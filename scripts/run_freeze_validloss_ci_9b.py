#!/usr/bin/env python
"""Real 9B target-scale candidate-vs-surrogate freeze A/B verdict.

This is the **Category-C measurement the loop has been asked to produce**:
a real ``Qwen/Qwen3.5-9B`` QLoRA run, on real public data, of the GOAL §4
question — *does an output-first progressive-freeze schedule retain quality
significantly better than a random-order freeze, or could seed noise alone
explain the gap?* — answered by feeding the real 9B valid_loss samples to the
same :func:`src.tg_lora.freeze_surrogate_ci.surrogate_valid_loss_ci` the
proxy-scale harness uses.

It is the union of the two half-instruments the repo already had:

* :mod:`scripts.probe_9b_memory_frontier` proved the **public 9B model-loading
  path** (``load_base_model`` → ``apply_lora`` → ``configure_trainable_lora_scope``)
  runs a real 9B forward+backward+step on a 12 GB GPU at ``seq_len=1024`` under
  the suffix-only scope (``last_25_percent``) — TASK-0155 closed the *memory*
  axis. It bypassed data with a synthetic batch, so it measured memory, not loss.
* :mod:`scripts.run_freeze_validloss_ci` is the **A/B instrument**: it drives the
  real :class:`~src.tg_lora.progressive_freeze.ProgressiveFreezeController` for an
  output-first candidate vs :func:`~src.tg_lora.freeze_schedule.random_freeze_order`
  surrogates across seeds and hands the real valid_loss samples to
  :func:`surrogate_valid_loss_ci`. It runs on a 24-hidden *proxy* model and tags
  the result ``proxy_scale=True``; its docstring states the drop-in contract —
  *"a target-scale run deposits its own samples through the same function with
  ``proxy_scale=False`` and the label upgrades with no code change."*

This script is that target-scale deposit. The make-or-break wiring question —
whether the freeze controller's ``layer_idx`` maps to the real model — is closed
by construction: both :func:`~src.model.lora_utils.configure_trainable_lora_scope`
and :class:`ProgressiveFreezeController` derive layer indices from the *same*
:func:`~src.model.lora_utils.iter_all_lora_params_by_layer` (regex
``layers.(\\d+).``), so the freeze loop runs on the real Qwen layer indices
identically to the proxy (verified at construction: the active scope returned by
``configure_trainable_lora_scope`` is passed straight through as the controller's
``active_layer_indices``, exactly as ``src/training/train_tg_lora.py`` wires it).

The data is real public Dolly (``databricks/databricks-dolly-15k``), tokenized
with the real Qwen tokenizer under the public SFT contract
(``train_on_prompt=False``: the ChatML user-turn is masked with ``labels=-100``,
only the assistant response is supervised). No private ``src.data`` is used —
the private pipeline's quality filtering is absent, which is noted honestly in
the deposit (candidate and surrogate share the identical data, so the A/B's
internal validity is unaffected; only absolute loss levels differ from a
filtered-data run).

Honesty (GOAL §7)
-----------------
This is a **real 9B + real data** A/B (``proxy_scale=False``), but a
**reduced-budget** one: few seeds and short training, not the config's full
``max_steps=1500`` multi-seed run. The deposit therefore carries
``reduced_budget=True`` and ``citable_as_full_section4_verdict=False`` — the
same honesty shape as the seq256 reduced probe (``e99e3c7``). It is the *first*
real-9B real-data A/B sample; whatever the bootstrap CI says
(``SURPASSES`` / ``TIES`` / ``UNDERSHOOTS``) is the measurement, never
pre-decided.

Usage::

    # The one-shot real-9B A/B (auto CUDA, suffix-only config, Dolly data).
    make freeze-validloss-ci-9b
    python -m scripts.run_freeze_validloss_ci_9b

    # Reduced budget for a fast smoke; write the JSON deposit.
    python -m scripts.run_freeze_validloss_ci_9b --total-steps 20 \\
        --n-candidate 1 --n-surrogate 1 --seq-len 1024 \\
        --json --output tests/fixtures/freeze_validloss_ci_9b_surrogate.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
from typing import Sequence

# Fragmentation control: a multi-step 9B QLoRA loop (variable per-example
# lengths + the compute_local_loss extra forwards + cached xin) fragments the
# CUDA cache far more than the single-shot memory probe did. expandable_segments
# is the documented lever that lets the suffix-only seq1024 stack fit a sustained
# run on 12 GB (set before torch initializes CUDA). Harmless on CPUs / larger GPUs.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import torch
from omegaconf import OmegaConf

# Shared A/B instrument pieces (the harness this deposits through).
from scripts.run_freeze_validloss_ci import resolve_device
from src.model.lora_utils import (
    configure_trainable_lora_scope,
    iter_all_lora_params,
)
from src.model.load_model import apply_lora, load_base_model, load_tokenizer
from src.tg_lora.activation_matching import ActivationMatchingLoss
from src.tg_lora.freeze_schedule import (
    FreezeSchedule,
    FreezeScheduleConfig,
    input_first_order,
    random_freeze_order,
)
from src.tg_lora.freeze_surrogate_ci import (
    format_surrogate_valid_loss_ci,
    surrogate_valid_loss_ci,
)
from src.tg_lora.progressive_freeze import ProgressiveFreezeController

logger = logging.getLogger("freeze-validloss-ci-9b")

# ---------------------------------------------------------------------------
# Defaults — a real-9B run that fits a 12 GB GPU (suffix-only, seq1024) with a
# budget small enough to complete an A/B sweep, large enough for the model to
# learn (valid_loss drops below the uniform floor) so the freeze has a
# trajectory to act on. All overridable on the CLI.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/9b_baseline_suffix_only_last25.yaml"
DEFAULT_SEQ_LEN = 1024            # suffix-only fits seq1024 on 12GB (TASK-0155)
DEFAULT_TRAIN_EXAMPLES = 12       # batch_size=1 batches the arms cycle through
DEFAULT_VALID_EXAMPLES = 16       # held-out eval set (distinct from train)
DEFAULT_TOTAL_STEPS = 40
DEFAULT_WARMUP_STEPS = 8          # freeze begins after this many steps
DEFAULT_DEPTH = 5                 # freeze this many of the active-scope layers
DEFAULT_SPACING = 4               # one freeze every N steps
DEFAULT_N_CANDIDATE = 2
DEFAULT_N_SURROGATE = 2
DEFAULT_BASE_SEED = 0
DEFAULT_DATASET = "databricks/databricks-dolly-15k"
DEFAULT_MAX_DATASET_ROWS = 4000   # cap the download for a bounded run

# ChatML (Qwen2.5/Qwen3.5 share it). Mirrors scripts/prepare_data.CHAT_TEMPLATE.
_CHATML_USER = "<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
_CHATML_RESPONSE = "{response}<|im_end|>"


# ---------------------------------------------------------------------------
# Public SFT data adapter (no src.data dependency)
# ---------------------------------------------------------------------------


def build_sft_example(
    tokenizer,
    instruction: str,
    response: str,
    context: str = "",
    *,
    max_seq_len: int = DEFAULT_SEQ_LEN,
) -> dict[str, torch.Tensor] | None:
    """Tokenize one Dolly record into a prompt-masked SFT example.

    Faithful to the public SFT contract (``train_on_prompt=False``): the ChatML
    user-turn prefix (``<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n``)
    is masked with ``labels=-100``; only the assistant response (plus its
    closing ``<|im_end|>``) is supervised. Right-truncated to ``max_seq_len``;
    returns ``None`` if the supervised portion is empty after truncation (so the
    caller drops prompt-dominant records rather than emitting an all-masked
    example — the same intent as the private ``min_supervised_tokens`` audit,
    which is absent on this mirror and noted honestly in the deposit).

    ``batch_size=1`` tensors; the caller pads/stacks into batches.
    """
    body = f"{instruction}\n\nContext: {context}" if context.strip() else instruction
    prefix = _CHATML_USER.format(body=body)
    completion = _CHATML_RESPONSE.format(response=response)

    # add_special_tokens=False: the ChatML markers are themselves the special
    # tokens; letting the tokenizer prepend its BOS would double-mark.
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix + completion, add_special_tokens=False).input_ids

    # Right-truncate the full sequence to the budget.
    full_ids = full_ids[:max_seq_len]
    labels = list(full_ids)
    prompt_len = min(len(prefix_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    # Drop examples whose supervised tail was entirely truncated away.
    if all(lab == -100 for lab in labels):
        return None

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    label_tensor = torch.tensor([labels], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_tensor,
    }


def _load_dolly_records(
    dataset: str, max_rows: int, seed: int
) -> list[dict]:
    """Load + shuffle Dolly records via the public ``datasets`` path.

    Seeded locally so candidate and surrogate arms see the identical record
    order independent of the global RNG / arm call order (the same
    reproducibility property :func:`make_batches` gives the proxy harness).
    """
    import random as _random
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train", streaming=True)
    records: list[dict] = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        records.append(
            {
                "instruction": row.get("instruction", ""),
                "context": row.get("context", ""),
                "response": row.get("response", ""),
            }
        )
    rng = _random.Random(seed)
    rng.shuffle(records)
    return records


def build_real_batches(
    tokenizer,
    records: Sequence[dict],
    *,
    n_examples: int,
    device,
    max_seq_len: int = DEFAULT_SEQ_LEN,
    offset: int = 0,
) -> list[dict]:
    """Tokenize the first ``n_examples`` usable records into batch_size=1 batches.

    ``offset`` selects a disjoint slice (train vs held-out valid from one
    record list). Records that truncate to an all-masked example are skipped,
    so the returned list may draw from slightly past ``offset + n_examples``.
    Every batch is moved to ``device``.
    """
    batches: list[dict] = []
    skipped = 0
    idx = offset
    while len(batches) < n_examples and idx < len(records):
        rec = records[idx]
        idx += 1
        ex = build_sft_example(
            tokenizer,
            rec["instruction"],
            rec["response"],
            context=rec.get("context", ""),
            max_seq_len=max_seq_len,
        )
        if ex is None:
            skipped += 1
            continue
        batches.append({k: v.to(device) for k, v in ex.items()})
    if len(batches) < n_examples:
        raise RuntimeError(
            f"Only {len(batches)} usable SFT examples from offset={offset} "
            f"(skipped {skipped} prompt-dominant); needed {n_examples}."
        )
    return batches


# ---------------------------------------------------------------------------
# Real-9B arm: load model, drive the freeze A/B loop, return final valid_loss.
# ---------------------------------------------------------------------------


def eval_loss_9b(model, valid_batches: list[dict]) -> float:
    """Mean HF cross-entropy over the held-out valid set (forward-only)."""
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in valid_batches:
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                labels=b["labels"],
            )
            tot += float(out.loss.item())
            n += 1
    model.train()
    return tot / max(n, 1)


def _reset_lora_for_arm(
    model, scope: str, seed: int
) -> set[int]:
    """Reset the ONE shared 9B model for a fresh A/B arm, in place.

    Reloading a 9B QLoRA model per arm fragments the CUDA cache on a 12 GB GPU:
    the bitsandbytes/accelerate-loaded base does not release cleanly until the
    process exits, so a second arm's load piles a second ~5.5 GB model on the
    first and OOMs (verified empirically — the surrogate arm died at a full-CE
    ``logits.float()`` step with two models resident). The memory probe
    (:mod:`scripts.probe_9b_memory_frontier`) avoided this by loading once and
    sweeping on the same model; this does the same for the A/B.

    So the model is loaded **once** and each arm re-initializes the LoRA adapter
    in place:

    * re-apply the suffix-only scope — ``configure_trainable_lora_scope`` sets
      ``requires_grad`` from the scope, which **un-freezes** the prior arm's
      frozen layers (the same call the trainer makes at build time);
    * re-initialize every LoRA ``lora_A`` (Kaiming uniform, matching PEFT's
      default init) and zero every ``lora_B`` (PEFT's zero init) under a fresh
      per-arm seed, so each arm is an independent LoRA init on the identical
      frozen 4-bit base — seed-independent, the only varying factor is
      ``order`` + seed, exactly as the reload-per-arm design intended.

    The fp32 norm casts and the input-require-grads hook set up once by
    ``_prepare_model_for_qlora_training`` are untouched; a fresh controller +
    optimizer per arm then holds no stale state (the ``xin`` cache lives inside
    the controller on CPU, and its capture hook is removed after each freeze).

    Returns the resolved active layer-index set (asserted equal across arms by
    the caller).
    """
    _names, active_indices = configure_trainable_lora_scope(model, scope)
    torch.manual_seed(seed)
    for name, p in iter_all_lora_params(model):
        if "lora_A" in name:
            torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif "lora_B" in name:
            p.detach().zero_()
    return active_indices


def arm_valid_loss_9b(
    model,
    order: Sequence[int],
    seed: int,
    *,
    scope: str,
    active_indices: set[int],
    train_batches: list[dict],
    valid_batches: list[dict],
    device,
    total_steps: int,
    warmup_steps: int,
    depth: int,
    spacing: int,
    lr: float,
    use_local_loss: bool = True,
) -> tuple[float, dict]:
    """One real-9B A/B arm on the shared model: reset, freeze under ``order``.

    Mirrors :func:`scripts.run_freeze_validloss_ci.arm_valid_loss` (the proxy
    instrument) so the 9B result is directly comparable through the same
    :func:`surrogate_valid_loss_ci`: reset the LoRA adapter on the shared model
    (:func:`_reset_lora_for_arm`), drive the real
    :class:`ProgressiveFreezeController` with a ``convergence_order`` schedule
    built from ``order`` (candidate and surrogate share one code path), train
    for ``total_steps`` optimizer steps on the boundary local loss once frozen
    (else the full CE task loss), and read the final valid_loss off the real
    forward pass.

    The caller owns the shared ``model``; this arm does not load or free it. It
    only clears the per-arm transient GPU state (optimizer + controller) so the
    next arm resets onto a clean cache.

    Returns ``(valid_loss, provenance)`` where provenance records the frozen
    layer set and trainable-param count for honest deposit.
    """
    resolved_indices = _reset_lora_for_arm(model, scope, seed)
    if resolved_indices != active_indices:
        raise RuntimeError(
            "active scope drifted across arms: "
            f"{sorted(resolved_indices)} != {sorted(active_indices)}"
        )
    model.train()

    scope_sorted = sorted(active_indices)
    if depth > len(scope_sorted):
        raise ValueError(
            f"depth {depth} exceeds active scope size {len(scope_sorted)}"
        )
    schedule_cfg = FreezeScheduleConfig(
        active_layer_indices=scope_sorted,
        num_epochs=total_steps,
        max_depth=depth,
        start_epoch=warmup_steps,
        spacing=spacing,
        policy="convergence_order",
        convergence_order=tuple(order),
    )
    schedule = FreezeSchedule.plan(schedule_cfg)
    ctrl = ProgressiveFreezeController(
        start_cycle=warmup_steps,
        active_layer_indices=set(scope_sorted),
        schedule=schedule,
    )
    loss_fn = ActivationMatchingLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)

    last_loss = float("nan")
    for step in range(total_steps):
        if ctrl.schedule is not None and ctrl.layers_due_at(step):
            ctrl.progress(model, step, train_batches, device)
        b = train_batches[step % len(train_batches)]
        opt.zero_grad(set_to_none=True)
        if use_local_loss and ctrl.frozen_layers:
            boundary = min(ctrl.frozen_layers)
            # The boundary local loss MUST run on the cached-xin batch
            # (train_batches[0]): ``apply_freeze_layer`` captured that layer's
            # ``xin`` from the dataloader's first batch, so the predicted front
            # activation must come from the same sequence shape. (The proxy
            # harness pins this to ``b_idx=0`` for the same reason; variable
            # per-example lengths on real data make the pinning explicit here.)
            loss = ctrl.compute_local_loss(
                model, train_batches[0], loss_fn, batch_idx=0,
                device=device, layer_idx=boundary,
            )
        else:
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                labels=b["labels"],
            )
            loss = out.loss
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    valid_loss = eval_loss_9b(model, valid_batches)
    # Mean full-CE over the TRAIN set under the final adapter — the
    # memorization-vs-generalization diagnostic for the regime the arm ran in.
    # ``last_train_loss`` above is the last *optimizer step's* loss, which is the
    # boundary activation-matching local loss once layers are frozen
    # (structurally ≈0 and not comparable across arms); ``final_ce_train_loss``
    # is the honest full cross-entropy on the train examples that reveals whether
    # the regime memorized (train_CE ≈ 0 ≪ valid_loss) or generalized
    # (train_CE well above 0, comparable to valid_loss). Without it the deposit
    # cannot tell a memorization-regime verdict from a generalization-regime one.
    final_ce_train_loss = eval_loss_9b(model, train_batches)
    provenance = {
        "frozen_layers": sorted(ctrl.frozen_layers),
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "last_train_loss": last_loss,
        "final_ce_train_loss": final_ce_train_loss,
    }
    # Clear per-arm transient state; the shared model stays loaded for the next arm.
    del opt, trainable, ctrl, loss_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return valid_loss, provenance


def candidate_order_9b(active_indices: set[int]) -> tuple[int, ...]:
    """Output-first order over the real active scope: highest layer first.

    Identical semantics to the proxy's :func:`output_first_order` (freeze the
    output side first = descending layer index), expressed over the real
    Qwen scope indices so it flows through the same ``convergence_order``
    planner as the surrogate — the apples-to-apples property the A/B requires.
    """
    return tuple(sorted(active_indices, reverse=True))


def control_order_9b(active_indices: set[int]) -> tuple[int, ...]:
    """Input-side contiguous control order over the real active scope.

    The DIRECTION-ISOLATION control for the §4 verdict (constitution P0: rule
    out a misattributed verdict). The candidate (:func:`candidate_order_9b`)
    freezes a **contiguous output-side block** while every random surrogate
    freezes a **scattered** set, so a candidate ``SURPASSES`` could be the
    output-side direction OR mere freeze-set contiguity. This order freezes the
    contiguous **input-side** block (lowest layer indices first =
    :func:`~src.tg_lora.freeze_schedule.input_first_order` over the real scope),
    so a candidate-vs-control comparison holds contiguity + depth + timing fixed
    and varies only direction — attributing (or refusing to attribute) the
    surrogate ``SURPASSES`` to the output side. Feeds the same
    ``convergence_order`` planner as candidate and surrogate (no separate branch).
    """
    return input_first_order(active_indices)


# ---------------------------------------------------------------------------
# Assembly: run candidate + surrogate arms, feed surrogate_valid_loss_ci.
# ---------------------------------------------------------------------------


def _is_reduced_budget(total_steps: int, max_steps: int) -> bool:
    """A run is reduced-budget unless it trained for the config's full
    ``max_steps`` (the §4 verdict's intended training length).

    Keeps :data:`reduced_budget` *honest*: a hardcoded ``True`` would silently
    lie about a future full-length run, and the citation gate
    (:attr:`citable_as_full_section4_verdict`) that keys off it would stay
    permanently closed no matter how long a run trained. With this, a run that
    reaches ``max_steps`` clears the flag (and a non-thin, target-scale one
    becomes citable). ``max_steps <= 0`` (absent / unparsed config) is treated
    as reduced — the conservative call, never silently promoting a run.
    """
    if max_steps <= 0:
        return True
    return total_steps < max_steps


def run_ci_9b(
    *,
    cfg,
    device,
    seq_len: int,
    train_examples: int,
    valid_examples: int,
    total_steps: int,
    warmup_steps: int,
    depth: int,
    spacing: int,
    n_candidate: int,
    n_surrogate: int,
    base_seed: int,
    dataset: str,
    max_dataset_rows: int,
    use_local_loss: bool = True,
    n_control: int = 0,
    n_baseline: int = 0,
) -> dict:
    """Run the real-9B candidate+surrogate sweep and return the §4 verdict dict.

    Loads the tokenizer + 9B model **once** (the bitsandbytes base does not
    release cleanly across reloads on a 12 GB GPU — see
    :func:`_reset_lora_for_arm`), derives the active scope from that one load
    (so the order vectors are built from the real layer indices), tokenizes a
    shared train/valid split of Dolly, then runs each arm as an in-place LoRA
    reset on the shared model. The resulting real valid_loss samples feed
    :func:`surrogate_valid_loss_ci` with ``proxy_scale=False`` (real 9B + real
    data) — the first such verdict grounded in target-scale numbers.

    ``n_control > 0`` adds an optional DIRECTION-CONTROL arm (input-side
    contiguous, :func:`control_order_9b`) and a ``direction_ci`` that compares
    candidate (output-contiguous) vs control (input-contiguous) — contiguity
    held fixed — to attribute the surrogate verdict to the output-side
    direction (constitution P0). ``n_control=0`` (default) runs no control and
    the §4 surrogate verdict is byte-identical to before.

    ``n_baseline > 0`` adds the GOAL §4 **full-backprop baseline** arm and a
    ``baseline_ci`` that compares candidate (output-first progressive freeze +
    activation-matching boundary loss) vs baseline (no freeze at all — every
    active-scope layer trained on the full CE task loss throughout). This is
    §4's other success axis: the surrogate verdict above is "the output-side
    order beats a random order" (does the schedule matter?); the baseline
    verdict is "the method's valid_loss stays within tolerance of full
    backprop" (GOAL §4 line 247 — does freezing cost quality?). The baseline
    arm reuses :func:`arm_valid_loss_9b` with ``depth=0`` (``max_depth=0`` →
    the schedule plans zero freezes → ``frozen_layers`` stays empty → the arm
    always takes the full-CE branch), so the full-backprop control runs through
    the identical training path as the candidate, varying only the freeze.
    ``n_baseline=0`` (default) runs no baseline and the deposit is byte-identical
    to before.
    """
    scope_label = cfg.training.get("trainable_lora_scope", "all")
    logger.info("Loading tokenizer + model for %s ...", cfg.model.name_or_path)
    tokenizer = load_tokenizer(cfg)
    torch.manual_seed(base_seed)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    _scope_names, active_indices = configure_trainable_lora_scope(model, scope_label)
    scope_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Active scope (%s): %d layers %s (%d trainable params)",
        scope_label, len(active_indices), sorted(active_indices), scope_trainable,
    )

    logger.info("Loading + tokenizing Dolly (%s) ...", dataset)
    records = _load_dolly_records(dataset, max_dataset_rows, base_seed)
    train_batches = build_real_batches(
        tokenizer, records, n_examples=train_examples, device=device,
        max_seq_len=seq_len, offset=0,
    )
    valid_batches = build_real_batches(
        tokenizer, records, n_examples=valid_examples, device=device,
        max_seq_len=seq_len, offset=train_examples,
    )
    logger.info(
        "Data: %d train batches, %d valid batches (seq_len<=%d, batch_size=1)",
        len(train_batches), len(valid_batches), seq_len,
    )

    cand_order = candidate_order_9b(active_indices)
    scope_sorted = sorted(active_indices)
    control_order = control_order_9b(active_indices)
    lr = float(cfg.training.learning_rate)
    cfg_max_steps = int(cfg.training.get("max_steps", 0))
    reduced_budget = _is_reduced_budget(total_steps, cfg_max_steps)
    if not reduced_budget:
        logger.info(
            "Full-budget run: total_steps=%d reaches config max_steps=%d "
            "(reduced_budget=False).", total_steps, cfg_max_steps,
        )

    try:
        candidate_results = [
            arm_valid_loss_9b(
                model, cand_order, base_seed + i, scope=scope_label,
                active_indices=active_indices,
                train_batches=train_batches, valid_batches=valid_batches, device=device,
                total_steps=total_steps, warmup_steps=warmup_steps, depth=depth,
                spacing=spacing, lr=lr, use_local_loss=use_local_loss,
            )
            for i in range(n_candidate)
        ]
        candidate_losses = [v for v, _ in candidate_results]
        surrogate_results = [
            arm_valid_loss_9b(
                model, random_freeze_order(scope_sorted, base_seed + 1000 + i),
                base_seed + 100 + i, scope=scope_label,
                active_indices=active_indices,
                train_batches=train_batches, valid_batches=valid_batches, device=device,
                total_steps=total_steps, warmup_steps=warmup_steps, depth=depth,
                spacing=spacing, lr=lr, use_local_loss=use_local_loss,
            )
            for i in range(n_surrogate)
        ]
        surrogate_losses = [v for v, _ in surrogate_results]
        # DIRECTION-CONTROL arm (constitution P0): the input-side contiguous
        # control. Only runs when n_control > 0; n_control=0 leaves this list
        # empty and the §4 surrogate verdict byte-identical to before. Distinct
        # seed offset (base_seed + 200) so the control's LoRA init never collides
        # with a candidate (base_seed + i) or surrogate (base_seed + 100 + i) arm.
        control_results = [
            arm_valid_loss_9b(
                model, control_order, base_seed + 200 + i, scope=scope_label,
                active_indices=active_indices,
                train_batches=train_batches, valid_batches=valid_batches, device=device,
                total_steps=total_steps, warmup_steps=warmup_steps, depth=depth,
                spacing=spacing, lr=lr, use_local_loss=use_local_loss,
            )
            for i in range(n_control)
        ]
        control_losses = [v for v, _ in control_results]
        # FULL-BACKPROP BASELINE arm (GOAL §4 line 247 control (i)): no freeze
        # at all — ``depth=0`` so ``max_depth=0`` plans zero freezes and the arm
        # always takes the full-CE branch. Every active-scope layer trains on the
        # task loss throughout. This is the "valid_loss vs full backprop"
        # success axis the surrogate verdict does NOT measure (the surrogate is a
        # random-order freeze, itself a freeze). Distinct seed offset
        # (base_seed + 300) so the baseline's LoRA init never collides with
        # candidate (base_seed + i) / surrogate (base_seed + 100 + i) / control
        # (base_seed + 200 + i). The ``order`` arg is inert at depth=0 (nothing
        # is scheduled to freeze); the scope order is passed for a valid,
        # non-empty convergence_order vector the config validator accepts.
        baseline_results = [
            arm_valid_loss_9b(
                model, tuple(scope_sorted), base_seed + 300 + i, scope=scope_label,
                active_indices=active_indices,
                train_batches=train_batches, valid_batches=valid_batches, device=device,
                total_steps=total_steps, warmup_steps=warmup_steps, depth=0,
                spacing=spacing, lr=lr, use_local_loss=use_local_loss,
            )
            for i in range(n_baseline)
        ]
        baseline_losses = [v for v, _ in baseline_results]
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ci = surrogate_valid_loss_ci(
        candidate_losses, surrogate_losses, seed=base_seed
    )
    # Direction-isolation CI (constitution P0): candidate (output-contiguous)
    # vs control (input-contiguous) — contiguity + depth + timing held fixed,
    # only DIRECTION varies. ``surrogate_valid_loss_ci`` is a generic two-sample
    # bootstrap on ``mean(b) - mean(a)``; the control arm occupies the
    # "surrogate" slot. None when no control arm ran (n_control=0).
    direction_ci = (
        surrogate_valid_loss_ci(candidate_losses, control_losses, seed=base_seed)
        if control_losses
        else None
    )
    # Full-backprop baseline CI (GOAL §4 line 247): candidate (progressive
    # freeze + activation matching) vs baseline (no freeze, full CE). The
    # baseline occupies the "surrogate" slot of the generic two-sample
    # bootstrap, so its mean is ``surrogate_mean`` internally and relabeled
    # ``baseline_mean`` in the deposit (mirrors the direction arm's
    # control_mean relabel). None when no baseline arm ran (n_baseline=0).
    baseline_ci = (
        surrogate_valid_loss_ci(candidate_losses, baseline_losses, seed=base_seed)
        if baseline_losses
        else None
    )
    return {
        "ci": ci,
        "candidate_losses": candidate_losses,
        "surrogate_losses": surrogate_losses,
        "candidate_order": list(cand_order),
        "device": str(device),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "depth": depth,
        "spacing": spacing,
        "n_candidate": n_candidate,
        "n_surrogate": n_surrogate,
        "base_seed": base_seed,
        "active_scope": sorted(active_indices),
        "scope_label": scope_label,
        "n_active_layers": len(active_indices),
        "scope_trainable_params": scope_trainable,
        "seq_len": seq_len,
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "dataset": dataset,
        "model": cfg.model.name_or_path,
        "use_local_loss": use_local_loss,
        # Real 9B + real data. ``reduced_budget`` is honest about the step
        # budget: True unless ``total_steps`` reaches the config's full
        # ``max_steps`` (the §4 verdict's intended training length). See
        # :func:`_is_reduced_budget` — never a hardcoded flag.
        "proxy_scale": False,
        "reduced_budget": reduced_budget,
        "cfg_max_steps": cfg_max_steps,
        "candidate_provenance": [p for _, p in candidate_results],
        "surrogate_provenance": [p for _, p in surrogate_results],
        # Direction-isolation control arm (None-valued / empty when n_control=0,
        # so the §4 surrogate verdict is unchanged). When present, these plus
        # ``direction_ci`` attribute the surrogate SURPASSES to direction (or
        # refuse to, if the control ties) — the P0 confound isolation.
        "control_losses": control_losses,
        "control_order": list(control_order),
        "n_control": n_control,
        "control_provenance": [p for _, p in control_results],
        "direction_ci": direction_ci,
        # Full-backprop baseline arm (GOAL §4 line 247 control (i)). Empty /
        # None-valued when n_baseline=0, so a no-baseline run deposits
        # byte-identically to before. When present, ``baseline`` is the
        # candidate (progressive freeze) vs baseline (no freeze, full CE) CI —
        # §4's "valid_loss within tolerance of full backprop" axis. The baseline
        # never freezes, so there is no freeze order to record (baseline_order
        # is the empty list, not a meaningful vector).
        "baseline_losses": baseline_losses,
        "n_baseline": n_baseline,
        "baseline_provenance": [p for _, p in baseline_results],
        "baseline_ci": baseline_ci,
    }


def result_to_json(result: dict) -> dict:
    """JSON-deposit shape: verdict + real samples + full provenance + honesty."""
    ci = result["ci"]
    reduced = bool(result["reduced_budget"])
    return {
        "verdict": ci.significance_verdict,
        "passes": ci.passes,
        "significant_surpasses": ci.significant_surpasses,
        "is_material": ci.is_material,
        "is_thin_evidence": ci.is_thin_evidence,
        "candidate_mean": ci.candidate_mean,
        "surrogate_mean": ci.surrogate_mean,
        "point_improvement": ci.point_improvement,
        "lower": ci.lower,
        "upper": ci.upper,
        "confidence": ci.confidence,
        "n_bootstrap": ci.n_bootstrap,
        "candidate_losses": result["candidate_losses"],
        "surrogate_losses": result["surrogate_losses"],
        "candidate_order": list(result["candidate_order"]),
        "device": result["device"],
        "model": result["model"],
        "dataset": result["dataset"],
        "total_steps": result["total_steps"],
        "warmup_steps": result["warmup_steps"],
        "depth": result["depth"],
        "spacing": result["spacing"],
        "n_candidate": result["n_candidate"],
        "n_surrogate": result["n_surrogate"],
        "base_seed": result["base_seed"],
        "active_scope": list(result["active_scope"]),
        "scope_label": result["scope_label"],
        "n_active_layers": result["n_active_layers"],
        "scope_trainable_params": result["scope_trainable_params"],
        "seq_len": result["seq_len"],
        "train_examples": result["train_examples"],
        "valid_examples": result["valid_examples"],
        "use_local_loss": result["use_local_loss"],
        # GOAL §7 scale + budget honesty.
        "proxy_scale": result["proxy_scale"],
        "reduced_budget": reduced,
        "cfg_max_steps": result.get("cfg_max_steps"),
        "candidate_provenance": result["candidate_provenance"],
        "surrogate_provenance": result["surrogate_provenance"],
        # Direction-isolation control arm (constitution P0). Empty / None-valued
        # when n_control=0, so a no-control run deposits byte-identically to
        # before. When present, ``direction`` is the candidate(output-contiguous)
        # vs control(input-contiguous) CI that attributes the surrogate
        # SURPASSES to the output-side direction — or refuses to, on a TIES
        # (which would mean contiguity, not direction, earned the lead).
        "control_losses": result["control_losses"],
        "control_order": list(result["control_order"]),
        "n_control": result["n_control"],
        "control_provenance": result["control_provenance"],
        "direction": _direction_ci_to_json(result["direction_ci"]),
        # Full-backprop baseline arm (GOAL §4 line 247 control (i)): the
        # candidate-vs-full-backprop CI. Empty / None-valued when n_baseline=0,
        # so a no-baseline run deposits byte-identically to before. When
        # present, ``baseline`` is the §4 "valid_loss within tolerance of full
        # backprop" axis — the success half the surrogate/direction verdicts
        # (themselves freeze-vs-freeze) do not measure.
        "baseline_losses": result["baseline_losses"],
        "n_baseline": result["n_baseline"],
        "baseline_provenance": result["baseline_provenance"],
        "baseline": _baseline_ci_to_json(result["baseline_ci"]),
        # Machine-readable citation gate: a run is the full §4 verdict ONLY when
        # it is target-scale (not proxy), full-budget (reached config max_steps,
        # not reduced), AND non-thin (enough seeds for the bootstrap to capture
        # variance). The private ``src.data`` quality filter is a further axis
        # this gate cannot see (absent on the mirror) — it is noted in the
        # report, never silently assumed away. The direction-isolation analysis
        # is an *attribution* caveat on the verdict's interpretation, not a
        # scale/budget axis, so it never opens or closes this gate by itself.
        "citable_as_target_scale": (not result["proxy_scale"]),
        "citable_as_full_section4_verdict": (
            (not result["proxy_scale"]) and (not reduced) and (not ci.is_thin_evidence)
        ),
    }


def _direction_ci_to_json(direction_ci) -> dict | None:
    """Serialize the direction-isolation CI (or ``None``) for the deposit.

    ``direction_ci`` reuses :func:`surrogate_valid_loss_ci` with the
    input-contiguous CONTROL arm in the "surrogate" slot, so its
    :attr:`~SurrogateValidLossCI.surrogate_mean` is the control (input-side)
    mean — relabeled ``control_mean`` here so the deposit reads honestly rather
    than calling an input-side control a "surrogate". ``None`` (no control arm
    ran) round-trips as JSON ``null``.
    """
    if direction_ci is None:
        return None
    return {
        "verdict": direction_ci.significance_verdict,
        "candidate_mean": direction_ci.candidate_mean,
        "control_mean": direction_ci.surrogate_mean,
        "point_improvement": direction_ci.point_improvement,
        "lower": direction_ci.lower,
        "upper": direction_ci.upper,
        "confidence": direction_ci.confidence,
        "is_thin_evidence": direction_ci.is_thin_evidence,
        "n_candidate": direction_ci.n_candidate,
        "n_control": direction_ci.n_surrogate,
    }


def _baseline_ci_to_json(baseline_ci) -> dict | None:
    """Serialize the full-backprop baseline CI (or ``None``) for the deposit.

    ``baseline_ci`` reuses :func:`surrogate_valid_loss_ci` with the no-freeze
    full-CE baseline arm in the "surrogate" slot, so its
    :attr:`~SurrogateValidLossCI.surrogate_mean` is the baseline mean — relabeled
    ``baseline_mean`` here so the deposit reads honestly (a full-backprop
    control must not be called a "surrogate", which on this deposit means a
    random-order *freeze*). ``None`` (no baseline arm ran, ``n_baseline=0``)
    round-trips as JSON ``null``.

    Verdict reading for §4 line 247 (valid_loss within tolerance of full
    backprop): ``SURPASSES`` (candidate < baseline) or ``TIES`` (candidate ≈
    baseline) satisfies "within tolerance"; ``UNDERSHOOTS`` (candidate > baseline
    beyond tolerance) means the freeze cost quality — condition (a) failed at
    this budget.
    """
    if baseline_ci is None:
        return None
    return {
        "verdict": baseline_ci.significance_verdict,
        "candidate_mean": baseline_ci.candidate_mean,
        "baseline_mean": baseline_ci.surrogate_mean,
        "point_improvement": baseline_ci.point_improvement,
        "lower": baseline_ci.lower,
        "upper": baseline_ci.upper,
        "confidence": baseline_ci.confidence,
        "is_thin_evidence": baseline_ci.is_thin_evidence,
        "n_candidate": baseline_ci.n_candidate,
        "n_baseline": baseline_ci.n_surrogate,
    }


def format_report_9b(result: dict) -> str:
    ci = result["ci"]
    lines = [
        "freeze_valid_loss_ci_9b — GOAL §4 REAL 9B target-scale A/B verdict",
        f"  model: {result['model']}  dataset: {result['dataset']}",
        f"  device: {result['device']}  seq_len={result['seq_len']}  "
        f"scope={result['scope_label']} ({result['n_active_layers']} layers "
        f"{result['active_scope']}, {result['scope_trainable_params']} trainable params)",
        f"  budget: total_steps={result['total_steps']} warmup={result['warmup_steps']} "
        f"depth={result['depth']} spacing={result['spacing']}  "
        f"n_candidate={result['n_candidate']} n_surrogate={result['n_surrogate']} "
        f"base_seed={result['base_seed']}  use_local_loss={result['use_local_loss']}",
        f"  candidate_order: {tuple(result['candidate_order'])}",
        "",
        format_surrogate_valid_loss_ci(ci),
        "",
        f"  candidate valid_loss samples: {[round(v, 6) for v in result['candidate_losses']]}",
        f"  surrogate valid_loss samples: {[round(v, 6) for v in result['surrogate_losses']]}",
        "  note: REAL_9B_TARGET_SCALE — real Qwen/Qwen3.5-9B + real public Dolly "
        "data (proxy_scale=False); the verdict above is grounded in numbers from "
        "an actual 9B run, not a proxy.",
    ]
    # Direction-isolation block (constitution P0): only when a control arm ran.
    # Attributes the surrogate SURPASSES above to the output-side direction — or
    # refuses to (a TIES means contiguity, not direction, earned the lead). The
    # control occupies the "surrogate" slot of the CI, so its mean is labeled
    # control_mean here, not surrogate_mean.
    if result.get("direction_ci") is not None:
        dci = result["direction_ci"]
        lines += [
            "",
            "  direction_isolation — candidate (output-contiguous) vs control "
            "(input-contiguous); contiguity + depth + timing held fixed:",
            f"    candidate_mean={dci.candidate_mean:.6f} vs "
            f"control_mean={dci.surrogate_mean:.6f}  "
            f"point={dci.point_improvement:.6f} "
            f"ci[{dci.confidence:.0%}]=[{dci.lower:.6f}, {dci.upper:.6f}]  "
            f"verdict={dci.significance_verdict}",
            f"    control valid_loss samples: "
            f"{[round(v, 6) for v in result['control_losses']]}",
        ]
        if dci.is_thin_evidence:
            lines.append(
                "    note: THIN_EVIDENCE — the direction CI has <3 seeds in an "
                "arm; do not read the direction verdict as confirmed."
            )
        lines.append(
            "    note: this ATTRIBUTES the §4 surrogate SURPASSES above. "
            "SURPASSES here => the output-side DIRECTION matters (not just "
            "contiguity); TIES => contiguity, not direction, earned the lead."
        )
    # Full-backprop baseline block (GOAL §4 line 247): only when a baseline arm
    # ran. The candidate (progressive freeze + activation matching) vs the
    # no-freeze full-CE baseline — the "valid_loss within tolerance of full
    # backprop" success axis. The baseline occupies the CI's "surrogate" slot,
    # so its mean is labeled baseline_mean here.
    if result.get("baseline_ci") is not None:
        bci = result["baseline_ci"]
        lines += [
            "",
            "  full_backprop_baseline — candidate (progressive freeze) vs "
            "baseline (no freeze, full CE); the §4 line-247 "
            "'valid_loss within tolerance of full backprop' axis:",
            f"    candidate_mean={bci.candidate_mean:.6f} vs "
            f"baseline_mean={bci.surrogate_mean:.6f}  "
            f"point={bci.point_improvement:.6f} "
            f"ci[{bci.confidence:.0%}]=[{bci.lower:.6f}, {bci.upper:.6f}]  "
            f"verdict={bci.significance_verdict}",
            f"    baseline valid_loss samples: "
            f"{[round(v, 6) for v in result['baseline_losses']]}",
        ]
        if bci.is_thin_evidence:
            lines.append(
                "    note: THIN_EVIDENCE — the baseline CI has <3 seeds in an "
                "arm; do not read the baseline verdict as confirmed."
            )
        lines.append(
            "    note: this is the OTHER §4 axis from the surrogate/direction "
            "verdicts above (those are freeze-vs-freeze). SURPASSES or TIES "
            "here => freezing did NOT cost quality vs full backprop (§4 line "
            "247 satisfied); UNDERSHOOTS => it did, at this budget."
        )
    # Honesty notes are flag-driven so the report never contradicts the
    # machine-readable labels (a hardcoded "reduced" string would lie about a
    # full-budget run, exactly the defect _is_reduced_budget fixes).
    if result["reduced_budget"]:
        lines.append(
            f"  note: REDUCED_BUDGET — total_steps={result['total_steps']} is short "
            f"of the config's full max_steps={result.get('cfg_max_steps')} multi-seed "
            "run, so this is a target-scale data point, NOT yet the complete §4 "
            "verdict (citable_as_full_section4_verdict=False)."
        )
    else:
        lines.append(
            f"  note: FULL_BUDGET — total_steps={result['total_steps']} reaches the "
            f"config's max_steps={result.get('cfg_max_steps')}."
        )
    if ci.is_thin_evidence:
        lines.append(
            "  note: THIN_EVIDENCE — fewer than 3 seeds in an arm; the recorded "
            "verdict is not confirmed (the formatter says so plainly)."
        )
    lines.append(
        "  note: the private src.data quality filter is absent on this mirror "
        "(absolute loss levels differ from a filtered-data run; the A/B internal "
        "validity is unaffected since candidate and surrogate share identical data)."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_freeze_validloss_ci_9b",
        description=(
            "Real GOAL §4 9B target-scale A/B verdict: trains the real 9B "
            "Qwen/Qwen3.5-9B QLoRA (suffix-only scope) on real public Dolly data "
            "for an output-first candidate vs random-order surrogates, and feeds "
            "the real valid_loss samples to surrogate_valid_loss_ci "
            "(proxy_scale=False, reduced budget)."
        ),
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="9B config to bind.")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu.")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    p.add_argument("--valid-examples", type=int, default=DEFAULT_VALID_EXAMPLES)
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--spacing", type=int, default=DEFAULT_SPACING)
    p.add_argument("--n-candidate", type=int, default=DEFAULT_N_CANDIDATE)
    p.add_argument("--n-surrogate", type=int, default=DEFAULT_N_SURROGATE)
    p.add_argument(
        "--n-control", type=int, default=0,
        help=(
            "DIRECTION-CONTROL arm seeds: run an input-side contiguous control "
            "(input_first_order) alongside candidate+surrogate and emit a "
            "direction-isolation CI (candidate output-contiguous vs control "
            "input-contiguous, contiguity held fixed) that attributes the "
            "surrogate SURPASSES to the output-side direction. Default 0 = no "
            "control (the §4 surrogate verdict is unchanged). Set >=3 for a "
            "non-thin direction verdict."
        ),
    )
    p.add_argument(
        "--n-baseline", type=int, default=0,
        help=(
            "FULL-BACKPROP BASELINE arm seeds: run a no-freeze full-CE baseline "
            "(depth=0) alongside candidate+surrogate+control and emit a "
            "candidate-vs-baseline CI (the §4 line-247 'valid_loss within "
            "tolerance of full backprop' axis — the success half the "
            "freeze-vs-freeze surrogate/direction verdicts do not measure). "
            "Default 0 = no baseline (the §4 surrogate verdict is unchanged). "
            "Set >=3 for a non-thin baseline verdict."
        ),
    )
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--max-dataset-rows", type=int, default=DEFAULT_MAX_DATASET_ROWS)
    p.add_argument(
        "--no-local-loss", action="store_true",
        help=(
            "Disable the boundary activation-matching local loss on frozen steps "
            "(use the full CE task loss throughout). Default faithfully mirrors "
            "the proxy A/B instrument, which uses the local loss once frozen."
        ),
    )
    p.add_argument("--json", action="store_true", help="emit JSON evidence to stdout.")
    p.add_argument("--output", default=None, help="write the report/JSON to this path.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        logger.error("CUDA not available — a real 9B run requires a GPU. Aborting.")
        return 2

    cfg = OmegaConf.load(args.config)
    device = resolve_device(args.device)

    result = run_ci_9b(
        cfg=cfg,
        device=device,
        seq_len=args.seq_len,
        train_examples=args.train_examples,
        valid_examples=args.valid_examples,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        depth=args.depth,
        spacing=args.spacing,
        n_candidate=args.n_candidate,
        n_surrogate=args.n_surrogate,
        base_seed=args.base_seed,
        dataset=args.dataset,
        max_dataset_rows=args.max_dataset_rows,
        use_local_loss=not args.no_local_loss,
        n_control=args.n_control,
        n_baseline=args.n_baseline,
    )

    payload = (
        json.dumps(result_to_json(result), indent=2)
        if args.json
        else format_report_9b(result)
    )
    print(payload, flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
