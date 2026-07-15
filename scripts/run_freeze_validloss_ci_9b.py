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

The citation gate ``citable_as_full_section4_verdict`` opens only when FOUR
honesty axes clear together: target-scale (not proxy), full-budget (reached
``max_steps``), non-thin (≥3 seeds/arm), AND generalization-regime (the
candidate arm generalized rather than memorized — see :func:`_classify_regime`).
The regime axis is load-bearing: a ``--total-steps 1500`` run paired with the
default small ``--train-examples`` would clear the budget axis but do tens of
epochs and memorize, so without the regime check it would be mislabeled as the
complete §4 verdict. ``make freeze-validloss-ci-9b-full`` sizes the train set
(~600 examples) so a full 1500-step run stays at ~2.5 epochs — generalizing.

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
import hashlib
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

from pathlib import Path

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
from src.tg_lora.freeze_cost import (
    LEVEL1_REALIZED_REDUCTION_CEILING,
    FreezeCostAccountant,
    LayerBackwardCost,
    realizable_reduction,
)
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

# Exit codes main() can return — keep the table explicit so a polling loop /
# the resumable --ledger workflow can branch on them:
#   0  success (deposit written)
#   1  unexpected error (uncaught exception)
#   2  CUDA unavailable
#   3  IncompleteResumeError (ledger banked only one side of the A/B; re-run)
#   75 GPU free-memory tempfail (EX_TEMPFAIL from sysexits.h: "retry later")
EXIT_GPU_TEMPFAIL = 75

# Free-GPU-memory floor for the pre-flight (GiB). Calibrated to defer only when
# another process is clearly holding the card — the seq1024 suffix-only peak is
# ~11.2 GB (probe da4fa4f / TASK-0155), the 12 GB card has ~11 GB free when
# otherwise idle, and a concurrent run (e.g. the private repo's own ~8.2 GB
# verdict) drops free to ~4 GB. 10 GiB separates those two regimes without
# spuriously firing at idle. It sits *below* the run's peak on purpose: this is
# a "is another big process holding the GPU" check, not a fit-guarantee (the
# card is fundamentally tight at seq1024 — see probe da4fa4f). --min-free-gib 0
# disables.
DEFAULT_MIN_FREE_GIB = 10.0

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
    loss_curve_sink: list[float] | None = None,
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

    ``loss_curve_sink`` is the opt-in per-step training-loss capture: when a
    list is passed, each optimizer step's loss is appended to it in order, so
    the caller can persist the full loss trajectory as a reproducible run-log
    artifact (see :func:`_write_run_log`). ``None`` (default) captures nothing
    and the arm is byte-identical to the pre-run-log path — the capture is a
    pure side effect that never touches the returned verdict or provenance.
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
        # Opt-in per-step loss-curve capture for the reproducible run-log artifact
        # (see :func:`_write_run_log`). ``None`` → no-op, so an arm run without a
        # sink is byte-identical to the pre-run-log path.
        if loss_curve_sink is not None:
            loss_curve_sink.append(last_loss)

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
# Heterogeneous architecture (per-layer asymmetric LoRA rank) — the target-scale
# realization of the proxy's discriminating positive control. At proxy scale the
# order-sensitivity diagnosis proved freeze ORDER is structurally non-resolvable
# (Var(order)=0.000) because a uniform-rank stack + full-rank learnable head is
# robust to WHICH layer froze; resolving an order effect needs real per-layer
# specialization. Heterogeneous ranks inject exactly that asymmetry — output-side
# active layers carry higher adapter capacity — so an output-first freeze
# structurally changes which capacity survives, the condition under which the
# homogeneous §4 verdict's TIES could resolve. Same public-Dolly data path as
# every homogeneous leg (architecture-independent); same verdict gate + cost
# accounting (architecture-independent). NOT harness plumbing — the one remaining
# open §4 research leg, now wired to target scale.
# ---------------------------------------------------------------------------
HOMOGENEOUS = "homogeneous"
HETEROGENEOUS = "heterogeneous"
ARCHITECTURES = (HOMOGENEOUS, HETEROGENEOUS)


def heterogeneous_ranks_9b(active_layers, base_rank: int) -> tuple[int, ...]:
    """Per-layer LoRA rank rising geometrically toward the output side.

    The target-scale mirror of the proxy harness's ``heterogeneous_ranks``: a
    geometric schedule ``base_rank ** (i / (n - 1))`` over the ``n`` active
    layers (sorted ascending), so the output-most layer carries ``base_rank``
    (the config's homogeneous ``r``, keeping the total adapter budget
    comparable) and earlier active layers carry progressively less capacity.
    This realizes the GOAL §1.5/§8 non-uniform per-layer-cost asymmetry as
    per-layer adapter *capacity* — the specialization signal a uniform-rank
    stack lacks and the order-sensitivity diagnosis identified as the missing
    condition for a freeze-order effect to resolve at target scale.
    """
    layers = sorted(active_layers)
    n = len(layers)
    if n <= 1:
        return (base_rank,) if n == 1 else ()
    return tuple(
        max(1, int(round(base_rank ** (i / (n - 1))))) for i in range(n)
    )


def _decoder_layer_count(model) -> int:
    """Number of decoder layers on a HF decoder model, read pre-LoRA.

    Heterogeneous ranks must be set at ``get_peft_model`` time (rank is
    structural), which is BEFORE :func:`configure_trainable_lora_scope` derives
    the active set from the applied LoRA params — so the active scope has to be
    computable from the base model itself. The CausalLM wraps its base in
    ``.model`` and the base exposes ``.layers`` (an ``nn.ModuleList``); the
    config's ``num_hidden_layers`` is the fallback. :func:`run_ci_9b` asserts
    this pre-LoRA scope matches the post-LoRA one so a model-structure surprise
    fails loud rather than seeding wrong ranks.
    """
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if isinstance(layers, torch.nn.ModuleList):
        return len(layers)
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if n:
        return int(n)
    raise ValueError(
        "Cannot determine decoder-layer count for heterogeneous ranks "
        "(model has no .model.layers ModuleList and no config.num_hidden_layers)."
    )


def _active_scope_pre_lora(model, scope_label: str) -> list[int]:
    """Active decoder-layer indices read from the base model, pre-LoRA.

    Mirrors :func:`src.model.lora_utils.get_last_fraction_lora_layer_indices`
    exactly (``ceil`` of the fraction, the LAST layers) but over the base
    model's layer count instead of the post-LoRA LoRA-param indices — so the
    rank pattern can be built for exactly the layers that will be trainable.
    """
    n = _decoder_layer_count(model)
    if scope_label == "all":
        return list(range(n))
    if scope_label == "last_25_percent":
        target = max(1, math.ceil(n * 0.25))
        return list(range(n - target, n))
    raise ValueError(f"Unsupported trainable_lora_scope: {scope_label}")


def build_rank_pattern(
    active_layers, architecture: str, base_rank: int
) -> tuple[dict, dict, dict]:
    """PEFT ``rank_pattern`` / ``alpha_pattern`` for the active layers.

    Returns ``(rank_pattern, alpha_pattern, layer_ranks)``: for ``HETEROGENEOUS``
    a per-layer geometric rank schedule over the active layers (each layer's
    regex keys ``alpha`` to ``2 * rank`` so ``alpha / rank`` stays the constant
    LoRA scaling and the only varying factor is capacity), plus a human-readable
    ``{layer: rank}`` dict for deposit provenance. For ``HOMOGENEOUS`` (the
    default) both patterns are empty — every layer takes ``base_rank`` from the
    config, byte-identical to before. The regex form ``layers\\.{i}\\..*`` is the
    full-match form PEFT's :func:`peft.utils.other.get_pattern_key` requires.
    """
    if architecture == HOMOGENEOUS:
        return {}, {}, {}
    if architecture != HETEROGENEOUS:
        raise ValueError(f"Unsupported architecture: {architecture}")
    ranks = heterogeneous_ranks_9b(active_layers, base_rank)
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    layer_ranks: dict[str, int] = {}
    for layer, rank in zip(sorted(active_layers), ranks):
        key = rf"layers\.{layer}\..*"
        rank_pattern[key] = rank
        alpha_pattern[key] = 2 * rank
        layer_ranks[str(layer)] = rank
    return rank_pattern, alpha_pattern, layer_ranks


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


# ── training-regime honesty ─────────────────────────────────────────────────
#
# The 9th-iteration generalization-regime verdict established that the §4 A/B is
# only externally valid when measured on a GENERALIZING model: in the
# memorization regime (few train examples × many epochs) the adapter drives train
# cross-entropy toward 0 and the held-out valid_loss is dominated by the frozen
# base, so a "SURPASSES" read off a memorized model is an artifact, not the §4
# question. The per-arm ``final_ce_train_loss`` diagnostic (mean full-CE over the
# train set under the final adapter) makes that regime machine-readable.
#
# The citation gate (:func:`result_to_json`) therefore needs a REGIME axis in
# addition to scale / budget / thin-ness: without it, a future ``--total-steps
# 1500`` run (which clears ``reduced_budget``) paired with the default small
# ``--train-examples`` would do tens of epochs, memorize, and STILL flip
# ``citable_as_full_section4_verdict=True`` — silently mislabeling a
# memorization artifact as the complete §4 verdict. The thresholds below are
# grounded in the committed 9B deposits, not picked from thin air:
#   * generalization-regime candidate arms (freeze_validloss_ci_9b_generalization
#     / _baseline) have final_ce_train_loss ≈ 1.507 with valid ≈ 1.515 → gap
#     ≈ 0.008;
#   * the full-backprop BASELINE arm overfits with final_ce 0.77 ≪ valid 1.54
#     → gap 0.77;
#   * memorization-regime arms (8 train × 20 step) collapse train CE toward 0.
# So a train-CE floor of 0.5 separates memorization (~0) from generalization
# (~1.5) with margin, and a train-valid gap threshold of 0.5 separates
# generalization (~0.01) from overfit (~0.77) with margin.
REGIME_GENERALIZATION = "generalization"
REGIME_MEMORIZATION = "memorization"
REGIME_OVERFIT = "overfit"
REGIME_UNKNOWN = "unknown"

_MEMORIZATION_TRAIN_CE_FLOOR = 0.5
_OVERFIT_GAP_THRESHOLD = 0.5


def _classify_regime(final_ce_train_loss, valid_loss):
    """Classify a run's training regime from the candidate arm's train/valid CE.

    ``final_ce_train_loss`` is the candidate arm's mean full cross-entropy over
    the *train* set under the final adapter; ``valid_loss`` is the candidate
    arm's mean held-out valid_loss (:attr:`SurrogateValidLossCI.candidate_mean`).
    Returns one of the :data:`REGIME_*` constants. Anything missing or
    non-finite (e.g. a deposit recorded before the ``final_ce_train_loss``
    diagnostic existed) classifies as :data:`REGIME_UNKNOWN` — the conservative
    call, which never opens the full-§4 citation gate on a regime it cannot
    verify.
    """
    try:
        ce = float(final_ce_train_loss)
        vl = float(valid_loss)
    except (TypeError, ValueError):
        return REGIME_UNKNOWN
    if not (math.isfinite(ce) and math.isfinite(vl)):
        return REGIME_UNKNOWN
    if ce < _MEMORIZATION_TRAIN_CE_FLOOR:
        return REGIME_MEMORIZATION
    if (vl - ce) > _OVERFIT_GAP_THRESHOLD:
        return REGIME_OVERFIT
    return REGIME_GENERALIZATION


def _full_section4_verdict_gate(
    *, proxy_scale: bool, reduced_budget: bool, is_thin_evidence: bool, regime: str
) -> bool:
    """The 4-conjunct citation gate, as one source of truth.

    A run is citable as the COMPLETE §4 verdict ONLY when it clears all four
    axes: target-scale (not a proxy), full-budget (reached config ``max_steps``,
    not reduced), non-thin (enough seeds for the bootstrap to capture variance),
    AND in the generalization regime (the candidate generalized rather than
    memorized — see :func:`_classify_regime`). Extracted from
    :func:`result_to_json` so the gate is a single testable expression: the
    serializer and the deposit self-consistency test
    (``TestDepositGateSelfConsistency``) both call this, so a future conjunct
    added here is the ONE place it must change, and any committed deposit whose
    stored boolean predates the change is flagged by that test — the gate
    definition cannot drift from the committed deposits unnoticed. The private
    ``src.data`` quality filter is a further axis this gate cannot see on the
    mirror; it is noted in the report, never silently assumed away.
    """
    return (
        (not proxy_scale)
        and (not reduced_budget)
        and (not is_thin_evidence)
        and (regime == REGIME_GENERALIZATION)
    )


def _candidate_final_ce_mean(result: dict):
    """Mean candidate-arm ``final_ce_train_loss``, ignoring unrecorded arms.

    Arms recorded before the diagnostic existed carry no ``final_ce_train_loss``
    (or a non-finite one); they are skipped rather than counted as 0 (which
    would falsely label a real generalization run as memorization). Returns
    ``None`` when no candidate arm recorded a finite train CE.
    """
    fces = []
    for prov in result.get("candidate_provenance", []):
        try:
            fv = float(prov.get("final_ce_train_loss"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            fces.append(fv)
    if not fces:
        return None
    return sum(fces) / len(fces)


def _candidate_cost_reduction(result: dict, *, level: int = 1) -> dict | None:
    """Backward-FLOPs reduction the candidate arm's freeze schedule achieves.

    GOAL §4 success is TWO-HEADED — (a) quality preserved (the valid_loss A/B
    the deposit already carries) AND (b) cost reduced vs full backprop (SYSTEM_
    CONSTITUTION condition (b) / the P3 cost gate). This closes the second head:
    it replans the candidate arm's EXACT freeze schedule — the same
    :class:`FreezeScheduleConfig` :func:`arm_valid_loss_9b` builds from
    ``candidate_order`` / ``depth`` / ``warmup_steps`` / ``spacing`` /
    ``total_steps`` — and feeds the realized ``frozen_at_epoch`` to a
    :class:`FreezeCostAccountant` with uniform per-layer cost. That is P3's
    "削減率 = 1 − progressive / full" in model-free first-order arithmetic: the
    reduction is a ratio, so uniform costs give the exact first-order figure,
    and real per-layer costs (DeltaNet vs. Attention, GOAL §1.5/§8) are the
    [UNVERIFIED] model-specific refinement that does not change it.

    Level-1 honesty (§6.2 / constitution verifiability): the candidate arm runs
    Level 1 (weight-grad stop; the activation gradient still traverses the frozen
    layer), so the accountant's arithmetic ``reduction_rate`` OVERSTATES what is
    realized in vivo. ``realized_reduction_rate`` caps it at the validated
    :data:`LEVEL1_REALIZED_REDUCTION_CEILING` (0.0) via
    :func:`realizable_reduction`, so the deposit never presents the arithmetic
    figure as a realized saving — the same realizability correction the §7 speed
    gate applies.

    Returns ``None`` only when the result lacks the candidate-arm schedule keys
    (a partial / legacy result); a plannable schedule always yields a dict
    (depth 0 → reduction 0.0, i.e. full backprop). A present-but-malformed
    schedule raises loudly (a deposit must not silently serialize a broken
    candidate cost axis).
    """
    try:
        order = list(result["candidate_order"])
        active_scope = list(result["active_scope"])
        depth = int(result["depth"])
        warmup_steps = int(result["warmup_steps"])
        spacing = int(result["spacing"])
        total_steps = int(result["total_steps"])
    except (KeyError, TypeError, ValueError):
        return None
    if not active_scope or total_steps < 1 or spacing < 1:
        return None
    # Same config the candidate arm trains under (arm_valid_loss_9b), so the
    # cost axis is measured on the schedule that actually produced the A/B, not
    # a re-derivation with different timing.
    schedule_cfg = FreezeScheduleConfig(
        active_layer_indices=sorted(active_scope),
        num_epochs=total_steps,
        max_depth=depth,
        start_epoch=warmup_steps,
        spacing=spacing,
        policy="convergence_order",
        convergence_order=tuple(order),
    )
    schedule = FreezeSchedule.plan(schedule_cfg)
    layer_costs = {
        idx: LayerBackwardCost(weight_grad_flops=1.0, act_grad_flops=1.0)
        for idx in sorted(active_scope)
    }
    accountant = FreezeCostAccountant(
        layer_costs=layer_costs,
        steps_per_epoch=1,
        num_epochs=total_steps,
        frozen_at_epoch=schedule.frozen_at_epoch,
    )
    summary = accountant.summary(level)
    realized = realizable_reduction(accountant, level)
    return {
        "level": level,
        # P3 headline: the arithmetic backward-FLOPs reduction.
        "reduction_rate": summary.reduction_rate,
        "progressive_backward_flops": summary.progressive_backward_flops,
        "full_backward_flops": summary.full_backward_flops,
        # §6.2: the arithmetic figure corrected for what Level 1 realizes in
        # vivo (~0 under the validated ceiling). This — not reduction_rate — is
        # the realized saving a reader may quote.
        "realized_reduction_rate": realized.realized_reduction,
        "level1_realization_ceiling": LEVEL1_REALIZED_REDUCTION_CEILING,
        "realized_depth": schedule.realized_depth,
        "frozen_at_epoch": {
            str(idx): epoch for idx, epoch in schedule.frozen_at_epoch.items()
        },
        # Flags the uniform first-order model so a reader does not mistake the
        # ratio for measured per-layer FLOPs (the [UNVERIFIED] refinement).
        "cost_model": "uniform_per_layer",
    }


# ── resumable per-arm ledger ─────────────────────────────────────────────────
#
# The full-budget verdict run (`make freeze-validloss-ci-9b-full`) is ~hours of
# 9B GPU (3 candidate + 3 surrogate + 3 baseline arms × 1500 steps). On the
# shared 12 GB card it is routinely preempted — a concurrent private-repo run
# holds the GPU, the OS OOM-killer can claim the process mid-arm, or the session
# can end. ``run_ci_9b`` below collects every arm's result in memory and
# serializes the deposit only at the very end, so a single interruption deep in
# the run discards ALL completed arms and the verdict stays blocked indefinitely
# — the next free-GPU window starts again from zero.
#
# This ledger turns each free-GPU window into banked progress. Each completed
# arm streams to a JSONL ledger keyed by ``(role, index)``; a re-run loads the
# ledger, skips arms whose run-config fingerprint matches, and executes only the
# missing ones — so an interrupted run RESUMES rather than restarts. The ledger
# is opt-in (``--ledger PATH``): with no ledger the run is byte-identical to
# before, so the committed reduced-budget deposits and their replay tests are
# unaffected. The ledger lives under ``runs/`` (gitignored run state), never the
# committed ``tests/fixtures/`` deposit.

LEDGER_VERSION = 1


class IncompleteResumeError(RuntimeError):
    """The resumed ledger lacks a runnable candidate/surrogate pair.

    Raised after loading the ledger when either the candidate or surrogate arm
    set is empty: the headline §4 A/B (:func:`surrogate_valid_loss_ci`) needs at
    least one sample in each arm, so a ledger that banked only candidate arms
    cannot assemble a verdict yet. The completed arms are safe in the ledger —
    the fix is to re-run the same command so the missing arms execute.
    """


class OutputPathDiedDuringRun(RuntimeError):
    """The run's CWD vanished mid-run and the output paths can't survive it.

    Raised at an arm boundary when :func:`cwd_is_alive` is False AND at least one
    of ``--output`` / ``--ledger`` is relative. The host removing the run's
    worktree (e.g. AI Hub recycling a per-instruction worktree while a multi-hour
    ``--total-steps 1500`` run is still on the card) leaves the process training
    — its model/data live in memory — but relative writes resolve to a deleted
    directory, so every remaining arm banks to an unrecoverable void and the
    deposit crashes at the end. Failing loud at the next arm boundary bounds the
    wasted GPU to the arm in progress instead of the whole run. Absolute
    ``--output`` / ``--ledger`` survive CWD death (their writes don't depend on
    it), so the check is skipped for them — the robust fire for a host that
    recycles worktrees faster than a full-budget run completes.
    """


def _config_fingerprint(
    *,
    total_steps: int,
    warmup_steps: int,
    depth: int,
    spacing: int,
    seq_len: int,
    train_examples: int,
    valid_examples: int,
    model: str,
    scope_label: str,
    active_scope,
    dataset: str,
    use_local_loss: bool,
    base_seed: int,
    architecture: str = HOMOGENEOUS,
) -> dict:
    """The run-config identity that defines an arm's result.

    Two arms are interchangeable across runs ONLY when every field matches: the
    same model + scope + data, trained under the same step/depth/spacing/seq-len
    /loss regime, with the same per-arm seed base. A change to any field (e.g.
    bumping ``total_steps`` from a 96-step smoke to the 1500-step full run)
    produces a different fingerprint, so a stale ledger from the old config is
    ignored rather than silently seeding wrong arms. ``active_scope`` (the sorted
    real layer indices) is included — not just its size — so two scopes that
    happen to share a layer count do not collide. ``architecture`` is included
    so a heterogeneous run never reuses a homogeneous arm banked in a ledger
    (or vice versa). ``ledger_version`` gates the whole shape: a future schema
    change reads as a stale ledger.
    """
    return {
        "ledger_version": LEDGER_VERSION,
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "depth": int(depth),
        "spacing": int(spacing),
        "seq_len": int(seq_len),
        "train_examples": int(train_examples),
        "valid_examples": int(valid_examples),
        "model": str(model),
        "scope_label": str(scope_label),
        "active_scope": list(active_scope),
        "dataset": str(dataset),
        "use_local_loss": bool(use_local_loss),
        "base_seed": int(base_seed),
        "architecture": str(architecture),
    }


def _arm_specs(
    *,
    active_indices,
    scope_sorted,
    base_seed: int,
    depth: int,
    n_candidate: int,
    n_surrogate: int,
    n_control: int,
    n_baseline: int,
) -> list[dict]:
    """The ordered list of A/B arms a run must execute.

    Each spec is ``{role, index, order, seed, depth}`` — a complete description
    of one arm, independent of the shared model/batches the runner closes over.
    The ``(role, index)`` pair is the stable resume key; ``order`` is recomputed
    deterministically from ``base_seed``/scope so a resumed arm matches the arm
    that originally banked it. The role/seed/depth assignments mirror the inline
    comprehensions in :func:`run_ci_9b` EXACTLY (candidate: same output-first
    order, seed ``base_seed + i``; surrogate: per-arm random order
    ``random_freeze_order(scope_sorted, base_seed + 1000 + i)``, seed
    ``base_seed + 100 + i``; control: same input-first order, seed
    ``base_seed + 200 + i``; baseline: depth-0 no-freeze, seed
    ``base_seed + 300 + i``) so a fresh run with no ledger is identical to before.
    """
    cand_order = candidate_order_9b(active_indices)
    control_order = control_order_9b(active_indices)
    specs: list[dict] = []
    for i in range(n_candidate):
        specs.append({"role": "candidate", "index": i, "order": tuple(cand_order),
                      "seed": base_seed + i, "depth": depth})
    for i in range(n_surrogate):
        specs.append({"role": "surrogate", "index": i,
                      "order": tuple(random_freeze_order(scope_sorted, base_seed + 1000 + i)),
                      "seed": base_seed + 100 + i, "depth": depth})
    for i in range(n_control):
        specs.append({"role": "control", "index": i, "order": tuple(control_order),
                      "seed": base_seed + 200 + i, "depth": depth})
    for i in range(n_baseline):
        specs.append({"role": "baseline", "index": i, "order": tuple(scope_sorted),
                      "seed": base_seed + 300 + i, "depth": 0})
    return specs


def _ledger_header(fingerprint: dict) -> dict:
    return {"type": "header", "fingerprint": fingerprint}


def _fingerprint_matches(stored, fingerprint: dict) -> bool:
    """True iff the stored fingerprint dict equals the requested one exactly."""
    if not isinstance(stored, dict):
        return False
    return stored == fingerprint


def _ledger_has_matching_header(path, fingerprint: dict) -> bool:
    """True iff ``path`` exists and its first line is a matching header.

    A missing file, an unreadable first line, a non-header first line, or a
    header whose fingerprint differs (incl. ``ledger_version``) all return
    False — the caller then (re)writes a fresh ledger.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        with p.open("r", encoding="utf-8") as fh:
            first = fh.readline().strip()
        rec = json.loads(first)
    except (json.JSONDecodeError, OSError):
        return False
    return rec.get("type") == "header" and _fingerprint_matches(
        rec.get("fingerprint"), fingerprint
    )


def load_ledger(path, fingerprint: dict) -> dict:
    """Read a ledger, returning ``{(role, index): (valid_loss, provenance)}``.

    Returns ``{}`` when the file is absent. The first JSONL line is the header;
    if its ``fingerprint`` does not match, the whole file is treated as stale (a
    previous config's run) and ``{}`` is returned — the run then executes every
    arm fresh and rewrites the ledger with the new header. Malformed lines (a
    partially-flushed trailing line from a crashed append) are skipped with a
    warning rather than aborting the resume, so a torn write cannot brick the
    ledger.
    """
    p = Path(path)
    if not p.exists():
        return {}
    cached: dict = {}
    header_seen = False
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Ledger %s:%d: skipping malformed line — likely a torn "
                    "write from a crashed run; the resume continues.",
                    p, lineno,
                )
                continue
            if not header_seen:
                header_seen = True
                if rec.get("type") != "header" or not _fingerprint_matches(
                    rec.get("fingerprint"), fingerprint
                ):
                    logger.info(
                        "Ledger %s: fingerprint mismatch (stale config) — "
                        "ignoring cached arms and rewriting.", p,
                    )
                    return {}
                continue
            if rec.get("type") == "arm":
                cached[(rec.get("role"), int(rec.get("index")))] = (
                    float(rec["valid_loss"]), rec.get("provenance"),
                )
    return cached


def append_arm_to_ledger(path, fingerprint: dict, spec: dict,
                         valid_loss: float, provenance) -> None:
    """Append one completed arm to the ledger, (re)writing the header first.

    When the file is absent or its header fingerprint is stale (a previous
    config), the file is truncated and a fresh header is written before this
    arm — so a config change never orphans old arms under a new header. When the
    header already matches, only this arm's record is appended. A single
    ``write`` + ``flush`` per arm is the durability unit: an arm is only
    considered banked once its line is on disk, so a crash after the runner
    returns loses at most the in-flight arm.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _ledger_has_matching_header(path, fingerprint):
        mode = "a"
        write_header = False
    else:
        mode = "w"
        write_header = True
    rec = {
        "type": "arm",
        "role": spec["role"],
        "index": int(spec["index"]),
        "seed": int(spec["seed"]),
        "order": list(spec["order"]),
        "valid_loss": float(valid_loss),
        "provenance": provenance,
    }
    with p.open(mode, encoding="utf-8") as fh:
        if write_header:
            fh.write(json.dumps(_ledger_header(fingerprint)) + "\n")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def _collect_arms(
    specs,
    runner,
    *,
    ledger_path=None,
    fingerprint=None,
    output=None,
):
    """Execute the arm specs, banking each to / replaying each from the ledger.

    ``runner`` is a ``callable(spec) -> (valid_loss, provenance)``; the real run
    passes a closure over :func:`arm_valid_loss_9b` (GPU), tests pass a stub.
    Returns ``(collected, n_resumed)`` where ``collected`` maps each role to its
    ordered ``[(valid_loss, provenance), ...]`` list and ``n_resumed`` counts the
    arms served from the ledger (0 when no ledger). With ``ledger_path=None`` no
    ledger is read or written and every spec is executed — the byte-identical
    legacy path.
    """
    cached = load_ledger(ledger_path, fingerprint) if ledger_path is not None else {}
    collected: dict = {"candidate": [], "surrogate": [], "control": [], "baseline": []}
    n_resumed = 0
    for spec in specs:
        role = spec["role"]
        key = (role, spec["index"])
        if key in cached:
            collected[role].append(cached[key])
            n_resumed += 1
            logger.info(
                "Ledger hit: %s[%d] cached (valid_loss=%.6f) — skipping GPU arm.",
                role, spec["index"], cached[key][0],
            )
            continue
        # Dead-CWD trap, mid-run: if the host removed this run's worktree since
        # startup (cwd_is_alive() was True at pre-flight) and a relative
        # --output/--ledger can't survive it, stop NOW rather than spend GPU on
        # an arm that banks to an unrecoverable void. Binds the waste to the arm
        # in progress instead of the whole multi-hour run. Absolute paths skip
        # the check — their writes outlive the CWD (the robust fire).
        if not cwd_is_alive() and not _outputs_are_absolute(output, ledger_path):
            raise OutputPathDiedDuringRun(
                "CWD removed mid-run (the worktree was deleted by the host) and "
                "--output / --ledger are relative, so the remaining arms would "
                "bank to an unrecoverable directory and the deposit would never "
                "land. Stopping at this arm boundary to spare the GPU. The arms "
                "already banked in the --ledger survive a re-fire. Re-fire from "
                "a live worktree, or with absolute --output / --ledger to "
                "survive worktree recycling."
            )
        valid_loss, prov = runner(spec)
        collected[role].append((valid_loss, prov))
        if ledger_path is not None:
            append_arm_to_ledger(ledger_path, fingerprint, spec, valid_loss, prov)
            logger.info(
                "Ledger banked: %s[%d] (valid_loss=%.6f).",
                role, spec["index"], valid_loss,
            )
    return collected, n_resumed


def _require_runnable_arms(candidate_losses, surrogate_losses) -> None:
    """Raise :class:`IncompleteResumeError` unless both headline arms exist.

    The §4 A/B needs ≥1 candidate and ≥1 surrogate to compute the bootstrap CI.
    A resumed ledger that banked only one side cannot assemble a verdict yet;
    the completed arms stay banked and the user re-runs to fill the gap.
    """
    if not candidate_losses or not surrogate_losses:
        raise IncompleteResumeError(
            f"Resume incomplete: candidate arms={len(candidate_losses)}, "
            f"surrogate arms={len(surrogate_losses)} (need >=1 each for the "
            "headline A/B). Completed arms are banked in the ledger; re-run the "
            "same command to execute the missing arms."
        )


def _make_arm_runner(
    model,
    *,
    scope_label,
    active_indices,
    train_batches,
    valid_batches,
    device,
    total_steps,
    warmup_steps,
    spacing,
    lr,
    use_local_loss,
    capture_loss_curve: bool = False,
):
    """Build the per-spec GPU arm runner closure for :func:`_collect_arms`.

    The shared ``model`` is bound as a parameter of this factory (rather than a
    free variable of a closure defined inside :func:`run_ci_9b`) so the binding
    is statically unambiguous — ``run_ci_9b`` later ``del``s the model in its
    ``finally`` to release GPU memory, and a closure capturing that same name in
    the same scope is a false-positive ``F821`` trap. Here ``model`` is a plain
    parameter (never deleted in this scope), and the returned ``runner(spec)``
    dispatches each spec's ``order``/``seed``/``depth`` to
    :func:`arm_valid_loss_9b` on that one shared model — identical to the legacy
    inline comprehensions.

    ``capture_loss_curve`` enables the per-step loss-curve capture for the
    reproducible run-log artifact: the orchestrator seeds each spec with an
    empty ``_loss_curve`` list and this closure forwards it to
    :func:`arm_valid_loss_9b` as the ``loss_curve_sink``. ``False`` (default)
    forwards ``None`` and the arm is byte-identical to the pre-run-log path —
    specs carry no ``_loss_curve`` key and no capture occurs.
    """
    def _runner(spec):
        sink = spec.get("_loss_curve") if capture_loss_curve else None
        return arm_valid_loss_9b(
            model, spec["order"], spec["seed"], scope=scope_label,
            active_indices=active_indices,
            train_batches=train_batches, valid_batches=valid_batches, device=device,
            total_steps=total_steps, warmup_steps=warmup_steps,
            depth=spec["depth"], spacing=spacing, lr=lr,
            use_local_loss=use_local_loss,
            loss_curve_sink=sink,
        )
    return _runner


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
    ledger_path=None,
    output=None,
    architecture: str = HOMOGENEOUS,
    run_log_path=None,
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

    ``ledger_path`` (default ``None``) opts into the RESUMABLE per-arm ledger:
    each completed arm streams to that JSONL file, and a re-run skips arms whose
    run-config fingerprint matches — so a multi-hour full-budget run interrupted
    by GPU preemption / OOM resumes instead of restarting from zero. ``None``
    (the default, and what every committed reduced-budget deposit uses) reads
    and writes no ledger and is byte-identical to a one-shot run.
    """
    scope_label = cfg.training.get("trainable_lora_scope", "all")
    logger.info("Loading tokenizer + model for %s ...", cfg.model.name_or_path)
    tokenizer = load_tokenizer(cfg)
    torch.manual_seed(base_seed)
    model = load_base_model(cfg)
    # Heterogeneous per-layer ranks are structural — they must be set at
    # ``get_peft_model`` time, BEFORE ``configure_trainable_lora_scope`` derives
    # the active set from the applied LoRA params. So the active scope is read
    # pre-LoRA from the base model and (under heterogeneous) asserted to match
    # the post-LoRA scope, so a model-structure surprise fails loud rather than
    # seeding ranks on the wrong layers.
    pre_scope = set(_active_scope_pre_lora(model, scope_label))
    rank_pattern, alpha_pattern, layer_ranks = build_rank_pattern(
        pre_scope, architecture, base_rank=int(cfg.lora.r),
    )
    if architecture == HETEROGENEOUS:
        logger.info(
            "Heterogeneous architecture: per-layer ranks %s over active scope %s",
            layer_ranks, sorted(pre_scope),
        )
    model = apply_lora(
        model, cfg, rank_pattern=rank_pattern, alpha_pattern=alpha_pattern,
    )
    _scope_names, active_indices = configure_trainable_lora_scope(model, scope_label)
    if architecture == HETEROGENEOUS and active_indices != pre_scope:
        raise RuntimeError(
            "Pre-LoRA active scope drifted from the post-LoRA scope under "
            f"heterogeneous ranks: {sorted(pre_scope)} != {sorted(active_indices)}"
        )
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

    # Build the full arm spec list up front (the same candidate/surrogate/
    # control/baseline assignments the inline comprehensions used to make) and
    # the run-config fingerprint that keys the resume ledger. With no ledger
    # these flow straight into the runner and the result is byte-identical to
    # the legacy one-shot path; with a ledger, completed arms replay and only
    # the missing ones hit the GPU.
    specs = _arm_specs(
        active_indices=active_indices, scope_sorted=scope_sorted,
        base_seed=base_seed, depth=depth,
        n_candidate=n_candidate, n_surrogate=n_surrogate,
        n_control=n_control, n_baseline=n_baseline,
    )
    fingerprint = _config_fingerprint(
        total_steps=total_steps, warmup_steps=warmup_steps, depth=depth,
        spacing=spacing, seq_len=seq_len, train_examples=train_examples,
        valid_examples=valid_examples, model=cfg.model.name_or_path,
        scope_label=scope_label, active_scope=sorted(active_indices),
        dataset=dataset, use_local_loss=use_local_loss, base_seed=base_seed,
        architecture=architecture,
    )
    if ledger_path is not None:
        logger.info(
            "Resume ledger enabled: %s (%d arms planned).", ledger_path, len(specs),
        )
    # Seed the per-spec loss-curve sink when a run log was requested, so every
    # planned arm — including ones later replayed from the ledger, which stay
    # empty as an honest "dynamics unavailable for this resumed arm" — carries a
    # ``_loss_curve`` key the runner forwards to ``arm_valid_loss_9b``. No
    # ``run_log_path`` → no seeding → specs carry no key → the runner forwards
    # ``None`` → byte-identical to the pre-run-log path.
    if run_log_path is not None:
        for spec in specs:
            spec["_loss_curve"] = []

    _runner = _make_arm_runner(
        model,
        scope_label=scope_label, active_indices=active_indices,
        train_batches=train_batches, valid_batches=valid_batches, device=device,
        total_steps=total_steps, warmup_steps=warmup_steps,
        spacing=spacing, lr=lr, use_local_loss=use_local_loss,
        capture_loss_curve=bool(run_log_path),
    )

    n_resumed = 0
    try:
        collected, n_resumed = _collect_arms(
            specs, _runner, ledger_path=ledger_path, fingerprint=fingerprint,
            output=output,
        )
        candidate_results = collected["candidate"]
        surrogate_results = collected["surrogate"]
        control_results = collected["control"]
        baseline_results = collected["baseline"]
        candidate_losses = [v for v, _ in candidate_results]
        surrogate_losses = [v for v, _ in surrogate_results]
        control_losses = [v for v, _ in control_results]
        baseline_losses = [v for v, _ in baseline_results]
        _require_runnable_arms(candidate_losses, surrogate_losses)
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
    result = {
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
        # Architecture (homogeneous default = uniform rank, the byte-identical
        # legacy path; heterogeneous = per-layer asymmetric rank, the one open §4
        # leg). ``lora_rank_pattern`` is the {layer: rank} provenance of that
        # asymmetry — None for homogeneous, present for heterogeneous so the
        # deposit is machine-readable about WHICH layers carried which capacity.
        "architecture": architecture,
        "lora_rank_pattern": layer_ranks or None,
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
        # Resume-ledger provenance (GOAL §7): how many arms replayed from the
        # ledger vs ran fresh, and the ledger path (None when no ledger). 0 /
        # None for every one-shot run — the committed reduced-budget deposits —
        # so a resumed full-budget run carries an honest trace of which arms
        # were banked across interruptions rather than recomputed this session.
        "resumed_arm_count": n_resumed,
        "ledger_path": (str(ledger_path) if ledger_path is not None else None),
    }
    # Persist the per-step loss-curve run-log artifact (GOAL §7 reproducibility)
    # when ``--run-log`` was given. Built from the captured per-spec curves AFTER
    # the result dict is assembled (so the run-config header reflects the run
    # that actually executed), then the content hash + path are stamped onto the
    # result for ``result_to_json`` to surface in the deposit. Skipped entirely
    # (no key set) when ``run_log_path`` is None — the deposit stays byte-identical.
    if run_log_path is not None:
        arm_curves = _gather_arm_curves(specs, collected)
        result["run_log_sha256"] = _write_run_log(run_log_path, result, arm_curves)
        result["run_log_path"] = str(run_log_path)
        logger.info(
            "Wrote run log %s (sha256=%s) — %d arm trajectories persisted.",
            run_log_path, result["run_log_sha256"], len(arm_curves),
        )
    return result


# GOAL §7 reproducibility provenance. The deposit's verdict, gate, and regime
# are all DERIVED from the raw measurements; the verdict-replay tests
# (``TestDepositReplayFaithfulness`` / ``TestFullBudgetDepositVerdictPin`` /
# ``TestHeterogeneousTargetScaleDeposit``) guard that derivation — a verdict
# painted on that disagrees with the stored floats fails red. What those tests
# cannot catch is a COORDINATED repaint: editing the committed floats, their CI
# bounds, the verdict label, and the per-arm provenance TOGETHER so every
# derived check still passes. :func:`_evidence_hash` freezes the EVIDENCE
# bytes (the raw measurements a real run produced, never the self-declared
# verdict/gate/regime labels) behind a content hash pinned in the test suite,
# so any such repaint — or any accidental byte drift — becomes a visible,
# reviewable test change instead of silent source-of-truth erosion.
#
# This is the "attach a content hash so the verdict is independently
# reproducible" guard. It does NOT certify that a GPU produced the bytes (only
# a run log / a fresh independent reproduction can — the heterogeneous run has
# no surviving log and the GPU was contended this iteration, recorded openly as
# the next action); it certifies the COMMITTED bytes are the immutable,
# auditable record every derived claim rests on.
EVIDENCE_HASH_KEYS = (
    # Raw held-out measurements + the freeze orders they were taken under.
    "candidate_losses", "surrogate_losses", "control_losses", "baseline_losses",
    "candidate_order", "control_order",
    # Per-arm provenance: which layers froze, how many params trained, and the
    # train-CE diagnostics that classify the regime — all run-determined.
    "candidate_provenance", "surrogate_provenance",
    "control_provenance", "baseline_provenance",
    # Run-determining config: identifies WHICH run produced these measurements.
    "model", "architecture", "lora_rank_pattern", "dataset",
    "total_steps", "warmup_steps", "depth", "spacing",
    "active_scope", "seq_len", "train_examples", "valid_examples",
    "n_candidate", "n_surrogate", "n_control", "n_baseline", "base_seed",
)


def _evidence_hash(deposit: dict) -> str:
    """SHA-256 hex over the deposit's evidence bytes, for reproducibility pinning.

    Canonicalizes a fixed, ordered subset of EVIDENCE keys (see
    :data:`EVIDENCE_HASH_KEYS`) — the raw measurements and run-determining
    config, never the derived verdict/gate/regime labels — to a stable JSON
    encoding (sorted keys, compact separators) and returns the SHA-256 hex.
    ``evidence_hash`` is itself absent from :data:`EVIDENCE_HASH_KEYS`, so
    stamping it is idempotent: a key missing from an older deposit contributes
    ``None`` and the hash never includes itself.
    """
    payload = {k: deposit.get(k) for k in EVIDENCE_HASH_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# GOAL §7 reproducibility — the run-log / loss-curve artifact. ``evidence_hash``
# freezes the deposit's terminal measurements; it cannot certify a GPU produced
# them (only a surviving run log or a fresh reproduction can — the heterogeneous
# run landed without one). The run log is that companion artifact: it carries
# each arm's full per-step training-loss trajectory (the loss curve) plus the
# run-determining config, so an independent reproduction can be checked against
# the recorded *dynamics*, not just the terminal valid_loss samples. Its own
# content hash (``_run_log_sha256``, over canonical bytes — independent of file
# indentation, mirroring :func:`_evidence_hash`) is what the deposit links via
# ``run_log_sha256``: a verifier fetches the artifact, recomputes the hash, and
# confirms the loss curve behind a verdict is the recorded one. Opt-in
# (``--run-log``); unset, the deposit carries ``run_log_sha256: null`` and is
# byte-identical to the pre-run-log shape.
RUN_LOG_SCHEMA_VERSION = 1

# Run-determining config recorded once in the artifact header so the per-arm
# curves are self-identifying independent of the deposit. Mirrors the
# run-config subset of :data:`EVIDENCE_HASH_KEYS`; the per-arm measurements
# (losses, frozen layers) live in each arm entry rather than the header.
_RUN_LOG_CONFIG_KEYS = (
    "model", "architecture", "lora_rank_pattern", "dataset",
    "total_steps", "warmup_steps", "depth", "spacing",
    "active_scope", "seq_len", "train_examples", "valid_examples",
    "n_candidate", "n_surrogate", "n_control", "n_baseline", "base_seed",
)


def _run_log_sha256(payload: dict) -> str:
    """SHA-256 hex over the canonical compact encoding of a run-log payload.

    The artifact is written human-readable (``indent=2, sort_keys=True``) but
    the hash is over the sorted-key compact form, so it is stable across
    whitespace changes and recomputable from the parsed bytes — a verifier
    reads the file, re-canonicalizes, and checks the stamp. Same canonicalization
    discipline as :func:`_evidence_hash`.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _gather_arm_curves(specs, collected) -> list[dict]:
    """Pair each planned spec with its result + captured loss curve.

    Walks ``specs`` in plan order, consuming each role's ``(valid_loss,
    provenance)`` results from ``collected`` in the same order
    :func:`_collect_arms` appended them (so spec ↔ result correspondence holds
    even under resume), and reads the per-step ``_loss_curve`` the orchestrator
    seeded on the spec. A resumed arm (replayed from the ledger, never re-run)
    carries an empty curve — an honest "dynamics unavailable for this arm",
    never a fabricated trajectory. Returns the per-arm entry list the run log
    serializes; pure (no I/O), so it is unit-testable without a GPU.
    """
    role_results = {role: list(seq) for role, seq in collected.items()}
    role_cursor = {role: 0 for role in role_results}
    arms: list[dict] = []
    for spec in specs:
        role = spec["role"]
        cursor = role_cursor[role]
        valid_loss, prov = role_results[role][cursor]
        role_cursor[role] = cursor + 1
        arms.append({
            "role": role,
            "index": int(spec["index"]),
            "seed": int(spec["seed"]),
            "order": list(spec["order"]),
            "frozen_layers": list(prov.get("frozen_layers", [])),
            "n_trainable_params": prov.get("n_trainable_params"),
            "final_valid_loss": float(valid_loss),
            "last_train_loss": prov.get("last_train_loss"),
            "final_ce_train_loss": prov.get("final_ce_train_loss"),
            "loss_curve": [float(x) for x in spec.get("_loss_curve", [])],
        })
    return arms


def _run_log_payload(result: dict, arm_curves: list[dict]) -> dict:
    """Build the run-log artifact body: schema version + run config + arms."""
    return {
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "run_config": {k: result.get(k) for k in _RUN_LOG_CONFIG_KEYS},
        "arms": arm_curves,
    }


def _write_run_log(path, result: dict, arm_curves: list[dict]) -> str:
    """Write the loss-curve run-log artifact; return its content hash.

    Returns ``_run_log_sha256`` over the canonical payload (independent of the
    indented on-disk formatting), so the deposit's ``run_log_sha256`` certifies
    the artifact *content* the verifier recomputes from the file, not its
    whitespace. The parent directory is created best-effort (the pre-flight
    :func:`output_paths_writable` already fail-loud on a dead/unwritable path).
    """
    payload = _run_log_payload(result, arm_curves)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _run_log_sha256(payload)


def result_to_json(result: dict) -> dict:
    """JSON-deposit shape: verdict + real samples + full provenance + honesty."""
    ci = result["ci"]
    reduced = bool(result["reduced_budget"])
    # Training-regime honesty (GOAL §7): the candidate arm's mean train CE vs its
    # held-out valid_loss. A full-budget run that memorized (train CE → 0) must
    # NOT be citable as the complete §4 verdict even though it cleared the budget
    # axis — see :func:`_classify_regime`.
    candidate_ce_mean = _candidate_final_ce_mean(result)
    candidate_valid = ci.candidate_mean
    regime = _classify_regime(candidate_ce_mean, candidate_valid)
    candidate_train_valid_gap = (
        candidate_valid - candidate_ce_mean
        if candidate_ce_mean is not None else None
    )
    deposit = {
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
        # Architecture provenance (ADDITIVE — legacy homogeneous deposits simply
        # lack these and read as the homogeneous default). ``lora_rank_pattern``
        # is None for the uniform-rank homogeneous leg and a {layer: rank} dict
        # for heterogeneous, so the deposit is explicit about which layers
        # carried which adapter capacity.
        "architecture": result.get("architecture", HOMOGENEOUS),
        "lora_rank_pattern": result.get("lora_rank_pattern"),
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
        # GOAL §7 regime honesty: the candidate arm's train CE and the
        # train-valid gap that classifies the run's regime. ADDITIVE — legacy
        # deposits (pre-diagnostic) simply lack these and read as REGIME_UNKNOWN.
        "candidate_final_ce_train_loss_mean": candidate_ce_mean,
        "candidate_train_valid_gap": candidate_train_valid_gap,
        "regime": regime,
        # GOAL §4 cost-reduction head (SYSTEM_CONSTITUTION condition (b) / the
        # P3 "削減率 = 1 − progressive / full" cost gate): the backward-FLOPs
        # reduction the candidate arm's freeze schedule achieves vs full
        # backprop, replanned from the SAME schedule the arm ran under. ADDITIVE
        # — ``None`` only when the result lacks the candidate-arm schedule keys
        # (a partial / legacy result). Level-1 honesty: the realized saving is
        # ``realized_reduction_rate`` (~0 under the validated ceiling), NOT the
        # arithmetic ``reduction_rate``.
        "candidate_cost_reduction": _candidate_cost_reduction(result),
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
        # not reduced), non-thin (enough seeds for the bootstrap to capture
        # variance), AND in the generalization regime (the candidate generalized
        # rather than memorized — without this axis a full-budget run on the
        # default small train set would do tens of epochs, memorize, and still
        # claim the crown). The private ``src.data`` quality filter is a further
        # axis this gate cannot see (absent on the mirror) — it is noted in the
        # report, never silently assumed away. The direction-isolation analysis
        # is an *attribution* caveat on the verdict's interpretation, not a
        # scale/budget axis, so it never opens or closes this gate by itself.
        "citable_as_target_scale": (not result["proxy_scale"]),
        "citable_as_full_section4_verdict": _full_section4_verdict_gate(
            proxy_scale=result["proxy_scale"],
            reduced_budget=reduced,
            is_thin_evidence=ci.is_thin_evidence,
            regime=regime,
        ),
        # Resume-ledger trace (GOAL §7). ADDITIVE — legacy deposits (and any
        # result dict built without the ledger) default to 0 resumed arms and no
        # ledger path, so this never changes a one-shot deposit's verdict.
        "resumed_arm_count": int(result.get("resumed_arm_count", 0) or 0),
        "ledger_path": result.get("ledger_path"),
        # Loss-curve run-log artifact (GOAL §7 reproducibility). ADDITIVE —
        # ``None`` when no ``--run-log`` was given, so a one-shot run deposits
        # byte-identically to the pre-run-log shape. When present, ``run_log_path``
        # is the persisted per-step loss-trajectory artifact and ``run_log_sha256``
        # is its content hash (over canonical bytes, independent of indentation)
        # — the reproducibility companion to ``evidence_hash``: the deposit's
        # evidence hash freezes the terminal measurements; the run log certifies a
        # GPU produced the dynamics behind them. Deliberately NOT in
        # :data:`EVIDENCE_HASH_KEYS` (the path is machine-specific and the run log
        # carries its OWN hash), so it never perturbs the pinned evidence hash.
        "run_log_path": result.get("run_log_path"),
        "run_log_sha256": result.get("run_log_sha256"),
    }
    # GOAL §7 reproducibility provenance — see :func:`_evidence_hash`. Computed
    # last, over the evidence keys only (never the verdict/gate/regime it would
    # be circular to hash), so the stamp is honest about the bytes that follow.
    deposit["evidence_hash"] = _evidence_hash(deposit)
    return deposit


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
    # Regime honesty (GOAL §7): flag-driven so the report never contradicts the
    # machine-readable ``regime`` label. A full-budget run that memorized still
    # fails ``citable_as_full_section4_verdict`` on this axis — the report says
    # so plainly rather than letting a 1500-step memorization run claim the crown.
    _ce_mean = _candidate_final_ce_mean(result)
    _gap = (ci.candidate_mean - _ce_mean) if _ce_mean is not None else None
    _regime = _classify_regime(_ce_mean, ci.candidate_mean)
    _ce_str = f"{_ce_mean:.4f}" if _ce_mean is not None else "n/a"
    _gap_str = f"{_gap:.4f}" if _gap is not None else "n/a"
    lines.append(
        f"  note: REGIME={_regime} — candidate train_CE={_ce_str} vs valid="
        f"{ci.candidate_mean:.4f} (gap={_gap_str}); the complete §4 verdict "
        "requires GENERALIZATION (memorization/overfit/unknown block it even at "
        "full budget)."
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
        "--architecture", default=HOMOGENEOUS, choices=ARCHITECTURES,
        help=(
            "LoRA architecture: homogeneous (default — uniform rank, the "
            "byte-identical legacy path) or heterogeneous (per-layer "
            "asymmetric rank over the active scope — the target-scale "
            "realization of the proxy's discriminating positive control; "
            "the one open §4 research leg). Architecture-independent data "
            "path, verdict gate, and cost accounting."
        ),
    )
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
    p.add_argument(
        "--ledger", default=None,
        help=(
            "Resume ledger path (JSONL). Each completed arm streams to this "
            "file; a re-run skips arms whose run-config fingerprint matches and "
            "executes only the missing ones — so a multi-hour full-budget run "
            "interrupted by GPU preemption / OOM RESUMES instead of restarting "
            "from zero. Recommended for ``--total-steps 1500``: "
            "--ledger runs/freeze_validloss_ci_9b_full_ledger.jsonl. Default "
            "None = no ledger (byte-identical to a one-shot run)."
        ),
    )
    p.add_argument(
        "--run-log", default=None,
        help=(
            "Run-log / loss-curve artifact path (JSON). When set, each arm's "
            "full per-step training-loss trajectory is persisted here alongside "
            "the run-determining config and per-arm provenance, and the deposit "
            "carries run_log_path + a content hash (run_log_sha256) linking it — "
            "the reproducibility companion to evidence_hash (which freezes the "
            "terminal measurements but cannot certify a GPU produced them). "
            "Recommended for any citable run so the verdict is independently "
            "reproducible from the recorded dynamics. Default None = no run log "
            "(the deposit is byte-identical to a no-run-log run)."
        ),
    )
    p.add_argument(
        "--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB,
        help=(
            "Free-GPU-memory floor in GiB for the pre-flight check. If the card "
            "has less free memory than this (a concurrent process is holding it), "
            "the run defers with exit 75 (EX_TEMPFAIL) instead of crashing on "
            "OOM minutes into the first arm — so a poll-loop / resumable "
            "--ledger workflow can simply re-run on the next free-GPU window and "
            "keep the arms it already banked. 0 disables the check. Default "
            f"{DEFAULT_MIN_FREE_GIB} (sits below the seq1024 suffix-only peak; "
            "fires only when another big process is clearly holding the card)."
        ),
    )
    return p


def gpu_free_mib() -> int | None:
    """Free GPU memory in MiB, or ``None`` if it cannot be read.

    Failing open (returning None) is deliberate: this guard exists to spare a
    fragmented-GPU workflow a mid-arm OOM crash, not to block runs on cards
    where ``mem_get_info`` is unavailable (e.g. some driver/WSL setups). The
    caller treats None as "don't know, don't defer."
    """
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except (RuntimeError, AttributeError):
        return None
    return int(free_bytes) // (1024 * 1024)


def gpu_free_memory_deferred(min_free_gib: float) -> str | None:
    """Return a deferral reason string if free GPU memory is below the floor.

    ``None`` means "proceed" (enough free, or the floor is disabled, or free
    memory could not be read — fail open). A non-None string means "defer":
    the caller logs it and exits ``EXIT_GPU_TEMPFAIL`` (75).
    """
    if min_free_gib <= 0:
        return None
    free_mib = gpu_free_mib()
    if free_mib is None:
        return None
    min_free_mib = min_free_gib * 1024.0
    if free_mib < min_free_mib:
        return (
            f"Insufficient free GPU memory: {free_mib} MiB free < "
            f"{min_free_mib:.0f} MiB floor (--min-free-gib {min_free_gib}). "
            f"A concurrent process is likely holding the card. Deferring — "
            f"exit {EXIT_GPU_TEMPFAIL} (EX_TEMPFAIL, retry later). Re-run on "
            f"the next free-GPU window; completed arms stay banked in the "
            f"--ledger."
        )
    return None


def output_paths_writable(
    output: str | None, ledger: str | None, run_log: str | None = None,
) -> str | None:
    """Return a fatal reason if ``--output`` / ``--ledger`` / ``--run-log`` can't be written.

    ``None`` means "both writable, proceed". A non-None string means "abort":
    the parent directory of the given path either does not exist or is not
    writable. This catches the *dead-CWD trap*: a background run launched from
    a worktree that has since been removed keeps training (its model and data
    live in memory / absolute HF-cache paths) but its *relative* output paths
    — ``runs/...ledger.jsonl``, ``tests/fixtures/...json`` — resolve to a
    deleted, now-empty working directory, so the ledger append (first arm
    completion) and the final deposit write both raise ``FileNotFoundError``
    hours later and the verdict never lands. Checking at pre-flight makes that
    failure loud and immediate instead of surfacing only after GPU has been
    spent.

    A missing/unwritable parent is FATAL (the caller exits 1), NOT a tempfail
    (75): waiting on a free GPU does not resurrect a deleted directory, so the
    operator must re-fire from a live worktree. Deliberately checked in main()
    BEFORE the CUDA / GPU-memory gates so a dead CWD is never mislabeled as a
    retryable tempfail and retried forever.

    ``Path.is_dir`` / ``os.access`` resolve the (relative) parent against the
    process CWD; on a removed CWD both return False (ENOENT is swallowed), so
    the trap is detected without a fragile ``os.getcwd()`` that would itself
    raise.
    """
    for flag, path in (
        ("--output", output), ("--ledger", ledger), ("--run-log", run_log),
    ):
        if not path:
            continue
        parent = Path(path).parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            return (
                f"{flag} path {path!r} is not writable: its parent directory "
                f"{parent} does not exist or is not writable. If this run was "
                f"launched from a worktree that has since been removed, every "
                f"relative output path resolves to a deleted directory and the "
                f"verdict deposit can never land (the dead-CWD trap). Re-fire "
                f"this command from a live working directory. Aborting (fatal, "
                f"not a retryable tempfail)."
            )
    return None


def cwd_is_alive() -> bool:
    """True iff the process working directory still exists on disk.

    A background run whose worktree was removed by the host (AI Hub recycling a
    per-instruction worktree mid-run) keeps training — model/data live in memory
    / absolute HF-cache paths — but ``Path.cwd()`` raises ``FileNotFoundError``
    once the directory is unlinked. That is the robust *mid-run* signal for the
    dead-CWD trap: a recreated ``runs/`` subdir would fool a ``parent.is_dir()``
    check (``append_arm_to_ledger`` mkdir()s at write time, so it can resurrect
    ``runs/`` inside the unlinked inode), but the CWD path itself is never
    recreated, so this stays honest after the trap fires.
    """
    try:
        Path.cwd()
        return True
    except (FileNotFoundError, OSError):
        return False


def _outputs_are_absolute(
    output: str | None, ledger: str | None, run_log: str | None = None,
) -> bool:
    """True when every given output path is absolute.

    Absolute ``--output`` / ``--ledger`` / ``--run-log`` survive the run's CWD
    being removed (their writes never resolve against CWD), so a mid-run dead
    CWD is harmless for them — the check in :func:`_collect_arms` is skipped. A
    single relative path makes the run CWD-dependent, so this returns False.
    ``None`` paths impose no constraint.
    """
    for path in (output, ledger, run_log):
        if path and not Path(path).is_absolute():
            return False
    return True


def _ensure_output_parent_dirs(
    output: str | None, ledger: str | None, run_log: str | None = None,
) -> None:
    """Create the ``--output`` / ``--ledger`` / ``--run-log`` parent dirs up front.

    ``append_arm_to_ledger`` and :func:`_write_run_log` already
    ``mkdir(parents=True, exist_ok=True)`` at write time, so a missing parent (a
    fresh worktree with no ``runs/``) is a trivial setup step — NOT the dead-CWD
    trap. Doing it at pre-flight (after the :func:`cwd_is_alive` gate, so a truly
    dead CWD is still caught first) makes a fresh-worktree
    ``make freeze-validloss-ci-9b-full-bg`` proceed instead of FATALing on a
    ``runs/`` that the run would have created anyway. Silently best-effort: a
    path whose parent cannot be created is left for :func:`output_paths_writable`
    to report.
    """
    for path in (output, ledger, run_log):
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Output integrity is the most fundamental pre-flight: the deposit is the
    # entire point of a multi-hour run, checked before CUDA/GPU-memory so it is
    # never mislabeled as the retryable tempfail (75) the launcher would loop on
    # forever. The dead-CWD trap has two faces, disambiguated here:
    #   (1) the CWD itself is gone (worktree removed by the host) — FATAL, the
    #       run can persist NOTHING relative; re-fire live or with absolute
    #       paths. cwd_is_alive() is the robust signal (a recreated runs/ subdir
    #       cannot fool it).
    #   (2) the CWD is live but a parent dir is missing (fresh worktree, no
    #       runs/) — trivially created, NOT fatal; append_arm_to_ledger mkdir()s
    #       at write time anyway.
    if not cwd_is_alive():
        logger.error(
            "CWD has been removed (the process working directory no longer "
            "exists on disk). A background run launched from a worktree the "
            "host has since deleted cannot persist any relative --output / "
            "--ledger, so the verdict deposit can never land (the dead-CWD "
            "trap). Re-fire from a live working directory, or with absolute "
            "--output / --ledger to survive worktree recycling. Aborting "
            "(fatal, not a retryable tempfail)."
        )
        return 1
    _ensure_output_parent_dirs(args.output, args.ledger, args.run_log)
    unwritable = output_paths_writable(args.output, args.ledger, args.run_log)
    if unwritable is not None:
        logger.error("%s", unwritable)
        return 1

    if not torch.cuda.is_available():
        logger.error("CUDA not available — a real 9B run requires a GPU. Aborting.")
        return 2

    deferred = gpu_free_memory_deferred(args.min_free_gib)
    if deferred is not None:
        logger.error("%s", deferred)
        return EXIT_GPU_TEMPFAIL

    cfg = OmegaConf.load(args.config)
    device = resolve_device(args.device)

    try:
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
            ledger_path=args.ledger,
            output=args.output,
            architecture=args.architecture,
            run_log_path=args.run_log,
        )
    except IncompleteResumeError as exc:
        # The resumed ledger banked only one side of the A/B, so no verdict can
        # assemble yet. The completed arms are safe in the ledger; exit nonzero
        # WITHOUT writing a deposit (an honest "not done yet") so the user knows
        # to re-run the same command and fill the gap.
        logger.error("%s", exc)
        return 3
    except OutputPathDiedDuringRun as exc:
        # The worktree was removed mid-run; remaining arms can't persist. FATAL
        # (1) — NOT retryable: the self-retrying launcher would otherwise respawn
        # the worker into the same dead state forever. The arms already banked
        # in the --ledger survive a re-fire from a live CWD.
        logger.error("%s", exc)
        return 1

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
