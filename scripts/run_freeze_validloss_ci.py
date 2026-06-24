#!/usr/bin/env python
"""Real valid_loss-axis significance run: candidate vs random-order surrogate.

This is the **Category-C attack** the loop had been deferring. Until now every
GOAL §4 statistical layer consumed only *structural* or *fabricated* numbers:

* :func:`src.tg_lora.freeze_surrogate_gate.surrogate_exceedance` gives the
  *structural* verdict (the candidate clears the seeded surrogate on the
  deterministic FLOPs axis) — no valid_loss ever enters it.
* :func:`src.tg_lora.freeze_surrogate_ci.surrogate_valid_loss_ci` is the
  *significance* layer (a bootstrap CI on the candidate-vs-surrogate valid_loss
  difference), but its tests only ever fed it constructed constants — it had
  never seen a valid_loss that came out of an actual training run.

So the §4 question "does the candidate freeze schedule retain quality
*significantly* better than a random-order freeze, or could seed noise alone
explain the gap?" had no executable answer: the helper existed, the samples did
not. The 9B target run that would deposit them is Category-C (needs the private
``src.data`` pipeline, absent from this public mirror) and does not fit a 12 GB
card. This script closes the gap the other way the GOAL §4 statistics allow: it
runs the **real** progressive-freeze trio (the same
:class:`~src.tg_lora.progressive_freeze.ProgressiveFreezeController` +
boundary local loss the prod path uses) on a small but genuinely learnable
proxy, for a candidate (output-first) order and a set of random-order
surrogates across seeds, and hands the resulting *real* valid_loss samples to
:func:`surrogate_valid_loss_ci` — producing the **first significance verdict
grounded in numbers that came out of an actual run**, not a constant.

The harness is honest about what it is and is not (GOAL §7):

* **It is a real run.** The proxy genuinely trains (cross-entropy descends from
  near-uniform to a learned state), the freeze is applied through the real
  controller, and the valid_loss is read off the real forward pass — on GPU when
  one is available (``--device cuda`` / ``auto``), else CPU.
* **It is proxy-scale.** HIDDEN=24 / 6 layers is not the 9B target, so the
  verdict is tagged ``proxy_scale=True`` and the report says so plainly. A
  target-scale run deposits its own samples through the *same* function; the
  verdict then upgrades from proxy-scale to target-scale with zero code change
  — this harness is the wiring, the target run swaps the data source.
* **It does not pre-decide the verdict.** Whether the output-first order
  *significantly* beats a random order at this scale is exactly the empirical
  question; the harness emits whatever the bootstrap CI says
  (``SURPASSES`` / ``TIES`` / ``UNDERSHOOTS``) and never assumes a label.

The fixture is factored from ``tests/test_progressive_freeze_invivo.py`` (the
faithful learnable proxy that already drives the trio against a live backward
pass): a residual connection, a small frozen base (std=0.02), and a
parameter-free ``LayerNorm`` keep the stack stable over depth, so the model
learns rather than saturating. The recorded/proxy dataset is
:func:`make_batches` — a deterministic seeded batch (a fixed, reproducible
proxy dataset, the Category-C artifact the loop can re-run on demand).

Usage::

    # Auto device (cuda if available, else cpu) — the one-shot Category-C run.
    make freeze-validloss-ci
    python -m scripts.run_freeze_validloss_ci

    # Pin CPU for a deterministic CI-suite reproduction; write JSON evidence.
    python -m scripts.run_freeze_validloss_ci --device cpu \\
        --n-candidate 4 --n-surrogate 4 --json --output validloss_ci.json
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.lora_utils import set_trainable_lora_layers
from src.tg_lora.activation_matching import ActivationMatchingLoss
from src.tg_lora.freeze_schedule import (
    FreezeSchedule,
    FreezeScheduleConfig,
    random_freeze_order,
)
from src.tg_lora.freeze_surrogate_ci import (
    SurrogateValidLossCI,
    format_surrogate_valid_loss_ci,
    surrogate_valid_loss_ci,
)
from src.tg_lora.progressive_freeze import ProgressiveFreezeController

# Converged-regime defaults mirror tests/test_progressive_freeze_invivo.py
# (HIDDEN=24 / 6 layers genuinely learns 3.5 -> ~0.4 cross-entropy over 60
# epochs). The make target keeps these for a robust verdict; the test suite
# shrinks the budget via the CLI flags below for a fast, deterministic check.
NUM_LAYERS = 6
HIDDEN = 24
VOCAB = 32
LR = 1.0

# Default run budget (the make target's real GPU/CPU run).
DEFAULT_TOTAL = 60
DEFAULT_WARMUP = 45
DEFAULT_DEPTH = 3
DEFAULT_N_CANDIDATE = 5
DEFAULT_N_SURROGATE = 5
DEFAULT_BASE_SEED = 0


def output_first_order(num_layers: int) -> tuple[int, ...]:
    """The candidate order: freeze output-side first (descending layer index).

    Identical to ``FreezeSchedule``'s ``output_first`` policy resolution
    (``_resolve_order`` descends from the output side). Expressed as a
    ``convergence_order`` tuple so the candidate flows through the *same*
    planner / controller / accountant path as the surrogate — the
    apples-to-apples property :func:`random_freeze_order`'s docstring requires.
    """
    return tuple(range(num_layers - 1, -1, -1))


# ---------------------------------------------------------------------------
# Faithful learnable fixture — factored from tests/test_progressive_freeze_invivo.py
# (residual + small frozen base + param-free norm keeps a depth stack stable).
# ---------------------------------------------------------------------------


class _LoRALinear(nn.Module):
    """Frozen base + trainable LoRA, both on the forward graph."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.base.weight, std=0.02)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.lora_A.t() @ self.lora_B.t()


class _Layer(nn.Module):
    """Residual + parameter-free LayerNorm: keeps activations stable over depth."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = _LoRALinear(hidden)

    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        h = hidden_states + self.proj(hidden_states)  # residual
        return (F.layer_norm(h, (h.shape[-1],)),)  # param-free norm


class _ProxyModel(nn.Module):
    """Tiny causal-LM-shaped model that genuinely runs its decoder layers."""

    def __init__(self, num_layers: int = NUM_LAYERS, hidden: int = HIDDEN,
                 vocab: int = VOCAB) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Layer(hidden) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        del labels
        h = self.embed_tokens(input_ids)
        h = F.layer_norm(h, (HIDDEN,))
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        out = SimpleNamespace()
        out.loss = None
        out.logits = self.lm_head(h)
        return out


def build_model(seed: int, num_layers: int, hidden: int, device) -> _ProxyModel:
    """Seed-built proxy on ``device`` with every LoRA layer trainable."""
    torch.manual_seed(seed)
    model = _ProxyModel(num_layers=num_layers, hidden=hidden)
    set_trainable_lora_layers(model, set(range(num_layers)))
    return model.to(device)


def make_batches(seed: int, *, batch: int = 4, seq: int = 6,
                 device=None, vocab: int = VOCAB) -> list[dict]:
    """The recorded/proxy dataset: one deterministic seeded batch.

    A fixed, reproducible proxy dataset (the Category-C artifact the loop can
    re-run on demand) — identical ``(seed, device)`` yields the identical batch,
    so an arm's data draw is pinned independent of call order.
    """
    g = torch.Generator().manual_seed(seed)
    b = {
        "input_ids": torch.randint(0, vocab, (batch, seq), generator=g),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.randint(0, vocab, (batch, seq), generator=g),
    }
    if device is not None:
        b = {k: v.to(device) for k, v in b.items()}
    return [b]


def lm_loss(logits: torch.Tensor, batch: dict, vocab: int = VOCAB) -> torch.Tensor:
    labels = batch["labels"]
    mask = batch["attention_mask"][:, 1:].float()
    s_logits = logits[:, :-1, :].contiguous()
    s_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(
        s_logits.reshape(-1, vocab), s_labels.reshape(-1), reduction="none"
    ).reshape(s_labels.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def eval_loss(model: nn.Module, batches: list[dict]) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            tot += lm_loss(out.logits, b).item()
            n += 1
    model.train()
    return tot / n


def arm_valid_loss(
    order: Sequence[int],
    seed: int,
    *,
    device,
    total: int,
    warmup: int,
    depth: int,
    num_layers: int = NUM_LAYERS,
    hidden: int = HIDDEN,
) -> float:
    """Train the real progressive-freeze trio under ``order``; return final valid_loss.

    One Category-C sample: build the seeded proxy on ``device``, drive the real
    :class:`ProgressiveFreezeController` with a ``convergence_order`` schedule
    built from ``order`` (so candidate and surrogate share one code path), train
    for ``total`` epochs on the boundary local loss once frozen (else the full
    task loss), and read the final valid_loss off the real forward pass. The
    ``seed`` pins model init and the data draw; ``order`` is supplied by the
    caller (``output_first_order`` for the candidate,
    :func:`random_freeze_order` for a surrogate).
    """
    model = build_model(seed, num_layers, hidden, device)
    batches = make_batches(seed + 10_000, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR)
    config = FreezeScheduleConfig(
        active_layer_indices=list(range(num_layers)),
        num_epochs=total,
        max_depth=depth,
        start_epoch=warmup,
        spacing=1,
        policy="convergence_order",
        convergence_order=tuple(order),
    )
    schedule = FreezeSchedule.plan(config)
    ctrl = ProgressiveFreezeController(
        start_cycle=warmup,
        active_layer_indices=set(range(num_layers)),
        schedule=schedule,
    )
    loss_fn = ActivationMatchingLoss()
    for epoch in range(total):
        ctrl.progress(model, epoch, batches, device)
        for b_idx, b in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            if ctrl.frozen_layers:
                boundary = min(ctrl.frozen_layers)
                loss = ctrl.compute_local_loss(
                    model, b, loss_fn, batch_idx=b_idx, device=device, layer_idx=boundary
                )
            else:
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                loss = lm_loss(out.logits, b)
            loss.backward()
            opt.step()
    return eval_loss(model, batches)


def run_ci(
    *,
    device,
    total: int = DEFAULT_TOTAL,
    warmup: int = DEFAULT_WARMUP,
    depth: int = DEFAULT_DEPTH,
    n_candidate: int = DEFAULT_N_CANDIDATE,
    n_surrogate: int = DEFAULT_N_SURROGATE,
    base_seed: int = DEFAULT_BASE_SEED,
    num_layers: int = NUM_LAYERS,
) -> dict:
    """Run the candidate + surrogate arms and return the §4 significance verdict.

    The candidate arm runs the output-first order under ``n_candidate`` seeds
    (each a fresh model init + data draw); each surrogate arm runs a distinct
    :func:`random_freeze_order` under its own seed. The resulting real
    valid_loss samples feed :func:`surrogate_valid_loss_ci` for the
    significance-graded verdict — the first such verdict grounded in numbers
    from an actual run. Every RNG is locally seeded, so a fixed ``base_seed``
    reproduces the whole sweep bit-for-bit on a given device.
    """
    candidate_losses = [
        arm_valid_loss(
            output_first_order(num_layers), base_seed + i,
            device=device, total=total, warmup=warmup, depth=depth, num_layers=num_layers,
        )
        for i in range(n_candidate)
    ]
    surrogate_losses = [
        arm_valid_loss(
            random_freeze_order(range(num_layers), base_seed + 1000 + i),
            base_seed + 100 + i,
            device=device, total=total, warmup=warmup, depth=depth, num_layers=num_layers,
        )
        for i in range(n_surrogate)
    ]
    ci = surrogate_valid_loss_ci(candidate_losses, surrogate_losses, seed=base_seed)
    return {
        "ci": ci,
        "candidate_losses": candidate_losses,
        "surrogate_losses": surrogate_losses,
        "candidate_order": output_first_order(num_layers),
        "device": str(device),
        "total": total,
        "warmup": warmup,
        "depth": depth,
        "n_candidate": n_candidate,
        "n_surrogate": n_surrogate,
        "base_seed": base_seed,
        "num_layers": num_layers,
        # Proxy-scale honesty (GOAL §7): HIDDEN=24 / 6 layers is not the 9B
        # target, so the verdict is proxy-scale. A target run deposits its own
        # samples through the same function and the label upgrades with no code
        # change — this flag is what a reader checks before citing the verdict.
        "proxy_scale": True,
    }


def resolve_device(name: str):
    """``auto`` -> cuda if available else cpu; otherwise the named device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_freeze_validloss_ci",
        description=(
            "Real GOAL §4 valid_loss-axis significance run: trains the "
            "progressive-freeze trio for an output-first candidate vs "
            "random-order surrogates across seeds and feeds the real "
            "valid_loss samples to surrogate_valid_loss_ci (proxy-scale)."
        ),
    )
    p.add_argument(
        "--device", default="auto",
        help="torch device: 'auto' (cuda if available else cpu), 'cpu', or 'cuda'.",
    )
    p.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="training epochs per arm.")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="freeze start epoch.")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="freeze depth (layers frozen).")
    p.add_argument("--n-candidate", type=int, default=DEFAULT_N_CANDIDATE, help="candidate-arm seeds.")
    p.add_argument("--n-surrogate", type=int, default=DEFAULT_N_SURROGATE, help="surrogate-arm seeds.")
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="sweep base seed.")
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS, help="proxy stack depth.")
    p.add_argument("--json", action="store_true", help="emit JSON evidence to stdout.")
    p.add_argument("--output", default=None, help="write the report/JSON to this path too.")
    return p


def format_report(result: dict) -> str:
    """Human-readable §4 verdict block with full provenance + proxy-scale caveat."""
    ci: SurrogateValidLossCI = result["ci"]
    lines = [
        "freeze_valid_loss_ci — GOAL §4 real valid_loss-axis significance",
        f"  device: {result['device']}  proxy_scale={result['proxy_scale']}  "
        f"(HIDDEN={HIDDEN}, num_layers={result['num_layers']}, depth={result['depth']}, "
        f"epochs={result['total']}, warmup={result['warmup']})",
        f"  candidate_order: {tuple(result['candidate_order'])}  "
        f"n_candidate={result['n_candidate']} n_surrogate={result['n_surrogate']}  "
        f"base_seed={result['base_seed']}",
        "",
        format_surrogate_valid_loss_ci(ci),
        "",
        f"  candidate valid_loss samples: "
        f"{[round(v, 6) for v in result['candidate_losses']]}",
        f"  surrogate valid_loss samples: "
        f"{[round(v, 6) for v in result['surrogate_losses']]}",
    ]
    if result["proxy_scale"]:
        lines.append(
            "  note: PROXY_SCALE — the verdict is from a 24-hidden / "
            f"{result['num_layers']}-layer proxy, not the 9B target. A "
            "target-scale run deposits its own samples through the same "
            "surrogate_valid_loss_ci() and upgrades this label; do not cite "
            "this verdict as a target-scale §4 result."
        )
    return "\n".join(lines)


def result_to_json(result: dict) -> dict:
    ci: SurrogateValidLossCI = result["ci"]
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
        "total": result["total"],
        "warmup": result["warmup"],
        "depth": result["depth"],
        "n_candidate": result["n_candidate"],
        "n_surrogate": result["n_surrogate"],
        "base_seed": result["base_seed"],
        "num_layers": result["num_layers"],
        "proxy_scale": result["proxy_scale"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    result = run_ci(
        device=device,
        total=args.total,
        warmup=args.warmup,
        depth=args.depth,
        n_candidate=args.n_candidate,
        n_surrogate=args.n_surrogate,
        base_seed=args.base_seed,
        num_layers=args.num_layers,
    )
    payload = json.dumps(result_to_json(result), indent=2) if args.json else format_report(result)
    print(payload)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
