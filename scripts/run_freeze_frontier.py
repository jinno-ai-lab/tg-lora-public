#!/usr/bin/env python
"""Reproducible Phase 2 freeze-frontier sweep, exposed as a bare CLI.

Phase 2 (GOAL §3.1 Phase 2 / design §4.1) asks for the frontier curve of
*backward-FLOPs reduction vs freeze depth*, across the three ordering policies
(``output_first`` / ``convergence_order`` / ``compromise``) and both freeze
levels. The ``valid_loss`` axis needs a GPU run (Category C); the
FLOPs-reduction axis is pure arithmetic and is computed *before* any run by
:func:`src.tg_lora.freeze_frontier.frontier`. Until now that function had no
production launch path — the Phase 2 plan was reproducible only by hand-coding a
``FrontierSpec``. This script is the missing launch path (GOAL §7: verify the
mechanism before trusting a GPU run; PURPOSE.md MS-PF2 "frontier sweep の CLI
exposition").

The cost model is the homogeneous-stack first-order model — every active layer
carries the same backward cost — which mirrors
:func:`src.tg_lora.freeze_cost.uniform_layer_accountant`: the
:math:`\\text{reduction\\_rate}` is a ratio, so uniform costs give the exact
first-order reduction for a schedule. Model-specific per-layer costs (DeltaNet
vs Attention, GOAL §1.5/§8) are the [UNVERIFIED] refinement and do not change
the verdict's graduation.

Usage::

    # Default: Qwen3.5-9B's 32-block stack, all 3 policies, Level 1 only.
    python scripts/run_freeze_frontier.py

    # Reproducible machine-readable artifact for the Phase 2 plan.
    python scripts/run_freeze_frontier.py --num-epochs 12 --json --output runs/p2/frontier.json

    # Preview Level 2 (the suffix-cut experiment) alongside the prod Level 1.
    python scripts/run_freeze_frontier.py --levels 1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running as a standalone CLI (``python scripts/run_freeze_frontier.py``):
# a bare invocation puts ``scripts/`` — not the repo root — on sys.path, so make
# the repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tg_lora.freeze_cost import LayerBackwardCost  # noqa: E402
from src.tg_lora.freeze_frontier import FrontierPoint, FrontierSpec, frontier  # noqa: E402
from src.tg_lora.freeze_schedule import VALID_POLICIES  # noqa: E402

# The Qwen3.5-9B target is 32 homogeneous transformer blocks (SYSTEM_CONSTITUTION).
DEFAULT_NUM_LAYERS = 32


def _parse_int_list(raw: str) -> list[int]:
    return [int(tok) for tok in raw.split(",") if tok.strip() != ""]


def _parse_str_list(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip() != ""]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_freeze_frontier",
        description=(
            "Reproducible Phase 2 freeze-frontier sweep (GOAL §3.1 Phase 2 / §7). "
            "Prints the depth → backward-FLOPs-reduction frontier before any GPU run."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=(
            "Number of active layers in the homogeneous stack "
            f"(default {DEFAULT_NUM_LAYERS} = Qwen3.5-9B). Ignored when "
            "--active-layers is given."
        ),
    )
    parser.add_argument(
        "--active-layers",
        type=_parse_int_list,
        default=None,
        help="Comma-separated active layer indices (default: all of range(num-layers)).",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=8, help="Training epochs (default 8)."
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1,
        help="Optimizer steps per epoch (default 1).",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="First epoch at which a layer may freeze (default 1).",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=1,
        help="Epochs between consecutive freezes (default 1).",
    )
    parser.add_argument(
        "--policies",
        type=_parse_str_list,
        default=list(VALID_POLICIES),
        help=f"Comma-separated ordering policies, subset of {VALID_POLICIES} "
        f"(default: all three — the Phase 2 degrees of freedom).",
    )
    parser.add_argument(
        "--levels",
        type=_parse_int_list,
        default=[1],
        help=(
            "Comma-separated freeze levels, subset of 1,2 (default 1). "
            "GOAL §1.6.3: the prod loop drives Level 1 only; Level 2 "
            "(suffix cut) is the Phase 3 experiment, opt in with '1,2'."
        ),
    )
    parser.add_argument(
        "--convergence-order",
        type=_parse_int_list,
        default=None,
        help=(
            "Order for the 'convergence_order' policy (default: active layers in "
            "index order). Must cover the active set."
        ),
    )
    parser.add_argument(
        "--weight-grad-flops",
        type=float,
        default=1.0,
        help="Per-layer weight-gradient FLOPs in the homogeneous model (default 1.0).",
    )
    parser.add_argument(
        "--act-grad-flops",
        type=float,
        default=1.0,
        help="Per-layer activation-gradient propagation FLOPs (default 1.0).",
    )
    parser.add_argument(
        "--optim-state-bytes",
        type=int,
        default=0,
        help="Per-layer optimizer-state bytes freed on freeze (default 0; FLOP ratio only).",
    )
    parser.add_argument(
        "--act-grad-bytes",
        type=int,
        default=0,
        help="Per-layer activation-gradient VRAM bytes freed under Level 2 (default 0).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON to stdout instead of the human table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a reproducible JSON artifact to this path (in addition to stdout).",
    )
    return parser


def build_layer_costs(
    active_indices,
    *,
    weight_grad_flops: float,
    act_grad_flops: float,
    optim_state_bytes: int,
    act_grad_bytes: int,
) -> dict[int, LayerBackwardCost]:
    """Uniform per-layer costs for the homogeneous-stack first-order model."""
    return {
        i: LayerBackwardCost(
            weight_grad_flops=weight_grad_flops,
            act_grad_flops=act_grad_flops,
            optim_state_bytes=optim_state_bytes,
            act_grad_bytes=act_grad_bytes,
        )
        for i in active_indices
    }


def build_spec(args: argparse.Namespace) -> FrontierSpec:
    """Translate parsed CLI args into a validated :class:`FrontierSpec`."""
    active = (
        tuple(args.active_layers)
        if args.active_layers
        else tuple(range(args.num_layers))
    )
    layer_costs = build_layer_costs(
        active,
        weight_grad_flops=args.weight_grad_flops,
        act_grad_flops=args.act_grad_flops,
        optim_state_bytes=args.optim_state_bytes,
        act_grad_bytes=args.act_grad_bytes,
    )
    # convergence_order must cover the active set; the index order is a neutral
    # default the caller overrides with --convergence-order when they have one.
    convergence_order = (
        tuple(args.convergence_order) if args.convergence_order else active
    )
    return FrontierSpec(
        layer_costs=layer_costs,
        steps_per_epoch=args.steps_per_epoch,
        num_epochs=args.num_epochs,
        active_layer_indices=active,
        start_epoch=args.start_epoch,
        spacing=args.spacing,
        policies=tuple(args.policies),
        levels=tuple(args.levels),
        convergence_order=convergence_order,
    )


def format_table(points: list[FrontierPoint]) -> str:
    """Human-readable frontier table, sorted by policy → level → depth."""
    header = (
        f"{'policy':<18} {'lvl':>3} {'depth':>5} {'reduction':>10} "
        f"{'prog_bwd_flops':>15} {'full_bwd_flops':>15} {'vram_saved_B':>13}"
    )
    lines = [header, "-" * len(header)]
    for p in sorted(points, key=lambda q: (q.policy, q.level, q.depth)):
        lines.append(
            f"{p.policy:<18} {p.level:>3} {p.depth:>5} {p.reduction_rate * 100:>9.2f}% "
            f"{p.progressive_backward_flops:>15.1f} {p.full_backward_flops:>15.1f} "
            f"{p.peak_vram_saved_bytes:>13}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = build_spec(args)
    except ValueError as exc:
        # FrontierSpec.__post_init__ validates policies/levels/coverage; surface
        # its message cleanly rather than as a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    points = frontier(spec)

    if args.as_json:
        json.dump([asdict(p) for p in points], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_table(points))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(p) for p in points], indent=2) + "\n", encoding="utf-8"
        )
        print(f"[frontier] wrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
