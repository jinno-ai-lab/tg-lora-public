#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analysis.extrapolation_predictability import (
    analyze_update_predictability,
    load_update_steps_from_artifacts,
)


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated int list") from exc
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all N values must be positive")
    return values


def parse_float_list(value: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated float list"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one threshold")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate whether EMA velocity extrapolation predicts future LoRA "
            "update directions from saved trajectory delta artifacts."
        )
    )
    parser.add_argument(
        "artifact_dir",
        help="Directory containing trajectory_delta_artifacts/*.pt",
    )
    parser.add_argument(
        "--anchor-kind",
        default="after_optimizer_step",
        help="Artifact anchor_kind to analyze (default: after_optimizer_step)",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["cumulative", "direct"],
        default="cumulative",
        help="Whether artifacts store cumulative deltas or direct updates.",
    )
    parser.add_argument(
        "--n-values",
        type=parse_int_list,
        default=[1, 2, 5, 10, 20],
        help="Comma-separated future horizons to test.",
    )
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=10)
    parser.add_argument(
        "--consistency-thresholds",
        type=parse_float_list,
        default=[0.5, 0.7, 0.9],
        help="Comma-separated short/long EMA cosine thresholds.",
    )
    parser.add_argument(
        "--yes-threshold",
        type=float,
        default=0.5,
        help="Mean future cosine threshold used for predictable=yes.",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Disable random-direction and shuffled-trajectory control baselines.",
    )
    parser.add_argument(
        "--control-seed",
        type=int,
        default=0,
        help="Seed for random/shuffled control baselines.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    updates = load_update_steps_from_artifacts(
        args.artifact_dir,
        anchor_kind=args.anchor_kind,
        delta_mode=args.delta_mode,
    )
    report = analyze_update_predictability(
        updates,
        n_values=args.n_values,
        short_window=args.short_window,
        long_window=args.long_window,
        consistency_thresholds=args.consistency_thresholds,
        yes_threshold=args.yes_threshold,
        include_controls=not args.no_controls,
        control_seed=args.control_seed,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(_format_report(report))
    return 0


def _format_report(report: dict[str, Any]) -> str:
    lines = [
        "Extrapolation Predictability",
        f"updates={report['update_count']} short_window={report['short_window']} "
        f"long_window={report['long_window']} yes_threshold={report['yes_threshold']}",
        "",
        "N  samples  pred  cos_future_long  cos_future_short  cos_short_long",
    ]
    for n_key, entry in report["per_n"].items():
        if entry["sample_count"] == 0:
            lines.append(
                f"{n_key:>2} {0:>8}  no    n/a              n/a               n/a"
            )
            continue
        pred = "yes" if entry["predictable"] else "no"
        lines.append(
            f"{int(n_key):>2} {entry['sample_count']:>8}  {pred:<3}  "
            f"{entry['mean_future_cos_long']:+.4f}           "
            f"{entry['mean_future_cos_short']:+.4f}            "
            f"{entry['mean_consistency_cos']:+.4f}"
        )
    if report.get("include_controls"):
        lines.extend(
            [
                "",
                "Controls",
                "N  random_cos  shuffle_cos_long  first_half_cos  second_half_cos",
            ]
        )
        for n_key, entry in report["per_n"].items():
            if entry["sample_count"] == 0:
                lines.append(
                    f"{n_key:>2}  n/a         n/a               n/a             n/a"
                )
                continue
            split = entry["split_by_anchor"]
            lines.append(
                f"{int(n_key):>2}  "
                f"{_fmt_optional(entry['mean_random_cos'])}     "
                f"{_fmt_optional(entry['mean_shuffle_cos_long'])}           "
                f"{_fmt_optional(split['first_half']['mean_future_cos_long'])}         "
                f"{_fmt_optional(split['second_half']['mean_future_cos_long'])}"
            )
    return "\n".join(lines)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return " n/a   "
    return f"{value:+.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
