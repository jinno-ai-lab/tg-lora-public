#!/usr/bin/env python
"""CLI tool for generating training advice from run history.

Reads run_metrics.jsonl files and produces advisory reports using
the TrainingAdvisor module.

Usage::

    # Analyze a single run
    python scripts/advise_training.py runs/my_run/run_metrics.jsonl

    # Output JSON report
    python scripts/advise_training.py runs/my_run/run_metrics.jsonl --json -o report.json

    # Adjust sensitivity
    python scripts/advise_training.py runs/my_run/run_metrics.jsonl --patience 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a standalone CLI (``python scripts/advise_training.py``): a bare
# script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the
# repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tg_lora.training_advisor import (
    AdvisoryReport,
    AdvisorConfig,
    generate_advice_from_history,
)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return records as a list of dicts."""
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    with p.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num}: {e}", file=sys.stderr)
    return records


def _extract_cycle_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract cycle step records from run_metrics.jsonl data."""
    cycles: list[dict[str, Any]] = []
    for rec in records:
        rec_type = rec.get("type", "")
        if rec_type in ("cycle_step", "step"):
            cycles.append({
                "cycle": rec.get("cycle", rec.get("step", 0)),
                "train_loss": rec.get("loss_train", rec.get("train_loss", 0.0)),
                "valid_loss": rec.get("loss_valid", rec.get("valid_loss")),
                "grad_norm": rec.get("grad_norm"),
                "velocity_magnitude": rec.get("velocity_magnitude"),
                "loss_pilot": rec.get("loss_pilot", 0.0),
                "loss_after": rec.get("loss_after", 0.0),
                "tg_lora_accepted": rec.get("tg_lora_accepted"),
            })
    return cycles


def _compute_acceptance_rate(records: list[dict[str, Any]]) -> float | None:
    """Compute overall acceptance rate from cycle records."""
    accepted = sum(1 for r in records if r.get("tg_lora_accepted") is True)
    rejected = sum(1 for r in records if r.get("tg_lora_accepted") is False)
    total = accepted + rejected
    if total == 0:
        return None
    return accepted / total


def _format_report_text(report: AdvisoryReport) -> str:
    """Format advisory report as human-readable text."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Training Advisory Report")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Overall Health: {report.overall_health.upper()}")
    lines.append("=" * 60)

    if report.summary:
        lines.append(f"\nSummary: {report.summary}")

    if report.cycle_health:
        lines.append(f"\nCycle Health: {report.cycle_health.status}")
        if report.cycle_health.divergence.detected:
            d = report.cycle_health.divergence
            lines.append(f"  Divergence: {d.severity} in {d.metric} (value={d.current_value})")
        if report.cycle_health.stagnation.detected:
            s = report.cycle_health.stagnation
            lines.append(f"  Stagnation: {s.cycles_without_improvement} cycles without improvement")

    if report.trajectory_summary:
        lines.append("\nTrajectory Analysis:")
        for k, v in report.trajectory_summary.items():
            lines.append(f"  {k}: {v}")

    if report.actions:
        lines.append("\nRecommended Actions:")
        for i, action in enumerate(report.actions, 1):
            value_str = f" (suggested: {action.suggested_value:.2f})" if action.suggested_value else ""
            lines.append(
                f"  {i}. [{action.priority.upper()}] {action.action_type}: "
                f"{action.reason}{value_str}"
            )
    lines.append("=" * 60)
    return "\n".join(lines)


def _report_to_dict(report: AdvisoryReport) -> dict[str, Any]:
    """Convert AdvisoryReport to a JSON-serializable dict."""
    return {
        "overall_health": report.overall_health,
        "summary": report.summary,
        "timestamp": report.timestamp,
        "actions": [
            {
                "action_type": a.action_type,
                "priority": a.priority,
                "reason": a.reason,
                "suggested_value": a.suggested_value,
                "confidence": a.confidence,
            }
            for a in report.actions
        ],
        "cycle_health": {
            "status": report.cycle_health.status if report.cycle_health else None,
            "divergence": {
                "detected": report.cycle_health.divergence.detected if report.cycle_health else False,
                "severity": report.cycle_health.divergence.severity if report.cycle_health else "",
            } if report.cycle_health else {},
            "stagnation": {
                "detected": report.cycle_health.stagnation.detected if report.cycle_health else False,
                "cycles_without_improvement": (
                    report.cycle_health.stagnation.cycles_without_improvement
                    if report.cycle_health else 0
                ),
            } if report.cycle_health else {},
        } if report.cycle_health else None,
        "trajectory_summary": report.trajectory_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training advisory report from run_metrics.jsonl"
    )
    parser.add_argument(
        "metrics_path",
        help="Path to run_metrics.jsonl file",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of human-readable text",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Stagnation patience (cycles without improvement before warning)",
    )
    parser.add_argument(
        "--spike-threshold", type=float, default=2.0,
        help="Loss spike ratio threshold (default: 2.0)",
    )
    parser.add_argument(
        "--trajectory-window", type=int, default=5,
        help="Window size for trajectory trend analysis",
    )
    args = parser.parse_args()

    config = AdvisorConfig(
        stagnation_patience=args.patience,
        spike_threshold=args.spike_threshold,
        trajectory_window=args.trajectory_window,
    )

    records = _load_jsonl(args.metrics_path)
    if not records:
        print("ERROR: No records found in metrics file", file=sys.stderr)
        sys.exit(1)

    cycle_records = _extract_cycle_records(records)
    if not cycle_records:
        print("ERROR: No cycle step records found", file=sys.stderr)
        sys.exit(1)

    acceptance_rate = _compute_acceptance_rate(cycle_records)

    # Add acceptance rate to the last record for the advisor
    if acceptance_rate is not None and cycle_records:
        cycle_records[-1]["acceptance_rate"] = acceptance_rate

    report = generate_advice_from_history(cycle_records, config=config)

    if args.json:
        output = json.dumps(_report_to_dict(report), indent=2, ensure_ascii=False)
    else:
        output = _format_report_text(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n")
        print(f"Report written to {args.output}")
    else:
        print(output)

    # Exit with non-zero if training is in critical state
    sys.exit(2 if report.overall_health == "critical" else 0)


if __name__ == "__main__":
    main()
