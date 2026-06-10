#!/usr/bin/env python
"""Analyze training trajectory from run metrics.

Reads run_metrics.json or a JSONL log of training cycles and produces
a trajectory analysis report with convergence prediction and early-stop
recommendations.

Usage::

    python scripts/analyze_trajectory.py runs/.../run_metrics.json
    python scripts/analyze_trajectory.py --from-losses 2.5,2.3,2.1,1.9,1.8
    python scripts/analyze_trajectory.py --from-file metrics.jsonl --output report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.tg_lora.trajectory import TrajectoryAnalyzer
from src.training.trajectory_artifact_anomalies import \
    summarize_trajectory_artifact_anomalies


def _load_run_metrics(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(2)

    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in text.strip().splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            cycles = data.get("cycles", data.get("per_cycle", []))
            if isinstance(cycles, list):
                records = cycles
            else:
                records = [data]
        else:
            print(f"ERROR: unexpected format in {p}", file=sys.stderr)
            sys.exit(2)
    return records


def _parse_loss_list(s: str) -> list[float]:
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as e:
        print(f"ERROR: invalid loss list: {e}", file=sys.stderr)
        sys.exit(2)


def _build_report(
    analyzer: TrajectoryAnalyzer,
    target_loss: float | None = None,
    patience: int = 5,
    delta_artifact_anomalies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report = analyzer.full_report(target_loss=target_loss, patience=patience)

    return {
        "total_points": report.total_points,
        "loss_trend": report.loss_trend,
        "volatility": report.volatility,
        "convergence": {
            "converged": report.convergence.converged,
            "remaining_steps": report.convergence.remaining_steps,
            "predicted_final_loss": report.convergence.predicted_final_loss,
            "convergence_rate": report.convergence.convergence_rate,
            "confidence": report.convergence.confidence,
        },
        "early_stop": {
            "should_stop": report.early_stop.should_stop,
            "reason": report.early_stop.reason,
            "estimated_gain": report.early_stop.estimated_gain_from_continuing,
            "optimal_cycle": report.early_stop.optimal_cycle,
        },
        "anomalies": {
            "detected": report.anomaly_detected,
            "details": report.anomaly_details,
        },
        "delta_artifact_anomalies": delta_artifact_anomalies or [],
    }


def _print_report(report: dict[str, Any]) -> None:
    conv = report["convergence"]
    es = report["early_stop"]
    anom = report["anomalies"]

    print("=" * 55)
    print("  TG-LoRA Training Trajectory Analysis")
    print("=" * 55)
    print(f"  Data points analyzed: {report['total_points']}")
    print(f"  Loss trend (recent):  {report['loss_trend']:.6f}")
    print(f"  Volatility:           {report['volatility']:.6f}")
    print()
    print("  Convergence:")
    print(f"    Converged:          {conv['converged']}")
    print(f"    Convergence rate:   {conv['convergence_rate']:.6f}")
    print(f"    Remaining steps:    {conv['remaining_steps']}")
    print(f"    Predicted final:    {conv['predicted_final_loss']}")
    print(f"    Confidence:         {conv['confidence']:.2f}")
    print()
    print("  Early Stop Advice:")
    print(f"    Should stop:        {es['should_stop']}")
    print(f"    Reason:             {es['reason'] or 'continue training'}")
    print(f"    Est. remaining gain: {es['estimated_gain']:.6f}")
    print(f"    Best cycle:         {es['optimal_cycle']}")
    print()
    print("  Anomalies:")
    if anom["detected"]:
        for detail in anom["details"]:
            print(f"    - {detail}")
    else:
        print("    None detected")
    artifact_anomalies = report.get("delta_artifact_anomalies", [])
    print()
    print("  Delta Artifact Anomalies:")
    if artifact_anomalies:
        for anomaly in artifact_anomalies:
            cycle_or_step = (
                f"cycle {anomaly['cycle']}"
                if anomaly.get("cycle") is not None
                else f"step {anomaly['step']}"
            )
            print(
                "    - "
                f"{anomaly['anchor_kind']} @ {cycle_or_step}: "
                f"norm={anomaly['delta_total_norm']:.4f}, "
                f"z={anomaly['robust_z_score']:.2f}"
            )
            for example in anomaly.get("source_examples", [])[:3]:
                locator_bits = []
                if example.get("record_id") is not None:
                    locator_bits.append(f"id={example['record_id']}")
                locator_bits.append(f"idx={example['dataset_index']}")
                print(
                    f"      source [{', '.join(locator_bits)}]: {example['text_preview']}"
                )
    else:
        print("    None detected")
    print("=" * 55)

    status = "STOP" if es["should_stop"] else "CONTINUE"
    print(f"  Recommendation: {status}")
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze training trajectory and predict convergence"
    )
    parser.add_argument("metrics_file", nargs="?", help="Path to run_metrics.json or JSONL file")
    parser.add_argument(
        "--from-losses",
        help="Comma-separated loss values for quick analysis",
    )
    parser.add_argument(
        "--target-loss", type=float, default=None,
        help="Target loss for convergence estimation",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early-stop patience (default: 5)",
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="Trend analysis window (default: 5)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON report path",
    )
    args = parser.parse_args()

    if args.from_losses:
        losses = _parse_loss_list(args.from_losses)
        analyzer = TrajectoryAnalyzer.from_loss_history(
            losses, window=args.window,
        )
    elif args.metrics_file:
        records = _load_run_metrics(args.metrics_file)
        analyzer = TrajectoryAnalyzer.from_dicts(
            records, window=args.window,
        )
    else:
        print("ERROR: provide metrics_file or --from-losses", file=sys.stderr)
        parser.print_help()
        sys.exit(2)

    if len(analyzer.points) < 2:
        print("ERROR: need at least 2 data points for analysis", file=sys.stderr)
        sys.exit(2)

    artifact_anomalies: list[dict[str, Any]] = []
    if args.metrics_file:
        artifact_anomalies = summarize_trajectory_artifact_anomalies(
            Path(args.metrics_file).resolve().parent,
        )

    report = _build_report(
        analyzer,
        target_loss=args.target_loss,
        patience=args.patience,
        delta_artifact_anomalies=artifact_anomalies,
    )

    _print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
