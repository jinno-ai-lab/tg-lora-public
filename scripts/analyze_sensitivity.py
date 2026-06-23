#!/usr/bin/env python
"""Hyperparameter sensitivity analysis for TG-LoRA experiments.

Loads sweep results from multiple runs, computes correlation matrices between
hyperparameters and outcome metrics, and ranks parameters by sensitivity.

Usage::

    python scripts/analyze_sensitivity.py --run-dir runs/sweep/
    python scripts/analyze_sensitivity.py --run-dir runs/ --output sensitivity_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Allow running as a standalone CLI (``python scripts/analyze_sensitivity.py``): a bare
# script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the
# repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run_query import list_runs, parse_jsonl


def load_sweep_results(run_dir: str | Path) -> list[dict[str, Any]]:
    runs = list_runs(run_dir)
    results: list[dict[str, Any]] = []
    for run in runs:
        entry: dict[str, Any] = {
            "run_id": run.get("run_id"),
            "mode": run.get("mode"),
            "best_valid_loss": run.get("best_valid_loss"),
            "final_train_loss": run.get("final_train_loss"),
            "total_wall_seconds": run.get("total_wall_seconds"),
        }
        jsonl_path = Path(run.get("_jsonl_path", ""))
        if jsonl_path.exists():
            try:
                records = parse_jsonl(jsonl_path)
                steps = [r for r in records if r.get("type") == "step"]
                header = next((r for r in records if r.get("type") == "run_header"), {})
            except (ValueError, Exception):
                steps = []
                header = {}
        else:
            steps = []
            header = {}

        config_fields = [
            "tg_lora_K", "tg_lora_N", "tg_lora_alpha", "tg_lora_beta",
            "learning_rate", "batch_size", "lora_r", "lora_alpha",
        ]
        for field in config_fields:
            if field in header:
                entry[field] = header[field]
            elif steps:
                val = steps[-1].get(field)
                if val is not None:
                    entry[field] = val

        if steps:
            entry["initial_loss"] = steps[0].get("loss_train")
            entry["final_loss"] = steps[-1].get("loss_train")
            entry["total_backward_passes"] = steps[-1].get("total_backward_passes")

        results.append(entry)
    return results


def _pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
    var_x = sum((xi - mean_x) ** 2 for xi in x) / (n - 1)
    var_y = sum((yi - mean_y) ** 2 for yi in y) / (n - 1)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def compute_correlation_matrix(
    results: list[dict[str, Any]],
    params: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    if params is None:
        param_candidates = [
            "tg_lora_K", "tg_lora_N", "tg_lora_alpha", "tg_lora_beta",
            "learning_rate", "batch_size", "lora_r", "lora_alpha",
        ]
        params = [p for p in param_candidates if any(r.get(p) is not None for r in results)]

    if metrics is None:
        metrics = ["best_valid_loss", "final_train_loss", "total_wall_seconds"]
        metrics = [m for m in metrics if any(r.get(m) is not None for r in results)]

    matrix: dict[str, dict[str, float]] = {}
    for param in params:
        matrix[param] = {}
        for metric in metrics:
            pairs = [
                (float(r[param]), float(r[metric]))
                for r in results
                if r.get(param) is not None and r.get(metric) is not None
            ]
            if len(pairs) >= 2:
                xs, ys = zip(*pairs)
                matrix[param][metric] = round(_pearson_r(list(xs), list(ys)), 4)
            else:
                matrix[param][metric] = 0.0

    return matrix


def rank_sensitivity(
    correlations: dict[str, dict[str, float]],
) -> list[tuple[str, float]]:
    sensitivity: dict[str, float] = {}
    for param, metric_corrs in correlations.items():
        abs_corrs = [abs(v) for v in metric_corrs.values()]
        sensitivity[param] = sum(abs_corrs) / len(abs_corrs) if abs_corrs else 0.0

    ranked = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    return ranked


def generate_sensitivity_report(
    results: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    correlations = compute_correlation_matrix(results)
    rankings = rank_sensitivity(correlations)

    report: dict[str, Any] = {
        "num_experiments": len(results),
        "correlation_matrix": correlations,
        "sensitivity_ranking": [
            {"parameter": param, "avg_abs_correlation": round(score, 4)}
            for param, score in rankings
        ],
    }

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis")
    parser.add_argument("--run-dir", required=True, help="Directory containing sweep runs")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    parser.add_argument("--params", nargs="*", help="Parameters to analyze")
    parser.add_argument("--metrics", nargs="*", help="Metrics to analyze")
    args = parser.parse_args()

    results = load_sweep_results(args.run_dir)
    if not results:
        print(f"No runs found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    if args.params or args.metrics:
        correlations = compute_correlation_matrix(results, args.params, args.metrics)
        rankings = rank_sensitivity(correlations)
        report: dict[str, Any] = {
            "num_experiments": len(results),
            "correlation_matrix": correlations,
            "sensitivity_ranking": [
                {"parameter": p, "avg_abs_correlation": round(s, 4)}
                for p, s in rankings
            ],
        }
    else:
        report = generate_sensitivity_report(results, args.output)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
