#!/usr/bin/env python
"""Cross-configuration experiment matrix comparator for TG-LoRA.

Auto-discovers experiments under a runs directory, builds a comparison matrix
of config parameters vs outcome metrics, and ranks experiments.

Usage::

    python scripts/compare_experiment_configs.py --run-base runs/
    python scripts/compare_experiment_configs.py --run-base runs/ --format json
    python scripts/compare_experiment_configs.py --run-base runs/ --metric best_valid_loss
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from src.utils.run_query import list_runs, parse_jsonl


@dataclass
class ExperimentSummary:
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class ComparisonMatrix:
    experiments: list[ExperimentSummary] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    metrics_cols: list[str] = field(default_factory=list)


CONFIG_PARAMS = [
    "tg_lora_K", "tg_lora_N", "tg_lora_alpha", "tg_lora_beta",
    "learning_rate", "batch_size", "lora_r", "lora_alpha",
    "grad_accumulation", "seed",
]

OUTCOME_METRICS = [
    "best_valid_loss", "final_train_loss", "total_wall_seconds",
    "perplexity", "total_backward_passes",
]


def discover_experiments(run_base: str | Path) -> list[ExperimentSummary]:
    runs = list_runs(run_base)
    summaries: list[ExperimentSummary] = []

    for run in runs:
        exp = ExperimentSummary(
            run_id=run.get("run_id", "unknown"),
            config={},
            metrics={},
        )

        for param in CONFIG_PARAMS:
            if param in run and run[param] is not None:
                exp.config[param] = run[param]

        for metric in OUTCOME_METRICS:
            if metric in run and run[metric] is not None:
                exp.metrics[metric] = run[metric]

        jsonl_path = Path(run.get("_jsonl_path", ""))
        if jsonl_path.exists():
            try:
                records = parse_jsonl(jsonl_path)
                header = next((r for r in records if r.get("type") == "run_header"), {})
                for param in CONFIG_PARAMS:
                    if param not in exp.config and param in header:
                        exp.config[param] = header[param]

                steps = [r for r in records if r.get("type") == "step"]
                if steps:
                    exp.metrics.setdefault("initial_loss", steps[0].get("loss_train"))
                    exp.metrics.setdefault("final_loss", steps[-1].get("loss_train"))
                    accepted = sum(1 for s in steps if s.get("tg_lora_accepted"))
                    exp.metrics["acceptance_rate"] = accepted / len(steps) if steps else 0.0
            except (ValueError, orjson.JSONDecodeError) as e:
                exp.parse_warnings.append(f"Failed to parse {jsonl_path.name}: {e}")

        summaries.append(exp)

    return summaries


def build_comparison_matrix(
    experiments: list[ExperimentSummary],
) -> ComparisonMatrix:
    all_params: set[str] = set()
    all_metrics: set[str] = set()
    for exp in experiments:
        all_params.update(exp.config.keys())
        all_metrics.update(exp.metrics.keys())

    params = sorted(all_params)
    metrics = sorted(all_metrics)

    return ComparisonMatrix(
        experiments=experiments,
        parameters=params,
        metrics_cols=metrics,
    )


def rank_experiments(
    matrix: ComparisonMatrix,
    metric: str = "best_valid_loss",
) -> list[dict[str, Any]]:
    valid = [
        (i, exp) for i, exp in enumerate(matrix.experiments)
        if exp.metrics.get(metric) is not None
    ]

    lower_is_better = metric in ("best_valid_loss", "final_train_loss", "total_wall_seconds")

    sorted_exps = sorted(
        valid,
        key=lambda x: x[1].metrics[metric],
        reverse=not lower_is_better,
    )

    ranked: list[dict[str, Any]] = []
    for rank, (orig_idx, exp) in enumerate(sorted_exps, 1):
        ranked.append({
            "rank": rank,
            "run_id": exp.run_id,
            metric: exp.metrics[metric],
            "config": exp.config,
        })

    return ranked


def format_as_markdown(matrix: ComparisonMatrix, metric: str = "best_valid_loss") -> str:
    ranked = rank_experiments(matrix, metric)
    if not ranked:
        return "No experiments with the specified metric found.\n"

    cols = ["rank", "run_id", metric] + matrix.parameters
    cols = [c for c in cols if any(r.get("config", {}).get(c) is not None or c in r for r in ranked)]

    lines: list[str] = []
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(f"# Experiment Comparison (ranked by {metric})")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for r in ranked:
        vals = []
        for c in cols:
            if c == "rank":
                vals.append(str(r["rank"]))
            elif c == "run_id":
                vals.append(str(r["run_id"]))
            elif c == metric:
                v = r.get(metric)
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            else:
                v = r.get("config", {}).get(c)
                if v is None:
                    vals.append("—")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    all_warnings = [w for e in matrix.experiments for w in e.parse_warnings]
    if all_warnings:
        lines.append("")
        lines.append("## Parse Warnings")
        for w in all_warnings:
            lines.append(f"- {w}")

    return "\n".join(lines) + "\n"


def format_as_json(matrix: ComparisonMatrix, metric: str = "best_valid_loss") -> dict[str, Any]:
    ranked = rank_experiments(matrix, metric)
    all_warnings = [w for e in matrix.experiments for w in e.parse_warnings]
    result: dict[str, Any] = {
        "metric": metric,
        "num_experiments": len(matrix.experiments),
        "parameters": matrix.parameters,
        "ranked_experiments": ranked,
    }
    if all_warnings:
        result["parse_warnings"] = all_warnings
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-configuration experiment comparison")
    parser.add_argument("--run-base", required=True, help="Base directory containing experiment runs")
    parser.add_argument("--metric", default="best_valid_loss", help="Metric to rank by")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()

    experiments = discover_experiments(args.run_base)
    if not experiments:
        print(f"No experiments found in {args.run_base}", file=sys.stderr)
        sys.exit(1)

    for exp in experiments:
        for w in exp.parse_warnings:
            print(f"WARNING: {w}", file=sys.stderr)

    matrix = build_comparison_matrix(experiments)

    if args.format == "json":
        result = format_as_json(matrix, args.metric)
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = format_as_markdown(matrix, args.metric)

    print(output)

    if args.output:
        Path(args.output).write_text(output)
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
