"""Analyze benchmark evaluation results: baseline vs TG-LoRA metric deltas.

Usage:
    python scripts/analyze_benchmark.py --baseline reports/eval/baseline_results.json --tg-lora reports/eval/tg_lora_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_benchmark_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict) and "results" in data:
        return data
    return {"results": data}


def extract_metrics(data: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results = data.get("results", {})
    if isinstance(results, list):
        for entry in results:
            if isinstance(entry, dict):
                name = entry.get("task_name") or entry.get("config", {}).get("task", "unknown")
                for acc_key in ("acc_norm,none", "acc,none", "acc", "accuracy"):
                    if acc_key in entry:
                        metrics[f"{name}/{acc_key}"] = float(entry[acc_key])
                        break
                for ppl_key in ("perplexity,none", "perplexity", "ppl"):
                    if ppl_key in entry:
                        metrics[f"{name}/{ppl_key}"] = float(entry[ppl_key])
                        break
    elif isinstance(results, dict):
        for task_name, task_results in results.items():
            if isinstance(task_results, dict):
                for key, value in task_results.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{task_name}/{key}"] = float(value)
    return metrics


def compute_deltas(
    baseline_metrics: dict[str, float],
    tg_lora_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    all_keys = sorted(set(baseline_metrics) | set(tg_lora_metrics))
    deltas: dict[str, dict[str, float]] = {}
    for key in all_keys:
        b_val = baseline_metrics.get(key)
        t_val = tg_lora_metrics.get(key)
        entry: dict[str, float] = {}
        if b_val is not None:
            entry["baseline"] = b_val
        if t_val is not None:
            entry["tg_lora"] = t_val
        if b_val is not None and t_val is not None:
            entry["delta"] = t_val - b_val
        deltas[key] = entry
    return deltas


def format_report(deltas: dict[str, dict[str, float]]) -> str:
    lines = ["Benchmark Analysis: Baseline vs TG-LoRA", "=" * 50]
    for key, entry in deltas.items():
        b = entry.get("baseline")
        t = entry.get("tg_lora")
        d = entry.get("delta")
        parts = [key]
        if b is not None:
            parts.append(f"baseline={b:.6f}")
        else:
            parts.append("baseline=N/A")
        if t is not None:
            parts.append(f"tg_lora={t:.6f}")
        else:
            parts.append("tg_lora=N/A")
        if d is not None:
            parts.append(f"delta={d:+.6f}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def analyze(baseline_path: Path, tg_lora_path: Path) -> dict[str, dict[str, float]]:
    baseline_data = load_benchmark_json(baseline_path)
    tg_lora_data = load_benchmark_json(tg_lora_path)
    baseline_metrics = extract_metrics(baseline_data)
    tg_lora_metrics = extract_metrics(tg_lora_data)
    return compute_deltas(baseline_metrics, tg_lora_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark results: baseline vs TG-LoRA")
    parser.add_argument("--baseline", required=True, help="Path to baseline results JSON")
    parser.add_argument("--tg-lora", required=True, help="Path to TG-LoRA results JSON")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    deltas = analyze(Path(args.baseline), Path(args.tg_lora))
    report = format_report(deltas)
    print(report)

    if args.output:
        Path(args.output).write_text(json.dumps(deltas, indent=2), encoding="utf-8")
        print(f"\nJSON saved to {args.output}")


if __name__ == "__main__":
    main()
