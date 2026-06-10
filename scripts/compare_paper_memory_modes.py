#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_METRICS = [
    "warm_tg_wall_seconds",
    "tg_cache_load_seconds",
    "tg_cache_build_seconds",
    "warm_tg_runtime_offload_gpu_allocated_mb_before",
    "warm_tg_runtime_offload_gpu_allocated_mb_after",
    "warm_tg_runtime_offload_gpu_freed_mb",
    "warm_tg_best_valid_loss",
    "warm_tg_gpu_peak_mb",
    "warm_tg_loss_red_per_wall_minute",
    "tg_cache_warm_speedup_pct",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reuse vs one-shot paper-memory aggregate summaries"
    )
    parser.add_argument("--reuse-summary", required=True)
    parser.add_argument("--one-shot-summary", required=True)
    parser.add_argument(
        "--output-base",
        default=f"runs/paper_memory_mode_compare_{datetime.now():%Y%m%d_%H%M%S}",
        help="Output path prefix without extension",
    )
    return parser.parse_args()


def _load_summary(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if "aggregate" in data and "per_seed" in data:
        return data

    if "cold" in data and "warm" in data:
        cold = data["cold"]
        warm = data["warm"]
        row = {
            "seed": 0,
            "warm_tg_wall_seconds": warm["tg_lora"].get("wall_seconds"),
            "tg_cache_load_seconds": warm["tg_lora"].get(
                "prefix_feature_cache_total_load_seconds"
            ),
            "warm_tg_runtime_offload_gpu_allocated_mb_before": warm["tg_lora"].get(
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before"
            ),
            "warm_tg_runtime_offload_gpu_allocated_mb_after": warm["tg_lora"].get(
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after"
            ),
            "warm_tg_runtime_offload_gpu_freed_mb": warm["tg_lora"].get(
                "prefix_feature_cache_runtime_offload_gpu_freed_mb"
            ),
            "warm_tg_best_valid_loss": warm["tg_lora"].get("best_valid_loss"),
        }
        return {
            "seeds": [0],
            "per_seed": [row],
            "aggregate": {
                "warm_tg_wall_seconds": {
                    "mean": warm["tg_lora"].get("wall_seconds")
                },
                "tg_cache_load_seconds": {
                    "mean": warm["tg_lora"].get(
                        "prefix_feature_cache_total_load_seconds"
                    )
                },
                "tg_cache_build_seconds": {
                    "mean": cold["tg_lora"].get(
                        "prefix_feature_cache_total_build_seconds"
                    )
                },
                "warm_tg_runtime_offload_gpu_allocated_mb_before": {
                    "mean": warm["tg_lora"].get(
                        "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before"
                    )
                },
                "warm_tg_runtime_offload_gpu_allocated_mb_after": {
                    "mean": warm["tg_lora"].get(
                        "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after"
                    )
                },
                "warm_tg_runtime_offload_gpu_freed_mb": {
                    "mean": warm["tg_lora"].get(
                        "prefix_feature_cache_runtime_offload_gpu_freed_mb"
                    )
                },
                "warm_tg_best_valid_loss": {
                    "mean": warm["tg_lora"].get("best_valid_loss")
                },
                "warm_tg_gpu_peak_mb": {
                    "mean": warm["tg_lora"].get("gpu_peak_mb")
                },
                "warm_tg_loss_red_per_wall_minute": {
                    "mean": warm["tg_lora"].get("loss_red_per_wall_minute")
                },
                "tg_cache_warm_speedup_pct": {
                    "mean": data.get("delta", {}).get("tg_wall_speedup_pct")
                },
            },
        }

    raise ValueError(
        f"Unsupported paper-memory summary format at {path}; expected aggregate_summary.json or benchmark summary.json"
    )


def _series_mean(summary: dict[str, Any], key: str) -> float | None:
    series = summary.get("aggregate", {}).get(key, {})
    mean = series.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None


def _relative_delta(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference == 0:
        return None
    return (candidate - reference) / abs(reference) * 100.0


def build_mode_comparison(
    reuse_summary: dict[str, Any],
    one_shot_summary: dict[str, Any],
) -> dict[str, Any]:
    paired_by_seed: dict[int, dict[str, Any]] = {}
    reuse_rows = {int(row["seed"]): row for row in reuse_summary.get("per_seed", [])}
    one_shot_rows = {
        int(row["seed"]): row for row in one_shot_summary.get("per_seed", [])
    }
    for seed in sorted(set(reuse_rows) & set(one_shot_rows)):
        reuse_row = reuse_rows[seed]
        one_shot_row = one_shot_rows[seed]
        paired_by_seed[seed] = {
            "seed": seed,
            "warm_tg_wall_seconds": {
                "reuse": reuse_row.get("warm_tg_wall_seconds"),
                "one_shot": one_shot_row.get("warm_tg_wall_seconds"),
                "relative_delta_pct": _relative_delta(
                    reuse_row.get("warm_tg_wall_seconds"),
                    one_shot_row.get("warm_tg_wall_seconds"),
                ),
            },
            "tg_cache_load_seconds": {
                "reuse": reuse_row.get("tg_cache_load_seconds"),
                "one_shot": one_shot_row.get("tg_cache_load_seconds"),
                "relative_delta_pct": _relative_delta(
                    reuse_row.get("tg_cache_load_seconds"),
                    one_shot_row.get("tg_cache_load_seconds"),
                ),
            },
            "warm_tg_runtime_offload_gpu_allocated_mb_before": {
                "reuse": reuse_row.get("warm_tg_runtime_offload_gpu_allocated_mb_before"),
                "one_shot": one_shot_row.get("warm_tg_runtime_offload_gpu_allocated_mb_before"),
                "relative_delta_pct": _relative_delta(
                    reuse_row.get("warm_tg_runtime_offload_gpu_allocated_mb_before"),
                    one_shot_row.get("warm_tg_runtime_offload_gpu_allocated_mb_before"),
                ),
            },
            "warm_tg_runtime_offload_gpu_allocated_mb_after": {
                "reuse": reuse_row.get("warm_tg_runtime_offload_gpu_allocated_mb_after"),
                "one_shot": one_shot_row.get("warm_tg_runtime_offload_gpu_allocated_mb_after"),
                "relative_delta_pct": _relative_delta(
                    reuse_row.get("warm_tg_runtime_offload_gpu_allocated_mb_after"),
                    one_shot_row.get("warm_tg_runtime_offload_gpu_allocated_mb_after"),
                ),
            },
            "warm_tg_runtime_offload_gpu_freed_mb": {
                "reuse": reuse_row.get("warm_tg_runtime_offload_gpu_freed_mb"),
                "one_shot": one_shot_row.get("warm_tg_runtime_offload_gpu_freed_mb"),
                "relative_delta_pct": _relative_delta(
                    reuse_row.get("warm_tg_runtime_offload_gpu_freed_mb"),
                    one_shot_row.get("warm_tg_runtime_offload_gpu_freed_mb"),
                ),
            },
            "warm_tg_best_valid_loss": {
                "reuse": reuse_row.get("warm_tg_best_valid_loss"),
                "one_shot": one_shot_row.get("warm_tg_best_valid_loss"),
                "relative_delta_pct": _relative_delta(
                    reuse_row.get("warm_tg_best_valid_loss"),
                    one_shot_row.get("warm_tg_best_valid_loss"),
                ),
            },
        }

    aggregate_comparison: dict[str, Any] = {}
    for metric in DEFAULT_METRICS:
        reuse_mean = _series_mean(reuse_summary, metric)
        one_shot_mean = _series_mean(one_shot_summary, metric)
        aggregate_comparison[metric] = {
            "reuse_mean": reuse_mean,
            "one_shot_mean": one_shot_mean,
            "absolute_delta": (
                None
                if reuse_mean is None or one_shot_mean is None
                else one_shot_mean - reuse_mean
            ),
            "relative_delta_pct": _relative_delta(reuse_mean, one_shot_mean),
        }

    return {
        "reuse_seeds": reuse_summary.get("seeds", []),
        "one_shot_seeds": one_shot_summary.get("seeds", []),
        "paired_seeds": sorted(paired_by_seed),
        "aggregate": aggregate_comparison,
        "per_seed": [paired_by_seed[seed] for seed in sorted(paired_by_seed)],
    }


def _render_markdown(comparison: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Paper Memory Mode Comparison")
    lines.append("")
    lines.append(f"- reuse seeds: {comparison['reuse_seeds']}")
    lines.append(f"- one-shot seeds: {comparison['one_shot_seeds']}")
    lines.append(f"- paired seeds: {comparison['paired_seeds']}")
    lines.append("")
    lines.append("## Aggregate Means")
    lines.append("")
    lines.append("| Metric | Reuse Mean | One-shot Mean | Absolute Delta | Relative Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for metric, payload in comparison["aggregate"].items():
        reuse_mean = payload["reuse_mean"]
        one_shot_mean = payload["one_shot_mean"]
        absolute_delta = payload["absolute_delta"]
        relative_delta = payload["relative_delta_pct"]
        lines.append(
            "| {metric} | {reuse} | {one_shot} | {absolute} | {relative} |".format(
                metric=metric,
                reuse="-" if reuse_mean is None else f"{reuse_mean:.4f}",
                one_shot="-" if one_shot_mean is None else f"{one_shot_mean:.4f}",
                absolute="-" if absolute_delta is None else f"{absolute_delta:.4f}",
                relative="-" if relative_delta is None else f"{relative_delta:.2f}%",
            )
        )
    if comparison["per_seed"]:
        lines.append("")
        lines.append("## Per Seed")
        lines.append("")
        lines.append("| Seed | Warm TG Wall Reuse (s) | Warm TG Wall One-shot (s) | Load Reuse (s) | Load One-shot (s) | Warm Loss Reuse | Warm Loss One-shot |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in comparison["per_seed"]:
            lines.append(
                "| {seed} | {wall_reuse:.4f} | {wall_one_shot:.4f} | {load_reuse:.4f} | {load_one_shot:.4f} | {loss_reuse:.4f} | {loss_one_shot:.4f} |".format(
                    seed=row["seed"],
                    wall_reuse=row["warm_tg_wall_seconds"]["reuse"],
                    wall_one_shot=row["warm_tg_wall_seconds"]["one_shot"],
                    load_reuse=row["tg_cache_load_seconds"]["reuse"],
                    load_one_shot=row["tg_cache_load_seconds"]["one_shot"],
                    loss_reuse=row["warm_tg_best_valid_loss"]["reuse"],
                    loss_one_shot=row["warm_tg_best_valid_loss"]["one_shot"],
                )
            )
        lines.append("")
        lines.append("### Per Seed Offload Memory")
        lines.append("")
        lines.append("| Seed | GPU Before Reuse (MB) | GPU Before One-shot (MB) | GPU After Reuse (MB) | GPU After One-shot (MB) | GPU Freed Reuse (MB) | GPU Freed One-shot (MB) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in comparison["per_seed"]:
            lines.append(
                "| {seed} | {before_reuse:.4f} | {before_one_shot:.4f} | {after_reuse:.4f} | {after_one_shot:.4f} | {freed_reuse:.4f} | {freed_one_shot:.4f} |".format(
                    seed=row["seed"],
                    before_reuse=row["warm_tg_runtime_offload_gpu_allocated_mb_before"]["reuse"],
                    before_one_shot=row["warm_tg_runtime_offload_gpu_allocated_mb_before"]["one_shot"],
                    after_reuse=row["warm_tg_runtime_offload_gpu_allocated_mb_after"]["reuse"],
                    after_one_shot=row["warm_tg_runtime_offload_gpu_allocated_mb_after"]["one_shot"],
                    freed_reuse=row["warm_tg_runtime_offload_gpu_freed_mb"]["reuse"],
                    freed_one_shot=row["warm_tg_runtime_offload_gpu_freed_mb"]["one_shot"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    reuse_summary = _load_summary(args.reuse_summary)
    one_shot_summary = _load_summary(args.one_shot_summary)
    comparison = build_mode_comparison(reuse_summary, one_shot_summary)

    output_base = Path(args.output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")

    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(_render_markdown(comparison) + "\n")

    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"Mode comparison written to {json_path}")
    print(f"Markdown written to {md_path}")


if __name__ == "__main__":
    main()