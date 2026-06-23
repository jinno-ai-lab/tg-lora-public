#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as a standalone CLI (``python scripts/benchmark_prefix_cache.py``): a bare
# script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the
# repo root importable so ``src.*`` / ``scripts.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_runs import load_run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cold/warm benchmark for the persistent prefix feature cache "
            "using the existing baseline-vs-TG comparison pipeline."
        )
    )
    parser.add_argument("--budget", type=int, default=32, help="Backward-pass budget")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--quick-eval-examples", type=int, default=4)
    parser.add_argument("--eval-points", type=int, default=2)
    parser.add_argument(
        "--baseline-config",
        default="configs/9b_baseline_suffix_only_last25.yaml",
        help="Baseline YAML config",
    )
    parser.add_argument(
        "--tg-config",
        default="configs/9b_tg_lora_prefix_feature_cache_experimental.yaml",
        help="TG-LoRA YAML config",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/prefix_feature_cache_benchmark",
        help="Persistent prefix-cache directory shared across cold and warm runs",
    )
    parser.add_argument(
        "--output-base",
        default=f"runs/prefix_cache_benchmark_{datetime.now():%Y%m%d_%H%M%S}",
        help="Output directory for the cold/warm benchmark pair",
    )
    parser.add_argument(
        "--venv-python",
        default=os.environ.get("VENV_PYTHON", ".venv/bin/python"),
        help="Python executable used by scripts/run_comparison.sh",
    )
    parser.add_argument(
        "--mlflow-enabled",
        action="store_true",
        help="Enable MLflow logging for the comparison runs",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="Optional CUDA_VISIBLE_DEVICES value propagated to comparison subprocesses",
    )
    return parser.parse_args()


def _loss_reduction_per_wall_minute(records: list[dict[str, Any]], footer: dict[str, Any] | None) -> float | None:
    if not records or footer is None:
        return None
    first_loss = records[0].get("loss_train")
    best_valid = footer.get("best_valid_loss")
    wall_seconds = footer.get("total_wall_seconds")
    if not isinstance(first_loss, (float, int)):
        return None
    if not isinstance(best_valid, (float, int)):
        return None
    if not isinstance(wall_seconds, (float, int)) or wall_seconds <= 0:
        return None
    return float(first_loss - best_valid) / (float(wall_seconds) / 60.0)


def _find_report_path(run_dir: Path) -> str | None:
    reports_dir = run_dir / "reports"
    if not reports_dir.exists():
        return None
    matches = sorted(reports_dir.glob("comparison_*.md"))
    if not matches:
        return None
    return str(matches[-1])


def summarize_comparison_run(run_dir: Path) -> dict[str, Any]:
    baseline_header, baseline_records, baseline_footer = load_run(
        run_dir / "baseline" / "run_metrics.jsonl"
    )
    tg_header, tg_records, tg_footer = load_run(run_dir / "tg_lora" / "run_metrics.jsonl")
    tg_summary = (tg_footer or {}).get("tg_lora_summary", {}) or {}

    return {
        "run_dir": str(run_dir),
        "report_path": _find_report_path(run_dir),
        "baseline": {
            "best_valid_loss": (baseline_footer or {}).get("best_valid_loss"),
            "wall_seconds": (baseline_footer or {}).get("total_wall_seconds"),
            "gpu_peak_mb": (baseline_footer or {}).get("gpu_peak_mb"),
            "loss_red_per_wall_minute": _loss_reduction_per_wall_minute(
                baseline_records, baseline_footer
            ),
            "total_backward_passes": baseline_records[-1].get("total_backward_passes")
            if baseline_records
            else None,
        },
        "tg_lora": {
            "best_valid_loss": (tg_footer or {}).get("best_valid_loss"),
            "wall_seconds": (tg_footer or {}).get("total_wall_seconds"),
            "gpu_peak_mb": (tg_footer or {}).get("gpu_peak_mb"),
            "loss_red_per_wall_minute": _loss_reduction_per_wall_minute(
                tg_records, tg_footer
            ),
            "total_backward_passes": tg_records[-1].get("total_backward_passes")
            if tg_records
            else None,
            "extrapolation_steps": tg_summary.get("extrapolation_steps"),
            "accepted_extrapolations": tg_summary.get("accepted"),
            "acceptance_rate": tg_summary.get("acceptance_rate"),
            "prefix_feature_cache_dir": tg_summary.get("prefix_feature_cache_dir"),
            "prefix_feature_cache_total_build_seconds": tg_summary.get(
                "prefix_feature_cache_total_build_seconds"
            ),
            "prefix_feature_cache_total_load_seconds": tg_summary.get(
                "prefix_feature_cache_total_load_seconds"
            ),
            "prefix_feature_cache_valid_quick_source": tg_summary.get(
                "prefix_feature_cache_valid_quick_source"
            ),
            "prefix_feature_cache_valid_full_source": tg_summary.get(
                "prefix_feature_cache_valid_full_source"
            ),
            "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before": tg_summary.get(
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before"
            ),
            "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after": tg_summary.get(
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after"
            ),
            "prefix_feature_cache_runtime_offload_gpu_freed_mb": tg_summary.get(
                "prefix_feature_cache_runtime_offload_gpu_freed_mb"
            ),
            "prefix_feature_cache_offloaded_prefix_modules": tg_summary.get(
                "prefix_feature_cache_offloaded_prefix_modules"
            ),
            "prefix_feature_cache_offloaded_prefix_parameters": tg_summary.get(
                "prefix_feature_cache_offloaded_prefix_parameters"
            ),
        },
    }


def build_benchmark_summary(cold_dir: Path, warm_dir: Path) -> dict[str, Any]:
    cold = summarize_comparison_run(cold_dir)
    warm = summarize_comparison_run(warm_dir)

    cold_tg = cold["tg_lora"]
    warm_tg = warm["tg_lora"]

    cold_wall = cold_tg.get("wall_seconds")
    warm_wall = warm_tg.get("wall_seconds")
    cold_per_min = cold_tg.get("loss_red_per_wall_minute")
    warm_per_min = warm_tg.get("loss_red_per_wall_minute")

    wall_speedup_pct = None
    if isinstance(cold_wall, (float, int)) and isinstance(warm_wall, (float, int)) and cold_wall > 0:
        wall_speedup_pct = (float(cold_wall) - float(warm_wall)) / float(cold_wall) * 100.0

    per_min_delta_pct = None
    if isinstance(cold_per_min, (float, int)) and isinstance(warm_per_min, (float, int)) and cold_per_min != 0:
        per_min_delta_pct = (float(warm_per_min) - float(cold_per_min)) / abs(float(cold_per_min)) * 100.0

    return {
        "cold": cold,
        "warm": warm,
        "delta": {
            "tg_wall_speedup_pct": wall_speedup_pct,
            "tg_loss_red_per_wall_minute_delta_pct": per_min_delta_pct,
        },
    }


def _run_comparison_once(args: argparse.Namespace, *, run_dir: Path, force_rebuild: bool) -> None:
    env = os.environ.copy()
    env.update(
        {
            "VENV_PYTHON": args.venv_python,
            "BUDGET_PASSES": str(args.budget),
            "OUTPUT_BASE": str(run_dir),
            "BASELINE_CONFIG": args.baseline_config,
            "TG_LORA_CONFIG": args.tg_config,
            "MAX_SEQ_LEN": str(args.max_seq_len),
            "QUICK_EVAL_EXAMPLES": str(args.quick_eval_examples),
            "EVAL_POINTS": str(args.eval_points),
            "MLFLOW_ENABLED": "true" if args.mlflow_enabled else "false",
            "TG_PREFIX_CACHE_DIR": args.cache_dir,
            "TG_PREFIX_FORCE_REBUILD": "true" if force_rebuild else "false",
        }
    )
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    subprocess.run(
        ["bash", "scripts/run_comparison.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_base = repo_root / args.output_base
    cache_dir = repo_root / args.cache_dir
    cold_dir = output_base / "cold"
    warm_dir = output_base / "warm"

    output_base.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cache_dir, ignore_errors=True)

    _run_comparison_once(args, run_dir=cold_dir, force_rebuild=False)
    _run_comparison_once(args, run_dir=warm_dir, force_rebuild=False)

    summary = build_benchmark_summary(cold_dir, warm_dir)
    summary_path = output_base / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()