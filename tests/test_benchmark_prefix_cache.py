from pathlib import Path

import orjson

from scripts.benchmark_prefix_cache import (build_benchmark_summary,
                                            summarize_comparison_run)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record) + b"\n")


def _baseline_records(best_valid_loss: float, wall_seconds: float) -> list[dict]:
    return [
        {
            "type": "run_header",
            "model_name": "test-model",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 2e-4,
            "seed": 42,
        },
        {
            "type": "step",
            "step": 4,
            "loss_train": 3.0,
            "total_backward_passes": 32,
            "elapsed_seconds": wall_seconds,
        },
        {
            "type": "run_footer",
            "total_wall_seconds": wall_seconds,
            "best_valid_loss": best_valid_loss,
            "final_train_loss": 2.4,
            "best_valid_step": 4,
            "gpu_peak_mb": 8000,
        },
    ]


def _tg_records(
    *,
    best_valid_loss: float,
    wall_seconds: float,
    build_seconds: float,
    load_seconds: float,
    quick_source: str,
    full_source: str,
    offload_before_mb: float = 12000.0,
    offload_after_mb: float = 8200.0,
    offload_freed_mb: float = 3800.0,
) -> list[dict]:
    return [
        {
            "type": "run_header",
            "model_name": "test-model",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 2e-4,
            "seed": 42,
        },
        {
            "type": "step",
            "step": 8,
            "loss_train": 2.8,
            "total_backward_passes": 16,
            "elapsed_seconds": wall_seconds - 5,
            "tg_lora_N": 1,
            "tg_lora_accepted": True,
        },
        {
            "type": "run_footer",
            "total_wall_seconds": wall_seconds,
            "best_valid_loss": best_valid_loss,
            "final_train_loss": 2.0,
            "best_valid_step": 8,
            "gpu_peak_mb": 8200,
            "tg_lora_summary": {
                "prefix_feature_cache_dir": ".cache/example",
                "prefix_feature_cache_total_build_seconds": build_seconds,
                "prefix_feature_cache_total_load_seconds": load_seconds,
                "prefix_feature_cache_valid_quick_source": quick_source,
                "prefix_feature_cache_valid_full_source": full_source,
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before": offload_before_mb,
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after": offload_after_mb,
                "prefix_feature_cache_runtime_offload_gpu_freed_mb": offload_freed_mb,
            },
        },
    ]


def _make_run_dir(
    root: Path,
    *,
    best_valid_baseline: float,
    best_valid_tg: float,
    baseline_wall_seconds: float,
    tg_wall_seconds: float,
    build_seconds: float,
    load_seconds: float,
    quick_source: str,
    full_source: str,
) -> Path:
    run_dir = root
    _write_jsonl(
        run_dir / "baseline" / "run_metrics.jsonl",
        _baseline_records(best_valid_baseline, baseline_wall_seconds),
    )
    _write_jsonl(
        run_dir / "tg_lora" / "run_metrics.jsonl",
        _tg_records(
            best_valid_loss=best_valid_tg,
            wall_seconds=tg_wall_seconds,
            build_seconds=build_seconds,
            load_seconds=load_seconds,
            quick_source=quick_source,
            full_source=full_source,
        ),
    )
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "comparison_20260523_000000.md").write_text("report\n")
    return run_dir


def test_summarize_comparison_run_exposes_prefix_cache_sources(tmp_path: Path):
    run_dir = _make_run_dir(
        tmp_path / "cold",
        best_valid_baseline=2.0,
        best_valid_tg=1.8,
        baseline_wall_seconds=36.0,
        tg_wall_seconds=220.0,
        build_seconds=190.0,
        load_seconds=0.0,
        quick_source="built",
        full_source="built",
    )

    summary = summarize_comparison_run(run_dir)

    assert summary["report_path"].endswith("comparison_20260523_000000.md")
    assert summary["tg_lora"]["prefix_feature_cache_valid_quick_source"] == "built"
    assert summary["tg_lora"]["prefix_feature_cache_valid_full_source"] == "built"
    assert summary["tg_lora"]["prefix_feature_cache_total_build_seconds"] == 190.0
    assert summary["tg_lora"]["prefix_feature_cache_total_load_seconds"] == 0.0
    assert summary["tg_lora"]["prefix_feature_cache_runtime_offload_gpu_allocated_mb_before"] == 12000.0
    assert summary["tg_lora"]["prefix_feature_cache_runtime_offload_gpu_allocated_mb_after"] == 8200.0
    assert summary["tg_lora"]["prefix_feature_cache_runtime_offload_gpu_freed_mb"] == 3800.0


def test_build_benchmark_summary_reports_cold_to_warm_speedup(tmp_path: Path):
    cold_dir = _make_run_dir(
        tmp_path / "cold",
        best_valid_baseline=2.06,
        best_valid_tg=1.86,
        baseline_wall_seconds=36.0,
        tg_wall_seconds=220.0,
        build_seconds=190.0,
        load_seconds=0.0,
        quick_source="built",
        full_source="built",
    )
    warm_dir = _make_run_dir(
        tmp_path / "warm",
        best_valid_baseline=2.06,
        best_valid_tg=1.88,
        baseline_wall_seconds=36.0,
        tg_wall_seconds=28.0,
        build_seconds=0.0,
        load_seconds=0.9,
        quick_source="disk",
        full_source="disk",
    )

    summary = build_benchmark_summary(cold_dir, warm_dir)

    assert summary["cold"]["tg_lora"]["prefix_feature_cache_valid_quick_source"] == "built"
    assert summary["warm"]["tg_lora"]["prefix_feature_cache_valid_quick_source"] == "disk"
    assert summary["warm"]["tg_lora"]["prefix_feature_cache_total_load_seconds"] == 0.9
    assert summary["warm"]["tg_lora"]["prefix_feature_cache_runtime_offload_gpu_freed_mb"] == 3800.0
    assert summary["delta"]["tg_wall_speedup_pct"] > 80.0
    assert summary["delta"]["tg_loss_red_per_wall_minute_delta_pct"] > 0.0