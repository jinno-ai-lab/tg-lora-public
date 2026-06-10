import json

import pytest

from scripts.compare_paper_memory_modes import (
    _load_summary,
    _relative_delta,
    _render_markdown,
    _series_mean,
    build_mode_comparison,
)


def _summary(
    *,
    seeds: list[int],
    warm_tg_wall_seconds: float,
    tg_cache_load_seconds: float,
    tg_cache_build_seconds: float,
    warm_tg_runtime_offload_gpu_allocated_mb_before: float,
    warm_tg_runtime_offload_gpu_allocated_mb_after: float,
    warm_tg_runtime_offload_gpu_freed_mb: float,
    warm_tg_best_valid_loss: float,
    warm_tg_gpu_peak_mb: float,
    warm_tg_loss_red_per_wall_minute: float,
    tg_cache_warm_speedup_pct: float,
) -> dict:
    per_seed = []
    for seed in seeds:
        per_seed.append(
            {
                "seed": seed,
                "warm_tg_wall_seconds": warm_tg_wall_seconds,
                "tg_cache_load_seconds": tg_cache_load_seconds,
                "warm_tg_runtime_offload_gpu_allocated_mb_before": warm_tg_runtime_offload_gpu_allocated_mb_before,
                "warm_tg_runtime_offload_gpu_allocated_mb_after": warm_tg_runtime_offload_gpu_allocated_mb_after,
                "warm_tg_runtime_offload_gpu_freed_mb": warm_tg_runtime_offload_gpu_freed_mb,
                "warm_tg_best_valid_loss": warm_tg_best_valid_loss,
            }
        )
    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "warm_tg_wall_seconds": {"mean": warm_tg_wall_seconds},
            "tg_cache_load_seconds": {"mean": tg_cache_load_seconds},
            "tg_cache_build_seconds": {"mean": tg_cache_build_seconds},
            "warm_tg_runtime_offload_gpu_allocated_mb_before": {
                "mean": warm_tg_runtime_offload_gpu_allocated_mb_before
            },
            "warm_tg_runtime_offload_gpu_allocated_mb_after": {
                "mean": warm_tg_runtime_offload_gpu_allocated_mb_after
            },
            "warm_tg_runtime_offload_gpu_freed_mb": {
                "mean": warm_tg_runtime_offload_gpu_freed_mb
            },
            "warm_tg_best_valid_loss": {"mean": warm_tg_best_valid_loss},
            "warm_tg_gpu_peak_mb": {"mean": warm_tg_gpu_peak_mb},
            "warm_tg_loss_red_per_wall_minute": {
                "mean": warm_tg_loss_red_per_wall_minute
            },
            "tg_cache_warm_speedup_pct": {"mean": tg_cache_warm_speedup_pct},
        },
    }


def _full_summaries() -> tuple[dict, dict]:
    return (
        _summary(
            seeds=[42],
            warm_tg_wall_seconds=9.3,
            tg_cache_load_seconds=2.73,
            tg_cache_build_seconds=741.81,
            warm_tg_runtime_offload_gpu_allocated_mb_before=12000.0,
            warm_tg_runtime_offload_gpu_allocated_mb_after=8200.0,
            warm_tg_runtime_offload_gpu_freed_mb=3800.0,
            warm_tg_best_valid_loss=1.8281,
            warm_tg_gpu_peak_mb=7788.9,
            warm_tg_loss_red_per_wall_minute=3.8653,
            tg_cache_warm_speedup_pct=98.76,
        ),
        _summary(
            seeds=[42],
            warm_tg_wall_seconds=5.0,
            tg_cache_load_seconds=0.001,
            tg_cache_build_seconds=739.847,
            warm_tg_runtime_offload_gpu_allocated_mb_before=11900.0,
            warm_tg_runtime_offload_gpu_allocated_mb_after=7700.0,
            warm_tg_runtime_offload_gpu_freed_mb=4200.0,
            warm_tg_best_valid_loss=1.8281,
            warm_tg_gpu_peak_mb=7788.9,
            warm_tg_loss_red_per_wall_minute=5.9238,
            tg_cache_warm_speedup_pct=99.33,
        ),
    )


@pytest.mark.parametrize(
    "label,expect",
    [
        (
            "aggregate_deltas",
            {
                "paired_seeds": [42],
                ("aggregate", "warm_tg_wall_seconds", "reuse_mean"): 9.3,
                ("aggregate", "warm_tg_wall_seconds", "one_shot_mean"): 5.0,
                ("aggregate", "warm_tg_wall_seconds", "absolute_delta"): pytest.approx(-4.3),
                ("aggregate", "tg_cache_load_seconds", "one_shot_mean"): 0.001,
                ("aggregate", "warm_tg_runtime_offload_gpu_freed_mb", "reuse_mean"): 3800.0,
                ("aggregate", "warm_tg_runtime_offload_gpu_freed_mb", "one_shot_mean"): 4200.0,
            },
        ),
        (
            "per_seed_deltas",
            {
                ("per_seed", 0, "tg_cache_load_seconds", "reuse"): 2.73,
                ("per_seed", 0, "tg_cache_load_seconds", "one_shot"): 0.001,
                ("per_seed", 0, "warm_tg_runtime_offload_gpu_allocated_mb_before", "reuse"): 12000.0,
                ("per_seed", 0, "warm_tg_runtime_offload_gpu_allocated_mb_after", "one_shot"): 7700.0,
                ("per_seed", 0, "warm_tg_runtime_offload_gpu_freed_mb", "one_shot"): 4200.0,
            },
        ),
    ],
)
def test_build_mode_comparison_full_data(label: str, expect: dict) -> None:
    reuse, one_shot = _full_summaries()
    comparison = build_mode_comparison(reuse, one_shot)

    for path, expected in expect.items():
        if isinstance(path, tuple):
            node = comparison
            for key in path:
                node = node[key]
            assert node == expected
        else:
            assert comparison[path] == expected


def test_build_mode_comparison_handles_missing_means() -> None:
    reuse = {"seeds": [42], "per_seed": [{"seed": 42}], "aggregate": {}}
    one_shot = {"seeds": [42], "per_seed": [{"seed": 42}], "aggregate": {}}

    comparison = build_mode_comparison(reuse, one_shot)

    assert comparison["aggregate"]["warm_tg_wall_seconds"]["reuse_mean"] is None
    assert comparison["aggregate"]["warm_tg_wall_seconds"]["relative_delta_pct"] is None


class TestRelativeDelta:
    def test_positive_delta(self):
        assert _relative_delta(100.0, 120.0) == pytest.approx(20.0)

    def test_negative_delta(self):
        assert _relative_delta(100.0, 80.0) == pytest.approx(-20.0)

    def test_zero_reference_returns_none(self):
        assert _relative_delta(0.0, 10.0) is None

    def test_none_inputs_return_none(self):
        assert _relative_delta(None, 10.0) is None
        assert _relative_delta(10.0, None) is None

    def test_both_none_returns_none(self):
        assert _relative_delta(None, None) is None


class TestSeriesMean:
    def test_extracts_float_mean(self):
        summary = {"aggregate": {"warm_tg_wall_seconds": {"mean": 9.3}}}
        assert _series_mean(summary, "warm_tg_wall_seconds") == 9.3

    def test_extracts_int_mean_as_float(self):
        summary = {"aggregate": {"count": {"mean": 42}}}
        assert _series_mean(summary, "count") == 42.0
        assert isinstance(_series_mean(summary, "count"), float)

    def test_missing_key_returns_none(self):
        assert _series_mean({}, "nonexistent") is None

    def test_none_mean_returns_none(self):
        summary = {"aggregate": {"metric": {"mean": None}}}
        assert _series_mean(summary, "metric") is None

    def test_string_mean_returns_none(self):
        summary = {"aggregate": {"metric": {"mean": "not_a_number"}}}
        assert _series_mean(summary, "metric") is None


class TestLoadSummary:
    def test_loads_aggregate_format(self, tmp_path):
        data = {
            "seeds": [1, 2],
            "per_seed": [{"seed": 1}, {"seed": 2}],
            "aggregate": {"warm_tg_wall_seconds": {"mean": 5.0}},
        }
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(data))
        result = _load_summary(path)
        assert result["seeds"] == [1, 2]
        assert result["aggregate"]["warm_tg_wall_seconds"]["mean"] == 5.0

    def test_loads_legacy_benchmark_format(self, tmp_path):
        data = {
            "cold": {"tg_lora": {"prefix_feature_cache_total_build_seconds": 100.0}},
            "warm": {
                "tg_lora": {
                    "wall_seconds": 9.3,
                    "prefix_feature_cache_total_load_seconds": 2.0,
                    "best_valid_loss": 1.5,
                    "gpu_peak_mb": 8000.0,
                    "loss_red_per_wall_minute": 3.0,
                    "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before": 12000.0,
                    "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after": 8200.0,
                    "prefix_feature_cache_runtime_offload_gpu_freed_mb": 3800.0,
                }
            },
            "delta": {"tg_wall_speedup_pct": 98.5},
        }
        path = tmp_path / "benchmark.json"
        path.write_text(json.dumps(data))
        result = _load_summary(path)
        assert result["seeds"] == [0]
        assert result["aggregate"]["warm_tg_wall_seconds"]["mean"] == 9.3
        assert result["aggregate"]["tg_cache_build_seconds"]["mean"] == 100.0
        assert result["aggregate"]["tg_cache_warm_speedup_pct"]["mean"] == 98.5
        assert len(result["per_seed"]) == 1
        assert result["per_seed"][0]["warm_tg_wall_seconds"] == 9.3

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"unknown": "format"}))
        with pytest.raises(ValueError, match="Unsupported paper-memory summary"):
            _load_summary(path)


class TestRenderMarkdown:
    def test_renders_aggregate_table(self):
        reuse, one_shot = _full_summaries()
        comparison = build_mode_comparison(reuse, one_shot)
        md = _render_markdown(comparison)
        assert "# Paper Memory Mode Comparison" in md
        assert "## Aggregate Means" in md
        assert "warm_tg_wall_seconds" in md
        assert "relative_delta_pct" not in md

    def test_renders_per_seed_section(self):
        reuse, one_shot = _full_summaries()
        comparison = build_mode_comparison(reuse, one_shot)
        md = _render_markdown(comparison)
        assert "## Per Seed" in md
        assert "Per Seed Offload Memory" in md

    def test_no_per_seed_when_empty(self):
        reuse = {"seeds": [1], "per_seed": [], "aggregate": {}}
        one_shot = {"seeds": [2], "per_seed": [], "aggregate": {}}
        comparison = build_mode_comparison(reuse, one_shot)
        md = _render_markdown(comparison)
        assert "## Per Seed" not in md

    def test_dashes_for_none_values(self):
        reuse = {"seeds": [], "per_seed": [], "aggregate": {}}
        one_shot = {"seeds": [], "per_seed": [], "aggregate": {}}
        comparison = build_mode_comparison(reuse, one_shot)
        md = _render_markdown(comparison)
        assert " - |" in md


class TestMainIntegration:
    def test_main_writes_json_and_md(self, tmp_path, monkeypatch):
        reuse_data = {
            "seeds": [42],
            "per_seed": [
                {
                    "seed": 42,
                    "warm_tg_wall_seconds": 9.3,
                    "tg_cache_load_seconds": 2.73,
                    "warm_tg_runtime_offload_gpu_allocated_mb_before": 12000.0,
                    "warm_tg_runtime_offload_gpu_allocated_mb_after": 8200.0,
                    "warm_tg_runtime_offload_gpu_freed_mb": 3800.0,
                    "warm_tg_best_valid_loss": 1.8281,
                }
            ],
            "aggregate": {
                "warm_tg_wall_seconds": {"mean": 9.3},
                "tg_cache_load_seconds": {"mean": 2.73},
                "tg_cache_build_seconds": {"mean": 741.81},
                "warm_tg_runtime_offload_gpu_allocated_mb_before": {"mean": 12000.0},
                "warm_tg_runtime_offload_gpu_allocated_mb_after": {"mean": 8200.0},
                "warm_tg_runtime_offload_gpu_freed_mb": {"mean": 3800.0},
                "warm_tg_best_valid_loss": {"mean": 1.8281},
                "warm_tg_gpu_peak_mb": {"mean": 7788.9},
                "warm_tg_loss_red_per_wall_minute": {"mean": 3.8653},
                "tg_cache_warm_speedup_pct": {"mean": 98.76},
            },
        }
        one_shot_data = {
            "seeds": [42],
            "per_seed": [
                {
                    "seed": 42,
                    "warm_tg_wall_seconds": 5.0,
                    "tg_cache_load_seconds": 0.001,
                    "warm_tg_runtime_offload_gpu_allocated_mb_before": 11900.0,
                    "warm_tg_runtime_offload_gpu_allocated_mb_after": 7700.0,
                    "warm_tg_runtime_offload_gpu_freed_mb": 4200.0,
                    "warm_tg_best_valid_loss": 1.8281,
                }
            ],
            "aggregate": {
                "warm_tg_wall_seconds": {"mean": 5.0},
                "tg_cache_load_seconds": {"mean": 0.001},
                "tg_cache_build_seconds": {"mean": 739.847},
                "warm_tg_runtime_offload_gpu_allocated_mb_before": {"mean": 11900.0},
                "warm_tg_runtime_offload_gpu_allocated_mb_after": {"mean": 7700.0},
                "warm_tg_runtime_offload_gpu_freed_mb": {"mean": 4200.0},
                "warm_tg_best_valid_loss": {"mean": 1.8281},
                "warm_tg_gpu_peak_mb": {"mean": 7788.9},
                "warm_tg_loss_red_per_wall_minute": {"mean": 5.9238},
                "tg_cache_warm_speedup_pct": {"mean": 99.33},
            },
        }
        reuse_path = tmp_path / "reuse.json"
        one_shot_path = tmp_path / "one_shot.json"
        reuse_path.write_text(json.dumps(reuse_data))
        one_shot_path.write_text(json.dumps(one_shot_data))
        output_base = tmp_path / "output"
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare_paper_memory_modes",
                f"--reuse-summary={reuse_path}",
                f"--one-shot-summary={one_shot_path}",
                f"--output-base={output_base}",
            ],
        )
        from scripts.compare_paper_memory_modes import main

        main()
        assert (tmp_path / "output.json").exists()
        assert (tmp_path / "output.md").exists()
        result = json.loads((tmp_path / "output.json").read_text())
        assert result["paired_seeds"] == [42]
        md = (tmp_path / "output.md").read_text()
        assert "# Paper Memory Mode Comparison" in md