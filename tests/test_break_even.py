"""Tests for scripts/analyze_prefix_cache_break_even.py — TC-226-01."""

import json

import pytest

from scripts.analyze_prefix_cache_break_even import (
    analyze_break_even,
    _extract_from_single_run,
    _extract_from_aggregate,
)


def _make_single_run_summary():
    return {
        "warm": {
            "baseline": {"wall_seconds": 600.0, "gpu_peak_mb": 8000},
            "tg_lora": {"wall_seconds": 500.0, "gpu_peak_mb": 8200},
        },
        "cold": {
            "tg_lora": {
                "prefix_feature_cache_total_build_seconds": 1200.0,
                "wall_seconds": 1800.0,
            },
        },
    }


def _make_aggregate_summary():
    return {
        "aggregate": {
            "warm_baseline_wall_seconds": {"mean": 600.0},
            "warm_tg_wall_seconds": {"mean": 500.0},
            "tg_cache_build_seconds": {"mean": 1200.0},
            "warm_baseline_gpu_peak_mb": {"mean": 8000},
            "warm_tg_gpu_peak_mb": {"mean": 8200},
        },
        "per_seed": {"seed_42": {}},
    }


class TestTC226:
    """REQ-226: Prefix cache break-even analysis."""

    def test_tc226_01_single_run_break_even(self):
        """TC-226-01: analyze_prefix_cache_break_even.py calculates break_even_cycles
        from a single-run benchmark summary."""
        summary = _make_single_run_summary()
        paper = _extract_from_single_run(summary)
        result = analyze_break_even(paper, precompute=None)

        assert result["break_even_status"] == "warm_win"
        assert result["break_even_repeated_runs"] is not None
        # warm_delta = 600 - 500 = 100, cold_build = 1200
        # break_even = 1200 / 100 = 12.0
        assert result["break_even_repeated_runs"] == pytest.approx(12.0)
        assert result["cold_build_seconds"] == pytest.approx(1200.0)
        assert result["warm_baseline_wall_seconds"] == pytest.approx(600.0)
        assert result["warm_tg_wall_seconds"] == pytest.approx(500.0)
        assert "break_even" in json.dumps(result).lower()

    def test_tc226_01_aggregate_break_even(self):
        """TC-226-01: break-even analysis works with aggregate_summary.json format."""
        summary = _make_aggregate_summary()
        paper = _extract_from_aggregate(summary)
        result = analyze_break_even(paper, precompute=None)

        assert result["summary_type"] == "aggregate"
        assert result["break_even_status"] == "warm_win"
        assert result["break_even_repeated_runs"] == pytest.approx(12.0)

    def test_tc226_01_with_precompute_override(self):
        """TC-226-01: precompute summary overrides cold_build_seconds."""
        summary = _make_single_run_summary()
        paper = _extract_from_single_run(summary)
        precompute = {"overall_wall_seconds": 800.0}
        result = analyze_break_even(paper, precompute=precompute)

        assert result["cold_build_source"] == "parallel_precompute_summary"
        assert result["cold_build_seconds"] == pytest.approx(800.0)
        assert result["break_even_repeated_runs"] == pytest.approx(8.0)

    def test_no_warm_win_status(self):
        """When warm TG is slower than baseline, break_even_status is no_warm_win."""
        paper = {
            "summary_type": "single_run",
            "warm_baseline_wall_seconds": 400.0,
            "warm_tg_wall_seconds": 500.0,
            "cold_build_seconds": 1200.0,
            "warm_baseline_gpu_peak_mb": 8000,
            "warm_tg_gpu_peak_mb": 8200,
        }
        result = analyze_break_even(paper, precompute=None)
        assert result["break_even_status"] == "no_warm_win"
        assert result["break_even_repeated_runs"] is None
