"""Tests for scripts/analyze_prefix_cache_break_even.py.

Covers argument parsing, single-run and aggregate extraction,
paper summary loading/validation, break-even calculation, and CLI behavior.
"""

import json
import subprocess
import sys
from pathlib import Path

import orjson
import pytest

from scripts.analyze_prefix_cache_break_even import (
    _extract_from_aggregate,
    _extract_from_single_run,
    _load_paper_summary,
    analyze_break_even,
)


def _write_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
    return path


# ---------------------------------------------------------------------------
# Fixtures: synthetic paper summary data
# ---------------------------------------------------------------------------


def _single_run_summary(
    *,
    warm_baseline_wall: float = 300.0,
    warm_tg_wall: float = 240.0,
    cold_build_seconds: float = 600.0,
    warm_baseline_gpu_mb: float = 8200.0,
    warm_tg_gpu_mb: float = 8300.0,
) -> dict:
    return {
        "cold": {"tg_lora": {"prefix_feature_cache_total_build_seconds": cold_build_seconds}},
        "warm": {
            "baseline": {"wall_seconds": warm_baseline_wall, "gpu_peak_mb": warm_baseline_gpu_mb},
            "tg_lora": {"wall_seconds": warm_tg_wall, "gpu_peak_mb": warm_tg_gpu_mb},
        },
    }


def _aggregate_summary(
    *,
    warm_baseline_wall: float = 300.0,
    warm_tg_wall: float = 240.0,
    tg_cache_build: float = 600.0,
    warm_baseline_gpu_mb: float = 8200.0,
    warm_tg_gpu_mb: float = 8300.0,
) -> dict:
    return {
        "aggregate": {
            "warm_baseline_wall_seconds": {"mean": warm_baseline_wall},
            "warm_tg_wall_seconds": {"mean": warm_tg_wall},
            "tg_cache_build_seconds": {"mean": tg_cache_build},
            "warm_baseline_gpu_peak_mb": {"mean": warm_baseline_gpu_mb},
            "warm_tg_gpu_peak_mb": {"mean": warm_tg_gpu_mb},
        },
        "per_seed": [],
    }


# ---------------------------------------------------------------------------
# _extract_from_single_run
# ---------------------------------------------------------------------------


class TestExtractFromSingleRun:
    def test_extracts_wall_times(self):
        s = _single_run_summary(warm_baseline_wall=300.0, warm_tg_wall=240.0)
        result = _extract_from_single_run(s)
        assert result["warm_baseline_wall_seconds"] == 300.0
        assert result["warm_tg_wall_seconds"] == 240.0

    def test_extracts_cold_build_seconds(self):
        s = _single_run_summary(cold_build_seconds=600.0)
        result = _extract_from_single_run(s)
        assert result["cold_build_seconds"] == 600.0

    def test_extracts_gpu_peak(self):
        s = _single_run_summary(warm_baseline_gpu_mb=8200.0, warm_tg_gpu_mb=8300.0)
        result = _extract_from_single_run(s)
        assert result["warm_baseline_gpu_peak_mb"] == 8200.0
        assert result["warm_tg_gpu_peak_mb"] == 8300.0

    def test_summary_type_is_single_run(self):
        result = _extract_from_single_run(_single_run_summary())
        assert result["summary_type"] == "single_run"

    def test_missing_cold_build_gives_none(self):
        s = _single_run_summary()
        del s["cold"]["tg_lora"]["prefix_feature_cache_total_build_seconds"]
        result = _extract_from_single_run(s)
        assert result["cold_build_seconds"] is None


# ---------------------------------------------------------------------------
# _extract_from_aggregate
# ---------------------------------------------------------------------------


class TestExtractFromAggregate:
    def test_extracts_means(self):
        s = _aggregate_summary(warm_baseline_wall=300.0, warm_tg_wall=240.0, tg_cache_build=600.0)
        result = _extract_from_aggregate(s)
        assert result["warm_baseline_wall_seconds"] == 300.0
        assert result["warm_tg_wall_seconds"] == 240.0
        assert result["cold_build_seconds"] == 600.0

    def test_extracts_gpu_means(self):
        s = _aggregate_summary(warm_baseline_gpu_mb=8200.0, warm_tg_gpu_mb=8300.0)
        result = _extract_from_aggregate(s)
        assert result["warm_baseline_gpu_peak_mb"] == 8200.0
        assert result["warm_tg_gpu_peak_mb"] == 8300.0

    def test_summary_type_is_aggregate(self):
        result = _extract_from_aggregate(_aggregate_summary())
        assert result["summary_type"] == "aggregate"


# ---------------------------------------------------------------------------
# _load_paper_summary
# ---------------------------------------------------------------------------


class TestLoadPaperSummary:
    def test_loads_single_run(self, tmp_path):
        path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = _load_paper_summary(path)
        assert result["summary_type"] == "single_run"

    def test_loads_aggregate(self, tmp_path):
        path = _write_json(tmp_path / "aggregate_summary.json", _aggregate_summary())
        result = _load_paper_summary(path)
        assert result["summary_type"] == "aggregate"

    def test_rejects_non_dict(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="must resolve to a JSON object"):
            _load_paper_summary(path)

    def test_rejects_unsupported_format(self, tmp_path):
        path = _write_json(tmp_path / "unknown.json", {"foo": "bar"})
        with pytest.raises(ValueError, match="Unsupported paper summary format"):
            _load_paper_summary(path)


# ---------------------------------------------------------------------------
# analyze_break_even
# ---------------------------------------------------------------------------


class TestAnalyzeBreakEven:
    def _paper(self, **kwargs) -> dict:
        return _extract_from_single_run(_single_run_summary(**kwargs))

    def test_warm_win_break_even(self):
        paper = self._paper(warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=600.0)
        result = analyze_break_even(paper, None)

        assert result["break_even_status"] == "warm_win"
        assert result["warm_wall_delta_seconds"] == pytest.approx(60.0)
        assert result["break_even_repeated_runs"] == pytest.approx(10.0)
        assert result["cold_build_seconds"] == pytest.approx(600.0)
        assert "Warm TG is already faster" in result["interpretation"]

    def test_no_warm_win(self):
        paper = self._paper(warm_baseline_wall=240.0, warm_tg_wall=300.0, cold_build_seconds=600.0)
        result = analyze_break_even(paper, None)

        assert result["break_even_status"] == "no_warm_win"
        assert result["break_even_repeated_runs"] is None
        assert result["warm_wall_delta_seconds"] == pytest.approx(-60.0)
        assert "does not yet beat baseline" in result["interpretation"]

    def test_equal_warm_times(self):
        paper = self._paper(warm_baseline_wall=300.0, warm_tg_wall=300.0, cold_build_seconds=600.0)
        result = analyze_break_even(paper, None)

        assert result["break_even_status"] == "no_warm_win"
        assert result["warm_wall_delta_seconds"] == pytest.approx(0.0)

    def test_one_run_total_includes_cold_build(self):
        paper = self._paper(warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=600.0)
        result = analyze_break_even(paper, None)

        assert result["one_run_total_tg_seconds_including_cold_build"] == pytest.approx(840.0)
        assert result["one_run_total_delta_seconds"] == pytest.approx(300.0 - 840.0)

    def test_cold_build_source_paper_summary(self):
        paper = self._paper()
        result = analyze_break_even(paper, None)
        assert result["cold_build_source"] == "paper_summary"

    def test_cold_build_source_overridden_by_precompute(self):
        paper = self._paper(cold_build_seconds=600.0)
        precompute = {"overall_wall_seconds": 400.0}
        result = analyze_break_even(paper, precompute)

        assert result["cold_build_source"] == "parallel_precompute_summary"
        assert result["cold_build_seconds"] == pytest.approx(400.0)

    def test_break_even_with_precompute_overrides_cold_build(self):
        paper = self._paper(warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=600.0)
        precompute = {"overall_wall_seconds": 120.0}
        result = analyze_break_even(paper, precompute)

        assert result["break_even_repeated_runs"] == pytest.approx(2.0)

    def test_raises_when_cold_build_missing_and_no_precompute(self):
        paper = self._paper()
        paper["cold_build_seconds"] = None
        with pytest.raises(ValueError, match="cold_build_seconds is required"):
            analyze_break_even(paper, None)

    def test_gpu_peak_mb_forwarded(self):
        paper = self._paper(warm_baseline_gpu_mb=8200.0, warm_tg_gpu_mb=8300.0)
        result = analyze_break_even(paper, None)
        assert result["warm_baseline_gpu_peak_mb"] == pytest.approx(8200.0)
        assert result["warm_tg_gpu_peak_mb"] == pytest.approx(8300.0)

    def test_aggregate_paper_break_even(self):
        paper = _extract_from_aggregate(_aggregate_summary(
            warm_baseline_wall=300.0, warm_tg_wall=240.0, tg_cache_build=600.0,
        ))
        result = analyze_break_even(paper, None)

        assert result["summary_type"] == "aggregate"
        assert result["break_even_status"] == "warm_win"
        assert result["break_even_repeated_runs"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_with_single_run_summary(self, tmp_path):
        summary_path = _write_json(
            tmp_path / "summary.json", _single_run_summary()
        )
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py",
             "--paper-summary", str(summary_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        output = json.loads(result.stdout.split("Break-even analysis")[0])
        assert output["break_even_status"] == "warm_win"

        out_file = tmp_path / "summary_break_even.json"
        assert out_file.exists()

    def test_cli_with_aggregate_summary(self, tmp_path):
        summary_path = _write_json(
            tmp_path / "aggregate_summary.json", _aggregate_summary()
        )
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py",
             "--paper-summary", str(summary_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_with_precompute_summary(self, tmp_path):
        summary_path = _write_json(
            tmp_path / "summary.json", _single_run_summary()
        )
        precompute_path = _write_json(
            tmp_path / "precompute.json", {"overall_wall_seconds": 400.0}
        )
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py",
             "--paper-summary", str(summary_path),
             "--precompute-summary", str(precompute_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        output = json.loads(result.stdout.split("Break-even analysis")[0])
        assert output["cold_build_source"] == "parallel_precompute_summary"
        assert output["cold_build_seconds"] == pytest.approx(400.0)

    def test_cli_custom_output_path(self, tmp_path):
        summary_path = _write_json(
            tmp_path / "summary.json", _single_run_summary()
        )
        output_path = tmp_path / "custom_output.json"
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py",
             "--paper-summary", str(summary_path),
             "--output", str(output_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_path.exists()

    def test_cli_missing_paper_summary_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py",
             "--paper-summary", "/nonexistent/path"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_cli_no_args_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_cli_rejects_non_dict_summary(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("[1, 2, 3]")
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py",
             "--paper-summary", str(bad_path)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "must resolve to a JSON object" in result.stderr


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_small_warm_delta_large_break_even(self):
        paper = _extract_from_single_run(_single_run_summary(
            warm_baseline_wall=300.0, warm_tg_wall=299.99, cold_build_seconds=600.0,
        ))
        result = analyze_break_even(paper, None)
        assert result["break_even_status"] == "warm_win"
        assert result["break_even_repeated_runs"] == pytest.approx(600.0 / 0.01)

    def test_large_cold_build_slow_amortization(self):
        paper = _extract_from_single_run(_single_run_summary(
            warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=60000.0,
        ))
        result = analyze_break_even(paper, None)
        assert result["break_even_repeated_runs"] == pytest.approx(1000.0)

    def test_precompute_with_missing_wall_seconds_raises(self):
        paper = _extract_from_single_run(_single_run_summary(cold_build_seconds=600.0))
        precompute = {}
        with pytest.raises(ValueError, match="cold_build_seconds is required"):
            analyze_break_even(paper, precompute)

    def test_one_run_total_delta_negative_when_cold_expensive(self):
        paper = _extract_from_single_run(_single_run_summary(
            warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=600.0,
        ))
        result = analyze_break_even(paper, None)
        # total_tg = 600 + 240 = 840 > baseline 300
        assert result["one_run_total_delta_seconds"] == pytest.approx(-540.0)
