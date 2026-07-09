"""Tests for scripts/analyze_prefix_cache_break_even.py.

Covers argument parsing, single-run and aggregate extraction,
paper summary loading/validation, break-even calculation, and CLI behavior.
"""

import json
import os
import shutil
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
    evaluate_gates,
)

# The end-to-end tests (TestEndToEndPipelineGate) drive the REAL benchmark
# producer over run_metrics.jsonl. That producer pulls a heavier import chain
# (scripts.benchmark_prefix_cache -> scripts.compare_runs -> src.utils.* ) than
# the analyze script itself needs, so import it lazily and skip those tests when
# the chain is unavailable rather than failing collection.
try:  # pragma: no cover - environment-dependent import chain
    from scripts.benchmark_prefix_cache import build_benchmark_summary

    _PRODUCER_AVAILABLE = True
except Exception:  # pragma: no cover
    build_benchmark_summary = None  # type: ignore[assignment]
    _PRODUCER_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "cold": {
            "tg_lora": {"prefix_feature_cache_total_build_seconds": cold_build_seconds}
        },
        "warm": {
            "baseline": {
                "wall_seconds": warm_baseline_wall,
                "gpu_peak_mb": warm_baseline_gpu_mb,
            },
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
        s = _aggregate_summary(
            warm_baseline_wall=300.0, warm_tg_wall=240.0, tg_cache_build=600.0
        )
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
        paper = self._paper(
            warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=600.0
        )
        result = analyze_break_even(paper, None)

        assert result["break_even_status"] == "warm_win"
        assert result["warm_wall_delta_seconds"] == pytest.approx(60.0)
        assert result["break_even_repeated_runs"] == pytest.approx(10.0)
        assert result["cold_build_seconds"] == pytest.approx(600.0)
        assert "Warm TG is already faster" in result["interpretation"]

    def test_no_warm_win(self):
        paper = self._paper(
            warm_baseline_wall=240.0, warm_tg_wall=300.0, cold_build_seconds=600.0
        )
        result = analyze_break_even(paper, None)

        assert result["break_even_status"] == "no_warm_win"
        assert result["break_even_repeated_runs"] is None
        assert result["warm_wall_delta_seconds"] == pytest.approx(-60.0)
        assert "does not yet beat baseline" in result["interpretation"]

    def test_equal_warm_times(self):
        paper = self._paper(
            warm_baseline_wall=300.0, warm_tg_wall=300.0, cold_build_seconds=600.0
        )
        result = analyze_break_even(paper, None)

        assert result["break_even_status"] == "no_warm_win"
        assert result["warm_wall_delta_seconds"] == pytest.approx(0.0)

    def test_one_run_total_includes_cold_build(self):
        paper = self._paper(
            warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=600.0
        )
        result = analyze_break_even(paper, None)

        assert result["one_run_total_tg_seconds_including_cold_build"] == pytest.approx(
            840.0
        )
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
        paper = self._paper(
            warm_baseline_wall=300.0, warm_tg_wall=240.0, cold_build_seconds=600.0
        )
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
        paper = _extract_from_aggregate(
            _aggregate_summary(
                warm_baseline_wall=300.0,
                warm_tg_wall=240.0,
                tg_cache_build=600.0,
            )
        )
        result = analyze_break_even(paper, None)

        assert result["summary_type"] == "aggregate"
        assert result["break_even_status"] == "warm_win"
        assert result["break_even_repeated_runs"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# evaluate_gates — the consumer that turns display-only metrics into a verdict
# ---------------------------------------------------------------------------


class TestEvaluateGates:
    """The gate is the consumer the steering input asked for: it makes the
    previously display-only break_even_status / break_even_repeated_runs /
    one_run_total_delta_seconds metrics drive a pass/fail decision instead of
    being a 4th symmetric aggregation axis with no consumer."""

    def _result(self, **kwargs) -> dict:
        # default fixture: warm_baseline=300, warm_tg=240, cold_build=600 ->
        #   break_even_status='warm_win', break_even_repeated_runs=10.0,
        #   one_run_total_delta_seconds=-540.0 (cold build dominates one run)
        paper = _extract_from_single_run(_single_run_summary(**kwargs))
        return analyze_break_even(paper, None)

    def test_no_gates_enabled_yields_no_failures(self):
        result = self._result()
        assert (
            evaluate_gates(
                result,
                require_warm_win=False,
                max_break_even_runs=None,
                require_one_run_win=False,
            )
            == []
        )

    def test_require_warm_win_passes_when_warm_tg_beats_baseline(self):
        result = self._result()
        assert (
            evaluate_gates(
                result,
                require_warm_win=True,
                max_break_even_runs=None,
                require_one_run_win=False,
            )
            == []
        )

    def test_require_warm_win_fails_when_no_warm_win(self):
        result = self._result(warm_baseline_wall=240.0, warm_tg_wall=300.0)
        failures = evaluate_gates(
            result,
            require_warm_win=True,
            max_break_even_runs=None,
            require_one_run_win=False,
        )
        assert len(failures) == 1
        assert failures[0]["gate"] == "--require-warm-win"
        assert "no_warm_win" in failures[0]["message"]

    def test_max_break_even_runs_passes_within_budget(self):
        result = self._result()  # ber=10.0
        assert (
            evaluate_gates(
                result,
                require_warm_win=False,
                max_break_even_runs=20.0,
                require_one_run_win=False,
            )
            == []
        )

    def test_max_break_even_runs_boundary_is_inclusive(self):
        result = self._result()  # ber=10.0 exactly
        assert (
            evaluate_gates(
                result,
                require_warm_win=False,
                max_break_even_runs=10.0,
                require_one_run_win=False,
            )
            == []
        )

    def test_max_break_even_runs_fails_over_budget(self):
        result = self._result()  # ber=10.0
        failures = evaluate_gates(
            result,
            require_warm_win=False,
            max_break_even_runs=5.0,
            require_one_run_win=False,
        )
        assert len(failures) == 1
        assert failures[0]["gate"] == "--max-break-even-runs"
        assert "exceeds budget 5.0" in failures[0]["message"]

    def test_max_break_even_runs_fails_when_no_warm_win(self):
        result = self._result(warm_baseline_wall=240.0, warm_tg_wall=300.0)
        failures = evaluate_gates(
            result,
            require_warm_win=False,
            max_break_even_runs=20.0,
            require_one_run_win=False,
        )
        assert len(failures) == 1
        assert failures[0]["gate"] == "--max-break-even-runs"
        assert "None" in failures[0]["message"]

    def test_require_one_run_win_fails_when_cold_build_dominates(self):
        result = self._result()  # one_run_total_delta=-540.0
        failures = evaluate_gates(
            result,
            require_warm_win=False,
            max_break_even_runs=None,
            require_one_run_win=True,
        )
        assert len(failures) == 1
        assert failures[0]["gate"] == "--require-one-run-win"

    def test_require_one_run_win_passes_when_cold_build_cheap(self):
        # cold_build=50 -> one run incl cold = 290 < baseline 300 -> delta +10
        result = self._result(cold_build_seconds=50.0)
        assert result["one_run_total_delta_seconds"] > 0
        assert (
            evaluate_gates(
                result,
                require_warm_win=False,
                max_break_even_runs=None,
                require_one_run_win=True,
            )
            == []
        )

    # -- VRAM-budget gate: the consumer for the otherwise display-only ----------
    # warm_*_gpu_peak_mb metrics. Constitution P3 (VRAM cost accounting) + the
    # RTX 3060 12GB target: the wall-clock gates above are moot if the cache-on
    # arm OOMs the budget. Default fixture has baseline=8200 MB, tg=8300 MB.

    def test_max_warm_gpu_peak_mb_passes_within_budget(self):
        result = self._result()  # baseline=8200, tg=8300
        assert evaluate_gates(result, max_warm_gpu_peak_mb=9000.0) == []

    def test_max_warm_gpu_peak_mb_boundary_is_inclusive(self):
        result = self._result()  # tg=8300.0 exactly
        assert evaluate_gates(result, max_warm_gpu_peak_mb=8300.0) == []

    def test_max_warm_gpu_peak_mb_fails_when_tg_exceeds(self):
        result = self._result()  # tg=8300 > 8250; baseline=8200 ok
        failures = evaluate_gates(result, max_warm_gpu_peak_mb=8250.0)
        assert len(failures) == 1
        assert failures[0]["gate"] == "--max-warm-gpu-peak-mb"
        assert failures[0]["arm"] == "warm_tg_gpu_peak_mb"
        assert "warm_tg_gpu_peak_mb" in failures[0]["message"]
        assert "exceeds budget 8250.0" in failures[0]["message"]

    def test_max_warm_gpu_peak_mb_fails_when_baseline_exceeds(self):
        result = self._result(
            warm_baseline_gpu_mb=9000.0
        )  # baseline=9000 > 8500; tg=8300 ok
        failures = evaluate_gates(result, max_warm_gpu_peak_mb=8500.0)
        assert len(failures) == 1
        assert failures[0]["gate"] == "--max-warm-gpu-peak-mb"
        assert failures[0]["arm"] == "warm_baseline_gpu_peak_mb"
        assert "warm_baseline_gpu_peak_mb" in failures[0]["message"]

    def test_max_warm_gpu_peak_mb_reports_each_offending_arm(self):
        # both arms over an 8000 MB budget (baseline set to 9000, tg default 8300)
        result = self._result(warm_baseline_gpu_mb=9000.0)
        failures = evaluate_gates(result, max_warm_gpu_peak_mb=8000.0)
        assert len(failures) == 2
        # Structured per-arm attribution: each offending arm names itself in the
        # `arm` field (the metric key), so a consumer enumerates failing arms
        # without substring-matching the human-readable message.
        assert {f["arm"] for f in failures} == {
            "warm_baseline_gpu_peak_mb",
            "warm_tg_gpu_peak_mb",
        }
        messages = "\n".join(f["message"] for f in failures)
        assert "warm_baseline_gpu_peak_mb" in messages
        assert "warm_tg_gpu_peak_mb" in messages

    def test_max_warm_gpu_peak_mb_fails_loud_when_unmeasured(self):
        # gpu_peak_mb=None => the peak was never recorded (CPU run / legacy run);
        # a budget cannot be certified against an unmeasured peak, so fail loud
        # (constitution §7: don't conclude without measuring) rather than pass.
        result = self._result(warm_tg_gpu_mb=None)
        failures = evaluate_gates(result, max_warm_gpu_peak_mb=12288.0)
        assert len(failures) == 1
        assert failures[0]["gate"] == "--max-warm-gpu-peak-mb"
        assert failures[0]["arm"] == "warm_tg_gpu_peak_mb"
        assert "not recorded" in failures[0]["message"]

    def test_vram_arm_field_is_structured_not_message_embedded(self):
        # Load-bearing contract test: the per-arm attribution is a STRUCTURED
        # field (the metric key), not text buried in `message`. A consumer must
        # be able to read which arm broke budget via failures[i]["arm"] alone —
        # independent of how the message is worded. If a future edit rephrases
        # the message and drops the bare metric key, the `arm` field still names
        # the offending arm, so the enforced CI gate's per-arm verdict cannot
        # silently degrade to "some gate failed, no idea which arm".
        result = self._result(warm_tg_gpu_mb=14000.0)  # only TG over a 12288 budget
        failures = evaluate_gates(result, max_warm_gpu_peak_mb=12288.0)
        assert len(failures) == 1
        assert failures[0]["arm"] == "warm_tg_gpu_peak_mb"
        assert failures[0]["arm"] != "warm_baseline_gpu_peak_mb"
        # The arm key matches the result-dict key the verdict reports the value under:
        assert failures[0]["arm"] in result

    def test_non_arm_gate_failures_carry_none_arm(self):
        # Cross-arm gates (warm-win / break-even-runs / one-run-win) compare the
        # two arms holistically, so no single arm is at fault. Their failure
        # records carry arm=None so a consumer uniformly reads f["arm"] and gets
        # either the metric key or a clear "not arm-specific" sentinel — never a
        # KeyError, never a misleading arm.
        result = self._result(warm_baseline_wall=240.0, warm_tg_wall=300.0)
        failures = evaluate_gates(
            result,
            require_warm_win=True,
            max_break_even_runs=5.0,
            require_one_run_win=True,
        )
        assert {f["gate"] for f in failures} == {
            "--require-warm-win",
            "--max-break-even-runs",
            "--require-one-run-win",
        }
        assert all(f["arm"] is None for f in failures)

    def test_multiple_enabled_gates_collect_every_failure(self):
        result = self._result(warm_baseline_wall=240.0, warm_tg_wall=300.0)
        failures = evaluate_gates(
            result,
            require_warm_win=True,
            max_break_even_runs=5.0,
            require_one_run_win=True,
        )
        gates = {f["gate"] for f in failures}
        assert gates == {
            "--require-warm-win",
            "--max-break-even-runs",
            "--require-one-run-win",
        }


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_with_single_run_summary(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Stdout is a clean JSON document (the "written to ..." status line is on
        # stderr), so json.loads(stdout) parses directly — no fragile splitting.
        output = json.loads(result.stdout)
        assert output["break_even_status"] == "warm_win"

        out_file = tmp_path / "summary_break_even.json"
        assert out_file.exists()

    def test_cli_with_aggregate_summary(self, tmp_path):
        summary_path = _write_json(
            tmp_path / "aggregate_summary.json", _aggregate_summary()
        )
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_with_precompute_summary(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        precompute_path = _write_json(
            tmp_path / "precompute.json", {"overall_wall_seconds": 400.0}
        )
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--precompute-summary",
                str(precompute_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["cold_build_source"] == "parallel_precompute_summary"
        assert output["cold_build_seconds"] == pytest.approx(400.0)

    def test_cli_custom_output_path(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        output_path = tmp_path / "custom_output.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_path.exists()

    def test_cli_missing_paper_summary_exits_nonzero(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                "/nonexistent/path",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_no_args_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "scripts/analyze_prefix_cache_break_even.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_rejects_non_dict_summary(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("[1, 2, 3]")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(bad_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "must resolve to a JSON object" in result.stderr

    # -- decision gates: the metrics now drive a CI exit code -----------------

    def test_cli_no_gate_flag_keeps_exit_zero_even_with_no_warm_win(self, tmp_path):
        """Backward-compat: gates are strictly opt-in. Without a gate flag the
        script stays display-only (exit 0) even when the verdict is bad."""
        summary_path = _write_json(
            tmp_path / "summary.json",
            _single_run_summary(warm_baseline_wall=240.0, warm_tg_wall=300.0),
        )
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"display-only path must stay exit 0: {result.stderr}"
        )

    def test_cli_require_warm_win_passes(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--require-warm-win",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"gate should pass: {result.stderr}"

    def test_cli_require_warm_win_fails(self, tmp_path):
        summary_path = _write_json(
            tmp_path / "summary.json",
            _single_run_summary(warm_baseline_wall=240.0, warm_tg_wall=300.0),
        )
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--require-warm-win",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "--require-warm-win" in result.stderr

    def test_cli_max_break_even_runs_passes_within_budget(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--max-break-even-runs",
                "20",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"gate should pass: {result.stderr}"

    def test_cli_max_break_even_runs_fails_over_budget(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--max-break-even-runs",
                "5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "--max-break-even-runs" in result.stderr
        assert "exceeds budget" in result.stderr

    def test_cli_require_one_run_win_fails_when_cold_dominates(self, tmp_path):
        # default fixture: cold build 600 -> one run incl cold = 840 > 300
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--require-one-run-win",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "--require-one-run-win" in result.stderr

    def test_cli_max_warm_gpu_peak_mb_passes_within_budget(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--max-warm-gpu-peak-mb",
                "9000",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"VRAM gate should pass: {result.stderr}"

    def test_cli_max_warm_gpu_peak_mb_fails_over_budget(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--max-warm-gpu-peak-mb",
                "8000",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "--max-warm-gpu-peak-mb" in result.stderr
        assert "exceeds budget" in result.stderr

    def test_cli_max_warm_gpu_peak_mb_unmeasured_fails_loud(self, tmp_path):
        # tg gpu_peak_mb stripped -> the VRAM budget cannot be certified
        summary = _single_run_summary()
        summary["warm"]["tg_lora"]["gpu_peak_mb"] = None
        summary_path = _write_json(tmp_path / "summary.json", summary)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--max-warm-gpu-peak-mb",
                "12288",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "not recorded" in result.stderr

    def test_cli_max_warm_gpu_peak_mb_verdict_recorded_in_output_json(self, tmp_path):
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        out_path = tmp_path / "out.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--max-warm-gpu-peak-mb",
                "8000",
                "--output",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        record = json.loads(out_path.read_bytes())
        assert record["gates"]["passed"] is False
        assert record["gates"]["requested"] == ["--max-warm-gpu-peak-mb=8000.0"]
        assert record["gates"]["failures"][0]["gate"] == "--max-warm-gpu-peak-mb"

    def test_cli_gate_verdict_recorded_in_output_json(self, tmp_path):
        """The consumer leaves a paper trail: when a gate is requested the output
        JSON carries the verdict, so a deposit artifact records the decision."""
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        out_path = tmp_path / "out.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--max-break-even-runs",
                "5",
                "--output",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        record = json.loads(out_path.read_bytes())
        assert "gates" in record
        assert record["gates"]["passed"] is False
        assert record["gates"]["requested"] == ["--max-break-even-runs=5.0"]
        assert record["gates"]["failures"][0]["gate"] == "--max-break-even-runs"

    def test_cli_no_gate_flag_leaves_output_byte_identical(self, tmp_path):
        """No gate flag -> no `gates` key in the output JSON (existing consumers
        are unaffected; the gate is purely additive)."""
        summary_path = _write_json(tmp_path / "summary.json", _single_run_summary())
        out_path = tmp_path / "out.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_prefix_cache_break_even.py",
                "--paper-summary",
                str(summary_path),
                "--output",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        record = json.loads(out_path.read_bytes())
        assert "gates" not in record


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_small_warm_delta_large_break_even(self):
        paper = _extract_from_single_run(
            _single_run_summary(
                warm_baseline_wall=300.0,
                warm_tg_wall=299.99,
                cold_build_seconds=600.0,
            )
        )
        result = analyze_break_even(paper, None)
        assert result["break_even_status"] == "warm_win"
        assert result["break_even_repeated_runs"] == pytest.approx(600.0 / 0.01)

    def test_large_cold_build_slow_amortization(self):
        paper = _extract_from_single_run(
            _single_run_summary(
                warm_baseline_wall=300.0,
                warm_tg_wall=240.0,
                cold_build_seconds=60000.0,
            )
        )
        result = analyze_break_even(paper, None)
        assert result["break_even_repeated_runs"] == pytest.approx(1000.0)

    def test_precompute_with_missing_wall_seconds_raises(self):
        paper = _extract_from_single_run(_single_run_summary(cold_build_seconds=600.0))
        precompute = {}
        with pytest.raises(ValueError, match="cold_build_seconds is required"):
            analyze_break_even(paper, precompute)

    def test_one_run_total_delta_negative_when_cold_expensive(self):
        paper = _extract_from_single_run(
            _single_run_summary(
                warm_baseline_wall=300.0,
                warm_tg_wall=240.0,
                cold_build_seconds=600.0,
            )
        )
        result = analyze_break_even(paper, None)
        # total_tg = 600 + 240 = 840 > baseline 300
        assert result["one_run_total_delta_seconds"] == pytest.approx(-540.0)


# ---------------------------------------------------------------------------
# End-to-end pipeline gate — closes the fixture-vs-pipeline gap
# ---------------------------------------------------------------------------
#
# Steering input (AI_HUB_MAKE_RUN_FEEDBACK): every other gate test feeds a
# hand-pruned 4-key fixture. These tests instead drive the REAL
# `build_benchmark_summary` producer over run_metrics.jsonl inputs (the format
# train_* emits) and then exercise the REAL `make analyze-prefix-break-even-ci`
# target — so the gate's accept/reject boundary is validated against genuine
# pipeline output, not only unit fixtures. The producer output is far denser
# (18 tg_lora keys + delta) than the _single_run_summary unit fixture, which is
# why this is real coverage rather than a duplicate of TestCLI.


def _write_run_metrics_jsonl(
    path: Path,
    *,
    wall_seconds: float,
    gpu_peak_mb: float,
    build_seconds: float | None = None,
    tg: bool = False,
) -> None:
    """Write a run_metrics.jsonl the real `load_run`/`summarize_comparison_run`
    producer can consume: a run_header line, step records, and a run_footer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"type": "step", "step": 0, "loss_train": 2.5, "total_backward_passes": 100},
        {"type": "step", "step": 1, "loss_train": 2.4, "total_backward_passes": 200},
    ]
    footer: dict = {
        "type": "run_footer",
        "total_wall_seconds": wall_seconds,
        "gpu_peak_mb": gpu_peak_mb,
        "best_valid_loss": 2.3,
    }
    if tg:
        # Mirror the full tg_lora_summary surface the real producer forwards.
        footer["tg_lora_summary"] = {
            "extrapolation_steps": 50,
            "accepted": 40,
            "acceptance_rate": 0.8,
            "prefix_feature_cache_dir": str(path.parent / "cache"),
            "prefix_feature_cache_total_build_seconds": build_seconds,
            "prefix_feature_cache_total_load_seconds": 5.0,
            "prefix_feature_cache_valid_quick_source": "quick",
            "prefix_feature_cache_valid_full_source": "full",
            "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before": 8000,
            "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after": 4000,
            "prefix_feature_cache_runtime_offload_gpu_freed_mb": 4000,
            "prefix_feature_cache_offloaded_prefix_modules": 12,
            "prefix_feature_cache_offloaded_prefix_parameters": 1024,
        }
    lines = [orjson.dumps({"type": "run_header", "config": "x"})]
    lines += [orjson.dumps(r) for r in records]
    lines.append(orjson.dumps(footer))
    path.write_bytes(b"\n".join(lines) + b"\n")


@pytest.mark.skipif(
    not _PRODUCER_AVAILABLE, reason="benchmark_prefix_cache import chain unavailable"
)
@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
class TestEndToEndPipelineGate:
    """Validate the gate target against REAL producer output, end to end."""

    @staticmethod
    def _venv_root() -> str:
        # The Makefile recipe is `$(VENV)/bin/python`. Point VENV at the python
        # running pytest so the gate target resolves without a repo .venv.
        candidate = Path(sys.executable).resolve().parent.parent
        if (candidate / "bin" / "python").exists():
            return str(candidate)
        pytest.skip(
            "sys.executable is not <venv>/bin/python; cannot derive VENV for make target"
        )

    @staticmethod
    def _build_real_summary(
        tmp_path: Path,
        *,
        warm_tg_wall: float,
        warm_tg_gpu: float,
        warm_baseline_wall: float = 300.0,
        warm_baseline_gpu: float = 8200.0,
        cold_tg_build: float = 600.0,
    ) -> Path:
        """Drive the REAL `build_benchmark_summary` producer over jsonl inputs
        to emit a pipeline-faithful summary.json. Returns its path."""
        run_root = tmp_path / "run"
        _write_run_metrics_jsonl(
            run_root / "cold" / "baseline" / "run_metrics.jsonl",
            wall_seconds=300.0,
            gpu_peak_mb=8200.0,
        )
        _write_run_metrics_jsonl(
            run_root / "cold" / "tg_lora" / "run_metrics.jsonl",
            wall_seconds=360.0,
            gpu_peak_mb=8300.0,
            build_seconds=cold_tg_build,
            tg=True,
        )
        _write_run_metrics_jsonl(
            run_root / "warm" / "baseline" / "run_metrics.jsonl",
            wall_seconds=warm_baseline_wall,
            gpu_peak_mb=warm_baseline_gpu,
        )
        _write_run_metrics_jsonl(
            run_root / "warm" / "tg_lora" / "run_metrics.jsonl",
            wall_seconds=warm_tg_wall,
            gpu_peak_mb=warm_tg_gpu,
            build_seconds=cold_tg_build,
            tg=True,
        )
        summary = build_benchmark_summary(run_root / "cold", run_root / "warm")
        out = tmp_path / "summary.json"
        out.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
        return out

    @staticmethod
    def _run_make_gate(env_overrides: dict) -> subprocess.CompletedProcess:
        env = {**os.environ, **env_overrides}
        return subprocess.run(
            ["make", "analyze-prefix-break-even-ci"],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_producer_output_is_dense_pipeline_shape_not_minimal_fixture(
        self, tmp_path
    ):
        """Guard: the producer-emitted summary carries the full real key surface
        (>=18 tg_lora keys + delta), far denser than the 4-key _single_run_summary
        unit fixture — which is WHY this end-to-end exercise is a real coverage
        gain and not a duplicate of TestCLI. If the producer drops a field this
        pins, the gate's extractor may be reading a stale shape."""
        summary_path = self._build_real_summary(
            tmp_path, warm_tg_wall=240.0, warm_tg_gpu=8300.0
        )
        produced = orjson.loads(summary_path.read_bytes())
        tg = produced["warm"]["tg_lora"]
        # Real keys the minimal unit fixture omits but the pipeline always emits:
        for dense_key in (
            "prefix_feature_cache_total_load_seconds",
            "prefix_feature_cache_offloaded_prefix_modules",
            "prefix_feature_cache_runtime_offload_gpu_freed_mb",
            "extrapolation_steps",
            "acceptance_rate",
            "loss_red_per_wall_minute",
            "total_backward_passes",
        ):
            assert dense_key in tg, (
                f"producer output missing real pipeline key: {dense_key}"
            )
        assert "delta" in produced
        assert "tg_wall_speedup_pct" in produced["delta"]
        assert len(tg) >= 18
        # And the extractor still pulls the right values out of the dense shape:
        extracted = _extract_from_single_run(produced)
        assert extracted["warm_baseline_gpu_peak_mb"] == 8200.0
        assert extracted["warm_tg_gpu_peak_mb"] == 8300.0
        assert extracted["cold_build_seconds"] == 600.0

    def test_vram_violation_fires_nonzero_with_per_arm_record(self, tmp_path):
        """Real pipeline output: warm TG WINS on wall-clock (240<300) but BLOWS
        the VRAM budget (14000>12288). The VRAM gate must fire naming the TG arm
        while --require-warm-win passes — per-arm / per-gate independence on
        genuine producer output. Non-zero exit + verdict JSON recorded."""
        summary_path = self._build_real_summary(
            tmp_path, warm_tg_wall=240.0, warm_tg_gpu=14000.0
        )
        verdict = tmp_path / "verdict.json"
        result = self._run_make_gate(
            {
                "VENV": self._venv_root(),
                "PAPER_SUMMARY": str(summary_path),
                "OUTPUT_PATH": str(verdict),
                "REQUIRE_WARM_WIN": "1",
                "MAX_WARM_GPU_PEAK_MB": "12288",
            }
        )
        assert result.returncode != 0, "VRAM-violating warm arm must exit non-zero"
        assert "--max-warm-gpu-peak-mb" in result.stderr
        assert "warm_tg_gpu_peak_mb" in result.stderr
        assert "exceeds budget" in result.stderr
        # The baseline arm (8200 MB) is under budget and must NOT be named:
        assert "warm_baseline_gpu_peak_mb" not in result.stderr
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is False
        assert {f["gate"] for f in record["gates"]["failures"]} == {
            "--max-warm-gpu-peak-mb"
        }
        # Structured per-arm attribution on REAL producer output: the failing arm
        # is named by the `arm` field (the metric key), not only by stderr text —
        # so a downstream control plane reads verdict.gates.failures[*].arm to
        # learn which arm to shrink/retry, without parsing the message string.
        assert {f["arm"] for f in record["gates"]["failures"]} == {
            "warm_tg_gpu_peak_mb"
        }
        # The wall-clock gate passed even though the VRAM gate failed:
        assert record["break_even_status"] == "warm_win"
        requested = record["gates"]["requested"]
        assert "--require-warm-win" in requested
        assert "--max-warm-gpu-peak-mb=12288.0" in requested

    def test_wallclock_loss_fires_require_warm_win(self, tmp_path):
        """Real pipeline output where warm TG LOSES on wall-clock (350>300).
        --require-warm-win must fire with no_warm_win; an unrequested VRAM gate
        stays out of the verdict entirely."""
        summary_path = self._build_real_summary(
            tmp_path, warm_tg_wall=350.0, warm_tg_gpu=8300.0
        )
        verdict = tmp_path / "verdict.json"
        result = self._run_make_gate(
            {
                "VENV": self._venv_root(),
                "PAPER_SUMMARY": str(summary_path),
                "OUTPUT_PATH": str(verdict),
                "REQUIRE_WARM_WIN": "1",
            }
        )
        assert result.returncode != 0
        assert "--require-warm-win" in result.stderr
        assert "no_warm_win" in result.stderr
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is False
        assert record["gates"]["failures"][0]["gate"] == "--require-warm-win"
        assert record["break_even_status"] == "no_warm_win"
        assert record["gates"]["requested"] == ["--require-warm-win"]

    def test_all_green_path_exits_zero_when_warm_wins_and_under_budget(self, tmp_path):
        """Positive control: real pipeline output that satisfies every enabled
        gate exits ZERO with gates.passed=True — so the target's accept path is
        exercised, not only the reject path (both sides of the boundary)."""
        summary_path = self._build_real_summary(
            tmp_path, warm_tg_wall=240.0, warm_tg_gpu=8300.0
        )
        verdict = tmp_path / "verdict.json"
        result = self._run_make_gate(
            {
                "VENV": self._venv_root(),
                "PAPER_SUMMARY": str(summary_path),
                "OUTPUT_PATH": str(verdict),
                "REQUIRE_WARM_WIN": "1",
                "MAX_WARM_GPU_PEAK_MB": "12288",
                "MAX_BREAK_EVEN_RUNS": "20",  # ber=10.0 <= 20
            }
        )
        assert result.returncode == 0, f"all-green path must exit 0: {result.stderr}"
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is True
        assert record["gates"]["failures"] == []

    # The two AMORTIZATION gates — --max-break-even-runs and --require-one-run-win
    # — answer the actual break-even question ("does the cold build pay off?").
    # Before these tests they were exercised ONLY on the minimal 4-key
    # _single_run_summary unit fixture (TestCLI) and hand-built dicts
    # (TestEvaluateGates) — never against the dense 18-key real producer output,
    # never through the real `make` target. That left the fixture-vs-pipeline gap
    # open for 2 of the 4 gates. The three tests below close it: each reject
    # boundary drives the REAL producer so a regression that only manifests on
    # the dense pipeline shape (e.g. misreading cold_build_seconds out of the
    # full tg_lora_summary surface) is caught here, not only in a unit stub.

    def test_max_break_even_runs_fires_on_real_producer_output(self, tmp_path):
        """Real producer output: warm TG wins on wall-clock (240<300) and is under
        VRAM budget, so --require-warm-win PASSES — but break_even_repeated_runs
        = cold_build(600)/warm_delta(60) = 10.0 EXCEEDS the --max-break-even-runs
        5 budget. The amortization gate must fire INDEPENDENTLY of the warm-win
        gate, naming only itself, with the cold_build/warm_delta decomposition."""
        summary_path = self._build_real_summary(
            tmp_path, warm_tg_wall=240.0, warm_tg_gpu=8300.0
        )
        verdict = tmp_path / "verdict.json"
        result = self._run_make_gate(
            {
                "VENV": self._venv_root(),
                "PAPER_SUMMARY": str(summary_path),
                "OUTPUT_PATH": str(verdict),
                "REQUIRE_WARM_WIN": "1",
                "MAX_BREAK_EVEN_RUNS": "5",  # ber=10.0 > 5
            }
        )
        assert result.returncode != 0, (
            "amortization-boundary violation must exit non-zero"
        )
        assert "--max-break-even-runs" in result.stderr
        assert "break_even_repeated_runs=10.000 exceeds budget 5.0" in result.stderr
        # warm-win passed, so it must NOT be named alongside the amortization gate:
        assert "--require-warm-win" not in result.stderr
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is False
        assert {f["gate"] for f in record["gates"]["failures"]} == {
            "--max-break-even-runs"
        }
        # The wall-clock prerequisite held even though amortization did not:
        assert record["break_even_status"] == "warm_win"
        assert record["break_even_repeated_runs"] == 10.0

    def test_require_one_run_win_fires_on_real_producer_output(self, tmp_path):
        """Real producer output where a single run INCLUDING the cold build does
        NOT come out ahead: one_run_total = cold_build(600) + warm_tg(240) = 840 >
        baseline 300, so one_run_total_delta = -540 <= 0. --require-one-run-win
        must fire naming only itself, with the decomposition that a single run
        (cold build + warm TG) loses to the baseline."""
        summary_path = self._build_real_summary(
            tmp_path, warm_tg_wall=240.0, warm_tg_gpu=8300.0
        )
        verdict = tmp_path / "verdict.json"
        result = self._run_make_gate(
            {
                "VENV": self._venv_root(),
                "PAPER_SUMMARY": str(summary_path),
                "OUTPUT_PATH": str(verdict),
                "REQUIRE_ONE_RUN_WIN": "1",
            }
        )
        assert result.returncode != 0, (
            "single-run-does-not-win violation must exit non-zero"
        )
        assert "--require-one-run-win" in result.stderr
        assert "one_run_total_delta_seconds=-540.000 <= 0" in result.stderr
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is False
        assert {f["gate"] for f in record["gates"]["failures"]} == {
            "--require-one-run-win"
        }
        assert record["gates"]["requested"] == ["--require-one-run-win"]
        assert record["one_run_total_delta_seconds"] == -540.0

    def test_amortization_gates_pass_when_cold_build_is_small(self, tmp_path):
        """Positive control for BOTH amortization gates on real producer output:
        a small cold build (50s) means a single run including the build
        (50+240=290) beats the baseline (300) → one_run_total_delta=+10 > 0, and
        break_even_repeated_runs=50/60=0.833 <= 20. Both amortization gates pass
        (exit 0) on the dense pipeline shape — the accept side these gates had no
        real-output coverage for before. (VRAM/warm-win are left unrequested so
        only the amortization gates' accept path is exercised.)"""
        summary_path = self._build_real_summary(
            tmp_path, warm_tg_wall=240.0, warm_tg_gpu=8300.0, cold_tg_build=50.0
        )
        verdict = tmp_path / "verdict.json"
        result = self._run_make_gate(
            {
                "VENV": self._venv_root(),
                "PAPER_SUMMARY": str(summary_path),
                "OUTPUT_PATH": str(verdict),
                "REQUIRE_ONE_RUN_WIN": "1",
                "MAX_BREAK_EVEN_RUNS": "20",  # ber=0.833 <= 20
            }
        )
        assert result.returncode == 0, (
            f"both amortization gates must pass on small cold build: {result.stderr}"
        )
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is True
        assert record["gates"]["failures"] == []
        assert record["one_run_total_delta_seconds"] == pytest.approx(10.0)
        assert record["break_even_repeated_runs"] == pytest.approx(50.0 / 60.0)


# ---------------------------------------------------------------------------
# make gates-ci — the loop's GATE SEQUENCE (the aggregate that runs EVERY
# GPU-free gate in one target). This is the seam AI_HUB_MAKE_RUN_FEEDBACK #4
# names: "confirm the gates are wired into the loop's gate sequence, else they
# are inert". Until this class NO test exercised `make gates-ci` as a whole —
# only its individual sub-targets (`analyze-prefix-break-even-ci` above,
# `bench-velocity-ops-ci` standalone). Two boundaries are pinned here:
#   accept — the canonical default config exits 0 (every gate green)
#   reject — a PAPER_SUMMARY override pointing the sequence at a producer-
#            faithful VRAM-VIOLATING summary propagates non-zero through the
#            aggregate, proving (a) the sequence now accepts real pipeline
#            output instead of being locked to the fixture and (b) a single
#            gate failing fails the whole sequence (the gate is not inert in
#            the loop's sequence). This closes both the fixture-vs-pipeline
#            gap at the SEQUENCE level and the loop-wiring seam in one exercise.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PRODUCER_AVAILABLE, reason="benchmark_prefix_cache import chain unavailable"
)
@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
class TestGatesCiLoopSequence:
    """The loop's gate sequence (`make gates-ci`), exercised end to end.

    Drives the REAL aggregate target — which itself runs the velocity-ops gate
    then the break-even gate — so a regression in EITHER gate's wiring into the
    sequence, or in the sequence's failure aggregation, fails here. Reuses the
    producer-driven summary builder and venv-derivation from the e2e class
    above (no fixture hand-pruning)."""

    @staticmethod
    def _run_make_gates_ci(env_overrides: dict) -> subprocess.CompletedProcess:
        env = {**os.environ, **env_overrides}
        return subprocess.run(
            ["make", "gates-ci"],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_default_canonical_config_exits_zero(self, tmp_path):
        """Accept boundary for the LOOP SEQUENCE: with no PAPER_SUMMARY
        override, `make gates-ci` runs every GPU-free gate against the checked-in
        canonical fixture (and the velocity-ops micro-benchmark) and exits 0.
        Pins that the aggregate is green on the reviewed default — not only its
        individual sub-targets — so the sequence cannot silently drop a gate."""
        verdict = tmp_path / "gates_default_verdict.json"
        result = self._run_make_gates_ci(
            {
                "VENV": TestEndToEndPipelineGate._venv_root(),
                "OUTPUT_PATH": str(verdict),
            }
        )
        assert result.returncode == 0, (
            f"make gates-ci must exit 0 on the canonical default:\n{result.stderr}"
        )
        # The break-even sub-target recorded its all-green verdict:
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is True
        assert record["gates"]["failures"] == []

    def test_override_propagates_break_even_failure(self, tmp_path):
        """Reject boundary for the LOOP SEQUENCE: point `make gates-ci` at a
        producer-faithful VRAM-VIOLATING summary (warm TG 14000 > 12288 budget)
        via the PAPER_SUMMARY override. The aggregate must exit NON-zero and
        name only the TG arm — proving the sequence (a) now accepts real
        pipeline output instead of being locked to the fixture (the
        fixture-vs-pipeline gap at the sequence level) and (b) propagates a
        break-even gate failure to a non-zero aggregate exit (the gate is not
        inert in the loop's sequence). The under-budget baseline arm must NOT
        be named, pinning per-arm independence through the full sequence."""
        summary_path = TestEndToEndPipelineGate._build_real_summary(
            tmp_path, warm_tg_wall=240.0, warm_tg_gpu=14000.0
        )
        verdict = tmp_path / "gates_violating_verdict.json"
        result = self._run_make_gates_ci(
            {
                "VENV": TestEndToEndPipelineGate._venv_root(),
                "PAPER_SUMMARY": str(summary_path),
                "OUTPUT_PATH": str(verdict),
            }
        )
        assert result.returncode != 0, (
            "VRAM-violating override must fail the whole gates-ci sequence:\n"
            f"{result.stdout}"
        )
        assert "--max-warm-gpu-peak-mb" in result.stderr
        assert "warm_tg_gpu_peak_mb" in result.stderr
        # The under-budget baseline arm (8200 MB) is NOT named:
        assert "warm_baseline_gpu_peak_mb" not in result.stderr
        record = orjson.loads(verdict.read_bytes())
        assert record["gates"]["passed"] is False
        assert {f["gate"] for f in record["gates"]["failures"]} == {
            "--max-warm-gpu-peak-mb"
        }


# ---------------------------------------------------------------------------
# Checked-in canonical fixture — the STABLE paper-summary the GPU-free `gates`
# CI job runs the break-even gate against (AI_HUB_MAKE_RUN_FEEDBACK #4: prove the
# gate target is non-inert by invoking it on every push). CI cannot run a real
# GPU A/B and should not invoke the producer in the `gates` job, so the fixture
# is the portable input. These tests pin that it stays producer-faithful
# (drift-guard) and carries a known budget-compliant verdict + fire boundary.
# ---------------------------------------------------------------------------

_CANONICAL_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "prefix_break_even_canonical_summary.json"
)

# Filesystem paths the producer embeds that vary per tmpdir — they are not part
# of the gate's contract, so the drift-guard exempts only these when comparing
# the checked-in fixture to a freshly produced one.
_VOLATILE_PATH_KEYS = frozenset({"run_dir", "prefix_feature_cache_dir"})


def _strip_volatile_paths(node: object) -> object:
    """Recursively drop per-tmpdir filesystem-path keys so a checked-in fixture
    can be compared for structural + value equality against a fresh producer run
    (whose tmp paths necessarily differ)."""
    if isinstance(node, dict):
        return {
            k: _strip_volatile_paths(v)
            for k, v in node.items()
            if k not in _VOLATILE_PATH_KEYS
        }
    if isinstance(node, list):
        return [_strip_volatile_paths(v) for v in node]
    return node


def _build_canonical_real_summary(tmp_path: Path) -> Path:
    """Drive the REAL `build_benchmark_summary` producer over the CANONICAL
    budget-compliant inputs that produced the checked-in
    `prefix_break_even_canonical_summary.json`. Used by the drift-guard to prove
    that fixture stays byte-faithful to the live producer."""
    run_root = tmp_path / "run"
    _write_run_metrics_jsonl(
        run_root / "cold" / "baseline" / "run_metrics.jsonl",
        wall_seconds=300.0,
        gpu_peak_mb=8200.0,
    )
    _write_run_metrics_jsonl(
        run_root / "cold" / "tg_lora" / "run_metrics.jsonl",
        wall_seconds=360.0,
        gpu_peak_mb=8300.0,
        build_seconds=600.0,
        tg=True,
    )
    _write_run_metrics_jsonl(
        run_root / "warm" / "baseline" / "run_metrics.jsonl",
        wall_seconds=300.0,
        gpu_peak_mb=8200.0,
    )
    _write_run_metrics_jsonl(
        run_root / "warm" / "tg_lora" / "run_metrics.jsonl",
        wall_seconds=240.0,
        gpu_peak_mb=10000.0,
        build_seconds=600.0,
        tg=True,
    )
    summary = build_benchmark_summary(run_root / "cold", run_root / "warm")
    out = tmp_path / "canonical_summary.json"
    out.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    return out


class TestCheckedInCanonicalFixture:
    """The fixture is reviewed-once but exercised forever after; these tests make
    sure it cannot silently drift. None of them need `make` or the producer's
    import chain except the drift-guard, so the boundary pins run in every
    environment (including CI's `gates` job's mental model of 'just a fixture')."""

    def test_fixture_exists_and_is_dense_pipeline_shape(self):
        """The checked-in fixture carries the full producer key surface (>=18
        warm.tg_lora keys), not a hand-pruned 4-key stub — otherwise re-running
        the gate against it in CI would re-open the fixture-vs-pipeline gap."""
        assert _CANONICAL_FIXTURE.exists(), (
            f"missing canonical fixture: {_CANONICAL_FIXTURE}"
        )
        checked_in = orjson.loads(_CANONICAL_FIXTURE.read_bytes())
        assert len(checked_in["warm"]["tg_lora"]) >= 18
        # The gate extractor pulls real values out of this dense shape:
        extracted = _extract_from_single_run(checked_in)
        assert extracted["warm_tg_wall_seconds"] == 240.0
        assert extracted["warm_baseline_wall_seconds"] == 300.0
        assert extracted["cold_build_seconds"] == 600.0

    @pytest.mark.skipif(
        not _PRODUCER_AVAILABLE,
        reason="benchmark_prefix_cache import chain unavailable",
    )
    def test_checked_in_fixture_matches_real_producer_output(self, tmp_path):
        """Drift-guard: regenerate from the canonical inputs via the REAL producer
        and assert structural + value equality with the checked-in fixture
        (exempting only per-tmpdir path fields). Catches a dropped producer field
        (shape drift) AND a changed canonical input (value drift) — so the
        fixture can never silently regress to a stub, and the canonical inputs
        cannot drift from what the fixture actually encodes."""
        regenerated = orjson.loads(_build_canonical_real_summary(tmp_path).read_bytes())
        checked_in = orjson.loads(_CANONICAL_FIXTURE.read_bytes())
        assert _strip_volatile_paths(regenerated) == _strip_volatile_paths(checked_in)

    def test_canonical_fixture_passes_the_ci_gate_config(self):
        """The checked-in fixture MUST satisfy the exact gate config the CI
        `gates` job uses (--require-warm-win --max-warm-gpu-peak-mb 12288) —
        otherwise CI goes red the day this fixture drifts. Pins the accept
        boundary against the stable artifact, in-process (no make, no GPU)."""
        paper = _load_paper_summary(_CANONICAL_FIXTURE)
        result = analyze_break_even(paper, None)
        failures = evaluate_gates(
            result,
            require_warm_win=True,
            max_warm_gpu_peak_mb=12288,
        )
        assert failures == [], (
            f"canonical fixture must pass the CI gate config: {failures}"
        )
        assert result["break_even_status"] == "warm_win"
        # Both warm arms under the 12288 MB RTX 3060 budget:
        assert result["warm_baseline_gpu_peak_mb"] == 8200.0
        assert result["warm_tg_gpu_peak_mb"] == 10000.0

    def test_violating_mutation_fires_naming_only_the_tg_arm(self, tmp_path):
        """The fire boundary against the STABLE fixture: bump warm TG VRAM over
        budget and the gate must fire naming ONLY the TG arm (the baseline arm
        stays under). This is the per-arm independence pin that the CI `gates`
        job's reject-path assertion relies on."""
        mutated = json.loads(_CANONICAL_FIXTURE.read_text())
        mutated["warm"]["tg_lora"]["gpu_peak_mb"] = 14000.0  # over the 12288 budget
        mutated_path = tmp_path / "violating.json"
        mutated_path.write_text(json.dumps(mutated))
        paper = _load_paper_summary(mutated_path)
        result = analyze_break_even(paper, None)
        failures = evaluate_gates(
            result,
            require_warm_win=True,
            max_warm_gpu_peak_mb=12288,
        )
        assert failures, "VRAM-violating mutation must fail the gate"
        assert {f["gate"] for f in failures} == {"--max-warm-gpu-peak-mb"}
        # Structured per-arm attribution: the TG arm is named by the `arm` field;
        # the under-budget baseline arm carries no failure record at all. This is
        # the per-arm independence pin the CI `gates` job's reject assertion reads.
        assert {f["arm"] for f in failures} == {"warm_tg_gpu_peak_mb"}
        # The wall-clock gate still passes (240 < 300) — only VRAM fails:
        assert result["break_even_status"] == "warm_win"
        # The TG arm is named; the under-budget baseline arm is NOT:
        assert all("warm_tg_gpu_peak_mb" in f["message"] for f in failures)
        assert all("warm_baseline_gpu_peak_mb" not in f["message"] for f in failures)
