"""Tests for scripts/evaluate_paper_gates.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/evaluate_paper_gates.py")


def _make_summary(
    *,
    seeds: list[int] | None = None,
    tg_eff: list[float] | None = None,
    bl_eff: list[float] | None = None,
    tg_bp: list[int] | None = None,
    bl_bp: list[int] | None = None,
    tg_loss: list[float] | None = None,
    bl_loss: list[float] | None = None,
    tg_peak: list[float] | None = None,
    bl_peak: list[float] | None = None,
    freed_mb: list[float] | None = None,
) -> dict:
    seeds = seeds or [42, 43, 44]
    n = len(seeds)
    tg_eff = tg_eff or [4.0] * n
    bl_eff = bl_eff or [2.0] * n
    tg_bp = tg_bp or [120] * n
    bl_bp = bl_bp or [240] * n
    tg_loss = tg_loss or [2.5] * n
    bl_loss = bl_loss or [2.5] * n
    tg_peak = tg_peak or [6000.0] * n
    bl_peak = bl_peak or [8000.0] * n
    freed_mb = freed_mb or [4600.0] * n

    import statistics

    per_seed = []
    for i, s in enumerate(seeds):
        per_seed.append({
            "seed": s,
            "warm_tg_loss_red_per_wall_minute": tg_eff[i],
            "warm_baseline_loss_red_per_wall_minute": bl_eff[i],
            "warm_tg_backward_passes": tg_bp[i],
            "warm_baseline_backward_passes": bl_bp[i],
            "warm_tg_best_valid_loss": tg_loss[i],
            "warm_baseline_best_valid_loss": bl_loss[i],
            "warm_tg_gpu_peak_mb": tg_peak[i],
            "warm_baseline_gpu_peak_mb": bl_peak[i],
            "warm_tg_runtime_offload_gpu_freed_mb": freed_mb[i],
        })

    def _agg(vals):
        clean = [v for v in vals if v is not None]
        return {"values": clean, "mean": statistics.mean(clean) if clean else None, "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0}

    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "warm_tg_loss_red_per_wall_minute": _agg(tg_eff),
            "warm_baseline_loss_red_per_wall_minute": _agg(bl_eff),
            "warm_tg_best_valid_loss": _agg(tg_loss),
            "warm_baseline_best_valid_loss": _agg(bl_loss),
            "warm_tg_gpu_peak_mb": _agg(tg_peak),
            "warm_baseline_gpu_peak_mb": _agg(bl_peak),
            "warm_tg_runtime_offload_gpu_freed_mb": _agg(freed_mb),
        },
    }


@pytest.fixture
def summary_dir(tmp_path):
    return tmp_path


def _write_summary(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "aggregate_summary.json"
    p.write_text(json.dumps(data, indent=2))
    return p


class TestG0Hygiene:
    def test_passes_with_valid_summary(self, summary_dir):
        from scripts.evaluate_paper_gates import _check_g0
        summary = _make_summary()
        result = _check_g0(summary)
        assert result["passed"]

    def test_fails_with_no_seeds(self):
        from scripts.evaluate_paper_gates import _check_g0
        result = _check_g0({"aggregate": {}})
        assert not result["passed"]


class TestG1Efficiency:
    def test_passes_when_tg_dominates(self):
        from scripts.evaluate_paper_gates import _check_g1
        summary = _make_summary(
            tg_eff=[4.0, 5.0, 3.5],
            bl_eff=[2.0, 2.0, 2.0],
            tg_bp=[120, 120, 120],
            bl_bp=[240, 240, 240],
            tg_loss=[2.5, 2.5, 2.5],
            bl_loss=[2.5, 2.5, 2.5],
        )
        result = _check_g1(summary)
        assert result["passed"]

    def test_fails_when_one_seed_tg_worse(self):
        from scripts.evaluate_paper_gates import _check_g1
        summary = _make_summary(
            seeds=[42, 43, 44],
            tg_eff=[4.0, 1.5, 3.5],
            bl_eff=[2.0, 2.0, 2.0],
        )
        result = _check_g1(summary)
        assert not result["passed"]

    def test_fails_when_ratio_below_2x(self):
        from scripts.evaluate_paper_gates import _check_g1
        summary = _make_summary(
            tg_eff=[2.5, 2.5, 2.5],
            bl_eff=[2.0, 2.0, 2.0],
        )
        result = _check_g1(summary)
        assert not result["passed"]

    def test_fails_when_quality_degrades(self):
        from scripts.evaluate_paper_gates import _check_g1
        summary = _make_summary(
            tg_loss=[2.6, 2.6, 2.6],
            bl_loss=[2.5, 2.5, 2.5],
        )
        result = _check_g1(summary, quality_tolerance=0.01)
        assert not result["passed"]


class TestG2Memory:
    def test_passes_with_major_memory_reduction(self):
        from scripts.evaluate_paper_gates import _check_g2
        summary = _make_summary(
            tg_peak=[6000.0, 6000.0, 6000.0],
            bl_peak=[8000.0, 8000.0, 8000.0],
            freed_mb=[4600.0, 4600.0, 4600.0],
        )
        result = _check_g2(summary)
        g21 = next(c for c in result["checks"] if c["check"].startswith("G2.1"))
        assert g21["pass"]

    def test_fails_with_insufficient_reduction(self):
        from scripts.evaluate_paper_gates import _check_g2
        summary = _make_summary(
            tg_peak=[7500.0, 7500.0, 7500.0],
            bl_peak=[8000.0, 8000.0, 8000.0],
        )
        result = _check_g2(summary)
        g21 = next(c for c in result["checks"] if c["check"].startswith("G2.1"))
        assert not g21["pass"]

    def test_g23_no_frontier_report_is_fail(self):
        from scripts.evaluate_paper_gates import _check_g2
        summary = _make_summary()
        result = _check_g2(summary)
        g23 = next(c for c in result["checks"] if c["check"].startswith("G2.3"))
        assert not g23["pass"]
        assert "No frontier_report.json" in g23["detail"]

    def test_g23_frontier_detected_passes(self, tmp_path):
        from scripts.evaluate_paper_gates import _check_g2
        frontier = {
            "frontier_separation_detected": True,
            "frontier_boundary": 2048,
            "runs": [
                {"seq_len": 1024, "baseline_status": "completed", "tg_status": "completed", "frontier_separation": False},
                {"seq_len": 2048, "baseline_status": "oom", "tg_status": "completed", "frontier_separation": True},
            ],
        }
        fp = tmp_path / "frontier_report.json"
        fp.write_text(json.dumps(frontier))

        summary = _make_summary()
        result = _check_g2(summary, frontier_report_path=str(fp))
        g23 = next(c for c in result["checks"] if c["check"].startswith("G2.3"))
        assert g23["pass"]
        assert "2048" in g23["detail"]

    def test_g23_no_frontier_detected_fails(self, tmp_path):
        from scripts.evaluate_paper_gates import _check_g2
        frontier = {
            "frontier_separation_detected": False,
            "frontier_boundary": None,
            "runs": [
                {"seq_len": 1024, "baseline_status": "completed", "tg_status": "completed", "frontier_separation": False},
            ],
        }
        fp = tmp_path / "frontier_report.json"
        fp.write_text(json.dumps(frontier))

        summary = _make_summary()
        result = _check_g2(summary, frontier_report_path=str(fp))
        g23 = next(c for c in result["checks"] if c["check"].startswith("G2.3"))
        assert not g23["pass"]
        assert "No frontier separation" in g23["detail"]

    def test_g23_corrupt_frontier_report_fails(self, tmp_path):
        from scripts.evaluate_paper_gates import _check_g2
        fp = tmp_path / "frontier_report.json"
        fp.write_text("NOT JSON{{{")

        summary = _make_summary()
        result = _check_g2(summary, frontier_report_path=str(fp))
        g23 = next(c for c in result["checks"] if c["check"].startswith("G2.3"))
        assert not g23["pass"]
        assert "Failed to read" in g23["detail"]

    def test_g23_missing_frontier_file_fails(self, tmp_path):
        from scripts.evaluate_paper_gates import _check_g2
        summary = _make_summary()
        result = _check_g2(summary, frontier_report_path=str(tmp_path / "nonexistent.json"))
        g23 = next(c for c in result["checks"] if c["check"].startswith("G2.3"))
        assert not g23["pass"]
        assert "No frontier_report.json" in g23["detail"]


class TestFindFrontierReport:
    def test_discovers_in_same_directory(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_frontier_report
        summary = tmp_path / "aggregate_summary.json"
        summary.write_text("{}")
        frontier = tmp_path / "frontier_report.json"
        frontier.write_text("{}")
        assert _find_frontier_report(str(summary)) == frontier

    def test_discovers_in_parent_directory(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_frontier_report
        slen_dir = tmp_path / "slen_2048"
        slen_dir.mkdir()
        summary = slen_dir / "aggregate_summary.json"
        summary.write_text("{}")
        frontier = tmp_path / "frontier_report.json"
        frontier.write_text("{}")
        assert _find_frontier_report(str(summary)) == frontier

    def test_returns_none_when_not_found(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_frontier_report
        summary = tmp_path / "aggregate_summary.json"
        summary.write_text("{}")
        assert _find_frontier_report(str(summary)) is None

    def test_does_not_search_beyond_two_levels(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_frontier_report
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        summary = deep / "aggregate_summary.json"
        summary.write_text("{}")
        frontier = tmp_path / "frontier_report.json"
        frontier.write_text("{}")
        assert _find_frontier_report(str(summary)) is None


class TestCLIEndToEnd:
    def test_exit_0_on_passing_summary(self, summary_dir):
        summary = _make_summary()
        path = _write_summary(summary_dir, summary)
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G2", "G3", "G4"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "G0" in r.stdout
        assert "G1" in r.stdout

    def test_exit_1_on_failing_summary(self, summary_dir):
        summary = _make_summary(tg_eff=[1.0, 1.0, 1.0], bl_eff=[2.0, 2.0, 2.0])
        path = _write_summary(summary_dir, summary)
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G2", "G3", "G4"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1

    def test_json_report_output(self, summary_dir):
        summary = _make_summary()
        path = _write_summary(summary_dir, summary)
        out = summary_dir / "report.json"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G2", "G3", "G4", "-o", str(out)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        report = json.loads(out.read_text())
        assert "gates" in report
        assert "overall_passed" in report

    def test_exit_2_on_missing_file(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/nonexistent/summary.json"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2

    def test_frontier_report_cli_passes_g2(self, summary_dir):
        summary = _make_summary()
        path = _write_summary(summary_dir, summary)
        frontier = {
            "frontier_separation_detected": True,
            "frontier_boundary": 2048,
            "runs": [
                {"seq_len": 2048, "baseline_status": "oom", "tg_status": "completed", "frontier_separation": True},
            ],
        }
        fp = summary_dir / "frontier_report.json"
        fp.write_text(json.dumps(frontier))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G3", "G4",
             "--frontier-report", str(fp)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert "G2.3" in r.stdout

    def test_frontier_report_cli_no_separation_fails_g2(self, summary_dir):
        summary = _make_summary()
        path = _write_summary(summary_dir, summary)
        frontier = {
            "frontier_separation_detected": False,
            "frontier_boundary": None,
            "runs": [],
        }
        fp = summary_dir / "frontier_report.json"
        fp.write_text(json.dumps(frontier))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G3", "G4",
             "--frontier-report", str(fp)],
            capture_output=True, text=True,
        )
        assert r.returncode == 1

    def test_auto_discover_frontier_report_in_parent(self, tmp_path):
        """When frontier_report.json is in the parent dir of the summary, auto-discover it."""
        sweep_dir = tmp_path / "frontier_sweep"
        sweep_dir.mkdir()
        slen_dir = sweep_dir / "slen_2048"
        slen_dir.mkdir()
        summary = _make_summary()
        path = slen_dir / "aggregate_summary.json"
        path.write_text(json.dumps(summary))
        frontier = {
            "frontier_separation_detected": True,
            "frontier_boundary": 2048,
            "runs": [
                {"seq_len": 2048, "baseline_status": "oom", "tg_status": "completed", "frontier_separation": True},
            ],
        }
        (sweep_dir / "frontier_report.json").write_text(json.dumps(frontier))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G3", "G4"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert "Auto-discovered frontier report" in r.stdout
        assert "G2.3" in r.stdout


class TestG4CausalAttribution:
    """REQ-189: _check_g4 ablation comparison logic."""

    def test_no_ablation_summaries_is_informational(self):
        from scripts.evaluate_paper_gates import _check_g4
        summary = _make_summary()
        result = _check_g4(summary)
        assert not result["passed"]
        assert len(result["checks"]) == 1
        assert "Requires cold vs warm" in result["checks"][0]["detail"]

    def test_g41_warm_speedup_all_seeds_pass(self):
        from scripts.evaluate_paper_gates import _check_g4
        warm = _make_summary(tg_eff=[4.0, 5.0, 3.5])
        cold = _make_summary(tg_eff=[2.0, 2.5, 2.0])
        result = _check_g4(warm, cold_summary=cold)
        g41 = next(c for c in result["checks"] if c["check"].startswith("G4.1"))
        assert g41["pass"]

    def test_g41_warm_speedup_one_seed_worse_fails(self):
        from scripts.evaluate_paper_gates import _check_g4
        warm = _make_summary(seeds=[42, 43, 44], tg_eff=[4.0, 1.5, 3.5])
        cold = _make_summary(seeds=[42, 43, 44], tg_eff=[2.0, 2.0, 2.0])
        result = _check_g4(warm, cold_summary=cold)
        g41 = next(c for c in result["checks"] if c["check"].startswith("G4.1"))
        assert not g41["pass"]
        assert "seed 43" in g41["detail"]

    def test_g41_missing_cold_seed_fails(self):
        from scripts.evaluate_paper_gates import _check_g4
        warm = _make_summary(seeds=[42, 43, 44], tg_eff=[4.0, 4.0, 4.0])
        cold = _make_summary(seeds=[42, 43], tg_eff=[2.0, 2.0])
        result = _check_g4(warm, cold_summary=cold)
        g41 = next(c for c in result["checks"] if c["check"].startswith("G4.1"))
        assert not g41["pass"]

    def test_g41_empty_per_seed_fails(self):
        from scripts.evaluate_paper_gates import _check_g4
        warm = {"per_seed": [], "aggregate": {}}
        cold = {"per_seed": [], "aggregate": {}}
        result = _check_g4(warm, cold_summary=cold)
        g41 = next(c for c in result["checks"] if c["check"].startswith("G4.1"))
        assert not g41["pass"]

    def test_g42_cache_on_stronger_pass(self):
        from scripts.evaluate_paper_gates import _check_g4
        cache_on = _make_summary(seeds=[42], tg_peak=[5000.0], bl_peak=[8000.0])
        cache_off = _make_summary(seeds=[42], tg_peak=[7000.0], bl_peak=[8000.0])
        result = _check_g4(cache_on, no_cache_summary=cache_off)
        g42 = next(c for c in result["checks"] if c["check"].startswith("G4.2"))
        assert g42["pass"]

    def test_g42_cache_on_weaker_fails(self):
        from scripts.evaluate_paper_gates import _check_g4
        cache_on = _make_summary(seeds=[42], tg_peak=[7500.0], bl_peak=[8000.0])
        cache_off = _make_summary(seeds=[42], tg_peak=[6000.0], bl_peak=[8000.0])
        result = _check_g4(cache_on, no_cache_summary=cache_off)
        g42 = next(c for c in result["checks"] if c["check"].startswith("G4.2"))
        assert not g42["pass"]

    def test_g42_cache_on_equal_fails(self):
        from scripts.evaluate_paper_gates import _check_g4
        cache_on = _make_summary(seeds=[42], tg_peak=[6000.0], bl_peak=[8000.0])
        cache_off = _make_summary(seeds=[42], tg_peak=[6000.0], bl_peak=[8000.0])
        result = _check_g4(cache_on, no_cache_summary=cache_off)
        g42 = next(c for c in result["checks"] if c["check"].startswith("G4.2"))
        assert not g42["pass"]

    def test_g42_missing_cache_off_seed_fails(self):
        from scripts.evaluate_paper_gates import _check_g4
        cache_on = _make_summary(seeds=[42, 43, 44], tg_peak=[5000.0, 5000.0, 5000.0])
        cache_off = _make_summary(seeds=[42, 43], tg_peak=[7000.0, 7000.0])
        result = _check_g4(cache_on, no_cache_summary=cache_off)
        g42 = next(c for c in result["checks"] if c["check"].startswith("G4.2"))
        assert not g42["pass"]

    def test_both_summaries_both_pass(self):
        from scripts.evaluate_paper_gates import _check_g4
        warm = _make_summary(seeds=[42, 43], tg_eff=[4.0, 4.0], tg_peak=[5000.0, 5000.0], bl_peak=[8000.0, 8000.0])
        cold = _make_summary(seeds=[42, 43], tg_eff=[2.0, 2.0])
        cache_off = _make_summary(seeds=[42, 43], tg_peak=[7000.0, 7000.0], bl_peak=[8000.0, 8000.0])
        result = _check_g4(warm, cold_summary=cold, no_cache_summary=cache_off)
        assert result["passed"]

    def test_both_summaries_one_fails_overall_fails(self):
        from scripts.evaluate_paper_gates import _check_g4
        warm = _make_summary(seeds=[42, 43], tg_eff=[4.0, 1.5], tg_peak=[5000.0, 5000.0], bl_peak=[8000.0, 8000.0])
        cold = _make_summary(seeds=[42, 43], tg_eff=[2.0, 2.0])
        cache_off = _make_summary(seeds=[42, 43], tg_peak=[7000.0, 7000.0], bl_peak=[8000.0, 8000.0])
        result = _check_g4(warm, cold_summary=cold, no_cache_summary=cache_off)
        assert not result["passed"]


class TestG4CLIEndToEnd:
    def test_g4_with_cold_summary_passes(self, summary_dir):
        warm = _make_summary(tg_eff=[4.0, 4.0, 4.0])
        cold = _make_summary(tg_eff=[2.0, 2.0, 2.0])
        warm_path = _write_summary(summary_dir, warm)
        cold_path = summary_dir / "cold_summary.json"
        cold_path.write_text(json.dumps(cold))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(warm_path), "--skip-gates", "G2", "G3",
             "--cold-summary", str(cold_path)],
            capture_output=True, text=True,
        )
        assert "G4.1" in r.stdout

    def test_g4_with_no_cache_summary_passes(self, summary_dir):
        cache_on = _make_summary(seeds=[42], tg_peak=[5000.0], bl_peak=[8000.0])
        cache_off = _make_summary(seeds=[42], tg_peak=[7000.0], bl_peak=[8000.0])
        on_path = _write_summary(summary_dir, cache_on)
        off_path = summary_dir / "no_cache_summary.json"
        off_path.write_text(json.dumps(cache_off))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(on_path), "--skip-gates", "G2", "G3",
             "--no-cache-summary", str(off_path)],
            capture_output=True, text=True,
        )
        assert "G4.2" in r.stdout

    def test_g4_missing_cold_summary_file_exits_2(self, summary_dir):
        summary = _make_summary()
        path = _write_summary(summary_dir, summary)
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G2", "G3",
             "--cold-summary", "/nonexistent/cold.json"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2


class TestFindSiblingSummary:
    """Auto-discovery of cold/no-cache ablation summaries."""

    def test_discovers_cold_directory(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_sibling_summary
        suite = tmp_path / "suite"
        reuse_dir = suite / "reuse"
        reuse_dir.mkdir(parents=True)
        summary = reuse_dir / "aggregate_summary.json"
        summary.write_text("{}")
        cold_dir = suite / "cold"
        cold_dir.mkdir()
        (cold_dir / "aggregate_summary.json").write_text("{}")
        assert _find_sibling_summary(str(summary), suffix="cold") == cold_dir / "aggregate_summary.json"

    def test_discovers_no_cache_directory(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_sibling_summary
        suite = tmp_path / "suite"
        reuse_dir = suite / "reuse"
        reuse_dir.mkdir(parents=True)
        summary = reuse_dir / "aggregate_summary.json"
        summary.write_text("{}")
        nc_dir = suite / "no_cache"
        nc_dir.mkdir()
        (nc_dir / "aggregate_summary.json").write_text("{}")
        assert _find_sibling_summary(str(summary), suffix="no_cache") == nc_dir / "aggregate_summary.json"

    def test_tries_candidate_names_in_order(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_sibling_summary
        suite = tmp_path / "suite"
        reuse_dir = suite / "reuse"
        reuse_dir.mkdir(parents=True)
        summary = reuse_dir / "aggregate_summary.json"
        summary.write_text("{}")
        alt_dir = suite / "cache_off"
        alt_dir.mkdir()
        (alt_dir / "aggregate_summary.json").write_text("{}")
        result = _find_sibling_summary(
            str(summary), suffix="no_cache", candidates=["no_cache", "cache_off", "nocache"],
        )
        assert result == alt_dir / "aggregate_summary.json"

    def test_returns_none_when_no_sibling(self, tmp_path):
        from scripts.evaluate_paper_gates import _find_sibling_summary
        suite = tmp_path / "suite"
        reuse_dir = suite / "reuse"
        reuse_dir.mkdir(parents=True)
        summary = reuse_dir / "aggregate_summary.json"
        summary.write_text("{}")
        assert _find_sibling_summary(str(summary), suffix="cold") is None

    def test_cli_auto_discovers_cold_and_no_cache(self, tmp_path):
        """End-to-end: auto-discover cold/ and no_cache/ sibling dirs."""
        suite = tmp_path / "paper_suite"
        reuse_dir = suite / "reuse"
        reuse_dir.mkdir(parents=True)
        warm = _make_summary(
            seeds=[42], tg_eff=[4.0], tg_peak=[5000.0], bl_peak=[8000.0],
        )
        warm_path = reuse_dir / "aggregate_summary.json"
        warm_path.write_text(json.dumps(warm))

        cold_dir = suite / "cold"
        cold_dir.mkdir()
        cold = _make_summary(seeds=[42], tg_eff=[2.0])
        (cold_dir / "aggregate_summary.json").write_text(json.dumps(cold))

        nc_dir = suite / "no_cache"
        nc_dir.mkdir()
        cache_off = _make_summary(seeds=[42], tg_peak=[7000.0], bl_peak=[8000.0])
        (nc_dir / "aggregate_summary.json").write_text(json.dumps(cache_off))

        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(warm_path), "--skip-gates", "G2", "G3"],
            capture_output=True, text=True,
        )
        assert "Auto-discovered cold ablation summary" in r.stdout, r.stdout
        assert "Auto-discovered cache-off ablation summary" in r.stdout, r.stdout
        assert "G4.1" in r.stdout
        assert "G4.2" in r.stdout


class TestJsonReportTimestamp:
    """Verify generated_at in JSON output."""

    def test_json_report_includes_generated_at(self, summary_dir):
        summary = _make_summary()
        path = _write_summary(summary_dir, summary)
        out = summary_dir / "report.json"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--skip-gates", "G2", "G3", "G4", "-o", str(out)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        report = json.loads(out.read_text())
        assert "generated_at" in report
        assert report["generated_at"].startswith("20")  # sanity: looks like ISO timestamp
