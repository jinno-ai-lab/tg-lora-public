"""Tests for scripts/frontier_report.py — Stage 3 Memory Frontier Sweep logic."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.frontier_report import (
    _read_run_meta,
    _split_oom_log,
    build_frontier_report,
    detect_oom_from_log,
    determine_status,
    find_frontier_boundary,
)


def _make_summary(
    *,
    seeds: list[int] | None = None,
    tg_peak: list[float] | None = None,
    bl_peak: list[float] | None = None,
    tg_wall: list[float] | None = None,
    bl_wall: list[float] | None = None,
) -> dict:
    import statistics

    n = len(tg_peak) if tg_peak else len(seeds) if seeds else 2
    seeds = seeds or list(range(42, 42 + n))
    tg_peak = tg_peak or [6000.0] * n
    bl_peak = bl_peak or [8000.0] * n
    tg_wall = tg_wall or [120.0] * n
    bl_wall = bl_wall or [240.0] * n

    per_seed = []
    for i, s in enumerate(seeds):
        per_seed.append({
            "seed": s,
            "warm_tg_gpu_peak_mb": tg_peak[i],
            "warm_baseline_gpu_peak_mb": bl_peak[i],
            "warm_tg_wall_seconds": tg_wall[i],
            "warm_baseline_wall_seconds": bl_wall[i],
        })

    def _agg(vals):
        clean = [v for v in vals if v is not None]
        return {
            "values": clean,
            "mean": statistics.mean(clean) if clean else None,
            "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        }

    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "warm_tg_gpu_peak_mb": _agg(tg_peak),
            "warm_baseline_gpu_peak_mb": _agg(bl_peak),
            "warm_tg_wall_seconds": _agg(tg_wall),
            "warm_baseline_wall_seconds": _agg(bl_wall),
        },
    }


class TestOOMDetection:
    def test_cuda_oom_pattern(self):
        assert detect_oom_from_log("RuntimeError: CUDA out of memory")

    def test_cuda_error_pattern(self):
        assert detect_oom_from_log("CUDA error: an illegal memory access was encountered")

    def test_oom_in_stderr(self):
        log = "Epoch 5: loss=2.3\nRuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        assert detect_oom_from_log(log)

    def test_no_oom_passes(self):
        assert not detect_oom_from_log("Training completed successfully")

    def test_empty_string_passes(self):
        assert not detect_oom_from_log("")

    def test_kill_signal(self):
        assert detect_oom_from_log("Killed")


class TestStatusDetermination:
    def test_completed_run(self):
        assert determine_status(exit_code=0, log="", summary_exists=True) == "completed"

    def test_oom_from_exit_code(self):
        assert determine_status(exit_code=1, log="CUDA out of memory", summary_exists=False) == "oom"

    def test_oom_from_log_only(self):
        assert determine_status(exit_code=1, log="CUDA out of memory", summary_exists=True) == "oom"

    def test_killed(self):
        assert determine_status(exit_code=137, log="", summary_exists=False) == "oom"

    def test_failed_no_oom(self):
        assert determine_status(exit_code=1, log="ValueError: bad config", summary_exists=False) == "failed"


class TestFrontierBoundary:
    def test_single_frontier(self):
        runs = [
            {"seq_len": 1024, "baseline_status": "completed", "tg_status": "completed"},
            {"seq_len": 1536, "baseline_status": "completed", "tg_status": "completed"},
            {"seq_len": 2048, "baseline_status": "oom", "tg_status": "completed"},
            {"seq_len": 3072, "baseline_status": "oom", "tg_status": "oom"},
        ]
        result = find_frontier_boundary(runs)
        assert result == 2048

    def test_no_frontier(self):
        runs = [
            {"seq_len": 1024, "baseline_status": "completed", "tg_status": "completed"},
            {"seq_len": 1536, "baseline_status": "completed", "tg_status": "completed"},
        ]
        assert find_frontier_boundary(runs) is None

    def test_all_oom(self):
        runs = [
            {"seq_len": 2048, "baseline_status": "oom", "tg_status": "oom"},
            {"seq_len": 3072, "baseline_status": "oom", "tg_status": "oom"},
        ]
        assert find_frontier_boundary(runs) is None

    def test_first_is_frontier(self):
        runs = [
            {"seq_len": 1536, "baseline_status": "oom", "tg_status": "completed"},
            {"seq_len": 2048, "baseline_status": "oom", "tg_status": "oom"},
        ]
        assert find_frontier_boundary(runs) == 1536


class TestBuildFrontierReport:
    def test_mixed_results(self, tmp_path):
        seq1_dir = tmp_path / "slen_1024"
        seq1_dir.mkdir()
        summary1 = _make_summary(tg_peak=[6000.0], bl_peak=[8000.0])
        (seq1_dir / "aggregate_summary.json").write_text(json.dumps(summary1))

        seq2_dir = tmp_path / "slen_2048"
        seq2_dir.mkdir()
        summary2 = _make_summary(tg_peak=[9000.0], bl_peak=[11000.0])
        (seq2_dir / "aggregate_summary.json").write_text(json.dumps(summary2))

        run_infos = [
            {"seq_len": 1024, "run_dir": str(seq1_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
            {"seq_len": 2048, "run_dir": str(seq2_dir), "baseline_exit": 1, "tg_exit": 0, "baseline_log": "CUDA out of memory", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        assert report["seq_lens"] == [1024, 2048]
        assert len(report["runs"]) == 2

        r0 = report["runs"][0]
        assert r0["seq_len"] == 1024
        assert r0["baseline_status"] == "completed"
        assert r0["tg_status"] == "completed"
        assert not r0["frontier_separation"]

        r1 = report["runs"][1]
        assert r1["seq_len"] == 2048
        assert r1["baseline_status"] == "oom"
        assert r1["tg_status"] == "completed"
        assert r1["frontier_separation"]

        assert report["frontier_boundary"] == 2048
        assert report["frontier_separation_detected"]

    def test_all_completed_no_frontier(self, tmp_path):
        seq_dir = tmp_path / "slen_1024"
        seq_dir.mkdir()
        summary = _make_summary()
        (seq_dir / "aggregate_summary.json").write_text(json.dumps(summary))

        run_infos = [
            {"seq_len": 1024, "run_dir": str(seq_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        assert not report["frontier_separation_detected"]
        assert report["frontier_boundary"] is None

    def test_missing_summary_treated_as_failed(self, tmp_path):
        missing_dir = tmp_path / "slen_2048"
        missing_dir.mkdir()

        run_infos = [
            {"seq_len": 2048, "run_dir": str(missing_dir), "baseline_exit": 1, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        assert report["runs"][0]["baseline_status"] == "failed"
        assert report["runs"][0]["tg_status"] == "completed"

    def test_report_json_serializable(self, tmp_path):
        seq_dir = tmp_path / "slen_1024"
        seq_dir.mkdir()
        summary = _make_summary()
        (seq_dir / "aggregate_summary.json").write_text(json.dumps(summary))

        run_infos = [
            {"seq_len": 1024, "run_dir": str(seq_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        text = json.dumps(report, indent=2)
        parsed = json.loads(text)
        assert parsed["seq_lens"] == [1024]

    def test_peak_memory_included_from_summary(self, tmp_path):
        seq_dir = tmp_path / "slen_1536"
        seq_dir.mkdir()
        summary = _make_summary(tg_peak=[5500.0], bl_peak=[7500.0])
        (seq_dir / "aggregate_summary.json").write_text(json.dumps(summary))

        run_infos = [
            {"seq_len": 1536, "run_dir": str(seq_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        run = report["runs"][0]
        assert run["tg_peak_mb"] == 5500.0
        assert run["baseline_peak_mb"] == 7500.0

    def test_memory_delta_and_savings_pct(self, tmp_path):
        seq_dir = tmp_path / "slen_1536"
        seq_dir.mkdir()
        summary = _make_summary(tg_peak=[5000.0], bl_peak=[8000.0])
        (seq_dir / "aggregate_summary.json").write_text(json.dumps(summary))

        run_infos = [
            {"seq_len": 1536, "run_dir": str(seq_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        run = report["runs"][0]
        assert run["memory_delta_mb"] == pytest.approx(3000.0)
        assert run["memory_savings_pct"] == pytest.approx(37.5)

    def test_memory_delta_missing_when_no_summary(self, tmp_path):
        missing_dir = tmp_path / "slen_2048"
        missing_dir.mkdir()

        run_infos = [
            {"seq_len": 2048, "run_dir": str(missing_dir), "baseline_exit": 1, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        assert "memory_delta_mb" not in report["runs"][0]
        assert "memory_savings_pct" not in report["runs"][0]

    def test_avg_memory_savings_pct(self, tmp_path):
        seq1_dir = tmp_path / "slen_1024"
        seq1_dir.mkdir()
        summary1 = _make_summary(tg_peak=[6000.0], bl_peak=[8000.0])
        (seq1_dir / "aggregate_summary.json").write_text(json.dumps(summary1))

        seq2_dir = tmp_path / "slen_2048"
        seq2_dir.mkdir()
        summary2 = _make_summary(tg_peak=[9000.0], bl_peak=[12000.0])
        (seq2_dir / "aggregate_summary.json").write_text(json.dumps(summary2))

        run_infos = [
            {"seq_len": 1024, "run_dir": str(seq1_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
            {"seq_len": 2048, "run_dir": str(seq2_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        # Total delta = 2000 + 3000 = 5000, total baseline = 8000 + 12000 = 20000
        assert report["avg_memory_savings_pct"] == pytest.approx(25.0)

    def test_avg_memory_savings_none_when_no_completed(self, tmp_path):
        missing_dir = tmp_path / "slen_4096"
        missing_dir.mkdir()

        run_infos = [
            {"seq_len": 4096, "run_dir": str(missing_dir), "baseline_exit": 1, "tg_exit": 1, "baseline_log": "CUDA out of memory", "tg_log": "CUDA out of memory"},
        ]

        report = build_frontier_report(run_infos)
        assert report["avg_memory_savings_pct"] is None

    def test_report_includes_generated_at_timestamp(self, tmp_path):
        seq_dir = tmp_path / "slen_1024"
        seq_dir.mkdir()
        summary = _make_summary()
        (seq_dir / "aggregate_summary.json").write_text(json.dumps(summary))

        run_infos = [
            {"seq_len": 1024, "run_dir": str(seq_dir), "baseline_exit": 0, "tg_exit": 0, "baseline_log": "", "tg_log": ""},
        ]

        report = build_frontier_report(run_infos)
        assert "generated_at" in report
        assert "T" in report["generated_at"]  # ISO 8601 format


class TestSplitOomLog:
    def test_baseline_oom_line(self):
        bl, tg = _split_oom_log("baseline step 50: CUDA out of memory")
        assert "CUDA out of memory" in bl
        assert tg == ""

    def test_tg_oom_line(self):
        bl, tg = _split_oom_log("tg step 50: CUDA out of memory")
        assert bl == ""
        assert "CUDA out of memory" in tg

    def test_unattributed_oom_goes_to_both(self):
        bl, tg = _split_oom_log("RuntimeError: CUDA out of memory")
        assert "CUDA out of memory" in bl
        assert "CUDA out of memory" in tg

    def test_no_oom_lines(self):
        bl, tg = _split_oom_log("Training completed\nLoss: 2.3")
        assert bl == ""
        assert tg == ""

    def test_mixed_lines(self):
        log = (
            "baseline step 50: CUDA out of memory\n"
            "tg step 100: completed\n"
            "tg step 120: Killed\n"
            "normal line\n"
        )
        bl, tg = _split_oom_log(log)
        assert "baseline" in bl.lower()
        assert "Killed" in tg
        assert "completed" not in tg


class TestReadRunMeta:
    def test_reads_exit_code(self, tmp_path):
        (tmp_path / "make_exit_code").write_text("1\n")
        meta = _read_run_meta(tmp_path)
        assert meta["make_exit"] == 1

    def test_reads_log(self, tmp_path):
        (tmp_path / "make_output.log").write_text("CUDA out of memory\n")
        meta = _read_run_meta(tmp_path)
        assert "CUDA out of memory" in meta["make_log"]

    def test_missing_files_default(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        meta = _read_run_meta(empty_dir)
        assert meta["make_exit"] == 0
        assert meta["make_log"] == ""

    def test_corrupt_exit_code(self, tmp_path):
        (tmp_path / "make_exit_code").write_text("not_a_number\n")
        meta = _read_run_meta(tmp_path)
        assert meta["make_exit"] == 0

    def test_reads_run_metadata_json(self, tmp_path):
        """run_metadata.json is the primary source when present."""
        meta_json = {
            "seq_len": 2048,
            "make_exit": 1,
            "summary_exists": False,
            "oom_in_log": True,
        }
        (tmp_path / "run_metadata.json").write_text(json.dumps(meta_json))
        meta = _read_run_meta(tmp_path)
        assert meta["make_exit"] == 1
        assert meta["summary_exists"] is False
        assert meta["oom_in_log"] is True

    def test_metadata_json_overrides_individual_files(self, tmp_path):
        """When both run_metadata.json and make_exit_code exist, JSON wins."""
        meta_json = {"seq_len": 2048, "make_exit": 137, "summary_exists": False, "oom_in_log": True}
        (tmp_path / "run_metadata.json").write_text(json.dumps(meta_json))
        (tmp_path / "make_exit_code").write_text("0\n")
        meta = _read_run_meta(tmp_path)
        assert meta["make_exit"] == 137

    def test_corrupt_metadata_json_falls_back(self, tmp_path):
        (tmp_path / "run_metadata.json").write_text("NOT JSON{{{")
        (tmp_path / "make_exit_code").write_text("1\n")
        meta = _read_run_meta(tmp_path)
        assert meta["make_exit"] == 1


class TestMainIntegration:
    """Integration tests invoking frontier_report.py main() via subprocess.

    Simulates the directory structure written by run_frontier_sweep.sh:
    each run directory may contain make_exit_code, make_output.log, and
    optionally aggregate_summary.json.  Verifies the full CLI pipeline
    produces correct status classifications.
    """

    def _run_main(self, runs: list[str], output: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "scripts/frontier_report.py",
             "--runs", *runs,
             "--output", str(output)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )

    def test_completed_run(self, tmp_path):
        run_dir = tmp_path / "slen_1024"
        run_dir.mkdir()
        (run_dir / "make_exit_code").write_text("0\n")
        (run_dir / "make_output.log").write_text("Training completed\n")
        summary = _make_summary(tg_peak=[6000.0], bl_peak=[8000.0])
        (run_dir / "aggregate_summary.json").write_text(json.dumps(summary))
        output = tmp_path / "report.json"

        proc = self._run_main([f"1024:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        assert report["runs"][0]["baseline_status"] == "completed"
        assert report["runs"][0]["tg_status"] == "completed"

    def test_oom_run_from_log(self, tmp_path):
        run_dir = tmp_path / "slen_3072"
        run_dir.mkdir()
        (run_dir / "make_exit_code").write_text("1\n")
        (run_dir / "make_output.log").write_text(
            "baseline step 80: CUDA out of memory. Tried to allocate 4 GiB\n"
        )
        # No aggregate_summary.json — make exited before TG ran.
        output = tmp_path / "report.json"

        proc = self._run_main([f"3072:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        r = report["runs"][0]
        assert r["baseline_status"] == "oom"
        # TG never ran — no summary, make exited non-zero.
        assert r["tg_status"] == "failed"

    def test_failed_run_no_oom(self, tmp_path):
        run_dir = tmp_path / "slen_2048"
        run_dir.mkdir()
        (run_dir / "make_exit_code").write_text("1\n")
        (run_dir / "make_output.log").write_text("ValueError: bad config\n")
        output = tmp_path / "report.json"

        proc = self._run_main([f"2048:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        r = report["runs"][0]
        assert r["baseline_status"] == "failed"
        assert r["tg_status"] == "failed"

    def test_killed_run_signal_137(self, tmp_path):
        run_dir = tmp_path / "slen_4096"
        run_dir.mkdir()
        (run_dir / "make_exit_code").write_text("137\n")
        (run_dir / "make_output.log").write_text("Training...\n")
        output = tmp_path / "report.json"

        proc = self._run_main([f"4096:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        r = report["runs"][0]
        assert r["baseline_status"] == "oom"
        assert r["tg_status"] == "oom"

    def test_mixed_runs_frontier_detection(self, tmp_path):
        # Run 1: both succeed
        ok_dir = tmp_path / "slen_1024"
        ok_dir.mkdir()
        (ok_dir / "make_exit_code").write_text("0\n")
        (ok_dir / "make_output.log").write_text("Done\n")
        summary_ok = _make_summary(tg_peak=[5000.0], bl_peak=[7000.0])
        (ok_dir / "aggregate_summary.json").write_text(json.dumps(summary_ok))

        # Run 2: baseline OOM, TG succeeds (frontier!)
        frontier_dir = tmp_path / "slen_2048"
        frontier_dir.mkdir()
        (frontier_dir / "make_exit_code").write_text("1\n")
        (frontier_dir / "make_output.log").write_text(
            "baseline: CUDA out of memory\n"
        )
        summary_ft = _make_summary(tg_peak=[9000.0], bl_peak=[11000.0])
        (frontier_dir / "aggregate_summary.json").write_text(json.dumps(summary_ft))

        # Run 3: both OOM
        oom_dir = tmp_path / "slen_3072"
        oom_dir.mkdir()
        (oom_dir / "make_exit_code").write_text("1\n")
        (oom_dir / "make_output.log").write_text("CUDA out of memory\n")

        output = tmp_path / "report.json"
        proc = self._run_main(
            [f"1024:{ok_dir}", f"2048:{frontier_dir}", f"3072:{oom_dir}"],
            output,
        )
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        assert report["frontier_boundary"] == 2048
        assert report["frontier_separation_detected"]

        runs = {r["seq_len"]: r for r in report["runs"]}
        assert runs[1024]["baseline_status"] == "completed"
        assert runs[1024]["tg_status"] == "completed"
        assert runs[2048]["baseline_status"] == "oom"
        assert runs[2048]["tg_status"] == "completed"
        assert runs[3072]["baseline_status"] == "oom"
        assert runs[3072]["tg_status"] == "oom"

    def test_missing_metadata_backward_compat(self, tmp_path):
        """Run dir with no make_exit_code or make_output.log defaults gracefully."""
        run_dir = tmp_path / "slen_1536"
        run_dir.mkdir()
        # Only aggregate_summary.json — no exit code or log files.
        summary = _make_summary()
        (run_dir / "aggregate_summary.json").write_text(json.dumps(summary))
        output = tmp_path / "report.json"

        proc = self._run_main([f"1536:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        r = report["runs"][0]
        assert r["baseline_status"] == "completed"
        assert r["tg_status"] == "completed"


class TestMetadataJsonIntegration:
    """End-to-end tests using run_metadata.json as written by run_frontier_sweep.sh.

    These tests simulate the full data pipeline: the shell script writes
    ``run_metadata.json`` + ``make_output.log``, and frontier_report.py
    reads them to produce correct status classifications.
    """

    def _run_main(self, runs: list[str], output: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "scripts/frontier_report.py",
             "--runs", *runs,
             "--output", str(output)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )

    def test_oom_via_metadata_json(self, tmp_path):
        """OOM run with run_metadata.json produces status=oom."""
        run_dir = tmp_path / "slen_3072"
        run_dir.mkdir()
        meta_json = {
            "seq_len": 3072,
            "make_exit": 1,
            "summary_exists": False,
            "oom_in_log": True,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(meta_json))
        (run_dir / "make_output.log").write_text(
            "baseline step 80: CUDA out of memory. Tried to allocate 4 GiB\n"
        )
        output = tmp_path / "report.json"

        proc = self._run_main([f"3072:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        r = report["runs"][0]
        assert r["baseline_status"] == "oom"
        assert r["tg_status"] == "failed"

    def test_completed_via_metadata_json(self, tmp_path):
        """Completed run with run_metadata.json + summary produces status=completed."""
        run_dir = tmp_path / "slen_1024"
        run_dir.mkdir()
        meta_json = {
            "seq_len": 1024,
            "make_exit": 0,
            "summary_exists": True,
            "oom_in_log": False,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(meta_json))
        (run_dir / "make_output.log").write_text("Training completed\n")
        summary = _make_summary(tg_peak=[6000.0], bl_peak=[8000.0])
        (run_dir / "aggregate_summary.json").write_text(json.dumps(summary))
        output = tmp_path / "report.json"

        proc = self._run_main([f"1024:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        r = report["runs"][0]
        assert r["baseline_status"] == "completed"
        assert r["tg_status"] == "completed"
        assert r["tg_peak_mb"] == 6000.0
        assert r["memory_delta_mb"] == pytest.approx(2000.0)

    def test_killed_via_metadata_json(self, tmp_path):
        """Killed (signal 137) run via metadata JSON → status=oom."""
        run_dir = tmp_path / "slen_4096"
        run_dir.mkdir()
        meta_json = {
            "seq_len": 4096,
            "make_exit": 137,
            "summary_exists": False,
            "oom_in_log": True,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(meta_json))
        (run_dir / "make_output.log").write_text("Training...\nKilled\n")
        output = tmp_path / "report.json"

        proc = self._run_main([f"4096:{run_dir}"], output)
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        r = report["runs"][0]
        assert r["baseline_status"] == "oom"
        assert r["tg_status"] == "oom"

    def test_frontier_detection_via_metadata_json(self, tmp_path):
        """Mixed runs via metadata JSON detect frontier boundary correctly."""
        # Run 1: both succeed
        ok_dir = tmp_path / "slen_1024"
        ok_dir.mkdir()
        (ok_dir / "run_metadata.json").write_text(json.dumps({
            "seq_len": 1024, "make_exit": 0, "summary_exists": True, "oom_in_log": False,
        }))
        (ok_dir / "make_output.log").write_text("Done\n")
        (ok_dir / "aggregate_summary.json").write_text(
            json.dumps(_make_summary(tg_peak=[5000.0], bl_peak=[7000.0]))
        )

        # Run 2: baseline OOM, TG succeeds (frontier!)
        frontier_dir = tmp_path / "slen_2048"
        frontier_dir.mkdir()
        (frontier_dir / "run_metadata.json").write_text(json.dumps({
            "seq_len": 2048, "make_exit": 1, "summary_exists": True, "oom_in_log": True,
        }))
        (frontier_dir / "make_output.log").write_text("baseline: CUDA out of memory\n")
        (frontier_dir / "aggregate_summary.json").write_text(
            json.dumps(_make_summary(tg_peak=[9000.0], bl_peak=[11000.0]))
        )

        # Run 3: both OOM
        oom_dir = tmp_path / "slen_3072"
        oom_dir.mkdir()
        (oom_dir / "run_metadata.json").write_text(json.dumps({
            "seq_len": 3072, "make_exit": 1, "summary_exists": False, "oom_in_log": True,
        }))
        (oom_dir / "make_output.log").write_text("CUDA out of memory\n")

        output = tmp_path / "report.json"
        proc = self._run_main(
            [f"1024:{ok_dir}", f"2048:{frontier_dir}", f"3072:{oom_dir}"],
            output,
        )
        assert proc.returncode == 0, proc.stderr

        report = json.loads(output.read_text())
        assert report["frontier_boundary"] == 2048
        assert report["frontier_separation_detected"]

        runs = {r["seq_len"]: r for r in report["runs"]}
        assert runs[1024]["baseline_status"] == "completed"
        assert runs[1024]["tg_status"] == "completed"
        assert runs[2048]["baseline_status"] == "oom"
        assert runs[2048]["tg_status"] == "completed"
        assert runs[3072]["baseline_status"] == "oom"
        assert runs[3072]["tg_status"] == "oom"
