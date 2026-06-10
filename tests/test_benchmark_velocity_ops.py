"""Smoke tests for scripts/benchmark_velocity_ops.py.

Verifies import health, --help, --quick JSON output, required fields,
and baseline regression detection (--baseline/--save-baseline/--threshold).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.benchmark_velocity_ops import (
    benchmark_cap_update,
    benchmark_velocity_ema,
    _compare_with_baseline,
    _round_record,
)


class TestBenchmarkVelocityOpsImport:
    def test_import_sanity(self):
        """Module imports without error."""
        import scripts.benchmark_velocity_ops as mod

        assert hasattr(mod, "main")
        assert hasattr(mod, "benchmark_velocity_ema")
        assert hasattr(mod, "benchmark_cap_update")

    def test_help_flag(self):
        """--help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_velocity_ops.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Micro-benchmark" in result.stdout

    def test_quick_json_output(self):
        """--quick produces valid JSON with required top-level keys."""
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_velocity_ops.py", "--quick"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "velocity_ema" in data
        assert "cap_update" in data
        assert data["iterations"] == 50


class TestBenchmarkVelocityEma:
    def test_returns_required_fields(self):
        out = benchmark_velocity_ema(iterations=5)
        assert "velocity_ema_time_ms" in out
        assert "velocity_ema_per_iter_ms" in out
        assert "velocity_ema_mem_delta_kb" in out
        assert "velocity_ema_iterations" in out
        assert out["velocity_ema_iterations"] == 5

    def test_time_is_positive(self):
        out = benchmark_velocity_ema(iterations=5)
        assert out["velocity_ema_time_ms"] > 0

    def test_round_record(self):
        out = _round_record(benchmark_velocity_ema(iterations=5))
        assert isinstance(out["velocity_ema_time_ms"], float)


class TestBenchmarkCapUpdate:
    def test_returns_required_fields(self):
        out = benchmark_cap_update(iterations=5)
        assert "cap_update_time_ms" in out
        assert "cap_update_per_iter_ms" in out
        assert "cap_update_mem_delta_kb" in out
        assert "cap_update_nocap_time_ms" in out
        assert "cap_update_iterations" in out
        assert out["cap_update_iterations"] == 5

    def test_time_is_positive(self):
        out = benchmark_cap_update(iterations=5)
        assert out["cap_update_time_ms"] > 0

    def test_json_output_has_required_fields(self):
        """Full --quick run includes velocity_ema_time_ms and cap_update_time_ms."""
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_velocity_ops.py", "--quick"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "velocity_ema_time_ms" in data["velocity_ema"]
        assert "cap_update_time_ms" in data["cap_update"]


class TestBaselineRegressionDetection:
    """Tests for --baseline, --save-baseline, and --threshold CLI flags."""

    def test_save_baseline_creates_file(self, tmp_path):
        """--save-baseline writes a valid JSON file with benchmark results."""
        baseline_file = tmp_path / "baseline.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_velocity_ops.py",
                "--quick",
                f"--save-baseline={baseline_file}",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert baseline_file.exists()
        data = json.loads(baseline_file.read_text())
        assert "velocity_ema" in data
        assert "cap_update" in data

    def test_baseline_no_regression_exits_zero(self, tmp_path):
        """--baseline with a generous baseline exits 0 (no regression)."""
        baseline = {
            "velocity_ema": {"velocity_ema_per_iter_ms": 0.05},
            "cap_update": {
                "cap_update_per_iter_ms": 1.0,
                "cap_update_nocap_per_iter_ms": 0.5,
            },
        }
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(baseline))
        result = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_velocity_ops.py",
                "--quick",
                f"--baseline={baseline_file}",
                "--threshold=9999",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_baseline_regression_exits_nonzero(self, tmp_path):
        """--baseline with a very tight baseline exits 1 (regression detected)."""
        baseline = {
            "velocity_ema": {"velocity_ema_per_iter_ms": 1e-9},
            "cap_update": {
                "cap_update_per_iter_ms": 1e-9,
                "cap_update_nocap_per_iter_ms": 1e-9,
            },
        }
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(baseline))
        result = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_velocity_ops.py",
                "--quick",
                f"--baseline={baseline_file}",
                "--threshold=10",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, (
            f"Expected exit code 1 for regression, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "REGRESSION" in result.stderr

    def test_baseline_missing_file_exits_nonzero(self, tmp_path):
        """--baseline with a non-existent file exits non-zero."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_velocity_ops.py",
                "--quick",
                f"--baseline={tmp_path / 'nonexistent.json'}",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_compare_with_baseline_no_regression(self):
        """_compare_with_baseline returns no regressions when within threshold."""
        current = {
            "velocity_ema": {"velocity_ema_per_iter_ms": 0.12},
            "cap_update": {
                "cap_update_per_iter_ms": 2.5,
                "cap_update_nocap_per_iter_ms": 1.6,
            },
        }
        baseline = {
            "velocity_ema": {"velocity_ema_per_iter_ms": 0.10},
            "cap_update": {
                "cap_update_per_iter_ms": 2.0,
                "cap_update_nocap_per_iter_ms": 1.5,
            },
        }
        regressions = _compare_with_baseline(current, baseline, threshold_pct=50.0)
        assert len(regressions) == 0

    def test_compare_with_baseline_detects_regression(self):
        """_compare_with_baseline detects when current exceeds baseline + threshold."""
        current = {
            "velocity_ema": {"velocity_ema_per_iter_ms": 0.5},
            "cap_update": {
                "cap_update_per_iter_ms": 2.5,
                "cap_update_nocap_per_iter_ms": 1.6,
            },
        }
        baseline = {
            "velocity_ema": {"velocity_ema_per_iter_ms": 0.10},
            "cap_update": {
                "cap_update_per_iter_ms": 2.0,
                "cap_update_nocap_per_iter_ms": 1.5,
            },
        }
        regressions = _compare_with_baseline(current, baseline, threshold_pct=20.0)
        assert len(regressions) >= 1
        labels = [r["metric"] for r in regressions]
        assert "velocity_ema_per_iter_ms" in labels

    def test_threshold_flag_controls_sensitivity(self, tmp_path):
        """Higher threshold makes regression detection more lenient."""
        baseline = {
            "velocity_ema": {"velocity_ema_per_iter_ms": 0.08},
            "cap_update": {
                "cap_update_per_iter_ms": 1.5,
                "cap_update_nocap_per_iter_ms": 1.0,
            },
        }
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(baseline))

        # Tight threshold → regression
        result_tight = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_velocity_ops.py",
                "--quick",
                f"--baseline={baseline_file}",
                "--threshold=1",
            ],
            capture_output=True,
            text=True,
        )
        # Very loose threshold → no regression
        result_loose = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_velocity_ops.py",
                "--quick",
                f"--baseline={baseline_file}",
                "--threshold=9999",
            ],
            capture_output=True,
            text=True,
        )
        assert result_tight.returncode == 1
        assert result_loose.returncode == 0


class TestCIGateBaseline:
    """Tests for the checked-in CI gate baseline (REQ-149)."""

    BASELINE_PATH = Path("baselines/velocity_ops.json")

    def test_baseline_file_exists(self):
        assert self.BASELINE_PATH.exists(), (
            "baselines/velocity_ops.json must be checked into the repository"
        )

    def test_baseline_file_valid_json(self):
        data = json.loads(self.BASELINE_PATH.read_text())
        assert "velocity_ema" in data
        assert "cap_update" in data
        for section in ("velocity_ema", "cap_update"):
            assert f"{section}_per_iter_ms" in data[section] or (
                section == "cap_update"
                and "cap_update_per_iter_ms" in data[section]
            )

    def test_ci_gate_passes(self):
        """Simulate `make bench-velocity-ops-ci` — must exit 0 against checked-in baseline."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_velocity_ops.py",
                "--quick",
                "--baseline",
                str(self.BASELINE_PATH),
                "--threshold",
                "50",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"CI gate failed (regression detected).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        data = json.loads(result.stdout)
        assert data["baseline_comparison"]["regressed"] is False
