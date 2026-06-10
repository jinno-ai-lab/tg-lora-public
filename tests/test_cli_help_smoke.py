"""CLI smoke tests: verify all Python scripts import and respond to --help (TASK-0124).

Each argparse-based script must exit 0 with --help. Scripts using raw
sys.argv are verified via import only.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Scripts using argparse (exit 0 on --help)
ARGPARSE_SCRIPTS = [
    "scripts/advise_training.py",
    "scripts/analyze_benchmark.py",
    "scripts/analyze_prefix_cache_break_even.py",
    "scripts/analyze_sensitivity.py",
    "scripts/analyze_trajectory.py",
    "scripts/benchmark_optimizer_lifecycle.py",
    "scripts/benchmark_prefix_cache.py",
    "scripts/benchmark_velocity_ops.py",
    "scripts/compare_experiment_configs.py",
    "scripts/compare_paper_memory_modes.py",
    "scripts/compare_runs.py",
    "scripts/consolidate_paper_results.py",
    "scripts/diagnose.py",
    "scripts/download_data.py",
    "scripts/evaluate_paper_gates.py",
    "scripts/export_paper_results.py",
    "scripts/frontier_report.py",
    "scripts/generate_sweep_dashboard.py",
    "scripts/inspect_model.py",
    "scripts/lookup_batch_plan.py",
    "scripts/precompute_prefix_cache_parallel.py",
    "scripts/prepare_data.py",
    "scripts/recover.py",
    "scripts/run_paper_external_eval.py",
    "scripts/summarize_sweep.py",
]

# Scripts using raw sys.argv (no --help) — import-only check
IMPORT_ONLY_SCRIPTS = [
    "scripts/analyze_accel_sweep.py",
]


@pytest.fixture(name="argparse_script_path", params=ARGPARSE_SCRIPTS, ids=lambda s: Path(s).name)
def _argparse_script_path_fixture(request):
    return request.param


@pytest.fixture(name="import_script_path", params=IMPORT_ONLY_SCRIPTS, ids=lambda s: Path(s).name)
def _import_script_path_fixture(request):
    return request.param


class TestCLIHelpSmoke:
    """Verify argparse scripts respond to --help and non-argparse scripts import."""

    def test_argparse_help(self, argparse_script_path):
        """Script exits 0 and prints usage with --help."""
        r = subprocess.run(
            [sys.executable, argparse_script_path, "--help"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
            cwd=str(ROOT),
        )
        assert r.returncode == 0, (
            f"{argparse_script_path} --help exited {r.returncode}\n"
            f"stderr: {r.stderr[:500]}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "usage" in combined or "help" in combined, (
            f"{argparse_script_path} --help: no usage/help in output"
        )

    def test_import_only(self, import_script_path):
        """Non-argparse script can be imported without errors."""
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util; "
                    f"importlib.util.spec_from_file_location('m', '{import_script_path}')"
                ),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
            cwd=str(ROOT),
        )
        assert r.returncode == 0, f"Import check failed for {import_script_path}: {r.stderr}"
