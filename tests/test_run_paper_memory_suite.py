"""Tests for run_paper_memory_suite.sh dry-run mode (TASK-0111)."""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path("scripts/run_paper_memory_suite.sh")
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "paper_memory_suite"


@pytest.fixture()
def fake_configs(tmp_path: Path) -> dict[str, Path]:
    """Create minimal baseline and TG config fixtures."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    baseline = configs_dir / "baseline.yaml"
    baseline.write_text(textwrap.dedent("""\
        experiment:
          name: test_baseline
          seed: 42
        training:
          max_steps: 10
    """))

    tg = configs_dir / "tg.yaml"
    tg.write_text(textwrap.dedent("""\
        experiment:
          name: test_tg
          seed: 42
        training:
          max_steps: 10
        prefix_feature_cache:
          prefix_feature_cache_train: true
    """))

    return {"baseline": baseline, "tg": tg, "dir": configs_dir}


@pytest.fixture()
def tg_config_missing_pfc(tmp_path: Path) -> Path:
    """TG config without prefix_feature_cache_train key."""
    tg = tmp_path / "configs" / "tg_no_pfc.yaml"
    tg.parent.mkdir(parents=True, exist_ok=True)
    tg.write_text(textwrap.dedent("""\
        experiment:
          name: test_tg_no_pfc
          seed: 42
        training:
          max_steps: 10
        prefix_feature_cache:
          enabled: true
    """))
    return tg


def _run_dry_run(
    tmp_path: Path,
    baseline: Path,
    tg: Path,
    seeds: str = "42",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "VENV_PYTHON": "python3",
        "SEEDS": seeds,
        "OUTPUT_BASE": str(tmp_path / "output"),
        "BASELINE_CONFIG": str(baseline),
        "TG_CONFIG": str(tg),
        "CACHE_BASE": str(tmp_path / "cache"),
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


class TestDryRunSkipsTraining:
    """DRY_RUN=true must not execute benchmark_prefix_cache.py."""

    def test_no_benchmark_invocation(self, tmp_path: Path, fake_configs: dict):
        result = _run_dry_run(tmp_path, fake_configs["baseline"], fake_configs["tg"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "benchmark_prefix_cache" not in result.stdout
        assert "benchmark_prefix_cache" not in result.stderr

    def test_no_aggregation_run(self, tmp_path: Path, fake_configs: dict):
        result = _run_dry_run(tmp_path, fake_configs["baseline"], fake_configs["tg"])
        assert "DRY RUN" in result.stdout or "dry-run" in result.stdout.lower()

    def test_output_dir_created(self, tmp_path: Path, fake_configs: dict):
        output_base = tmp_path / "output"
        _run_dry_run(tmp_path, fake_configs["baseline"], fake_configs["tg"])
        assert output_base.exists()


class TestDryRunConfigValidation:
    """Dry-run validates config existence and content."""

    def test_missing_baseline_config_fails(self, tmp_path: Path, fake_configs: dict):
        nonexistent = tmp_path / "nonexistent_baseline.yaml"
        result = _run_dry_run(tmp_path, nonexistent, fake_configs["tg"])
        assert result.returncode != 0

    def test_missing_tg_config_fails(self, tmp_path: Path, fake_configs: dict):
        nonexistent = tmp_path / "nonexistent_tg.yaml"
        result = _run_dry_run(tmp_path, fake_configs["baseline"], nonexistent)
        assert result.returncode != 0


class TestDryRunPrefixFeatureCacheValidation:
    """Dry-run verifies prefix_feature_cache_train: true in TG config."""

    def test_missing_pfc_train_flag_warns(self, tmp_path: Path, fake_configs: dict, tg_config_missing_pfc: Path):
        result = _run_dry_run(tmp_path, fake_configs["baseline"], tg_config_missing_pfc)
        combined = result.stdout + result.stderr
        assert (
            "prefix_feature_cache_train" in combined
        ), "Should mention missing prefix_feature_cache_train setting"


class TestDryRunSeedExpansion:
    """Dry-run validates seed expansion without running training."""

    def test_multi_seed_expansion(self, tmp_path: Path, fake_configs: dict):
        result = _run_dry_run(tmp_path, fake_configs["baseline"], fake_configs["tg"], seeds="42 43 44")
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        output_base = tmp_path / "output"
        assert (output_base / "seed_42").exists()
        assert (output_base / "seed_43").exists()
        assert (output_base / "seed_44").exists()

    def test_single_seed(self, tmp_path: Path, fake_configs: dict):
        result = _run_dry_run(tmp_path, fake_configs["baseline"], fake_configs["tg"], seeds="42")
        assert result.returncode == 0
        output_base = tmp_path / "output"
        assert (output_base / "seed_42").exists()
        assert not (output_base / "seed_43").exists()


class TestDryRunValidationReport:
    """Dry-run produces a clear validation report."""

    def test_reports_all_checks(self, tmp_path: Path, fake_configs: dict):
        result = _run_dry_run(tmp_path, fake_configs["baseline"], fake_configs["tg"])
        combined = result.stdout + result.stderr
        assert "seed" in combined.lower()
        assert "config" in combined.lower()
