"""Tests for the PSA ablation/sweep shell scripts (TC-280-01, TC-281-01).

run_psa_ablation.sh and run_psa_gamma_sweep.sh orchestrate multi-hour GPU
training runs, so we verify the scripts' *structure* (the 3 base conditions run
in order; the γ sweep covers the documented grid) and bash syntax rather than
executing the full training. This mirrors how test_summarize_psa_sweep.py
treats the Python summary tool — it asserts on behaviour, not a live training
run.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ABLATION_SCRIPT = Path("scripts/run_psa_ablation.sh")
GAMMA_SWEEP_SCRIPT = Path("scripts/run_psa_gamma_sweep.sh")


def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path.read_text()


@pytest.fixture(scope="module", autouse=True)
def _require_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


class TestAblationScriptStructure:
    """TC-280-01: run_psa_ablation.sh runs the 3 base conditions in order
    (plain baseline → LAWA-only → PSA default) on a shared backward-pass budget."""

    def test_bash_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(ABLATION_SCRIPT)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_three_base_conditions_in_order(self):
        src = _read(ABLATION_SCRIPT)
        baseline = src.index('_run_baseline "baseline_plain"')
        lawa = src.index('_run_psa "lawa_only"')
        psa = src.index('_run_psa "psa_default"')
        assert baseline < lawa < psa, (
            "base conditions must run in order: "
            "baseline_plain, lawa_only, psa_default"
        )

    def test_uses_parity_matched_baseline_config(self):
        # GOAL §3.3 fair comparison: the baseline must be dropout/scope-parity
        # matched to the PSA config, not the default 9b_baseline.yaml.
        src = _read(ABLATION_SCRIPT)
        assert "9b_baseline_suffix_only_last25.yaml" in src


class TestGammaSweepScriptStructure:
    """TC-281-01: run_psa_gamma_sweep.sh sweeps γ over {0.0, 0.5, 1.0, 2.0}
    (γ=0.0 is the no-amplification ablation baseline)."""

    def test_bash_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(GAMMA_SWEEP_SCRIPT)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_gamma_values_grid(self):
        src = _read(GAMMA_SWEEP_SCRIPT)
        m = re.search(r"GAMMA_VALUES=\(([^)]*)\)", src)
        assert m, "GAMMA_VALUES array not found"
        values = m.group(1).split()
        assert values == ["0.0", "0.5", "1.0", "2.0"], values

    def test_experiment_naming_per_grid_point(self):
        # Each (γ, regime_reset) grid point produces a distinct run dir.
        src = _read(GAMMA_SWEEP_SCRIPT)
        assert 'experiment_name="gamma_${gamma}_reset_${reset_tag}"' in src
