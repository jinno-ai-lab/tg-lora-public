"""Smoke tests for operational shell scripts (REQ-210~215).

Validates: existence, executability, shebang, set -euo pipefail,
key variable references, and structural correctness.

TC-211-01 coverage: run_ablation_suite.sh structural validation.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS_UNDER_TEST = [
    "scripts/run_sweep.sh",
    "scripts/run_high_lr_comparison.sh",
    "scripts/run_kstep_rollback_test.sh",
    "scripts/run_accel_sweep_parallel.sh",
    "scripts/run_accel_sweep_auto.sh",
]


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text()


class TestScriptCommon:
    """Properties every operational script must satisfy."""

    @staticmethod
    def _params():
        return [(s, s.split("/")[-1]) for s in SCRIPTS_UNDER_TEST]

    def test_exists(self):
        for rel, _ in self._params():
            assert (REPO_ROOT / rel).is_file(), f"{rel} must exist"

    def test_executable(self):
        for rel, _ in self._params():
            mode = (REPO_ROOT / rel).stat().st_mode
            assert mode & 0o111, f"{rel} must be executable"

    def test_bash_shebang(self):
        for rel, _ in self._params():
            first = (REPO_ROOT / rel).read_text().splitlines()[0]
            assert "bash" in first, f"{rel} must have bash shebang, got: {first}"

    def test_set_strict(self):
        for rel, _ in self._params():
            text = _read(rel)
            assert "set -euo pipefail" in text, f"{rel} must set -euo pipefail"


# ── run_sweep.sh (REQ-210) ────────────────────────────────────────────────


class TestRunSweep:
    SRC = "scripts/run_sweep.sh"

    def test_references_sweep_budget(self):
        assert "SWEEP_BUDGET" in _read(self.SRC)

    def test_references_sweep_dir(self):
        assert "SWEEP_DIR" in _read(self.SRC)

    def test_references_config(self):
        assert "9b_tg_lora.yaml" in _read(self.SRC)

    def test_calls_summarize_sweep(self):
        assert "summarize_sweep.py" in _read(self.SRC)

    def test_sweep_grid_defined(self):
        assert "SWEEP_GRID" in _read(self.SRC)

    def test_sweep_grid_has_nine_configs(self):
        text = _read(self.SRC)
        count = sum(1 for line in text.splitlines() if line.strip().startswith('"') and "|" in line)
        assert count >= 9, f"Expected at least 9 sweep configs, found {count}"

    def test_uses_eval_rollback_tolerance_override(self):
        text = _read(self.SRC)
        assert "eval.rollback_tolerance" in text
        assert "tg_lora.rollback_tolerance" not in text

    def test_computes_cycles_after_overrides(self):
        text = _read(self.SRC)
        override_idx = text.index("for pair in '${overrides}'.split(','):")
        cycle_idx = text.index("cfg.training.max_cycles = max(1")
        assert override_idx < cycle_idx

    def test_defaults_to_single_visible_gpu(self):
        text = _read(self.SRC)
        assert 'CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"' in text
        assert 'env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora' in text

    def test_forces_single_gpu_config_defaults(self):
        text = _read(self.SRC)
        assert 'cfg.model.device_map = None' in text
        assert 'cfg.model.device = None' in text


# ── run_high_lr_comparison.sh (REQ-212) ───────────────────────────────────


class TestRunHighLrComparison:
    SRC = "scripts/run_high_lr_comparison.sh"

    def test_references_budget(self):
        assert "BUDGET" in _read(self.SRC)

    def test_references_comparison_mode(self):
        assert 'COMPARISON_MODE="${COMPARISON_MODE:-stability}"' in _read(self.SRC)

    def test_references_baseline_config(self):
        assert "9b_baseline.yaml" in _read(self.SRC)

    def test_references_max_seq_len(self):
        assert 'MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"' in _read(self.SRC)

    def test_references_quick_eval_examples(self):
        assert 'QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-64}"' in _read(self.SRC)

    def test_references_baseline_pretrain_reference_toggle(self):
        assert 'BASELINE_PRETRAIN_REFERENCE_EVAL_ENABLED="${BASELINE_PRETRAIN_REFERENCE_EVAL_ENABLED:-true}"' in _read(self.SRC)

    def test_references_tglora_config(self):
        assert "9b_tg_lora.yaml" in _read(self.SRC)

    def test_defines_experiments_array(self):
        assert "EXPERIMENTS" in _read(self.SRC)

    def test_includes_high_lr_variants(self):
        text = _read(self.SRC)
        assert "0.002" in text or "2e-3" in text
        assert "0.005" in text or "5e-3" in text

    def test_writes_run_metrics(self):
        assert "run_metrics.jsonl" in _read(self.SRC)

    def test_baseline_budget_converts_backward_passes_to_steps(self):
        text = _read(self.SRC)
        assert "cfg.training.max_steps = max(1, ${ACTUAL_BUDGET_PASSES} // cfg.training.grad_accumulation)" in text

    def test_parity_mode_normalizes_shared_budget_unit(self):
        text = _read(self.SRC)
        assert "math.lcm" in text
        assert "ACTUAL_BUDGET_PASSES=$(( BUDGET / COMPARABLE_BUDGET_UNIT * COMPARABLE_BUDGET_UNIT ))" in text

    def test_summary_python_uses_heredoc(self):
        text = _read(self.SRC)
        assert 'OUTPUT_BASE="${OUTPUT_BASE}" ${VENV_PYTHON} - <<\'PY\'' in text
        assert 'Path(os.environ["OUTPUT_BASE"])' in text

    def test_summary_includes_backward_passes_column(self):
        text = _read(self.SRC)
        assert "{'BP':>8}" in text

    def test_defaults_to_single_visible_gpu(self):
        text = _read(self.SRC)
        assert 'CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"' in text
        assert 'env "${RUN_ENV[@]}" "PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF_VALUE}"' in text

    def test_forces_single_gpu_config_defaults(self):
        text = _read(self.SRC)
        assert 'cfg.model.device_map = None' in text
        assert 'cfg.model.device = None' in text


# ── run_kstep_rollback_test.sh (REQ-213) ──────────────────────────────────


class TestRunKstepRollbackTest:
    SRC = "scripts/run_kstep_rollback_test.sh"

    def test_references_budget(self):
        assert "BUDGET" in _read(self.SRC)

    def test_references_tglora_config(self):
        assert "9b_tg_lora" in _read(self.SRC)

    def test_defines_k_parameter(self):
        text = _read(self.SRC)
        assert "K=" in text or '${K"' in text or '"$4"' in text

    def test_writes_run_metrics(self):
        assert "run_metrics.jsonl" in _read(self.SRC)

    def test_references_grad_accum(self):
        assert "GRAD_ACCUM" in _read(self.SRC)

    def test_converts_budget_to_tg_cycles_using_grad_accum(self):
        assert "local n_cycles=$((BUDGET / (K * GRAD_ACCUM)))" in _read(self.SRC)

    def test_converts_budget_to_baseline_steps_using_grad_accum(self):
        assert "local steps=$((BUDGET / GRAD_ACCUM))" in _read(self.SRC)

    def test_defaults_to_single_visible_gpu(self):
        text = _read(self.SRC)
        assert 'CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"' in text
        assert 'env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora' in text
        assert 'env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora' in text


# ── run_accel_sweep_parallel.sh (REQ-214) ─────────────────────────────────


class TestRunAccelSweepParallel:
    SRC = "scripts/run_accel_sweep_parallel.sh"

    def test_references_max_cycles(self):
        assert "MAX_CYCLES" in _read(self.SRC)

    def test_references_output_base(self):
        assert "OUTPUT_BASE" in _read(self.SRC)

    def test_references_cuda_devices(self):
        text = _read(self.SRC)
        assert "cuda:0" in text
        assert "cuda:1" in text

    def test_references_accel_configs(self):
        text = _read(self.SRC)
        assert "accel_no_accel" in text
        assert "accel_conservative" in text
        assert "accel_balanced" in text
        assert "accel_aggressive" in text

    def test_has_run_config_function(self):
        assert "run_config()" in _read(self.SRC)

    def test_references_report_dir(self):
        assert "REPORT_DIR" in _read(self.SRC)


# ── run_accel_sweep_auto.sh (REQ-215) ─────────────────────────────────────


class TestRunAccelSweepAuto:
    SRC = "scripts/run_accel_sweep_auto.sh"

    def test_references_gpu_index(self):
        assert "GPU_INDEX" in _read(self.SRC)

    def test_references_gpu_threshold(self):
        assert "GPU_THRESHOLD_MB" in _read(self.SRC)

    def test_references_poll_interval(self):
        assert "POLL_INTERVAL" in _read(self.SRC)

    def test_references_log_file(self):
        assert "LOG_FILE" in _read(self.SRC)

    def test_has_log_function(self):
        assert "log()" in _read(self.SRC)

    def test_calls_nvidia_smi(self):
        assert "nvidia-smi" in _read(self.SRC)

    def test_has_sleep_for_polling(self):
        assert "sleep" in _read(self.SRC)


# ── run_ablation_suite.sh (REQ-211) ──────────────────────────────────────


class TestRunAblationSuite:
    SRC = "scripts/run_ablation_suite.sh"

    def test_references_target_bp(self):
        assert "TARGET_BP" in _read(self.SRC)

    def test_references_baseline_config(self):
        assert "9b_baseline.yaml" in _read(self.SRC)

    def test_references_paper_poc_config(self):
        assert "9b_tg_lora_paper_poc.yaml" in _read(self.SRC)

    def test_references_adaptive_k5_config(self):
        assert "9b_tg_lora_adaptive_k5.yaml" in _read(self.SRC)

    def test_references_adaptive_no_conv_config(self):
        assert "9b_tg_lora_adaptive_k5_no_conv.yaml" in _read(self.SRC)

    def test_runs_all_four_variants(self):
        text = _read(self.SRC)
        assert "_run_baseline" in text
        assert "_run_tg" in text

    def test_writes_run_metrics(self):
        assert "run_metrics.jsonl" in _read(self.SRC)

    def test_passes_output_base_via_environment(self):
        text = _read(self.SRC)
        assert 'OUTPUT_BASE="${OUTPUT_BASE}" ${VENV_PYTHON} - <<\'PY\'' in text
        assert 'Path(os.environ["OUTPUT_BASE"])' in text

    def test_defaults_to_single_visible_gpu(self):
        text = _read(self.SRC)
        assert 'CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"' in text
        assert 'env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora' in text
        assert 'env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora' in text

    def test_forces_single_gpu_config_defaults(self):
        text = _read(self.SRC)
        assert 'cfg.model.device_map = None' in text
        assert 'cfg.model.device = None' in text


class TestRunFrontierSweep:
    SRC = "scripts/run_frontier_sweep.sh"

    def test_defaults_to_single_visible_gpu(self):
        text = _read(self.SRC)
        assert 'CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"' in text
        assert 'env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora' in text
        assert 'env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora' in text

    def test_overrides_model_device_map_to_none(self):
        text = _read(self.SRC)
        assert '--override "model.device_map=null"' in text
        assert '--override "model.device=null"' in text
