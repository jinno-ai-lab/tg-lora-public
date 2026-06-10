"""Tests for scripts/recover.py — verifies automated fault recovery."""

import textwrap
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.recover import (
    RecoveryResult,
    analyze_fault,
    apply_remediation,
    generate_recovery_config,
    sanitize_checkpoint,
)


# ---------------------------------------------------------------------------
# RecoveryResult
# ---------------------------------------------------------------------------


class TestRecoveryResult:
    def test_to_dict(self):
        r = RecoveryResult("test", "ok", "message", {"key": "val"})
        d = r.to_dict()
        assert d["action"] == "test"
        assert d["status"] == "ok"
        assert d["message"] == "message"
        assert d["details"] == {"key": "val"}

    def test_default_details(self):
        r = RecoveryResult("test", "ok", "msg")
        assert r.details == {}


# ---------------------------------------------------------------------------
# analyze_fault
# ---------------------------------------------------------------------------


class TestAnalyzeFault:
    def test_missing_directory(self):
        results = analyze_fault("/nonexistent/path")
        assert any(r.status == "error" and "not found" in r.message for r in results)

    def test_empty_directory(self, tmp_path):
        results = analyze_fault(str(tmp_path))
        assert any(r.status == "warn" and "No log files" in r.message for r in results)

    def test_oom_detected(self, tmp_path):
        (tmp_path / "train.log").write_text("Step 50: torch.cuda.OutOfMemoryError\n")
        results = analyze_fault(str(tmp_path))
        assert any("oom" in r.message and r.status == "warn" for r in results)
        assert any("OOM recovery" in r.message for r in results)

    def test_nan_detected(self, tmp_path):
        (tmp_path / "train.log").write_text("Step 100: loss is NaN\n")
        results = analyze_fault(str(tmp_path))
        assert any("nan_loss" in r.message and r.status == "warn" for r in results)
        assert any("NaN/instability recovery" in r.message for r in results)

    def test_cuda_error_detected(self, tmp_path):
        (tmp_path / "train.log").write_text(
            "CUDA error: device-side assert triggered\n"
        )
        results = analyze_fault(str(tmp_path))
        assert any("cuda_error" in r.message and r.status == "warn" for r in results)
        assert any("CUDA error recovery" in r.message for r in results)

    def test_multiple_faults(self, tmp_path):
        (tmp_path / "train.log").write_text(
            textwrap.dedent("""\
            Step 50: torch.cuda.OutOfMemoryError
            Step 51: loss is NaN
        """)
        )
        results = analyze_fault(str(tmp_path))
        faults = [r.message for r in results if "Fault detected" in r.message]
        assert len(faults) >= 2

    def test_oom_checkpoint_detected(self, tmp_path):
        (tmp_path / "oom_checkpoint").mkdir()
        results = analyze_fault(str(tmp_path))
        assert any("OOM checkpoint" in r.message and r.status == "ok" for r in results)

    def test_training_state_detected(self, tmp_path):
        torch.save({"cycle": 5}, tmp_path / "training_state.pt")
        results = analyze_fault(str(tmp_path))
        assert any("Training state" in r.message and r.status == "ok" for r in results)

    def test_clean_log_no_faults(self, tmp_path):
        (tmp_path / "train.log").write_text("Step 1: loss=2.5\nStep 2: loss=2.3\n")
        results = analyze_fault(str(tmp_path))
        assert any(
            "No fault patterns" in r.message and r.status == "ok" for r in results
        )


# ---------------------------------------------------------------------------
# sanitize_checkpoint
# ---------------------------------------------------------------------------


class TestSanitizeCheckpoint:
    def test_missing_directory(self):
        results = sanitize_checkpoint("/nonexistent")
        assert any(r.status == "error" and "not found" in r.message for r in results)

    def test_sanitize_nan_in_safetensors(self, tmp_path):
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text('{"r": 16}')

        tensor_with_nan = torch.tensor([1.0, float("nan"), 3.0])
        save_file(
            {"lora_A.weight": tensor_with_nan}, ckpt / "adapter_model.safetensors"
        )

        out_dir = tmp_path / "clean"
        results = sanitize_checkpoint(str(ckpt), str(out_dir))

        assert any("Sanitized" in r.message for r in results)

        # Verify the output is clean
        clean_tensors = load_file(out_dir / "adapter_model.safetensors")
        assert torch.isfinite(clean_tensors["lora_A.weight"]).all()
        assert clean_tensors["lora_A.weight"][1].item() == 0.0  # NaN → 0

    def test_already_clean_passes(self, tmp_path):
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text('{"r": 16}')
        save_file({"w": torch.ones(4)}, ckpt / "adapter_model.safetensors")

        results = sanitize_checkpoint(str(ckpt))
        assert any("already finite" in r.message for r in results)

    def test_config_copied_to_output(self, tmp_path):
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        config_content = '{"r": 16, "alpha": 32}'
        (ckpt / "adapter_config.json").write_text(config_content)
        save_file({"w": torch.ones(2)}, ckpt / "adapter_model.safetensors")

        out_dir = tmp_path / "output"
        sanitize_checkpoint(str(ckpt), str(out_dir))

        assert (out_dir / "adapter_config.json").exists()
        assert (out_dir / "adapter_model.safetensors").exists()

    def test_training_state_sanitized(self, tmp_path):
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text('{"r": 16}')
        save_file({"w": torch.ones(2)}, ckpt / "adapter_model.safetensors")

        state = {"Velocity": torch.tensor([1.0, float("nan"), 3.0])}
        torch.save(state, ckpt / "training_state.pt")

        results = sanitize_checkpoint(str(ckpt))
        assert any("training_state" in r.message for r in results)

    def test_no_safetensors_warns(self, tmp_path):
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        results = sanitize_checkpoint(str(ckpt))
        assert any("No adapter_model.safetensors" in r.message for r in results)

    def test_inplace_sanitization(self, tmp_path):
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text('{"r": 16}')
        bad_tensor = torch.tensor([float("nan"), 2.0])
        save_file({"w": bad_tensor}, ckpt / "adapter_model.safetensors")

        sanitize_checkpoint(str(ckpt))  # no output_dir → in-place

        clean = load_file(ckpt / "adapter_model.safetensors")
        assert torch.isfinite(clean["w"]).all()


# ---------------------------------------------------------------------------
# generate_recovery_config
# ---------------------------------------------------------------------------


class TestGenerateRecoveryConfig:
    def _write_config(self, tmp_path, **overrides) -> Path:
        config = {
            "data": {
                "max_seq_len": 2048,
                "train_path": "data/train.jsonl",
                "valid_quick_path": "data/valid.jsonl",
                "valid_full_path": "data/valid.jsonl",
            },
            "training": {
                "batch_size": 2,
                "grad_accumulation": 8,
                "learning_rate": 2e-4,
                "max_cycles": 100,
                "gradient_checkpointing": False,
            },
            "tg_lora": {
                "alpha_initial": 0.3,
                "relative_update_cap": 0.005,
                "K_initial": 3,
                "K_candidates": [2, 3, 5],
                "N_initial": 5,
                "N_candidates": [1, 3, 5],
                "beta_initial": 0.8,
                "beta_candidates": [0.5, 0.8],
                "alpha_min": 0.03,
                "alpha_max": 1.5,
                "alpha_log_sigma": 0.15,
                "lr_initial": 5e-4,
                "lr_min": 1e-5,
                "lr_max": 1e-3,
                "active_layer_strategy": "last_25_percent",
            },
            "experiment": {"name": "test", "seed": 42},
            "model": {"name_or_path": "test-model"},
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": "all-linear",
            },
            "logging": {"run_dir": "runs/test"},
            "eval": {
                "quick_eval_examples": 32,
                "full_eval_every_cycles": 10,
                "rollback_tolerance": 0.005,
            },
        }
        config.update(overrides)
        import yaml

        cfg_path = tmp_path / "test_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)
        return cfg_path

    def test_oom_recovery_reduces_seq_len(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        output = tmp_path / "recovery.yaml"
        result = generate_recovery_config(str(cfg_path), "oom", str(output))

        assert result.status == "ok"
        assert "max_seq_len" in str(result.details.get("changes", {}))

        import yaml

        with open(output) as f:
            recovered = yaml.safe_load(f)
        assert recovered["data"]["max_seq_len"] == 1024  # halved from 2048

    def test_oom_enables_gradient_checkpointing(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        output = tmp_path / "recovery.yaml"
        generate_recovery_config(str(cfg_path), "oom", str(output))

        import yaml

        with open(output) as f:
            recovered = yaml.safe_load(f)
        assert recovered["training"]["gradient_checkpointing"] is True

    def test_nan_recovery_halves_alpha(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        output = tmp_path / "recovery.yaml"
        result = generate_recovery_config(str(cfg_path), "nan_loss", str(output))

        assert result.status == "ok"
        import yaml

        with open(output) as f:
            recovered = yaml.safe_load(f)
        assert recovered["tg_lora"]["alpha_initial"] == pytest.approx(0.15)
        assert recovered["tg_lora"]["relative_update_cap"] == pytest.approx(0.0025)

    def test_cuda_recovery_reduces_batch(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        output = tmp_path / "recovery.yaml"
        result = generate_recovery_config(str(cfg_path), "cuda_error", str(output))

        assert result.status == "ok"
        import yaml

        with open(output) as f:
            recovered = yaml.safe_load(f)
        assert recovered["training"]["batch_size"] == 1
        assert recovered["training"]["grad_accumulation"] == 16

    def test_missing_config_returns_error(self):
        result = generate_recovery_config("/nonexistent.yaml", "oom", "/tmp/out.yaml")
        assert result.status == "error"

    def test_output_dir_created(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        output = tmp_path / "deep" / "nested" / "recovery.yaml"
        result = generate_recovery_config(str(cfg_path), "oom", str(output))
        assert result.status == "ok"
        assert output.exists()


# ---------------------------------------------------------------------------
# apply_remediation (integration)
# ---------------------------------------------------------------------------


class TestApplyRemediation:
    def _setup_oom_run(self, tmp_path) -> Path:
        """Create a simulated run directory with OOM fault indicators."""
        # Write a log with OOM
        (tmp_path / "train.log").write_text("Step 50: torch.cuda.OutOfMemoryError\n")

        # Create OOM checkpoint
        oom_dir = tmp_path / "oom_checkpoint"
        oom_dir.mkdir()
        (oom_dir / "adapter_config.json").write_text('{"r": 16}')
        save_file({"w": torch.ones(4)}, oom_dir / "adapter_model.safetensors")

        # Write a config
        import yaml

        config = {
            "data": {
                "max_seq_len": 2048,
                "train_path": "data/train.jsonl",
                "valid_quick_path": "data/valid.jsonl",
                "valid_full_path": "data/valid.jsonl",
            },
            "training": {
                "batch_size": 1,
                "grad_accumulation": 8,
                "learning_rate": 2e-4,
                "max_cycles": 100,
                "gradient_checkpointing": True,
            },
            "tg_lora": {
                "alpha_initial": 0.3,
                "relative_update_cap": 0.005,
                "K_initial": 3,
                "K_candidates": [2, 3, 5],
                "N_initial": 5,
                "N_candidates": [1, 3, 5],
                "beta_initial": 0.8,
                "beta_candidates": [0.5, 0.8],
                "alpha_min": 0.03,
                "alpha_max": 1.5,
                "alpha_log_sigma": 0.15,
                "lr_initial": 5e-4,
                "lr_min": 1e-5,
                "lr_max": 1e-3,
                "active_layer_strategy": "last_25_percent",
            },
            "experiment": {"name": "test", "seed": 42},
            "model": {"name_or_path": "test-model"},
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": "all-linear",
            },
            "logging": {"run_dir": "runs/test"},
            "eval": {
                "quick_eval_examples": 32,
                "full_eval_every_cycles": 10,
                "rollback_tolerance": 0.005,
            },
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)
        return cfg_path

    def test_full_oom_remediation(self, tmp_path):
        cfg_path = self._setup_oom_run(tmp_path)
        results = apply_remediation(str(tmp_path), str(cfg_path))

        # Should have analysis, sanitization, and config generation results
        actions = [r.action for r in results]
        assert "analyze" in actions
        assert "fix-config" in actions

        # Should produce a recovery config
        assert (tmp_path / "recovery_config.yaml").exists()

        # Recovery config should have reduced seq_len
        import yaml

        with open(tmp_path / "recovery_config.yaml") as f:
            recovered = yaml.safe_load(f)
        assert recovered["data"]["max_seq_len"] == 1024

    def test_no_fault_returns_early(self, tmp_path):
        (tmp_path / "train.log").write_text("Step 1: loss=2.5\n")

        import yaml

        config = {"data": {"max_seq_len": 2048}, "training": {"batch_size": 1}}
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)

        results = apply_remediation(str(tmp_path), str(cfg_path))
        assert any("No fault detected" in r.message for r in results)
        assert not (tmp_path / "recovery_config.yaml").exists()

    def test_nan_remediation_with_corrupted_checkpoint(self, tmp_path):
        (tmp_path / "train.log").write_text("Step 100: loss is NaN\n")

        import yaml

        config = {
            "data": {
                "max_seq_len": 2048,
                "train_path": "d",
                "valid_quick_path": "d",
                "valid_full_path": "d",
            },
            "training": {
                "batch_size": 1,
                "grad_accumulation": 8,
                "learning_rate": 2e-4,
                "max_cycles": 100,
            },
            "tg_lora": {
                "alpha_initial": 0.3,
                "relative_update_cap": 0.005,
                "K_initial": 3,
                "K_candidates": [2, 3, 5],
                "N_initial": 5,
                "N_candidates": [1, 3, 5],
                "beta_initial": 0.8,
                "beta_candidates": [0.5, 0.8],
                "alpha_min": 0.03,
                "alpha_max": 1.5,
                "alpha_log_sigma": 0.15,
                "lr_initial": 5e-4,
                "lr_min": 1e-5,
                "lr_max": 1e-3,
                "active_layer_strategy": "last_25_percent",
            },
            "experiment": {"name": "test", "seed": 42},
            "model": {"name_or_path": "test"},
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": "all-linear",
            },
            "logging": {"run_dir": "runs/test"},
            "eval": {
                "quick_eval_examples": 32,
                "full_eval_every_cycles": 10,
                "rollback_tolerance": 0.005,
            },
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)

        results = apply_remediation(str(tmp_path), str(cfg_path))
        assert any("NaN/instability recovery" in r.message for r in results)

        with open(tmp_path / "recovery_config.yaml") as f:
            recovered = yaml.safe_load(f)
        assert recovered["tg_lora"]["alpha_initial"] == pytest.approx(0.15)
