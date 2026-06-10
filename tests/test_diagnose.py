"""Tests for scripts/diagnose.py — verifies health check automation."""

import textwrap
from unittest.mock import patch


from scripts.diagnose import (
    CheckResult,
    check_config,
    check_checkpoint,
    check_logs,
    check_gpu,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# check_gpu
# ---------------------------------------------------------------------------


class TestCheckGPU:
    def test_no_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            # Force reimport path
            results = check_gpu()
            # When torch can't be imported, we get an error
            assert any(r.status == "error" for r in results)

    def test_no_cuda(self):
        import types

        import torch as real_torch
        import src.utils.device as _dev_mod

        _prev = _dev_mod._CACHED_DEVICE
        _dev_mod._CACHED_DEVICE = None
        try:
            mock_torch = types.ModuleType("torch")
            mock_torch.cuda = types.ModuleType("torch.cuda")
            mock_torch.cuda.is_available = lambda: False
            mock_torch.cuda.device_count = lambda: 0
            mock_torch.version = types.SimpleNamespace(cuda=None)
            mock_torch.backends = types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False, is_built=lambda: False)
            )
            mock_torch.device = real_torch.device

            with patch.dict("sys.modules", {"torch": mock_torch}), \
                 patch.object(_dev_mod, "torch", mock_torch):
                results = check_gpu()
                assert any("No GPU detected" in r.message for r in results)
        finally:
            _dev_mod._CACHED_DEVICE = _prev

    def test_gpu_with_low_memory(self):
        import types

        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = types.ModuleType("torch.cuda")
        mock_torch.cuda.is_available = lambda: True
        mock_torch.cuda.device_count = lambda: 1
        mock_torch.cuda.get_device_name = lambda i: "Test GPU 8GB"
        mock_torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(
            total_memory=8 * 1024 * 1024 * 1024
        )
        mock_torch.cuda.memory_allocated = lambda i: 0
        mock_torch.cuda.memory_reserved = lambda i: 0
        mock_torch.version = types.SimpleNamespace(cuda="12.1")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            results = check_gpu()
            # Should warn about low memory
            assert any("12GB" in r.message and r.status == "warn" for r in results)


# ---------------------------------------------------------------------------
# check_checkpoint
# ---------------------------------------------------------------------------


class TestCheckCheckpoint:
    def test_missing_directory(self, tmp_path):
        results = check_checkpoint(str(tmp_path / "nonexistent"))
        assert any(r.status == "error" and "not found" in r.message for r in results)

    def test_empty_directory(self, tmp_path):
        results = check_checkpoint(str(tmp_path))
        assert any(
            "No adapter weights" in r.message and r.status == "error" for r in results
        )

    def test_valid_safetensors(self, tmp_path):
        """Checkpoint with safetensors and config should pass basic checks."""
        (tmp_path / "adapter_config.json").write_text('{"r": 16}')
        from safetensors.torch import save_file
        import torch

        save_file(
            {"lora_A.weight": torch.zeros(4, 4)},
            str(tmp_path / "adapter_model.safetensors"),
        )

        results = check_checkpoint(str(tmp_path))
        assert any(r.status == "ok" and "finite" in r.message.lower() for r in results)

    def test_nan_in_safetensors(self, tmp_path):
        """NaN in weights should produce an error."""
        (tmp_path / "adapter_config.json").write_text('{"r": 16}')
        from safetensors.torch import save_file
        import torch

        tensor_with_nan = torch.tensor([1.0, float("nan"), 3.0])
        save_file(
            {"lora_A.weight": tensor_with_nan},
            str(tmp_path / "adapter_model.safetensors"),
        )

        results = check_checkpoint(str(tmp_path))
        assert any(r.status == "error" and "NaN" in r.message for r in results)

    def test_with_training_state(self, tmp_path):
        """training_state.pt with finite tensors should pass."""
        (tmp_path / "adapter_config.json").write_text('{"r": 16}')
        from safetensors.torch import save_file
        import torch

        save_file({"w": torch.zeros(2, 2)}, str(tmp_path / "adapter_model.safetensors"))

        state = {"CycleState": torch.tensor([1.0, 2.0]), "Velocity": torch.zeros(3)}
        torch.save(state, str(tmp_path / "training_state.pt"))

        results = check_checkpoint(str(tmp_path))
        assert any(r.status == "ok" and "training_state" in r.message for r in results)
        assert any(r.status == "ok" and "finite" in r.message.lower() for r in results)


# ---------------------------------------------------------------------------
# check_config
# ---------------------------------------------------------------------------


class TestCheckConfig:
    def test_missing_file(self, tmp_path):
        results = check_config(str(tmp_path / "missing.yaml"))
        assert any(r.status == "error" and "not found" in r.message for r in results)

    def test_valid_config(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            tg_lora:
              K_initial: 3
              N_initial: 5
              alpha_initial: 0.3
              beta_initial: 0.8
              lr_initial: 0.0005
              relative_update_cap: 0.005
            training:
              grad_accumulation: 8
              gradient_checkpointing: true
            data:
              max_seq_len: 1024
        """)
        )
        results = check_config(str(cfg))
        # Filter out the Pydantic fallback warning (partial config).
        range_warnings = [
            r
            for r in results
            if r.status == "warn" and "validation failed" not in r.message.lower()
        ]
        errors = [r for r in results if r.status == "error"]
        assert len(errors) == 0, f"Unexpected errors: {[r.message for r in errors]}"
        assert len(range_warnings) == 0, (
            f"Unexpected warnings: {[r.message for r in range_warnings]}"
        )

    def test_out_of_range_values(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            tg_lora:
              K_initial: 100
              alpha_initial: 5.0
              lr_initial: 0.1
        """)
        )
        results = check_config(str(cfg))
        range_warnings = [
            r
            for r in results
            if r.status == "warn" and "validation failed" not in r.message.lower()
        ]
        assert any("K_initial" in r.message for r in range_warnings)
        assert any("alpha_initial" in r.message for r in range_warnings)
        assert any("lr_initial" in r.message for r in range_warnings)

    def test_oom_risk_seq_len(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("data:\n  max_seq_len: 4096\n")
        results = check_config(str(cfg))
        assert any("OOM" in r.message and r.status == "warn" for r in results)

    def test_gradient_checkpointing_disabled(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("training:\n  gradient_checkpointing: false\n")
        results = check_config(str(cfg))
        assert any(
            "gradient_checkpointing" in r.message and r.status == "warn"
            for r in results
        )


# ---------------------------------------------------------------------------
# check_logs
# ---------------------------------------------------------------------------


class TestCheckLogs:
    def test_missing_directory(self, tmp_path):
        results = check_logs(str(tmp_path / "missing"))
        assert any(r.status == "error" and "not found" in r.message for r in results)

    def test_empty_directory(self, tmp_path):
        results = check_logs(str(tmp_path))
        assert any(r.status == "warn" and "No log files" in r.message for r in results)

    def test_clean_log(self, tmp_path):
        (tmp_path / "train.log").write_text("All good\nLoss: 1.23\n")
        results = check_logs(str(tmp_path))
        assert any(
            r.status == "ok" and "No error patterns" in r.message for r in results
        )

    def test_oom_in_log(self, tmp_path):
        (tmp_path / "train.log").write_text("Step 50: torch.cuda.OutOfMemoryError\n")
        results = check_logs(str(tmp_path))
        assert any(r.status == "error" and "OOM" in r.message for r in results)

    def test_cuda_error_in_log(self, tmp_path):
        (tmp_path / "train.log").write_text(
            "CUDA error: device-side assert triggered\n"
        )
        results = check_logs(str(tmp_path))
        assert any(r.status == "error" and "CUDA error" in r.message for r in results)

    def test_nan_in_log(self, tmp_path):
        (tmp_path / "train.log").write_text("Step 100: loss is NaN\n")
        results = check_logs(str(tmp_path))
        assert any("Non-finite" in r.message or "NaN" in r.message for r in results)

    def test_multiple_errors(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text(
            textwrap.dedent("""\
            Step 50: torch.cuda.OutOfMemoryError
            Step 51: CUDA error: illegal memory access
            Step 52: loss is NaN
            NumericalInstabilityError at step 53
        """)
        )
        results = check_logs(str(tmp_path))
        errors = [r for r in results if r.status == "error"]
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    def test_returns_categorized_results(self):
        results = run_all_checks(gpu=True)
        assert "GPU" in results
        assert isinstance(results["GPU"], list)

    def test_no_args_runs_all(self):
        results = run_all_checks()
        assert "GPU" in results


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_creation(self):
        r = CheckResult("ok", "test")
        assert r.status == "ok"
        assert r.message == "test"
        assert r.details == {}

    def test_with_details(self):
        r = CheckResult("warn", "test", {"key": "value"})
        assert r.details["key"] == "value"
