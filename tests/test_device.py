"""Tests for src/utils/device.py — device-agnostic GPU utilities."""


import pytest
import torch

from src.utils.device import (
    OOM_EXIT_CODE,
    detect_device,
    fault_exit_code,
    gpu_device_count,
    gpu_device_name,
    gpu_empty_cache,
    gpu_info_dict,
    gpu_memory_allocated_mb,
    gpu_peak_memory_mb,
    gpu_reset_peak_stats,
    gpu_total_memory_mb,
    is_gpu_oom_error,
    resolve_compute_dtype,
)


# ── detect_device ──────────────────────────────────────────────────

class TestDetectDevice:
    def test_returns_valid_device(self):
        dev = detect_device()
        assert dev.type in ("cuda", "mps", "cpu")

    def test_caching(self):
        """Second call returns the same device object."""
        import src.utils.device as mod
        mod._CACHED_DEVICE = None
        d1 = detect_device()
        d2 = detect_device()
        assert d1 == d2
        mod._CACHED_DEVICE = None

    def test_priority_cuda_over_mps(self):
        """On this machine, detect_device picks the best available."""
        import src.utils.device as mod
        mod._CACHED_DEVICE = None
        dev = detect_device()
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if has_cuda:
            assert dev.type == "cuda"
        elif has_mps:
            assert dev.type == "mps"
        else:
            assert dev.type == "cpu"
        mod._CACHED_DEVICE = None


# ── gpu_memory_allocated_mb ────────────────────────────────────────

class TestGpuMemory:
    def test_none_for_none_device(self):
        assert gpu_memory_allocated_mb(None) is None

    def test_none_for_cpu(self):
        assert gpu_memory_allocated_mb("cpu") is None

    def test_returns_float_on_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        val = gpu_memory_allocated_mb("cuda:0")
        assert isinstance(val, float)
        assert val >= 0

    def test_returns_float_on_mps(self):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("No MPS")
        val = gpu_memory_allocated_mb("mps")
        assert isinstance(val, float)
        assert val >= 0


# ── gpu_device_name ────────────────────────────────────────────────

class TestGpuDeviceName:
    def test_cpu(self):
        assert gpu_device_name("cpu") == "CPU"

    def test_none(self):
        assert gpu_device_name(None) == "CPU"

    def test_mps(self):
        assert gpu_device_name("mps") == "Apple MPS"

    def test_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        name = gpu_device_name("cuda:0")
        assert isinstance(name, str)
        assert len(name) > 0


# ── gpu_device_count ───────────────────────────────────────────────

class TestGpuDeviceCount:
    def test_returns_int(self):
        count = gpu_device_count()
        assert isinstance(count, int)
        assert count >= 0


# ── gpu_total_memory_mb ────────────────────────────────────────────

class TestGpuTotalMemory:
    def test_none_for_mps(self):
        assert gpu_total_memory_mb("mps") is None

    def test_none_for_cpu(self):
        assert gpu_total_memory_mb("cpu") is None

    def test_none_for_none(self):
        assert gpu_total_memory_mb(None) is None

    def test_cuda_returns_float(self):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        val = gpu_total_memory_mb("cuda:0")
        assert isinstance(val, float)
        assert val > 0


# ── gpu_peak_memory_mb ─────────────────────────────────────────────

class TestGpuPeakMemory:
    def test_none_for_mps(self):
        assert gpu_peak_memory_mb("mps") is None

    def test_none_for_cpu(self):
        assert gpu_peak_memory_mb("cpu") is None

    def test_none_for_none(self):
        assert gpu_peak_memory_mb(None) is None


# ── gpu_reset_peak_stats / gpu_empty_cache ─────────────────────────

class TestNoOpFunctions:
    def test_reset_peak_no_error_mps(self):
        gpu_reset_peak_stats("mps")

    def test_reset_peak_no_error_cpu(self):
        gpu_reset_peak_stats("cpu")

    def test_reset_peak_no_error_none(self):
        gpu_reset_peak_stats(None)

    def test_empty_cache_no_error_mps(self):
        gpu_empty_cache("mps")

    def test_empty_cache_no_error_cpu(self):
        gpu_empty_cache("cpu")

    def test_empty_cache_no_error_none(self):
        gpu_empty_cache(None)


# ── gpu_info_dict ──────────────────────────────────────────────────

class TestGpuInfoDict:
    def test_has_expected_keys(self):
        info = gpu_info_dict()
        assert "name" in info
        assert "total_mb" in info
        assert "type" in info

    def test_type_matches_device(self):
        info = gpu_info_dict()
        dev = detect_device()
        assert info["type"] == dev.type


# ── is_gpu_oom_error ───────────────────────────────────────────────

class TestIsGpuOomError:
    def test_cuda_oom(self):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
        assert is_gpu_oom_error(exc) is True

    def test_runtime_oom_message(self):
        assert is_gpu_oom_error(RuntimeError("CUDA out of memory. Tried to allocate")) is True

    def test_mps_oom_message(self):
        assert is_gpu_oom_error(RuntimeError("MPS backend out of memory allocating")) is True

    def test_non_gpu_error(self):
        assert is_gpu_oom_error(RuntimeError("some other error")) is False

    def test_type_error(self):
        assert is_gpu_oom_error(TypeError("wrong type")) is False

    def test_string_not_exception(self):
        assert is_gpu_oom_error("out of memory") is False

    def test_accelerator_error_device_dispatch_oom(self):
        # torch 2.12+ can surface a CUDA OOM through torch.AcceleratorError — a
        # RuntimeError sibling that is NOT a torch.cuda.OutOfMemoryError. Both
        # production trainers (train_tg_lora / train_baseline_qlora) classify a
        # deferrable OOM via is_gpu_oom_error, and the freeze-ci-9b harness fixed
        # this exact sibling class (is_cuda_oom). If this predicate stops
        # recognizing the device-dispatch path, a contended-GPU OOM is
        # misclassified as a fatal cuda_error and the AGENTS.md exit-3 defer/retry
        # contract silently breaks — so pin it on the real exception type.
        ae = getattr(torch, "AcceleratorError", None)
        if ae is None:
            pytest.skip("torch.AcceleratorError absent (pre-2.12 torch)")
        assert is_gpu_oom_error(ae("CUDA out of memory. Tried to allocate 3725.29 GiB.")) is True


# ── resolve_compute_dtype ──────────────────────────────────────────

class TestResolveComputeDtype:
    def test_bf16_on_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        dt = resolve_compute_dtype("cuda:0", "bf16")
        assert dt == torch.bfloat16

    def test_bf16_downgrade_on_mps(self):
        dt = resolve_compute_dtype("mps", "bf16")
        assert dt == torch.float16

    def test_fp16_stays_on_mps(self):
        dt = resolve_compute_dtype("mps", "fp16")
        assert dt == torch.float16

    def test_fp32_stays_on_mps(self):
        dt = resolve_compute_dtype("mps", "fp32")
        assert dt == torch.float32

    def test_bfloat16_string(self):
        dt = resolve_compute_dtype("cpu", "bfloat16")
        assert dt == torch.bfloat16

    def test_float16_string(self):
        dt = resolve_compute_dtype("cpu", "float16")
        assert dt == torch.float16

    def test_default_is_bf16(self):
        dt = resolve_compute_dtype("cpu", "unknown")
        assert dt == torch.bfloat16


# ── fault_exit_code (producer half of the OOM-defer exit-code contract) ──


class TestFaultExitCode:
    """``fault_exit_code`` maps trainer fault reasons to process exit codes.

    The contract (see AGENTS.md "Process exit codes"): a deferrable GPU OOM is
    the ONLY fault that exits ``OOM_EXIT_CODE`` (3) — every other trainer fault
    exits 2, and no-fault exits 0. This is what lets a control plane read 3 as
    "defer and retry" without parsing logs.
    """

    def test_oom_exit_code_is_three(self):
        # Pinned literal: the value is the contract, not an implementation detail.
        assert OOM_EXIT_CODE == 3

    def test_oom_maps_to_defer_exit_code(self):
        assert fault_exit_code("oom") == OOM_EXIT_CODE

    @pytest.mark.parametrize("reason", ["numerical_instability", "cuda_error"])
    def test_non_oom_faults_exit_two(self, reason):
        # A real fault is NOT a deferral candidate — retrying unchanged reproduces it.
        assert fault_exit_code(reason) == 2
        assert fault_exit_code(reason) != OOM_EXIT_CODE

    def test_no_fault_exits_zero(self):
        assert fault_exit_code(None) == 0

    def test_oom_distinct_from_other_faults(self):
        # The whole point: OOM must not share an exit code with a real fault.
        assert fault_exit_code("oom") != fault_exit_code("numerical_instability")
        assert fault_exit_code("oom") != fault_exit_code("cuda_error")

