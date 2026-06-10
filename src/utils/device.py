"""Device-agnostic GPU utility: auto-detect CUDA / MPS / CPU and wrap torch.cuda calls."""

from __future__ import annotations

import os

import torch

_CACHED_DEVICE: torch.device | None = None


def detect_device() -> torch.device:
    """Return the best available device: cuda:0 > mps > cpu.

    Set TG_LORA_BACKEND=mlx to force CPU (MLX manages its own Metal backend).
    """
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE

    backend = os.environ.get("TG_LORA_BACKEND", "").lower()
    if backend == "mlx":
        _CACHED_DEVICE = torch.device("cpu")
        return _CACHED_DEVICE

    if torch.cuda.is_available():
        _CACHED_DEVICE = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        _CACHED_DEVICE = torch.device("mps")
    else:
        _CACHED_DEVICE = torch.device("cpu")

    return _CACHED_DEVICE


def gpu_memory_allocated_mb(device: torch.device | str | None = None) -> float | None:
    """Return GPU memory allocated in MB, or None if unavailable."""
    if device is None:
        return None
    dev = torch.device(device)
    if dev.type == "cuda":
        idx = dev.index or 0
        return round(torch.cuda.memory_allocated(idx) / 1024**2, 1)
    if dev.type == "mps":
        try:
            return round(torch.mps.current_allocated_memory() / 1024**2, 1)
        except AttributeError:
            return None
    return None


def gpu_memory_reserved_mb(device: torch.device | str | None = None) -> float | None:
    """Return GPU memory reserved in MB, or None if unavailable."""
    if device is None:
        return None
    dev = torch.device(device)
    if dev.type == "cuda":
        idx = dev.index or 0
        return round(torch.cuda.memory_reserved(idx) / 1024**2, 1)
    return None


def gpu_empty_cache(device: torch.device | str | None = None) -> None:
    """Empty GPU cache. No-op on CPU or if device is None."""
    if device is None:
        return
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        try:
            torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            pass


def gpu_device_name(device: torch.device | str | None = None) -> str:
    """Return a human-readable GPU name."""
    if device is None:
        return "CPU"
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev.index or 0)
    if dev.type == "mps":
        return "Apple MPS"
    return "CPU"


def gpu_device_count() -> int:
    """Return the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 1
    return 0


def gpu_total_memory_mb(device: torch.device | str | None = None) -> float | None:
    """Return total GPU memory in MB, or None if unavailable."""
    if device is None:
        return None
    dev = torch.device(device)
    if dev.type == "cuda":
        return round(torch.cuda.get_device_properties(dev.index or 0).total_memory / 1024**2, 1)
    return None


def gpu_peak_memory_mb(device: torch.device | str | None = None) -> float | None:
    """Return peak GPU memory allocated in MB, or None if unavailable."""
    if device is None:
        return None
    dev = torch.device(device)
    if dev.type == "cuda":
        idx = dev.index or 0
        return round(torch.cuda.max_memory_allocated(idx) / 1024**2, 1)
    return None


def gpu_reset_peak_stats(device: torch.device | str | None = None) -> None:
    """Reset peak memory stats. No-op on non-CUDA."""
    if device is None:
        return
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev.index)


def gpu_synchronize(device: torch.device | str | None = None) -> None:
    """Synchronize GPU. No-op on CPU."""
    if device is None:
        return
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev.index)
    elif dev.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass


def gpu_info_dict() -> dict[str, str | float | None]:
    """Return a dict with GPU name, total memory (MB), and device type."""
    dev = detect_device()
    return {
        "name": gpu_device_name(dev),
        "total_mb": gpu_total_memory_mb(dev),
        "type": dev.type,
    }


def is_gpu_oom_error(exc: BaseException) -> bool:
    """Check whether *exc* is a GPU out-of-memory error (CUDA or MPS)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg:
            return True
        if "mps" in msg and ("alloc" in msg or "memory" in msg):
            return True
    return False


def resolve_compute_dtype(
    device: torch.device | str | None = None,
    dtype_str: str = "bf16",
) -> torch.dtype:
    """Resolve compute dtype, downgrading bf16 to fp16 on MPS."""
    _map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = _map.get(dtype_str, torch.bfloat16)
    dev = torch.device(device) if device is not None else detect_device()
    if dev.type == "mps" and dtype == torch.bfloat16:
        return torch.float16
    return dtype
