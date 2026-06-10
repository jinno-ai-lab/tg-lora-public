import torch

from src.utils.device import detect_device, gpu_memory_allocated_mb, gpu_memory_reserved_mb


def vram_usage_mb() -> dict[str, float]:
    result = {}
    device = detect_device()
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            result[f"gpu{i}_allocated_mb"] = round(allocated, 1)
            result[f"gpu{i}_reserved_mb"] = round(reserved, 1)
    elif device.type == "mps":
        alloc = gpu_memory_allocated_mb(device)
        if alloc is not None:
            result["gpu0_allocated_mb"] = alloc
        reserved = gpu_memory_reserved_mb(device)
        if reserved is not None:
            result["gpu0_reserved_mb"] = reserved
    return result


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
