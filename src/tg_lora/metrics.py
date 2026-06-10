import math
import re
import warnings

import torch


def cosine_similarity(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for k in a:
        if k not in b:
            continue
        va = a[k].float().flatten()
        vb = b[k].float().flatten()
        d = torch.dot(va, vb).item()
        na = torch.dot(va, va).item()
        nb = torch.dot(vb, vb).item()
        if not (math.isfinite(d) and math.isfinite(na) and math.isfinite(nb)):
            continue
        dot += d
        norm_a += na
        norm_b += nb
    denom = (norm_a**0.5) * (norm_b**0.5)
    if denom <= 1e-12 and (norm_a > 0 or norm_b > 0):
        warnings.warn(
            "cosine_similarity: near-zero denominator with non-zero norms — "
            "returning 0.0 (vectors may be orthogonal)",
            stacklevel=2,
        )
    return dot / denom if denom > 1e-12 else 0.0


def total_norm(state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for v in state.values():
        n = v.float().norm().item()
        if not math.isfinite(n):
            continue
        total += n**2
    return total**0.5


def per_layer_norms(state: dict[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, tensor in state.items():
        n = tensor.float().norm().item()
        if not math.isfinite(n):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        layer = m.group(1) if m else "other"
        key = f"layer_{layer}"
        result[key] = result.get(key, 0.0) + n**2
    return {k: v**0.5 for k, v in result.items()}
