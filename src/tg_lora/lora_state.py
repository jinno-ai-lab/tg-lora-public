import torch

from src.model.lora_utils import iter_lora_params


def snapshot_lora(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in iter_lora_params(model)}


def snapshot_lora_delta(
    model: torch.nn.Module, base: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Store only the difference from base snapshot (memory-efficient).

    For K intermediate snapshots, this saves ~(K-1)/K of memory compared
    to storing full snapshots, since deltas are typically sparse/small.
    """
    if not base:
        raise ValueError("base snapshot must not be empty")
    return {
        name: (p.detach().cpu() - base[name])
        for name, p in iter_lora_params(model)
        if name in base
    }


@torch.no_grad()
def apply_delta_snapshot(
    model: torch.nn.Module,
    base: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
) -> None:
    """Restore model to base + delta state."""
    for name, p in iter_lora_params(model):
        if name not in base or name not in delta:
            continue
        restored = base[name] + delta[name]
        p.copy_(restored.to(device=p.device, dtype=p.dtype))


@torch.no_grad()
def load_lora_snapshot(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    for name, p in iter_lora_params(model):
        if name not in state:
            continue
        saved = state[name]
        p.copy_(saved.to(device=p.device, dtype=p.dtype))


def diff_lora(
    after: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    if scale == 0.0:
        return {k: torch.zeros_like(after[k]) for k in before if k in after}
    if scale == 1.0:
        return {k: after[k] - before[k] for k in before if k in after}
    return {k: (after[k] - before[k]) * scale for k in before if k in after}
