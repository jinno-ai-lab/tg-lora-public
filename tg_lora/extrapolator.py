import logging

import torch

from tg_lora.lora_utils import iter_lora_params

logger = logging.getLogger("tg-lora")


def cap_update(
    update: torch.Tensor,
    ref: torch.Tensor,
    max_ratio: float = 0.01,
    eps: float = 1e-8,
) -> torch.Tensor:
    if max_ratio <= 0:
        raise ValueError(f"max_ratio must be positive, got {max_ratio}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not torch.isfinite(update).all():
        n_nan = int(torch.isnan(update).sum().item())
        n_inf = int(torch.isinf(update).sum().item())
        logger.warning(
            "cap_update: non-finite update detected (%d NaN, %d Inf of %d elements) "
            "— zeroing entire update to prevent corruption",
            n_nan, n_inf, update.numel(),
        )
        return torch.zeros_like(update)

    update_norm = update.norm()
    ref_norm = ref.norm().clamp(min=eps)
    max_norm = max_ratio * ref_norm

    if update_norm > max_norm:
        update.mul_(max_norm / update_norm)
    return update


@torch.no_grad()
def apply_extrapolation(
    model: torch.nn.Module,
    velocity: dict[str, torch.Tensor],
    active_names: set[str],
    alpha_by_name: dict[str, float],
    default_alpha: float,
    n_steps: int,
    relative_update_cap: float = 0.005,
) -> None:
    if n_steps <= 0:
        return
    if not velocity or not active_names:
        return
    for name, p in iter_lora_params(model):
        if name not in active_names:
            continue

        if name not in velocity:
            continue
        v = velocity[name].to(device=p.device, dtype=p.dtype)
        alpha = alpha_by_name.get(name, default_alpha)
        raw_update = n_steps * alpha * v
        capped = cap_update(raw_update, p.detach(), max_ratio=relative_update_cap)
        p.add_(capped)
