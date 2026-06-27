import logging
import math
import types
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.func import jvp

from src.model.lora_utils import iter_lora_params
from src.tg_lora.activation_cache import (
    _get_decoder_layers,
    forward_from_hidden_states,
    forward_suffix_hidden_states,
)
from src.tg_lora.velocity import OrthonormalBasis

logger = logging.getLogger("tg-lora")
_FORWARD_AD_FALLBACK_WARNED = False


@dataclass
class ExtrapolationStats:
    """Diagnostics for how much the trust-region cap suppressed extrapolation."""

    num_tensors: int = 0
    capped_tensors: int = 0
    raw_update_norm: float = 0.0
    applied_update_norm: float = 0.0
    min_cap_ratio: float = 1.0
    mean_cap_ratio: float = 1.0

    @property
    def global_cap_ratio(self) -> float:
        if self.raw_update_norm <= 1e-12:
            return 1.0
        if math.isinf(self.raw_update_norm):
            return 0.0
        return self.applied_update_norm / self.raw_update_norm

    @property
    def capped_fraction(self) -> float:
        if self.num_tensors == 0:
            return 0.0
        return self.capped_tensors / self.num_tensors


@dataclass
class ZerothOrderDirectionStats:
    loss_plus_mu: float
    loss_plus_2mu: float
    g: float
    h: float
    t: float
    used_curvature: bool


@dataclass
class ZerothOrderStepStats:
    attempted: bool = False
    accepted: bool = False
    dim: int = 0
    residual_norm: float = 0.0
    mu: float = 0.0
    loss_initial: float = float("nan")
    loss_new: float = float("nan")
    forward_count: int = 0
    raw_step_norm: float = 0.0
    applied_step_norm: float = 0.0
    cap_ratio: float = 1.0
    capped: bool = False
    rollback_triggered: bool = False
    termination_reason: str = ""
    directions: list[ZerothOrderDirectionStats] = field(default_factory=list)


@dataclass
class AlphaLineStepStats:
    alpha_before: float
    alpha_after: float
    loss: float
    grad_alpha: float


@dataclass
class AlphaLineFirstOrderCache:
    """Cached suffix output and its alpha-direction JVP for one batch."""

    hidden_base: torch.Tensor
    hidden_jvp: torch.Tensor
    output_split_layer_idx: int
    jvp_method: str = "jvp"


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
            n_nan,
            n_inf,
            update.numel(),
        )
        return torch.zeros_like(update)

    update_norm = update.norm()
    ref_norm = ref.norm().clamp(min=eps)
    max_norm = max_ratio * ref_norm

    if update_norm > max_norm:
        update.mul_(max_norm / update_norm)
    return update


def alpha_line_reconstruct_output(
    module: torch.nn.Module,
    cached_h: torch.Tensor,
    direction: dict[str, torch.Tensor],
    alpha: float | torch.Tensor,
    *,
    base_out: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor:
    """Reconstruct LoRA output after an alpha-scaled direction update.

    For a LoRA layer ``B @ A``, updating both factors gives
    ``(B + alpha*dB) @ (A + alpha*dA)``.  The exact output delta therefore has
    linear and quadratic terms in ``alpha``.  When only one factor moves this
    reduces to the requested ``base_out + alpha * (V @ h)`` form.
    """
    lora_A = _raw_lora_parameter(module, "lora_A")
    lora_B = _raw_lora_parameter(module, "lora_B")
    if lora_A is None or lora_B is None:
        raise TypeError(
            "alpha_line_reconstruct_output currently supports modules with raw "
            "lora_A and lora_B Parameters"
        )

    if base_out is None:
        base_out = module(cached_h)
    scale = _resolve_lora_scaling(module) if scaling is None else float(scaling)
    alpha_t = _as_alpha_tensor(alpha, cached_h)

    dA = _lookup_direction_tensor(direction, "lora_A", lora_A)
    dB = _lookup_direction_tensor(direction, "lora_B", lora_B)
    if dA is None and dB is None:
        return base_out

    hidden = cached_h
    linear_delta = torch.zeros_like(base_out)
    if dA is not None:
        a_delta = torch.nn.functional.linear(hidden, dA)
        linear_delta = linear_delta + torch.nn.functional.linear(a_delta, lora_B)
    if dB is not None:
        a_base = torch.nn.functional.linear(hidden, lora_A)
        linear_delta = linear_delta + torch.nn.functional.linear(a_base, dB)

    output = base_out + alpha_t * scale * linear_delta
    if dA is not None and dB is not None:
        a_delta = torch.nn.functional.linear(hidden, dA)
        quadratic_delta = torch.nn.functional.linear(a_delta, dB)
        output = output + alpha_t * alpha_t * scale * quadratic_delta
    return output


def alpha_line_step(
    module: torch.nn.Module,
    cached_h: torch.Tensor,
    direction: dict[str, torch.Tensor],
    alpha: float,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    alpha_lr: float,
    base_out: torch.Tensor | None = None,
    scaling: float | None = None,
) -> tuple[float, AlphaLineStepStats]:
    """Update scalar alpha using autograd only through alpha."""
    if alpha_lr <= 0:
        raise ValueError(f"alpha_lr must be positive, got {alpha_lr}")
    alpha_t = torch.tensor(
        float(alpha),
        device=cached_h.device,
        dtype=cached_h.dtype,
        requires_grad=True,
    )
    with torch.no_grad():
        cached_base = module(cached_h) if base_out is None else base_out
    output = alpha_line_reconstruct_output(
        module,
        cached_h,
        direction,
        alpha_t,
        base_out=cached_base.detach(),
        scaling=scaling,
    )
    loss = loss_fn(output)
    grad = torch.autograd.grad(loss, alpha_t, retain_graph=False)[0]
    grad_value = float(grad.detach().item())
    if not math.isfinite(grad_value):
        logger.warning("Non-finite gradient in alpha_line_step: %.4e", grad_value)
        return float(alpha), AlphaLineStepStats(
            alpha_before=float(alpha),
            alpha_after=float(alpha),
            loss=float(loss.detach().item()),
            grad_alpha=grad_value,
        )
    alpha_after = float(alpha) - alpha_lr * grad_value
    return alpha_after, AlphaLineStepStats(
        alpha_before=float(alpha),
        alpha_after=alpha_after,
        loss=float(loss.detach().item()),
        grad_alpha=grad_value,
    )


@contextmanager
def alpha_line_lora_context(
    model: torch.nn.Module,
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float | torch.Tensor,
    *,
    active_names: set[str] | None = None,
):
    """Temporarily make PEFT LoRA modules evaluate ``base + alpha * direction``.

    This is the module-level exact bracketing path for cached suffix forwards.
    It avoids constructing a functional parameter mapping for the whole model
    while preserving the exact LoRA factor update, including the alpha^2 term
    that appears when both A and B factors move.
    """
    originals: list[tuple[torch.nn.Module, Callable]] = []
    for module_name, module in model.named_modules():
        if not _is_peft_lora_module(module):
            continue
        for adapter in getattr(module, "active_adapters", []):
            for factor_name, factor_modules in (
                ("lora_A", module.lora_A),
                ("lora_B", module.lora_B),
            ):
                if adapter not in factor_modules:
                    continue
                factor_module = factor_modules[adapter]
                key = f"{module_name}.{factor_name}.{adapter}.weight"
                if active_names is not None and key not in active_names:
                    continue
                if key not in base or key not in direction:
                    continue
                original_forward = factor_module.forward
                patched = _make_alpha_line_linear_forward(
                    key=key,
                    original_forward=original_forward,
                    base=base,
                    direction=direction,
                    alpha=alpha,
                )
                factor_module.forward = types.MethodType(patched, factor_module)
                originals.append((factor_module, original_forward))
    try:
        yield
    finally:
        for module, original_forward in originals:
            module.forward = original_forward


def alpha_line_loss_exact(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float | torch.Tensor,
    *,
    active_names: set[str] | None = None,
) -> torch.Tensor:
    """Evaluate cached-suffix alpha-line loss with exact PEFT LoRA bracketing."""
    _validate_cached_alpha_batch(batch)
    with alpha_line_lora_context(
        model,
        base,
        direction,
        alpha,
        active_names=active_names,
    ):
        return forward_from_hidden_states(
            model,
            batch["hidden_states"],
            batch["attention_mask"],
            batch["labels"],
            split_layer_idx=batch["split_layer_idx"],
            position_ids=batch.get("position_ids"),
        )


def compute_alpha_line_base_out_jvp(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    *,
    active_names: set[str] | None = None,
    finite_diff_eps: float = 1e-3,
) -> AlphaLineFirstOrderCache:
    """Cache suffix hidden states and their alpha-direction JVP for one batch."""
    _validate_cached_alpha_batch(batch)
    if finite_diff_eps <= 0:
        raise ValueError(f"finite_diff_eps must be positive, got {finite_diff_eps}")
    decoder_layers = _get_decoder_layers(model)
    output_split_layer_idx = len(decoder_layers)
    hidden_ref = batch["hidden_states"]
    alpha0 = torch.tensor(
        0.0,
        device=hidden_ref.device,
        dtype=torch.float32,
    )
    tangent = torch.ones_like(alpha0)

    was_training = model.training
    model.eval()

    def suffix_hidden(alpha: torch.Tensor) -> torch.Tensor:
        with alpha_line_lora_context(
            model,
            base,
            direction,
            alpha,
            active_names=active_names,
        ):
            return forward_suffix_hidden_states(
                model,
                batch["hidden_states"],
                batch["attention_mask"],
                split_layer_idx=batch["split_layer_idx"],
                position_ids=batch.get("position_ids"),
                decoder_layers=decoder_layers,
            )

    try:
        try:
            hidden_base, hidden_jvp = jvp(suffix_hidden, (alpha0,), (tangent,))
            jvp_method = "jvp"
        except (RuntimeError, NotImplementedError) as exc:
            if not _is_forward_ad_unsupported(exc):
                raise
            global _FORWARD_AD_FALLBACK_WARNED
            if not _FORWARD_AD_FALLBACK_WARNED:
                logger.warning(
                    "alpha-line JVP unsupported by this backend (%s); "
                    "falling back to finite-difference tangent",
                    exc.__class__.__name__,
                )
                _FORWARD_AD_FALLBACK_WARNED = True
            with torch.no_grad():
                hidden_base = suffix_hidden(alpha0)
                alpha_eps = torch.tensor(
                    finite_diff_eps,
                    device=alpha0.device,
                    dtype=alpha0.dtype,
                )
                hidden_eps = suffix_hidden(alpha_eps)
                hidden_jvp = (hidden_eps - hidden_base) / finite_diff_eps
            jvp_method = "finite_difference"
    finally:
        if was_training:
            model.train()

    return AlphaLineFirstOrderCache(
        hidden_base=hidden_base.detach(),
        hidden_jvp=hidden_jvp.detach(),
        output_split_layer_idx=output_split_layer_idx,
        jvp_method=jvp_method,
    )


def alpha_line_loss_cached_zeroth(
    model: torch.nn.Module,
    cache: AlphaLineFirstOrderCache,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate loss from cached suffix output at alpha=0."""
    _validate_cached_alpha_batch(batch)
    return forward_from_hidden_states(
        model,
        cache.hidden_base,
        batch["attention_mask"],
        batch["labels"],
        split_layer_idx=cache.output_split_layer_idx,
        position_ids=batch.get("position_ids"),
    )


def alpha_line_loss_cached_first_order(
    model: torch.nn.Module,
    cache: AlphaLineFirstOrderCache,
    batch: dict[str, torch.Tensor],
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate alpha-line loss with first-order corrected suffix output."""
    _validate_cached_alpha_batch(batch)
    alpha_t = _as_alpha_tensor(alpha, cache.hidden_base)
    hidden = cache.hidden_base + alpha_t * cache.hidden_jvp
    return forward_from_hidden_states(
        model,
        hidden,
        batch["attention_mask"],
        batch["labels"],
        split_layer_idx=cache.output_split_layer_idx,
        position_ids=batch.get("position_ids"),
    )


@torch.no_grad()
def subspace_zeroth_order_step(
    model: torch.nn.Module,
    basis: OrthonormalBasis,
    active_names: set[str],
    loss_closure: Callable[[], float],
    *,
    mu_ratio: float = 0.001,
    eps_curv: float = 1e-8,
    eta_fallback_ratio: float = 1e-2,
    max_step_ratio: float = 0.02,
    tolerance: float = 0.005,
    disable_curvature: bool = False,
    stop_on_positive_primary_g: bool = True,
    primary_g_stop_epsilon: float = 0.0,
) -> ZerothOrderStepStats:
    """Take one data-driven zeroth-order step in the velocity subspace.

    ``basis`` must already be globally orthonormal over active LoRA tensors.
    All perturbation losses are evaluated on the same caller-provided batch via
    ``loss_closure``.
    """
    _validate_zeroth_order_hparams(
        mu_ratio=mu_ratio,
        eps_curv=eps_curv,
        eta_fallback_ratio=eta_fallback_ratio,
        max_step_ratio=max_step_ratio,
        tolerance=tolerance,
        primary_g_stop_epsilon=primary_g_stop_epsilon,
    )
    stats = ZerothOrderStepStats(
        attempted=True,
        dim=basis.dim,
        residual_norm=basis.residual_norm,
    )
    if basis.dim <= 0 or not basis.vectors or not active_names:
        return stats

    params = _active_lora_params(model, active_names)
    if not params:
        stats.dim = 0
        return stats

    ref_norm = _param_dict_norm(params)
    mu = mu_ratio * max(ref_norm, 1e-12)
    stats.mu = mu
    eta_fallback = eta_fallback_ratio * max(ref_norm, 1e-12)

    loss_initial = float(loss_closure())
    stats.loss_initial = loss_initial
    stats.forward_count += 1
    if not math.isfinite(loss_initial):
        stats.rollback_triggered = True
        return stats

    step_coefficients: list[float] = []
    for direction_index, direction in enumerate(basis.vectors):
        _apply_scaled_direction(params, direction, mu)
        loss_plus_mu = float(loss_closure())
        stats.forward_count += 1
        _apply_scaled_direction(params, direction, -mu)

        _apply_scaled_direction(params, direction, 2.0 * mu)
        loss_plus_2mu = float(loss_closure())
        stats.forward_count += 1
        _apply_scaled_direction(params, direction, -2.0 * mu)

        g = (4.0 * loss_plus_mu - 3.0 * loss_initial - loss_plus_2mu) / (
            2.0 * mu
        )
        h = (loss_plus_2mu - 2.0 * loss_plus_mu + loss_initial) / (mu * mu)
        used_curvature = (
            not disable_curvature
            and math.isfinite(h)
            and h > eps_curv
            and math.isfinite(g)
        )
        if used_curvature:
            t = -g / h
        elif math.isfinite(g):
            t = -eta_fallback * g
        else:
            t = 0.0
        if not math.isfinite(t):
            t = 0.0
            used_curvature = False
        step_coefficients.append(t)
        stats.directions.append(
            ZerothOrderDirectionStats(
                loss_plus_mu=loss_plus_mu,
                loss_plus_2mu=loss_plus_2mu,
                g=g,
                h=h,
                t=t,
                used_curvature=used_curvature,
            )
        )
        if (
            direction_index == 0
            and stop_on_positive_primary_g
            and g >= -primary_g_stop_epsilon
        ):
            stats.termination_reason = "primary_g_non_descent"
            return stats

    raw_step = _compose_step(params, basis.vectors, step_coefficients)
    stats.raw_step_norm = _tensor_dict_norm(raw_step)
    applied_step, cap_ratio = _global_cap_step(
        raw_step,
        ref_norm=ref_norm,
        max_ratio=max_step_ratio,
    )
    stats.cap_ratio = cap_ratio
    stats.capped = cap_ratio < 1.0 - 1e-6
    stats.applied_step_norm = _tensor_dict_norm(applied_step)

    _apply_step(params, applied_step)
    loss_new = float(loss_closure())
    stats.forward_count += 1
    stats.loss_new = loss_new

    accepted = (
        math.isfinite(loss_new)
        and loss_new <= loss_initial + tolerance
        and _all_lora_params_finite(params)
    )
    stats.accepted = accepted
    if not accepted:
        _apply_step(params, applied_step, scale=-1.0)
        stats.rollback_triggered = True
        stats.termination_reason = "loss_degraded"
    else:
        stats.termination_reason = "accepted"
    return stats


@torch.no_grad()
def apply_extrapolation(
    model: torch.nn.Module,
    velocity: dict[str, torch.Tensor],
    active_names: set[str],
    n_steps: int,
    lr: float,
    relative_update_cap: float | None = 0.005,
    alpha_by_name: dict[str, float] | None = None,
    default_alpha: float | None = None,
) -> ExtrapolationStats:
    del alpha_by_name, default_alpha
    stats = ExtrapolationStats()
    if n_steps <= 0:
        return stats
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"lr must be positive and finite, got {lr}")
    if not velocity or not active_names:
        return stats
    raw_norm_sq = 0.0
    applied_norm_sq = 0.0
    ratio_sum = 0.0

    for name, p in iter_lora_params(model):
        if name not in active_names:
            continue

        if name not in velocity:
            continue
        v = velocity[name].to(device=p.device, dtype=p.dtype)
        raw_update = n_steps * lr * v
        raw_norm = raw_update.norm().item()
        applied = raw_update
        applied_norm = raw_norm
        ratio = 1.0

        if relative_update_cap is not None:
            applied = cap_update(raw_update, p.detach(), max_ratio=relative_update_cap)
            applied_norm = applied.norm().item()
            if math.isfinite(raw_norm) and raw_norm > 1e-12:
                ratio = applied_norm / raw_norm
            elif not math.isfinite(raw_norm):
                ratio = 0.0
            if ratio < 1.0 - 1e-6:
                stats.capped_tensors += 1

        p.add_(applied)

        stats.num_tensors += 1
        if math.isfinite(raw_norm):
            raw_norm_sq += raw_norm**2
        elif math.isinf(raw_norm):
            raw_norm_sq = float("inf")
        if math.isfinite(applied_norm):
            applied_norm_sq += applied_norm**2
        elif math.isinf(applied_norm):
            applied_norm_sq = float("inf")
        ratio_sum += ratio
        stats.min_cap_ratio = min(stats.min_cap_ratio, ratio)

    stats.raw_update_norm = raw_norm_sq**0.5
    stats.applied_update_norm = applied_norm_sq**0.5
    stats.mean_cap_ratio = ratio_sum / stats.num_tensors if stats.num_tensors else 1.0
    return stats


def _validate_zeroth_order_hparams(
    *,
    mu_ratio: float,
    eps_curv: float,
    eta_fallback_ratio: float,
    max_step_ratio: float,
    tolerance: float,
    primary_g_stop_epsilon: float,
) -> None:
    if mu_ratio <= 0:
        raise ValueError(f"mu_ratio must be positive, got {mu_ratio}")
    if eps_curv <= 0:
        raise ValueError(f"eps_curv must be positive, got {eps_curv}")
    if eta_fallback_ratio <= 0:
        raise ValueError(
            f"eta_fallback_ratio must be positive, got {eta_fallback_ratio}"
        )
    if max_step_ratio <= 0:
        raise ValueError(f"max_step_ratio must be positive, got {max_step_ratio}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if primary_g_stop_epsilon < 0:
        raise ValueError(
            "primary_g_stop_epsilon must be non-negative, "
            f"got {primary_g_stop_epsilon}"
        )


def _active_lora_params(
    model: torch.nn.Module,
    active_names: set[str],
) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in iter_lora_params(model) if name in active_names}


def _param_dict_norm(params: dict[str, torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for param in params.values():
        value = param.detach().float().norm().item()
        if math.isfinite(value):
            total_sq += value**2
    return total_sq**0.5


def _tensor_dict_norm(tensors: dict[str, torch.Tensor]) -> float:
    total_sq = 0.0
    for tensor in tensors.values():
        value = tensor.detach().float().norm().item()
        if math.isfinite(value):
            total_sq += value**2
        else:
            return float("inf")
    return total_sq**0.5


def _apply_scaled_direction(
    params: dict[str, torch.nn.Parameter],
    direction: dict[str, torch.Tensor],
    scale: float,
) -> None:
    for name, param in params.items():
        if name not in direction:
            continue
        update = direction[name].to(device=param.device, dtype=param.dtype)
        param.add_(update, alpha=scale)


def _compose_step(
    params: dict[str, torch.nn.Parameter],
    basis_vectors: list[dict[str, torch.Tensor]],
    coefficients: list[float],
) -> dict[str, torch.Tensor]:
    step: dict[str, torch.Tensor] = {}
    for direction, coefficient in zip(basis_vectors, coefficients, strict=False):
        if coefficient == 0.0:
            continue
        for name, param in params.items():
            if name not in direction:
                continue
            update = direction[name].to(device=param.device, dtype=param.dtype).mul(
                float(coefficient)
            )
            if name in step:
                step[name].add_(update)
            else:
                step[name] = update
    return step


def _global_cap_step(
    step: dict[str, torch.Tensor],
    *,
    ref_norm: float,
    max_ratio: float,
    eps: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], float]:
    if not step:
        return step, 1.0
    raw_norm = _tensor_dict_norm(step)
    if not math.isfinite(raw_norm):
        logger.warning("subspace zeroth-order step has non-finite norm; zeroing step")
        return {name: torch.zeros_like(tensor) for name, tensor in step.items()}, 0.0
    max_norm = max_ratio * max(ref_norm, eps)
    if raw_norm <= max_norm or raw_norm <= eps:
        return step, 1.0
    ratio = max_norm / raw_norm
    for tensor in step.values():
        tensor.mul_(ratio)
    return step, ratio


def _apply_step(
    params: dict[str, torch.nn.Parameter],
    step: dict[str, torch.Tensor],
    *,
    scale: float = 1.0,
) -> None:
    for name, update in step.items():
        if name in params:
            params[name].add_(update, alpha=scale)


def _all_lora_params_finite(params: dict[str, torch.nn.Parameter]) -> bool:
    return all(torch.isfinite(param).all().item() for param in params.values())


def _raw_lora_parameter(
    module: torch.nn.Module,
    attr_name: str,
) -> torch.nn.Parameter | None:
    value = getattr(module, attr_name, None)
    return value if isinstance(value, torch.nn.Parameter) else None


def _resolve_lora_scaling(module: torch.nn.Module) -> float:
    scaling = getattr(module, "scaling", 1.0)
    if isinstance(scaling, dict):
        if len(scaling) != 1:
            raise ValueError("Cannot infer LoRA scaling from a multi-adapter dict")
        return float(next(iter(scaling.values())))
    return float(scaling)


def _as_alpha_tensor(
    alpha: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(alpha, torch.Tensor):
        return alpha.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(float(alpha), device=reference.device, dtype=reference.dtype)


def _lookup_direction_tensor(
    direction: dict[str, torch.Tensor],
    suffix: str,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    for key, value in direction.items():
        if key == suffix or key.endswith(f".{suffix}"):
            return value.to(device=reference.device, dtype=reference.dtype)
    return None


def _is_peft_lora_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "base_layer")
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and isinstance(getattr(module, "lora_A"), torch.nn.ModuleDict)
        and isinstance(getattr(module, "lora_B"), torch.nn.ModuleDict)
    )


def _validate_cached_alpha_batch(batch: dict[str, torch.Tensor]) -> None:
    required = {"hidden_states", "attention_mask", "labels", "split_layer_idx"}
    missing = required - batch.keys()
    if missing:
        raise KeyError(
            f"cached alpha-line batch is missing keys: {sorted(missing)}"
        )


def _is_forward_ad_unsupported(exc: BaseException) -> bool:
    message = str(exc)
    markers = (
        "forward AD",
        "functorch transforms",
        "setup_context",
        "jvp",
        "does not support it because it has not been implemented",
    )
    return any(marker in message for marker in markers)


def _make_alpha_line_linear_forward(
    *,
    key: str,
    original_forward: Callable,
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float | torch.Tensor,
) -> Callable:
    def _forward(_self, x: torch.Tensor) -> torch.Tensor:
        if key not in base or key not in direction:
            return original_forward(x)
        weight = _effective_alpha_line_weight(
            key=key,
            fallback=_self.weight,
            base=base,
            direction=direction,
            alpha=alpha,
        )
        return F.linear(x, weight, _self.bias)

    return _forward


def _effective_alpha_line_weight(
    *,
    key: str,
    fallback: torch.Tensor,
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    if key not in base or key not in direction:
        return fallback
    base_t = base[key].to(device=fallback.device, dtype=fallback.dtype)
    direction_t = direction[key].to(device=fallback.device, dtype=fallback.dtype)
    alpha_t = _as_alpha_tensor(alpha, fallback)
    return base_t + alpha_t * direction_t


def flatten_tensor_dict(tensor_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    sorted_keys = sorted(tensor_dict.keys())
    return torch.cat([tensor_dict[k].flatten() for k in sorted_keys])


def unflatten_tensor_dict(flat_vector: torch.Tensor, template_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sorted_keys = sorted(template_dict.keys())
    restored = {}
    offset = 0
    for k in sorted_keys:
        shape = template_dict[k].shape
        numel = template_dict[k].numel()
        restored[k] = flat_vector[offset : offset + numel].view(shape).clone()
        offset += numel
    return restored


@torch.no_grad()
def subspace_m9_fit_step(
    model: torch.nn.Module,
    history: list[dict[str, torch.Tensor]],
    active_names: set[str],
    batch: dict[str, torch.Tensor],
    loss_fn: Callable[[dict[str, torch.Tensor]], float],
    *,
    selected_N: int = 1,
    fd_epsilon: float = 1e-3,
    fit_lr: float = 0.5,
    fit_steps: int = 1,
    velocity_direction: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Prior-based Subspace Learning (M9) using finite-difference gradient fitting.
    
    1. Reconstructs subspace (v0, pc1, pc2) from history.
    2. Gram-Schmidt orthogonalizes PC1 and PC2 with respect to v0.
    3. Fits coefficients (alpha, beta1, beta2) using K steps of finite-difference gradient descent.
    """
    stats = {}
    if not history or not active_names:
        return {}, stats

    # 1. Gather template parameters and states
    params = {name: p for name, p in model.named_parameters() if name in active_names}
    original_state = {name: p.data.clone().cpu() for name, p in params.items()}

    def restore_model():
        for name, p in params.items():
            p.data.copy_(original_state[name].to(p.device))

    # 2. Build flat delta trajectories
    flat_deltas = []
    num_active_params = sum(params[name].numel() for name in active_names)
    for idx, d in enumerate(history):
        d_filtered = {k: v for k, v in d.items() if k in active_names}
        flat_d = flatten_tensor_dict(d_filtered).cpu().double()
        
        # Dimension verification assert
        if flat_d.numel() != num_active_params:
            raise ValueError(
                f"[M9 Dimension Mismatch] History step {idx} has {flat_d.numel()} elements, "
                f"but model active parameters have {num_active_params} elements. "
                f"Check trainable_lora_scope configuration."
            )
            
        # Zero element anomaly detection assert
        zero_ratio = float((flat_d == 0).sum().item()) / flat_d.numel()
        if zero_ratio >= 0.5:
            raise ValueError(
                f"[M9 Anomalous Zero Elements] History step {idx} has {zero_ratio:.1%} zero elements "
                f"(threshold: 50%). Trainable scope mismatch or zero-padding is highly likely."
            )
            
        flat_deltas.append(flat_d)
    deltas_stack = torch.stack(flat_deltas)

    # Median norm for scale
    norms = deltas_stack.norm(dim=1)
    w_traj = float(norms.median().item())
    stats["w_traj"] = w_traj

    # Prior direction v0
    if velocity_direction is not None:
        # Use Velocity EMA direction instead of raw history mean.
        # Raw deltas are nearly orthogonal (pairwise cos ≈ 0.03),
        # so their mean is noise. Velocity EMA captures the true
        # global descent direction (short/long cos ≈ 0.8).
        v0_flat = flatten_tensor_dict(
            {k: v for k, v in velocity_direction.items() if k in active_names}
        ).cpu().double()
        v0_norm = v0_flat.norm().item()
        if v0_norm > 1e-8:
            v0 = v0_flat / v0_norm
        else:
            v0 = torch.zeros(v0_flat.shape, dtype=torch.float64)
        stats["v0_source"] = "velocity_ema"
    else:
        mean_delta = deltas_stack.mean(dim=0)
        mean_delta_norm = mean_delta.norm().item()
        if mean_delta_norm > 1e-8:
            v0 = mean_delta / mean_delta_norm
        else:
            v0 = torch.zeros_like(mean_delta)
        stats["v0_source"] = "history_mean"

    # PCA for pc1, pc2 (only relevant when beta != 0)
    _center = deltas_stack.mean(dim=0)
    centered_deltas = deltas_stack - _center
    q = min(4, len(history))
    if q >= 2:
        U, S_val, V = torch.pca_lowrank(centered_deltas, q=q, niter=4)
        pc1 = V[:, 0]
        pc2 = V[:, 1]
    elif q == 1:
        pc1 = centered_deltas[0]
        if pc1.norm() > 1e-8:
            pc1 = pc1 / pc1.norm()
        pc2 = torch.zeros_like(pc1)
    else:
        pc1 = torch.zeros_like(v0)
        pc2 = torch.zeros_like(v0)

    # Gram-Schmidt Orthogonalization (v0 -> pc1 -> pc2)
    def gram_schmidt_flat(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        ortho = []
        for v in vectors:
            v_ortho = v.clone()
            for u in ortho:
                proj = torch.dot(v_ortho, u) * u
                v_ortho -= proj
            norm = v_ortho.norm()
            if norm > 1e-8:
                ortho.append(v_ortho / norm)
            else:
                ortho.append(torch.zeros_like(v))
        return ortho

    ortho_basis = gram_schmidt_flat([v0, pc1, pc2])
    v0_ortho = ortho_basis[0]
    u1 = ortho_basis[1]
    u2 = ortho_basis[2]

    # Convert back to dicts
    template_dict = {k: v.clone().cpu() for k, v in original_state.items()}
    v0_dict = unflatten_tensor_dict(v0_ortho.float(), template_dict)
    u1_dict = unflatten_tensor_dict(u1.float(), template_dict)
    u2_dict = unflatten_tensor_dict(u2.float(), template_dict)

    # Helper function to compute loss
    def compute_loss_at(a: float, b1: float, b2: float) -> float:
        for name, p in params.items():
            delta_val = selected_N * a * w_traj * v0_dict[name].to(p.device) + b1 * u1_dict[name].to(p.device) + b2 * u2_dict[name].to(p.device)
            p.data.copy_(original_state[name].to(p.device) + delta_val)
        loss = loss_fn(batch)
        restore_model()
        return loss

    # Fixed coefficients: FD fitting bypassed for diagnostic run.
    # Motivation: alpha std=4.46 noise in previous run made fitting harmful.
    # This uses pure v0 extrapolation: W_extrap = W_t + N * w_traj * v0
    alpha, beta1, beta2 = 1.0, 0.0, 0.0
    initial_loss = compute_loss_at(alpha, beta1, beta2)
    stats["loss_initial"] = initial_loss
    stats["loss_final"] = initial_loss  # no fitting → same loss
    stats["alpha_fit"] = alpha
    stats["beta1_fit"] = beta1
    stats["beta2_fit"] = beta2

    # Compose final update delta dict (without applying, caller will apply or commit)
    final_delta_dict = {}
    for name in params.keys():
        final_delta_dict[name] = (
            selected_N * alpha * w_traj * v0_dict[name]
            + beta1 * u1_dict[name]
            + beta2 * u2_dict[name]
        )

    return final_delta_dict, stats
