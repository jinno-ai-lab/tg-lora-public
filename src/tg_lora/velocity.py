import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger("tg-lora")


@dataclass
class OrthonormalBasis:
    vectors: list[dict[str, torch.Tensor]]
    dim: int
    residual_norm: float
    short_norm: float
    long_norm: float
    tau_dim: float


class Velocity:
    def __init__(
        self,
        max_history: int = 100,
        beta_short: float | None = None,
        beta_long: float | None = None,
    ) -> None:
        if max_history <= 0:
            raise ValueError(f"max_history must be positive, got {max_history}")
        self.beta_short = (
            beta_from_window(3) if beta_short is None else _validate_beta(beta_short)
        )
        self.beta_long = (
            beta_from_window(10) if beta_long is None else _validate_beta(beta_long)
        )
        self._state: dict[str, torch.Tensor] | None = None
        self._short_state: dict[str, torch.Tensor] | None = None
        self._long_state: dict[str, torch.Tensor] | None = None
        self._fixed_direction: dict[str, torch.Tensor] | None = None
        self._fixed_since_cycle: int | None = None
        self._magnitude_history: list[float] = []
        self._max_history = max_history
        self._update_count = 0

    @property
    def state(self) -> dict[str, torch.Tensor] | None:
        return self._state

    @property
    def short_state(self) -> dict[str, torch.Tensor] | None:
        return self._short_state

    @property
    def long_state(self) -> dict[str, torch.Tensor] | None:
        return self._long_state

    @property
    def fixed_since_cycle(self) -> int | None:
        return self._fixed_since_cycle

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def magnitudes(self) -> list[float]:
        return list(self._magnitude_history)

    def update(
        self,
        delta: dict[str, torch.Tensor],
        beta: float,
        lr: float = 1.0,
        K: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Update the velocity EMA with lr-normalized pilot deltas.

        ``delta`` is the raw cycle displacement ``W_K - W_0``.  Dividing by
        ``lr * K`` removes the cycle learning-rate and step-count scale, so the
        EMA tracks the update direction that AdamW would apply per unit lr.
        Extrapolation later multiplies this state by the current lr again.
        """
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"lr must be positive and finite, got {lr}")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        beta = _validate_beta(beta)
        scale = 1.0 / (lr * K)
        self._state = _ema_update_scaled(self._state, delta, beta, scale)
        self._short_state = _ema_update_scaled(
            self._short_state,
            delta,
            self.beta_short,
            scale,
        )
        self._long_state = _ema_update_scaled(
            self._long_state,
            delta,
            self.beta_long,
            scale,
        )
        self._update_count += 1

        self._record_magnitude()
        return self._state

    def build_direction(
        self,
        delta: dict[str, torch.Tensor],
        *,
        lr: float,
        K: int,
        cycle: int | None = None,
        active_names: set[str] | None = None,
        eps: float = 1e-12,
    ) -> dict[str, torch.Tensor]:
        """Build and freeze a global unit direction for alpha-line learning.

        ``delta`` follows the same convention as ``update``: raw displacement
        ``W_K - W_0``.  The stored direction is normalized by ``lr * K`` and
        then by its global L2 norm, so alpha has a stable scalar meaning across
        cycles with different pilot learning rates.
        """
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"lr must be positive and finite, got {lr}")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        filtered = _filter_state(delta, active_names)
        scaled = _scale_state(filtered, 1.0 / (lr * K))
        norm = _global_norm(scaled)
        if norm <= eps:
            self._fixed_direction = {}
        else:
            self._fixed_direction = _scale_state(scaled, 1.0 / norm)
        self._fixed_since_cycle = cycle
        return self.current_direction() or {}

    def current_direction(self) -> dict[str, torch.Tensor] | None:
        if self._fixed_direction is None:
            return None
        return {k: v.detach().clone() for k, v in self._fixed_direction.items()}

    def _record_magnitude(self) -> None:
        if self._state is None:
            return
        total_sq = 0.0
        for t in self._state.values():
            n = t.float().norm().item()
            if not math.isfinite(n):
                continue
            total_sq += n**2
        mag = total_sq**0.5
        if not math.isfinite(mag):
            return
        self._magnitude_history.append(mag)
        if len(self._magnitude_history) > self._max_history:
            self._magnitude_history.pop(0)

    def reset(self) -> None:
        self._state = None
        self._short_state = None
        self._long_state = None
        self._fixed_direction = None
        self._fixed_since_cycle = None
        self._magnitude_history.clear()
        self._update_count = 0

    def predicted_consistency(self) -> float:
        """Cosine between short- and long-horizon normalized update EMAs.

        The first update makes both EMAs collinear by construction, so it is not
        evidence of a stable direction.  Return 0.0 until at least two updates
        have contributed.
        """
        if self._update_count < 2:
            return 0.0
        if self._short_state is None or self._long_state is None:
            return 0.0
        return _global_cosine(self._short_state, self._long_state)

    def short_long_norm_ratio(self) -> float:
        if self._short_state is None or self._long_state is None:
            return 1.0
        long_norm = _global_norm(self._long_state)
        if long_norm <= 1e-12:
            return 1.0
        return _global_norm(self._short_state) / long_norm

    def build_orthonormal_basis(
        self,
        active_names: set[str] | None = None,
        tau_dim: float = 0.15,
        force_dim: int = 0,
        eps: float = 1e-12,
    ) -> OrthonormalBasis:
        """Build a global orthonormal basis from short/long velocity EMAs.

        The basis is defined over the flattened concatenation of active LoRA
        tensors.  ``force_dim=0`` uses the residual-norm rule, while 1 or 2 can
        be used for ablations.
        """
        if not (0.0 <= tau_dim <= 1.0):
            raise ValueError(f"tau_dim must be in [0, 1], got {tau_dim}")
        if force_dim not in (0, 1, 2):
            raise ValueError(f"force_dim must be 0, 1, or 2, got {force_dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if self._short_state is None or self._long_state is None:
            return OrthonormalBasis([], 0, 0.0, 0.0, 0.0, tau_dim)

        short = _filter_state(self._short_state, active_names)
        long = _filter_state(self._long_state, active_names)
        short_norm = _global_norm(short)
        long_norm = _global_norm(long)
        if short_norm <= eps or long_norm <= eps:
            return OrthonormalBasis([], 0, 0.0, short_norm, long_norm, tau_dim)

        e1 = _scale_state(short, 1.0 / short_norm)
        proj = _global_dot(long, e1)
        r2 = _subtract_projection(long, e1, proj)
        r2_norm = _global_norm(r2)
        residual_norm = r2_norm / long_norm if long_norm > eps else 0.0

        use_dim2 = residual_norm >= tau_dim
        if force_dim == 1:
            use_dim2 = False
        elif force_dim == 2:
            use_dim2 = True

        if use_dim2 and r2_norm > eps:
            e2 = _scale_state(r2, 1.0 / r2_norm)
            return OrthonormalBasis([e1, e2], 2, residual_norm, short_norm, long_norm, tau_dim)
        return OrthonormalBasis([e1], 1, residual_norm, short_norm, long_norm, tau_dim)

    def choose_N(
        self,
        N_candidates: list[int],
        c_threshold_map: dict[int, float],
    ) -> int:
        """Choose the largest candidate whose consistency threshold is met.
        
        Returns 0 if the consistency does not meet the minimum required threshold.
        """
        if not N_candidates:
            raise ValueError("N_candidates must not be empty")
        if any(n <= 0 for n in N_candidates):
            raise ValueError("N_candidates must be positive")

        consistency = self.predicted_consistency()
        candidates = sorted(set(int(n) for n in N_candidates), reverse=True)
        thresholds = {int(n): float(c) for n, c in c_threshold_map.items()}
        for n_steps in candidates:
            c_min = thresholds.get(n_steps)
            if c_min is not None and consistency >= c_min:
                return n_steps
                
        # If consistency is below the absolute minimum threshold, return 0 (no extrapolation)
        min_threshold = min(thresholds.values()) if thresholds else 0.70
        if consistency < min_threshold:
            return 0
            
        return min(candidates)

    def is_magnitude_anomalous(self, threshold_sigma: float = 3.0) -> bool:
        if len(self._magnitude_history) < 3:
            return False
        norms = self._magnitude_history[:-1]
        mean = sum(norms) / len(norms)
        var = sum((n - mean) ** 2 for n in norms) / len(norms)
        std = var**0.5
        latest = self._magnitude_history[-1]
        if std < 1e-12:
            return latest > mean * 2.0
        return latest > mean + threshold_sigma * std

    def magnitude_trend(self, window: int = 5) -> float:
        n = min(window, len(self._magnitude_history))
        if n < 2:
            return 0.0
        recent = self._magnitude_history[-n:]
        mean_x = (n - 1) / 2.0
        mean_y = sum(recent) / n
        cov = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(recent))
        var_x = sum((i - mean_x) ** 2 for i in range(n))
        return cov / var_x if var_x > 0 else 0.0

    def magnitude_acceleration(self, window: int = 5) -> float:
        """Second derivative of velocity magnitude over the last *window* entries.

        Positive acceleration means magnitudes are growing faster over time
        (potential instability). Negative means growth is slowing (convergence).
        Returns 0.0 when fewer than 3 entries are in the window.
        """
        n = min(window, len(self._magnitude_history))
        if n < 3:
            return 0.0
        recent = self._magnitude_history[-n:]
        slopes: list[float] = []
        for i in range(1, len(recent)):
            slopes.append(recent[i] - recent[i - 1])
        if len(slopes) < 2:
            return 0.0
        acc_sum = sum(slopes[i] - slopes[i - 1] for i in range(1, len(slopes)))
        return acc_sum / (len(slopes) - 1)

    def cosine_similarity(self, delta: dict[str, torch.Tensor]) -> float:
        if self._state is None:
            return 0.0

        dot = 0.0
        norm_v = 0.0
        norm_d = 0.0
        overlap = 0
        for k in delta:
            if k not in self._state:
                continue
            overlap += 1
            v = self._state[k].float().flatten()
            d = delta[k].float().flatten()
            d_val = torch.dot(v, d).item()
            nv_val = torch.dot(v, v).item()
            nd_val = torch.dot(d, d).item()
            if not (
                math.isfinite(d_val) and math.isfinite(nv_val) and math.isfinite(nd_val)
            ):
                continue
            dot += d_val
            norm_v += nv_val
            norm_d += nd_val

        if overlap == 0:
            logger.warning(
                "cosine_similarity: no overlapping keys between velocity "
                "state and delta — returning 0.0"
            )
            return 0.0

        denom = (norm_v**0.5) * (norm_d**0.5)
        if denom <= 1e-12 and (norm_v > 0 or norm_d > 0):
            logger.warning(
                "cosine_similarity: near-zero denominator %.2e with "
                "non-zero norms (v=%.2e, d=%.2e) — returning 0.0 "
                "(vectors may be orthogonal)",
                denom,
                norm_v,
                norm_d,
            )
            return 0.0
        return dot / denom if denom > 1e-12 else 0.0


def beta_from_window(window: int) -> float:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if window == 1:
        return 0.0
    return 1.0 - (1.0 / window)


def _validate_beta(beta: float) -> float:
    if not math.isfinite(beta) or not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be finite and in [0, 1), got {beta}")
    return float(beta)


def _ema_update_scaled(
    state: dict[str, torch.Tensor] | None,
    delta: dict[str, torch.Tensor],
    beta: float,
    scale: float,
) -> dict[str, torch.Tensor]:
    if state is None:
        return {k: v.clone().mul_(scale) for k, v in delta.items()}
    for key, tensor in delta.items():
        if key not in state:
            state[key] = tensor.clone().mul_(scale)
            continue
        state[key].mul_(beta).add_(tensor, alpha=(1.0 - beta) * scale)
    return state


def _global_cosine(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for key in left.keys() & right.keys():
        a = left[key].float().flatten()
        b = right[key].float().flatten()
        dot_val = torch.dot(a, b).item()
        left_val = torch.dot(a, a).item()
        right_val = torch.dot(b, b).item()
        if not (
            math.isfinite(dot_val)
            and math.isfinite(left_val)
            and math.isfinite(right_val)
        ):
            continue
        dot += dot_val
        left_sq += left_val
        right_sq += right_val
    denom = (left_sq**0.5) * (right_sq**0.5)
    return dot / denom if denom > 1e-12 else 0.0


def _global_dot(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    dot = 0.0
    for key in left.keys() & right.keys():
        a = left[key].float().flatten()
        b = right[key].float().flatten()
        value = torch.dot(a, b).item()
        if math.isfinite(value):
            dot += value
    return dot


def _global_norm(tensors: dict[str, torch.Tensor]) -> float:
    total_sq = 0.0
    for tensor in tensors.values():
        value = tensor.float().norm().item()
        if math.isfinite(value):
            total_sq += value**2
    return total_sq**0.5


def _filter_state(
    state: dict[str, torch.Tensor],
    active_names: set[str] | None,
) -> dict[str, torch.Tensor]:
    if active_names is None:
        return {k: v for k, v in state.items()}
    return {k: v for k, v in state.items() if k in active_names}


def _scale_state(
    state: dict[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().mul_(scale) for k, v in state.items()}


def _subtract_projection(
    state: dict[str, torch.Tensor],
    unit_direction: dict[str, torch.Tensor],
    coefficient: float,
) -> dict[str, torch.Tensor]:
    residual: dict[str, torch.Tensor] = {}
    for key in state.keys() | unit_direction.keys():
        if key in state:
            value = state[key].detach().clone()
        else:
            value = torch.zeros_like(unit_direction[key])
        if key in unit_direction:
            direction = unit_direction[key].to(device=value.device, dtype=value.dtype)
            value.add_(direction, alpha=-coefficient)
        residual[key] = value
    return residual
