import logging
import math
from collections import deque

import torch

logger = logging.getLogger("tg-lora")


class Velocity:
    def __init__(self, max_history: int = 100) -> None:
        if max_history <= 0:
            raise ValueError(f"max_history must be positive, got {max_history}")
        self._state: dict[str, torch.Tensor] | None = None
        self._magnitude_history: deque[float] = deque(maxlen=max_history)
        self._max_history = max_history

    @property
    def state(self) -> dict[str, torch.Tensor] | None:
        return self._state

    @property
    def magnitudes(self) -> list[float]:
        return list(self._magnitude_history)

    def update(
        self, delta: dict[str, torch.Tensor], beta: float
    ) -> dict[str, torch.Tensor]:
        if self._state is None:
            self._state = {k: v.clone() for k, v in delta.items()}
        else:
            for k in delta:
                if k not in self._state:
                    self._state[k] = delta[k].clone()
                    continue
                self._state[k].mul_(beta).add_(delta[k], alpha=(1.0 - beta))

        self._record_magnitude()
        return self._state

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

    def reset(self) -> None:
        self._state = None
        self._magnitude_history.clear()

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
            if not (math.isfinite(d_val) and math.isfinite(nv_val) and math.isfinite(nd_val)):
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
