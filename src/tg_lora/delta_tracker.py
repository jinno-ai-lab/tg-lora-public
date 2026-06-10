from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from src.tg_lora.lora_state import diff_lora

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def compute_mean_delta(
    after: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
    K: int,
) -> dict[str, torch.Tensor]:
    """Return the raw cycle delta ``W_K - W_0``.

    ``K`` is kept in the signature for compatibility and validation, but the
    step-count normalization belongs in ``Velocity.update()`` where the pilot
    learning rate is also available.  This keeps the extrapolation invariant to
    cycle-to-cycle lr changes by normalizing with ``lr * K`` in one place.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    return diff_lora(after, before)


@dataclass
class DeltaStats:
    """Per-cycle delta statistics."""

    total_norm: float
    per_layer_norm: dict[str, float]
    max_component: float
    mean_abs: float


@torch.no_grad()
def _compute_stats(delta: dict[str, torch.Tensor]) -> DeltaStats:
    total_sq = 0.0
    per_layer: dict[str, float] = {}
    abs_vals: list[float] = []
    max_comp = 0.0

    for name, t in delta.items():
        f = t.float()
        norm_val = f.norm().item()
        if not math.isfinite(norm_val):
            continue
        norm_sq = norm_val**2
        total_sq += norm_sq

        m = _LAYER_RE.search(name)
        layer_key = f"layer_{m.group(1)}" if m else "other"
        per_layer[layer_key] = per_layer.get(layer_key, 0.0) + norm_sq

        flat = f.flatten()
        abs_vals.append(flat.abs().mean().item())
        max_comp = max(max_comp, flat.abs().max().item())

    return DeltaStats(
        total_norm=total_sq**0.5,
        per_layer_norm={k: v**0.5 for k, v in per_layer.items()},
        max_component=max_comp,
        mean_abs=sum(abs_vals) / len(abs_vals) if abs_vals else 0.0,
    )


class DeltaTracker:
    """Tracks delta statistics across TG-LoRA training cycles.

    Wraps ``compute_mean_delta`` and records per-layer norms, anomaly
    detection, and a convergence trend derived from recent history.
    """

    def __init__(self, max_history: int = 100) -> None:
        if max_history <= 0:
            raise ValueError(f"max_history must be positive, got {max_history}")
        self._history: list[dict[str, torch.Tensor]] = []
        self._norm_history: list[float] = []
        self._max_history = max_history
        self._last_stats: DeltaStats | None = None

    @property
    def last_stats(self) -> DeltaStats | None:
        return self._last_stats

    @property
    def norm_history(self) -> list[float]:
        return list(self._norm_history)

    def compute_and_record(
        self,
        after: dict[str, torch.Tensor],
        before: dict[str, torch.Tensor],
        K: int,
    ) -> dict[str, torch.Tensor]:
        after_keys, before_keys = set(after.keys()), set(before.keys())
        if after_keys != before_keys:
            missing_in_before = after_keys - before_keys
            missing_in_after = before_keys - after_keys
            parts: list[str] = []
            if missing_in_before:
                parts.append(f"missing in before: {sorted(missing_in_before)}")
            if missing_in_after:
                parts.append(f"missing in after: {sorted(missing_in_after)}")
            raise ValueError(
                f"Key mismatch between after/before dicts: {'; '.join(parts)}"
            )
        delta = compute_mean_delta(after, before, K)
        self._history.append(delta)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._last_stats = _compute_stats(delta)
        norm = self._last_stats.total_norm
        if math.isfinite(norm):
            self._norm_history.append(norm)
        if len(self._norm_history) > self._max_history:
            self._norm_history.pop(0)
        return delta

    def is_anomalous(self, threshold_sigma: float = 3.0) -> bool:
        """Return True when the latest delta norm is an outlier.

        A delta is anomalous when there are at least 3 history entries and
        the latest norm exceeds ``mean + threshold_sigma * std`` of all
        recorded norms.
        """
        if len(self._norm_history) < 3 or self._last_stats is None:
            return False
        norms = self._norm_history[:-1]
        mean = sum(norms) / len(norms)
        var = sum((n - mean) ** 2 for n in norms) / len(norms)
        std = var**0.5
        if std < 1e-12:
            return self._last_stats.total_norm > mean * 2.0
        return self._last_stats.total_norm > mean + threshold_sigma * std

    def convergence_trend(self, window: int = 5) -> float:
        """Slope of delta norms over the last *window* entries.

        Negative means deltas are shrinking (converging). Returns 0.0 when
        there are fewer than 2 entries in the window.
        """
        n = min(window, len(self._norm_history))
        if n < 2:
            return 0.0
        recent = self._norm_history[-n:]
        mean_x = (n - 1) / 2.0
        mean_y = sum(recent) / n
        cov = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(recent))
        var_x = sum((i - mean_x) ** 2 for i in range(n))
        return cov / var_x if var_x > 0 else 0.0

    def summary(self) -> dict:
        stats = self._last_stats
        return {
            "total_norm": stats.total_norm if stats else 0.0,
            "max_component": stats.max_component if stats else 0.0,
            "mean_abs": stats.mean_abs if stats else 0.0,
            "anomalous": self.is_anomalous(),
            "convergence_trend": self.convergence_trend(),
            "history_length": len(self._norm_history),
        }
