"""Training cycle health monitor for TG-LoRA.

Detects divergence (loss spikes, NaN gradients) and stagnation (no improvement
over N cycles), and recommends interventions such as LR reduction, K increase,
or rollback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DivergenceReport:
    detected: bool = False
    metric: str = ""
    severity: str = ""
    current_value: float | None = None
    threshold: float | None = None


@dataclass
class StagnationReport:
    detected: bool = False
    cycles_without_improvement: int = 0
    best_value: float | None = None
    current_value: float | None = None


@dataclass
class HealthReport:
    status: str = "healthy"
    divergence: DivergenceReport = field(default_factory=DivergenceReport)
    stagnation: StagnationReport = field(default_factory=StagnationReport)
    recommendations: list[str] = field(default_factory=list)
    cycle_count: int = 0


class CycleMonitor:
    """Track training cycle health and detect anomalies."""

    def __init__(
        self,
        patience: int = 5,
        spike_threshold: float = 2.0,
        nan_detection: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if spike_threshold <= 0:
            raise ValueError("spike_threshold must be > 0")
        self.patience = patience
        self.spike_threshold = spike_threshold
        self.nan_detection = nan_detection

        self._history: list[dict[str, Any]] = []
        self._best_loss: float | None = None
        self._cycles_since_best: int = 0

    def update(self, cycle_data: dict[str, Any]) -> HealthReport:
        self._history.append(cycle_data)

        train_loss = cycle_data.get("train_loss")
        valid_loss = cycle_data.get("valid_loss")
        cycle_data.get("grad_norm")

        loss = valid_loss if valid_loss is not None else train_loss

        # Update best tracking
        if loss is not None and not math.isnan(loss) and not math.isinf(loss):
            if self._best_loss is None or loss < self._best_loss:
                self._best_loss = loss
                self._cycles_since_best = 0
            else:
                self._cycles_since_best += 1
        else:
            self._cycles_since_best += 1

        div = self.detect_divergence()
        stag = self.detect_stagnation()

        status = "healthy"
        if div.detected:
            status = "divergent"
        elif stag.detected:
            status = "stagnant"

        recs = self.recommend_intervention(div, stag)

        return HealthReport(
            status=status,
            divergence=div,
            stagnation=stag,
            recommendations=recs,
            cycle_count=len(self._history),
        )

    def detect_divergence(self) -> DivergenceReport:
        if len(self._history) < 2:
            return DivergenceReport()

        current = self._history[-1]
        previous = self._history[-2]

        train_loss = current.get("train_loss")
        valid_loss = current.get("valid_loss")
        prev_train = previous.get("train_loss")

        # NaN detection
        if self.nan_detection:
            for name, val in [("train_loss", train_loss), ("valid_loss", valid_loss)]:
                if val is not None and (math.isnan(val) or math.isinf(val)):
                    return DivergenceReport(
                        detected=True,
                        metric=name,
                        severity="critical",
                        current_value=val,
                        threshold=None,
                    )

        # Loss spike detection — skip when losses are near zero to avoid
        # false positives from ratio amplification of negligible absolute changes
        if train_loss is not None and prev_train is not None and prev_train > 1e-6:
            ratio = train_loss / prev_train
            if ratio >= self.spike_threshold:
                return DivergenceReport(
                    detected=True,
                    metric="train_loss",
                    severity="high",
                    current_value=train_loss,
                    threshold=prev_train * self.spike_threshold,
                )

        return DivergenceReport()

    def detect_stagnation(self) -> StagnationReport:
        if self._cycles_since_best >= self.patience and self._best_loss is not None:
            current_loss = None
            if self._history:
                last = self._history[-1]
                # Mirror the loss-selection rule in ``update`` (``valid_loss
                # if valid_loss is not None else train_loss``): a ``valid_loss``
                # of exactly ``0.0`` is a legitimate value (perfect/near-perfect
                # loss, e.g. the proxy memorize task), so ``... or train_loss``
                # is wrong — Python treats ``0.0`` as falsy and silently falls
                # through to ``train_loss``, surfacing the wrong loss in
                # ``current_value`` (consumed by ``health_summary`` and the
                # advisor reporting path).
                valid = last.get("valid_loss")
                current_loss = valid if valid is not None else last.get("train_loss")

            return StagnationReport(
                detected=True,
                cycles_without_improvement=self._cycles_since_best,
                best_value=self._best_loss,
                current_value=current_loss,
            )

        return StagnationReport()

    def recommend_intervention(
        self,
        divergence: DivergenceReport | None = None,
        stagnation: StagnationReport | None = None,
    ) -> list[str]:
        recommendations: list[str] = []

        if divergence is None:
            divergence = self.detect_divergence()
        if stagnation is None:
            stagnation = self.detect_stagnation()

        if divergence.detected:
            if divergence.severity == "critical":
                recommendations.append("rollback: NaN/Inf detected")
                recommendations.append("reduce_lr: critical divergence requires LR reduction")
            elif divergence.severity == "high":
                recommendations.append("reduce_lr: loss spike detected")
                recommendations.append("consider_rollback: if divergence persists")

        if stagnation.detected:
            recommendations.append("increase_K: stagnation suggests more extrapolation steps needed")
            if stagnation.cycles_without_improvement >= self.patience * 2:
                recommendations.append("rollback: prolonged stagnation")

        return recommendations

    def health_summary(self) -> dict[str, Any]:
        div = self.detect_divergence()
        stag = self.detect_stagnation()
        status = "healthy"
        if div.detected:
            status = "divergent"
        elif stag.detected:
            status = "stagnant"

        return {
            "status": status,
            "cycle_count": len(self._history),
            "best_loss": self._best_loss,
            "cycles_since_best": self._cycles_since_best,
            "divergence": {
                "detected": div.detected,
                "metric": div.metric,
                "severity": div.severity,
                "current_value": div.current_value,
                "threshold": div.threshold,
            },
            "stagnation": {
                "detected": stag.detected,
                "cycles_without_improvement": stag.cycles_without_improvement,
                "best_value": stag.best_value,
                "current_value": stag.current_value,
            },
            "recommendations": self.recommend_intervention(div, stag),
        }

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
