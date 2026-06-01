"""Training trajectory analysis for TG-LoRA.

Predicts convergence, estimates remaining steps, and provides early-stop
recommendations from loss history and velocity trends.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryPoint:
    cycle: int
    train_loss: float
    valid_loss: float | None = None
    grad_norm: float | None = None
    velocity_magnitude: float | None = None


@dataclass
class ConvergenceEstimate:
    converged: bool
    remaining_steps: int | None = None
    predicted_final_loss: float | None = None
    convergence_rate: float = 0.0
    confidence: float = 0.0


@dataclass
class EarlyStopAdvice:
    should_stop: bool
    reason: str = ""
    estimated_gain_from_continuing: float = 0.0
    optimal_cycle: int | None = None


@dataclass
class TrajectoryReport:
    total_points: int
    convergence: ConvergenceEstimate
    early_stop: EarlyStopAdvice
    loss_trend: float = 0.0
    volatility: float = 0.0
    anomaly_detected: bool = False
    anomaly_details: list[str] = field(default_factory=list)


class TrajectoryAnalyzer:
    """Analyze training loss trajectory and predict convergence."""

    def __init__(
        self,
        window: int = 5,
        convergence_threshold: float = 1e-4,
        min_points: int = 3,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold must be > 0, got {convergence_threshold}")
        if min_points < 2:
            raise ValueError(f"min_points must be >= 2, got {min_points}")
        self.window = window
        self.convergence_threshold = convergence_threshold
        self.min_points = min_points
        self._points: list[TrajectoryPoint] = []

    def add_point(self, point: TrajectoryPoint) -> None:
        if point.cycle < 0:
            raise ValueError(f"cycle must be non-negative, got {point.cycle}")
        if math.isnan(point.train_loss) or math.isinf(point.train_loss):
            raise ValueError(f"train_loss must be finite, got {point.train_loss}")
        self._points.append(point)

    def add_points(self, points: list[TrajectoryPoint]) -> None:
        for p in points:
            self.add_point(p)

    @property
    def points(self) -> list[TrajectoryPoint]:
        return list(self._points)

    def _losses(self) -> list[float]:
        return [
            p.valid_loss if p.valid_loss is not None else p.train_loss
            for p in self._points
        ]

    def compute_loss_trend(self) -> float:
        losses = self._losses()
        n = min(self.window, len(losses))
        if n < 2:
            return 0.0
        recent = losses[-n:]
        return _linear_slope(recent)

    def compute_volatility(self) -> float:
        losses = self._losses()
        if len(losses) < 2:
            return 0.0
        diffs = [abs(losses[i] - losses[i - 1]) for i in range(1, len(losses))]
        return sum(diffs) / len(diffs)

    def compute_convergence_rate(self) -> float:
        losses = self._losses()
        if len(losses) < 2:
            return 0.0
        n = min(self.window, len(losses))
        recent = losses[-n:]
        if len(recent) < 2:
            return 0.0
        initial = recent[0]
        if initial == 0:
            return 0.0
        final = recent[-1]
        steps = len(recent) - 1
        return (initial - final) / (initial * steps)

    def predict_steps_to_convergence(self, target_loss: float | None = None) -> int | None:
        losses = self._losses()
        if len(losses) < self.min_points:
            return None

        if target_loss is None:
            target_loss = self._estimate_asymptote()
        if target_loss is None:
            return None

        current = losses[-1]
        if current <= target_loss:
            return 0

        rate = abs(self.compute_convergence_rate())
        if rate < 1e-12:
            return None

        gap = current - target_loss
        steps = int(math.ceil(gap / (rate * current)))
        return max(steps, 1)

    def predict_loss_at_step(self, future_steps: int) -> float | None:
        if future_steps < 0:
            raise ValueError(f"future_steps must be non-negative, got {future_steps}")
        losses = self._losses()
        if len(losses) < 2:
            return None

        rate = self.compute_convergence_rate()
        current = losses[-1]
        predicted = current * (1 - rate) ** future_steps

        asymptote = self._estimate_asymptote()
        if asymptote is not None and predicted < asymptote:
            predicted = asymptote

        return predicted

    def _estimate_asymptote(self) -> float | None:
        losses = self._losses()
        if len(losses) < self.min_points:
            return None

        n = min(self.window * 2, len(losses))
        recent = losses[-n:]
        if len(recent) < 3:
            return None

        last_val = recent[-1]
        rate = self.compute_convergence_rate()
        if rate <= 1e-12:
            return last_val

        asymptote = last_val / (1 + rate)
        return max(asymptote, 0.0)

    def estimate_convergence(self, target_loss: float | None = None) -> ConvergenceEstimate:
        losses = self._losses()
        if len(losses) < self.min_points:
            return ConvergenceEstimate(
                converged=False,
                remaining_steps=None,
                predicted_final_loss=None,
                convergence_rate=0.0,
                confidence=0.0,
            )

        trend = self.compute_loss_trend()
        rate = self.compute_convergence_rate()
        converged = abs(trend) < self.convergence_threshold

        if target_loss is None:
            target_loss = self._estimate_asymptote()

        remaining = self.predict_steps_to_convergence(target_loss)
        predicted_final = self._estimate_asymptote()

        n = len(losses)
        confidence = min(1.0, n / (self.window * 3))

        return ConvergenceEstimate(
            converged=converged,
            remaining_steps=remaining,
            predicted_final_loss=predicted_final,
            convergence_rate=rate,
            confidence=confidence,
        )

    def detect_anomalies(self) -> list[str]:
        losses = self._losses()
        anomalies: list[str] = []

        if len(losses) < 3:
            return anomalies

        n = min(self.window, len(losses))
        recent = losses[-n:]
        mean_loss = sum(recent) / len(recent)
        var = sum((x - mean_loss) ** 2 for x in recent) / len(recent)
        std = var ** 0.5

        if std > 1e-12:
            latest = losses[-1]
            z_score = abs(latest - mean_loss) / std
            if z_score > 3.0:
                anomalies.append(f"loss anomaly: z-score {z_score:.2f} at cycle {self._points[-1].cycle}")

        # Check for loss increase after sustained decrease
        if len(losses) >= 4:
            prev_trend = _linear_slope(losses[-4:-1])
            if prev_trend < 0:
                last_diff = losses[-1] - losses[-2]
                if last_diff > 0 and abs(last_diff) > abs(prev_trend) * 2:
                    anomalies.append(f"loss reversal: increasing loss after {n}-cycle downward trend")

        # Check velocity magnitude trend if available
        vels = [p.velocity_magnitude for p in self._points if p.velocity_magnitude is not None]
        if len(vels) >= 3:
            vel_trend = _linear_slope(vels[-min(self.window, len(vels)):])
            if vel_trend > 0 and self.compute_loss_trend() > 0:
                anomalies.append("velocity divergence: velocity magnitude increasing while loss increasing")

        return anomalies

    def early_stop_advice(
        self,
        patience: int = 5,
        min_improvement: float = 1e-4,
    ) -> EarlyStopAdvice:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_improvement < 0:
            raise ValueError(f"min_improvement must be >= 0, got {min_improvement}")

        losses = self._losses()
        if len(losses) < self.min_points:
            return EarlyStopAdvice(
                should_stop=False,
                reason="insufficient data",
                estimated_gain_from_continuing=0.0,
            )

        best_loss = min(losses)
        best_idx = losses.index(best_loss)
        current = losses[-1]
        cycles_since_best = len(losses) - 1 - best_idx

        convergence = self.estimate_convergence()
        trend = self.compute_loss_trend()
        volatility = self.compute_volatility()

        estimated_gain = 0.0
        if convergence.predicted_final_loss is not None:
            estimated_gain = max(0.0, current - convergence.predicted_final_loss)

        should_stop = False
        reason = ""

        if convergence.converged:
            should_stop = True
            reason = "converged: loss trend below threshold"
        elif cycles_since_best >= patience and trend >= 0:
            should_stop = True
            reason = f"stagnant: no improvement for {cycles_since_best} cycles with flat/upward trend"
        elif volatility > abs(best_loss) * 0.1 and cycles_since_best >= patience // 2:
            should_stop = True
            reason = "unstable: high volatility with no consistent improvement"
        elif estimated_gain < min_improvement and cycles_since_best >= patience:
            should_stop = True
            reason = f"marginal: estimated remaining gain ({estimated_gain:.6f}) below threshold ({min_improvement})"

        optimal_cycle = self._points[best_idx].cycle
        return EarlyStopAdvice(
            should_stop=should_stop,
            reason=reason,
            estimated_gain_from_continuing=estimated_gain,
            optimal_cycle=optimal_cycle,
        )

    def full_report(
        self,
        target_loss: float | None = None,
        patience: int = 5,
    ) -> TrajectoryReport:
        convergence = self.estimate_convergence(target_loss)
        early_stop = self.early_stop_advice(patience)
        anomalies = self.detect_anomalies()

        return TrajectoryReport(
            total_points=len(self._points),
            convergence=convergence,
            early_stop=early_stop,
            loss_trend=self.compute_loss_trend(),
            volatility=self.compute_volatility(),
            anomaly_detected=len(anomalies) > 0,
            anomaly_details=anomalies,
        )

    @classmethod
    def from_loss_history(
        cls,
        losses: list[float],
        *,
        window: int = 5,
        convergence_threshold: float = 1e-4,
    ) -> TrajectoryAnalyzer:
        analyzer = cls(window=window, convergence_threshold=convergence_threshold)
        for i, loss in enumerate(losses):
            analyzer.add_point(TrajectoryPoint(cycle=i, train_loss=loss))
        return analyzer

    @classmethod
    def from_dicts(
        cls,
        records: list[dict[str, Any]],
        *,
        window: int = 5,
        convergence_threshold: float = 1e-4,
    ) -> TrajectoryAnalyzer:
        analyzer = cls(window=window, convergence_threshold=convergence_threshold)
        for rec in records:
            train_loss = rec.get("train_loss", rec.get("loss_train"))
            if train_loss is None:
                continue
            cycle = rec.get("cycle")
            if cycle is None:
                cycle = rec.get("step")
            if cycle is None:
                cycle = len(analyzer._points)

            valid_loss = rec.get("valid_loss", rec.get("loss_valid"))
            grad_norm = rec.get("grad_norm")
            velocity_magnitude = rec.get("velocity_magnitude")

            if analyzer._points and analyzer._points[-1].cycle == cycle:
                previous = analyzer._points[-1]
                analyzer._points[-1] = TrajectoryPoint(
                    cycle=cycle,
                    train_loss=train_loss,
                    valid_loss=(
                        valid_loss if valid_loss is not None else previous.valid_loss
                    ),
                    grad_norm=grad_norm if grad_norm is not None else previous.grad_norm,
                    velocity_magnitude=(
                        velocity_magnitude
                        if velocity_magnitude is not None
                        else previous.velocity_magnitude
                    ),
                )
                continue

            analyzer.add_point(TrajectoryPoint(
                cycle=cycle,
                train_loss=train_loss,
                valid_loss=valid_loss,
                grad_norm=grad_norm,
                velocity_magnitude=velocity_magnitude,
            ))
        return analyzer


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    cov = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values))
    var_x = sum((i - mean_x) ** 2 for i in range(n))
    return cov / var_x if var_x > 0 else 0.0
