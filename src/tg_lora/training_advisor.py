"""Training advisor — consolidates monitoring signals into actionable advice.

Integrates CycleMonitor (health), TrajectoryAnalyzer (convergence/trend),
and TrajectoryController (adaptive decisions) into a single advisory layer
that produces structured, prioritized recommendations.

Phase 61 of the TG-LoRA development roadmap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from src.tg_lora.cycle_monitor import CycleMonitor, HealthReport
from src.tg_lora.trajectory import (
    TrajectoryAnalyzer,
    TrajectoryPoint,
    TrajectoryReport,
)


ActionType = Literal[
    "reduce_lr",
    "increase_lr",
    "stop_training",
    "save_checkpoint",
    "increase_k",
    "decrease_k",
    "adjust_alpha",
    "rollback",
    "resume",
    "no_action",
]

Priority = Literal["critical", "high", "medium", "low"]

HealthStatus = Literal["healthy", "warning", "critical"]


@dataclass
class AdvisoryAction:
    """A single recommended action with justification."""

    action_type: ActionType
    priority: Priority
    reason: str
    suggested_value: float | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class AdvisoryReport:
    """Full advisory report produced by TrainingAdvisor."""

    overall_health: HealthStatus
    actions: list[AdvisoryAction] = field(default_factory=list)
    summary: str = ""
    cycle_health: HealthReport | None = None
    trajectory_summary: dict[str, Any] | None = None
    timestamp: str = ""

    def top_action(self) -> AdvisoryAction | None:
        if not self.actions:
            return None
        order: dict[Priority, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(self.actions, key=lambda a: order.get(a.priority, 99))[0]


@dataclass
class AdvisorConfig:
    """Configuration for TrainingAdvisor."""

    stagnation_patience: int = 5
    spike_threshold: float = 2.0
    trajectory_window: int = 5
    convergence_threshold: float = 1e-4
    plateau_lr_factor: float = 0.5
    anomaly_lr_factor: float = 0.7
    convergence_alpha_factor: float = 0.9
    plateau_alpha_factor: float = 1.3
    early_stop_min_cycles: int = 10
    save_checkpoint_on_best: bool = True

    def __post_init__(self) -> None:
        if self.stagnation_patience < 1:
            raise ValueError(f"stagnation_patience must be >= 1, got {self.stagnation_patience}")
        if self.spike_threshold <= 0:
            raise ValueError(f"spike_threshold must be > 0, got {self.spike_threshold}")
        if self.trajectory_window < 2:
            raise ValueError(f"trajectory_window must be >= 2, got {self.trajectory_window}")
        if self.convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold must be > 0, got {self.convergence_threshold}")
        if self.early_stop_min_cycles < 1:
            raise ValueError(f"early_stop_min_cycles must be >= 1, got {self.early_stop_min_cycles}")


class TrainingAdvisor:
    """Consolidates monitoring signals into prioritized training advice.

    Integrates:
    - CycleMonitor for divergence/stagnation detection
    - TrajectoryAnalyzer for convergence prediction and early-stop
    - Direct loss/gradient observation for real-time advice

    Usage::

        advisor = TrainingAdvisor()
        for cycle in range(max_cycles):
            report = advisor.evaluate(cycle, train_loss=loss, ...)
            if report.top_action():
                print(report.top_action().reason)
    """

    def __init__(self, config: AdvisorConfig | None = None) -> None:
        self._config = config or AdvisorConfig()
        self._monitor = CycleMonitor(
            patience=self._config.stagnation_patience,
            spike_threshold=self._config.spike_threshold,
        )
        self._analyzer = TrajectoryAnalyzer(
            window=self._config.trajectory_window,
            convergence_threshold=self._config.convergence_threshold,
        )
        self._cycle_count: int = 0
        self._best_loss: float | None = None
        self._best_cycle: int | None = None

    @property
    def monitor(self) -> CycleMonitor:
        return self._monitor

    @property
    def analyzer(self) -> TrajectoryAnalyzer:
        return self._analyzer

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def best_loss(self) -> float | None:
        return self._best_loss

    @property
    def best_cycle(self) -> int | None:
        return self._best_cycle

    def evaluate(
        self,
        cycle: int,
        *,
        train_loss: float,
        valid_loss: float | None = None,
        grad_norm: float | None = None,
        velocity_magnitude: float | None = None,
        loss_pilot: float = 0.0,
        loss_after: float = 0.0,
        acceptance_rate: float | None = None,
    ) -> AdvisoryReport:
        """Evaluate current training state and return advisory report.

        Parameters
        ----------
        cycle : int
            Current training cycle number.
        train_loss : float
            Training loss for this cycle.
        valid_loss : float | None
            Validation loss if available.
        grad_norm : float | None
            Gradient norm if available.
        velocity_magnitude : float | None
            Velocity magnitude if available.
        loss_pilot : float
            Pilot loss before the TG-LoRA cycle.
        loss_after : float
            Loss after the TG-LoRA cycle.
        acceptance_rate : float | None
            Fraction of accepted TG-LoRA proposals.

        Returns
        -------
        AdvisoryReport
            Structured advice with prioritized actions.
        """
        import math

        is_nonfinite = math.isnan(train_loss) or math.isinf(train_loss)
        self._cycle_count += 1

        # Track best loss
        effective_loss = valid_loss if valid_loss is not None else train_loss
        is_new_best = False
        if (
            effective_loss is not None
            and not math.isnan(effective_loss)
            and not math.isinf(effective_loss)
            and (self._best_loss is None or effective_loss < self._best_loss)
        ):
            self._best_loss = effective_loss
            self._best_cycle = cycle
            is_new_best = True

        # Feed cycle monitor (handles NaN/Inf internally)
        cycle_data: dict[str, Any] = {
            "cycle": cycle,
            "train_loss": train_loss,
        }
        if valid_loss is not None:
            cycle_data["valid_loss"] = valid_loss
        if grad_norm is not None:
            cycle_data["grad_norm"] = grad_norm
        health = self._monitor.update(cycle_data)

        # Feed trajectory analyzer (only finite loss values)
        if not is_nonfinite:
            point = TrajectoryPoint(
                cycle=cycle,
                train_loss=train_loss,
                valid_loss=valid_loss,
                grad_norm=grad_norm,
                velocity_magnitude=velocity_magnitude,
            )
            self._analyzer.add_point(point)

        # Build trajectory report (requires enough points)
        traj_summary: dict[str, Any] | None = None
        trajectory_report: TrajectoryReport | None = None
        if len(self._analyzer._points) >= self._analyzer.min_points:
            trajectory_report = self._analyzer.full_report(
                patience=self._config.stagnation_patience,
            )
            traj_summary = {
                "converged": trajectory_report.convergence.converged,
                "convergence_rate": trajectory_report.convergence.convergence_rate,
                "predicted_final_loss": trajectory_report.convergence.predicted_final_loss,
                "loss_trend": trajectory_report.loss_trend,
                "volatility": trajectory_report.volatility,
                "anomaly_detected": trajectory_report.anomaly_detected,
                "early_stop": trajectory_report.early_stop.should_stop,
            }

        # Generate actions
        actions = self._generate_actions(
            health=health,
            trajectory=trajectory_report,
            is_new_best=is_new_best,
            loss_pilot=loss_pilot,
            loss_after=loss_after,
            acceptance_rate=acceptance_rate,
        )

        # Determine overall health
        overall = self._determine_health(health, trajectory_report)

        # Build summary
        summary = self._build_summary(overall, health, trajectory_report, actions)

        return AdvisoryReport(
            overall_health=overall,
            actions=actions,
            summary=summary,
            cycle_health=health,
            trajectory_summary=traj_summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_actions(
        self,
        *,
        health: HealthReport,
        trajectory: TrajectoryReport | None,
        is_new_best: bool,
        loss_pilot: float,
        loss_after: float,
        acceptance_rate: float | None,
    ) -> list[AdvisoryAction]:
        actions: list[AdvisoryAction] = []

        # 1. Critical: divergence (NaN/Inf or severe spike)
        if health.divergence.detected:
            if health.divergence.severity == "critical":
                actions.append(
                    AdvisoryAction(
                        action_type="rollback",
                        priority="critical",
                        reason=f"NaN/Inf detected in {health.divergence.metric}",
                        confidence=1.0,
                    )
                )
                actions.append(
                    AdvisoryAction(
                        action_type="reduce_lr",
                        priority="critical",
                        reason="Critical divergence requires immediate LR reduction",
                        suggested_value=self._config.anomaly_lr_factor,
                        confidence=1.0,
                    )
                )
            elif health.divergence.severity == "high":
                actions.append(
                    AdvisoryAction(
                        action_type="reduce_lr",
                        priority="high",
                        reason=f"Loss spike detected: {health.divergence.current_value:.4f} "
                               f"exceeds threshold {health.divergence.threshold:.4f}",
                        suggested_value=self._config.plateau_lr_factor,
                        confidence=0.9,
                    )
                )

        # 2. Trajectory-based actions
        if trajectory is not None:
            # Early stop
            if trajectory.early_stop.should_stop:
                actions.append(
                    AdvisoryAction(
                        action_type="stop_training",
                        priority="high",
                        reason=f"Early stop recommended: {trajectory.early_stop.reason}",
                        confidence=max(0.0, min(1.0, trajectory.early_stop.estimated_gain_from_continuing)),
                    )
                )

            # Anomaly-driven
            if trajectory.anomaly_detected:
                actions.append(
                    AdvisoryAction(
                        action_type="reduce_lr",
                        priority="high",
                        reason="Anomaly detected in loss trajectory",
                        suggested_value=self._config.anomaly_lr_factor,
                        confidence=0.85,
                    )
                )
                actions.append(
                    AdvisoryAction(
                        action_type="adjust_alpha",
                        priority="medium",
                        reason="Reduce extrapolation strength during anomaly",
                        suggested_value=self._config.convergence_alpha_factor,
                        confidence=0.8,
                    )
                )

            # Convergence-driven
            if trajectory.convergence.converged:
                actions.append(
                    AdvisoryAction(
                        action_type="stop_training",
                        priority="medium",
                        reason="Loss has converged below threshold",
                        confidence=0.7,
                    )
                )

            # Plateau detection (non-negative trend, low volatility)
            if trajectory.loss_trend >= 0 and trajectory.volatility < 1e-3:
                actions.append(
                    AdvisoryAction(
                        action_type="increase_k",
                        priority="medium",
                        reason="Plateau detected — increase extrapolation steps to escape",
                        confidence=0.6,
                    )
                )
                actions.append(
                    AdvisoryAction(
                        action_type="adjust_alpha",
                        priority="low",
                        reason="Plateau — try increasing alpha for more aggressive extrapolation",
                        suggested_value=self._config.plateau_alpha_factor,
                        confidence=0.5,
                    )
                )

            # Strong downward trend — consider increasing LR
            if trajectory.loss_trend < -1e-3 and trajectory.volatility < 1e-2:
                if not trajectory.anomaly_detected:
                    actions.append(
                        AdvisoryAction(
                            action_type="increase_lr",
                            priority="low",
                            reason="Strong downward trend with low volatility — safe to increase LR slightly",
                            suggested_value=1.2,
                            confidence=0.4,
                        )
                    )

        # 3. Stagnation actions
        if health.stagnation.detected:
            actions.append(
                AdvisoryAction(
                    action_type="increase_k",
                    priority="high",
                    reason=f"Stagnation: {health.stagnation.cycles_without_improvement} "
                           f"cycles without improvement",
                    confidence=0.8,
                )
            )
            if health.stagnation.cycles_without_improvement >= self._config.stagnation_patience * 2:
                actions.append(
                    AdvisoryAction(
                        action_type="rollback",
                        priority="medium",
                        reason="Prolonged stagnation — consider rollback to best checkpoint",
                        confidence=0.7,
                    )
                )

        # 4. Checkpoint recommendation
        if is_new_best and self._config.save_checkpoint_on_best and self._cycle_count > 1:
            actions.append(
                AdvisoryAction(
                    action_type="save_checkpoint",
                    priority="medium",
                    reason=f"New best loss: {self._best_loss:.6f} at cycle {self._best_cycle}",
                    confidence=1.0,
                )
            )

        # 5. Low acceptance rate
        if acceptance_rate is not None and acceptance_rate < 0.3 and self._cycle_count >= 5:
            actions.append(
                AdvisoryAction(
                    action_type="decrease_k",
                    priority="medium",
                    reason=f"Low acceptance rate ({acceptance_rate:.1%}) — "
                           f"reduce extrapolation aggressiveness",
                    confidence=0.6,
                )
            )

        # 6. No actions means healthy
        if not actions:
            actions.append(
                AdvisoryAction(
                    action_type="no_action",
                    priority="low",
                    reason="Training is progressing normally",
                    confidence=1.0,
                )
            )

        return actions

    def _determine_health(
        self,
        health: HealthReport,
        trajectory: TrajectoryReport | None,
    ) -> HealthStatus:
        if health.divergence.detected and health.divergence.severity == "critical":
            return "critical"
        if trajectory is not None and trajectory.early_stop.should_stop:
            # Convergence is a successful outcome, not an error
            if trajectory.convergence.converged:
                pass  # fall through to warning/healthy checks
            else:
                return "critical"
        if health.divergence.detected:
            return "warning"
        if health.stagnation.detected:
            return "warning"
        if trajectory is not None and trajectory.anomaly_detected:
            return "warning"
        if trajectory is not None and trajectory.loss_trend >= 0 and trajectory.volatility < 1e-3:
            return "warning"
        return "healthy"

    def _build_summary(
        self,
        overall: HealthStatus,
        health: HealthReport,
        trajectory: TrajectoryReport | None,
        actions: list[AdvisoryAction],
    ) -> str:
        parts: list[str] = [f"Overall: {overall}"]

        if health.divergence.detected:
            parts.append(f"Divergence: {health.divergence.severity} in {health.divergence.metric}")

        if health.stagnation.detected:
            parts.append(f"Stagnation: {health.stagnation.cycles_without_improvement} cycles without improvement")

        if trajectory is not None:
            if trajectory.convergence.converged:
                parts.append("Converged")
            if trajectory.anomaly_detected:
                parts.append(f"Anomaly: {', '.join(trajectory.anomaly_details)}")

        critical = [a for a in actions if a.priority in ("critical", "high")]
        if critical:
            parts.append(f"Top action: {critical[0].action_type} — {critical[0].reason}")

        return "; ".join(parts)

    def summary(self) -> dict[str, Any]:
        """Return advisor state summary."""
        return {
            "cycle_count": self._cycle_count,
            "best_loss": self._best_loss,
            "best_cycle": self._best_cycle,
            "monitor_summary": self._monitor.health_summary(),
        }


def generate_advice_from_history(
    history: list[dict[str, Any]],
    *,
    config: AdvisorConfig | None = None,
) -> AdvisoryReport:
    """Generate advisory report from a list of cycle history records.

    Each record should have keys like 'cycle', 'train_loss', 'valid_loss',
    'grad_norm', 'velocity_magnitude', etc.

    Returns the advisory report for the last cycle.
    """
    advisor = TrainingAdvisor(config=config)
    report = AdvisoryReport(overall_health="healthy", summary="No data")
    for record in history:
        train_loss = record["train_loss"]
        # Skip non-finite losses that would crash the internal analyzer
        import math
        if math.isnan(train_loss) or math.isinf(train_loss):
            # Feed NaN to cycle monitor (which handles it) but skip analyzer
            cycle_data = {"cycle": record.get("cycle", 0), "train_loss": train_loss}
            if record.get("valid_loss") is not None:
                cycle_data["valid_loss"] = record["valid_loss"]
            advisor._monitor.update(cycle_data)
            advisor._cycle_count += 1
            # Build a minimal critical report
            report = AdvisoryReport(
                overall_health="critical",
                actions=[
                    AdvisoryAction(
                        action_type="rollback",
                        priority="critical",
                        reason=f"NaN/Inf in train_loss: {train_loss}",
                        confidence=1.0,
                    )
                ],
                summary=f"Critical: Non-finite train_loss {train_loss}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            continue
        report = advisor.evaluate(
            cycle=record.get("cycle", 0),
            train_loss=train_loss,
            valid_loss=record.get("valid_loss"),
            grad_norm=record.get("grad_norm"),
            velocity_magnitude=record.get("velocity_magnitude"),
            loss_pilot=record.get("loss_pilot", 0.0),
            loss_after=record.get("loss_after", 0.0),
            acceptance_rate=record.get("acceptance_rate"),
        )
    return report
