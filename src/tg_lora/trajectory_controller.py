"""Trajectory-informed adaptive controller.

Bridges :class:`TrajectoryAnalyzer` and :class:`RandomWalkController` so that
real-time trajectory insights (convergence, anomalies, early-stop signals)
inform hyperparameter proposals and training decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.tg_lora.random_walk_controller import (
    ControllerState,
    Proposal,
    RandomWalkController,
)
from src.tg_lora.trajectory import (
    TrajectoryAnalyzer,
    TrajectoryPoint,
    TrajectoryReport,
)


@dataclass
class CycleDecision:
    """Decision returned after recording a training cycle."""

    proposal: Proposal | None = None
    should_stop: bool = False
    stop_reason: str = ""
    anomaly_detected: bool = False
    anomaly_details: list[str] = field(default_factory=list)
    adaptive_adjustments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryControllerConfig:
    """Configuration for TrajectoryController."""

    trajectory_window: int = 5
    convergence_threshold: float = 1e-4
    early_stop_patience: int = 5
    early_stop_min_improvement: float = 1e-4
    enable_adaptive_alpha: bool = True
    enable_adaptive_lr: bool = True
    convergence_alpha_decay: float = 0.9
    plateau_alpha_boost: float = 1.3
    anomaly_lr_decay: float = 0.7
    anomaly_alpha_decay: float = 0.8
    analysis_interval: int = 1
    min_cycles_before_adaptation: int = 3
    lr_reject_decay_floor: float = 0.01
    alpha_max_ceiling: float = 50.0


class TrajectoryController:
    """Integrates trajectory analysis with adaptive control.

    After each training cycle, records the loss trajectory and analyzes
    convergence / anomaly patterns.  When the trajectory signals convergence
    or anomalies, the underlying :class:`RandomWalkController` parameters
    are adjusted to stabilize or accelerate training.
    """

    def __init__(
        self,
        controller: RandomWalkController,
        *,
        config: TrajectoryControllerConfig | None = None,
        analyzer: TrajectoryAnalyzer | None = None,
    ) -> None:
        if controller is None:
            raise ValueError("controller must not be None")
        self._controller = controller
        self._config = config or TrajectoryControllerConfig()
        self._analyzer = analyzer or TrajectoryAnalyzer(
            window=self._config.trajectory_window,
            convergence_threshold=self._config.convergence_threshold,
        )
        self._cycle_count: int = 0
        self._last_report: TrajectoryReport | None = None
        self._last_decision: CycleDecision | None = None
        self._cumulative_adaptations: dict[str, float] = {}

    @property
    def controller(self) -> RandomWalkController:
        return self._controller

    @property
    def analyzer(self) -> TrajectoryAnalyzer:
        return self._analyzer

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_report(self) -> TrajectoryReport | None:
        return self._last_report

    @property
    def last_decision(self) -> CycleDecision | None:
        return self._last_decision

    def record_cycle(
        self,
        cycle: int,
        train_loss: float,
        *,
        valid_loss: float | None = None,
        grad_norm: float | None = None,
        velocity_magnitude: float | None = None,
        loss_pilot: float = 0.0,
        loss_after: float = 0.0,
    ) -> CycleDecision:
        """Record a training cycle and return an adaptive decision.

        Parameters
        ----------
        cycle : int
            Current cycle number.
        train_loss : float
            Training loss for this cycle.
        valid_loss : float | None
            Validation loss (if available).
        grad_norm : float | None
            Gradient norm (if available).
        velocity_magnitude : float | None
            Velocity magnitude (if available).
        loss_pilot : float
            Pilot loss before the cycle.
        loss_after : float
            Loss after the cycle.

        Returns
        -------
        CycleDecision
            Adaptive decision including proposal, stop signal, and adjustments.
        """
        self._cycle_count += 1

        point = TrajectoryPoint(
            cycle=cycle,
            train_loss=train_loss,
            valid_loss=valid_loss,
            grad_norm=grad_norm,
            velocity_magnitude=velocity_magnitude,
        )
        self._analyzer.add_point(point)

        decision = CycleDecision()

        # Accept/reject with underlying controller
        if loss_pilot != 0.0 or loss_after != 0.0:
            accepted = self._controller.accept(loss_pilot, loss_after)
            if accepted:
                self._controller.reward(loss_pilot, loss_after)
            else:
                self._controller.penalize(loss_pilot, loss_after)
            decision.proposal = self._controller.propose()

        # Run trajectory analysis at intervals
        should_analyze = (
            self._cycle_count % self._config.analysis_interval == 0
            and self._cycle_count >= self._config.min_cycles_before_adaptation
        )

        if should_analyze:
            self._last_report = self._analyzer.full_report(
                patience=self._config.early_stop_patience,
            )
            decision = self._apply_trajectory_insights(decision)

        self._last_decision = decision
        return decision

    def _apply_trajectory_insights(self, decision: CycleDecision) -> CycleDecision:
        """Adjust controller parameters based on trajectory analysis."""
        if self._last_report is None:
            return decision

        report = self._last_report

        # Early stop check
        if report.early_stop.should_stop:
            decision.should_stop = True
            decision.stop_reason = report.early_stop.reason

        # Anomaly detection
        if report.anomaly_detected:
            decision.anomaly_detected = True
            decision.anomaly_details = list(report.anomaly_details)

            if self._config.enable_adaptive_lr:
                lr_factor = self._config.anomaly_lr_decay
                self._controller.state.lr_reject_decay *= lr_factor
                self._controller.state.lr_reject_decay = max(
                    self._controller.state.lr_reject_decay,
                    self._config.lr_reject_decay_floor,
                )
                decision.adaptive_adjustments["lr_decay"] = lr_factor
                self._cumulative_adaptations["lr_decay"] = (
                    self._cumulative_adaptations.get("lr_decay", 1.0) * lr_factor
                )

            if self._config.enable_adaptive_alpha:
                alpha_factor = self._config.anomaly_alpha_decay
                self._controller.alpha_max *= alpha_factor
                self._clamp_alpha_max()
                decision.adaptive_adjustments["alpha_decay"] = alpha_factor
                self._cumulative_adaptations["alpha_decay"] = (
                    self._cumulative_adaptations.get("alpha_decay", 1.0) * alpha_factor
                )

        # Convergence-driven adjustments
        convergence = report.convergence
        if convergence.converged:
            decision.adaptive_adjustments["converged"] = True
            if self._config.enable_adaptive_alpha:
                alpha_factor = self._config.convergence_alpha_decay
                self._controller.alpha_max *= alpha_factor
                self._clamp_alpha_max()
                decision.adaptive_adjustments["convergence_alpha"] = alpha_factor
                self._cumulative_adaptations["convergence_alpha"] = (
                    self._cumulative_adaptations.get("convergence_alpha", 1.0)
                    * alpha_factor
                )
        elif report.loss_trend >= 0 and report.volatility < 1e-3:
            # Plateau detected (no improvement, low volatility)
            if self._config.enable_adaptive_alpha:
                alpha_factor = self._config.plateau_alpha_boost
                self._controller.alpha_max *= alpha_factor
                self._clamp_alpha_max()
                decision.adaptive_adjustments["plateau_alpha"] = alpha_factor
                self._cumulative_adaptations["plateau_alpha"] = (
                    self._cumulative_adaptations.get("plateau_alpha", 1.0)
                    * alpha_factor
                )

        # Feed convergence trend into controller
        if convergence.convergence_rate != 0:
            self._controller.adapt_to_convergence(convergence.convergence_rate)

        return decision

    def _clamp_alpha_max(self) -> None:
        self._controller.alpha_max = max(
            self._controller.alpha_min,
            min(self._controller.alpha_max, self._config.alpha_max_ceiling),
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary of the trajectory controller state."""
        return {
            "cycle_count": self._cycle_count,
            "controller_summary": self._controller.summary(),
            "cumulative_adaptations": dict(self._cumulative_adaptations),
            "last_anomaly_detected": (
                self._last_report.anomaly_detected if self._last_report else None
            ),
            "last_converged": (
                self._last_report.convergence.converged if self._last_report else None
            ),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore controller and analyzer state from a dict."""
        if "controller_state" in state:
            self._controller.restore_state(
                ControllerState.from_dict(state["controller_state"])
            )
        if "cycle_count" in state:
            self._cycle_count = state["cycle_count"]
        if "cumulative_adaptations" in state:
            self._cumulative_adaptations = dict(state["cumulative_adaptations"])
        if "analyzer_points" in state:
            self._analyzer = TrajectoryAnalyzer(
                window=self._config.trajectory_window,
                convergence_threshold=self._config.convergence_threshold,
            )
            self._analyzer.add_points(
                [
                    TrajectoryPoint(**p)
                    for p in state["analyzer_points"]
                    if isinstance(p, dict)
                ]
            )

    def export_state(self) -> dict[str, Any]:
        """Export controller and analyzer state as a serializable dict."""
        points_data = []
        for p in self._analyzer.points:
            points_data.append(
                {
                    "cycle": p.cycle,
                    "train_loss": p.train_loss,
                    "valid_loss": p.valid_loss,
                    "grad_norm": p.grad_norm,
                    "velocity_magnitude": p.velocity_magnitude,
                }
            )
        return {
            "cycle_count": self._cycle_count,
            "cumulative_adaptations": dict(self._cumulative_adaptations),
            "analyzer_points": points_data,
        }
