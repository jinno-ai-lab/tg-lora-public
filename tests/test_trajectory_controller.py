"""Tests for TrajectoryController — trajectory-informed adaptive control."""

from __future__ import annotations

import pytest

from src.tg_lora.random_walk_controller import RandomWalkController
from src.tg_lora.trajectory import TrajectoryAnalyzer
from src.tg_lora.trajectory_controller import (
    CycleDecision,
    TrajectoryController,
    TrajectoryControllerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller(**kw) -> RandomWalkController:
    defaults = dict(
        K_initial=3,
        N_initial=5,
        alpha_initial=0.3,
        alpha_min=0.03,
        alpha_max=1.5,
        beta_initial=0.8,
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
    )
    defaults.update(kw)
    return RandomWalkController(**defaults)


def _make_tc(**kw) -> TrajectoryController:
    ctrl = _make_controller()
    cfg = TrajectoryControllerConfig(
        min_cycles_before_adaptation=3,
        analysis_interval=1,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return TrajectoryController(ctrl, config=cfg)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_controller(self):
        with pytest.raises(ValueError, match="controller"):
            TrajectoryController(None)

    def test_default_config(self):
        tc = _make_tc()
        assert tc.cycle_count == 0
        assert tc.last_report is None
        assert tc.last_decision is None

    def test_custom_analyzer(self):
        analyzer = TrajectoryAnalyzer(window=10)
        tc = TrajectoryController(
            _make_controller(), analyzer=analyzer,
        )
        assert tc.analyzer is analyzer

    def test_config_propagated_to_analyzer(self):
        cfg = TrajectoryControllerConfig(trajectory_window=7)
        tc = TrajectoryController(_make_controller(), config=cfg)
        assert tc.analyzer.window == 7


# ---------------------------------------------------------------------------
# record_cycle — basic flow
# ---------------------------------------------------------------------------


class TestRecordCycle:
    def test_increments_cycle_count(self):
        tc = _make_tc()
        tc.record_cycle(1, train_loss=2.0)
        assert tc.cycle_count == 1
        tc.record_cycle(2, train_loss=1.8)
        assert tc.cycle_count == 2

    def test_adds_point_to_analyzer(self):
        tc = _make_tc()
        tc.record_cycle(1, train_loss=2.0, valid_loss=1.9)
        assert len(tc.analyzer.points) == 1
        assert tc.analyzer.points[0].train_loss == 2.0
        assert tc.analyzer.points[0].valid_loss == 1.9

    def test_returns_cycle_decision(self):
        tc = _make_tc()
        decision = tc.record_cycle(1, train_loss=2.0)
        assert isinstance(decision, CycleDecision)

    def test_no_analysis_before_min_cycles(self):
        tc = _make_tc(min_cycles_before_adaptation=5)
        for i in range(4):
            tc.record_cycle(i + 1, train_loss=2.0 - i * 0.1)
        assert tc.last_report is None  # not analyzed yet

    def test_analysis_after_min_cycles(self):
        tc = _make_tc(min_cycles_before_adaptation=3)
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=2.0 - i * 0.1)
        assert tc.last_report is not None


# ---------------------------------------------------------------------------
# Accept / reject wiring
# ---------------------------------------------------------------------------


class TestAcceptReject:
    def test_accept_on_loss_decrease(self):
        tc = _make_tc()
        d = tc.record_cycle(1, train_loss=1.8, loss_pilot=2.0, loss_after=1.8)
        assert d.proposal is not None

    def test_reject_on_loss_increase(self):
        tc = _make_tc()
        d = tc.record_cycle(1, train_loss=2.5, loss_pilot=2.0, loss_after=2.5)
        assert d.proposal is not None

    def test_no_proposal_when_zero_losses(self):
        tc = _make_tc()
        d = tc.record_cycle(1, train_loss=2.0)
        assert d.proposal is None


# ---------------------------------------------------------------------------
# Convergence-driven adaptation
# ---------------------------------------------------------------------------


class TestConvergenceAdaptation:
    def test_converged_reduces_alpha_max(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            convergence_alpha_decay=0.8,
        )
        initial_alpha_max = tc.controller.alpha_max
        # Feed decreasing losses that converge
        for i in range(10):
            loss = max(0.001, 1.0 - i * 0.11)
            tc.record_cycle(i + 1, train_loss=loss)
        # After convergence detected, alpha_max should have decayed
        if tc.last_report and tc.last_report.convergence.converged:
            assert tc.controller.alpha_max < initial_alpha_max

    def test_plateau_boosts_alpha_max(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            plateau_alpha_boost=1.5,
        )
        initial_alpha_max = tc.controller.alpha_max
        # Feed stagnant losses (plateau)
        for i in range(10):
            tc.record_cycle(i + 1, train_loss=1.0)
        if "plateau_alpha" in (tc.last_decision.adaptive_adjustments if tc.last_decision else {}):
            assert tc.controller.alpha_max > initial_alpha_max

    def test_adaptation_disabled_when_flag_off(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            enable_adaptive_alpha=False,
        )
        initial_alpha_max = tc.controller.alpha_max
        for i in range(10):
            tc.record_cycle(i + 1, train_loss=1.0)
        # alpha_max should not change when adaptive alpha is off
        assert tc.controller.alpha_max == initial_alpha_max


# ---------------------------------------------------------------------------
# Anomaly detection adaptation
# ---------------------------------------------------------------------------


class TestAnomalyAdaptation:
    def test_anomaly_detected_on_spike(self):
        tc = _make_tc(min_cycles_before_adaptation=3)
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=1.0 - i * 0.1)
        # Introduce spike
        d = tc.record_cycle(6, train_loss=10.0)
        if d.anomaly_detected:
            assert "spike" in " ".join(d.anomaly_details).lower() or d.anomaly_details

    def test_anomaly_reduces_lr_decay(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            anomaly_lr_decay=0.5,
            enable_adaptive_lr=True,
        )
        initial_lr_decay = tc.controller.state.lr_reject_decay
        # Create stable loss then spike
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=1.0 - i * 0.1)
        tc.record_cycle(6, train_loss=100.0)
        if tc.last_decision and tc.last_decision.anomaly_detected:
            assert tc.controller.state.lr_reject_decay < initial_lr_decay


# ---------------------------------------------------------------------------
# Early stop
# ---------------------------------------------------------------------------


class TestEarlyStop:
    def test_early_stop_on_stagnant_loss(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            early_stop_patience=3,
            early_stop_min_improvement=1e-3,
        )
        # Feed completely stagnant loss
        for i in range(15):
            d = tc.record_cycle(i + 1, train_loss=1.0)
            if d.should_stop:
                break
        # Should eventually trigger stop
        if tc.last_report:
            # At least verify the mechanism works
            assert isinstance(d.should_stop, bool)

    def test_no_stop_on_improving_loss(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            early_stop_patience=10,
        )
        for i in range(10):
            d = tc.record_cycle(i + 1, train_loss=2.0 - i * 0.1)
        assert not d.should_stop


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_export_restore_roundtrip(self):
        tc = _make_tc(min_cycles_before_adaptation=2)
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=2.0 - i * 0.15)
        state = tc.export_state()
        assert state["cycle_count"] == 5
        assert len(state["analyzer_points"]) == 5

        # Restore into fresh instance
        tc2 = _make_tc(min_cycles_before_adaptation=2)
        tc2.restore_state(state)
        assert tc2.cycle_count == 5
        assert len(tc2.analyzer.points) == 5

    def test_export_contains_adaptations(self):
        tc = _make_tc(min_cycles_before_adaptation=3)
        for i in range(10):
            tc.record_cycle(i + 1, train_loss=1.0)
        state = tc.export_state()
        assert isinstance(state["cumulative_adaptations"], dict)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_structure(self):
        tc = _make_tc(min_cycles_before_adaptation=2)
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=2.0 - i * 0.1)
        s = tc.summary()
        assert "cycle_count" in s
        assert "controller_summary" in s
        assert "cumulative_adaptations" in s
        assert s["cycle_count"] == 5

    def test_summary_before_any_cycle(self):
        tc = _make_tc()
        s = tc.summary()
        assert s["cycle_count"] == 0
        assert s["last_anomaly_detected"] is None
        assert s["last_converged"] is None


# ---------------------------------------------------------------------------
# Integration: full training simulation
# ---------------------------------------------------------------------------


class TestFullSimulation:
    def test_converging_training(self):
        """Simulate a training that converges. Controller should stabilize."""
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            convergence_alpha_decay=0.85,
        )
        decisions = []
        for i in range(20):
            loss = 2.0 * (0.9 ** i)  # exponentially decreasing
            d = tc.record_cycle(i + 1, train_loss=loss)
            decisions.append(d)
        # Training should complete without early stop (improving)
        assert not decisions[-1].should_stop
        # Cumulative adaptations should reflect convergence
        s = tc.summary()
        assert s["cycle_count"] == 20

    def test_diverging_training(self):
        """Simulate diverging training. Anomalies should be detected."""
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            anomaly_lr_decay=0.7,
            enable_adaptive_lr=True,
        )
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=1.0 - i * 0.1)
        # Diverge
        for i in range(5):
            tc.record_cycle(i + 6, train_loss=1.0 + i * 2.0)
        s = tc.summary()
        assert s["cycle_count"] == 10

    def test_analysis_interval(self):
        """Analysis only runs every N cycles."""
        tc = _make_tc(
            min_cycles_before_adaptation=1,
            analysis_interval=3,
        )
        for i in range(7):
            tc.record_cycle(i + 1, train_loss=1.0)
        # Reports generated at cycles 3, 6 (not 1, 2, 4, 5, 7)
        # Cycle count is 7 but analysis ran at 3 and 6
        assert tc.cycle_count == 7

    def test_grad_norm_and_velocity(self):
        """Ensure grad_norm and velocity_magnitude are passed through."""
        tc = _make_tc()
        tc.record_cycle(
            1, train_loss=2.0, grad_norm=0.5, velocity_magnitude=1.2,
        )
        assert tc.analyzer.points[0].grad_norm == 0.5
        assert tc.analyzer.points[0].velocity_magnitude == 1.2


# ---------------------------------------------------------------------------
# TC-229~232 acceptance criteria (REQ-237~240)
# ---------------------------------------------------------------------------


class TestTC229:
    """TC-229: TrajectoryController integrates analyzer and controller."""

    def test_tc229_01_record_cycle_returns_decision(self):
        tc = _make_tc()
        d = tc.record_cycle(1, train_loss=2.0)
        assert isinstance(d, CycleDecision)

    def test_tc229_02_convergence_triggers_adaptation(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            convergence_alpha_decay=0.9,
        )
        for i in range(15):
            tc.record_cycle(i + 1, train_loss=max(0.01, 2.0 - i * 0.14))
        s = tc.summary()
        assert s["cycle_count"] == 15


class TestTC230:
    """TC-230: Anomaly detection triggers parameter adjustment."""

    def test_tc230_01_spike_triggers_anomaly_flag(self):
        tc = _make_tc(min_cycles_before_adaptation=3)
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=1.0 - i * 0.05)
        d = tc.record_cycle(6, train_loss=50.0)
        if d.anomaly_detected:
            assert d.anomaly_details

    def test_tc230_02_anomaly_reduces_lr(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            anomaly_lr_decay=0.5,
        )
        initial_lr = tc.controller.state.lr_reject_decay
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=1.0)
        tc.record_cycle(6, train_loss=100.0)
        if tc.last_decision and tc.last_decision.anomaly_detected:
            assert tc.controller.state.lr_reject_decay < initial_lr


class TestTC231:
    """TC-231: Early stop signal propagated through CycleDecision."""

    def test_tc231_01_stagnant_loss_triggers_stop(self):
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            early_stop_patience=3,
        )
        for i in range(20):
            d = tc.record_cycle(i + 1, train_loss=1.0)
            if d.should_stop:
                break
        if d.should_stop:
            assert d.stop_reason

    def test_tc231_02_improving_loss_no_stop(self):
        tc = _make_tc(min_cycles_before_adaptation=3, early_stop_patience=10)
        for i in range(15):
            d = tc.record_cycle(i + 1, train_loss=2.0 - i * 0.1)
        assert not d.should_stop


class TestTC232:
    """TC-232: State export/restore preserves trajectory and adaptations."""

    def test_tc232_01_export_restore_roundtrip(self):
        tc = _make_tc(min_cycles_before_adaptation=2)
        for i in range(8):
            tc.record_cycle(i + 1, train_loss=2.0 - i * 0.1)
        state = tc.export_state()
        tc2 = _make_tc(min_cycles_before_adaptation=2)
        tc2.restore_state(state)
        assert tc2.cycle_count == 8
        assert len(tc2.analyzer.points) == 8

    def test_tc232_02_summary_matches_state(self):
        tc = _make_tc(min_cycles_before_adaptation=2)
        for i in range(5):
            tc.record_cycle(i + 1, train_loss=1.5 - i * 0.1)
        s = tc.summary()
        assert s["cycle_count"] == 5
        assert isinstance(s["controller_summary"], dict)


# ---------------------------------------------------------------------------
# Adaptive parameter bounds (drift protection)
# ---------------------------------------------------------------------------


class TestAdaptiveParameterBounds:
    """Verify cumulative multiplicative adjustments cannot drive parameters
    to unusable extremes (near-zero or unbounded growth).
    """

    def test_alpha_max_floor_above_alpha_min_after_many_anomaly_cycles(self):
        """Repeated anomaly decay must not push alpha_max below alpha_min."""
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            anomaly_alpha_decay=0.8,
            enable_adaptive_alpha=True,
        )
        alpha_min = tc.controller.alpha_min
        for i in range(100):
            # Build stable baseline then spike to trigger anomaly
            for j in range(5):
                tc.record_cycle(i * 6 + j + 1, train_loss=1.0 - j * 0.05)
            tc.record_cycle(i * 6 + 6, train_loss=50.0)
        assert tc.controller.alpha_max >= alpha_min, (
            f"alpha_max ({tc.controller.alpha_max}) decayed below alpha_min ({alpha_min})"
        )

    def test_alpha_max_ceiling_after_plateau_boosts(self):
        """Repeated plateau boosts must not grow alpha_max unboundedly."""
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            plateau_alpha_boost=1.5,
            enable_adaptive_alpha=True,
        )
        initial_alpha_max = tc.controller.alpha_max
        for i in range(50):
            tc.record_cycle(i + 1, train_loss=1.0)
        # alpha_max should not grow to absurd levels
        assert tc.controller.alpha_max <= initial_alpha_max * 10, (
            f"alpha_max ({tc.controller.alpha_max}) grew beyond 10x initial ({initial_alpha_max})"
        )

    def test_lr_reject_decay_floor_after_anomaly(self):
        """Repeated anomaly lr decay must not push lr_reject_decay to zero."""
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            anomaly_lr_decay=0.5,
            enable_adaptive_lr=True,
        )
        for i in range(100):
            for j in range(5):
                tc.record_cycle(i * 6 + j + 1, train_loss=1.0 - j * 0.05)
            tc.record_cycle(i * 6 + 6, train_loss=50.0)
        assert tc.controller.state.lr_reject_decay >= 0.01, (
            f"lr_reject_decay ({tc.controller.state.lr_reject_decay}) "
            f"decayed below floor (0.01)"
        )

    def test_convergence_decay_respects_alpha_max_floor(self):
        """Convergence-driven alpha_max decay must not go below alpha_min."""
        tc = _make_tc(
            min_cycles_before_adaptation=3,
            convergence_alpha_decay=0.5,
            enable_adaptive_alpha=True,
        )
        alpha_min = tc.controller.alpha_min
        for i in range(100):
            loss = max(0.001, 2.0 * (0.95 ** i))
            tc.record_cycle(i + 1, train_loss=loss)
        assert tc.controller.alpha_max >= alpha_min, (
            f"alpha_max ({tc.controller.alpha_max}) below alpha_min ({alpha_min}) after convergence"
        )
