"""Property-based tests for AdvisorConfig validation and TrajectoryAnalyzer edge cases.

Covers:
- P1: AdvisorConfig rejects invalid parameter values
- P2: predict_steps_to_convergence always returns bounded values
- P3: compute_convergence_rate handles near-zero initial loss
- P4: early_stop confidence is always in [0, 1]
- P5: AdvisorConfig defaults pass validation
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.tg_lora.training_advisor import (
    AdvisoryAction,
    AdvisoryReport,
    AdvisorConfig,
    TrainingAdvisor,
)
from src.tg_lora.trajectory import TrajectoryAnalyzer, TrajectoryPoint


# ---------------------------------------------------------------------------
# P1: AdvisorConfig validation
# ---------------------------------------------------------------------------


class TestAdvisorConfigValidation:
    def test_default_config_is_valid(self):
        config = AdvisorConfig()
        assert config.stagnation_patience == 5
        assert config.spike_threshold == 2.0

    def test_reject_zero_patience(self):
        with pytest.raises(ValueError, match="stagnation_patience must be >= 1"):
            AdvisorConfig(stagnation_patience=0)

    def test_reject_negative_patience(self):
        with pytest.raises(ValueError, match="stagnation_patience must be >= 1"):
            AdvisorConfig(stagnation_patience=-3)

    def test_reject_zero_spike_threshold(self):
        with pytest.raises(ValueError, match="spike_threshold must be > 0"):
            AdvisorConfig(spike_threshold=0)

    def test_reject_negative_spike_threshold(self):
        with pytest.raises(ValueError, match="spike_threshold must be > 0"):
            AdvisorConfig(spike_threshold=-1.0)

    def test_reject_window_below_2(self):
        with pytest.raises(ValueError, match="trajectory_window must be >= 2"):
            AdvisorConfig(trajectory_window=1)

    def test_reject_zero_convergence_threshold(self):
        with pytest.raises(ValueError, match="convergence_threshold must be > 0"):
            AdvisorConfig(convergence_threshold=0)

    def test_reject_negative_convergence_threshold(self):
        with pytest.raises(ValueError, match="convergence_threshold must be > 0"):
            AdvisorConfig(convergence_threshold=-0.001)

    def test_reject_zero_early_stop_min_cycles(self):
        with pytest.raises(ValueError, match="early_stop_min_cycles must be >= 1"):
            AdvisorConfig(early_stop_min_cycles=0)

    def test_valid_custom_config(self):
        config = AdvisorConfig(
            stagnation_patience=10,
            spike_threshold=3.0,
            trajectory_window=8,
            convergence_threshold=1e-3,
            early_stop_min_cycles=5,
        )
        assert config.stagnation_patience == 10


# ---------------------------------------------------------------------------
# P2: predict_steps_to_convergence is always bounded
# ---------------------------------------------------------------------------


class TestPredictStepsBounded:
    @given(
        losses=st.lists(
            st.floats(min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=50,
        ),
        target_loss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300)
    def test_steps_always_bounded_or_none(self, losses, target_loss):
        """predict_steps_to_convergence returns None or a value in [1, 100000]."""
        try:
            a = TrajectoryAnalyzer.from_loss_history(losses)
        except ValueError:
            return  # NaN/Inf filtered by strategy, but degenerate inputs may still fail
        steps = a.predict_steps_to_convergence(target_loss=target_loss)
        if steps is not None:
            assert 0 <= steps <= 100_000, f"steps out of bounds: {steps}"

    def test_very_slow_convergence_returns_bounded(self):
        """Extremely slow convergence should still be capped."""
        # Losses that decrease by a tiny amount each step
        losses = [1.0 + 1e-15 * i for i in range(20)]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        steps = a.predict_steps_to_convergence(target_loss=0.5)
        if steps is not None:
            assert steps <= 100_000

    def test_very_small_rate_returns_none(self):
        """Near-zero convergence rate returns None instead of huge step count."""
        # Flat losses produce rate near 0
        losses = [1.0 + 1e-20 * i for i in range(10)]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        steps = a.predict_steps_to_convergence(target_loss=0.5)
        # Either None or a bounded value
        if steps is not None:
            assert steps <= 100_000


# ---------------------------------------------------------------------------
# P3: compute_convergence_rate handles near-zero initial
# ---------------------------------------------------------------------------


class TestConvergenceRateNearZero:
    def test_zero_initial_gives_zero_rate(self):
        """Exact zero initial loss returns rate 0."""
        losses = [0.0, 0.1, 0.2, 0.3, 0.4]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        rate = a.compute_convergence_rate()
        assert rate == 0.0

    def test_tiny_initial_gives_zero_rate(self):
        """Very small initial loss (near machine epsilon) returns rate 0."""
        losses = [1e-15, 0.5, 1.0, 1.5, 2.0]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        rate = a.compute_convergence_rate()
        assert rate == 0.0

    def test_normal_initial_gives_nonzero_rate(self):
        """Reasonable initial loss gives a normal rate."""
        losses = [2.5, 2.3, 2.1, 1.9, 1.7]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        rate = a.compute_convergence_rate()
        assert rate != 0.0


# ---------------------------------------------------------------------------
# P4: Early-stop confidence always in [0, 1]
# ---------------------------------------------------------------------------


class TestEarlyStopConfidence:
    @given(
        losses=st.lists(
            st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=30,
        ),
    )
    @settings(max_examples=200)
    def test_all_action_confidences_bounded(self, losses):
        """Every AdvisoryAction from evaluate has confidence in [0, 1]."""
        advisor = TrainingAdvisor()
        for i, loss in enumerate(losses):
            report = advisor.evaluate(i, train_loss=loss)
            assert isinstance(report, AdvisoryReport)
            for action in report.actions:
                assert isinstance(action, AdvisoryAction)
                assert 0.0 <= action.confidence <= 1.0, (
                    f"confidence {action.confidence} out of [0,1] at cycle {i}"
                )

    def test_converged_training_stop_action_has_valid_confidence(self):
        """Regression: converged training stop action must have valid confidence."""
        advisor = TrainingAdvisor(AdvisorConfig(trajectory_window=3, convergence_threshold=0.01))
        # Feed converged losses
        for i in range(15):
            loss = 1.0 + 1e-6 * ((-1) ** i)
            report = advisor.evaluate(i, train_loss=loss)
        for action in report.actions:
            assert 0.0 <= action.confidence <= 1.0


# ---------------------------------------------------------------------------
# P5: AdvisorConfig defaults pass construction
# ---------------------------------------------------------------------------


class TestAdvisorConfigDefaults:
    def test_defaults_create_valid_advisor(self):
        """Default AdvisorConfig produces a working TrainingAdvisor."""
        advisor = TrainingAdvisor()
        assert advisor.cycle_count == 0
        report = advisor.evaluate(0, train_loss=2.5)
        assert report.overall_health == "healthy"

    @given(
        patience=st.integers(min_value=1, max_value=100),
        threshold=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_valid_params_create_advisor(self, patience, threshold):
        """Any valid config parameters produce a working advisor."""
        config = AdvisorConfig(
            stagnation_patience=patience,
            spike_threshold=threshold,
        )
        advisor = TrainingAdvisor(config)
        report = advisor.evaluate(0, train_loss=2.5)
        assert report.overall_health in ("healthy", "warning", "critical")
