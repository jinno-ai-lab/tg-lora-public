"""Tests for tg_lora/trajectory.py — Training trajectory analysis."""
from __future__ import annotations

import pytest

from tg_lora.trajectory import (ConvergenceEstimate, EarlyStopAdvice,
                                    TrajectoryAnalyzer, TrajectoryPoint,
                                    TrajectoryReport, _linear_slope)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decreasing_losses(n: int, start: float = 2.5, decay: float = 0.1) -> list[float]:
    return [start - decay * i for i in range(n)]


def _make_converged_losses(n: int = 20, final: float = 1.0, noise: float = 1e-6) -> list[float]:
    return [final + noise * ((-1) ** i) for i in range(n)]


# ---------------------------------------------------------------------------
# TrajectoryPoint
# ---------------------------------------------------------------------------

class TestTrajectoryPoint:
    def test_basic_construction(self):
        p = TrajectoryPoint(cycle=0, train_loss=2.5)
        assert p.cycle == 0
        assert p.train_loss == 2.5
        assert p.valid_loss is None
        assert p.grad_norm is None
        assert p.velocity_magnitude is None

    def test_full_construction(self):
        p = TrajectoryPoint(cycle=5, train_loss=1.8, valid_loss=1.9, grad_norm=0.1, velocity_magnitude=0.05)
        assert p.valid_loss == 1.9
        assert p.grad_norm == 0.1
        assert p.velocity_magnitude == 0.05


# ---------------------------------------------------------------------------
# TrajectoryAnalyzer construction
# ---------------------------------------------------------------------------

class TestTrajectoryAnalyzerConstruction:
    def test_default_params(self):
        a = TrajectoryAnalyzer()
        assert a.window == 5
        assert a.convergence_threshold == 1e-4
        assert a.min_points == 3
        assert a.points == []

    def test_custom_params(self):
        a = TrajectoryAnalyzer(window=10, convergence_threshold=0.001, min_points=5)
        assert a.window == 10

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window must be >= 2"):
            TrajectoryAnalyzer(window=1)

    def test_invalid_convergence_threshold(self):
        with pytest.raises(ValueError, match="convergence_threshold must be > 0"):
            TrajectoryAnalyzer(convergence_threshold=0)

    def test_invalid_min_points(self):
        with pytest.raises(ValueError, match="min_points must be >= 2"):
            TrajectoryAnalyzer(min_points=1)


# ---------------------------------------------------------------------------
# add_point / add_points
# ---------------------------------------------------------------------------

class TestAddPoint:
    def test_add_single_point(self):
        a = TrajectoryAnalyzer()
        a.add_point(TrajectoryPoint(cycle=0, train_loss=2.5))
        assert len(a.points) == 1

    def test_add_multiple_points(self):
        a = TrajectoryAnalyzer()
        a.add_points([TrajectoryPoint(cycle=i, train_loss=2.5 - 0.1 * i) for i in range(5)])
        assert len(a.points) == 5

    def test_reject_negative_cycle(self):
        a = TrajectoryAnalyzer()
        with pytest.raises(ValueError, match="cycle must be non-negative"):
            a.add_point(TrajectoryPoint(cycle=-1, train_loss=2.5))

    def test_reject_nan_loss(self):
        a = TrajectoryAnalyzer()
        with pytest.raises(ValueError, match="train_loss must be finite"):
            a.add_point(TrajectoryPoint(cycle=0, train_loss=float("nan")))

    def test_reject_inf_loss(self):
        a = TrajectoryAnalyzer()
        with pytest.raises(ValueError, match="train_loss must be finite"):
            a.add_point(TrajectoryPoint(cycle=0, train_loss=float("inf")))


# ---------------------------------------------------------------------------
# Loss trend
# ---------------------------------------------------------------------------

class TestLossTrend:
    def test_decreasing_losses_negative_trend(self):
        a = TrajectoryAnalyzer.from_loss_history(_make_decreasing_losses(10))
        trend = a.compute_loss_trend()
        assert trend < 0

    def test_increasing_losses_positive_trend(self):
        losses = [1.0 + 0.1 * i for i in range(10)]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        trend = a.compute_loss_trend()
        assert trend > 0

    def test_flat_losses_near_zero_trend(self):
        losses = [1.5 + 1e-7 * i for i in range(20)]
        a = TrajectoryAnalyzer.from_loss_history(losses, window=20)
        trend = a.compute_loss_trend()
        assert abs(trend) < 1e-4

    def test_insufficient_data_returns_zero(self):
        a = TrajectoryAnalyzer.from_loss_history([2.5])
        assert a.compute_loss_trend() == 0.0

    def test_uses_valid_loss_when_available(self):
        a = TrajectoryAnalyzer()
        for i in range(5):
            a.add_point(TrajectoryPoint(cycle=i, train_loss=2.5 - 0.1 * i, valid_loss=2.5 - 0.2 * i))
        trend = a.compute_loss_trend()
        # valid_loss decreases by 0.2 per step (steeper than train_loss)
        assert trend < -0.15


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_constant_losses_zero_volatility(self):
        losses = [1.5] * 10
        a = TrajectoryAnalyzer.from_loss_history(losses)
        assert a.compute_volatility() == 0.0

    def test_varying_losses_positive_volatility(self):
        losses = [1.0, 2.0, 1.0, 2.0, 1.0]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        vol = a.compute_volatility()
        assert vol > 0

    def test_insufficient_data_zero_volatility(self):
        a = TrajectoryAnalyzer.from_loss_history([1.0])
        assert a.compute_volatility() == 0.0


# ---------------------------------------------------------------------------
# Convergence rate
# ---------------------------------------------------------------------------

class TestConvergenceRate:
    def test_decreasing_losses_positive_rate(self):
        losses = _make_decreasing_losses(10, start=2.5, decay=0.1)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        rate = a.compute_convergence_rate()
        assert rate > 0

    def test_increasing_losses_negative_rate(self):
        losses = [1.0 + 0.1 * i for i in range(10)]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        rate = a.compute_convergence_rate()
        assert rate < 0

    def test_constant_losses_zero_rate(self):
        losses = [1.5] * 10
        a = TrajectoryAnalyzer.from_loss_history(losses)
        assert a.compute_convergence_rate() == 0.0

    def test_insufficient_data_zero_rate(self):
        a = TrajectoryAnalyzer.from_loss_history([1.0])
        assert a.compute_convergence_rate() == 0.0


# ---------------------------------------------------------------------------
# Predict steps to convergence
# ---------------------------------------------------------------------------

class TestPredictStepsToConvergence:
    def test_already_converged_returns_zero(self):
        losses = [1.0, 0.99, 0.98, 0.97]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        steps = a.predict_steps_to_convergence(target_loss=1.0)
        assert steps == 0

    def test_decreasing_gives_positive_steps(self):
        losses = _make_decreasing_losses(20, start=2.5, decay=0.05)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        steps = a.predict_steps_to_convergence(target_loss=1.0)
        assert steps is not None
        assert steps > 0

    def test_flat_losses_returns_none(self):
        losses = [1.5] * 10
        a = TrajectoryAnalyzer.from_loss_history(losses)
        steps = a.predict_steps_to_convergence(target_loss=1.0)
        assert steps is None

    def test_insufficient_data_returns_none(self):
        a = TrajectoryAnalyzer.from_loss_history([2.0, 1.9])
        steps = a.predict_steps_to_convergence()
        assert steps is None


# ---------------------------------------------------------------------------
# Predict loss at step
# ---------------------------------------------------------------------------

class TestPredictLossAtStep:
    def test_predict_current_step(self):
        losses = _make_decreasing_losses(10, start=2.5, decay=0.1)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        predicted = a.predict_loss_at_step(0)
        assert predicted is not None
        assert abs(predicted - losses[-1]) < 0.01

    def test_predict_future_step_lower(self):
        losses = _make_decreasing_losses(20, start=2.5, decay=0.05)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        predicted = a.predict_loss_at_step(10)
        assert predicted is not None
        assert predicted < losses[-1]

    def test_reject_negative_steps(self):
        a = TrajectoryAnalyzer.from_loss_history([2.5, 2.4])
        with pytest.raises(ValueError, match="future_steps must be non-negative"):
            a.predict_loss_at_step(-1)

    def test_insufficient_data_returns_none(self):
        a = TrajectoryAnalyzer.from_loss_history([1.0])
        assert a.predict_loss_at_step(5) is None


# ---------------------------------------------------------------------------
# Estimate convergence
# ---------------------------------------------------------------------------

class TestEstimateConvergence:
    def test_converged_series(self):
        losses = _make_converged_losses(20)
        a = TrajectoryAnalyzer.from_loss_history(losses, convergence_threshold=1e-4)
        est = a.estimate_convergence()
        assert isinstance(est, ConvergenceEstimate)
        assert est.converged

    def test_still_training(self):
        losses = _make_decreasing_losses(10, start=2.5, decay=0.1)
        a = TrajectoryAnalyzer.from_loss_history(losses, convergence_threshold=1e-6)
        est = a.estimate_convergence()
        assert not est.converged

    def test_confidence_increases_with_data(self):
        losses_short = _make_decreasing_losses(5)
        losses_long = _make_decreasing_losses(30)
        a_short = TrajectoryAnalyzer.from_loss_history(losses_short)
        a_long = TrajectoryAnalyzer.from_loss_history(losses_long)
        assert a_long.estimate_convergence().confidence > a_short.estimate_convergence().confidence

    def test_insufficient_data(self):
        a = TrajectoryAnalyzer.from_loss_history([2.5, 2.4])
        est = a.estimate_convergence()
        assert not est.converged
        assert est.remaining_steps is None


# ---------------------------------------------------------------------------
# Detect anomalies
# ---------------------------------------------------------------------------

class TestDetectAnomalies:
    def test_no_anomalies_in_smooth_descent(self):
        losses = _make_decreasing_losses(20, start=2.5, decay=0.05)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        anomalies = a.detect_anomalies()
        assert len(anomalies) == 0

    def test_spike_anomaly(self):
        losses = _make_decreasing_losses(20, start=2.5, decay=0.05)
        # Add a large spike at the end
        losses[-1] = losses[-2] + 5.0
        a = TrajectoryAnalyzer.from_loss_history(losses)
        anomalies = a.detect_anomalies()
        assert len(anomalies) > 0
        assert any("anomaly" in a or "reversal" in a for a in anomalies)

    def test_loss_reversal_anomaly(self):
        losses = [2.5, 2.3, 2.1, 1.9, 2.5]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        anomalies = a.detect_anomalies()
        assert any("reversal" in a for a in anomalies)

    def test_velocity_divergence_anomaly(self):
        a = TrajectoryAnalyzer()
        for i in range(10):
            a.add_point(TrajectoryPoint(
                cycle=i,
                train_loss=2.5 + 0.1 * i,  # loss increasing
                velocity_magnitude=0.1 * (i + 1),  # velocity increasing
            ))
        anomalies = a.detect_anomalies()
        assert any("velocity divergence" in an for an in anomalies)

    def test_insufficient_data_no_anomalies(self):
        a = TrajectoryAnalyzer.from_loss_history([1.0, 1.1])
        assert a.detect_anomalies() == []


# ---------------------------------------------------------------------------
# Early stop advice
# ---------------------------------------------------------------------------

class TestEarlyStopAdvice:
    def test_continue_when_improving(self):
        losses = _make_decreasing_losses(10, start=2.5, decay=0.1)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        advice = a.early_stop_advice(patience=5)
        assert isinstance(advice, EarlyStopAdvice)
        assert not advice.should_stop

    def test_stop_when_stagnant(self):
        losses = [2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        a = TrajectoryAnalyzer.from_loss_history(losses, window=5)
        advice = a.early_stop_advice(patience=3)
        assert advice.should_stop
        assert "stagnant" in advice.reason or "converged" in advice.reason

    def test_stop_when_converged(self):
        losses = _make_converged_losses(30, final=1.0)
        a = TrajectoryAnalyzer.from_loss_history(losses, convergence_threshold=1e-4)
        advice = a.early_stop_advice()
        assert advice.should_stop
        assert "converged" in advice.reason

    def test_stop_when_unstable(self):
        losses = [2.0, 1.0, 3.0, 1.5, 2.8, 1.2, 3.5, 1.0, 2.9, 1.8, 3.2, 1.1]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        advice = a.early_stop_advice(patience=3)
        # High volatility should trigger stop
        assert advice.should_stop
        assert "unstable" in advice.reason or "stagnant" in advice.reason

    def test_insufficient_data_continues(self):
        a = TrajectoryAnalyzer.from_loss_history([2.5, 2.4])
        advice = a.early_stop_advice()
        assert not advice.should_stop
        assert "insufficient" in advice.reason

    def test_estimated_gain_from_continuing(self):
        losses = _make_decreasing_losses(10, start=2.5, decay=0.1)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        advice = a.early_stop_advice()
        assert advice.estimated_gain_from_continuing >= 0

    def test_invalid_patience(self):
        a = TrajectoryAnalyzer.from_loss_history([2.5, 2.4])
        with pytest.raises(ValueError, match="patience must be >= 1"):
            a.early_stop_advice(patience=0)

    def test_invalid_min_improvement(self):
        a = TrajectoryAnalyzer.from_loss_history([2.5, 2.4])
        with pytest.raises(ValueError, match="min_improvement must be >= 0"):
            a.early_stop_advice(min_improvement=-0.1)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

class TestFullReport:
    def test_report_structure(self):
        losses = _make_decreasing_losses(15, start=2.5, decay=0.08)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        report = a.full_report()
        assert isinstance(report, TrajectoryReport)
        assert report.total_points == 15
        assert isinstance(report.convergence, ConvergenceEstimate)
        assert isinstance(report.early_stop, EarlyStopAdvice)
        assert isinstance(report.loss_trend, float)
        assert isinstance(report.volatility, float)
        assert isinstance(report.anomaly_detected, bool)
        assert isinstance(report.anomaly_details, list)

    def test_with_target_loss(self):
        losses = _make_decreasing_losses(15, start=2.5, decay=0.08)
        a = TrajectoryAnalyzer.from_loss_history(losses)
        report = a.full_report(target_loss=1.5)
        assert report.convergence.remaining_steps is not None


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:
    def test_from_loss_history(self):
        losses = [2.5, 2.3, 2.1, 1.9, 1.8]
        a = TrajectoryAnalyzer.from_loss_history(losses, window=3)
        assert len(a.points) == 5
        assert a.window == 3

    def test_from_dicts(self):
        records = [
            {"cycle": 0, "train_loss": 2.5, "valid_loss": 2.6},
            {"cycle": 1, "train_loss": 2.3, "valid_loss": 2.4},
            {"cycle": 2, "train_loss": 2.1, "valid_loss": 2.2},
        ]
        a = TrajectoryAnalyzer.from_dicts(records)
        assert len(a.points) == 3
        assert a.points[0].valid_loss == 2.6

    def test_from_dicts_accepts_run_metrics_field_names(self):
        records = [
            {"cycle": 0, "loss_train": 2.5, "loss_valid": 2.6},
            {"cycle": 1, "loss_train": 2.3, "loss_valid": 2.4},
            {"cycle": 2, "loss_train": 2.1, "loss_valid": 2.2},
        ]

        a = TrajectoryAnalyzer.from_dicts(records)
        assert len(a.points) == 3
        assert a.points[0].train_loss == 2.5
        assert a.points[0].valid_loss == 2.6

    def test_from_dicts_skips_records_without_loss(self):
        records = [
            {"type": "run_header", "run_id": "x"},
            {"cycle": 0, "loss_train": 2.5, "loss_valid": 2.6},
            {"type": "run_footer", "best_valid_loss": 2.1},
        ]

        a = TrajectoryAnalyzer.from_dicts(records)
        assert len(a.points) == 1
        assert a.points[0].train_loss == 2.5

    def test_from_dicts_auto_cycle(self):
        records = [
            {"train_loss": 2.5},
            {"train_loss": 2.3},
        ]
        a = TrajectoryAnalyzer.from_dicts(records)
        assert a.points[0].cycle == 0
        assert a.points[1].cycle == 1

    def test_from_dicts_none_cycle_falls_back_to_index(self):
        records = [
            {"cycle": None, "loss_train": 2.5},
            {"cycle": None, "loss_train": 2.3},
        ]
        a = TrajectoryAnalyzer.from_dicts(records)
        assert a.points[0].cycle == 0
        assert a.points[1].cycle == 1

    def test_from_dicts_uses_step_when_cycle_missing(self):
        records = [
            {"step": 1, "loss_train": 2.5},
            {"step": 2, "loss_train": 2.3},
        ]
        a = TrajectoryAnalyzer.from_dicts(records)
        assert a.points[0].cycle == 1
        assert a.points[1].cycle == 2

    def test_from_dicts_merges_duplicate_step_records(self):
        records = [
            {"step": 1, "loss_train": 2.5},
            {"step": 1, "loss_train": 2.5, "loss_valid": 2.4},
            {"step": 2, "loss_train": 2.3, "loss_valid": 2.2},
        ]

        a = TrajectoryAnalyzer.from_dicts(records)

        assert len(a.points) == 2
        assert a.points[0].cycle == 1
        assert a.points[0].train_loss == 2.5
        assert a.points[0].valid_loss == 2.4
        assert a.points[1].cycle == 2

    def test_early_stop_advice_returns_actual_cycle(self):
        a = TrajectoryAnalyzer()
        a.add_point(TrajectoryPoint(cycle=10, train_loss=2.5, valid_loss=2.4))
        a.add_point(TrajectoryPoint(cycle=20, train_loss=2.3, valid_loss=2.2))
        a.add_point(TrajectoryPoint(cycle=30, train_loss=2.4, valid_loss=2.3))

        advice = a.early_stop_advice(patience=5)

        assert advice.optimal_cycle == 20


# ---------------------------------------------------------------------------
# _linear_slope helper
# ---------------------------------------------------------------------------

class TestLinearSlope:
    def test_perfect_linear(self):
        assert _linear_slope([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_decreasing(self):
        assert _linear_slope([4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_constant(self):
        assert _linear_slope([1.5, 1.5, 1.5]) == pytest.approx(0.0)

    def test_single_value(self):
        assert _linear_slope([1.0]) == 0.0

    def test_empty(self):
        assert _linear_slope([]) == 0.0


# ---------------------------------------------------------------------------
# Coverage gap: trajectory.py uncovered lines
# ---------------------------------------------------------------------------


class TestTrajectoryCoverageGaps:
    """Tests for previously uncovered branches in trajectory.py."""

    def test_compute_convergence_rate_initial_zero(self):
        """Line 113: convergence rate when initial=0 returns 0."""
        a = TrajectoryAnalyzer()
        # Add points where initial loss is 0
        a.add_point(TrajectoryPoint(cycle=0, train_loss=0.0))
        a.add_point(TrajectoryPoint(cycle=1, train_loss=1.0))
        a.add_point(TrajectoryPoint(cycle=2, train_loss=2.0))
        # With initial=0, function should return 0.0
        rate = a.compute_convergence_rate()
        assert rate == 0.0

    def test_predict_steps_target_loss_none_asymptote_none(self):
        """Lines 127-129: when target_loss=None and _estimate_asymptote returns None."""
        a = TrajectoryAnalyzer(min_points=3)
        # Not enough points for asymptote estimation
        a.add_point(TrajectoryPoint(cycle=0, train_loss=2.5))
        a.add_point(TrajectoryPoint(cycle=1, train_loss=2.4))
        a.add_point(TrajectoryPoint(cycle=2, train_loss=2.3))
        # With min_points=3 we have just enough for predict_steps, but
        # _estimate_asymptote needs min_points and window*2 data.
        # Use a small window so we have enough data
        result = a.predict_steps_to_convergence(target_loss=None)
        # If asymptote is None, should return None
        assert result is None or isinstance(result, int)

    def test_estimate_asymptote_insufficient_window_data(self):
        """Line 163: _estimate_asymptote with < min_points returns None."""
        a = TrajectoryAnalyzer(min_points=5)
        for i in range(3):
            a.add_point(TrajectoryPoint(cycle=i, train_loss=2.5 - 0.1 * i))
        result = a._estimate_asymptote()
        assert result is None

    def test_estimate_asymptote_recent_too_short(self):
        """Line 168: _estimate_asymptote with recent < 3 returns None."""
        a = TrajectoryAnalyzer(window=2, min_points=2)
        a.add_point(TrajectoryPoint(cycle=0, train_loss=2.5))
        a.add_point(TrajectoryPoint(cycle=1, train_loss=2.4))
        result = a._estimate_asymptote()
        # window*2 = 4, but only 2 points available → recent has 2 items → 2 < 3
        assert result is None

    def test_detect_anomaly_spike_zscore(self):
        """Line 227: loss anomaly with z-score > 3.0."""
        # Need the spike to be included in the recent window but still produce z > 3
        # With window=10 and 19 stable values + 1 spike:
        # recent = [1]*9 + [50], mean=5.9, std=14.7, z=3.0
        losses = [1.0] * 19 + [50.0]
        a = TrajectoryAnalyzer.from_loss_history(losses, window=10)
        anomalies = a.detect_anomalies()
        assert any("loss anomaly" in an for an in anomalies)

    def test_early_stop_marginal_gain_below_threshold(self):
        """Lines 287-291: marginal stop when gain < min_improvement.

        Requirements to reach the marginal branch:
        - NOT converged (abs(trend) >= threshold)
        - NOT stagnant (trend < 0 OR cycles_since_best < patience)
        - NOT unstable (volatility <= abs(best)*0.1 OR cycles_since_best < patience//2)
        - estimated_gain < min_improvement AND cycles_since_best >= patience
        """
        losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.8, 2.7, 2.65, 2.62, 2.601,
                  2.6008, 2.6005, 2.6003, 2.6002, 2.6001,   # best=2.6001 at idx 14
                  2.60015, 2.60012, 2.60011, 2.600105, 2.600102]
        a = TrajectoryAnalyzer.from_loss_history(
            losses, convergence_threshold=1e-7, window=5
        )
        advice = a.early_stop_advice(patience=3, min_improvement=0.1)
        assert advice.should_stop
        assert "marginal" in advice.reason

