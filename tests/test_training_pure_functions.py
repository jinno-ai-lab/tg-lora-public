"""Unit tests for pure functions extracted from train_tg_lora.py (TASK-0012)."""

import math

import pytest

from src.training.train_tg_lora import (_compute_pilot_average,
                                        _decide_accept_rollback,
                                        _evaluate_full_eval_outcome,
                                        _format_cycle_progress,
                                        _should_fallback_to_baseline_like)


# ---------------------------------------------------------------------------
# _compute_pilot_average
# ---------------------------------------------------------------------------
class TestComputePilotAverage:
    def test_normal_values(self):
        avg, m = _compute_pilot_average([2.0, 4.0, 6.0], K=3)
        assert avg == pytest.approx(4.0)
        assert m["K"] == 3
        assert m["count"] == 3
        assert m["min_loss"] == 2.0
        assert m["max_loss"] == 6.0

    def test_single_value(self):
        avg, m = _compute_pilot_average([3.5], K=1)
        assert avg == pytest.approx(3.5)
        assert m["count"] == 1
        assert m["min_loss"] == m["max_loss"] == 3.5

    def test_empty_list_returns_nan(self):
        avg, m = _compute_pilot_average([], K=3)
        assert math.isnan(avg)
        assert m["count"] == 0
        assert math.isnan(m["avg_loss"])

    def test_nan_in_losses_filtered(self):
        """Non-finite losses are filtered out; average is from finite values only."""
        avg, m = _compute_pilot_average([1.0, float("nan"), 3.0], K=3)
        assert avg == pytest.approx(2.0)
        assert m["finite_count"] == 2
        assert m["count"] == 3

    def test_all_nan_losses_returns_nan(self):
        avg, m = _compute_pilot_average([float("nan"), float("inf")], K=2)
        assert math.isnan(avg)
        assert m["finite_count"] == 0
        assert m["total_count"] == 2

    def test_inf_in_losses_filtered(self):
        avg, m = _compute_pilot_average([1.0, float("inf"), 5.0], K=3)
        assert avg == pytest.approx(3.0)
        assert m["finite_count"] == 2

    def test_negative_inf_filtered(self):
        avg, m = _compute_pilot_average([2.0, float("-inf"), 4.0], K=3)
        assert avg == pytest.approx(3.0)

    def test_large_k_mismatch(self):
        """K parameter is informational; average uses len(step_losses)."""
        avg, m = _compute_pilot_average([1.0, 2.0], K=5)
        assert avg == pytest.approx(1.5)
        assert m["K"] == 5
        assert m["count"] == 2

    def test_negative_losses(self):
        avg, m = _compute_pilot_average([-1.0, -3.0], K=2)
        assert avg == pytest.approx(-2.0)
        assert m["min_loss"] == -3.0
        assert m["max_loss"] == -1.0

    def test_identical_values(self):
        avg, m = _compute_pilot_average([5.0, 5.0, 5.0, 5.0], K=4)
        assert avg == pytest.approx(5.0)
        assert m["min_loss"] == m["max_loss"] == 5.0


# ---------------------------------------------------------------------------
# _decide_accept_rollback
# ---------------------------------------------------------------------------
class TestDecideAcceptRollback:
    def test_improvement(self):
        accepted, reason = _decide_accept_rollback(2.0, 1.5, 0.01)
        assert accepted is True
        assert reason == "improvement"

    def test_exact_equality(self):
        accepted, reason = _decide_accept_rollback(2.0, 2.0, 0.0)
        assert accepted is True
        assert reason == "improvement"

    def test_within_tolerance(self):
        accepted, reason = _decide_accept_rollback(2.0, 2.005, 0.01)
        assert accepted is True
        assert reason == "within_tolerance"

    def test_tolerance_boundary_accepted(self):
        accepted, reason = _decide_accept_rollback(2.0, 2.01, 0.01)
        assert accepted is True
        assert reason == "within_tolerance"

    def test_tolerance_boundary_rejected(self):
        accepted, reason = _decide_accept_rollback(2.0, 2.02, 0.01)
        assert accepted is False
        assert "degradation" in reason

    def test_large_degradation(self):
        accepted, reason = _decide_accept_rollback(1.0, 5.0, 0.01)
        assert accepted is False
        assert "degradation" in reason

    def test_zero_tolerance(self):
        accepted, reason = _decide_accept_rollback(2.0, 2.0001, 0.0)
        assert accepted is False

    def test_zero_tolerance_improvement(self):
        accepted, reason = _decide_accept_rollback(2.0, 1.9999, 0.0)
        assert accepted is True
        assert reason == "improvement"

    def test_near_zero_pilot_loss_rejects(self):
        accepted, reason = _decide_accept_rollback(0.0, 0.001, 0.01)
        assert accepted is False
        assert "degradation" in reason

    def test_near_zero_pilot_loss_improvement(self):
        accepted, reason = _decide_accept_rollback(0.001, 0.0, 0.01)
        assert accepted is True
        assert reason == "improvement"

    def test_negative_losses(self):
        accepted, reason = _decide_accept_rollback(-1.0, -1.5, 0.01)
        assert accepted is True
        assert reason == "improvement"


# ---------------------------------------------------------------------------
# _format_cycle_progress
# ---------------------------------------------------------------------------
class TestFormatCycleProgress:
    def test_accepted_format(self):
        s = _format_cycle_progress(5, 1.23456, True, 0.8912, 0.15, 3, 5)
        assert "c=5" in s
        assert "loss=1.2346" in s
        assert "Y" in s
        assert "cos=0.891" in s
        assert "K=3" in s
        assert "N=5" in s

    def test_rejected_format(self):
        s = _format_cycle_progress(0, 0.5, False, 0.0, 0.0, 1, 1)
        assert "N" in s
        assert "c=0" in s

    def test_zero_reduction_rate(self):
        s = _format_cycle_progress(0, 1.0, True, 0.5, 0.0, 2, 3)
        assert "red=0.0%" in s

    def test_precision_loss(self):
        s = _format_cycle_progress(10, 1.23456789, True, 0.123456, 0.5678, 4, 7)
        # loss has 4 decimal places
        assert "loss=1.2346" in s
        # cos_sim has 3 decimal places
        assert "cos=0.123" in s

    def test_large_cycle_number(self):
        s = _format_cycle_progress(99999, 1.0, True, 0.5, 0.25, 10, 20)
        assert "c=99999" in s


# ---------------------------------------------------------------------------
# _evaluate_full_eval_outcome
# ---------------------------------------------------------------------------
class TestEvaluateFullEvalOutcome:
    def test_new_best(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            1.0,
            2.0,
            stale_cycles=3,
            patience=5,
            min_cycles=10,
            current_cycle=12,
        )
        assert is_best is True
        assert stop is False
        assert "new_best" in reason


# ---------------------------------------------------------------------------
# _should_fallback_to_baseline_like
# ---------------------------------------------------------------------------
class TestShouldFallbackToBaselineLike:
    def test_disabled_guard_never_falls_back(self):
        fallback, reason = _should_fallback_to_baseline_like(
            proposal_N=5,
            total_cycles=10,
            acceptance_rate=0.0,
            pilot_loss=10.0,
            previous_valid_loss=1.0,
            acceleration=1.0,
            velocity_anomalous=True,
            enabled=False,
            warmup_cycles=2,
            min_acceptance_rate=0.5,
            pilot_margin=0.01,
            max_positive_acceleration=0.0,
        )
        assert fallback is False
        assert reason == "disabled"

    def test_warmup_defers_fallback(self):
        fallback, reason = _should_fallback_to_baseline_like(
            proposal_N=5,
            total_cycles=1,
            acceptance_rate=0.0,
            pilot_loss=10.0,
            previous_valid_loss=1.0,
            acceleration=1.0,
            velocity_anomalous=True,
            enabled=True,
            warmup_cycles=2,
            min_acceptance_rate=0.5,
            pilot_margin=0.01,
            max_positive_acceleration=0.0,
        )
        assert fallback is False
        assert reason == "warmup"

    def test_velocity_anomaly_triggers_fallback(self):
        fallback, reason = _should_fallback_to_baseline_like(
            proposal_N=5,
            total_cycles=3,
            acceptance_rate=0.9,
            pilot_loss=1.0,
            previous_valid_loss=1.0,
            acceleration=0.0,
            velocity_anomalous=True,
            enabled=True,
            warmup_cycles=2,
            min_acceptance_rate=0.5,
            pilot_margin=0.01,
            max_positive_acceleration=0.0,
        )
        assert fallback is True
        assert reason == "velocity_anomaly"

    def test_positive_acceleration_with_unstable_pilot_triggers_fallback(self):
        fallback, reason = _should_fallback_to_baseline_like(
            proposal_N=5,
            total_cycles=3,
            acceptance_rate=0.9,
            pilot_loss=1.05,
            previous_valid_loss=1.0,
            acceleration=0.03,
            velocity_anomalous=False,
            enabled=True,
            warmup_cycles=2,
            min_acceptance_rate=0.5,
            pilot_margin=0.01,
            max_positive_acceleration=0.0,
        )
        assert fallback is True
        assert reason.startswith("positive_acceleration")

    def test_positive_acceleration_with_stable_pilot_keeps_tg(self):
        fallback, reason = _should_fallback_to_baseline_like(
            proposal_N=5,
            total_cycles=3,
            acceptance_rate=0.9,
            pilot_loss=1.005,
            previous_valid_loss=1.0,
            acceleration=0.03,
            velocity_anomalous=False,
            enabled=True,
            warmup_cycles=2,
            min_acceptance_rate=0.5,
            pilot_margin=0.01,
            max_positive_acceleration=0.0,
        )
        assert fallback is False
        assert reason == "linearity_ok"

    def test_low_acceptance_and_unstable_pilot_trigger_fallback(self):
        fallback, reason = _should_fallback_to_baseline_like(
            proposal_N=5,
            total_cycles=5,
            acceptance_rate=0.2,
            pilot_loss=1.5,
            previous_valid_loss=1.0,
            acceleration=0.0,
            velocity_anomalous=False,
            enabled=True,
            warmup_cycles=2,
            min_acceptance_rate=0.5,
            pilot_margin=0.01,
            max_positive_acceleration=0.0,
        )
        assert fallback is True
        assert reason == "low_acceptance_and_unstable_pilot"

    def test_stable_pilot_can_keep_tg_even_with_low_acceptance(self):
        fallback, reason = _should_fallback_to_baseline_like(
            proposal_N=5,
            total_cycles=5,
            acceptance_rate=0.2,
            pilot_loss=1.005,
            previous_valid_loss=1.0,
            acceleration=0.0,
            velocity_anomalous=False,
            enabled=True,
            warmup_cycles=2,
            min_acceptance_rate=0.5,
            pilot_margin=0.01,
            max_positive_acceleration=0.0,
        )
        assert fallback is False
        assert reason == "linearity_ok"

    def test_no_improvement_within_patience(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            3.0,
            2.0,
            stale_cycles=2,
            patience=5,
            min_cycles=10,
            current_cycle=12,
        )
        assert is_best is False
        assert stop is False
        assert "no_improvement" in reason
        assert "stale=3" in reason

    def test_early_stop_triggered(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            3.0,
            2.0,
            stale_cycles=4,
            patience=5,
            min_cycles=10,
            current_cycle=12,
        )
        assert is_best is False
        assert stop is True
        assert "early_stop" in reason
        assert "stale=5" in reason

    def test_no_stop_below_min_cycles(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            3.0,
            2.0,
            stale_cycles=10,
            patience=5,
            min_cycles=15,
            current_cycle=12,
        )
        assert stop is False

    def test_patience_none_no_stop(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            3.0,
            2.0,
            stale_cycles=100,
            patience=None,
            min_cycles=10,
            current_cycle=50,
        )
        assert stop is False
        assert is_best is False

    def test_exact_patience_boundary(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            3.0,
            2.0,
            stale_cycles=4,
            patience=5,
            min_cycles=5,
            current_cycle=5,
        )
        # stale becomes 5, which equals patience=5
        assert stop is True

    def test_new_best_resets_stale(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            1.0,
            2.0,
            stale_cycles=100,
            patience=5,
            min_cycles=10,
            current_cycle=50,
        )
        assert is_best is True
        assert stop is False

    def test_equal_loss_not_new_best(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            2.0,
            2.0,
            stale_cycles=0,
            patience=5,
            min_cycles=10,
            current_cycle=12,
        )
        assert is_best is False
        assert "stale=1" in reason
