"""Tests for src/tg_lora/trajectory.py — Training trajectory analysis (Phase 59)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.tg_lora.trajectory import (ConvergenceEstimate, EarlyStopAdvice,
                                    TrajectoryAnalyzer, TrajectoryPoint,
                                    TrajectoryReport, _linear_slope)

SCRIPT = Path("scripts/analyze_trajectory.py")

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

    def test_zero_best_loss_does_not_collapse_volatility_threshold(self):
        # A run that converged to the loss floor — best_loss == 0.0 is a
        # LEGITIMATE signal here (e.g. the proxy memorize task, where
        # valid_loss=0.0; see cycle_monitor.detect_stagnation) — and then
        # oscillates at the floor must NOT be mislabeled "unstable". The
        # relative threshold ``abs(best_loss) * 0.1`` collapses to 0.0 there,
        # so any nonzero volatility trips the half-patience branch with a
        # wrong reason (mirrors detect_divergence's ``prev_train > 1e-6``
        # near-zero guard against ratio amplification). cycles_since_best (2)
        # is >= patience//2 (2) but < patience (5), so ONLY the volatility
        # branch can fire pre-fix.
        losses = [2.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.0, 0.00005, 0.0001]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        advice = a.early_stop_advice(patience=5)
        assert "unstable" not in advice.reason
        assert not advice.should_stop

    def test_above_floor_unstable_still_flags(self):
        # Non-regression: the near-zero guard must NOT suppress a genuinely
        # unstable run whose best_loss is meaningfully above the floor.
        losses = [1.0, 0.6, 0.9, 0.5, 0.85, 0.55, 0.9, 0.5, 0.88, 0.52]
        a = TrajectoryAnalyzer.from_loss_history(losses)
        advice = a.early_stop_advice(patience=5)
        assert advice.should_stop
        assert "unstable" in advice.reason

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
# CLI end-to-end
# ---------------------------------------------------------------------------

class TestCLI:
    def test_from_losses(self, tmp_path):
        output = tmp_path / "report.json"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--from-losses", "2.5,2.3,2.1,1.9,1.8",
             "--output", str(output)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["total_points"] == 5
        assert data["loss_trend"] < 0
        assert "convergence" in data

    def test_from_file(self, tmp_path):
        metrics = [
            {"cycle": i, "train_loss": 2.5 - 0.1 * i, "valid_loss": 2.6 - 0.1 * i}
            for i in range(10)
        ]
        metrics_file = tmp_path / "run_metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        output = tmp_path / "report.json"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(metrics_file),
             "--output", str(output)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        data = json.loads(output.read_text())
        assert data["total_points"] == 10

    def test_from_jsonl_file(self, tmp_path):
        lines = [
            json.dumps({"cycle": i, "train_loss": 2.5 - 0.1 * i})
            for i in range(5)
        ]
        jsonl_file = tmp_path / "metrics.jsonl"
        jsonl_file.write_text("\n".join(lines))

        output = tmp_path / "report.json"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(jsonl_file),
             "--output", str(output)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

    def test_from_jsonl_file_merges_duplicate_step_records(self, tmp_path):
        lines = [
            json.dumps({"step": 1, "loss_train": 2.5}),
            json.dumps({"step": 1, "loss_train": 2.5, "loss_valid": 2.4}),
            json.dumps({"step": 2, "loss_train": 2.3, "loss_valid": 2.2}),
        ]
        jsonl_file = tmp_path / "metrics.jsonl"
        jsonl_file.write_text("\n".join(lines))

        output = tmp_path / "report.json"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(jsonl_file),
             "--output", str(output)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

        data = json.loads(output.read_text())
        assert data["total_points"] == 2

    def test_missing_file_exits_2(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/nonexistent/file.json"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2

    def test_no_args_shows_error(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True, text=True,
        )
        assert r.returncode != 0

    def test_target_loss(self, tmp_path):
        output = tmp_path / "report.json"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--from-losses", "2.5,2.3,2.1,1.9,1.8",
             "--target-loss", "1.0", "--output", str(output)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        data = json.loads(output.read_text())
        assert data["convergence"]["remaining_steps"] is not None

    def test_prints_recommendation_continue(self, tmp_path):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--from-losses", "2.5,2.3,2.1,1.9,1.8"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "CONTINUE" in r.stdout or "STOP" in r.stdout

    def test_prints_recommendation_stop(self):
        losses = ",".join(["1.0"] * 20)
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--from-losses", losses],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "STOP" in r.stdout or "CONTINUE" in r.stdout


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

class TestPublicAPIExports:
    def test_trajectory_classes_importable_from_package(self):
        from src.tg_lora import TrajectoryAnalyzer, TrajectoryPoint
        assert TrajectoryAnalyzer is not None
        assert TrajectoryPoint is not None

    def test_all_trajectory_classes_in___all__(self):
        import src.tg_lora as pkg
        for name in ["TrajectoryAnalyzer", "TrajectoryPoint", "ConvergenceEstimate",
                      "EarlyStopAdvice", "TrajectoryReport"]:
            assert name in pkg.__all__
