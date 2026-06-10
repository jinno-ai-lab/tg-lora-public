"""Tests for src/tg_lora/cycle_monitor.py — training cycle health monitor."""
from __future__ import annotations


import pytest

from src.tg_lora.cycle_monitor import (
    CycleMonitor,
    HealthReport,
)


class TestCycleMonitorInit:
    def test_defaults(self):
        m = CycleMonitor()
        assert m.patience == 5
        assert m.spike_threshold == 2.0
        assert m.nan_detection is True

    def test_custom(self):
        m = CycleMonitor(patience=10, spike_threshold=3.0, nan_detection=False)
        assert m.patience == 10
        assert m.spike_threshold == 3.0
        assert m.nan_detection is False

    def test_invalid_patience(self):
        with pytest.raises(ValueError, match="patience"):
            CycleMonitor(patience=0)

    def test_invalid_spike_threshold(self):
        with pytest.raises(ValueError, match="spike_threshold"):
            CycleMonitor(spike_threshold=0)


class TestDivergenceDetection:
    def test_healthy_on_few_cycles(self):
        m = CycleMonitor()
        report = m.update({"train_loss": 1.0, "valid_loss": 0.9})
        assert report.status == "healthy"
        assert not report.divergence.detected

    def test_loss_spike_detected(self):
        m = CycleMonitor(spike_threshold=2.0)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": 2.5})
        assert report.divergence.detected
        assert report.divergence.metric == "train_loss"
        assert report.divergence.severity == "high"
        assert report.status == "divergent"

    def test_no_spike_below_threshold(self):
        m = CycleMonitor(spike_threshold=2.0)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": 1.5})
        assert not report.divergence.detected

    def test_nan_train_loss(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": float("nan")})
        assert report.divergence.detected
        assert report.divergence.severity == "critical"
        assert report.status == "divergent"

    def test_nan_valid_loss(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0, "valid_loss": 0.9})
        report = m.update({"train_loss": 0.95, "valid_loss": float("nan")})
        assert report.divergence.detected
        assert report.divergence.metric == "valid_loss"
        assert report.divergence.severity == "critical"

    def test_inf_loss(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": float("inf")})
        assert report.divergence.detected
        assert report.divergence.severity == "critical"

    def test_nan_detection_disabled(self):
        m = CycleMonitor(nan_detection=False)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": float("nan")})
        assert not report.divergence.detected


class TestStagnationDetection:
    def test_no_stagnation_within_patience(self):
        m = CycleMonitor(patience=3)
        m.update({"train_loss": 1.0, "valid_loss": 0.9})
        m.update({"train_loss": 0.95, "valid_loss": 0.85})
        m.update({"train_loss": 0.9, "valid_loss": 0.88})
        report = m.update({"train_loss": 0.88, "valid_loss": 0.87})
        assert not report.stagnation.detected

    def test_stagnation_after_patience(self):
        m = CycleMonitor(patience=3)
        best = 0.8
        m.update({"train_loss": 1.0, "valid_loss": best})
        m.update({"train_loss": 0.9, "valid_loss": 0.85})
        m.update({"train_loss": 0.88, "valid_loss": 0.83})
        m.update({"train_loss": 0.86, "valid_loss": 0.82})
        report = m.update({"train_loss": 0.85, "valid_loss": 0.81})
        assert report.stagnation.detected
        assert report.stagnation.cycles_without_improvement >= 3
        assert report.status == "stagnant"

    def test_improvement_resets_counter(self):
        m = CycleMonitor(patience=3)
        m.update({"train_loss": 1.0, "valid_loss": 0.9})
        m.update({"train_loss": 0.9, "valid_loss": 0.85})
        m.update({"train_loss": 0.88, "valid_loss": 0.83})
        report = m.update({"train_loss": 0.85, "valid_loss": 0.80})  # new best
        assert not report.stagnation.detected


class TestRecommendations:
    def test_nan_recommendations(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": float("nan")})
        assert "rollback: NaN/Inf detected" in report.recommendations
        assert "reduce_lr" in " ".join(report.recommendations)

    def test_spike_recommendations(self):
        m = CycleMonitor(spike_threshold=2.0)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": 3.0})
        assert "reduce_lr" in " ".join(report.recommendations)

    def test_stagnation_recommendations(self):
        m = CycleMonitor(patience=2)
        m.update({"train_loss": 1.0, "valid_loss": 0.5})
        m.update({"train_loss": 0.9, "valid_loss": 0.6})
        report = m.update({"train_loss": 0.85, "valid_loss": 0.7})
        assert "increase_K" in " ".join(report.recommendations)

    def test_healthy_no_recommendations(self):
        m = CycleMonitor()
        report = m.update({"train_loss": 1.0, "valid_loss": 0.9})
        assert report.recommendations == []


class TestHealthSummary:
    def test_healthy_summary(self):
        m = CycleMonitor()
        m.update({"train_loss": 1.0, "valid_loss": 0.9})
        summary = m.health_summary()
        assert summary["status"] == "healthy"
        assert summary["cycle_count"] == 1
        assert summary["best_loss"] == 0.9

    def test_divergent_summary(self):
        m = CycleMonitor()
        m.update({"train_loss": 1.0})
        m.update({"train_loss": float("nan")})
        summary = m.health_summary()
        assert summary["status"] == "divergent"
        assert summary["divergence"]["detected"]

    def test_stagnant_summary(self):
        m = CycleMonitor(patience=2)
        m.update({"train_loss": 1.0, "valid_loss": 0.5})
        m.update({"train_loss": 0.9, "valid_loss": 0.6})
        m.update({"train_loss": 0.85, "valid_loss": 0.7})
        summary = m.health_summary()
        assert summary["status"] == "stagnant"
        assert summary["stagnation"]["detected"]


class TestHistory:
    def test_history_recorded(self):
        m = CycleMonitor()
        m.update({"train_loss": 1.0})
        m.update({"train_loss": 0.9})
        assert len(m.history) == 2
        assert m.history[0]["train_loss"] == 1.0

    def test_history_is_copy(self):
        m = CycleMonitor()
        m.update({"train_loss": 1.0})
        h = m.history
        h.clear()
        assert len(m.history) == 1


class TestHealthReport:
    def test_default_fields(self):
        r = HealthReport()
        assert r.status == "healthy"
        assert r.recommendations == []
        assert r.cycle_count == 0

    def test_with_data(self):
        r = HealthReport(
            status="divergent",
            recommendations=["rollback"],
            cycle_count=5,
        )
        assert r.status == "divergent"
        assert len(r.recommendations) == 1


class TestNearZeroLossSpikeSuppression:
    """Spike detection should not false-positive when losses are near zero."""

    def test_no_false_spike_at_near_zero_loss(self):
        m = CycleMonitor(spike_threshold=2.0)
        m.update({"train_loss": 1e-8})
        report = m.update({"train_loss": 1e-3})
        assert not report.divergence.detected, (
            "Spike detected from 1e-8 to 1e-3 — ratio is 1e5 but absolute "
            "change is negligible"
        )

    def test_spike_still_detected_at_normal_scale(self):
        m = CycleMonitor(spike_threshold=2.0)
        m.update({"train_loss": 0.1})
        report = m.update({"train_loss": 0.5})
        assert report.divergence.detected
        assert report.divergence.severity == "high"

    def test_no_spike_at_zero_prev_loss(self):
        m = CycleMonitor(spike_threshold=2.0)
        m.update({"train_loss": 0.0})
        report = m.update({"train_loss": 10.0})
        assert not report.divergence.detected

    def test_boundary_just_above_eps(self):
        m = CycleMonitor(spike_threshold=2.0)
        m.update({"train_loss": 2e-6})
        report = m.update({"train_loss": 1e-2})
        assert report.divergence.detected
