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


class TestGradNormDivergence:
    """A non-finite ``grad_norm`` is a divergence precursor the monitor's
    docstring promises to catch ("NaN gradients") and the ``nan_detection``
    flag gates.

    Gradient explosion that ``clip_grad_norm_`` bounded leaves the *forward*
    loss finite (the clipped step stayed small) while the *pre-clip* grad norm
    the producer records is ``inf``/``NaN``. ``detect_divergence`` previously
    read ``grad_norm`` in ``update`` and threw it away, checking only the
    losses — so such a cycle reported ``healthy``. The ``advise_training`` CLI
    streams the real producer's ``grad_norm`` straight into the monitor and
    gates its exit-code-2 / ``critical`` automation on this signal, so the gap
    was operator-visible: a diverging run reported healthy (exit 0).
    """

    def test_nan_grad_norm_detected_with_finite_loss(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0, "grad_norm": 0.5})
        report = m.update({"train_loss": 0.95, "grad_norm": float("nan")})
        assert report.divergence.detected
        assert report.divergence.metric == "grad_norm"
        assert report.divergence.severity == "critical"
        assert report.status == "divergent"

    def test_inf_grad_norm_detected_with_finite_loss(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0, "grad_norm": 0.5})
        report = m.update({"train_loss": 0.95, "grad_norm": float("inf")})
        assert report.divergence.detected
        assert report.divergence.metric == "grad_norm"
        assert report.divergence.severity == "critical"

    def test_finite_grad_norm_not_flagged(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0, "grad_norm": 0.5})
        report = m.update({"train_loss": 0.95, "grad_norm": 1.2})
        assert not report.divergence.detected

    def test_nan_grad_norm_recommends_rollback(self):
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0, "grad_norm": 0.5})
        report = m.update({"train_loss": 0.95, "grad_norm": float("inf")})
        assert "rollback: NaN/Inf detected" in report.recommendations
        assert "reduce_lr" in " ".join(report.recommendations)

    def test_nan_grad_norm_respects_nan_detection_disabled(self):
        m = CycleMonitor(nan_detection=False)
        m.update({"train_loss": 1.0, "grad_norm": 0.5})
        report = m.update({"train_loss": 0.95, "grad_norm": float("nan")})
        assert not report.divergence.detected

    def test_nan_train_loss_precedence_over_grad_norm(self):
        """When both train_loss and grad_norm are non-finite, the loss signal
        (checked first) wins — documents the loop order, not a regression."""
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0, "grad_norm": 0.5})
        report = m.update({"train_loss": float("nan"), "grad_norm": float("nan")})
        assert report.divergence.detected
        assert report.divergence.metric == "train_loss"

    def test_grad_norm_absent_does_not_crash_or_flag(self):
        """A cycle with no grad_norm key (older/loss-only fixtures) must behave
        exactly as before — no crash, no spurious flag."""
        m = CycleMonitor(nan_detection=True)
        m.update({"train_loss": 1.0})
        report = m.update({"train_loss": 0.95})
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


class TestStagnationCurrentLossSelection:
    """``detect_stagnation().current_value`` must report the same loss that
    ``update`` selects for best-tracking — ``valid_loss if valid_loss is not
    None else train_loss`` (the rule at the top of ``update``). The stagnation
    path previously used ``last.get("valid_loss") or last.get("train_loss")``,
    which treats a legitimate ``valid_loss == 0.0`` as falsy and silently falls
    through to ``train_loss``: the surfaced ``current_value`` (consumed by
    ``health_summary`` and the advisor/``advise_training`` reporting path) then
    reports the wrong loss. This is the stagnation-side companion to
    ``TestNearZeroLossSpikeSuppression`` — a zero loss is a real value, not a
    sentinel for "missing".
    """

    def test_zero_valid_loss_is_not_swallowed(self):
        """``valid_loss == 0.0`` (perfect/near-perfect loss — e.g. the proxy
        memorize task) must be reported as ``current_value``, not replaced by
        ``train_loss`` via a falsy ``or``."""
        m = CycleMonitor(patience=2)
        m.update({"train_loss": 1.50, "valid_loss": 0.0})  # best = 0.0
        m.update({"train_loss": 1.45, "valid_loss": 0.0})  # cycles_since_best = 1
        report = m.update({"train_loss": 1.43, "valid_loss": 0.0})  # -> 2: detect
        assert report.stagnation.detected
        assert report.stagnation.current_value == 0.0, (
            f"valid_loss=0.0 must be reported as current_value, got "
            f"{report.stagnation.current_value} (train_loss leaked through `or`)"
        )

    @pytest.mark.parametrize(
        "valid_loss, train_loss",
        [
            (0.0, 1.43),  # the falsy-swallow case (0.0 is a real value)
            (0.5, 1.0),  # normal scale, truthy
            (1e-12, 2.0),  # tiny-but-nonzero is truthy, still must round-trip
        ],
    )
    def test_current_value_matches_best_loss_rule(self, valid_loss, train_loss):
        """``current_value`` must equal the same selection ``update`` applies
        for best-tracking, magnitude- and falsy-invariant across inputs."""
        m = CycleMonitor(patience=2)
        m.update({"train_loss": 10.0, "valid_loss": valid_loss})  # best
        m.update({"train_loss": 9.0, "valid_loss": valid_loss + 0.1})  # csb = 1
        report = m.update({"train_loss": train_loss, "valid_loss": valid_loss})
        assert report.stagnation.detected
        expected = valid_loss if valid_loss is not None else train_loss
        assert report.stagnation.current_value == pytest.approx(expected), (
            f"valid_loss={valid_loss}, train_loss={train_loss}: expected "
            f"current_value={expected}, got {report.stagnation.current_value}"
        )

    def test_falls_back_to_train_loss_when_valid_absent(self):
        """When ``valid_loss`` is genuinely absent (key missing -> None),
        ``current_value`` falls back to ``train_loss``. Guards that the fix
        does not break the legitimate missing-valid path."""
        m = CycleMonitor(patience=2)
        m.update({"train_loss": 1.00})  # best tracked from train_loss
        m.update({"train_loss": 1.01})
        report = m.update({"train_loss": 1.02})
        assert report.stagnation.detected
        assert report.stagnation.current_value == pytest.approx(1.02)


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
