"""Property-based tests for health status classification in TrainingAdvisor.

Uses Hypothesis to verify invariants of _determine_health across the
full parameter space of spike_threshold, stagnation_patience, loss
sequences, and NaN/Inf inputs. Prevents regressions of the bug where
loss spikes were misclassified as critical instead of warning.
"""
from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.tg_lora.training_advisor import (
    AdvisoryReport,
    AdvisorConfig,
    TrainingAdvisor,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_pos_loss = st.floats(
    min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
)
spike_threshold = st.floats(min_value=1.5, max_value=10.0, allow_nan=False)
stagnation_patience = st.integers(min_value=2, max_value=20)


# ---------------------------------------------------------------------------
# Invariant P1: NaN/Inf always produces "critical"
# ---------------------------------------------------------------------------


class TestNaNInfAlwaysCritical:
    """Non-finite train_loss must always produce overall_health == 'critical'."""

    @given(
        loss=st.one_of(
            st.just(float("nan")),
            st.just(float("inf")),
            st.just(float("-inf")),
        ),
        threshold=spike_threshold,
        patience=stagnation_patience,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_nan_inf_yields_critical(self, loss, threshold, patience):
        advisor = TrainingAdvisor(
            config=AdvisorConfig(
                spike_threshold=threshold,
                stagnation_patience=patience,
            )
        )
        advisor.evaluate(0, train_loss=1.0)
        report = advisor.evaluate(1, train_loss=loss)
        assert report.overall_health == "critical", (
            f"Non-finite loss {loss} produced {report.overall_health}, expected 'critical'"
        )

    @given(
        losses=st.lists(finite_pos_loss, min_size=2, max_size=10),
        threshold=spike_threshold,
        patience=stagnation_patience,
    )
    @settings(max_examples=30)
    def test_nan_mid_sequence_yields_critical(self, losses, threshold, patience):
        advisor = TrainingAdvisor(
            config=AdvisorConfig(
                spike_threshold=threshold,
                stagnation_patience=patience,
            )
        )
        for i, l in enumerate(losses):
            advisor.evaluate(i, train_loss=l)
        report = advisor.evaluate(len(losses), train_loss=float("nan"))
        assert report.overall_health == "critical"


# ---------------------------------------------------------------------------
# Invariant P2: Spike (ratio >= threshold) always produces "warning", never "critical"
# Only guaranteed in short sequences where trajectory early-stop hasn't triggered.
# ---------------------------------------------------------------------------


class TestSpikeAlwaysWarning:
    """Loss spikes must produce 'warning', never 'critical' (short sequences)."""

    @given(
        prev_loss=finite_pos_loss,
        threshold=spike_threshold,
        patience=stagnation_patience,
    )
    @settings(max_examples=60)
    def test_spike_yields_warning_not_critical(self, prev_loss, threshold, patience):
        """In a 2-cycle sequence, spike must be warning (no trajectory yet)."""
        spiked_loss = prev_loss * threshold * 1.1
        advisor = TrainingAdvisor(
            config=AdvisorConfig(
                spike_threshold=threshold,
                stagnation_patience=patience,
            )
        )
        advisor.evaluate(0, train_loss=prev_loss)
        report = advisor.evaluate(1, train_loss=spiked_loss)
        assert report.overall_health == "warning", (
            f"Spike ratio {spiked_loss / prev_loss:.2f} (threshold {threshold}) "
            f"produced {report.overall_health}, expected 'warning'"
        )

    @given(
        base_loss=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        multiplier=st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
        threshold=spike_threshold,
    )
    @settings(max_examples=50)
    def test_spike_invariant_to_absolute_loss(self, base_loss, multiplier, threshold):
        """Spike classification depends on ratio, not absolute loss value."""
        spiked = base_loss * multiplier
        ratio = spiked / base_loss
        advisor = TrainingAdvisor(config=AdvisorConfig(spike_threshold=threshold))
        advisor.evaluate(0, train_loss=base_loss)
        report = advisor.evaluate(1, train_loss=spiked)

        if ratio >= threshold:
            assert report.overall_health != "healthy", (
                f"Ratio {ratio:.2f} >= threshold {threshold} but health is 'healthy'"
            )
            assert report.overall_health == "warning", (
                f"Spike produced {report.overall_health} instead of 'warning'"
            )


# ---------------------------------------------------------------------------
# Invariant P3: Monotonically decreasing finite losses produce "healthy"
# ---------------------------------------------------------------------------


class TestHealthyProgression:
    """Monotonically decreasing finite losses should produce 'healthy'."""

    @given(
        start=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
        decay=st.floats(min_value=0.8, max_value=0.99, allow_nan=False),
        n_cycles=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=40)
    def test_monotonic_decrease_is_healthy(self, start, decay, n_cycles):
        advisor = TrainingAdvisor()
        report = AdvisoryReport(overall_health="healthy")
        for i in range(n_cycles):
            loss = start * (decay ** i)
            report = advisor.evaluate(i, train_loss=loss)
        assert report.overall_health == "healthy", (
            f"Monotonically decreasing losses produced {report.overall_health}"
        )

    @given(
        losses=st.lists(
            st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
            min_size=3, max_size=8,
        ).filter(lambda ls: all(ls[i] > ls[i + 1] for i in range(len(ls) - 1)))
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
    def test_strictly_decreasing_sequence_healthy(self, losses):
        advisor = TrainingAdvisor()
        for i, l in enumerate(losses):
            report = advisor.evaluate(i, train_loss=l)
        assert report.overall_health == "healthy"


# ---------------------------------------------------------------------------
# Invariant P4: Health status is always a valid literal
# ---------------------------------------------------------------------------


class TestHealthStatusIsValid:
    """_determine_health must always return a valid HealthStatus literal."""

    VALID = {"healthy", "warning", "critical"}

    @given(
        train_losses=st.lists(
            st.one_of(
                finite_pos_loss,
                st.just(float("nan")),
            ),
            min_size=1, max_size=15,
        ),
        threshold=spike_threshold,
        patience=stagnation_patience,
    )
    @settings(max_examples=80, suppress_health_check=[HealthCheck.filter_too_much])
    def test_always_valid_status(self, train_losses, threshold, patience):
        advisor = TrainingAdvisor(
            config=AdvisorConfig(
                spike_threshold=threshold,
                stagnation_patience=patience,
            )
        )
        report = AdvisoryReport(overall_health="healthy")
        for i, loss in enumerate(train_losses):
            report = advisor.evaluate(i, train_loss=loss)
        assert report.overall_health in self.VALID, (
            f"Invalid health status: {report.overall_health!r}"
        )


# ---------------------------------------------------------------------------
# Invariant P5: Below-threshold ratio never triggers divergence
# ---------------------------------------------------------------------------


class TestBelowThresholdNoDivergence:
    """Loss ratio strictly below spike_threshold must not trigger divergence."""

    @given(
        prev_loss=finite_pos_loss,
        threshold=spike_threshold,
    )
    @settings(max_examples=60)
    def test_below_threshold_no_divergence(self, prev_loss, threshold):
        next_loss = prev_loss * threshold * 0.9  # 10% below threshold
        advisor = TrainingAdvisor(config=AdvisorConfig(spike_threshold=threshold))
        advisor.evaluate(0, train_loss=prev_loss)
        report = advisor.evaluate(1, train_loss=next_loss)
        if report.cycle_health is not None:
            assert not report.cycle_health.divergence.detected, (
                f"Loss ratio {next_loss / prev_loss:.2f} < threshold {threshold} "
                f"but divergence was detected"
            )

    @given(
        prev_loss=finite_pos_loss,
        threshold=spike_threshold,
    )
    @settings(max_examples=40)
    def test_at_threshold_triggers_divergence(self, prev_loss, threshold):
        # Use a small multiplier above threshold to avoid floating-point edge cases
        # where prev_loss * threshold / prev_loss != threshold exactly
        next_loss = prev_loss * threshold * (1 + 1e-9)
        advisor = TrainingAdvisor(config=AdvisorConfig(spike_threshold=threshold))
        advisor.evaluate(0, train_loss=prev_loss)
        report = advisor.evaluate(1, train_loss=next_loss)
        if report.cycle_health is not None:
            assert report.cycle_health.divergence.detected, (
                f"Loss ratio >= {threshold} but divergence not detected"
            )
            assert report.cycle_health.divergence.severity == "high", (
                f"Non-NaN spike severity should be 'high', got "
                f"{report.cycle_health.divergence.severity!r}"
            )


# ---------------------------------------------------------------------------
# Invariant P6: Stagnation produces at least "warning" (may be "critical" via early-stop)
# ---------------------------------------------------------------------------


class TestStagnationYieldsWarningOrCritical:
    """After patience+ cycles without improvement, health must be warning or critical."""

    @given(
        loss=finite_pos_loss,
        patience=stagnation_patience,
        threshold=spike_threshold,
    )
    @settings(max_examples=40)
    def test_stagnation_not_healthy(self, loss, patience, threshold):
        """Stagnation must never produce 'healthy'."""
        advisor = TrainingAdvisor(
            config=AdvisorConfig(
                spike_threshold=threshold,
                stagnation_patience=patience,
            )
        )
        advisor.evaluate(0, train_loss=loss * 2.0)  # establish a best
        for i in range(1, patience + 2):
            report = advisor.evaluate(i, train_loss=loss)
        assert report.overall_health in ("warning", "critical"), (
            f"After {patience + 1} stagnant cycles, health is {report.overall_health}"
        )

    @given(
        loss=finite_pos_loss,
        patience=stagnation_patience,
        threshold=spike_threshold,
    )
    @settings(max_examples=30)
    def test_stagnation_detected_in_cycle_health(self, loss, patience, threshold):
        """Stagnation detection must be reflected in cycle_health."""
        advisor = TrainingAdvisor(
            config=AdvisorConfig(
                spike_threshold=threshold,
                stagnation_patience=patience,
            )
        )
        advisor.evaluate(0, train_loss=loss * 2.0)
        for i in range(1, patience + 2):
            report = advisor.evaluate(i, train_loss=loss)
        assert report.cycle_health is not None
        assert report.cycle_health.stagnation.detected, (
            f"After {patience + 1} stagnant cycles, stagnation not detected"
        )


# ---------------------------------------------------------------------------
# Invariant P7: NaN always overrides everything to "critical"
# ---------------------------------------------------------------------------


class TestHealthPriorityOrdering:
    """NaN divergence overrides any other signal to critical."""

    @given(
        base_loss=finite_pos_loss,
        patience=stagnation_patience,
    )
    @settings(max_examples=30)
    def test_nan_overrides_stagnation(self, base_loss, patience):
        advisor = TrainingAdvisor(
            config=AdvisorConfig(stagnation_patience=patience)
        )
        for i in range(patience + 2):
            advisor.evaluate(i, train_loss=base_loss)
        report = advisor.evaluate(patience + 2, train_loss=float("nan"))
        assert report.overall_health == "critical"

    @given(
        prev_loss=finite_pos_loss,
        threshold=spike_threshold,
    )
    @settings(max_examples=30)
    def test_spike_detected_as_high_severity(self, prev_loss, threshold):
        """Spike (non-NaN) must have severity 'high', never 'critical'."""
        spiked = prev_loss * threshold * 1.5
        advisor = TrainingAdvisor(config=AdvisorConfig(spike_threshold=threshold))
        advisor.evaluate(0, train_loss=prev_loss)
        report = advisor.evaluate(1, train_loss=spiked)
        assert report.cycle_health is not None
        assert report.cycle_health.divergence.detected
        assert report.cycle_health.divergence.severity == "high", (
            f"Spike severity should be 'high', got "
            f"{report.cycle_health.divergence.severity!r}"
        )


# ---------------------------------------------------------------------------
# Invariant P8: Spike classification is independent of stagnation_patience
# ---------------------------------------------------------------------------


class TestConfigIndependence:
    """Spike classification should be independent of stagnation_patience."""

    @given(
        prev=finite_pos_loss,
        threshold=spike_threshold,
        p1=stagnation_patience,
        p2=stagnation_patience,
    )
    @settings(max_examples=40)
    def test_spike_invariant_to_patience(self, prev, threshold, p1, p2):
        spiked = prev * threshold * 1.2
        adv1 = TrainingAdvisor(config=AdvisorConfig(spike_threshold=threshold, stagnation_patience=p1))
        adv2 = TrainingAdvisor(config=AdvisorConfig(spike_threshold=threshold, stagnation_patience=p2))
        adv1.evaluate(0, train_loss=prev)
        adv2.evaluate(0, train_loss=prev)
        r1 = adv1.evaluate(1, train_loss=spiked)
        r2 = adv2.evaluate(1, train_loss=spiked)
        assert r1.overall_health == r2.overall_health == "warning"


# ---------------------------------------------------------------------------
# Invariant P9: Converged training must never be "critical" without NaN/Inf
# ---------------------------------------------------------------------------


class TestConvergenceNeverCritical:
    """When convergence triggers early-stop, health must not be "critical".

    Convergence is a successful outcome. Only NaN/Inf divergence or
    non-convergent stagnation/instability should produce "critical".
    """

    @given(
        start=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
        decay=st.floats(min_value=0.7, max_value=0.95, allow_nan=False),
    )
    @settings(max_examples=40)
    def test_converged_never_critical(self, start, decay):
        """Exponentially decaying loss for enough cycles to trigger convergence."""
        advisor = TrainingAdvisor()
        for i in range(25):
            loss = start * (decay ** i)
            report = advisor.evaluate(i, train_loss=loss)
        if report.trajectory_summary is not None and report.trajectory_summary.get("converged"):
            assert report.overall_health != "critical", (
                f"Converged training produced 'critical' health: "
                f"start={start}, decay={decay}"
            )
