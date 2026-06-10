"""Property-based tests for TrainingAdvisor health status classification.

Uses Hypothesis to generate arbitrary combinations of HealthReport and
TrajectoryReport states, verifying that _determine_health always produces
a valid status and that the classification invariants hold.

Regression guard for the spike→warning bug (was incorrectly critical).
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.tg_lora.cycle_monitor import (
    DivergenceReport,
    HealthReport,
    StagnationReport,
)
from src.tg_lora.training_advisor import TrainingAdvisor
from src.tg_lora.trajectory import (
    ConvergenceEstimate,
    EarlyStopAdvice,
    TrajectoryReport,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

divergence_severity = st.sampled_from(["", "high", "critical"])

divergence_report = st.builds(
    DivergenceReport,
    detected=st.booleans(),
    metric=st.text(min_size=0, max_size=20),
    severity=divergence_severity,
    current_value=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
    threshold=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
)

stagnation_report = st.builds(
    StagnationReport,
    detected=st.booleans(),
    cycles_without_improvement=st.integers(min_value=0, max_value=100),
    best_value=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    current_value=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
)

health_report = st.builds(
    HealthReport,
    status=st.sampled_from(["healthy", "divergent", "stagnant"]),
    divergence=divergence_report,
    stagnation=stagnation_report,
    cycle_count=st.integers(min_value=0, max_value=1000),
)

convergence_estimate = st.builds(
    ConvergenceEstimate,
    converged=st.booleans(),
    remaining_steps=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
    predicted_final_loss=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    convergence_rate=st.floats(min_value=-1.0, max_value=1.0),
    confidence=st.floats(min_value=0.0, max_value=1.0),
)

early_stop_advice = st.builds(
    EarlyStopAdvice,
    should_stop=st.booleans(),
    reason=st.text(min_size=0, max_size=50),
    estimated_gain_from_continuing=st.floats(min_value=-1.0, max_value=1.0),
    optimal_cycle=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
)

trajectory_report = st.builds(
    TrajectoryReport,
    total_points=st.integers(min_value=0, max_value=100),
    convergence=convergence_estimate,
    early_stop=early_stop_advice,
    loss_trend=st.floats(min_value=-10.0, max_value=10.0),
    volatility=st.floats(min_value=0.0, max_value=10.0),
    anomaly_detected=st.booleans(),
)

# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

VALID_STATUSES: set[str] = {"healthy", "warning", "critical"}


class TestDetermineHealthInvariants:
    """Property-based tests for TrainingAdvisor._determine_health."""

    @given(health=health_report, trajectory=st.one_of(st.none(), trajectory_report))
    @settings(max_examples=500)
    def test_result_is_always_valid_status(self, health, trajectory):
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result in VALID_STATUSES

    @given(health=health_report, trajectory=st.one_of(st.none(), trajectory_report))
    @settings(max_examples=500)
    def test_critical_divergence_implies_critical(self, health, trajectory):
        """When divergence severity is 'critical', overall must be 'critical'."""
        if not (health.divergence.detected and health.divergence.severity == "critical"):
            return
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result == "critical"

    @given(health=health_report, trajectory=trajectory_report)
    @settings(max_examples=500)
    def test_early_stop_implies_critical(self, health, trajectory):
        """When early_stop.should_stop is True and training has NOT converged,
        overall must be 'critical'. Convergence-triggered stop is a success,
        not an error."""
        if not trajectory.early_stop.should_stop:
            return
        if trajectory.convergence.converged:
            return  # convergence is a successful outcome, not critical
        # But critical divergence takes precedence — if that's also present,
        # we still get critical (which satisfies the invariant either way).
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result == "critical"

    @given(health=health_report, trajectory=st.one_of(st.none(), trajectory_report))
    @settings(max_examples=500)
    def test_high_severity_spike_never_critical(self, health, trajectory):
        """Regression: high-severity divergence (spike) must produce 'warning', not 'critical'.

        This was the bug fixed in 02916f7: spike was incorrectly mapped to
        'critical' instead of 'warning'.
        """
        if not health.divergence.detected:
            return
        if health.divergence.severity == "critical":
            return  # critical severity correctly maps to critical
        # If there's an early_stop, that independently makes it critical,
        # so only assert when early_stop is not active
        if trajectory is not None and trajectory.early_stop.should_stop:
            return
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result == "warning", (
            f"Expected 'warning' for high-severity divergence, got '{result}'"
        )

    @given(
        health=st.builds(
            HealthReport,
            divergence=st.builds(DivergenceReport, detected=st.just(False), severity=st.just("")),
            stagnation=st.builds(StagnationReport, detected=st.just(False)),
        ),
        trajectory=st.builds(
            TrajectoryReport,
            total_points=st.integers(min_value=1, max_value=10),
            convergence=convergence_estimate,
            early_stop=st.builds(EarlyStopAdvice, should_stop=st.just(False)),
            loss_trend=st.floats(min_value=-10.0, max_value=-0.001),
            volatility=st.floats(min_value=0.01, max_value=10.0),
            anomaly_detected=st.just(False),
        ),
    )
    @settings(max_examples=200)
    def test_no_signals_and_downward_trend_implies_healthy(self, health, trajectory):
        """No divergence, no stagnation, no anomaly, downward trend → healthy."""
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result == "healthy"

    @given(
        health=st.builds(
            HealthReport,
            divergence=st.builds(
                DivergenceReport, detected=st.just(True), severity=st.just("high")
            ),
            stagnation=stagnation_report,
        ),
        trajectory=st.one_of(st.none(), trajectory_report),
    )
    @settings(max_examples=300)
    def test_any_non_critical_divergence_at_least_warning(self, health, trajectory):
        """Any detected divergence (non-critical severity) → at least 'warning'."""
        if trajectory is not None and trajectory.early_stop.should_stop:
            return  # early_stop would bump to critical, still valid
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result in ("warning", "critical")

    @given(
        health=st.builds(
            HealthReport,
            divergence=st.builds(DivergenceReport, detected=st.just(False), severity=st.just("")),
            stagnation=st.builds(StagnationReport, detected=st.just(True)),
        ),
        trajectory=st.builds(
            TrajectoryReport,
            total_points=st.integers(min_value=1, max_value=10),
            convergence=convergence_estimate,
            early_stop=st.builds(EarlyStopAdvice, should_stop=st.just(False)),
            anomaly_detected=st.just(False),
            loss_trend=st.floats(min_value=-10.0, max_value=-0.001),
            volatility=st.floats(min_value=0.01, max_value=10.0),
        ),
    )
    @settings(max_examples=200)
    def test_stagnation_without_divergence_is_warning(self, health, trajectory):
        """Stagnation without divergence or anomaly → 'warning'."""
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result == "warning"

    @given(
        health=st.builds(
            HealthReport,
            divergence=st.builds(DivergenceReport, detected=st.just(False), severity=st.just("")),
            stagnation=st.builds(StagnationReport, detected=st.just(False)),
        ),
        trajectory=st.builds(
            TrajectoryReport,
            total_points=st.integers(min_value=1, max_value=10),
            convergence=st.builds(
                ConvergenceEstimate,
                converged=st.just(True),
                remaining_steps=st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
                predicted_final_loss=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
                convergence_rate=st.floats(min_value=-1.0, max_value=1.0),
                confidence=st.floats(min_value=0.0, max_value=1.0),
            ),
            early_stop=st.builds(EarlyStopAdvice, should_stop=st.just(True)),
            loss_trend=st.floats(min_value=-10.0, max_value=-0.001),
            volatility=st.floats(min_value=0.01, max_value=10.0),
            anomaly_detected=st.just(False),
        ),
    )
    @settings(max_examples=300)
    def test_converged_early_stop_never_critical(self, health, trajectory):
        """Regression: convergence + early_stop must never produce 'critical'.

        This was the bug fixed in 74e00bb: converged training was
        misclassified as critical because early_stop.should_stop triggered
        the "return critical" path without checking convergence first.
        """
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result != "critical", (
            f"Converged training with early_stop should not be critical, "
            f"got '{result}'"
        )

    @given(
        health=st.builds(
            HealthReport,
            divergence=st.builds(
                DivergenceReport, detected=st.just(True), severity=st.just("critical")
            ),
            stagnation=stagnation_report,
        ),
        trajectory=st.builds(
            TrajectoryReport,
            total_points=st.integers(min_value=1, max_value=10),
            convergence=st.builds(
                ConvergenceEstimate,
                converged=st.just(True),
                remaining_steps=st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
                predicted_final_loss=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
                convergence_rate=st.floats(min_value=-1.0, max_value=1.0),
                confidence=st.floats(min_value=0.0, max_value=1.0),
            ),
            early_stop=st.builds(EarlyStopAdvice, should_stop=st.just(True)),
            anomaly_detected=st.booleans(),
        ),
    )
    @settings(max_examples=200)
    def test_critical_divergence_overrides_convergence(self, health, trajectory):
        """Even with convergence + early_stop, critical divergence wins."""
        advisor = TrainingAdvisor()
        result = advisor._determine_health(health, trajectory)
        assert result == "critical"
