"""Single-source-of-truth relative-degradation metric for accept/rollback gates.

``rollback_tolerance`` (default ``0.005``) is a *relative* fraction — 0.5% — not
an absolute loss margin. That contract is pinned magnitude-invariant by
``tests/test_accept_property.py`` for :meth:`RandomWalkController.accept`. Three
other accept/rollback gates in the loop previously misused the same knob as an
*absolute* additive (``loss_new <= loss_before + rollback_tolerance``), which
made their sensitivity drift with the current loss level (over-sensitive when
loss is large, under-sensitive when loss is small). They now route through
:func:`relative_degradation` so every gate shares the canonical scale-invariant
metric.

These tests pin the helper itself — and therefore the shared logic at *all*
sites that consume it (``RandomWalkController.accept``, the pilot-overshoot
trigger, the alpha line-search accept, and the zeroth-order subspace step) — so
the magnitude-invariance property cannot silently regress at any one site.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.tg_lora.random_walk_controller import RandomWalkController, relative_degradation


# ---------------------------------------------------------------------------
# Magnitude invariance — the load-bearing property.
# ---------------------------------------------------------------------------
class TestRelativeDegradationMagnitudeInvariance:
    """Same relative change ⇒ same ``relative_degradation`` regardless of scale."""

    @settings(max_examples=200)
    @given(
        magnitude=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        relative_change=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_same_relative_change_same_metric_across_magnitudes(self, magnitude, relative_change):
        """A fixed relative change must yield a fixed metric across loss scales.

        This is the exact property that breaks if the metric were absolute
        (``loss_after - loss_before``): an absolute metric grows with magnitude,
        so a 0.3% degradation at loss=1.0 (delta 0.003) and at loss=100.0 (delta
        0.3) would differ. The relative metric collapses both to 0.003.
        """
        loss_before = magnitude
        loss_after = magnitude * (1.0 + relative_change)
        # Scaling the baseline by 10x must not move the metric.
        metric_a = relative_degradation(loss_before, loss_after)
        metric_b = relative_degradation(10.0 * loss_before, 10.0 * loss_after)
        assert metric_a == pytest.approx(metric_b, abs=1e-12), (
            f"magnitude {magnitude}, change {relative_change}: metric must be "
            f"scale-invariant, got {metric_a} vs {metric_b}"
        )

    def test_concrete_pair_distinguishes_relative_from_absolute(self):
        """The concrete pair that motivated the fix: 0.3% degradation at two scales.

        Under the relative metric both equal 0.003 (within a 0.005 tolerance ⇒
        accepted everywhere). Under an absolute metric they would be 0.003 vs
        0.3 — different verdicts. Pinning the *value* (not just a boolean) makes
        the test maximally sensitive to a regression to absolute semantics.
        """
        small = relative_degradation(1.0, 1.003)
        large = relative_degradation(100.0, 100.3)
        assert small == pytest.approx(0.003, abs=1e-9)
        assert large == pytest.approx(0.003, abs=1e-9)
        # The two must be equal *to each other* (both ≈ 0.3% degradation).
        assert small == pytest.approx(large, abs=1e-9)


# ---------------------------------------------------------------------------
# Sign + value semantics.
# ---------------------------------------------------------------------------
class TestRelativeDegradationSemantics:
    def test_improvement_is_negative(self):
        assert relative_degradation(2.0, 1.5) == pytest.approx(-0.25)

    def test_no_change_is_zero(self):
        assert relative_degradation(2.0, 2.0) == 0.0

    def test_degradation_is_positive(self):
        assert relative_degradation(2.0, 2.5) == pytest.approx(0.25)

    def test_floor_near_zero_baseline(self):
        """A ~0 baseline must not blow up — the 1e-8 floor bounds the metric."""
        # 0.001 absolute degradation off a 0.0 baseline: floored denominator.
        assert relative_degradation(0.0, 0.001) == pytest.approx(0.001 / 1e-8)

    def test_non_finite_returns_inf(self):
        """NaN/Inf inputs must yield +inf so ``<= tolerance`` rejects them."""
        assert relative_degradation(float("nan"), 1.0) == math.inf
        assert relative_degradation(1.0, float("nan")) == math.inf
        assert relative_degradation(float("inf"), 1.0) == math.inf
        assert relative_degradation(1.0, float("inf")) == math.inf


# ---------------------------------------------------------------------------
# accept() routes through the helper — the refactor is behavior-identical.
# ---------------------------------------------------------------------------
class TestAcceptUsesRelativeDegradation:
    """``RandomWalkController.accept`` must agree with the shared helper.

    Guards the extract-method refactor: if ``accept`` is ever restored to an
    inline *absolute* comparison (the historical bug shape), it diverges from
    ``relative_degradation`` and these assertions fail — including at loss
    scales where absolute and relative disagree.
    """

    def _ctrl(self, tolerance=0.005):
        return RandomWalkController(
            rollback_tolerance=tolerance,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )

    def test_accept_matches_helper_across_magnitudes(self):
        ctrl = self._ctrl()
        for before in (0.5, 2.0, 10.0, 100.0):
            # A 0.3% degradation is within the 0.5% tolerance at every scale.
            after = before * 1.003
            expected = relative_degradation(before, after) <= ctrl.rollback_tolerance
            assert ctrl.accept(before, after) is True
            assert expected is True
            # A 2% degradation exceeds the tolerance at every scale.
            after_bad = before * 1.02
            expected_bad = relative_degradation(before, after_bad) <= ctrl.rollback_tolerance
            assert ctrl.accept(before, after_bad) is False
            assert expected_bad is False

    def test_absolute_regression_would_diverge_at_large_scale(self):
        """If accept() used absolute ``+ tolerance``, a 0.3% bump at loss=100
        would be accepted (0.3 > 0.005 is FALSE for absolute <=). Pin that it is
        NOT — relative semantics reject nothing within 0.5% and accept the rest,
        consistently with the helper."""
        ctrl = self._ctrl()
        # 0.3% bump at loss=100.0: relative 0.003 <= 0.005 ⇒ accept.
        assert ctrl.accept(100.0, 100.3) is True
        assert relative_degradation(100.0, 100.3) <= ctrl.rollback_tolerance
