"""TASK-0045: Property-based tests for RandomWalkController.accept().

Uses hypothesis to verify that the relative-tolerance accept() logic behaves
consistently across loss-value magnitudes and handles edge cases correctly.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.tg_lora.random_walk_controller import RandomWalkController


# Shared controller with default rollback_tolerance=0.005
def _ctrl(tolerance=0.005):
    return RandomWalkController(rollback_tolerance=tolerance, k_explore_prob=0.0, n_explore_prob=0.0, beta_explore_prob=0.0, strategy_explore_prob=0.0, lr_explore_prob=0.0)


# --- TC-071-P19-01: loss_after <= loss_pilot always accepted ---


@settings(max_examples=200)
@given(
    loss_pilot=st.floats(
        min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
    improvement=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_accept_improvement_always_true(loss_pilot, improvement):
    """When loss_after <= loss_pilot, accept() must always return True."""
    loss_after = loss_pilot * (1.0 - improvement)
    # Guard against floating point producing values > pilot due to rounding
    loss_after = min(loss_after, loss_pilot)
    ctrl = _ctrl()
    assert ctrl.accept(loss_pilot, loss_after) is True


# --- TC-071-P19-02: NaN/Inf inputs always rejected ---


_nonfinite = st.one_of(
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
)


@settings(max_examples=200)
@given(loss_pilot=_nonfinite, loss_after=_nonfinite)
def test_accept_nonfinite_always_false(loss_pilot, loss_after):
    """NaN/Inf inputs must always cause accept() to return False."""
    ctrl = _ctrl()
    assert ctrl.accept(loss_pilot, loss_after) is False


@settings(max_examples=100)
@given(
    good=st.floats(
        min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
    bad=_nonfinite,
)
def test_accept_one_nonfinite_false(good, bad):
    """If either argument is non-finite, accept() returns False."""
    ctrl = _ctrl()
    assert ctrl.accept(good, bad) is False
    assert ctrl.accept(bad, good) is False


# --- TC-071-P19-03: Relative tolerance magnitude consistency ---


@settings(max_examples=200)
@given(
    magnitude=st.floats(
        min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
    relative_degradation=st.floats(
        min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False
    ),
)
def test_accept_magnitude_consistency(magnitude, relative_degradation):
    """Same relative degradation should give same accept/reject regardless of magnitude."""
    ctrl = _ctrl(tolerance=0.005)
    loss_pilot = magnitude
    loss_after = magnitude * (1.0 + relative_degradation)

    expected = relative_degradation <= ctrl.rollback_tolerance
    result = ctrl.accept(loss_pilot, loss_after)
    assert result == expected, (
        f"Magnitude {magnitude}: degradation={relative_degradation:.6f}, "
        f"tolerance={ctrl.rollback_tolerance}, expected={expected}, got={result}"
    )


# --- TC-071-P19-04: 1e-8 floor behavior ---


@settings(max_examples=200)
@given(
    pilot_near_floor=st.floats(
        min_value=1e-10, max_value=1e-6, allow_nan=False, allow_infinity=False
    ),
    small_worsening=st.floats(
        min_value=1e-12, max_value=1e-7, allow_nan=False, allow_infinity=False
    ),
)
def test_accept_near_floor_uses_absolute_denominator(pilot_near_floor, small_worsening):
    """Near 1e-8 floor, the denominator max(abs(pilot), 1e-8) prevents division issues."""
    ctrl = _ctrl()
    loss_after = pilot_near_floor + small_worsening

    # Should not crash or produce unexpected results
    result = ctrl.accept(pilot_near_floor, loss_after)
    assert isinstance(result, bool)

    # When pilot is below 1e-8, the floor ensures safe division
    if pilot_near_floor < 1e-8:
        relative = small_worsening / 1e-8
        expected = relative <= ctrl.rollback_tolerance
        assert result == expected


def test_accept_idempotence():
    """Calling accept() twice with same inputs returns same result."""
    ctrl = _ctrl()
    pairs = [
        (2.5, 2.4),
        (0.001, 0.001001),
        (1000.0, 1005.0),
        (1e-6, 2e-6),
    ]
    for pilot, after in pairs:
        r1 = ctrl.accept(pilot, after)
        r2 = ctrl.accept(pilot, after)
        assert r1 == r2, f"Idempotence violated for ({pilot}, {after})"
