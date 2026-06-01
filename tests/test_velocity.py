import math

import pytest
import torch

from tg_lora.velocity import Velocity


def test_velocity_first_update():
    v = Velocity()
    delta = {"w": torch.tensor([1.0, 2.0, 3.0])}
    result = v.update(delta, beta=0.8)

    assert torch.allclose(result["w"], delta["w"])


def test_velocity_ema():
    v = Velocity()
    d1 = {"w": torch.tensor([1.0, 0.0, 0.0])}
    v.update(d1, beta=0.8)

    d2 = {"w": torch.tensor([0.0, 1.0, 0.0])}
    result = v.update(d2, beta=0.8)

    # 0.8 * [1,0,0] + 0.2 * [0,1,0] = [0.8, 0.2, 0]
    expected = torch.tensor([0.8, 0.2, 0.0])
    assert torch.allclose(result["w"], expected)


def test_cosine_similarity():
    v = Velocity()
    d1 = {"w": torch.tensor([1.0, 0.0, 0.0])}
    v.update(d1, beta=0.8)

    # Same direction
    d2 = {"w": torch.tensor([2.0, 0.0, 0.0])}
    sim = v.cosine_similarity(d2)
    assert abs(sim - 1.0) < 1e-6

    # Orthogonal
    d3 = {"w": torch.tensor([0.0, 1.0, 0.0])}
    sim = v.cosine_similarity(d3)
    assert abs(sim) < 1e-6


def test_cosine_similarity_no_state():
    v = Velocity()
    d = {"w": torch.tensor([1.0, 0.0, 0.0])}
    assert v.cosine_similarity(d) == 0.0


def test_velocity_reset():
    v = Velocity()
    d = {"w": torch.tensor([1.0, 2.0])}
    v.update(d, beta=0.8)
    assert v.state is not None

    v.reset()
    assert v.state is None


def test_cosine_similarity_mismatched_keys():
    v = Velocity()
    d1 = {"a": torch.tensor([1.0, 0.0])}
    v.update(d1, beta=0.8)

    d2 = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 0.0])}
    sim = v.cosine_similarity(d2)
    assert abs(sim - 1.0) < 1e-6


def test_velocity_update_with_new_keys():
    """Regression: delta containing keys not in existing state must not raise KeyError."""
    v = Velocity()
    d1 = {"a": torch.tensor([1.0, 0.0])}
    v.update(d1, beta=0.8)

    # New key "b" appears in second delta
    d2 = {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 0.0])}
    result = v.update(d2, beta=0.8)

    assert "a" in result
    assert "b" in result
    # "a" follows EMA: 0.8 * [1,0] + 0.2 * [0,1] = [0.8, 0.2]
    assert torch.allclose(result["a"], torch.tensor([0.8, 0.2]))
    # "b" is new, should be stored as-is (or initialized reasonably)
    assert "b" in v.state


def test_magnitude_tracking_first_update():
    v = Velocity()
    delta = {"w": torch.tensor([3.0, 4.0])}
    v.update(delta, beta=0.8)
    # L2 norm of [3, 4] = 5.0
    assert len(v.magnitudes) == 1
    assert abs(v.magnitudes[0] - 5.0) < 1e-5


def test_magnitude_tracking_ema():
    v = Velocity()
    d1 = {"w": torch.tensor([3.0, 4.0])}
    v.update(d1, beta=0.8)  # magnitude = 5.0

    d2 = {"w": torch.tensor([0.0, 0.0])}
    v.update(d2, beta=0.8)  # state = 0.8*[3,4] + 0.2*[0,0] = [2.4, 3.2], mag=4.0
    assert len(v.magnitudes) == 2
    assert abs(v.magnitudes[1] - 4.0) < 1e-5


def test_magnitude_reset():
    v = Velocity()
    v.update({"w": torch.tensor([1.0])}, beta=0.8)
    v.update({"w": torch.tensor([2.0])}, beta=0.8)
    assert len(v.magnitudes) == 2

    v.reset()
    assert v.magnitudes == []


def test_is_magnitude_anomalous_false_few_entries():
    v = Velocity()
    v.update({"w": torch.tensor([100.0])}, beta=0.8)
    v.update({"w": torch.tensor([100.0])}, beta=0.8)
    assert not v.is_magnitude_anomalous()


def test_is_magnitude_anomalous_detects_spike():
    v = Velocity()
    for _ in range(5):
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
    assert not v.is_magnitude_anomalous()

    # Spike: magnitude far above the stable history
    v.update({"w": torch.tensor([50.0])}, beta=0.8)
    assert v.is_magnitude_anomalous(threshold_sigma=2.0)


def test_is_magnitude_anomalous_near_zero_std():
    v = Velocity()
    for _ in range(5):
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
    # All magnitudes are nearly identical → std ≈ 0, uses mean*2 fallback
    assert not v.is_magnitude_anomalous()

    v.update({"w": torch.tensor([3.0])}, beta=0.8)
    # Still within mean*2 threshold for these small EMA changes
    assert not v.is_magnitude_anomalous()


def test_magnitude_trend_negative():
    v = Velocity()
    # Decreasing magnitudes → negative trend (converging)
    for val in [10.0, 8.0, 6.0, 4.0, 2.0]:
        v._magnitude_history.append(val)
    trend = v.magnitude_trend(window=5)
    assert trend < 0


def test_magnitude_trend_insufficient_data():
    v = Velocity()
    v.update({"w": torch.tensor([1.0])}, beta=0.8)
    assert v.magnitude_trend() == 0.0


def test_max_history_trims_magnitudes():
    v = Velocity(max_history=3)
    for val in [1.0, 2.0, 3.0, 4.0]:
        v.update({"w": torch.tensor([val])}, beta=0.8)
    assert len(v.magnitudes) == 3
    assert v.magnitudes[0] != 1.0  # oldest was trimmed


def test_cosine_similarity_warns_no_overlap(caplog):
    v = Velocity()
    v.update({"a": torch.tensor([1.0, 0.0])}, beta=0.8)

    with caplog.at_level("WARNING", logger="tg-lora"):
        sim = v.cosine_similarity({"b": torch.tensor([1.0, 0.0])})

    assert sim == 0.0
    assert "no overlapping keys" in caplog.text


def test_cosine_similarity_warns_orthogonal_vectors(caplog):
    v = Velocity()
    # Small orthogonal vectors → denom ≈ 1e-14 but norms are non-zero
    v.update({"w": torch.tensor([1e-7, 0.0])}, beta=0.8)

    with caplog.at_level("WARNING", logger="tg-lora"):
        sim = v.cosine_similarity({"w": torch.tensor([0.0, 1e-7])})

    assert sim == 0.0
    assert "near-zero denominator" in caplog.text


def test_cosine_similarity_no_warn_normal_case(caplog):
    v = Velocity()
    v.update({"w": torch.tensor([1.0, 0.0])}, beta=0.8)

    with caplog.at_level("WARNING", logger="tg-lora"):
        sim = v.cosine_similarity({"w": torch.tensor([2.0, 0.0])})

    assert abs(sim - 1.0) < 1e-6
    assert caplog.text == ""


# --- TASK-0079: data_ptr preservation tests ---


class TestVelocityInPlaceDataPtr:
    """Verify that velocity.update() EMA updates preserve tensor data_ptr (in-place ops)."""

    def test_ema_update_preserves_data_ptr(self):
        v = Velocity()
        d1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
        v.update(d1, beta=0.8)
        ptr_after_first = v.state["w"].data_ptr()

        d2 = {"w": torch.tensor([4.0, 5.0, 6.0])}
        v.update(d2, beta=0.8)
        ptr_after_second = v.state["w"].data_ptr()

        assert ptr_after_first == ptr_after_second

    def test_new_key_gets_different_data_ptr(self):
        v = Velocity()
        d1 = {"a": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)
        ptr_a = v.state["a"].data_ptr()

        d2 = {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 0.0])}
        v.update(d2, beta=0.8)

        assert "b" in v.state
        assert v.state["b"].data_ptr() != ptr_a

    def test_existing_key_ptr_preserved_with_new_key_added(self):
        v = Velocity()
        d1 = {"a": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)
        ptr_a_before = v.state["a"].data_ptr()

        d2 = {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 0.0])}
        v.update(d2, beta=0.8)

        # "a" should still be in-place updated despite "b" being new
        assert v.state["a"].data_ptr() == ptr_a_before

    def test_multiple_keys_all_preserve_ptr(self):
        v = Velocity()
        d1 = {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}
        v.update(d1, beta=0.9)
        ptr_x = v.state["x"].data_ptr()
        ptr_y = v.state["y"].data_ptr()

        d2 = {"x": torch.tensor([3.0]), "y": torch.tensor([4.0])}
        v.update(d2, beta=0.9)

        assert v.state["x"].data_ptr() == ptr_x
        assert v.state["y"].data_ptr() == ptr_y

    def test_new_key_tensor_is_clone_not_alias(self):
        v = Velocity()
        d1 = {"a": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)

        b_tensor = torch.tensor([5.0, 6.0])
        b_ptr_original = b_tensor.data_ptr()
        d2 = {"a": torch.tensor([0.0, 1.0]), "b": b_tensor}
        v.update(d2, beta=0.8)

        # Stored "b" is a clone, not the same underlying storage
        assert v.state["b"].data_ptr() != b_ptr_original


class TestVelocityNonFiniteMagnitude:
    """Verify that non-finite magnitudes are silently dropped from history."""

    def test_nan_delta_does_not_pollute_magnitude_history(self):
        v = Velocity()
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        assert len(v.magnitudes) == 1

        # Inject NaN into state directly to simulate corrupted velocity
        v._state["w"].fill_(float("nan"))
        # Next update with a normal delta will still have NaN in state due to EMA
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        # NaN magnitude should be skipped — history should not grow
        assert all(math.isfinite(m) for m in v.magnitudes)

    def test_inf_delta_does_not_pollute_magnitude_history(self):
        v = Velocity()
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        assert len(v.magnitudes) == 1

        v._state["w"].fill_(float("inf"))
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        assert all(math.isfinite(m) for m in v.magnitudes)

    def test_anomaly_detection_remains_valid_after_nonfinite(self):
        v = Velocity()
        # Build stable history
        for _ in range(5):
            v.update({"w": torch.tensor([1.0])}, beta=0.8)
        assert not v.is_magnitude_anomalous()

        # Corrupt state temporarily
        v._state["w"].fill_(float("nan"))
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        # Anomaly detection should still work with only finite entries
        assert not v.is_magnitude_anomalous()


class TestCosineSimilarityNonFinite:
    """Verify that cosine_similarity skips NaN/Inf tensors and returns a finite value."""

    def test_nan_state_tensor_skipped(self):
        v = Velocity()
        d1 = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)
        v._state["bad"].fill_(float("nan"))

        d2 = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([1.0, 0.0])}
        sim = v.cosine_similarity(d2)
        assert math.isfinite(sim)
        assert abs(sim - 1.0) < 1e-6

    def test_nan_delta_tensor_skipped(self):
        v = Velocity()
        d1 = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)

        d2 = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([float("nan"), 1.0])}
        sim = v.cosine_similarity(d2)
        assert math.isfinite(sim)
        assert abs(sim - 1.0) < 1e-6

    def test_inf_state_tensor_skipped(self):
        v = Velocity()
        d1 = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)
        v._state["bad"].fill_(float("inf"))

        d2 = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([1.0, 0.0])}
        sim = v.cosine_similarity(d2)
        assert math.isfinite(sim)

    def test_all_nonfinite_returns_zero(self):
        v = Velocity()
        d1 = {"bad": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)
        v._state["bad"].fill_(float("nan"))

        d2 = {"bad": torch.tensor([1.0, 0.0])}
        sim = v.cosine_similarity(d2)
        assert sim == 0.0

    def test_mixed_finite_and_nan_gives_valid_result(self):
        v = Velocity()
        d1 = {"w1": torch.tensor([1.0, 0.0]), "w2": torch.tensor([1.0, 0.0])}
        v.update(d1, beta=0.8)
        v._state["w2"].fill_(float("nan"))

        d2 = {"w1": torch.tensor([0.0, 1.0]), "w2": torch.tensor([1.0, 0.0])}
        sim = v.cosine_similarity(d2)
        assert math.isfinite(sim)
        assert abs(sim) < 1e-6


class TestMagnitudeAcceleration:
    """Verify magnitude_acceleration() computes the second derivative correctly."""

    def test_insufficient_data_returns_zero(self):
        v = Velocity()
        assert v.magnitude_acceleration() == 0.0
        v._magnitude_history.append(1.0)
        assert v.magnitude_acceleration() == 0.0
        v._magnitude_history.append(2.0)
        assert v.magnitude_acceleration() == 0.0

    def test_constant_magnitude_zero_acceleration(self):
        v = Velocity()
        for _ in range(5):
            v._magnitude_history.append(5.0)
        assert v.magnitude_acceleration() == pytest.approx(0.0, abs=1e-10)

    def test_linear_growth_zero_acceleration(self):
        v = Velocity()
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            v._magnitude_history.append(val)
        # Linear growth: slopes are all 1.0, so acceleration = 0
        assert v.magnitude_acceleration() == pytest.approx(0.0, abs=1e-10)

    def test_accelerating_growth_positive(self):
        v = Velocity()
        # Quadratic: 1, 4, 9, 16, 25 → slopes: 3, 5, 7, 9 → acc: +2
        for val in [1.0, 4.0, 9.0, 16.0, 25.0]:
            v._magnitude_history.append(val)
        acc = v.magnitude_acceleration()
        assert acc > 0

    def test_decelerating_growth_negative(self):
        v = Velocity()
        # Decelerating: 1, 3, 5, 6, 6.5 → slopes: 2, 2, 1, 0.5 → acc negative
        for val in [1.0, 3.0, 5.0, 6.0, 6.5]:
            v._magnitude_history.append(val)
        acc = v.magnitude_acceleration()
        assert acc < 0

    def test_shrinking_magnitudes_negative_acceleration(self):
        v = Velocity()
        # 25, 16, 9, 4, 1 → slopes: -9, -7, -5, -3 → acc: +2 (deceleration of shrink)
        for val in [25.0, 16.0, 9.0, 4.0, 1.0]:
            v._magnitude_history.append(val)
        acc = v.magnitude_acceleration()
        # Slopes are becoming less negative → positive acceleration
        assert acc > 0

    def test_window_parameter(self):
        v = Velocity()
        for val in [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]:
            v._magnitude_history.append(val)
        # Window=3 uses last 3 entries [10, 20, 30]: slopes [10, 10] → acc=0
        acc_small = v.magnitude_acceleration(window=3)
        assert acc_small == pytest.approx(0.0, abs=1e-10)

        # Window=6 uses all: [1,2,3,10,20,30] → slopes [1,1,7,10,10] → acc varies
        acc_full = v.magnitude_acceleration(window=6)
        assert acc_full != pytest.approx(0.0, abs=1e-3)


# --- Parameter validation ---


def test_rejects_zero_max_history():
    with pytest.raises(ValueError, match="max_history must be positive"):
        Velocity(max_history=0)


def test_rejects_negative_max_history():
    with pytest.raises(ValueError, match="max_history must be positive"):
        Velocity(max_history=-1)


def test_accepts_positive_max_history():
    v = Velocity(max_history=1)
    assert v.magnitudes == []
