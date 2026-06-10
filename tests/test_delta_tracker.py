import torch
import math

from src.tg_lora.delta_tracker import (
    DeltaTracker,
    _compute_stats,
    compute_mean_delta,
)


# --- compute_mean_delta (historical name; now raw cycle delta) ---


def test_compute_mean_delta():
    K = 5
    before = {"w": torch.tensor([1.0, 2.0, 3.0])}
    after = {"w": torch.tensor([6.0, 7.0, 8.0])}

    delta = compute_mean_delta(after, before, K=K)

    expected = torch.tensor([5.0, 5.0, 5.0])
    assert torch.allclose(delta["w"], expected)


def test_compute_mean_delta_empty():
    delta = compute_mean_delta({}, {}, K=1)
    assert delta == {}


# --- _compute_stats ---


def test_compute_stats_single_tensor():
    delta = {"w": torch.tensor([3.0, 4.0])}
    stats = _compute_stats(delta)

    assert math.isclose(stats.total_norm, 5.0, rel_tol=1e-6)
    assert math.isclose(stats.max_component, 4.0, rel_tol=1e-6)
    assert math.isclose(stats.mean_abs, 3.5, rel_tol=1e-6)


def test_compute_stats_per_layer():
    delta = {
        "model.layers.0.lora_A": torch.tensor([1.0, 0.0]),
        "model.layers.0.lora_B": torch.tensor([0.0, 1.0]),
        "model.layers.3.lora_A": torch.tensor([2.0, 0.0]),
    }
    stats = _compute_stats(delta)

    assert "layer_0" in stats.per_layer_norm
    assert "layer_3" in stats.per_layer_norm
    assert math.isclose(stats.per_layer_norm["layer_0"], math.sqrt(2.0), rel_tol=1e-6)
    assert math.isclose(stats.per_layer_norm["layer_3"], 2.0, rel_tol=1e-6)


def test_compute_stats_other_layer_key():
    delta = {"embedding.weight": torch.tensor([1.0, 1.0])}
    stats = _compute_stats(delta)

    assert "other" in stats.per_layer_norm
    assert math.isclose(stats.per_layer_norm["other"], math.sqrt(2.0), rel_tol=1e-6)


def test_compute_stats_empty_delta():
    stats = _compute_stats({})
    assert stats.total_norm == 0.0
    assert stats.per_layer_norm == {}
    assert stats.max_component == 0.0
    assert stats.mean_abs == 0.0


# --- DeltaTracker ---


def test_tracker_initial_state():
    tracker = DeltaTracker()
    assert tracker.last_stats is None
    assert tracker.norm_history == []


def test_tracker_compute_and_record():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(3)}
    after = {"w": torch.tensor([1.0, 2.0, 3.0])}

    delta = tracker.compute_and_record(after, before, K=1)

    assert "w" in delta
    assert tracker.last_stats is not None
    assert math.isclose(tracker.last_stats.total_norm, math.sqrt(14.0), rel_tol=1e-6)
    assert len(tracker.norm_history) == 1


def test_tracker_multiple_cycles():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(2)}

    tracker.compute_and_record({"w": torch.tensor([1.0, 0.0])}, before, K=1)
    tracker.compute_and_record({"w": torch.tensor([0.5, 0.0])}, before, K=1)
    tracker.compute_and_record({"w": torch.tensor([0.25, 0.0])}, before, K=1)

    assert len(tracker.norm_history) == 3
    assert tracker.norm_history[0] > tracker.norm_history[1] > tracker.norm_history[2]


def test_tracker_max_history():
    tracker = DeltaTracker(max_history=3)
    before = {"w": torch.zeros(1)}

    for i in range(5):
        tracker.compute_and_record({"w": torch.tensor([float(i)])}, before, K=1)

    assert len(tracker.norm_history) == 3
    assert len(tracker._history) == 3


# --- is_anomalous ---


def test_anomalous_insufficient_history():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}
    tracker.compute_and_record({"w": torch.tensor([1.0])}, before, K=1)

    assert tracker.is_anomalous() is False


def test_anomalous_not_anomalous():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    for v in [1.0, 1.1, 1.05, 0.95, 1.0]:
        tracker.compute_and_record({"w": torch.tensor([v])}, before, K=1)

    assert tracker.is_anomalous() is False


def test_anomalous_detected():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
        tracker.compute_and_record({"w": torch.tensor([v])}, before, K=1)

    tracker.compute_and_record({"w": torch.tensor([100.0])}, before, K=1)

    assert tracker.is_anomalous() is True


def test_anomalous_custom_threshold():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    # History with some variance: std ≈ 0.14
    for v in [1.0, 0.9, 1.1, 0.95, 1.05]:
        tracker.compute_and_record({"w": torch.tensor([v])}, before, K=1)

    # Latest value = 1.5, mean ≈ 1.0, std ≈ 0.07
    tracker.compute_and_record({"w": torch.tensor([1.5])}, before, K=1)

    assert tracker.is_anomalous(threshold_sigma=10.0) is False
    assert tracker.is_anomalous(threshold_sigma=1.0) is True


def test_anomalous_zero_std():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    for _ in range(4):
        tracker.compute_and_record({"w": torch.tensor([1.0])}, before, K=1)

    assert tracker.is_anomalous() is False


def test_anomalous_zero_std_exceeds_double_mean():
    """When std < 1e-12 and latest norm > 2*mean, it is flagged anomalous."""
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    # Build constant history so std ≈ 0
    for _ in range(4):
        tracker.compute_and_record({"w": torch.tensor([1.0])}, before, K=1)

    # Spike to 3.0, which is > 2 * mean(1.0)
    tracker.compute_and_record({"w": torch.tensor([3.0])}, before, K=1)

    assert tracker.is_anomalous() is True


# --- convergence_trend ---


def test_convergence_trend_insufficient_data():
    tracker = DeltaTracker()
    assert tracker.convergence_trend() == 0.0

    before = {"w": torch.zeros(1)}
    tracker.compute_and_record({"w": torch.tensor([1.0])}, before, K=1)
    assert tracker.convergence_trend() == 0.0


def test_convergence_trend_decreasing():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    for v in [10.0, 8.0, 6.0, 4.0, 2.0]:
        tracker.compute_and_record({"w": torch.tensor([v])}, before, K=1)

    assert tracker.convergence_trend() < 0.0


def test_convergence_trend_increasing():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tracker.compute_and_record({"w": torch.tensor([v])}, before, K=1)

    assert tracker.convergence_trend() > 0.0


def test_convergence_trend_window():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}

    for v in [10.0, 10.0, 10.0, 1.0, 1.0]:
        tracker.compute_and_record({"w": torch.tensor([v])}, before, K=1)

    trend_default = tracker.convergence_trend()
    trend_window2 = tracker.convergence_trend(window=2)

    assert trend_default < 0.0
    assert math.isclose(trend_window2, 0.0, abs_tol=1e-6)


# --- summary ---


def test_summary_no_data():
    tracker = DeltaTracker()
    s = tracker.summary()

    assert s["total_norm"] == 0.0
    assert s["max_component"] == 0.0
    assert s["mean_abs"] == 0.0
    assert s["anomalous"] is False
    assert s["convergence_trend"] == 0.0
    assert s["history_length"] == 0


def test_summary_with_data():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(2)}

    tracker.compute_and_record({"w": torch.tensor([3.0, 4.0])}, before, K=1)
    tracker.compute_and_record({"w": torch.tensor([1.0, 0.0])}, before, K=1)

    s = tracker.summary()

    assert s["total_norm"] == 1.0
    assert s["max_component"] == 1.0
    assert s["history_length"] == 2
    assert isinstance(s["anomalous"], bool)
    assert isinstance(s["convergence_trend"], float)


# --- NaN/Inf sanitization ---


def test_compute_stats_skips_nan_tensor():
    delta = {
        "good": torch.tensor([3.0, 4.0]),
        "bad": torch.tensor([float("nan"), 1.0]),
    }
    stats = _compute_stats(delta)
    assert math.isfinite(stats.total_norm)
    assert math.isclose(stats.total_norm, 5.0, rel_tol=1e-6)


def test_compute_stats_skips_inf_tensor():
    delta = {
        "good": torch.tensor([3.0, 4.0]),
        "bad": torch.tensor([float("inf"), 0.0]),
    }
    stats = _compute_stats(delta)
    assert math.isfinite(stats.total_norm)
    assert math.isclose(stats.total_norm, 5.0, rel_tol=1e-6)


def test_compute_stats_all_nan_returns_zeros():
    delta = {"w": torch.tensor([float("nan")])}
    stats = _compute_stats(delta)
    assert stats.total_norm == 0.0
    assert stats.max_component == 0.0
    assert stats.mean_abs == 0.0


def test_tracker_nan_norm_not_appended_to_history():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}
    tracker.compute_and_record({"w": torch.tensor([1.0])}, before, K=1)
    assert len(tracker.norm_history) == 1

    # NaN delta: diff_lora produces nan * scale = nan
    after_nan = {"w": torch.tensor([float("nan")])}
    tracker.compute_and_record(after_nan, before, K=1)
    # The NaN norm should be skipped from history
    assert all(math.isfinite(n) for n in tracker.norm_history)


def test_tracker_inf_norm_not_appended_to_history():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}
    tracker.compute_and_record({"w": torch.tensor([1.0])}, before, K=1)

    after_inf = {"w": torch.tensor([float("inf")])}
    tracker.compute_and_record(after_inf, before, K=1)
    assert all(math.isfinite(n) for n in tracker.norm_history)


def test_compute_stats_all_inf_returns_zeros():
    delta = {"w": torch.tensor([float("inf"), float("inf")])}
    stats = _compute_stats(delta)
    assert stats.total_norm == 0.0
    assert stats.max_component == 0.0
    assert stats.mean_abs == 0.0


def test_compute_stats_neg_inf_tensor():
    delta = {
        "good": torch.tensor([3.0, 4.0]),
        "neg_inf": torch.tensor([float("-inf")]),
    }
    stats = _compute_stats(delta)
    assert math.isfinite(stats.total_norm)
    assert math.isclose(stats.total_norm, 5.0, rel_tol=1e-6)


def test_convergence_trend_flat_nonzero_history():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}
    for _ in range(5):
        tracker.compute_and_record({"w": torch.tensor([3.0])}, before, K=1)
    assert math.isclose(tracker.convergence_trend(), 0.0, abs_tol=1e-10)


def test_tracker_mixed_nan_and_inf_deltas():
    tracker = DeltaTracker()
    before = {"w": torch.zeros(1)}
    tracker.compute_and_record({"w": torch.tensor([1.0])}, before, K=1)
    assert len(tracker.norm_history) == 1

    # NaN/Inf tensors are skipped in _compute_stats, yielding norm=0.0 (finite)
    tracker.compute_and_record({"w": torch.tensor([float("nan")])}, before, K=1)
    tracker.compute_and_record({"w": torch.tensor([float("inf")])}, before, K=1)
    tracker.compute_and_record({"w": torch.tensor([2.0])}, before, K=1)

    # 4 entries: [1.0, 0.0, 0.0, 2.0] — all finite, no NaN/Inf leaked
    assert len(tracker.norm_history) == 4
    assert all(math.isfinite(n) for n in tracker.norm_history)
    assert tracker.norm_history == [1.0, 0.0, 0.0, 2.0]


def test_compute_mean_delta_rejects_zero_K():
    import pytest

    before = {"w": torch.tensor([1.0])}
    after = {"w": torch.tensor([2.0])}
    with pytest.raises(ValueError, match="K must be positive"):
        compute_mean_delta(after, before, K=0)


def test_compute_mean_delta_rejects_negative_K():
    import pytest

    before = {"w": torch.tensor([1.0])}
    after = {"w": torch.tensor([2.0])}
    with pytest.raises(ValueError, match="K must be positive"):
        compute_mean_delta(after, before, K=-3)


def test_compute_stats_no_grad_leak_with_requires_grad_tensors():
    """_compute_stats must not create autograd graph even with grad-tracked inputs."""
    t = torch.tensor([3.0, 4.0], requires_grad=True)
    delta = {"w": t}

    stats = _compute_stats(delta)

    assert math.isclose(stats.total_norm, 5.0, rel_tol=1e-6)
    # No autograd functions should be accumulated
    assert t.grad_fn is None or not t.requires_grad
    # The function should not retain any grad computation
    assert stats.total_norm == 5.0


# --- Parameter validation ---


def test_rejects_zero_max_history():
    import pytest

    with pytest.raises(ValueError, match="max_history must be positive"):
        DeltaTracker(max_history=0)


def test_rejects_negative_max_history():
    import pytest

    with pytest.raises(ValueError, match="max_history must be positive"):
        DeltaTracker(max_history=-1)


def test_accepts_positive_max_history():
    tracker = DeltaTracker(max_history=1)
    assert tracker.norm_history == []
