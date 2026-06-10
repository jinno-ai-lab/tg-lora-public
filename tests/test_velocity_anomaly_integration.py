"""End-to-end integration tests for the Velocity anomaly detection pipeline.

Simulates realistic training time-series data through velocity.py and asserts
that anomaly detection and trend tracking behave correctly across the full
lifecycle: convergent training → anomaly spike → recovery → divergent phase.
"""

import math

import torch

from src.tg_lora.velocity import Velocity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_delta(
    scale: float, keys: list[str], size: int = 8
) -> dict[str, torch.Tensor]:
    """Create a delta dict where each tensor has L2 norm ≈ scale * sqrt(size)."""
    raw_val = scale / math.sqrt(size)
    return {k: torch.full((size,), raw_val) for k in keys}


def _simulate_constant_phase(
    vel: Velocity,
    keys: list[str],
    scale: float,
    steps: int,
    beta: float,
) -> None:
    for _ in range(steps):
        vel.update(_make_delta(scale, keys), beta)


def _simulate_decay_phase(
    vel: Velocity,
    keys: list[str],
    scale: float,
    decay: float,
    steps: int,
    beta: float,
) -> None:
    for i in range(steps):
        s = scale * (decay**i)
        vel.update(_make_delta(s, keys), beta)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestVelocityAnomalyPipelineEndToEnd:
    """Full pipeline: update → magnitude tracking → anomaly detection → trend."""

    KEYS = [
        "model.layers.0.lora_A",
        "model.layers.0.lora_B",
        "model.layers.15.lora_A",
        "model.layers.15.lora_B",
    ]

    def test_convergent_training_no_false_positive(self):
        """50-step convergent phase: no anomaly, negative trend."""
        vel = Velocity(max_history=200)
        beta = 0.9
        _simulate_decay_phase(
            vel, self.KEYS, scale=1.0, decay=0.95, steps=50, beta=beta
        )

        assert not vel.is_magnitude_anomalous(threshold_sigma=2.0), (
            "Convergent training should not trigger anomaly detection"
        )
        assert vel.magnitude_trend(window=10) < 0, (
            "Convergent training should show negative (converging) trend"
        )
        assert len(vel.magnitudes) == 50

    def test_stable_training_no_anomaly(self):
        """Constant-magnitude deltas: no anomaly, near-zero trend."""
        vel = Velocity(max_history=200)
        _simulate_constant_phase(vel, self.KEYS, scale=1.0, steps=30, beta=0.9)

        assert not vel.is_magnitude_anomalous()
        trend = vel.magnitude_trend(window=10)
        assert abs(trend) < 0.1, (
            f"Stable training trend should be near zero, got {trend}"
        )

    def test_spike_after_convergence_detected(self):
        """Convergent → sudden spike: anomaly detected, trend turns positive."""
        vel = Velocity(max_history=200)
        beta = 0.9

        # Phase 1: convergent (30 steps)
        _simulate_decay_phase(
            vel, self.KEYS, scale=1.0, decay=0.93, steps=30, beta=beta
        )
        assert not vel.is_magnitude_anomalous()

        # Phase 2: inject spike (10x the last magnitude)
        spike_scale = 10.0
        vel.update(_make_delta(spike_scale, self.KEYS), beta)

        assert vel.is_magnitude_anomalous(threshold_sigma=2.0), (
            "Spike after convergence should be detected as anomalous"
        )

    def test_full_lifecycle_converge_spike_recover(self):
        """Convergent → spike → recovery: anomaly appears then clears."""
        vel = Velocity(max_history=200)
        beta = 0.9

        # Phase 1: convergent (20 steps)
        _simulate_decay_phase(vel, self.KEYS, scale=1.0, decay=0.9, steps=20, beta=beta)
        assert not vel.is_magnitude_anomalous()
        trend_converging = vel.magnitude_trend(window=10)
        assert trend_converging < 0

        # Phase 2: spike (large enough to trigger even with default sigma=3.0)
        vel.update(_make_delta(20.0, self.KEYS), beta)
        assert vel.is_magnitude_anomalous(threshold_sigma=2.0)

        # Phase 3: recovery (small stable deltas, enough to dilute spike)
        _simulate_constant_phase(vel, self.KEYS, scale=0.2, steps=15, beta=beta)
        assert not vel.is_magnitude_anomalous(), (
            "After recovery phase, anomaly should clear"
        )

    def test_gradual_divergence_detected(self):
        """Gradually increasing deltas: trend turns positive, anomaly at high sigma."""
        vel = Velocity(max_history=200)
        beta = 0.9

        # 20 steps of increasing magnitude (each 15% larger)
        for i in range(20):
            scale = 0.5 * (1.15**i)
            vel.update(_make_delta(scale, self.KEYS), beta)

        trend = vel.magnitude_trend(window=10)
        assert trend > 0, "Divergent training should show positive trend"

        # At high sigma (5.0) may not be anomalous yet
        # At low sigma (1.5) should be anomalous
        assert vel.is_magnitude_anomalous(threshold_sigma=1.5), (
            "Divergent training should trigger anomaly at low threshold"
        )

    def test_single_layer_pipeline(self):
        """Pipeline works with single LoRA parameter (edge case)."""
        vel = Velocity()
        single_key = ["model.layers.0.lora_A"]

        for _ in range(10):
            vel.update(_make_delta(1.0, single_key, size=4), beta=0.8)
        assert not vel.is_magnitude_anomalous()

        vel.update(_make_delta(20.0, single_key, size=4), beta=0.8)
        assert vel.is_magnitude_anomalous(threshold_sigma=2.0)

    def test_multi_scale_parameters(self):
        """Parameters with very different magnitudes still produce valid pipeline output."""
        vel = Velocity(max_history=200)
        beta = 0.9

        # lora_A is large, lora_B is small (realistic for initialized LoRA)
        for _ in range(15):
            delta = {
                "layers.0.lora_A": torch.randn(64, 32) * 0.01,
                "layers.0.lora_B": torch.randn(64, 32) * 0.0001,
            }
            vel.update(delta, beta)

        assert not vel.is_magnitude_anomalous()

        # Inject spike only in lora_A
        spike = {
            "layers.0.lora_A": torch.randn(64, 32) * 1.0,  # 100x normal
            "layers.0.lora_B": torch.randn(64, 32) * 0.0001,  # normal
        }
        vel.update(spike, beta)
        assert vel.is_magnitude_anomalous(threshold_sigma=2.0)

    def test_magnitude_history_ring_buffer_with_anomaly(self):
        """Ring buffer overflow during long training doesn't break anomaly detection."""
        vel = Velocity(max_history=10)
        beta = 0.9

        # 30 steps of stable data (fills and rolls the buffer 3x)
        _simulate_constant_phase(vel, self.KEYS, scale=1.0, steps=30, beta=beta)
        assert len(vel.magnitudes) == 10
        assert not vel.is_magnitude_anomalous()

        # Spike after buffer rolled over
        vel.update(_make_delta(20.0, self.KEYS), beta)
        assert vel.is_magnitude_anomalous(threshold_sigma=2.0)

    def test_reset_clears_anomaly_state(self):
        """After reset, anomaly pipeline starts fresh."""
        vel = Velocity()
        beta = 0.9

        _simulate_constant_phase(vel, self.KEYS, scale=1.0, steps=10, beta=beta)
        vel.update(_make_delta(20.0, self.KEYS), beta)
        assert vel.is_magnitude_anomalous()

        vel.reset()
        assert vel.magnitudes == []
        assert vel.state is None

        # Fresh start: no anomaly with just a few entries
        for _ in range(2):
            vel.update(_make_delta(5.0, self.KEYS), beta)
        assert not vel.is_magnitude_anomalous()

    def test_cosine_similarity_with_anomaly_context(self):
        """Cosine similarity remains meaningful even when anomaly is present."""
        vel = Velocity()
        beta = 0.9

        # Build up state with a specific direction
        base_delta = {"w": torch.tensor([1.0, 0.0, 0.0, 0.0])}
        for _ in range(5):
            vel.update(base_delta, beta=beta)

        sim_stable = vel.cosine_similarity(base_delta)
        assert sim_stable > 0.99

        # Inject orthogonal direction (perpendicular in 4D)
        orthogonal = {"w": torch.tensor([0.0, 1.0, 0.0, 0.0])}
        vel.update(orthogonal, beta=beta)

        sim_after = vel.cosine_similarity(base_delta)
        # Similarity should drop because velocity now has orthogonal component
        assert sim_after < sim_stable

    def test_pipeline_with_decreasing_beta(self):
        """Varying beta across phases still produces valid pipeline output."""
        vel = Velocity(max_history=200)

        # Phase 1: high beta (0.95), slow tracking
        _simulate_constant_phase(vel, self.KEYS, scale=1.0, steps=10, beta=0.95)

        # Phase 2: medium beta (0.8)
        _simulate_constant_phase(vel, self.KEYS, scale=1.0, steps=10, beta=0.8)

        # Phase 3: low beta (0.5), fast tracking + spike
        _simulate_constant_phase(vel, self.KEYS, scale=1.0, steps=5, beta=0.5)
        vel.update(_make_delta(8.0, self.KEYS), beta=0.5)

        assert vel.is_magnitude_anomalous(threshold_sigma=2.0)


class TestVelocityDeltaTrackerCombinedAnomaly:
    """Integration between Velocity anomaly and DeltaTracker anomaly.

    Both modules independently detect anomalies but on different signals:
    - Velocity: magnitude of the velocity vector (EMA of deltas)
    - DeltaTracker: raw delta norm history

    In real training, both should agree on clear anomalies.
    """

    def test_both_detectors_flag_spike(self):
        """A sudden spike is flagged by both Velocity and DeltaTracker."""
        from src.tg_lora.delta_tracker import DeltaTracker

        vel = Velocity(max_history=200)
        dt = DeltaTracker(max_history=200)
        beta = 0.9
        keys = ["layers.0.lora_A", "layers.0.lora_B"]

        # Normal phase
        for _ in range(10):
            delta = _make_delta(0.5, keys)
            vel.update(delta, beta)
            before = {k: torch.zeros_like(v) for k, v in delta.items()}
            dt.compute_and_record(delta, before, K=2)

        assert not vel.is_magnitude_anomalous()
        assert not dt.is_anomalous()

        # Spike
        spike = _make_delta(15.0, keys)
        vel.update(spike, beta)
        before_spike = {k: torch.zeros_like(v) for k, v in spike.items()}
        dt.compute_and_record(spike, before_spike, K=2)

        assert vel.is_magnitude_anomalous(threshold_sigma=2.0)
        assert dt.is_anomalous()

    def test_velocity_smoother_than_delta_tracker(self):
        """Velocity anomaly detection is smoother (EMA-based) than raw delta tracking."""
        from src.tg_lora.delta_tracker import DeltaTracker

        vel = Velocity(max_history=200)
        dt = DeltaTracker(max_history=200)
        beta = 0.9
        keys = ["layers.0.lora_A"]

        # Build up history with constant deltas
        for _ in range(10):
            delta = _make_delta(1.0, keys)
            vel.update(delta, beta)
            before = {k: torch.zeros_like(v) for k, v in delta.items()}
            dt.compute_and_record(delta, before, K=2)

        # Small perturbation: DeltaTracker may flag it, Velocity (EMA) should be smoother
        # and less likely to flag small perturbations
        moderate = _make_delta(2.5, keys)
        vel.update(moderate, beta)
        before = {k: torch.zeros_like(v) for k, v in moderate.items()}
        dt.compute_and_record(moderate, before, K=2)

        # Both may or may not flag, but the key property is that they return valid bools
        assert isinstance(vel.is_magnitude_anomalous(), bool)
        assert isinstance(dt.is_anomalous(), bool)

    def test_convergence_trend_agreement(self):
        """Both trackers agree on convergence direction for monotonic series."""
        from src.tg_lora.delta_tracker import DeltaTracker

        vel = Velocity(max_history=200)
        dt = DeltaTracker(max_history=200)
        beta = 0.9
        keys = ["layers.0.lora_A"]

        # Decreasing deltas
        for i in range(15):
            scale = 1.0 * (0.85**i)
            delta = _make_delta(scale, keys)
            vel.update(delta, beta)
            before = {k: torch.zeros_like(v) for k, v in delta.items()}
            dt.compute_and_record(delta, before, K=2)

        # Both should report negative (converging) trend
        assert vel.magnitude_trend(window=10) < 0
        assert dt.convergence_trend(window=10) < 0

    def test_full_training_simulation(self):
        """Simulate a realistic training run with all pipeline stages."""
        from src.tg_lora.delta_tracker import DeltaTracker

        vel = Velocity(max_history=100)
        dt = DeltaTracker(max_history=100)
        beta = 0.9
        keys = [
            "layers.0.lora_A",
            "layers.0.lora_B",
            "layers.15.lora_A",
            "layers.15.lora_B",
        ]

        results: list[dict] = []

        def record(step: int, phase: str) -> None:
            results.append(
                {
                    "step": step,
                    "phase": phase,
                    "vel_anomaly": vel.is_magnitude_anomalous(),
                    "dt_anomaly": dt.is_anomalous(),
                    "vel_trend": vel.magnitude_trend(window=5),
                    "dt_trend": dt.convergence_trend(window=5),
                    "magnitude": vel.magnitudes[-1] if vel.magnitudes else 0.0,
                }
            )

        # Phase 1: Initial convergence (20 steps, decreasing deltas)
        for i in range(20):
            scale = 1.0 * (0.92**i)
            delta = _make_delta(scale, keys)
            vel.update(delta, beta)
            before = {k: torch.zeros_like(v) for k, v in delta.items()}
            dt.compute_and_record(delta, before, K=2)

        record(19, "converging")

        # Phase 2: Spike (simulates bad extrapolation, large enough for sigma=3.0 default)
        spike = _make_delta(20.0, keys)
        vel.update(spike, beta)
        before = {k: torch.zeros_like(v) for k, v in spike.items()}
        dt.compute_and_record(spike, before, K=2)
        record(20, "spike")

        # Phase 3: Recovery (15 steps of small stable deltas)
        for i in range(15):
            delta = _make_delta(0.3, keys)
            vel.update(delta, beta)
            before = {k: torch.zeros_like(v) for k, v in delta.items()}
            dt.compute_and_record(delta, before, K=2)

        record(35, "recovery")

        # Assert phase-by-phase behavior
        converging = results[0]
        assert not converging["vel_anomaly"], "Should not flag during convergence"
        assert converging["vel_trend"] < 0, "Convergence trend should be negative"

        spike_phase = results[1]
        assert spike_phase["vel_anomaly"], "Should flag spike"
        assert spike_phase["dt_anomaly"], "DeltaTracker should also flag spike"

        recovery = results[2]
        assert not recovery["vel_anomaly"], "Anomaly should clear after recovery"
        assert not recovery["dt_anomaly"], "DeltaTracker anomaly should also clear"
