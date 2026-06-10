"""Tests for regime detection (RegimeDetector)."""

import torch

from src.tg_lora.regime import Regime, RegimeDetector


class TestRegimeDetector:
    def test_stable_with_decreasing_loss(self):
        det = RegimeDetector(window=8, min_history=3)
        for i in range(10):
            loss = 5.0 - i * 0.1
            regime = det.update(loss)
        assert regime == Regime.STABLE

    def test_transition_on_sudden_change(self):
        det = RegimeDetector(window=8, min_history=3, transition_z=1.5)
        # Stable decreasing phase
        for i in range(6):
            det.update(5.0 - i * 0.1)
        # Sudden jump (regime change)
        regime = det.update(5.0)
        # Should detect transition due to outlier velocity
        assert regime == Regime.TRANSITION
        assert det.should_reset_priors

    def test_plateau_detection(self):
        det = RegimeDetector(window=8, min_history=3, plateau_eps=1e-3)
        # Decreasing then flat
        for i in range(6):
            det.update(5.0 - i * 0.1)
        # Plateau: loss barely changes
        for _ in range(4):
            det.update(4.401)
        assert det.regime == Regime.PLATEAU

    def test_reset_clears_state(self):
        det = RegimeDetector(window=8, min_history=3)
        for i in range(5):
            det.update(5.0 - i * 0.1)
        det.reset()
        assert det.regime == Regime.STABLE
        assert det.transition_count == 0

    def test_nonfinite_loss_ignored(self):
        det = RegimeDetector(window=8, min_history=3)
        det.update(5.0)
        regime = det.update(float("nan"))
        assert regime == Regime.STABLE  # unchanged
        regime = det.update(float("inf"))
        assert regime == Regime.STABLE

    def test_transition_count_increments(self):
        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        # Phase 1: stable
        for i in range(6):
            det.update(5.0 - i * 0.1)
        assert det.transition_count == 0
        # Phase 2: sudden jump → transition
        det.update(5.0)
        assert det.transition_count == 1
        # Continue in new regime (no re-trigger until next transition)
        det.update(4.9)
        det.update(4.8)
        assert det.transition_count == 1

    def test_should_reset_is_one_shot(self):
        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        for i in range(6):
            det.update(5.0 - i * 0.1)
        det.update(5.0)
        assert det.should_reset_priors  # first call: True
        # Next update clears the signal
        det.update(4.9)
        assert not det.should_reset_priors

    def test_short_history_stays_stable(self):
        det = RegimeDetector(window=8, min_history=3)
        det.update(5.0)
        det.update(10.0)  # big jump but not enough history
        assert det.regime == Regime.STABLE

    def test_configurable_parameters(self):
        det = RegimeDetector(window=4, plateau_eps=0.1, transition_z=1.0, min_history=2)
        assert det.window == 4
        assert det.plateau_eps == 0.1
        assert det.transition_z == 1.0
        assert det.min_history == 2

    def test_consume_reset_signal_clears_flag(self):
        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        for i in range(6):
            det.update(5.0 - i * 0.1)
        det.update(5.0)
        assert det.consume_reset_signal() is True
        # Signal consumed — second read returns False without needing update
        assert det.consume_reset_signal() is False
        assert det.should_reset_priors is False

    def test_consume_reset_signal_no_transition(self):
        det = RegimeDetector(window=8, min_history=3)
        for i in range(10):
            det.update(5.0 - i * 0.1)
        assert det.consume_reset_signal() is False


class TestRegimePSAIntegration:
    """Integration test: regime detector triggers PSA prior reset."""

    def test_transition_triggers_psa_reset(self):
        from src.tg_lora.psa import PSAPrior

        model = _make_model_with_grads()
        prior = PSAPrior(history_length=6, gain=0.5)

        # Build up priors with stable deltas
        dominant = _dominant_direction(model)
        for t in range(5):
            delta = {
                n: dominant[n] * (t + 1) + torch.randn_like(p) * 0.05
                for n, p in model.named_parameters()
                if "lora_" in n
                for p in [p]
            }
            prior.record_delta(delta)
        prior.extract_priors()
        assert len(prior.priors) > 0

        # Simulate regime detection → reset
        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        for i in range(6):
            det.update(5.0 - i * 0.1)
        det.update(5.0)  # Sudden jump

        assert det.consume_reset_signal()
        prior.reset_priors()

        assert len(prior.priors) == 0
        assert prior.history_count == 0

    def test_no_reset_when_consume_already_called(self):
        from src.tg_lora.psa import PSAPrior

        prior = PSAPrior(history_length=4, gain=0.5)
        prior.priors["x"] = torch.randn(8)

        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        for i in range(6):
            det.update(5.0 - i * 0.1)
        det.update(5.0)

        # First consume → True, triggers reset
        assert det.consume_reset_signal()
        prior.reset_priors()
        assert len(prior.priors) == 0

        # Rebuild priors
        prior.priors["x"] = torch.randn(8)

        # Second consume → False, no reset
        assert not det.consume_reset_signal()
        assert len(prior.priors) == 1  # Preserved — no spurious reset


def _make_model_with_grads():
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Parameter(torch.randn(2, 8) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(8, 2) * 0.01)

    return M()


def _dominant_direction(model):
    d = {}
    for n, p in model.named_parameters():
        if "lora_" in n:
            v = torch.randn_like(p)
            d[n] = v / v.norm()
    return d
