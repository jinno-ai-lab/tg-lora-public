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


class TestRegimeDetectorStateRoundtrip:
    """state_dict / load_state_dict for the resume-state-loss axis.

    Mirrors ``TestPSAPriorStateRoundtrip`` / the activation-regime round-trip: the
    run-wide accumulation (loss / velocity classification windows + current
    regime + the run-wide transition count) must round-trip so a fault/periodic
    resume does not reset the per-cycle ``psa_regime_transitions`` (persisted to
    ``run_metrics.jsonl``) to 0 — a silent resume-state-loss sibling to the fixed
    psa_state / act_regime_state gaps.
    """

    def test_roundtrip_preserves_accumulated_state(self):
        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        for i in range(6):
            det.update(5.0 - i * 0.1)
        det.update(5.0)  # sudden jump → transition_count = 1
        assert det.transition_count == 1

        restored = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        restored.load_state_dict(det.state_dict())

        assert list(restored._losses) == list(det._losses)
        assert list(restored._velocities) == list(det._velocities)
        assert restored._regime == det._regime
        assert restored.transition_count == det.transition_count

    def test_transition_count_survives_roundtrip(self):
        # transition_count is the field reported as psa_regime_transitions.
        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        det._transition_count = 7
        state = det.state_dict()
        assert state["transition_count"] == 7

        restored = RegimeDetector(window=8, min_history=3)
        restored.load_state_dict(state)
        assert restored.transition_count == 7

    def test_load_none_is_noop(self):
        det = RegimeDetector(window=8, min_history=3)
        det.update(5.0)
        det.update(4.0)
        before = (list(det._losses), list(det._velocities), det._regime, det.transition_count)
        det.load_state_dict(None)  # pre-fix / PSA-disabled checkpoint
        after = (list(det._losses), list(det._velocities), det._regime, det.transition_count)
        assert before == after  # untouched — no fabricated state

    def test_load_empty_dict_is_noop(self):
        det = RegimeDetector(window=8, min_history=3)
        det.update(5.0)
        det.load_state_dict({})
        assert det.transition_count == 0
        assert det.regime == Regime.STABLE

    def test_partial_dict_tolerant(self):
        # A legacy checkpoint carrying only transition_count.
        det = RegimeDetector(window=8, min_history=3)
        det.load_state_dict({"transition_count": 3})
        assert det.transition_count == 3
        # Untouched keys stay at constructed defaults.
        assert list(det._losses) == []
        assert det.regime == Regime.STABLE

    def test_window_maxlen_rebuilt_with_constructed_window(self):
        # The deques are rebuilt with the CONSTRUCTED maxlen (from config), not
        # the checkpoint's, so a config-window change trims the way a live run
        # would and a too-large checkpoint window is not padded.
        det = RegimeDetector(window=4, min_history=3, transition_z=1.0)
        det._losses.extend([1.0, 2.0, 3.0])
        det._velocities.extend([0.1, 0.2])
        state = det.state_dict()

        restored = RegimeDetector(window=4, min_history=3, transition_z=1.0)
        restored.load_state_dict(state)
        assert restored._losses.maxlen == 5  # window + 1
        assert restored._velocities.maxlen == 4
        assert list(restored._losses) == [1.0, 2.0, 3.0]

    def test_reset_signal_not_persisted(self):
        # _reset_signaled is a transient one-shot consumed each cycle; a loaded
        # checkpoint must not re-fire a prior-reset the loop already consumed.
        det = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        for i in range(6):
            det.update(5.0 - i * 0.1)
        det.update(5.0)
        assert det.consume_reset_signal() is True

        state = det.state_dict()
        assert "reset_signaled" not in state  # transient — not serialized

        restored = RegimeDetector(window=8, min_history=3, transition_z=1.0)
        restored.load_state_dict(state)
        # Cold as on the run's first cycle — no stale prior-reset signal.
        assert restored.should_reset_priors is False

    def test_regime_value_roundtrips_through_string(self):
        det = RegimeDetector(window=8, min_history=3, plateau_eps=1e-3)
        for i in range(6):
            det.update(5.0 - i * 0.1)
        for _ in range(4):
            det.update(4.401)  # flat → PLATEAU
        assert det.regime == Regime.PLATEAU

        state = det.state_dict()
        assert state["regime"] == "plateau"  # serialized as the .value string

        restored = RegimeDetector(window=8, min_history=3)
        restored.load_state_dict(state)
        assert restored.regime == Regime.PLATEAU


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
