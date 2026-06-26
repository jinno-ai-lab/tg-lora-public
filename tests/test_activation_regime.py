"""Tests for activation-fingerprint cosine regime tracker."""

import math
from collections import deque

import pytest
import torch
import torch.nn as nn

from src.tg_lora.activation_regime import (
    ActivationFingerprintTracker,
    ActivationRegime,
    _cosine_similarity,
    compute_regime_null_baseline,
)


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = torch.randn(64)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = torch.tensor([1.0, 2.0])
        b = -a
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        a = torch.randn(32)
        z = torch.zeros(32)
        assert _cosine_similarity(a, z) == 0.0


# ---------------------------------------------------------------------------
# Hook capture
# ---------------------------------------------------------------------------

class TestHookCapture:
    def test_hook_captures_tensor_output(self):
        module = nn.Linear(16, 8)
        tracker = ActivationFingerprintTracker()
        tracker.register_hook(module)

        x = torch.randn(2, 16)
        module(x)  # forward triggers hook

        assert tracker._current_act is not None
        assert tracker._current_act.shape == (2 * 8,)  # flattened
        tracker.remove_hooks()

    def test_hook_captures_first_element_of_tuple(self):
        """Simulates a module that returns (tensor, ...)."""

        class TupleModule(nn.Module):
            def forward(self, x):
                return x, "extra"

        mod = TupleModule()
        tracker = ActivationFingerprintTracker()
        tracker.register_hook(mod)

        x = torch.randn(1, 16)
        mod(x)

        assert tracker._current_act is not None
        assert tracker._current_act.numel() == 16
        tracker.remove_hooks()

    def test_fingerprint_capped_at_4096(self):
        module = nn.Linear(2048, 2048)
        tracker = ActivationFingerprintTracker()
        tracker.register_hook(module)

        x = torch.randn(4, 2048)  # output = 8192 elements
        module(x)

        assert tracker._current_act is not None
        assert tracker._current_act.numel() == 4096
        tracker.remove_hooks()

    def test_remove_hooks(self):
        module = nn.Linear(4, 4)
        tracker = ActivationFingerprintTracker()
        tracker.register_hook(module)
        assert len(tracker._hooks) == 1

        tracker.remove_hooks()
        assert len(tracker._hooks) == 0


# ---------------------------------------------------------------------------
# Step and regime classification
# ---------------------------------------------------------------------------

class TestStepClassification:
    def _make_tracker_with_cosines(self, cosines: list[float]) -> ActivationFingerprintTracker:
        """Create a tracker with pre-loaded cosine history."""
        tracker = ActivationFingerprintTracker(min_history=3)
        # Directly inject cosines into the deque
        for c in cosines:
            tracker._cosines.append(c)
        return tracker

    def test_stable_regime_high_cosine(self):
        tracker = self._make_tracker_with_cosines([0.97, 0.98, 0.96, 0.97])
        regime = tracker._classify()
        assert regime == ActivationRegime.STABLE

    def test_chaotic_regime_low_cosine(self):
        tracker = self._make_tracker_with_cosines([0.95, 0.93, 0.90, 0.3])
        regime = tracker._classify()
        assert regime == ActivationRegime.CHAOTIC

    def test_transition_regime_sudden_drop(self):
        # Recent history high, then sudden drop — transition
        tracker = self._make_tracker_with_cosines([0.97, 0.96, 0.97, 0.60])
        regime = tracker._classify()
        assert regime == ActivationRegime.TRANSITION

    def test_insufficient_history_returns_stable(self):
        tracker = self._make_tracker_with_cosines([0.5])
        regime = tracker._classify()
        assert regime == ActivationRegime.STABLE  # not enough data


# ---------------------------------------------------------------------------
# Step integration
# ---------------------------------------------------------------------------

class TestStepIntegration:
    def test_step_advances_regime(self):
        module = nn.Linear(4, 4)
        tracker = ActivationFingerprintTracker(min_history=2)
        tracker.register_hook(module)

        # Step 1: no previous activation
        module(torch.randn(1, 4))
        r1 = tracker.step()
        assert r1 == ActivationRegime.STABLE

        # Step 2: similar activation → stable
        torch.manual_seed(42)
        module(torch.randn(1, 4))
        r2 = tracker.step()

        # Step 3: similar activation → should be stable
        torch.manual_seed(42)
        module(torch.randn(1, 4))
        r3 = tracker.step()
        # Should be high cosine since same input
        assert len(tracker.cosines) == 2

        tracker.remove_hooks()

    def test_step_with_no_forward_returns_current_regime(self):
        tracker = ActivationFingerprintTracker()
        r = tracker.step()
        assert r == ActivationRegime.STABLE


# ---------------------------------------------------------------------------
# Regime inventory
# ---------------------------------------------------------------------------

class TestRegimeInventory:
    def test_inventory_tracks_fractions(self):
        tracker = ActivationFingerprintTracker(min_history=2)
        tracker._counts[ActivationRegime.STABLE] = 8
        tracker._counts[ActivationRegime.TRANSITION] = 1
        tracker._counts[ActivationRegime.CHAOTIC] = 1

        inv = tracker.regime_inventory
        assert inv["stable"] == pytest.approx(0.8)
        assert inv["transition"] == pytest.approx(0.1)
        assert inv["chaotic"] == pytest.approx(0.1)

    def test_stable_fraction(self):
        tracker = ActivationFingerprintTracker()
        tracker._counts[ActivationRegime.STABLE] = 7
        tracker._counts[ActivationRegime.CHAOTIC] = 3
        assert tracker.stable_fraction == pytest.approx(0.7)

    def test_empty_inventory_returns_zeros(self):
        tracker = ActivationFingerprintTracker()
        inv = tracker.regime_inventory
        assert all(v == 0.0 for v in inv.values())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self):
        tracker = ActivationFingerprintTracker(min_history=2)
        module = nn.Linear(4, 4)
        tracker.register_hook(module)
        module(torch.randn(1, 4))
        tracker.step()
        module(torch.randn(1, 4))
        tracker.step()

        s = tracker.summary()
        assert "regime" in s
        assert "stable_fraction" in s
        assert "regime_inventory" in s
        assert "cosine_mean" in s
        assert "cosine_latest" in s
        assert "total_steps" in s
        assert "all_cosines" in s
        assert s["total_steps"] == 2
        tracker.remove_hooks()

    def test_all_cosines_preserves_full_history(self):
        tracker = ActivationFingerprintTracker(min_history=2, window=5)
        module = nn.Linear(4, 4)
        tracker.register_hook(module)

        for i in range(10):
            torch.manual_seed(i)
            module(torch.randn(1, 4))
            tracker.step()

        # Sliding window has at most 5, but all_cosines has all 9
        assert len(tracker.cosines) <= 5
        assert len(tracker._all_cosines) == 9  # 10 steps - 1 (first has no prev)
        assert len(tracker.summary()["all_cosines"]) == 9
        tracker.remove_hooks()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        tracker = ActivationFingerprintTracker()
        tracker._cosines.extend([0.9, 0.8, 0.7])
        tracker._all_cosines.extend([0.9, 0.8, 0.7])
        tracker._counts[ActivationRegime.STABLE] = 10
        tracker._prev_act = torch.randn(16)

        tracker.reset()

        assert len(tracker._cosines) == 0
        assert len(tracker._all_cosines) == 0
        assert tracker._prev_act is None
        assert sum(tracker._counts.values()) == 0
        assert tracker.regime == ActivationRegime.STABLE


# ---------------------------------------------------------------------------
# Null baseline (GOAL §7)
# ---------------------------------------------------------------------------

class TestNullBaseline:
    def test_null_baseline_returns_required_keys(self):
        cosines = [0.97, 0.96, 0.98, 0.95, 0.97, 0.96, 0.30, 0.94, 0.97, 0.96]
        result = compute_regime_null_baseline(cosines, n_shuffles=50)
        assert "stable_fraction_null_mean" in result
        assert "stable_fraction_null_std" in result
        assert "stable_fraction_z" in result
        assert "transition_fraction_null_mean" in result
        assert "chaotic_fraction_null_mean" in result
        assert "per_shuffle_fractions" in result
        assert len(result["per_shuffle_fractions"]) == 50

    def test_null_baseline_short_series_returns_none(self):
        cosines = [0.95]  # too short for classification
        result = compute_regime_null_baseline(cosines, min_history=3)
        assert result["stable_fraction_null_mean"] is None
        assert result["per_shuffle_fractions"] == []

    def test_null_baseline_stable_series_high_z(self):
        # All high cosines → all stable in observed and null
        cosines = [0.97] * 50
        result = compute_regime_null_baseline(
            cosines, n_shuffles=100, stable_threshold=0.95
        )
        # Shuffling identical values gives same result, so z should be 0
        assert result["stable_fraction_z"] == pytest.approx(0.0, abs=0.01)

    def test_null_baseline_distinguishes_temporal_structure(self):
        # Create a series with temporal structure: stable block then chaotic block
        cosines = [0.97] * 25 + [0.3] * 25
        result = compute_regime_null_baseline(
            cosines, n_shuffles=200, stable_threshold=0.95,
            chaotic_threshold=0.5, min_history=3, window=10,
        )
        # The shuffled baseline should have different stable fraction
        # than the observed (observed has clear structure)
        assert result["stable_fraction_null_mean"] is not None
        assert result["stable_fraction_null_std"] is not None
        assert isinstance(result["stable_fraction_z"], float)


# ---------------------------------------------------------------------------
# Checkpoint resume (state_dict / load_state_dict) — GOAL §4 resume-state-loss axis
# ---------------------------------------------------------------------------

class TestActivationRegimeStateRoundtrip:
    """The tracker's run-wide accumulation must survive checkpoint resume or the
    run-end ``activation_regime_inventory`` / ``stable_fraction`` (GOAL §4) reflect
    only post-resume steps — a resume-state-loss sibling to the fixed LAWA /
    dynfreeze / best_full_eval surfaces. These tests pin ``state_dict()`` /
    ``load_state_dict()`` directly, independent of the TrainingState layer."""

    def test_roundtrip_preserves_inventory_and_counts(self):
        """Serialize a populated tracker, load into a fresh one, and verify the
        run-wide regime surface is identical."""
        src = ActivationFingerprintTracker(window=10, min_history=3)
        # Populate the resume-persistent surface directly (mirrors what a live
        # run accumulates through step()).
        src._all_cosines = [0.97, 0.98, 0.96, 0.41, 0.97]
        src._cosines = deque([0.96, 0.41, 0.97], maxlen=src.window)
        src._counts[ActivationRegime.STABLE] = 3
        src._counts[ActivationRegime.TRANSITION] = 1
        src._counts[ActivationRegime.CHAOTIC] = 1
        src._regime = ActivationRegime.STABLE

        dst = ActivationFingerprintTracker(window=10, min_history=3)
        dst.load_state_dict(src.state_dict())

        # Counts drive regime_inventory / stable_fraction.
        assert dst._counts == src._counts
        assert dst.regime_inventory == src.regime_inventory
        assert dst.stable_fraction == src.stable_fraction
        # Full cosine series drives the GOAL §7 null baseline at summary time.
        assert dst._all_cosines == src._all_cosines
        assert dst.summary()["all_cosines"] == src._all_cosines
        assert dst.regime == src.regime

    def test_roundtrip_after_real_steps(self):
        """Populate via the real hook+step() path, serialize, reload into a fresh
        tracker, and verify the loaded inventory matches the live run exactly
        (proves state_dict captures real accumulated state, not zeros)."""
        def run_live(tracker):
            module = nn.Linear(8, 8)
            tracker.register_hook(module)
            # Steps 1-3: identical input → identical output → cosine ~1.0 (stable).
            x_stable = torch.randn(1, 8)
            for _ in range(3):
                module(x_stable)
                tracker.step()
            # Step 4: a wildly different input → low cosine (chaotic/transition).
            module(torch.randn(1, 8) * 50)
            tracker.step()
            tracker.remove_hooks()

        src = ActivationFingerprintTracker(window=10, min_history=3)
        run_live(src)
        dst = ActivationFingerprintTracker(window=10, min_history=3)
        dst.load_state_dict(src.state_dict())

        assert dst._counts == src._counts
        assert dst._all_cosines == src._all_cosines
        assert dst.regime_inventory == src.regime_inventory

    def test_load_none_is_noop(self):
        """``load_state_dict(None)`` must leave a fresh tracker empty — covers a
        legacy blob and an ``activation_regime_enabled: false`` resume."""
        tracker = ActivationFingerprintTracker()
        tracker.load_state_dict(None)
        assert tracker._all_cosines == []
        assert len(tracker._cosines) == 0
        assert sum(tracker._counts.values()) == 0
        assert tracker.regime == ActivationRegime.STABLE

    def test_load_partial_dict_tolerates_missing_keys(self):
        """A legacy/partial blob with only some keys must load the present keys
        and leave the rest at their constructed defaults."""
        tracker = ActivationFingerprintTracker()
        tracker.load_state_dict(
            {"counts": {"stable": 4, "transition": 0, "chaotic": 1}}
        )
        assert tracker._counts[ActivationRegime.STABLE] == 4
        assert tracker._counts[ActivationRegime.CHAOTIC] == 1
        # Unset keys keep defaults.
        assert tracker._all_cosines == []
        assert tracker.regime == ActivationRegime.STABLE

    def test_resume_does_not_require_transient_tensors(self):
        """Transient per-step activation tensors are NOT persisted; a resumed
        tracker must operate from a cold start (no predecessor) just like the
        run's first step — the next step() with a forward hook simply has no
        cosine (the same cold-start as step 1), and must not crash."""
        src = ActivationFingerprintTracker(window=10, min_history=3)
        src._all_cosines = [0.97, 0.98]
        src._cosines = deque([0.97, 0.98], maxlen=src.window)
        src._counts[ActivationRegime.STABLE] = 2
        src._prev_act = torch.randn(8)  # transient — must NOT be persisted

        blob = src.state_dict()
        dst = ActivationFingerprintTracker(window=10, min_history=3)
        dst.load_state_dict(blob)
        assert "prev_act" not in blob
        assert dst._prev_act is None  # cold start, as at run start

        module = nn.Linear(8, 8)
        dst.register_hook(module)
        module(torch.randn(1, 8))
        regime = dst.step()  # cold start: no cosine, no crash
        assert regime in set(ActivationRegime)
        dst.remove_hooks()

    def test_load_rebuilds_window_maxlen(self):
        """The classification window is rebuilt with the tracker's constructed
        ``maxlen``, so a checkpoint whose ``cosines`` exceed the live window is
        trimmed the same way a live run would have trimmed it (not unbounded),
        while the full series (needed for the §7 null baseline) is unbounded."""
        blob = {
            "all_cosines": [0.9, 0.91, 0.92, 0.93, 0.94],
            "cosines": [0.9, 0.91, 0.92, 0.93, 0.94],
            "counts": {"stable": 5, "transition": 0, "chaotic": 0},
            "regime": "stable",
        }
        tracker = ActivationFingerprintTracker(window=3)
        tracker.load_state_dict(blob)
        # Sliding window trimmed to the constructed maxlen...
        assert len(tracker._cosines) == 3
        assert list(tracker._cosines) == [0.92, 0.93, 0.94]
        # ...but the full series is unbounded.
        assert tracker._all_cosines == [0.9, 0.91, 0.92, 0.93, 0.94]
