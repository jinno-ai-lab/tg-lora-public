"""TASK-0091: validation hardening + coverage gap closure.

Tests for:
1. delta_tracker key-mismatch validation in compute_and_record
2. rollback_manager max_history positive-int validation
3. velocity.py uncovered edge-case lines (40, 49, 96)
"""

import math

import pytest
import torch

from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.rollback_manager import RollbackManager
from src.tg_lora.velocity import Velocity



# --- delta_tracker key-mismatch validation ---


class TestDeltaTrackerKeyMismatch:
    def test_extra_key_in_after_raises(self):
        tracker = DeltaTracker()
        before = {"w": torch.zeros(2)}
        after = {"w": torch.ones(2), "extra": torch.ones(2)}
        with pytest.raises(ValueError, match="Key mismatch"):
            tracker.compute_and_record(after, before, K=1)

    def test_missing_key_in_after_raises(self):
        tracker = DeltaTracker()
        before = {"w": torch.zeros(2), "v": torch.zeros(2)}
        after = {"w": torch.ones(2)}
        with pytest.raises(ValueError, match="Key mismatch"):
            tracker.compute_and_record(after, before, K=1)

    def test_completely_disjoint_keys_raises(self):
        tracker = DeltaTracker()
        before = {"a": torch.zeros(2)}
        after = {"b": torch.ones(2)}
        with pytest.raises(ValueError, match="Key mismatch"):
            tracker.compute_and_record(after, before, K=1)

    def test_matching_keys_succeeds(self):
        tracker = DeltaTracker()
        before = {"w": torch.zeros(2)}
        after = {"w": torch.ones(2)}
        delta = tracker.compute_and_record(after, before, K=1)
        assert "w" in delta


# --- rollback_manager max_history validation ---


class TestRollbackManagerMaxHistory:
    def test_zero_max_history_raises(self):
        with pytest.raises(ValueError, match="max_history must be positive"):
            RollbackManager(max_history=0)

    def test_negative_max_history_raises(self):
        with pytest.raises(ValueError, match="max_history must be positive"):
            RollbackManager(max_history=-5)

    def test_positive_max_history_works(self):
        mgr = RollbackManager(max_history=10)
        assert mgr._max_history == 10


# --- velocity.py edge-case coverage ---


class TestVelocityRecordMagnitudeEdgeCases:
    """Cover velocity.py lines 40 (state is None guard) and 49 (non-finite mag)."""

    def test_record_magnitude_after_reset_skips(self):
        """After reset(), _record_magnitude should hit the early return at line 40."""
        v = Velocity()
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        v.reset()
        # Directly call _record_magnitude — state is None
        v._record_magnitude()
        assert v.magnitudes == []

    def test_nonfinite_magnitude_not_recorded(self):
        """When magnitude is non-finite, line 49 early-returns without appending."""
        v = Velocity()
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        assert len(v.magnitudes) == 1

        # Corrupt state to produce overflow magnitude
        v._state["w"].fill_(float("inf"))
        v._record_magnitude()
        # The inf magnitude should not be appended (line 49 guard)
        assert all(math.isfinite(m) for m in v.magnitudes)

        # Another corrupted call: very large but finite value
        v._state["w"].fill_(1e30)
        v._record_magnitude()
        # Either it was appended (finite but huge) or skipped
        assert all(math.isfinite(m) for m in v.magnitudes)


class TestVelocityMagnitudeAccelerationSlopeGuard:
    """Cover velocity.py line 96 (slopes < 2 guard in magnitude_acceleration)."""

    def test_exactly_two_entries_returns_zero(self):
        """With 2 history entries → slopes has length 1 → early return at line 96."""
        v = Velocity()
        v._magnitude_history.append(1.0)
        v._magnitude_history.append(2.0)
        assert v.magnitude_acceleration() == 0.0

    def test_exactly_three_entries_computes_acceleration(self):
        """With 3 entries → slopes has length 2 → real computation, no early return."""
        v = Velocity()
        v._magnitude_history.append(1.0)
        v._magnitude_history.append(4.0)
        v._magnitude_history.append(9.0)
        acc = v.magnitude_acceleration()
        # slopes: [3, 5], acceleration: (5-3)/1 = 2.0
        assert acc == pytest.approx(2.0, abs=1e-10)
