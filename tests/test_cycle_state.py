import pytest

from tg_lora.cycle_state import CycleState

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestCycleStateInit:
    def test_default_values(self):
        cs = CycleState()
        assert cs.cycle == 0
        assert cs.full_backward_passes == 0
        assert cs.extrapolation_steps == 0
        assert cs.best_loss == float("inf")
        assert cs.best_step == 0
        assert cs.stale_cycles == 0
        assert cs.last_train_loss == 0.0
        assert cs.accepted_count == 0
        assert cs.rejected_count == 0


# ---------------------------------------------------------------------------
# record_cycle
# ---------------------------------------------------------------------------


class TestRecordCycle:
    def test_increments_counters(self):
        cs = CycleState()
        cs.record_cycle(K=5, N=10, grad_accum=2, train_loss=1.5)
        assert cs.cycle == 1
        assert cs.full_backward_passes == 10  # 5 * 2
        assert cs.extrapolation_steps == 10
        assert cs.last_train_loss == 1.5
        assert cs.accepted_count == 1
        assert cs.rejected_count == 0

    def test_rejected_cycle(self):
        cs = CycleState()
        cs.record_cycle(K=3, N=5, grad_accum=1, train_loss=2.0, accepted=False)
        assert cs.accepted_count == 0
        assert cs.rejected_count == 1

    def test_neutral_cycle_does_not_change_acceptance_counts(self):
        cs = CycleState()
        cs.record_cycle(K=3, N=5, grad_accum=1, train_loss=2.0, accepted=None)
        assert cs.accepted_count == 0
        assert cs.rejected_count == 0

    def test_accumulates_across_multiple_cycles(self):
        cs = CycleState()
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0)
        cs.record_cycle(K=3, N=5, grad_accum=2, train_loss=0.8)
        assert cs.cycle == 2
        assert cs.full_backward_passes == 5 + 6  # 5*1 + 3*2
        assert cs.extrapolation_steps == 10 + 5

    def test_updates_best_loss_on_improvement(self):
        cs = CycleState(best_loss=1.0)
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=0.5, valid_loss=0.9)
        assert cs.best_loss == 0.9
        assert cs.best_step == 5
        assert cs.stale_cycles == 0

    def test_increments_stale_on_no_improvement(self):
        cs = CycleState(best_loss=0.5)
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=0.5, valid_loss=0.8)
        assert cs.best_loss == 0.5  # unchanged
        assert cs.stale_cycles == 1

    def test_stale_resets_on_improvement(self):
        cs = CycleState(best_loss=0.5, stale_cycles=3)
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=0.3, valid_loss=0.4)
        assert cs.stale_cycles == 0
        assert cs.best_loss == 0.4

    def test_no_valid_loss_skips_best_tracking(self):
        cs = CycleState()
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0)
        assert cs.best_loss == float("inf")
        assert cs.stale_cycles == 0


# ---------------------------------------------------------------------------
# reduction_rate
# ---------------------------------------------------------------------------


class TestReductionRate:
    def test_zero_when_no_steps(self):
        cs = CycleState()
        assert cs.reduction_rate == 0.0

    def test_computes_correctly(self):
        cs = CycleState(
            full_backward_passes=100,
            extrapolation_steps=300,
        )
        # reduction = 1 - 100/400 = 0.75
        assert cs.reduction_rate == pytest.approx(0.75)

    def test_no_extrapolation(self):
        cs = CycleState(full_backward_passes=100, extrapolation_steps=0)
        assert cs.reduction_rate == pytest.approx(0.0)

    def test_all_extrapolation(self):
        cs = CycleState(full_backward_passes=0, extrapolation_steps=100)
        assert cs.reduction_rate == pytest.approx(1.0)

    def test_updates_after_record_cycle(self):
        cs = CycleState()
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0)
        # 5 backward, 10 extrapolation → 1 - 5/15 ≈ 0.667
        assert cs.reduction_rate == pytest.approx(1.0 - 5.0 / 15.0)


# ---------------------------------------------------------------------------
# acceptance_rate
# ---------------------------------------------------------------------------


class TestAcceptanceRate:
    def test_zero_when_no_cycles(self):
        cs = CycleState()
        assert cs.acceptance_rate == 0.0

    def test_all_accepted(self):
        cs = CycleState(accepted_count=10, rejected_count=0)
        assert cs.acceptance_rate == 1.0

    def test_mixed(self):
        cs = CycleState(accepted_count=7, rejected_count=3)
        assert cs.acceptance_rate == pytest.approx(0.7)

    def test_updates_with_record(self):
        cs = CycleState()
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0, accepted=True)
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0, accepted=True)
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0, accepted=False)
        assert cs.acceptance_rate == pytest.approx(2.0 / 3.0)

    def test_neutral_cycles_do_not_affect_acceptance_rate(self):
        cs = CycleState()
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0, accepted=True)
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0, accepted=None)
        cs.record_cycle(K=5, N=10, grad_accum=1, train_loss=1.0, accepted=False)
        assert cs.acceptance_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# total_cycles
# ---------------------------------------------------------------------------


class TestTotalCycles:
    def test_zero_initially(self):
        assert CycleState().total_cycles == 0

    def test_counts_accepted_and_rejected(self):
        cs = CycleState(accepted_count=5, rejected_count=3)
        assert cs.total_cycles == 8


# ---------------------------------------------------------------------------
# should_stop (early stopping)
# ---------------------------------------------------------------------------


class TestShouldStop:
    def test_never_when_patience_none(self):
        cs = CycleState(stale_cycles=100, cycle=100)
        assert cs.should_stop(patience=None) is False

    def test_stops_when_patience_exceeded(self):
        cs = CycleState(stale_cycles=5, cycle=20)
        assert cs.should_stop(patience=5, min_cycles=10) is True

    def test_no_stop_below_min_cycles(self):
        cs = CycleState(stale_cycles=5, cycle=5)
        assert cs.should_stop(patience=3, min_cycles=10) is False

    def test_no_stop_below_patience(self):
        cs = CycleState(stale_cycles=2, cycle=20)
        assert cs.should_stop(patience=5, min_cycles=10) is False

    def test_exact_boundary(self):
        cs = CycleState(stale_cycles=3, cycle=10)
        assert cs.should_stop(patience=3, min_cycles=10) is True


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_returns_all_fields(self):
        cs = CycleState()
        cs.record_cycle(K=5, N=10, grad_accum=2, train_loss=1.0, valid_loss=0.9)
        s = cs.summary()
        assert s["cycles"] == 1
        assert s["full_backward_passes"] == 10
        assert s["extrapolation_steps"] == 10
        assert s["reduction_rate"] == pytest.approx(0.5)
        assert s["best_valid_loss"] == 0.9
        assert s["best_valid_step"] == 10
        assert s["stale_cycles"] == 0
        assert s["acceptance_rate"] == 1.0
        assert s["accepted_count"] == 1
        assert s["rejected_count"] == 0
        assert s["final_train_loss"] == 1.0

    def test_summary_after_multiple_cycles(self):
        cs = CycleState()
        cs.record_cycle(
            K=5, N=10, grad_accum=1, train_loss=2.0, valid_loss=1.5, accepted=True
        )
        cs.record_cycle(
            K=3, N=5, grad_accum=1, train_loss=1.8, valid_loss=1.8, accepted=False
        )
        s = cs.summary()
        assert s["cycles"] == 2
        assert s["full_backward_passes"] == 8
        assert s["extrapolation_steps"] == 15
        assert s["reduction_rate"] == pytest.approx(1.0 - 8.0 / 23.0)
        assert s["best_valid_loss"] == 1.5
        assert s["stale_cycles"] == 1
        assert s["acceptance_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_round_trip_empty(self):
        cs = CycleState()
        restored = CycleState.from_dict(cs.summary())
        assert restored.cycle == cs.cycle
        assert restored.full_backward_passes == cs.full_backward_passes
        assert restored.extrapolation_steps == cs.extrapolation_steps
        assert restored.best_loss == cs.best_loss
        assert restored.best_step == cs.best_step
        assert restored.stale_cycles == cs.stale_cycles
        assert restored.last_train_loss == cs.last_train_loss
        assert restored.accepted_count == cs.accepted_count
        assert restored.rejected_count == cs.rejected_count

    def test_round_trip_after_training(self):
        cs = CycleState()
        cs.record_cycle(
            K=5, N=10, grad_accum=1, train_loss=2.0, valid_loss=1.5, accepted=True
        )
        cs.record_cycle(
            K=3, N=5, grad_accum=2, train_loss=1.8, valid_loss=1.8, accepted=False
        )
        cs.record_cycle(
            K=4, N=8, grad_accum=1, train_loss=1.6, valid_loss=1.3, accepted=True
        )
        s = cs.summary()
        restored = CycleState.from_dict(s)
        assert restored.cycle == 3
        assert restored.full_backward_passes == 5 + 6 + 4
        assert restored.extrapolation_steps == 10 + 5 + 8
        assert restored.best_loss == pytest.approx(1.3)
        assert restored.best_step == 5 + 6 + 4
        assert restored.stale_cycles == 0
        assert restored.last_train_loss == pytest.approx(1.6)
        assert restored.accepted_count == 2
        assert restored.rejected_count == 1
        assert restored.reduction_rate == pytest.approx(cs.reduction_rate)
        assert restored.acceptance_rate == pytest.approx(cs.acceptance_rate)

    def test_from_empty_dict(self):
        cs = CycleState.from_dict({})
        assert cs.cycle == 0
        assert cs.best_loss == float("inf")
        assert cs.accepted_count == 0

    def test_from_partial_dict(self):
        cs = CycleState.from_dict({"cycles": 5, "best_valid_loss": 0.8})
        assert cs.cycle == 5
        assert cs.best_loss == pytest.approx(0.8)
        assert cs.full_backward_passes == 0

    def test_preserves_stale_cycles(self):
        cs = CycleState(stale_cycles=7, best_loss=0.5)
        cs.record_cycle(K=2, N=3, grad_accum=1, train_loss=0.6, valid_loss=0.7)
        restored = CycleState.from_dict(cs.summary())
        assert restored.stale_cycles == 8

    def test_from_legacy_checkpoint_keys(self):
        """from_dict accepts old checkpoint-format keys (cycle, best_loss, ...)."""
        checkpoint_data = {
            "cycle": 5,
            "full_backward_passes": 100,
            "extrapolation_steps": 200,
            "best_loss": 0.85,
            "best_step": 50,
            "stale_cycles": 3,
            "last_train_loss": 0.9,
            "accepted_count": 4,
            "rejected_count": 1,
        }
        restored = CycleState.from_dict(checkpoint_data)
        assert restored.cycle == 5
        assert restored.full_backward_passes == 100
        assert restored.best_loss == pytest.approx(0.85)
        assert restored.best_step == 50
        assert restored.stale_cycles == 3
        assert restored.last_train_loss == pytest.approx(0.9)

    def test_prefers_summary_keys_over_legacy(self):
        """When both key formats present, summary keys take precedence."""
        data = {"cycles": 10, "cycle": 5}
        restored = CycleState.from_dict(data)
        assert restored.cycle == 10


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestValidation:
    """CycleState.__post_init__ rejects invalid parameter values."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("cycle", -1),
            ("full_backward_passes", -1),
            ("extrapolation_steps", -1),
            ("best_step", -1),
            ("stale_cycles", -1),
            ("accepted_count", -1),
            ("rejected_count", -1),
        ],
    )
    def test_rejects_negative_field(self, field, value):
        with pytest.raises(ValueError, match=f"{field} must be non-negative"):
            CycleState(**{field: value})

    @pytest.mark.parametrize(
        "field",
        ["cycle", "full_backward_passes", "extrapolation_steps", "best_step", "stale_cycles", "accepted_count", "rejected_count"],
    )
    def test_accepts_zero(self, field):
        CycleState(**{field: 0})

    def test_default_values_all_valid(self):
        CycleState()
