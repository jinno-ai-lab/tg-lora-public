"""Unit tests for FreezeSchedule (GOAL §3.1 Phase 2 freeze-schedule planner).

Phase 2 sweeps three schedule degrees — order / depth / timing
(docs/design/10_progressive_freezing.md §4, GOAL §3.1 Phase 2) — to find the
frontier curve of valid_loss degradation vs backward-FLOPs reduction. The
planner turns a (policy, depth, timing) request into the ``frozen_at_epoch``
map that :class:`FreezeCostAccountant` already consumes, so any candidate
schedule's cost can be predicted *before* a GPU run.

All expected values are hand-computed so the GOAL Phase 2 / design §4.1
arithmetic is locked down independently of the implementation. The same
planner also produces the random-order freeze schedule that GOAL §7 / design
Phase 2 control (ii) require as the null baseline.
"""

import pytest

from src.tg_lora.freeze_cost import FreezeCostAccountant, LayerBackwardCost
from src.tg_lora.freeze_schedule import (
    VALID_POLICIES,
    FreezeSchedule,
    FreezeScheduleConfig,
)

# A representative output-side active layer set (Qwen-style 32-layer model,
# design §1.5 / GOAL §1.5: 24 GDN + 8 attention, indices 24..31).
ACTIVE = [24, 25, 26, 27, 28, 29, 30, 31]


# ---------------------------------------------------------------------------
# Output-first policy (design §5.3 candidate 1, GOAL §3.1 Phase 2 candidate 1)
# ---------------------------------------------------------------------------


class TestOutputFirst:
    def test_freezes_high_index_first_descending(self):
        # depth=3, start=4, spacing=2 -> {31:4, 30:6, 29:8}
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=3,
                start_epoch=4,
                spacing=2,
                policy="output_first",
            )
        )
        assert sched.frozen_at_epoch == {31: 4, 30: 6, 29: 8}
        # Output side freezes first: the realized order is ascending epoch.
        assert sched.order == (31, 30, 29)
        assert sched.realized_depth == 3

    def test_max_depth_truncates_candidate_order(self):
        # Only the single deepest layer requested.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=1,
                start_epoch=2,
                policy="output_first",
            )
        )
        assert sched.frozen_at_epoch == {31: 2}

    def test_spacing_spreads_freezes_across_epochs(self):
        # spacing=1 freezes one layer per epoch from the output side.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=20,
                max_depth=4,
                start_epoch=5,
                spacing=1,
                policy="output_first",
            )
        )
        assert sched.frozen_at_epoch == {31: 5, 30: 6, 29: 7, 28: 8}

    def test_drops_freezes_landing_past_num_epochs(self):
        # start=8, spacing=2, depth=3 -> epochs 8, 10, 12; num_epochs=10 keeps
        # only the first two (epoch 10 is < 10? no: 10 is not < 10 -> dropped).
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=3,
                start_epoch=8,
                spacing=2,
                policy="output_first",
            )
        )
        assert sched.frozen_at_epoch == {31: 8}  # 30@10 and 29@12 dropped
        assert sched.realized_depth == 1


# ---------------------------------------------------------------------------
# Convergence-order policy (candidate 2) and the random surrogate (control ii)
# ---------------------------------------------------------------------------


class TestConvergenceOrder:
    def test_freezes_in_explicit_stability_order(self):
        # Layers reached stability in the order [25, 30, 28]; spacing=2.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=3,
                start_epoch=3,
                spacing=2,
                policy="convergence_order",
                convergence_order=(25, 30, 28),
            )
        )
        assert sched.frozen_at_epoch == {25: 3, 30: 5, 28: 7}
        assert sched.order == (25, 30, 28)

    def test_shuffled_order_is_the_random_surrogate(self):
        # A permutation of the active set uses the identical planner path:
        # this is the GOAL §7 / design Phase 2 control-(ii) random-order
        # freeze, no extra code path required.
        shuffled = (29, 24, 31, 27)
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=20,
                max_depth=len(shuffled),
                start_epoch=0,
                spacing=1,
                policy="convergence_order",
                convergence_order=shuffled,
            )
        )
        assert sched.frozen_at_epoch == {29: 0, 24: 1, 31: 2, 27: 3}

    def test_convergence_order_truncated_by_max_depth(self):
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=20,
                max_depth=2,
                start_epoch=1,
                spacing=1,
                policy="convergence_order",
                convergence_order=(26, 30, 28),
            )
        )
        assert sched.frozen_at_epoch == {26: 1, 30: 2}


# ---------------------------------------------------------------------------
# Compromise policy (candidate 3): output-side order, stability-gated timing
# ---------------------------------------------------------------------------


class TestCompromise:
    def test_defers_freeze_until_stability_threshold(self):
        # Output-side order [31, 30, 29, ...]; start=3, spacing=1.
        # stability floor pushes layer 30 to epoch 7 (past its nominal 4) and
        # leaves the others at their nominal position.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=3,
                start_epoch=3,
                spacing=1,
                policy="compromise",
                stability_epoch={31: 2, 30: 7, 29: 4},
            )
        )
        assert sched.frozen_at_epoch == {31: 3, 30: 7, 29: 5}

    def test_layer_pushed_past_num_epochs_is_dropped(self):
        # Same as above but num_epochs=6 -> layer 30 (epoch 7) never freezes.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=6,
                max_depth=3,
                start_epoch=3,
                spacing=1,
                policy="compromise",
                stability_epoch={31: 2, 30: 7, 29: 4},
            )
        )
        assert sched.frozen_at_epoch == {31: 3, 29: 5}
        assert sched.realized_depth == 2

    def test_missing_stability_entry_means_no_extra_delay(self):
        # Layer 28 has no stability entry -> nominal position only.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=2,
                start_epoch=2,
                spacing=2,
                policy="compromise",
                stability_epoch={31: 5},  # 31 deferred to 5; 30 nominal 4
            )
        )
        assert sched.frozen_at_epoch == {31: 5, 30: 4}


# ---------------------------------------------------------------------------
# Degenerate / baseline schedules
# ---------------------------------------------------------------------------


class TestDegenerateSchedules:
    def test_max_depth_zero_freezes_nothing(self):
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=0,
                start_epoch=2,
                policy="output_first",
            )
        )
        assert sched.frozen_at_epoch == {}
        assert sched.order == ()
        assert sched.realized_depth == 0

    def test_start_epoch_past_num_epochs_freezes_nothing(self):
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=5,
                max_depth=3,
                start_epoch=5,  # nothing lands within [0, 5)
                policy="output_first",
            )
        )
        assert sched.frozen_at_epoch == {}

    def test_full_depth_freezes_every_active_layer(self):
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=100,
                max_depth=len(ACTIVE),
                start_epoch=10,
                spacing=1,
                policy="output_first",
            )
        )
        assert set(sched.frozen_at_epoch) == set(ACTIVE)
        assert sched.realized_depth == len(ACTIVE)


# ---------------------------------------------------------------------------
# Integration with FreezeCostAccountant (GOAL §5): schedule feeds the accountant
# ---------------------------------------------------------------------------


def _uniform_cost() -> LayerBackwardCost:
    # weight_grad=10, act_grad=10 -> per-step full per layer = 20.
    return LayerBackwardCost(
        weight_grad_flops=10.0,
        act_grad_flops=10.0,
        optim_state_bytes=100,
        act_grad_bytes=50,
    )


class TestCostAccountantIntegration:
    def test_schedule_feeds_accountant_reduction_rate(self):
        # 2 layers, steps_per_epoch=1, num_epochs=4 -> full = 20*2*4 = 160.
        # Output-first, depth=1, start=2 freezes layer 1 at epoch 2.
        # Layer 1 active [0,2)=2 epochs (20*2=40), frozen [2,4)=2 epochs.
        # Level 1 frozen keeps act_grad: 10*2=20. Layer1 total = 60.
        # Layer 0 never frozen: 20*4=80. Progressive = 140.
        # reduction_rate(1) = 1 - 140/160 = 0.125.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=[0, 1],
                num_epochs=4,
                max_depth=1,
                start_epoch=2,
                policy="output_first",
            )
        )
        accountant = FreezeCostAccountant(
            layer_costs={0: _uniform_cost(), 1: _uniform_cost()},
            steps_per_epoch=1,
            num_epochs=4,
            frozen_at_epoch=sched.frozen_at_epoch,
        )
        assert sched.frozen_at_epoch == {1: 2}
        assert accountant.full_backward_flops() == 160.0
        assert accountant.progressive_backward_flops(level=1) == 140.0
        assert accountant.reduction_rate(level=1) == pytest.approx(0.125)

    def test_empty_schedule_means_no_reduction(self):
        # depth=0 -> empty schedule -> progressive == full -> zero reduction.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=[0, 1],
                num_epochs=4,
                max_depth=0,
                start_epoch=2,
                policy="output_first",
            )
        )
        accountant = FreezeCostAccountant(
            layer_costs={0: _uniform_cost(), 1: _uniform_cost()},
            steps_per_epoch=1,
            num_epochs=4,
            frozen_at_epoch=sched.frozen_at_epoch,
        )
        assert accountant.reduction_rate(level=1) == 0.0
        assert accountant.reduction_rate(level=2) == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_policy_rejected(self):
        with pytest.raises(ValueError, match="policy"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=1,
                start_epoch=2,
                policy="sideways",
            )

    def test_duplicate_active_layers_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            FreezeScheduleConfig(
                active_layer_indices=[24, 24, 25],
                num_epochs=10,
                max_depth=1,
                start_epoch=2,
            )

    def test_negative_spacing_rejected(self):
        with pytest.raises(ValueError, match="spacing"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=1,
                start_epoch=2,
                spacing=0,
            )

    def test_max_depth_exceeds_active_count(self):
        with pytest.raises(ValueError, match="max_depth"):
            FreezeScheduleConfig(
                active_layer_indices=[0, 1],
                num_epochs=10,
                max_depth=3,
                start_epoch=2,
            )

    def test_convergence_order_required_for_policy(self):
        with pytest.raises(ValueError, match="convergence_order"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=2,
                start_epoch=2,
                policy="convergence_order",
            )

    def test_convergence_order_exceeds_depth_request(self):
        with pytest.raises(ValueError, match="max_depth"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=3,
                start_epoch=2,
                policy="convergence_order",
                convergence_order=(25, 30),  # only 2 entries for depth 3
            )

    def test_convergence_order_unknown_layer_rejected(self):
        with pytest.raises(ValueError, match="unknown layer"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=2,
                start_epoch=2,
                policy="convergence_order",
                convergence_order=(25, 99),  # 99 not active
            )

    def test_convergence_order_duplicate_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=2,
                start_epoch=2,
                policy="convergence_order",
                convergence_order=(25, 25),
            )

    def test_stability_epoch_wrong_policy_rejected(self):
        # stability_epoch only meaningful under compromise; supplying it under
        # output_first is a caller bug.
        with pytest.raises(ValueError, match="stability_epoch"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=2,
                start_epoch=2,
                policy="output_first",
                stability_epoch={31: 3},
            )

    def test_stability_epoch_unknown_layer_rejected(self):
        with pytest.raises(ValueError, match="unknown layer"):
            FreezeScheduleConfig(
                active_layer_indices=ACTIVE,
                num_epochs=10,
                max_depth=2,
                start_epoch=2,
                policy="compromise",
                stability_epoch={99: 3},
            )

    def test_all_policies_are_valid(self):
        # Sanity: VALID_POLICIES exactly matches the three GOAL candidates.
        assert set(VALID_POLICIES) == {
            "output_first",
            "convergence_order",
            "compromise",
        }
