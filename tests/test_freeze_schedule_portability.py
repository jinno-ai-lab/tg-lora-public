"""Schedule-portability tests — GOAL §3.1 Phase 4 / §4 step 5 (MS-PF4 Category-A).

Phase 4 is the cross-condition check: "is the optimal schedule found in
Phase 2/3 *portable* — does the same procedure still resolve when applied to a
different condition (LR / data / r / seed)?" GOAL §3.1 Phase 4 is explicit
(line 177) that what is reused is the **手順 (procedure: when and how many
layers to freeze)**, *not* the specific cached activations ("target xin の使い
回し" is out of scope). The procedure is predicted to be more robust to
condition drift than concrete activations.

GPU is required to measure the *valid_loss* axis of portability (Category-C,
blocked). The **手順-resolution axis is pure CPU (Category-A)** and is what
these tests lock down: a single :class:`ScheduleProcedure` (policy / depth /
timing, independent of the layer set) re-bound to several different
``active_layer_indices`` sets must

* resolve without crashing for any set with at least ``max_depth`` layers,
* produce an *identical* freeze-epoch sequence across equal-size sets (the
  手順 is invariant; only the absolute layer indices change),
* freeze the output-side suffix regardless of the absolute indices,
* degrade *safely* — a smaller set yields a clear ``ValueError``, never a
  silent planner crash, and
* feed :class:`FreezeCostAccountant` with an identical reduction rate across
  equal-size uniform-cost sets (the whole candidate-vs-cost pipeline is
  index-independent).

All expected epoch sequences are hand-computed so the GOAL Phase 4 arithmetic
is locked down independently of the implementation.
"""

import pytest

from src.tg_lora.freeze_cost import FreezeCostAccountant, LayerBackwardCost
from src.tg_lora.freeze_schedule import FreezeSchedule, ScheduleProcedure

# Two disjoint equal-size active sets — same procedure applied to each is the
# Phase 4 "different condition" proxy (a different run's trainable layer set).
SET_A = [24, 25, 26, 27, 28, 29, 30, 31]  # Qwen-style output block (design §1.5)
SET_B = [0, 1, 2, 3, 4, 5, 6, 7]  # a different condition's active set, same size
SET_C = [100, 101, 102, 103]  # smaller, disjoint, non-contiguous offsets


class TestProcedureInvariance:
    """The 手順 is invariant to which condition's layer set it binds to."""

    def test_identical_freeze_epoch_sequence_across_equal_size_sets(self):
        # output_first, depth=3, start=4, spacing=2 -> epochs {4, 6, 8} for the
        # top-3 layers, regardless of the absolute indices.
        proc = ScheduleProcedure(
            policy="output_first", max_depth=3, start_epoch=4, spacing=2
        )
        sched_a = FreezeSchedule.plan(proc.bind(SET_A, num_epochs=10))
        sched_b = FreezeSchedule.plan(proc.bind(SET_B, num_epochs=10))
        # Epoch sequence identical — only the layer identities differ.
        assert sorted(sched_a.frozen_at_epoch.values()) == [4, 6, 8]
        assert sorted(sched_b.frozen_at_epoch.values()) == [4, 6, 8]
        # Realized depth is portable: depth bound does not leak with the set.
        assert sched_a.realized_depth == sched_b.realized_depth == 3

    def test_output_side_suffix_portable_regardless_of_absolute_indices(self):
        # Under output_first the frozen layers are always the output-side suffix
        # (top-max_depth by index), independent of the absolute numbering.
        proc = ScheduleProcedure(
            policy="output_first", max_depth=3, start_epoch=4, spacing=2
        )
        sched_a = FreezeSchedule.plan(proc.bind(SET_A, num_epochs=10))
        sched_b = FreezeSchedule.plan(proc.bind(SET_B, num_epochs=10))
        assert set(sched_a.frozen_at_epoch) == {31, 30, 29}
        assert set(sched_b.frozen_at_epoch) == {7, 6, 5}

    def test_procedure_binds_to_a_noncontiguous_offset_set(self):
        # A condition whose active set lives at arbitrary offsets still resolves
        # to the same epoch sequence; portability is not an artifact of the
        # canonical 24..31 block.
        proc = ScheduleProcedure(
            policy="output_first", max_depth=2, start_epoch=1, spacing=3
        )
        sched_c = FreezeSchedule.plan(proc.bind(SET_C, num_epochs=20))
        assert set(sched_c.frozen_at_epoch) == {103, 102}
        assert sorted(sched_c.frozen_at_epoch.values()) == [1, 4]

    def test_run_length_is_the_only_thing_that_truncates(self):
        # Same procedure, shorter run: freezes landing past num_epochs drop
        # identically across conditions — the truncation rule is portable too.
        proc = ScheduleProcedure(
            policy="output_first", max_depth=3, start_epoch=8, spacing=2
        )
        sched_a = FreezeSchedule.plan(proc.bind(SET_A, num_epochs=10))
        sched_b = FreezeSchedule.plan(proc.bind(SET_B, num_epochs=10))
        # epochs 8, 10, 12 -> only the first lands within [0, 10).
        assert sched_a.frozen_at_epoch == {31: 8}
        assert sched_b.frozen_at_epoch == {7: 8}
        assert sched_a.realized_depth == sched_b.realized_depth == 1


class TestGracefulDegradation:
    """Portability means the procedure degrades safely on a new condition."""

    def test_smaller_active_set_rejected_with_clear_error(self):
        # A condition with fewer trainable layers than max_depth cannot host the
        # procedure. It must raise a clear ValueError — never crash the planner.
        proc = ScheduleProcedure(policy="output_first", max_depth=3, start_epoch=2)
        small_condition = [100, 101]  # 2 trainable layers < max_depth 3
        with pytest.raises(ValueError, match="max_depth"):
            proc.bind(small_condition, num_epochs=10)

    def test_depth_zero_procedure_is_portably_a_no_op(self):
        # The degenerate "freeze nothing" procedure is portable to any set.
        proc = ScheduleProcedure(policy="output_first", max_depth=0, start_epoch=2)
        for active in (SET_A, SET_B, SET_C):
            sched = FreezeSchedule.plan(proc.bind(active, num_epochs=10))
            assert sched.frozen_at_epoch == {}
            assert sched.realized_depth == 0


class TestConditionSpecificInputs:
    """convergence_order / stability_epoch are re-derived per condition; the
    *procedure* (policy/depth/timing) is what carries over."""

    def test_convergence_order_procedure_portable_with_redrived_order(self):
        # Same procedure (convergence_order, depth=2, start=1, spacing=1); each
        # condition re-supplies its own stability order. Epoch sequence {1, 2}
        # is invariant; the layer identities follow the per-condition order.
        proc = ScheduleProcedure(
            policy="convergence_order", max_depth=2, start_epoch=1, spacing=1
        )
        sched_a = FreezeSchedule.plan(
            proc.bind(SET_A, num_epochs=10, convergence_order=(25, 30))
        )
        sched_b = FreezeSchedule.plan(
            proc.bind(SET_B, num_epochs=10, convergence_order=(1, 5))
        )
        assert sched_a.frozen_at_epoch == {25: 1, 30: 2}
        assert sched_b.frozen_at_epoch == {1: 1, 5: 2}

    def test_convergence_order_without_fresh_order_rejected_safely(self):
        # Binding a convergence_order procedure to a new condition without
        # re-deriving the order must fail loudly, not silently plan garbage.
        proc = ScheduleProcedure(policy="convergence_order", max_depth=2, start_epoch=1)
        with pytest.raises(ValueError, match="convergence_order"):
            proc.bind(SET_B, num_epochs=10)  # no fresh order supplied

    def test_compromise_procedure_portable_with_per_condition_stability(self):
        # Same compromise procedure (depth=2, start=2, spacing=2); each condition
        # re-supplies its stability floor. Epoch sequence {4, 5} is invariant.
        proc = ScheduleProcedure(
            policy="compromise", max_depth=2, start_epoch=2, spacing=2
        )
        sched_a = FreezeSchedule.plan(
            proc.bind(SET_A, num_epochs=10, stability_epoch={31: 5})
        )
        sched_b = FreezeSchedule.plan(
            proc.bind(SET_B, num_epochs=10, stability_epoch={7: 5})
        )
        # top-2 layers: 31->max(2,5)=5, 30->4 ; 7->max(2,5)=5, 6->4
        assert sched_a.frozen_at_epoch == {31: 5, 30: 4}
        assert sched_b.frozen_at_epoch == {7: 5, 6: 4}


class TestCostAccountantPipeline:
    """The whole candidate→cost pipeline is index-independent: an equal-size
    uniform-cost condition yields an identical reduction rate."""

    def test_reduction_rate_identical_across_equal_size_conditions(self):
        # output_first, depth=1, start=2, num_epochs=4, uniform cost
        # (weight=10, act=10). The frozen layer saves weight_grad over [2,4)=2
        # epochs regardless of which layer/index it is, so reduction_rate(1)
        # must be identical across conditions. Hand value: 0.125.
        proc = ScheduleProcedure(policy="output_first", max_depth=1, start_epoch=2)

        def accountant_for(active: list[int]) -> FreezeCostAccountant:
            sched = FreezeSchedule.plan(proc.bind(active, num_epochs=4))
            cost = _uniform_cost()
            return FreezeCostAccountant(
                layer_costs={idx: cost for idx in active},
                steps_per_epoch=1,
                num_epochs=4,
                frozen_at_epoch=sched.frozen_at_epoch,
            )

        acc_a = accountant_for(SET_A[:2])  # [24, 25] -> freezes 25 @ epoch 2
        acc_b = accountant_for(SET_B[:2])  # [0, 1]   -> freezes 1  @ epoch 2
        assert acc_a.reduction_rate(level=1) == pytest.approx(0.125)
        assert acc_b.reduction_rate(level=1) == pytest.approx(0.125)
        # The strong portability claim: identical rate, disjoint indices.
        assert acc_a.reduction_rate(level=1) == acc_b.reduction_rate(level=1)


class TestProcedureValidation:
    def test_invalid_policy_rejected(self):
        with pytest.raises(ValueError, match="policy"):
            ScheduleProcedure(policy="sideways", max_depth=1, start_epoch=2)

    def test_invalid_spacing_rejected(self):
        with pytest.raises(ValueError, match="spacing"):
            ScheduleProcedure(
                policy="output_first", max_depth=1, start_epoch=2, spacing=0
            )

    def test_bind_returns_plannable_config(self):
        # bind() yields a FreezeScheduleConfig that plans end-to-end (the
        # procedure is a real schedule-producing primitive, not a stub).
        proc = ScheduleProcedure(
            policy="output_first", max_depth=2, start_epoch=1, spacing=1
        )
        cfg = proc.bind(SET_A, num_epochs=10)
        sched = FreezeSchedule.plan(cfg)
        assert isinstance(sched, FreezeSchedule)
        assert sched.realized_depth == 2


def _uniform_cost() -> LayerBackwardCost:
    # weight_grad=10, act_grad=10 -> per-step full per layer = 20.
    return LayerBackwardCost(
        weight_grad_flops=10.0,
        act_grad_flops=10.0,
        optim_state_bytes=100,
        act_grad_bytes=50,
    )
