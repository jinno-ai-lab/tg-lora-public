"""Unit tests for the GOAL §4 surrogate-exceedance gate.

GOAL §4 makes the random-order freeze (design Phase 2 control-(ii)) the null any
real schedule must beat: "ランダム順フリーズ（サロゲート）を超えた削減・性能だ
けを有効と認定", and "「計算が減った」「性能が保てた」も対照を超えて初めて
主張". :func:`random_freeze_order` already feeds the identical planner/accountant
path; :func:`surrogate_exceedance` is the *judgement* — does a candidate
schedule's reduction clear the distribution of random-order surrogates over
GOAL §4's required multiple seeds?

The keystone honesty test: on a *homogeneous* stack the FLOPs reduction depends
only on (depth, timing), not order, so a real schedule cannot beat random — the
verdict is TIES, never SURPASSES. A discriminating SURPASSES needs non-uniform
per-layer costs (GOAL §1.5/§8: DeltaNet vs Attention), which the two cost-gap
cases below exercise.
"""

import pytest

from src.tg_lora.freeze_cost import LayerBackwardCost
from src.tg_lora.freeze_frontier import FrontierSpec
from src.tg_lora.freeze_surrogate_gate import (
    DEFAULT_SURROGATE_SEEDS,
    SURPASSES,
    TIES,
    UNDERSHOOTS,
    format_surrogate_exceedance,
    surrogate_exceedance,
)


def _cost(weight: float = 10.0, act: float = 10.0) -> LayerBackwardCost:
    return LayerBackwardCost(weight_grad_flops=weight, act_grad_flops=act)


def _spec(layer_costs: dict[int, LayerBackwardCost], **overrides) -> FrontierSpec:
    base: dict = {
        "steps_per_epoch": 1,
        "num_epochs": 10,
        "active_layer_indices": tuple(sorted(layer_costs)),
        "start_epoch": 0,
        "spacing": 1,
    }
    base.update(overrides)
    return FrontierSpec(layer_costs=layer_costs, **base)


# A homogeneous 4-layer stack: order cannot matter, so no real schedule beats
# random on the FLOPs axis (the honesty keystone).
_HOMOGENEOUS = {i: _cost() for i in range(4)}

# An 8-layer stack whose per-layer cost rises with index. output_first freezes
# the expensive output-side layers first (rank 0 = frozen longest = most
# savings), which is the unique FLOPs-optimal permutation by the rearrangement
# inequality — so it strictly beats every other (random) ordering. A cheap-first
# (ascending) order is the unique worst and undershoots random.
_ASCENDING_COST = {i: _cost(weight=float(i + 1) * 10.0) for i in range(8)}
_ASCENDING = tuple(range(8))           # freeze cheap-first (layer 0 .. 7)
_DESCENDING = tuple(range(7, -1, -1))   # freeze expensive-first (layer 7 .. 0)


class TestHomogeneousStackCannotBeatRandom:
    def test_flops_axis_is_ties(self):
        result = surrogate_exceedance(
            _spec(_HOMOGENEOUS), policy="output_first", depth=4
        )
        # Every surrogate freezes the same cost-profile layers over the same
        # (depth, timing), so candidate == every surrogate -> TIES, not SURPASSES.
        assert result.flops_verdict == TIES
        assert result.overall_verdict == TIES
        assert result.passes is False

    def test_candidate_equals_every_surrogate_reduction(self):
        result = surrogate_exceedance(
            _spec(_HOMOGENEOUS), policy="output_first", depth=2
        )
        assert all(
            s == pytest.approx(result.candidate_flops_reduction)
            for s in result.surrogate_flops_reductions
        )

    def test_valid_loss_axis_is_unverified_until_a_gpu_run_deposits_it(self):
        result = surrogate_exceedance(
            _spec(_HOMOGENEOUS), policy="output_first", depth=2
        )
        assert result.valid_loss_verdict is None
        assert result.valid_loss_unverified is True


class TestNonUniformCostsAreDiscriminating:
    def test_expensive_first_freeze_surpasses_random(self):
        # output_first == the unique FLOPs-optimal order for ascending costs.
        result = surrogate_exceedance(
            _spec(_ASCENDING_COST), policy="output_first", depth=8
        )
        assert result.flops_verdict == SURPASSES
        assert result.overall_verdict == SURPASSES
        assert result.passes is True
        # The candidate strictly clears the luckiest random arm.
        assert result.candidate_flops_reduction > max(result.surrogate_flops_reductions)

    def test_cheap_first_freeze_undershoots_random(self):
        # Ascending order is the unique FLOPs-worst permutation -> worse than
        # even the unluckiest random ordering.
        spec = _spec(
            _ASCENDING_COST, convergence_order=_ASCENDING, policies=("convergence_order",)
        )
        result = surrogate_exceedance(
            spec, policy="convergence_order", depth=8
        )
        assert result.flops_verdict == UNDERSHOOTS
        assert result.overall_verdict == UNDERSHOOTS
        assert result.candidate_flops_reduction < min(result.surrogate_flops_reductions)


class TestReproducibilityAndStructure:
    def test_same_seeds_give_identical_surrogates_and_verdict(self):
        spec = _spec(_ASCENDING_COST)
        a = surrogate_exceedance(spec, policy="output_first", depth=6)
        b = surrogate_exceedance(spec, policy="output_first", depth=6)
        assert a.surrogate_flops_reductions == b.surrogate_flops_reductions
        assert a.flops_verdict == b.flops_verdict
        assert a.candidate_flops_reduction == pytest.approx(b.candidate_flops_reduction)

    def test_surrogate_count_matches_seeds(self):
        result = surrogate_exceedance(
            _spec(_ASCENDING_COST), policy="output_first", depth=4, seeds=(11, 22, 33)
        )
        assert len(result.surrogate_flops_reductions) == 3
        assert result.seeds == (11, 22, 33)

    def test_default_seeds_satisfy_goal_multi_seed_requirement(self):
        # GOAL §4 "各条件は複数シードで回す": the default surrogate sample must
        # carry more than one seed, else "exceeds random" is one anecdote.
        assert len(DEFAULT_SURROGATE_SEEDS) >= 3


class TestValidLossAxisGatesOverall:
    def _surpasses_on_flops(self):
        # Reuse the discriminating case so the FLOPs axis SURPASSes by design.
        return _spec(_ASCENDING_COST)

    def test_overall_ties_when_valid_loss_does_not_surpass(self):
        # FLOPs SURPASS but valid_loss is indistinguishable from the random
        # controls -> GOAL §4 refuses the combined claim (overall TIES).
        result = surrogate_exceedance(
            self._surpasses_on_flops(),
            policy="output_first",
            depth=8,
            candidate_valid_loss=1.0,
            surrogate_valid_losses=(1.0, 1.0, 1.0, 1.0, 1.0),
        )
        assert result.flops_verdict == SURPASSES
        assert result.valid_loss_verdict == TIES
        assert result.overall_verdict == TIES
        assert result.passes is False
        assert result.valid_loss_unverified is False

    def test_overall_surpasses_only_when_both_axes_clear(self):
        # FLOPs SURPASS and the candidate's valid_loss degradation is strictly
        # below every random control's -> both axes clear -> overall SURPASSES.
        result = surrogate_exceedance(
            self._surpasses_on_flops(),
            policy="output_first",
            depth=8,
            candidate_valid_loss=0.80,  # lower (less degradation) is better
            surrogate_valid_losses=(1.0, 1.0, 1.0, 1.0, 1.0),
        )
        assert result.flops_verdict == SURPASSES
        assert result.valid_loss_verdict == SURPASSES
        assert result.overall_verdict == SURPASSES
        assert result.passes is True


class TestFormatter:
    def test_is_deterministic_and_carries_the_verdict(self):
        result = surrogate_exceedance(
            _spec(_ASCENDING_COST), policy="output_first", depth=8
        )
        text = format_surrogate_exceedance(result)
        assert text == format_surrogate_exceedance(result)
        assert result.overall_verdict in text
        assert "flops_axis" in text  # the candidate-vs-surrogate FLOPs line
        assert "surrogate" in text
        assert "schedule:" in text  # provenance: policy/depth/level/margin
