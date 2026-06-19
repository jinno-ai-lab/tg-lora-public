"""Unit tests for the Phase 2 freeze-frontier predictor (GOAL §3.1 Phase 2).

The frontier glues :class:`FreezeSchedule.plan` to :class:`FreezeCostAccountant`
(GOAL §5) so a candidate schedule's FLOPs/VRAM savings are predicted before any
GPU run. Reference setup, reused throughout (matches test_freeze_schedule's
hand-computed example so the two stay consistent):

    2 layers, each weight_grad=10, act_grad=10  ->  per-step full per layer = 20
    steps_per_epoch = 1, num_epochs = 4         ->  full = 160

    output_first, start=2, spacing=1:
      depth 0 -> {}                           -> reduction 0
      depth 1 -> {1: 2}                       -> progressive 140 -> reduction 0.125
      depth 2 -> {1: 2, 0: 3}                 -> progressive 130 -> reduction 0.1875
"""

import pytest

from src.tg_lora.freeze_cost import LayerBackwardCost
from src.tg_lora.freeze_frontier import (
    FrontierPoint,
    FrontierSpec,
    evaluate_schedule,
    frontier,
)


def _cost(
    weight: float = 10.0,
    act: float = 10.0,
    optim_bytes: int = 100,
    act_bytes: int = 50,
) -> LayerBackwardCost:
    return LayerBackwardCost(
        weight_grad_flops=weight,
        act_grad_flops=act,
        optim_state_bytes=optim_bytes,
        act_grad_bytes=act_bytes,
    )


def _two_layer_spec(**overrides) -> FrontierSpec:
    base = dict(
        layer_costs={0: _cost(), 1: _cost()},
        steps_per_epoch=1,
        num_epochs=4,
        active_layer_indices=(0, 1),
        start_epoch=2,
    )
    base.update(overrides)
    return FrontierSpec(**base)


# ---------------------------------------------------------------------------
# Contract glue: planner -> accountant (the production connection)
# ---------------------------------------------------------------------------


class TestEvaluateSchedule:
    def test_output_first_depth1_matches_hand_computed_reduction(self):
        # Layer 1 active [0,2)=2 (20*2=40), frozen [2,4)=2 keeps act_grad
        # (10*2=20) -> 60. Layer 0 never frozen -> 20*4=80. Progressive=140.
        pt = evaluate_schedule(_two_layer_spec(), "output_first", depth=1, level=1)
        assert pt.policy == "output_first"
        assert pt.level == 1
        assert pt.depth == 1
        assert pt.frozen_at_epoch == {1: 2}
        assert pt.full_backward_flops == 160.0
        assert pt.progressive_backward_flops == 140.0
        assert pt.reduction_rate == pytest.approx(0.125)

    def test_depth2_freezes_two_layers_higher_reduction(self):
        # {1: 2, 0: 3}: layer1=60, layer0 active[0,3)=3 (20*3=60)+frozen1(10)=70.
        # Progressive=130 -> reduction = 1 - 130/160 = 0.1875.
        pt = evaluate_schedule(_two_layer_spec(), "output_first", depth=2, level=1)
        assert pt.depth == 2
        assert pt.frozen_at_epoch == {1: 2, 0: 3}
        assert pt.progressive_backward_flops == 130.0
        assert pt.reduction_rate == pytest.approx(0.1875)

    def test_level2_suffix_cut_reduces_more_than_level1(self):
        # Same depth-1 schedule, but Level 2 cuts layer 1's act_grad too:
        # layer1=40 (active only), layer0=80 -> progressive=120 -> 0.25.
        level1 = evaluate_schedule(_two_layer_spec(), "output_first", 1, level=1)
        level2 = evaluate_schedule(_two_layer_spec(), "output_first", 1, level=2)
        assert level2.progressive_backward_flops == 120.0
        assert level2.reduction_rate == pytest.approx(0.25)
        assert level2.reduction_rate > level1.reduction_rate

    def test_reduction_rate_is_ratio_of_flops(self):
        pt = evaluate_schedule(_two_layer_spec(), "output_first", depth=1, level=1)
        assert pt.reduction_rate == pytest.approx(
            1.0 - pt.progressive_backward_flops / pt.full_backward_flops
        )

    def test_invalid_level_rejected(self):
        with pytest.raises(ValueError, match="level"):
            evaluate_schedule(_two_layer_spec(), "output_first", 1, level=3)


# ---------------------------------------------------------------------------
# The frontier sweep (GOAL §3.1 Phase 2: depth -> FLOPs reduction)
# ---------------------------------------------------------------------------


class TestFrontier:
    def test_single_policy_two_levels_has_expected_shape(self):
        # 1 policy x 2 levels x (1 origin + 2 depths) = 6 points.
        pts = frontier(_two_layer_spec(policies=("output_first",), levels=(1, 2)))
        assert len(pts) == 6
        # Sorted by (policy, level, depth) so each series is contiguous.
        keys = [(p.policy, p.level, p.depth) for p in pts]
        assert keys == sorted(keys)
        assert all(isinstance(p, FrontierPoint) for p in pts)

    def test_includes_depth0_origin_with_zero_reduction(self):
        pts = frontier(_two_layer_spec(policies=("output_first",), levels=(1,)))
        origin = pts[0]
        assert origin.depth == 0
        assert origin.frozen_at_epoch == {}
        assert origin.reduction_rate == 0.0
        assert origin.progressive_backward_flops == origin.full_backward_flops
        assert origin.peak_vram_saved_bytes == 0

    def test_reduction_monotonic_nondecreasing_in_depth(self):
        pts = frontier(_two_layer_spec(policies=("output_first",), levels=(1, 2)))
        # Group by (policy, level) and assert each series is non-decreasing.
        from collections import defaultdict

        series: dict[tuple[str, int], list[float]] = defaultdict(list)
        for p in pts:
            series[(p.policy, p.level)].append(p.reduction_rate)
        for rates in series.values():
            assert rates == sorted(rates), rates

    def test_vram_saved_monotonic_nondecreasing_in_depth(self):
        pts = frontier(_two_layer_spec(policies=("output_first",), levels=(1,)))
        level1 = [p for p in pts if p.level == 1]
        # depth 0 -> 0, depth 1 -> 100 (layer 1), depth 2 -> 200 (layers 0+1).
        vram = [p.peak_vram_saved_bytes for p in level1]
        assert vram == [0, 100, 200]

    def test_full_backward_flops_constant_across_frontier(self):
        pts = frontier(_two_layer_spec())
        fulls = {p.full_backward_flops for p in pts}
        assert fulls == {160.0}

    def test_max_depth_freezes_every_active_layer(self):
        pts = frontier(_two_layer_spec(policies=("output_first",), levels=(1,)))
        deepest = max(pts, key=lambda p: p.depth)
        assert deepest.depth == 2  # both active layers frozen
        assert set(deepest.frozen_at_epoch) == {0, 1}
        assert deepest.reduction_rate == max(p.reduction_rate for p in pts)


class TestPolicySweep:
    def test_each_policy_yields_its_own_series(self):
        spec = _two_layer_spec(
            policies=("output_first", "convergence_order"),
            levels=(1,),
            convergence_order=(0, 1),
        )
        pts = frontier(spec)
        policies_seen = {p.policy for p in pts}
        assert policies_seen == {"output_first", "convergence_order"}
        # Same number of points per policy (origin + N depths).
        from collections import Counter

        counts = Counter(p.policy for p in pts)
        assert counts["output_first"] == 3
        assert counts["convergence_order"] == 3

    def test_convergence_order_freezes_first_stable_layer_first(self):
        # convergence_order=(0, 1): depth 1 freezes layer 0, not layer 1.
        spec = _two_layer_spec(
            policies=("convergence_order",),
            levels=(1,),
            convergence_order=(0, 1),
        )
        pt = evaluate_schedule(spec, "convergence_order", depth=1, level=1)
        assert pt.frozen_at_epoch == {0: 2}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_active_layers_rejected(self):
        with pytest.raises(ValueError, match="active_layer_indices must be non-empty"):
            FrontierSpec(
                layer_costs={0: _cost()},
                steps_per_epoch=1,
                num_epochs=4,
                active_layer_indices=(),
                start_epoch=2,
            )

    def test_duplicate_active_layers_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            FrontierSpec(
                layer_costs={0: _cost()},
                steps_per_epoch=1,
                num_epochs=4,
                active_layer_indices=(0, 0),
                start_epoch=2,
            )

    def test_active_layer_missing_from_costs_rejected(self):
        with pytest.raises(ValueError, match="missing from layer_costs"):
            FrontierSpec(
                layer_costs={0: _cost()},
                steps_per_epoch=1,
                num_epochs=4,
                active_layer_indices=(0, 1),  # layer 1 has no cost
                start_epoch=2,
            )

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="policies must be a subset"):
            FrontierSpec(
                layer_costs={0: _cost(), 1: _cost()},
                steps_per_epoch=1,
                num_epochs=4,
                active_layer_indices=(0, 1),
                start_epoch=2,
                policies=("bogus",),
            )

    def test_invalid_level_rejected(self):
        with pytest.raises(ValueError, match="levels must be a subset"):
            FrontierSpec(
                layer_costs={0: _cost(), 1: _cost()},
                steps_per_epoch=1,
                num_epochs=4,
                active_layer_indices=(0, 1),
                start_epoch=2,
                levels=(1, 3),
            )

    def test_convergence_order_required_when_policy_requested(self):
        with pytest.raises(ValueError, match="requires spec.convergence_order"):
            FrontierSpec(
                layer_costs={0: _cost(), 1: _cost()},
                steps_per_epoch=1,
                num_epochs=4,
                active_layer_indices=(0, 1),
                start_epoch=2,
                policies=("convergence_order",),
            )

    def test_convergence_order_must_cover_active_set(self):
        with pytest.raises(ValueError, match="must cover the active set"):
            FrontierSpec(
                layer_costs={0: _cost(), 1: _cost()},
                steps_per_epoch=1,
                num_epochs=4,
                active_layer_indices=(0, 1),
                start_epoch=2,
                policies=("convergence_order",),
                convergence_order=(0,),  # only one layer for a 2-layer sweep
            )
