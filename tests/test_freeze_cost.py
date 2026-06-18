"""Unit tests for FreezeCostAccountant (GOAL §5 freeze cost accounting).

All expected values are hand-computed so the GOAL §5 arithmetic is locked
down independently of the implementation. Reference schedule used throughout:

    2 layers, each weight_grad=10, act_grad=10  ->  per-step full = 40
    steps_per_epoch = 1, num_epochs = 4        ->  full = 160

A layer frozen at epoch f is active for epochs [0, f) and frozen for [f, 4).
"""

import pytest

from src.tg_lora.freeze_cost import (
    FreezeCostAccountant,
    FreezeCostSummary,
    LayerBackwardCost,
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


def _two_layers(frozen_at_epoch: dict[int, int] | None = None) -> FreezeCostAccountant:
    return FreezeCostAccountant(
        layer_costs={0: _cost(), 1: _cost()},
        steps_per_epoch=1,
        num_epochs=4,
        frozen_at_epoch=frozen_at_epoch or {},
    )


class TestFullBackwardFlops:
    def test_formula(self):
        acc = _two_layers()
        # 2 layers * (10 + 10) * 1 step * 4 epochs = 160
        assert acc.full_backward_flops() == 160.0

    def test_scales_with_steps_and_epochs(self):
        acc = FreezeCostAccountant(
            layer_costs={0: _cost()},
            steps_per_epoch=3,
            num_epochs=5,
            frozen_at_epoch={},
        )
        # 1 layer * 20 * 3 steps * 5 epochs = 300
        assert acc.full_backward_flops() == 300.0


class TestNoFreeze:
    def test_reduction_is_zero(self):
        acc = _two_layers()
        assert acc.reduction_rate(level=1) == 0.0
        assert acc.reduction_rate(level=2) == 0.0

    def test_progressive_equals_full(self):
        acc = _two_layers()
        assert acc.progressive_backward_flops(level=1) == acc.full_backward_flops()
        assert acc.progressive_backward_flops(level=2) == acc.full_backward_flops()


class TestFreezeFromEpochZero:
    """Both layers frozen from epoch 0: the maximum single-run freeze."""

    def test_level1_skips_only_weight_grad(self):
        # Frozen 4 epochs each: act_grad 10 * 4 = 40 per layer, * 2 = 80.
        acc = _two_layers(frozen_at_epoch={0: 0, 1: 0})
        assert acc.progressive_backward_flops(level=1) == 80.0
        assert acc.reduction_rate(level=1) == pytest.approx(1 - 80 / 160)

    def test_level2_skips_weight_and_act_grad(self):
        # Both skipped while frozen -> progressive 0 -> reduction 1.0.
        acc = _two_layers(frozen_at_epoch={0: 0, 1: 0})
        assert acc.progressive_backward_flops(level=2) == 0.0
        assert acc.reduction_rate(level=2) == 1.0


class TestSingleLayerMidFreeze:
    """Layer 1 frozen at epoch 2 — the Phase 1 single-layer analog."""

    def test_level1(self):
        # Layer 0: active all 4 -> 20 * 4 = 80.
        # Layer 1: active 2 epochs -> 20 * 2 = 40; frozen 2 -> act 10 * 2 = 20.
        # progressive = 80 + 60 = 140; reduction = 1 - 140/160 = 0.125.
        acc = _two_layers(frozen_at_epoch={1: 2})
        assert acc.progressive_backward_flops(level=1) == 140.0
        assert acc.reduction_rate(level=1) == pytest.approx(0.125)

    def test_level2(self):
        # Layer 1 frozen contributes 0 -> 80 + 40 = 120; reduction = 0.25.
        acc = _two_layers(frozen_at_epoch={1: 2})
        assert acc.progressive_backward_flops(level=2) == 120.0
        assert acc.reduction_rate(level=2) == pytest.approx(0.25)

    def test_layer0_unaffected_by_freeze(self):
        acc = _two_layers(frozen_at_epoch={1: 2})
        # Layer 0 is never frozen, so it stays active for all 4 epochs.
        assert acc._active_epochs(0) == 4


class TestLevelOrdering:
    """Level 2 skips at least as much compute as Level 1."""

    @pytest.mark.parametrize(
        "schedule",
        [{}, {0: 0, 1: 0}, {1: 2}, {0: 1}, {0: 0, 1: 3}],
    )
    def test_level2_reduction_geq_level1(self, schedule):
        acc = _two_layers(frozen_at_epoch=schedule)
        assert acc.reduction_rate(level=2) >= acc.reduction_rate(level=1)

    @pytest.mark.parametrize("level", [1, 2])
    def test_progressive_never_exceeds_full(self, level):
        for schedule in [{}, {0: 0, 1: 0}, {1: 2}, {0: 1, 1: 2}]:
            acc = _two_layers(frozen_at_epoch=schedule)
            assert acc.progressive_backward_flops(level) <= acc.full_backward_flops()


class TestPeakVramSaved:
    def test_level1_counts_only_optimizer_state(self):
        # Both frozen epoch 0: 2 layers * 100 optim bytes = 200.
        acc = _two_layers(frozen_at_epoch={0: 0, 1: 0})
        assert acc.peak_vram_saved_bytes(level=1) == 200

    def test_level2_adds_activation_gradient_buffer(self):
        # Level 2: optimizer (200) + act_grad buffer (50 * 2 = 100) = 300.
        acc = _two_layers(frozen_at_epoch={0: 0, 1: 0})
        assert acc.peak_vram_saved_bytes(level=2) == 300

    def test_counts_only_layers_frozen_during_run(self):
        # Frozen at epoch == num_epochs -> never freezes during run -> 0 saved.
        acc = _two_layers(frozen_at_epoch={0: 4, 1: 0})
        # Only layer 1 freezes; layer 0 scheduled after the run.
        assert acc.peak_vram_saved_bytes(level=2) == 100 + 50

    def test_no_freeze_saves_nothing(self):
        acc = _two_layers()
        assert acc.peak_vram_saved_bytes(level=1) == 0
        assert acc.peak_vram_saved_bytes(level=2) == 0


class TestEdgeCases:
    def test_zero_epochs_no_division_by_zero(self):
        acc = FreezeCostAccountant(
            layer_costs={0: _cost()},
            steps_per_epoch=1,
            num_epochs=0,
            frozen_at_epoch={0: 0},
        )
        assert acc.full_backward_flops() == 0.0
        assert acc.reduction_rate(level=1) == 0.0
        assert acc.reduction_rate(level=2) == 0.0
        assert acc.peak_vram_saved_bytes(level=1) == 0  # no run -> nothing freed

    def test_zero_steps_no_division_by_zero(self):
        acc = FreezeCostAccountant(
            layer_costs={0: _cost()},
            steps_per_epoch=0,
            num_epochs=4,
            frozen_at_epoch={0: 0},
        )
        assert acc.full_backward_flops() == 0.0
        assert acc.reduction_rate(level=1) == 0.0

    def test_steps_scale_full_and_progressive_equally(self):
        base = _two_layers(frozen_at_epoch={1: 2})
        scaled = FreezeCostAccountant(
            layer_costs={0: _cost(), 1: _cost()},
            steps_per_epoch=5,
            num_epochs=4,
            frozen_at_epoch={1: 2},
        )
        # 5x steps -> both totals 5x -> reduction rate unchanged.
        assert scaled.full_backward_flops() == 5 * base.full_backward_flops()
        assert scaled.reduction_rate(level=1) == pytest.approx(base.reduction_rate(1))


class TestSummary:
    def test_level1_fields_consistent(self):
        acc = _two_layers(frozen_at_epoch={1: 2})
        s = acc.summary(level=1)
        assert isinstance(s, FreezeCostSummary)
        assert s.level == 1
        assert s.full_backward_flops == acc.full_backward_flops()
        assert s.progressive_backward_flops == 140.0
        assert s.reduction_rate == pytest.approx(1 - 140 / 160)
        assert s.peak_vram_saved_bytes == acc.peak_vram_saved_bytes(level=1)

    def test_reduction_rate_identity(self):
        acc = _two_layers(frozen_at_epoch={0: 0, 1: 0})
        s = acc.summary(level=2)
        assert s.reduction_rate == pytest.approx(
            1 - s.progressive_backward_flops / s.full_backward_flops
        )


class TestValidation:
    def test_negative_steps(self):
        with pytest.raises(ValueError, match="steps_per_epoch"):
            FreezeCostAccountant(
                layer_costs={0: _cost()},
                steps_per_epoch=-1,
                num_epochs=1,
                frozen_at_epoch={},
            )

    def test_negative_epochs(self):
        with pytest.raises(ValueError, match="num_epochs"):
            FreezeCostAccountant(
                layer_costs={0: _cost()},
                steps_per_epoch=1,
                num_epochs=-1,
                frozen_at_epoch={},
            )

    def test_unknown_layer_in_schedule(self):
        with pytest.raises(KeyError, match="unknown layer"):
            FreezeCostAccountant(
                layer_costs={0: _cost()},
                steps_per_epoch=1,
                num_epochs=1,
                frozen_at_epoch={5: 0},
            )

    def test_negative_freeze_epoch(self):
        with pytest.raises(ValueError, match="non-negative"):
            FreezeCostAccountant(
                layer_costs={0: _cost()},
                steps_per_epoch=1,
                num_epochs=1,
                frozen_at_epoch={0: -1},
            )

    @pytest.mark.parametrize("bad_level", [0, 3, -1])
    def test_invalid_level(self, bad_level):
        acc = _two_layers(frozen_at_epoch={1: 2})
        with pytest.raises(ValueError, match="level"):
            acc.reduction_rate(level=bad_level)
