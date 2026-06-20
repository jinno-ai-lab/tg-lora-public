"""Unit tests for FreezeCostAccountant (GOAL §5 freeze cost accounting).

All expected values are hand-computed so the GOAL §5 arithmetic is locked
down independently of the implementation. Reference schedule used throughout:

    2 layers, each weight_grad=10, act_grad=10  ->  per-step full = 40
    steps_per_epoch = 1, num_epochs = 4        ->  full = 160

A layer frozen at epoch f is active for epochs [0, f) and frozen for [f, 4).
"""

import pytest

from src.tg_lora.freeze_cost import (
    LEVEL1_REALIZED_REDUCTION_CEILING,
    PROXY_VALIDATED_MAX_WIDTH,
    SPEED_GATE_THRESHOLD,
    ExtrapolationConfidence,
    FreezeCostAccountant,
    FreezeCostSummary,
    GatedReduction,
    LayerBackwardCost,
    RealizedReduction,
    SpeedGateVerdict,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PROVISIONAL_PASS,
    VERDICT_REQUIRES_SCALE_MEASUREMENT,
    extrapolation_confidence,
    gate_reduction,
    realizable_reduction,
    speed_gate_verdict,
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


class TestExtrapolationConfidence:
    """Width-extrapolation bound that stops the gate over-trusting a proxy number.

    Reference: validated_max_width = 2048 (PROXY_VALIDATED_MAX_WIDTH).

        target_width / 2048   confidence   requires_scale_measurement (floor 0.5)
        -------------------   ----------   -------------------------------------
                512             1.0           False
               2048             1.0           False
               4096             0.5           False   (exactly at the floor)
               8192             0.25          True    (below the floor)
    """

    def test_default_validated_max_matches_proxy_envelope(self):
        assert PROXY_VALIDATED_MAX_WIDTH == 2048

    @pytest.mark.parametrize(
        "width, expected_conf",
        [(512, 1.0), (1024, 1.0), (2048, 1.0), (4096, 0.5), (8192, 0.25)],
    )
    def test_confidence_is_one_inside_envelope_and_decays_beyond(
        self, width, expected_conf
    ):
        c = extrapolation_confidence(width)
        assert c.confidence == pytest.approx(expected_conf)
        assert c.extrapolation_ratio == pytest.approx(width / 2048)

    def test_confidence_clamped_to_one_when_target_below_envelope(self):
        # A sub-proxy target is still fully trusted: confidence never exceeds 1.
        c = extrapolation_confidence(256)
        assert c.confidence == 1.0
        assert c.target_width == 256

    def test_requires_scale_measurement_flips_beyond_floor(self):
        # At 2x width confidence == floor (0.5): not required, just discounted.
        assert extrapolation_confidence(4096).requires_scale_measurement is False
        # At 4x width confidence (0.25) < floor: a scale measurement is required.
        assert extrapolation_confidence(8192).requires_scale_measurement is True
        # Custom floor raises the bar: a 1.5x target now demands measurement.
        assert (
            extrapolation_confidence(
                3072, scale_measurement_floor=0.75
            ).requires_scale_measurement
            is True
        )

    def test_discount_scales_proxy_value_by_confidence(self):
        # A proxy 0.30 reduction at 4x width is credited at 0.30 * 0.25 = 0.075.
        c = extrapolation_confidence(8192)
        assert c.discount(0.30) == pytest.approx(0.075)
        # Inside the envelope the proxy is credited in full.
        assert extrapolation_confidence(512).discount(0.30) == pytest.approx(0.30)

    @pytest.mark.parametrize("bad_width", [0, -1])
    def test_non_positive_target_width_raises(self, bad_width):
        with pytest.raises(ValueError, match="target_width"):
            extrapolation_confidence(bad_width)

    def test_invalid_floor_raises(self):
        with pytest.raises(ValueError, match="scale_measurement_floor"):
            extrapolation_confidence(4096, scale_measurement_floor=0.0)
        with pytest.raises(ValueError, match="scale_measurement_floor"):
            extrapolation_confidence(4096, scale_measurement_floor=1.5)


class TestGateReduction:
    """The acceptance gate discounts the accountant's proxy reduction by width.

    Reference accountant: 2 layers (weight=act=10), steps=1, epochs=4, layer 1
    frozen at epoch 2 -> Level-2 reduction_rate = 0.25 (see TestSingleLayerMidFreeze).
    """

    def _acc(self) -> FreezeCostAccountant:
        return _two_layers(frozen_at_epoch={1: 2})

    def test_inside_envelope_trusts_proxy(self):
        # At h=2048 confidence is 1.0: effective == proxy, gate clears threshold.
        gated = gate_reduction(self._acc(), level=2, target_width=2048)
        assert isinstance(gated, GatedReduction)
        assert gated.proxy_reduction == pytest.approx(0.25)
        assert gated.confidence.confidence == 1.0
        assert gated.effective_reduction == pytest.approx(0.25)
        assert gated.requires_scale_measurement is False
        assert gated.passes(threshold=0.10) is True

    def test_nine_b_is_discounted_provisionally(self):
        # At h=4096 (9B, 2x width) confidence is 0.5: effective 0.25*0.5 = 0.125.
        # Still clears the 0.10 threshold, so the gate PASSes provisionally rather
        # than silently trusting the raw 0.25 proxy number.
        gated = gate_reduction(self._acc(), level=2, target_width=4096)
        assert gated.effective_reduction == pytest.approx(0.125)
        assert gated.requires_scale_measurement is False
        assert gated.passes(threshold=0.10) is True

    def test_proxy_fails_gate_once_discounted(self):
        # A 0.15 proxy reduction would PASS raw, but at 9B it discounts to 0.075
        # < 0.10: the gate correctly refuses to credit the proxy number at scale.
        gated = gate_reduction(self._acc(), level=2, target_width=4096)
        assert gated.proxy_reduction == pytest.approx(0.25)
        # Synthetic check: build the discounted value directly and compare to a
        # threshold the proxy would clear but the discount does not.
        discounted = ExtrapolationConfidence(
            target_width=4096,
            validated_max_width=2048,
            extrapolation_ratio=2.0,
            confidence=0.5,
            requires_scale_measurement=False,
        ).discount(0.15)
        assert discounted < 0.10

    def test_extreme_width_requires_scale_measurement(self):
        # At 4x width the gate refuses proxy-only PASS regardless of margin.
        gated = gate_reduction(self._acc(), level=2, target_width=8192)
        assert gated.requires_scale_measurement is True
        assert gated.passes(threshold=0.01) is False

    def test_gate_credits_level1_as_unrealized(self):
        # Level-1 freeze-only realizes ~0 backward reduction in vivo: the
        # weight-grad FLOPs it credits never become fewer traversals (see
        # test_progressive_freeze_invivo.py ::
        # test_accountant_level1_overstates_realizable_savings_in_vivo). The
        # gate must not credit that number: the raw arithmetic (0.125) is kept
        # for transparency, but the realizability correction caps the credited
        # reduction at 0, so the gate cannot PASS the 10% speed bar at any width.
        gated = gate_reduction(self._acc(), level=1, target_width=2048)
        assert gated.proxy_reduction == pytest.approx(0.125)  # raw arithmetic kept
        assert gated.realized_reduction == 0.0  # realizability cap
        assert gated.effective_reduction == 0.0
        assert gated.passes(threshold=0.10) is False


class TestRealizableReduction:
    """Realizability correction: only Level-2 (the trio) is realized in vivo.

    Level-1 freeze-only credits weight-grad FLOPs that never become fewer
    backward traversals, so the realized reduction is ~0 while the accountant
    reports > 0 — the overstatement the in-vivo suite pins down in
    test_progressive_freeze_invivo.py :: test_accountant_level1_overstates_
    realizable_savings_in_vivo. This is orthogonal to the width bound
    (TestExtrapolationConfidence): it is a width-independent, empirically proven
    over-trust of the accountant's own Level-1 model.
    """

    def _acc(self) -> FreezeCostAccountant:
        return _two_layers(frozen_at_epoch={1: 2})

    def test_level2_passes_through_unchanged(self):
        # The trio's suffix cut is realized exactly in vivo, so the creditable
        # reduction equals the accountant's Level-2 arithmetic.
        r = realizable_reduction(self._acc(), level=2)
        assert isinstance(r, RealizedReduction)
        assert r.proxy_reduction == pytest.approx(0.25)
        assert r.realized_reduction == r.proxy_reduction
        assert r.is_realized is True

    def test_level1_capped_to_ceiling(self):
        # Level-1 reports a positive reduction in arithmetic (0.125) but realizes
        # ~0 in vivo; the correction caps it at the Level-1 ceiling (0.0) while
        # still exposing the raw proxy figure for transparency.
        r = realizable_reduction(self._acc(), level=1)
        assert r.proxy_reduction == pytest.approx(0.125)
        assert r.realized_reduction == LEVEL1_REALIZED_REDUCTION_CEILING
        assert r.realized_reduction == 0.0
        assert r.is_realized is False

    def test_no_freeze_realizes_zero_for_both_levels(self):
        # A schedule that freezes nothing saves nothing (reduction_rate == 0);
        # the realizability correction stays at 0 for both levels.
        acc = _two_layers(frozen_at_epoch={})
        assert realizable_reduction(acc, level=1).realized_reduction == 0.0
        assert realizable_reduction(acc, level=2).realized_reduction == 0.0

    @pytest.mark.parametrize("bad_level", [0, 3, -1])
    def test_invalid_level_raises(self, bad_level):
        with pytest.raises(ValueError, match="level"):
            realizable_reduction(self._acc(), level=bad_level)


class TestSpeedGateVerdict:
    """§7 first gate judged from proxy accounting, as a graduated verdict.

    Reference accountant: layer 1 frozen at epoch 2 -> Level-2 reduction 0.25,
    Level-1 reduction 0.125 (see TestSingleLayerMidFreeze). The verdict categories
    are what a bare ``passes()`` boolean cannot express, and are what stop the
    gate silently trusting a proxy number (10_guard_experiment.md §6.1/§6.2/§7).
    """

    def _acc(self) -> FreezeCostAccountant:
        return _two_layers(frozen_at_epoch={1: 2})

    def test_default_threshold_is_ten_percent(self):
        assert SPEED_GATE_THRESHOLD == 0.10

    def test_validated_width_is_clean_pass(self):
        # h=2048: confidence 1.0, effective 0.25 >= 0.10 -> a clean, full PASS.
        v = speed_gate_verdict(self._acc(), level=2, target_width=2048)
        assert isinstance(v, SpeedGateVerdict)
        assert v.verdict == VERDICT_PASS
        assert v.effective_reduction == pytest.approx(0.25)
        assert v.passes is True
        assert v.requires_scale_measurement is False

    def test_nine_b_passes_only_provisionally(self):
        # h=4096 (9B, 2x width): confidence 0.5 -> effective 0.125 still clears
        # 0.10, but at a partly-validated width, so PROVISIONAL_PASS — the proxy
        # number is credited partly, never silently in full.
        v = speed_gate_verdict(self._acc(), level=2, target_width=4096)
        assert v.verdict == VERDICT_PROVISIONAL_PASS
        assert v.effective_reduction == pytest.approx(0.125)
        assert v.confidence.confidence == pytest.approx(0.5)
        assert v.passes is True

    def test_extreme_width_requires_scale_measurement(self):
        # h=8192 (4x width): the gate refuses any verdict from the proxy and
        # demands a real measurement — it draws neither PASS nor FAIL.
        v = speed_gate_verdict(self._acc(), level=2, target_width=8192)
        assert v.verdict == VERDICT_REQUIRES_SCALE_MEASUREMENT
        assert v.passes is False
        assert v.requires_scale_measurement is True

    def test_no_realizable_reduction_fails_at_validated_width(self):
        # A layer scheduled to freeze at/after the run end realizes nothing: at
        # the fully-validated width the gate correctly FAILs on zero reduction.
        acc = _two_layers(frozen_at_epoch={1: 4})
        v = speed_gate_verdict(acc, level=2, target_width=2048)
        assert v.verdict == VERDICT_FAIL
        assert v.effective_reduction == 0.0
        assert v.passes is False

    def test_level1_always_fails_regardless_of_width(self):
        # Level-1 realizes ~0 in vivo (§6.2); the gate credits none of it, at any
        # width — even at 4x width this is a realizability FAIL, not a "needs
        # scale measurement" (the reduction is zero either way).
        for width in (2048, 4096, 8192):
            v = speed_gate_verdict(self._acc(), level=1, target_width=width)
            assert v.verdict == VERDICT_FAIL
            assert v.realized_reduction == 0.0
            assert v.effective_reduction == 0.0
            assert v.passes is False

    def test_custom_threshold_can_flip_pass_to_fail(self):
        # A 0.20 bar makes the 9B effective 0.125 FAIL even though it would
        # PROVISIONAL_PASS under the default 0.10 bar.
        v = speed_gate_verdict(self._acc(), level=2, target_width=4096, threshold=0.20)
        assert v.verdict == VERDICT_FAIL

    def test_provenance_fields_expose_full_audit_trail(self):
        # The verdict keeps every figure an auditor needs: the raw proxy, the
        # realized figure, the discounted effective, the confidence, the width.
        v = speed_gate_verdict(self._acc(), level=2, target_width=4096)
        assert v.target_width == 4096
        assert v.proxy_reduction == pytest.approx(0.25)  # raw, kept for transparency
        assert v.realized_reduction == pytest.approx(0.25)  # Level-2 realized fully
        assert v.threshold == 0.10
