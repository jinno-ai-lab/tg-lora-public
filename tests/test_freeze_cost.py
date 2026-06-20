"""Unit tests for FreezeCostAccountant (GOAL §5 freeze cost accounting).

All expected values are hand-computed so the GOAL §5 arithmetic is locked
down independently of the implementation. Reference schedule used throughout:

    2 layers, each weight_grad=10, act_grad=10  ->  per-step full = 40
    steps_per_epoch = 1, num_epochs = 4        ->  full = 160

A layer frozen at epoch f is active for epochs [0, f) and frozen for [f, 4).
"""

import pytest

from src.tg_lora.freeze_cost import (
    CALIBRATION_EMPIRICAL_ENVELOPE,
    CALIBRATION_NORMAL,
    LEVEL1_REALIZED_REDUCTION_CEILING,
    MIN_SAMPLE_FOR_CONFIDENCE_BAND,
    PROXY_VALIDATED_MAX_WIDTH,
    SPEED_GATE_THRESHOLD,
    ConfidenceBand,
    ExtrapolationConfidence,
    FreezeCostAccountant,
    FreezeCostSummary,
    GatedReduction,
    LayerBackwardCost,
    LevelComparison,
    RealizedReduction,
    ReductionSample,
    SpeedGateVerdict,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PROVISIONAL_PASS,
    VERDICT_REQUIRES_SCALE_MEASUREMENT,
    calibrate_reduction_band,
    compare_freeze_levels,
    extrapolation_confidence,
    format_level_comparison,
    format_reduction_band,
    format_speed_gate_verdict,
    frozen_at_epoch_from_freeze_log,
    gate_reduction,
    per_cycle_realized_reductions,
    realizable_reduction,
    speed_gate_verdict,
    uniform_layer_accountant,
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


class TestUniformLayerAccountant:
    """Homogeneous-stack first-order accountant for the §7 proxy path.

    ``reduction_rate`` is a ratio, so uniform per-layer costs give the exact
    first-order reduction for a schedule — equal to a hand-built accountant with
    explicit uniform costs. Reference: 4 layers (weight=act=1), 10 epochs, layer
    3 frozen at epoch 5.

        full     = 4 * (1+1) * 10         = 80
        L2 prog  = 3*(2*10) + (2*5 + 0*5) = 70  -> reduction 0.125
        L1 prog  = 3*(2*10) + (2*5 + 1*5) = 75  -> reduction 0.0625
    """

    def test_level2_and_level1_match_hand_calc(self):
        acc = uniform_layer_accountant(4, 10, {3: 5})
        assert acc.reduction_rate(level=2) == pytest.approx(0.125)
        assert acc.reduction_rate(level=1) == pytest.approx(0.0625)

    def test_equals_explicit_uniform_cost_accountant(self):
        # The helper is a convenience wrapper: identical to an accountant built
        # by hand with the same uniform per-layer costs.
        helper = uniform_layer_accountant(4, 10, {3: 5})
        explicit = FreezeCostAccountant(
            layer_costs={i: LayerBackwardCost(1.0, 1.0) for i in range(4)},
            steps_per_epoch=1,
            num_epochs=10,
            frozen_at_epoch={3: 5},
        )
        assert helper.full_backward_flops() == explicit.full_backward_flops()
        assert helper.reduction_rate(level=2) == pytest.approx(
            explicit.reduction_rate(level=2)
        )

    def test_feeds_speed_gate_verdict_graduation(self):
        # A schedule clearing the 10% bar raw (L2 reduction 0.125) graduates by
        # width when judged as a proxy: clean PASS at a validated width, FAIL at
        # 9B once the 0.5 discount drops 0.0625 below the bar, and
        # REQUIRES_SCALE_MEASUREMENT at 4x width regardless of margin.
        acc = uniform_layer_accountant(4, 10, {3: 5})
        assert (
            speed_gate_verdict(acc, level=2, target_width=2048).verdict == VERDICT_PASS
        )
        v9b = speed_gate_verdict(acc, level=2, target_width=4096)
        assert v9b.verdict == VERDICT_FAIL
        assert v9b.effective_reduction == pytest.approx(0.0625)
        assert (
            speed_gate_verdict(acc, level=2, target_width=8192).verdict
            == VERDICT_REQUIRES_SCALE_MEASUREMENT
        )

    @pytest.mark.parametrize("bad_n", [0, -1])
    def test_non_positive_layer_count_raises(self, bad_n):
        with pytest.raises(ValueError, match="num_layers"):
            uniform_layer_accountant(bad_n, 10, {})

    def test_unknown_layer_in_schedule_raises(self):
        # frozen_at_epoch referencing a layer >= num_layers is rejected: the
        # homogeneous model only owns range(num_layers).
        with pytest.raises(KeyError, match="unknown layer"):
            uniform_layer_accountant(4, 10, {9: 5})


class TestFrozenAtEpochFromFreezeLog:
    """Earliest-cycle parser turning a per-cycle freeze log into frozen_at_epoch."""

    def test_earliest_cycle_per_layer(self):
        # Output-side suffix growing: layer 7 first seen at cycle 2, 6 at 3, 5 at 4.
        log = {2: {7}, 3: {6, 7}, 4: {5, 6, 7}}
        assert frozen_at_epoch_from_freeze_log(log) == {7: 2, 6: 3, 5: 4}

    def test_int_value_is_single_layer(self):
        assert frozen_at_epoch_from_freeze_log({2: 7, 3: 6}) == {7: 2, 6: 3}

    def test_keeps_earliest_when_seen_again_later(self):
        # A layer observed at cycle 5 then cycle 3 keeps the earliest (cycle 3).
        assert frozen_at_epoch_from_freeze_log({5: {4}, 3: {4}}) == {4: 3}

    def test_never_seen_layer_is_omitted(self):
        log = {2: {7}, 3: {6}}
        assert frozen_at_epoch_from_freeze_log(log) == {7: 2, 6: 3}
        assert 5 not in log.get(2, set())  # layer 5 never appears

    def test_empty_log_returns_empty(self):
        assert frozen_at_epoch_from_freeze_log({}) == {}

    def test_feeds_uniform_accountant_to_proxy_verdict(self):
        # End-to-end substrate: a freeze log -> frozen_at_epoch -> uniform
        # accountant -> a concrete proxy verdict (Level-2, validated width PASS).
        # 4-layer stack: layer 3 frozen @2, layer 2 frozen @4 -> reduction 0.35.
        log = {2: {3}, 4: {2, 3}}
        acc = uniform_layer_accountant(4, 10, frozen_at_epoch_from_freeze_log(log))
        v = speed_gate_verdict(acc, level=2, target_width=2048)
        assert v.verdict == VERDICT_PASS
        assert v.realized_reduction == pytest.approx(0.35)


class TestFormatSpeedGateVerdict:
    """The §7 proxy verdict rendered as an honest, auditable provenance block.

    Reference accountant: layer 1 frozen at epoch 2 -> Level-2 reduction 0.25,
    Level-1 reduction 0.125 (see TestSingleLayerMidFreeze). The rendered text
    keeps the raw proxy visible while making clear only the effective figure is
    credited — the honesty gradation a bare boolean cannot express.
    """

    def _acc(self) -> FreezeCostAccountant:
        return _two_layers(frozen_at_epoch={1: 2})

    def test_pass_at_validated_width(self):
        text = format_speed_gate_verdict(
            speed_gate_verdict(self._acc(), level=2, target_width=2048)
        )
        assert VERDICT_PASS in text
        assert "passes=True" in text
        assert "target_width=2048" in text
        assert "effective_reduction=0.2500" in text

    def test_provisional_pass_at_nine_b(self):
        text = format_speed_gate_verdict(
            speed_gate_verdict(self._acc(), level=2, target_width=4096)
        )
        assert VERDICT_PROVISIONAL_PASS in text
        assert "confidence=0.500" in text
        assert "effective_reduction=0.1250" in text
        # Raw proxy stays visible for transparency, distinct from the credited figure.
        assert "proxy_reduction=0.2500" in text

    def test_requires_scale_measurement_at_extreme_width(self):
        text = format_speed_gate_verdict(
            speed_gate_verdict(self._acc(), level=2, target_width=8192)
        )
        assert VERDICT_REQUIRES_SCALE_MEASUREMENT in text
        assert "passes=False" in text

    def test_fail_exposes_zero_realized_for_level1(self):
        text = format_speed_gate_verdict(
            speed_gate_verdict(self._acc(), level=1, target_width=2048)
        )
        assert VERDICT_FAIL in text
        assert "realized_reduction=0.0000" in text
        assert "effective_reduction=0.0000" in text

    def test_threshold_rendered(self):
        text = format_speed_gate_verdict(
            speed_gate_verdict(self._acc(), level=2, target_width=2048)
        )
        assert "threshold=0.10" in text


class TestReductionSample:
    """Observed-reduction statistics that record the measured spread (§6.3).

    All expected values are hand-computed. Sample [0.0, 0.1, 0.2, 0.3, 0.4]:
    n=5, min=0.0, max=0.4, mean=0.2, stddev=sqrt(0.025)=0.15811…, and n >= 3 so
    it is not thin evidence (the boundary is MIN_SAMPLE_FOR_CONFIDENCE_BAND = 3).
    """

    def test_min_sample_for_confidence_band_is_three(self):
        # The steering critique ("two reproductions of a median is thin
        # evidence") is enforced as a named constant, not prose.
        assert MIN_SAMPLE_FOR_CONFIDENCE_BAND == 3

    def test_statistics_hand_computed(self):
        s = ReductionSample.from_values([0.0, 0.1, 0.2, 0.3, 0.4])
        assert s.n == 5
        assert s.min == 0.0
        assert s.max == 0.4
        assert s.mean == pytest.approx(0.2)
        assert s.stddev == pytest.approx(0.15811388, abs=1e-6)
        assert s.is_empty is False
        assert s.is_thin_evidence is False  # n=5 >= 3

    def test_thin_evidence_below_three_observations(self):
        # n=2 reproductions: too thin to call a confidence band.
        s = ReductionSample.from_values([0.1, 0.2])
        assert s.n == 2
        assert s.is_thin_evidence is True

    def test_boundary_three_is_not_thin(self):
        # Exactly MIN_SAMPLE_FOR_CONFIDENCE_BAND observations clear the bar.
        s = ReductionSample.from_values([0.0, 0.25, 0.5])
        assert s.n == 3
        assert s.is_thin_evidence is False

    def test_single_observation_is_thin(self):
        s = ReductionSample.from_values([0.3])
        assert s.n == 1
        assert s.is_thin_evidence is True
        # stddev is undefined below two observations and reports 0.0.
        assert s.stddev == 0.0
        assert s.min == s.max == s.mean == 0.3

    def test_empty_sample(self):
        s = ReductionSample.from_values([])
        assert s.is_empty is True
        assert s.n == 0
        assert s.min == 0.0 and s.max == 0.0 and s.mean == 0.0

    def test_accepts_any_iterable(self):
        # A generator (the shape per_cycle_realized_reductions returns) works.
        s = ReductionSample.from_values(x / 4 for x in range(4))
        assert s.n == 4
        assert s.observations == (0.0, 0.25, 0.5, 0.75)

    def test_rejects_negative_reduction(self):
        # A freeze cannot increase backward work, so a reduction is non-negative.
        with pytest.raises(ValueError, match="non-negative"):
            ReductionSample.from_values([-0.1, 0.2])

    def test_from_runs_accumulates_across_runs(self):
        # Each positional series is one run's per-cycle observations; they flatten
        # into one sample so the band is calibrated *across runs* (§6.3), not over
        # a single run's ramp — the steering feedback's 'across runs' path.
        s = ReductionSample.from_runs([0.0, 0.25], [0.1, 0.4])
        assert s.observations == (0.0, 0.25, 0.1, 0.4)
        assert s.n == 4
        assert s.min == 0.0
        assert s.max == 0.4

    def test_from_runs_single_series_matches_from_values(self):
        assert ReductionSample.from_runs([0.0, 0.25, 0.5]) == ReductionSample.from_values(
            [0.0, 0.25, 0.5]
        )

    def test_from_runs_rejects_negative(self):
        # The non-negative invariant holds across the combined series too.
        with pytest.raises(ValueError, match="non-negative"):
            ReductionSample.from_runs([0.1, 0.2], [-0.05, 0.3])


class TestCalibrateReductionBand:
    """Band width calibrated against the sample's measured spread (§6.3).

    Reference sample [0.0, 0.25, 0.5]: n=3, mean=0.25, stddev=0.25. The
    empirical envelope is [min, max] = [0.0, 0.5]; the normal band is
    mean ± z·stddev. This retires the containment-only band the steering
    feedback named — the width now comes from what was measured, and thin
    evidence is flagged rather than printed as a confidence band.
    """

    def _sample(self) -> ReductionSample:
        return ReductionSample.from_values([0.0, 0.25, 0.5])

    def test_empirical_envelope_uses_observed_range(self):
        band = calibrate_reduction_band(self._sample())
        assert isinstance(band, ConfidenceBand)
        assert band.method == CALIBRATION_EMPIRICAL_ENVELOPE  # default
        assert band.lower == 0.0
        assert band.upper == 0.5
        assert band.center == pytest.approx(0.25)
        assert band.half_width == pytest.approx(0.25)
        assert band.width == pytest.approx(0.5)
        assert band.n == 3
        assert band.is_thin_evidence is False

    def test_normal_band_is_mean_plus_minus_z_stddev(self):
        # stddev=0.25, z=2 -> half_width 0.5 -> [-0.25, 0.75].
        band = calibrate_reduction_band(
            self._sample(), method=CALIBRATION_NORMAL, z=2.0
        )
        assert band.method == CALIBRATION_NORMAL
        assert band.lower == pytest.approx(-0.25)
        assert band.upper == pytest.approx(0.75)
        assert band.center == pytest.approx(0.25)
        assert band.width == pytest.approx(1.0)
        # Symmetric normal interval may dip below zero for a low-mean sample —
        # reductions are non-negative, so prefer the empirical envelope then.
        assert band.lower < 0.0

    def test_contains_inclusive_bounds(self):
        band = calibrate_reduction_band(self._sample())
        assert band.contains(0.0) is True
        assert band.contains(0.5) is True
        assert band.contains(0.25) is True
        assert band.contains(-0.001) is False
        assert band.contains(0.501) is False

    def test_thin_evidence_is_flagged_not_hidden(self):
        # Two observations: the band is still computed for the record, but it is
        # explicitly thin evidence — a gate must not present it as calibrated.
        band = calibrate_reduction_band(ReductionSample.from_values([0.1, 0.2]))
        assert band.is_thin_evidence is True
        assert band.n == 2
        assert band.lower == 0.1 and band.upper == 0.2

    def test_single_observation_collapses_to_a_point(self):
        band = calibrate_reduction_band(ReductionSample.from_values([0.3]))
        assert band.is_thin_evidence is True
        assert band.lower == band.upper == 0.3
        assert band.width == 0.0

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calibrate_reduction_band(ReductionSample.from_values([]))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            calibrate_reduction_band(self._sample(), method="quartile")

    def test_non_positive_z_raises(self):
        with pytest.raises(ValueError, match="z"):
            calibrate_reduction_band(self._sample(), method=CALIBRATION_NORMAL, z=0.0)

    def test_format_records_provenance_and_thin_flag(self):
        calibrated = format_reduction_band(calibrate_reduction_band(self._sample()))
        assert "empirical_envelope" in calibrated
        assert "calibrated" in calibrated
        assert "n=3" in calibrated
        assert "width=0.5000" in calibrated
        thin = format_reduction_band(
            calibrate_reduction_band(ReductionSample.from_values([0.1, 0.2]))
        )
        assert "THIN_EVIDENCE" in thin
        assert "n=2" in thin

    def test_band_carries_measured_spread_provenance(self):
        # The §6.3 audit must record the full measured spread — min/max/stddev —
        # not only the calibrated interval. stddev was computed on the sample but
        # lost on the band; surfacing it retires that gap (steering feedback:
        # "min/max/stddev and N"). Reference [0.0, 0.25, 0.5]: stddev 0.25.
        band = calibrate_reduction_band(self._sample())
        assert band.min_obs == 0.0
        assert band.max_obs == 0.5
        assert band.stddev == pytest.approx(0.25)

    def test_normal_band_keeps_raw_min_max_distinct_from_interval(self):
        # For the normal method the interval (mean ± z·stddev) is NOT the observed
        # range; the band must still carry the raw min/max so the audit shows both
        # the calibrated interval and what was actually observed.
        band = calibrate_reduction_band(
            self._sample(), method=CALIBRATION_NORMAL, z=2.0
        )
        assert band.lower == pytest.approx(-0.25)  # interval can dip below zero
        assert band.min_obs == 0.0  # ...but the observed range stays non-negative
        assert band.max_obs == 0.5
        assert band.stddev == pytest.approx(0.25)

    def test_format_records_measured_spread_line(self):
        out = format_reduction_band(calibrate_reduction_band(self._sample()))
        # The headline interval line is unchanged ...
        assert "lower=0.0000" in out
        assert "upper=0.5000" in out
        # ... and a measured_spread line records min/max/mean/stddev explicitly,
        # so the headline number is never presented without its measured spread.
        assert "measured_spread:" in out
        assert "min=0.0000" in out
        assert "max=0.5000" in out
        assert "stddev=0.2500" in out


class TestPerCycleRealizedReductions:
    """The per-cycle observed spread a confidence band is calibrated over (§6.3).

    Reference accountant: layer 1 frozen at epoch 2 -> Level-2 reduction 0.25
    (see TestSingleLayerMidFreeze). As the suffix freezes, the realized Level-2
    reduction ramps cycle by cycle; Level-1 stays ~0 (the §6.2 ceiling).
    """

    def _acc(self) -> FreezeCostAccountant:
        return _two_layers(frozen_at_epoch={1: 2})

    def test_level2_ramps_as_suffix_freezes(self):
        # Cycle t counts layers frozen by epoch <= t: nothing until t=2, then the
        # full 0.25 reduction is realized and held to the end of the run.
        series = per_cycle_realized_reductions(self._acc(), level=2, num_cycles=4)
        assert series == [0.0, 0.0, pytest.approx(0.25), pytest.approx(0.25)]

    def test_level1_stays_zero_every_cycle(self):
        # Level-1 realizes ~0 in vivo regardless of cycle (the §6.2 ceiling).
        series = per_cycle_realized_reductions(self._acc(), level=1, num_cycles=4)
        assert series == [0.0, 0.0, 0.0, 0.0]

    def test_default_num_cycles_is_num_epochs(self):
        assert len(per_cycle_realized_reductions(self._acc(), level=2)) == 4

    def test_multi_freeze_ramp(self):
        # layer0 frozen@1, layer1 frozen@3: reduction steps 0 -> 0.375 -> 0.5.
        # t=1: only layer0 frozen -> 1 - (20 + 80)/160 = 0.375.
        # t=3: both frozen -> 1 - (20 + 60)/160 = 0.5.
        acc = _two_layers(frozen_at_epoch={0: 1, 1: 3})
        series = per_cycle_realized_reductions(acc, level=2, num_cycles=4)
        assert series == [
            0.0,
            pytest.approx(0.375),
            pytest.approx(0.375),
            pytest.approx(0.5),
        ]

    @pytest.mark.parametrize("bad_level", [0, 3, -1])
    def test_invalid_level_raises(self, bad_level):
        with pytest.raises(ValueError, match="level"):
            per_cycle_realized_reductions(self._acc(), level=bad_level)

    def test_feeds_calibrated_band_containing_headline(self):
        # The headline realized reduction (0.25) sits at the band's upper edge,
        # and the band records the full per-cycle spread rather than one number.
        series = per_cycle_realized_reductions(self._acc(), level=2, num_cycles=4)
        band = calibrate_reduction_band(ReductionSample.from_values(series))
        assert band.is_thin_evidence is False  # n=4
        assert band.contains(
            realizable_reduction(self._acc(), level=2).realized_reduction
        )
        assert band.upper == pytest.approx(0.25)
        assert band.lower == 0.0

    def test_few_cycles_is_thin_evidence(self):
        # Two cycles before any freeze: the sample is too thin for a band.
        series = per_cycle_realized_reductions(self._acc(), level=2, num_cycles=2)
        band = calibrate_reduction_band(ReductionSample.from_values(series))
        assert band.is_thin_evidence is True
        assert band.n == 2

    def test_cross_run_band_calibrated_over_combined_spread(self):
        # End-to-end across-runs path (§6.3): two runs' per-cycle series feed one
        # sample, so N counts every observed cycle across both runs and the band
        # spans their combined measured spread — not one run's ramp in isolation.
        run_a = per_cycle_realized_reductions(self._acc(), level=2, num_cycles=4)
        run_b = per_cycle_realized_reductions(
            _two_layers(frozen_at_epoch={1: 1}), level=2, num_cycles=4
        )
        combined = ReductionSample.from_runs(run_a, run_b)
        assert combined.observations == tuple(run_a) + tuple(run_b)
        band = calibrate_reduction_band(combined)
        assert band.n == len(run_a) + len(run_b)
        assert band.stddev > 0.0


class TestCompareFreezeLevels:
    """Level-1-vs-Level-2 quantitative comparison (GOAL §5 / §1.6.3 / Phase 3).

    Reference accountant: ``_two_layers(frozen_at_epoch={1: 2})`` (see module
    docstring). Layer 1 frozen at epoch 2 -> Level-1 arithmetic reduction 0.125
    (realized ~0 under the §6.2 ceiling), Level-2 arithmetic reduction 0.25
    (realized exactly). The comparison bundles both verdicts and the marginal
    reduction Level 2's suffix cut buys on top of the Level 1 baseline — the
    quantity GOAL Phase 3 weighs against Level 2's proxy-loss quality risk.
    """

    def _acc(self) -> FreezeCostAccountant:
        return _two_layers(frozen_at_epoch={1: 2})

    def test_marginal_deltas_match_verdict_difference(self):
        # The additional_* fields are exactly level2 minus level1 across all
        # three provenance quantities, derived from the bundled verdicts.
        comp = compare_freeze_levels(self._acc(), target_width=2048)
        assert comp.additional_arithmetic_reduction == pytest.approx(
            comp.level2.proxy_reduction - comp.level1.proxy_reduction
        )
        assert comp.additional_realized_reduction == pytest.approx(
            comp.level2.realized_reduction - comp.level1.realized_reduction
        )
        assert comp.additional_effective_reduction == pytest.approx(
            comp.level2.effective_reduction - comp.level1.effective_reduction
        )

    def test_level1_realizes_zero_so_additional_realized_is_level2(self):
        # §6.2: Level 1 realizes ~0 in vivo regardless of width, so the entire
        # realizable backward reduction is carried by Level 2's suffix cut.
        # The marginal realized reduction therefore equals Level 2's realized.
        comp = compare_freeze_levels(self._acc(), target_width=2048)
        assert comp.level1.realized_reduction == pytest.approx(0.0)
        assert comp.additional_realized_reduction == pytest.approx(
            comp.level2.realized_reduction
        )
        assert comp.additional_realized_reduction == pytest.approx(0.25)

    def test_additional_arithmetic_positive_with_activation_grad_cost(self):
        # Weight-grad AND act-grad both cost 10: skipping act-grad on top of
        # weight-grad adds the 0.125 arithmetic reduction (0.25 - 0.125).
        comp = compare_freeze_levels(self._acc(), target_width=2048)
        assert comp.level1.proxy_reduction == pytest.approx(0.125)
        assert comp.level2.proxy_reduction == pytest.approx(0.25)
        assert comp.additional_arithmetic_reduction == pytest.approx(0.125)

    def test_additional_arithmetic_zero_without_activation_grad_cost(self):
        # No activation-gradient work to skip -> Level 1 and Level 2 agree
        # arithmetically, so the extra cut buys zero additional FLOPs. The §6.2
        # ceiling still holds Level 1's realization at 0, so the suffix cut is
        # nonetheless the only level whose reduction realizes in vivo.
        acc = FreezeCostAccountant(
            layer_costs={0: _cost(act=0.0), 1: _cost(act=0.0)},
            steps_per_epoch=1,
            num_epochs=4,
            frozen_at_epoch={1: 2},
        )
        comp = compare_freeze_levels(acc, target_width=2048)
        assert comp.additional_arithmetic_reduction == pytest.approx(0.0)
        assert comp.level1.realized_reduction == pytest.approx(0.0)
        assert comp.additional_realized_reduction == pytest.approx(
            comp.level2.realized_reduction
        )

    def test_additional_passes_at_validated_and_nine_b_widths(self):
        # Level 1 always FAILs (realized ~0); Level 2 PASSes at the validated
        # width and PROVISIONAL_PASSes at 9B (2x). In both cases the suffix cut
        # is what carries the gate, so additional_passes is True.
        assert (
            compare_freeze_levels(self._acc(), target_width=2048).additional_passes
            is True
        )
        nine_b = compare_freeze_levels(self._acc(), target_width=4096)
        assert nine_b.level1.verdict == VERDICT_FAIL
        assert nine_b.level2.verdict == VERDICT_PROVISIONAL_PASS
        assert nine_b.additional_passes is True

    def test_additional_passes_false_when_level2_requires_scale(self):
        # At 4x width Level 2 refuses to PASS on the proxy alone, so even though
        # it is the only level that could realize reduction, additional_passes
        # is False — a CUDA/scale measurement is mandatory first.
        extreme = compare_freeze_levels(self._acc(), target_width=8192)
        assert extreme.level1.verdict == VERDICT_FAIL
        assert extreme.level2.verdict == VERDICT_REQUIRES_SCALE_MEASUREMENT
        assert extreme.additional_passes is False

    def test_verdicts_share_target_width_threshold_and_confidence(self):
        # Both levels are judged at the same width, so their width-confidence
        # (§6.1) is identical — the marginal numbers are an apples-to-apples
        # delta, not a confound of differing gate settings.
        comp = compare_freeze_levels(self._acc(), target_width=4096)
        assert comp.level1.target_width == comp.target_width
        assert comp.level2.target_width == comp.target_width
        assert comp.level1.threshold == comp.level2.threshold == SPEED_GATE_THRESHOLD
        assert (
            comp.level1.confidence.confidence
            == comp.level2.confidence.confidence
            == pytest.approx(0.5)
        )

    def test_agrees_with_independently_built_verdicts(self):
        # The comparison's verdicts are identical to building each level's
        # verdict by hand — the function is pure bundling, no extra arithmetic.
        acc = self._acc()
        comp = compare_freeze_levels(acc, target_width=2048)
        assert isinstance(comp, LevelComparison)
        assert comp.level1 == speed_gate_verdict(acc, level=1, target_width=2048)
        assert comp.level2 == speed_gate_verdict(acc, level=2, target_width=2048)

    def test_uniform_accountant_substrate_feeds_comparison(self):
        # The §7 proxy substrate (freeze log -> accountant -> verdict) reaches
        # the comparison too: a homogeneous-stack schedule produces a comparison
        # whose marginal realized reduction is exactly Level 2's realized.
        log = {2: {3}, 4: {2, 3}}
        acc = uniform_layer_accountant(
            4, 10, frozen_at_epoch_from_freeze_log(log)
        )
        comp = compare_freeze_levels(acc, target_width=2048)
        assert comp.level1.realized_reduction == pytest.approx(0.0)
        assert comp.additional_realized_reduction == pytest.approx(
            comp.level2.realized_reduction
        )
        assert comp.additional_realized_reduction > 0.0

    def test_marginal_deltas_are_non_negative(self):
        # Level 2 skips a superset of Level 1's skipped work, so every marginal
        # delta is non-negative across a range of schedules and widths.
        schedules = [
            {1: 2},
            {0: 1, 1: 3},
            {},
        ]
        for frozen in schedules:
            acc = _two_layers(frozen_at_epoch=frozen)
            for width in (2048, 4096, 8192):
                comp = compare_freeze_levels(acc, target_width=width)
                assert comp.additional_arithmetic_reduction >= 0.0
                assert comp.additional_realized_reduction >= 0.0
                assert comp.additional_effective_reduction >= -1e-12

    def test_format_renders_both_verdicts_and_marginal_line(self):
        comp = compare_freeze_levels(self._acc(), target_width=2048)
        text = format_level_comparison(comp)
        assert "level_comparison: target_width=2048" in text
        assert "level1 (progressive freeze):" in text
        assert "level2 (suffix cut):" in text
        assert "additional (level2 - level1):" in text
        # Both verdict categories and the hand-computed marginal realized (0.25)
        # appear in the rendered audit block.
        assert VERDICT_FAIL in text
        assert VERDICT_PASS in text
        assert "realized=0.2500" in text
