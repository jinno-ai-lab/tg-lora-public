"""Unit tests for the GOAL §3.1 Phase-3 loss-ablation harness.

The harness (:mod:`src.tg_lora.loss_ablation`) is the Category-A scaffold for
the Phase-3 loss-function ablation (GOAL §3.1 line 166:
"MSE 単独 vs MSE+cos vs 分布も合わせる版"). The loss terms themselves are
verified in ``test_activation_matching.py``; here we verify the *harness*: that
the three arms are named and pinned, that weighting is config-driven, and that
the side-by-side sweep is observable and faithful to the underlying combiner
(GOAL §7: observe the mechanism on CPU before any GPU run).
"""

import pytest
import torch

from src.tg_lora.activation_matching import ActivationMatchingLoss
from src.tg_lora.loss_ablation import (
    LOSS_ARM_NAMES,
    LOSS_ARMS,
    LossArmConfig,
    build_matching_loss,
    run_loss_ablation,
)


class TestNamedArms:
    def test_canonical_three_arms(self):
        # GOAL line 166 names exactly these three arms, in this ablation order.
        assert LOSS_ARM_NAMES == ("mse", "mse_cos", "dist")

    def test_mse_arm_is_pure_mse(self):
        # MSE 単独: only the base term, Phase-1 gate.
        assert LOSS_ARMS["mse"] == LossArmConfig(arm="mse").resolve_weights()
        w = LOSS_ARMS["mse"]
        assert (w.mse, w.cosine, w.dist) == (1.0, 0.0, 0.0)

    def test_mse_cos_adds_cosine(self):
        # MSE+cos: base plus exactly the cosine term.
        w = LOSS_ARMS["mse_cos"]
        assert (w.mse, w.cosine, w.dist) == (1.0, 1.0, 0.0)

    def test_dist_arm_adds_distribution(self):
        # 分布も合わせる版: base plus the distribution term (factorial, not
        # cumulative — isolates distribution's marginal contribution).
        w = LOSS_ARMS["dist"]
        assert (w.mse, w.cosine, w.dist) == (1.0, 0.0, 1.0)


class TestLossArmConfig:
    def test_unknown_arm_rejected(self):
        with pytest.raises(ValueError, match="unknown ablation arm"):
            LossArmConfig(arm="nope")

    def test_preset_resolves_without_overrides(self):
        assert LossArmConfig(arm="mse_cos").resolve_weights() == LOSS_ARMS["mse_cos"]

    def test_explicit_override_applies_on_top_of_preset(self):
        # arm pins cosine=1; the override raises dist to 2 while keeping cosine.
        w = LossArmConfig(arm="mse_cos", dist_weight=2.0).resolve_weights()
        assert (w.mse, w.cosine, w.dist) == (1.0, 1.0, 2.0)

    def test_arm_none_falls_back_to_explicit_weights(self):
        # No preset -> explicit weights ARE the config (defaults to pure MSE).
        assert LossArmConfig(arm=None).resolve_weights() == LOSS_ARMS["mse"]
        w = LossArmConfig(arm=None, mse_weight=0.0, dist_weight=1.0).resolve_weights()
        assert (w.mse, w.cosine, w.dist) == (0.0, 0.0, 1.0)

    def test_build_delegates_weight_validation(self):
        # An all-zero config is rejected by the combiner at build time.
        with pytest.raises(ValueError, match="at least one matching weight"):
            build_matching_loss(
                LossArmConfig(arm=None, mse_weight=0.0, cosine_weight=0.0, dist_weight=0.0)
            )


class TestBuildMatchingLoss:
    def test_build_matches_combiner_with_same_weights(self):
        cfg = LossArmConfig(arm="mse_cos")
        predicted = torch.randn(4, 3, 6)
        target = torch.randn(4, 3, 6)
        via_config = build_matching_loss(cfg)
        by_hand = ActivationMatchingLoss(mse_weight=1.0, cosine_weight=1.0, dist_weight=0.0)
        assert torch.allclose(via_config(predicted, target), by_hand(predicted, target))


class TestRunLossAblation:
    def test_default_runs_canonical_three(self):
        predicted = torch.randn(4, 2, 5)
        target = torch.randn(4, 2, 5)
        result = run_loss_ablation(predicted, target)
        assert [r.arm for r in result.arms] == ["mse", "mse_cos", "dist"]
        assert result.loss_by_arm().keys() == {"mse", "mse_cos", "dist"}

    def test_breakdown_sums_to_loss(self):
        # The detached harness breakdown must match the differentiable combiner
        # exactly — the harness is faithful, not an approximation.
        predicted = torch.randn(6, 3, 4)
        target = torch.randn(6, 3, 4)
        result = run_loss_ablation(predicted, target)
        for arm in result.arms:
            assert torch.allclose(arm.loss, arm.breakdown.total)
            # And faithful to a directly-built combiner on the same input.
            direct = build_matching_loss(LossArmConfig(arm=arm.arm))(predicted, target)
            assert torch.allclose(arm.loss, direct)

    def test_mse_arm_carries_only_mse_term(self):
        predicted = torch.randn(4, 3, 5)
        target = torch.randn(4, 3, 5)
        mse_arm = run_loss_ablation(predicted, target, arms=["mse"]).arms[0]
        assert mse_arm.breakdown.cosine == 0.0
        assert mse_arm.breakdown.dist == 0.0
        assert torch.allclose(mse_arm.loss, mse_arm.breakdown.mse)

    def test_accepts_names_or_configs(self):
        predicted = torch.randn(3, 2, 4)
        target = torch.randn(3, 2, 4)
        mixed = run_loss_ablation(
            predicted, target, arms=["mse", LossArmConfig(arm="dist")]
        )
        assert [r.arm for r in mixed.arms] == ["mse", "dist"]

    def test_empty_arms_rejected(self):
        with pytest.raises(ValueError, match="at least one arm"):
            run_loss_ablation(torch.randn(2, 2, 2), torch.randn(2, 2, 2), arms=[])

    def test_dist_term_is_permutation_invariant_mse_is_not(self):
        # The crux of the ablation: the distribution arm discriminates from MSE
        # exactly where MSE is blind. Permuting target rows breaks the per-sample
        # pairing (MSE moves) but leaves the batch distribution unchanged (the
        # distribution term holds). The harness makes this observable per term.
        torch.manual_seed(0)
        predicted = torch.randn(8, 3, 5)
        target = torch.randn(8, 3, 5)
        perm = torch.randperm(8)
        target_perm = target[perm]

        res = run_loss_ablation(predicted, target, arms=["mse", "dist"])
        res_perm = run_loss_ablation(predicted, target_perm, arms=["mse", "dist"])

        mse_term = res.arms[0].breakdown.mse
        mse_term_perm = res_perm.arms[0].breakdown.mse
        dist_term = res.arms[1].breakdown.dist
        dist_term_perm = res_perm.arms[1].breakdown.dist

        # MSE is per-sample: the broken pairing changes it.
        assert not torch.allclose(mse_term, mse_term_perm)
        # Distribution is permutation-invariant: identical moments -> same term.
        assert torch.allclose(dist_term, dist_term_perm)

    def test_as_rows_is_comparable_table(self):
        predicted = torch.randn(3, 2, 4)
        target = torch.randn(3, 2, 4)
        rows = run_loss_ablation(predicted, target).as_rows()
        assert len(rows) == 3
        keys = set(rows[0])
        assert keys == {
            "arm",
            "mse_weight",
            "cosine_weight",
            "dist_weight",
            "loss",
            "mse_term",
            "cosine_term",
            "dist_term",
        }
        labels = [r["arm"] for r in rows]
        assert labels == ["mse", "mse_cos", "dist"]
