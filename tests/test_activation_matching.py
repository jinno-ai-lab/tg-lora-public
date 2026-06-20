"""Unit tests for activation-matching local loss (Progressive Freezing Level 2).

The matching loss is the learning signal that trains the front layer against
the cached ``xin`` after the backward-graph suffix is cut (GOAL §1.6.1, design
§3). All expected values are hand-computed so the Phase 1 mechanism is locked
down before any downstream freeze experiment trusts it (GOAL §7).
"""

import pytest
import torch

from src.tg_lora.activation_matching import (
    ActivationMatchingLoss,
    cosine_matching_loss,
    distribution_matching_loss,
    mse_matching_loss,
)


class TestMseMatching:
    def test_exact_match_is_zero(self):
        # The "no learning signal" state (design §3.3): front layer already
        # reproduces xin -> loss 0 -> no gradient.
        x = torch.randn(4, 3, 8)
        assert torch.equal(mse_matching_loss(x, x), torch.tensor(0.0))

    def test_hand_computed(self):
        predicted = torch.tensor([[0.0, 0.0]])  # [N=1, H=2]
        target = torch.tensor([[2.0, 0.0]])
        # diff_sq = [4, 0]; mean over 2 = 2.0
        assert torch.allclose(mse_matching_loss(predicted, target), torch.tensor(2.0))

    def test_matches_torch_mse(self):
        predicted = torch.randn(3, 5, 7)
        target = torch.randn(3, 5, 7)
        expected = ((predicted - target) ** 2).mean()
        assert torch.allclose(mse_matching_loss(predicted, target), expected)

    def test_scales_quadratically_with_distance(self):
        target = torch.zeros(2, 4)
        a = torch.full((2, 4), 1.0)
        b = torch.full((2, 4), 2.0)
        # 2x the distance -> 4x the MSE
        assert torch.allclose(
            mse_matching_loss(b, target), 4.0 * mse_matching_loss(a, target)
        )


class TestCosineMatching:
    def test_aligned_is_zero(self):
        v = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(cosine_matching_loss(v, v), torch.tensor(0.0))

    def test_opposite_is_two(self):
        predicted = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[-1.0, 0.0, 0.0]])
        assert torch.allclose(
            cosine_matching_loss(predicted, target), torch.tensor(2.0)
        )

    def test_orthogonal_is_one(self):
        predicted = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(
            cosine_matching_loss(predicted, target), torch.tensor(1.0)
        )

    def test_matches_one_minus_cos(self):
        predicted = torch.randn(3, 5, 7)
        target = torch.randn(3, 5, 7)
        cos = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        assert torch.allclose(cosine_matching_loss(predicted, target), 1.0 - cos.mean())


class TestDistributionMatching:
    """The Phase 3 distribution-consistency arm (GOAL §3.1 Phase 3, design §6.2).

    MSE measures point-wise agreement and cosine measures per-vector direction;
    both miss the *joint* second-order structure (covariance, correlation) of the
    activation batch. Distribution matching closes that gap by matching the
    batch's per-feature mean and covariance — the ``分布も合わせる版`` arm of the
    Phase 3 loss ablation. All expectations are hand-computed (GOAL §7).
    """

    def test_exact_match_is_zero(self):
        # The "no learning signal" state: identical batches share mean and
        # covariance -> loss 0 -> no gradient.
        x = torch.randn(4, 3, 8)
        assert torch.equal(distribution_matching_loss(x, x), torch.tensor(0.0))

    def test_hand_computed(self):
        # predicted=[[0,0],[0,0]], target=[[2,0],[0,2]] (no mask), H=2, D=2.
        # mean: mu_p=[0,0], mu_t=[1,1] -> mean of (0-1)^2 over 2 features = 1.0.
        # cov: predicted covariance is 0; target centered=[[1,-1],[-1,1]],
        #   cov_t=(c^T c)/D = [[2,-2],[-2,2]]/2 = [[1,-1],[-1,1]].
        #   cov_term = mean of squares of [[1,-1],[-1,1]] = 4/4 = 1.0.
        # loss = 1.0 + 1.0 = 2.0.
        predicted = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        target = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        assert torch.allclose(
            distribution_matching_loss(predicted, target), torch.tensor(2.0)
        )

    def test_mean_shift_only(self):
        # A pure constant shift moves the mean but leaves covariance at 0 for
        # both -> only the mean term fires. mu_p=[0,0], mu_t=[1,1] -> 1.0.
        predicted = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        target = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        assert torch.allclose(
            distribution_matching_loss(predicted, target), torch.tensor(1.0)
        )

    def test_distribution_shift_is_nonzero(self):
        # Scaling the target shifts both mean and covariance -> loss > 0, and
        # strictly larger than a mean-only shift (covariance mismatch fires).
        predicted = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        target = torch.tensor([[4.0, 0.0], [0.0, 4.0]])  # 2x scale
        shifted = distribution_matching_loss(predicted, target)
        mean_only = distribution_matching_loss(
            predicted, torch.tensor([[3.0, 3.0], [3.0, 3.0]])
        )
        assert shifted > 0.0
        assert shifted > mean_only  # covariance mismatch adds on top of the mean

    def test_permutation_invariant_the_honest_null_baseline(self):
        # GOAL §7 null baseline, framed honestly for a *distribution* statistic:
        # matching the batch distribution is, by design, blind to which sample
        # pairs with which xin (the complement of MSE's per-sample signal). A
        # permutation of the target rows leaves its mean and covariance
        # unchanged, so the distribution loss stays 0 even though MSE > 0.
        predicted = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        permuted = torch.tensor([[0.0, 2.0], [2.0, 0.0]])  # rows swapped
        assert torch.allclose(
            distribution_matching_loss(predicted, permuted), torch.tensor(0.0)
        )
        assert mse_matching_loss(predicted, permuted) > 0.0  # MSE is NOT blind to it

    def test_padding_excluded(self):
        # Same fixture as the MSE mask test: keep token 0, drop token 1.
        # With one effective row the covariance is 0 on both sides; only the
        # mean term fires: mu_p=[0,0], mu_t=[2,0] -> mean of (4,0) = 2.0.
        predicted = torch.zeros(1, 2, 2)
        target = torch.tensor([[[2.0, 0.0], [9.0, 9.0]]])
        mask = torch.tensor([[[1.0], [0.0]]])
        assert torch.allclose(
            distribution_matching_loss(predicted, target, mask=mask),
            torch.tensor(2.0),
        )

    def test_mask_reduces_to_unmasked_when_all_kept(self):
        predicted = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, 1)
        assert torch.allclose(
            distribution_matching_loss(predicted, target, mask=mask),
            distribution_matching_loss(predicted, target),
        )

    def test_gradient_flows_only_through_predicted(self):
        # xin is a cached, detached target, so only the front-layer output
        # receives a gradient — never the target (same contract as MSE/cosine).
        predicted = torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3)  # requires_grad=False, like cached xin
        distribution_matching_loss(predicted, target).backward()
        assert predicted.grad is not None
        assert target.grad is None
        assert not target.requires_grad


class TestGradient:
    def test_gradient_pushes_predicted_toward_target(self):
        # The whole point of the loss: pulling the front-layer output toward
        # the cached xin. One SGD step from a zero start must reduce distance.
        predicted = torch.zeros(2, requires_grad=True)
        target = torch.tensor([2.0, 0.0])
        loss = mse_matching_loss(predicted, target)
        loss.backward()
        grad = predicted.grad
        # d/dp mean((p-t)^2) = 2(p-t)/n -> at p=0: [-2, 0]
        assert torch.allclose(grad, torch.tensor([-2.0, 0.0]))

        with torch.no_grad():
            stepped = predicted - 0.5 * grad  # lr=0.5 -> [1, 0]
        assert torch.norm(stepped - target) < torch.norm(predicted - target)

    def test_gradient_flows_only_through_predicted(self):
        # xin is a cached, detached target (cache_xin stores .detach().cpu()),
        # so only the front-layer output receives a gradient — never the target.
        predicted = torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3)  # requires_grad=False, like cached xin
        mse_matching_loss(predicted, target).backward()
        assert predicted.grad is not None
        assert target.grad is None
        assert not target.requires_grad


class TestShapeGuard:
    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="identical shapes"):
            mse_matching_loss(torch.zeros(2, 3), torch.zeros(3, 2))
        with pytest.raises(ValueError, match="identical shapes"):
            cosine_matching_loss(torch.zeros(2, 3), torch.zeros(2, 4))
        with pytest.raises(ValueError, match="identical shapes"):
            distribution_matching_loss(torch.zeros(2, 3), torch.zeros(3, 2))


class TestMask:
    def test_padding_excluded_from_mse(self):
        # [N=1, T=2, H=2]; token 1 differs hugely but is masked out (padding).
        predicted = torch.zeros(1, 2, 2)
        target = torch.tensor([[[2.0, 0.0], [9.0, 9.0]]])
        mask = torch.tensor([[[1.0], [0.0]]])  # keep token 0, drop token 1
        # Only token 0 contributes: diff_sq [4, 0] -> 4/2 = 2.0
        assert torch.allclose(
            mse_matching_loss(predicted, target, mask=mask), torch.tensor(2.0)
        )

    def test_unmasked_mse_includes_padding(self):
        predicted = torch.zeros(1, 2, 2)
        target = torch.tensor([[[2.0, 0.0], [9.0, 9.0]]])
        # Unmasked: (4 + 0 + 81 + 81) / 4 = 41.5
        assert torch.allclose(mse_matching_loss(predicted, target), torch.tensor(41.5))

    def test_mask_reduces_to_unmasked_when_all_kept(self):
        predicted = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, 1)
        assert torch.allclose(
            mse_matching_loss(predicted, target, mask=mask),
            mse_matching_loss(predicted, target),
        )


class TestNullBaselinePairing:
    """GOAL §7: every metric needs a null baseline.

    A matching loss that only fit the batch mean would be invariant to which
    sample pairs with which xin. Here the correct pairing must beat a shuffled
    (wrong) pairing, proving the loss carries per-sample signal rather than a
    batch-average artefact.
    """

    def test_correct_pairing_beats_shuffled(self):
        predicted = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        target = predicted.clone()  # front layer reproduces each xin
        shuffled = torch.tensor([[0.0, 2.0], [2.0, 0.0]])  # swap samples
        correct = mse_matching_loss(predicted, target)
        null = mse_matching_loss(predicted, shuffled)
        assert correct == 0.0
        assert null > correct  # shuffled pairing (4.0) is strictly worse


class TestDeviceDtypeAlignment:
    def test_target_cast_to_predicted_dtype(self):
        # cache_xin stores xin as float32 on CPU; the activation path may be
        # bf16 (GOAL §1.5). The loss must align without an explicit caller cast.
        predicted = torch.tensor([[0.0, 0.0]], dtype=torch.bfloat16)
        target = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
        loss = mse_matching_loss(predicted, target)
        assert torch.isfinite(loss)
        assert loss.dtype == torch.bfloat16


class TestCombiner:
    def test_pure_mse_equals_mse(self):
        combiner = ActivationMatchingLoss(mse_weight=1.0, cosine_weight=0.0)
        predicted = torch.randn(3, 5, 7)
        target = torch.randn(3, 5, 7)
        assert torch.allclose(
            combiner(predicted, target), mse_matching_loss(predicted, target)
        )

    def test_pure_cosine_equals_cosine(self):
        combiner = ActivationMatchingLoss(mse_weight=0.0, cosine_weight=1.0)
        predicted = torch.randn(3, 5, 7)
        target = torch.randn(3, 5, 7)
        assert torch.allclose(
            combiner(predicted, target), cosine_matching_loss(predicted, target)
        )

    def test_weighted_sum(self):
        combiner = ActivationMatchingLoss(mse_weight=2.0, cosine_weight=3.0)
        predicted = torch.randn(3, 5, 7)
        target = torch.randn(3, 5, 7)
        expected = 2.0 * mse_matching_loss(
            predicted, target
        ) + 3.0 * cosine_matching_loss(predicted, target)
        assert torch.allclose(combiner(predicted, target), expected)

    def test_passes_mask_through(self):
        combiner = ActivationMatchingLoss(mse_weight=1.0, cosine_weight=1.0)
        predicted = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, 1)
        assert torch.isfinite(combiner(predicted, target, mask=mask))

    def test_default_is_phase1_pure_mse(self):
        # GOAL §3.1 Phase 1: "starting with MSE". Default weights encode that.
        assert ActivationMatchingLoss() == ActivationMatchingLoss(
            mse_weight=1.0, cosine_weight=0.0, dist_weight=0.0
        )

    def test_pure_distribution_equals_distribution(self):
        combiner = ActivationMatchingLoss(
            mse_weight=0.0, cosine_weight=0.0, dist_weight=1.0
        )
        predicted = torch.randn(3, 5, 7)
        target = torch.randn(3, 5, 7)
        assert torch.allclose(
            combiner(predicted, target),
            distribution_matching_loss(predicted, target),
        )

    def test_weighted_sum_three_terms(self):
        combiner = ActivationMatchingLoss(mse_weight=2.0, cosine_weight=3.0, dist_weight=4.0)
        predicted = torch.randn(3, 5, 7)
        target = torch.randn(3, 5, 7)
        expected = (
            2.0 * mse_matching_loss(predicted, target)
            + 3.0 * cosine_matching_loss(predicted, target)
            + 4.0 * distribution_matching_loss(predicted, target)
        )
        assert torch.allclose(combiner(predicted, target), expected)


class TestCombinerValidation:
    def test_negative_mse_weight_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ActivationMatchingLoss(mse_weight=-0.1, cosine_weight=1.0)

    def test_negative_cosine_weight_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ActivationMatchingLoss(mse_weight=1.0, cosine_weight=-0.1)

    def test_all_zero_weights_rejected(self):
        # A zeroed combiner emits no gradient — reject rather than silently
        # stall training (GOAL §7: no signal is not a signal).
        with pytest.raises(ValueError, match="positive"):
            ActivationMatchingLoss(mse_weight=0.0, cosine_weight=0.0, dist_weight=0.0)

    def test_negative_dist_weight_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ActivationMatchingLoss(mse_weight=1.0, cosine_weight=0.0, dist_weight=-0.1)

    def test_dist_only_allowed(self):
        # The pure-distribution arm of the Phase 3 ablation must be selectable
        # on its own (distribution alone still emits a gradient).
        loss = ActivationMatchingLoss(
            mse_weight=0.0, cosine_weight=0.0, dist_weight=1.0
        )
        predicted = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        assert torch.isfinite(loss(predicted, target))
