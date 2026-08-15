"""§7 null baseline for the PSA PC1 prior — behavior lock (GOAL §3.2 / §7).

PSA (``src/tg_lora/psa.py``) amplifies gradients along the per-tensor PC1 prior
``v_PSA`` extracted from ΔW history. Its validity rests on whether that prior
captures more gradient-direction energy than a random unit vector — the random-
surrogate null GOAL §7 mandates and §4's 統計の歯止め recognizes as the only
valid claim. ``layer_delta_analysis`` already pins the rank-1 *eigenvalue* null
(``rank1_z`` vs Marchenko-Pastur, ``test_rank1_null_calibration.py``); this file
pins the null for the PC1 *direction* PSA amplifies, which that eigenvalue test
does not cover (a high rank-1 dominance says a spike exists, not that the
spike's eigenvector aligns with the gradients PSA would amplify).

Measured verdict (GPU-free, synthetic — thresholds below are tuned to these):

  * iid-noise deltas + iid held-out grads  -> alignment_ratio ≈ 1.0  (NULL: the
    prior is no better than random; PSA would inject gradient noise — NO false
    signal), across every realistic (H, numel) regime;
  * planted spike shared by deltas AND held-out grads -> alignment_ratio >> 1.0
    (SIGNAL: the prior captures a real gradient direction);
  * a RANDOM prior collapses to ≈ 1.0 even on the signal setup — proves the
    metric DISCRIMINATES (the signal result needs the real extracted prior).

The two together close the prerequisite honesty gate before §3.2 PSA can be
reactivated as the post-§4 research axis (``docs/psa_axis_research_question.md``).
"""

import pytest
import torch

from src.tg_lora.psa import _power_iteration_pc1
from src.tg_lora.psa_null_baseline import (
    ALIGNMENT_NULL,
    ALIGNMENT_SIGNAL,
    alignment_z_score,
    decide_alignment_signal,
    null_alignment_ratio_distribution,
    prior_vs_surrogate_alignment,
    random_unit_directions,
)


def _extract_prior(deltas: list[torch.Tensor], seed_for_iter: int | None = None) -> torch.Tensor:
    """Run the REAL PSA PC1 extraction (power iteration) on a delta history.

    Uses the production ``_power_iteration_pc1`` so these tests exercise the
    exact direction PSA would amplify, not an independent reimplementation.
    """
    if seed_for_iter is not None:
        torch.manual_seed(seed_for_iter)
    mat = torch.stack([d.flatten().to(torch.float32) for d in deltas])  # [H, numel]
    v = _power_iteration_pc1(mat, n_iters=50)
    return v / (v.norm() + 1e-12)


class TestNullNoFalsePositive:
    """On pure iid noise the extracted prior must NOT beat the random surrogate
    (alignment_ratio ≈ 1.0). This is the §7 null: NO false signal — otherwise
    PSA would be greenlit on noise."""

    def test_iid_noise_prior_is_null(self):
        torch.manual_seed(2026)
        numel = 1024
        deltas = [torch.randn(numel) * 0.1 for _ in range(16)]  # iid noise, no spike
        prior = _extract_prior(deltas)
        grad_samples = torch.randn(64, numel) * 0.3  # held-out, disjoint from deltas

        result = prior_vs_surrogate_alignment(
            prior, grad_samples, n_surrogate=128, generator=torch.Generator().manual_seed(7)
        )
        assert result["alignment_ratio"] == pytest.approx(1.0, abs=0.25), (
            f"iid-noise prior must be a NULL (ratio≈1.0), got "
            f"{result['alignment_ratio']:.3f} — the §7 null must NOT report a false signal"
        )

    def test_iid_null_stays_below_signal_threshold_across_regimes(self):
        """False-positive sweep across realistic (H, numel) regimes, mirroring
        ``test_rank1_null_calibration.py``'s multi-regime null measurement."""
        for n_rows, numel in [(8, 256), (16, 4096), (32, 4096)]:
            torch.manual_seed(1000 + n_rows)
            deltas = [torch.randn(numel) * 0.1 for _ in range(n_rows)]
            prior = _extract_prior(deltas)
            grads = torch.randn(max(64, numel // 4), numel) * 0.3
            result = prior_vs_surrogate_alignment(
                prior,
                grads,
                n_surrogate=128,
                generator=torch.Generator().manual_seed(n_rows),
            )
            assert result["alignment_ratio"] < 1.7, (
                f"iid null false signal at regime ({n_rows},{numel}): "
                f"ratio={result['alignment_ratio']:.3f}"
            )


class TestSignalDetected:
    """When deltas AND held-out grads share a real spike, the prior must beat the
    random surrogate (alignment_ratio >> 1.0)."""

    def test_planted_spike_prior_beats_surrogate(self):
        torch.manual_seed(2027)
        numel = 1024
        u = torch.randn(numel)
        u = u / u.norm()
        # ΔW history dominated by u (spike well above the MP edge).
        deltas = [u * (t + 1) * 2.0 + torch.randn(numel) * 0.05 for t in range(16)]
        prior = _extract_prior(deltas)
        # Held-out grads concentrated along u (the signal PSA should amplify).
        grad_samples = u.unsqueeze(0) * 1.0 + torch.randn(64, numel) * 0.05

        result = prior_vs_surrogate_alignment(
            prior, grad_samples, n_surrogate=128, generator=torch.Generator().manual_seed(9)
        )
        assert result["alignment_ratio"] > 50.0, (
            f"planted-spike prior must capture signal (ratio>>1), got "
            f"{result['alignment_ratio']:.3f} — extraction or metric is broken"
        )
        # Sanity: the extracted prior actually tracks the planted direction u.
        assert abs(torch.dot(prior, u).item()) > 0.95


class TestMutationProofs:
    """Prove the metric DISCRIMINATES: a random prior collapses the signal, so
    the signal result needs the real extracted prior (not any unit vector)."""

    def test_random_prior_collapses_to_null_on_signal_data(self):
        """On the SIGNAL setup, swapping in a RANDOM prior must report ≈1.0 —
        if it reported high, the metric would fire on noise and the §7 gate
        would be defeated."""
        torch.manual_seed(2028)
        numel = 1024
        u = torch.randn(numel)
        u = u / u.norm()
        grad_samples = u.unsqueeze(0) + torch.randn(64, numel) * 0.05
        random_prior = torch.randn(numel)
        random_prior = random_prior / random_prior.norm()

        result = prior_vs_surrogate_alignment(
            random_prior,
            grad_samples,
            n_surrogate=128,
            generator=torch.Generator().manual_seed(11),
        )
        assert result["alignment_ratio"] < 1.7, (
            f"a RANDOM prior must be null even on signal data, got "
            f"{result['alignment_ratio']:.3f} — metric does not discriminate"
        )


class TestEdgeCases:
    def test_determinism_with_generator(self):
        torch.manual_seed(0)
        numel = 256
        prior = torch.randn(numel)
        prior = prior / prior.norm()
        grads = torch.randn(32, numel)
        r1 = prior_vs_surrogate_alignment(
            prior, grads, n_surrogate=64, generator=torch.Generator().manual_seed(123)
        )
        r2 = prior_vs_surrogate_alignment(
            prior, grads, n_surrogate=64, generator=torch.Generator().manual_seed(123)
        )
        assert r1 == r2

    def test_zero_norm_grads_return_zero_not_crash(self):
        numel = 16
        prior = torch.ones(numel)
        prior = prior / prior.norm()
        grads = torch.zeros(4, numel)  # all-zero norm
        result = prior_vs_surrogate_alignment(prior, grads, n_surrogate=8)
        assert result["alignment_ratio"] == 0.0
        assert result["prior_alignment"] == 0.0

    def test_wrong_shape_raises(self):
        prior = torch.randn(8)
        grads = torch.randn(4, 16)  # mismatched last dim
        with pytest.raises(ValueError, match="must match prior numel"):
            prior_vs_surrogate_alignment(prior, grads)

    def test_non_unit_prior_normalized_internally(self):
        """A non-unit prior must not inflate the metric — internal normalization
        means scaling the prior leaves the ratio invariant (no measurement
        footgun)."""
        torch.manual_seed(1)
        numel = 128
        v = torch.randn(numel)
        v = v / v.norm()
        grads = torch.randn(32, numel)
        r_unit = prior_vs_surrogate_alignment(
            v, grads, n_surrogate=32, generator=torch.Generator().manual_seed(5)
        )
        r_scaled = prior_vs_surrogate_alignment(
            v * 100.0, grads, n_surrogate=32, generator=torch.Generator().manual_seed(5)
        )
        assert r_unit["alignment_ratio"] == pytest.approx(
            r_scaled["alignment_ratio"], rel=1e-5
        )


class TestRandomUnitDirections:
    def test_unit_norm_and_shape(self):
        d = random_unit_directions(20, 64, torch.Generator().manual_seed(0))
        assert d.shape == (20, 64)
        norms = d.norm(dim=1)
        assert torch.allclose(norms, torch.ones(20), atol=1e-5)

    def test_zero_returns_empty(self):
        assert random_unit_directions(0, 8).shape == (0, 8)
        assert random_unit_directions(5, 0).shape == (5, 0)


# ── §7 calibrated decision boundary (the rank1_z analog for the PC1 direction) ─


# Realistic per-tensor calibration regimes: (n_history, numel, n_grad).
# n_history = PSA ring-buffer snapshots; numel = one LoRA tensor's flat size;
# n_grad = held-out gradient samples.
_CALIB_REGIMES = [(16, 4096, 64), (8, 256, 64), (32, 4096, 128)]


def _null_dist(n_history: int, numel: int, n_grad: int, *, seed: int = 2026, trials: int = 200):
    return null_alignment_ratio_distribution(
        numel,
        n_history,
        n_grad,
        n_trials=trials,
        n_surrogate=64,
        generator=torch.Generator().manual_seed(seed),
    )


def _fresh_iid_ratio(n_history: int, numel: int, n_grad: int, seed: int) -> float:
    """One iid-noise measurement from a SEPARATE seed stream (not a null sample)."""
    g = torch.Generator().manual_seed(seed)
    mat = torch.randn(n_history, numel, generator=g, dtype=torch.float32) * 0.1
    guess = torch.randn(numel, generator=g, dtype=torch.float32)
    prior = _power_iteration_pc1(mat, n_iters=20, initial_guess=guess)
    prior = prior / (prior.norm() + 1e-12)
    grads = torch.randn(n_grad, numel, generator=g, dtype=torch.float32) * 0.3
    return prior_vs_surrogate_alignment(prior, grads, n_surrogate=64, generator=g)[
        "alignment_ratio"
    ]


def _planted_spike_ratio(n_history: int, numel: int, n_grad: int, seed: int = 2027) -> float:
    """A history AND held-out grads sharing one dominant direction -> real signal."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(numel, generator=g, dtype=torch.float32)
    u = u / u.norm()
    mat = torch.stack(
        [u * (t + 1) * 2.0 + torch.randn(numel, generator=g, dtype=torch.float32) * 0.05 for t in range(n_history)]
    )
    guess = torch.randn(numel, generator=g, dtype=torch.float32)
    prior = _power_iteration_pc1(mat, n_iters=20, initial_guess=guess)
    prior = prior / (prior.norm() + 1e-12)
    grads = u.unsqueeze(0) + torch.randn(n_grad, numel, generator=g, dtype=torch.float32) * 0.05
    return prior_vs_surrogate_alignment(prior, grads, n_surrogate=64, generator=g)[
        "alignment_ratio"
    ]


class TestNullDecisionCalibration:
    """The §7 decision basis — a calibrated iid null turns one measured
    ``alignment_ratio`` into a go/no-go. Mirrors ``test_rank1_null_calibration.py``'s
    discipline for the PC1 *direction* (``rank1_z`` covers the eigenvalue): the
    null must center near 1.0, never false-positive on noise, and fire on signal.

    This closes the gap ``docs/psa_axis_research_question.md`` §3/§4 left open —
    §3 calibrated the null center and the signal but derived no decision boundary,
    so the 9B live ``alignment_ratio`` would be a number with no threshold.
    """

    def test_iid_null_centers_near_one_across_regimes(self):
        for n_history, numel, n_grad in _CALIB_REGIMES:
            null = _null_dist(n_history, numel, n_grad)
            assert null["mean"] == pytest.approx(1.0, abs=0.35), (
                f"regime ({n_history},{numel},{n_grad}): null mean={null['mean']:.3f} "
                "— the iid null must center near 1.0 (prior ≈ random direction), "
                "else the decision boundary is biased against real signal."
            )

    def test_iid_null_decides_no_false_positive(self):
        """A fresh iid measurement must be decided NULL, not SIGNAL — no false
        positive on noise (the §7 honesty gate). Mirrors rank1's frac(z>3)<0.05."""
        for n_history, numel, n_grad in _CALIB_REGIMES:
            null = _null_dist(n_history, numel, n_grad)
            false_signals = sum(
                1
                for s in range(60)
                if decide_alignment_signal(
                    _fresh_iid_ratio(n_history, numel, n_grad, seed=9000 + s), null
                )
                == ALIGNMENT_SIGNAL
            )
            assert false_signals / 60 < 0.05, (
                f"regime ({n_history},{numel},{n_grad}): {false_signals}/60 fresh iid "
                "measurements decided SIGNAL — the null false-positives, defeating "
                "the §7 gate (a noise prior would greenlight PSA)."
            )

    def test_planted_spike_decides_signal(self):
        """Positive control: a real shared spike must be decided SIGNAL with z≫1.
        Without this the null tests above could pass vacuously."""
        n_history, numel, n_grad = 16, 1024, 64
        null = _null_dist(n_history, numel, n_grad, seed=2026)
        ratio = _planted_spike_ratio(n_history, numel, n_grad)
        assert decide_alignment_signal(ratio, null) == ALIGNMENT_SIGNAL, (
            f"planted-spike ratio={ratio:.3f} vs null p99={null['p99']:.3f} — a real "
            "shared spike must clear the calibrated boundary."
        )
        assert alignment_z_score(ratio, null) > 10.0, (
            f"planted-spike z={alignment_z_score(ratio, null):.1f} — well above the "
            "null, not borderline (else the gate is too lenient)."
        )

    def test_zscore_is_zero_at_null_mean(self):
        null = _null_dist(16, 4096, 64)
        assert alignment_z_score(null["mean"], null) == pytest.approx(0.0, abs=1e-6)

    def test_decision_boundary_is_p99(self):
        """Pin the conservative rule: just below p99 -> NULL, just above -> SIGNAL.
        Triangulated with the null (below) and signal (above) tests above."""
        null = _null_dist(16, 4096, 64)
        assert decide_alignment_signal(null["p99"] * 0.999, null) == ALIGNMENT_NULL
        assert decide_alignment_signal(null["p99"] * 1.001, null) == ALIGNMENT_SIGNAL

    def test_null_distribution_is_deterministic(self):
        a = _null_dist(16, 4096, 64, seed=555)
        b = _null_dist(16, 4096, 64, seed=555)
        assert a == b
