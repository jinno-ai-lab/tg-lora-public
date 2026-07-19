"""Unit tests for the GOAL §4 valid_loss-difference bootstrap CI.

GOAL §4's statistical brake makes the bootstrap CI on the valid_loss difference
the judge of the quality axis: "valid_loss 差はブートストラップ CI で評価" and
"ランダム順フリーズ（サロゲート）を超えた削減・性能だけを有効と認定".
:func:`src.tg_lora.freeze_surrogate_gate.surrogate_exceedance` already returns
the structural ``SURPASSES`` / ``TIES`` / ``UNDERSHOOTS`` verdict (deterministic,
clears the best surrogate arm); :func:`surrogate_valid_loss_ci` is the
*significance-graded* promotion — does the candidate's valid_loss lead over the
random-order surrogate survive resampling, or could sample noise explain it?

The verdicts below are pinned to effect sizes large enough that the seed does
not flip them (a constant arm resamples to itself, so a wide-enough gap gives a
deterministic CI), keeping the statistical assertions robust. The §7 honesty
tests exercise the two traps this layer keeps apart: significance vs.
materiality, and thin-evidence honesty.
"""

import math

import numpy as np
import pytest

from src.tg_lora.freeze_surrogate_ci import (
    DEFAULT_N_BOOTSTRAP,
    MIN_SAMPLE_FOR_BOOTSTRAP,
    bootstrap_difference_ci,
    format_surrogate_valid_loss_ci,
    surrogate_valid_loss_ci,
)
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

# Valid_loss: lower is better. A candidate SURPASSES the random-order surrogate
# on quality when its valid_loss sits *below* the surrogate's, so the signed
# improvement (surrogate_mean - candidate_mean) is positive.

# A clearly-better candidate: every candidate run beats every surrogate run by a
# wide margin -> the CI excludes zero on the candidate's side, deterministically
# (constant arms resample to themselves regardless of seed).
_BETTER_CANDIDATE = (1.0, 1.0, 1.0, 1.0, 1.0)
_BETTER_SURROGATE = (2.0, 2.0, 2.0, 2.0, 2.0)

# A clearly-worse candidate: candidate valid_loss above the surrogate's -> CI
# excludes zero against the candidate.
_WORSE_CANDIDATE = (2.0, 2.0, 2.0, 2.0, 2.0)
_WORSE_SURROGATE = (1.0, 1.0, 1.0, 1.0, 1.0)

# Indistinguishable arms: same multiset -> the resampled differences are
# symmetric about zero -> the CI straddles zero -> TIES, robustly across seeds.
_SAME_CANDIDATE = (1.0, 0.9, 1.1, 1.0, 1.05)
_SAME_SURROGATE = (1.05, 1.0, 1.1, 0.9, 1.0)


class TestSignificanceVerdictFromCI:
    def test_clearly_better_candidate_surpasses(self):
        ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE)
        assert ci.significance_verdict == SURPASSES
        assert ci.significant_surpasses is True
        # Constant arms -> the CI is a point exactly at the observed gap (1.0).
        assert ci.point_improvement == pytest.approx(1.0)
        assert ci.lower == pytest.approx(1.0)
        assert ci.upper == pytest.approx(1.0)

    def test_clearly_worse_candidate_undershoots(self):
        ci = surrogate_valid_loss_ci(_WORSE_CANDIDATE, _WORSE_SURROGATE)
        assert ci.significance_verdict == UNDERSHOOTS
        assert ci.point_improvement == pytest.approx(-1.0)
        assert ci.upper < 0.0

    def test_indistinguishable_arms_are_ties(self):
        ci = surrogate_valid_loss_ci(_SAME_CANDIDATE, _SAME_SURROGATE)
        assert ci.significance_verdict == TIES
        # Same multiset -> mean difference ~0 and the CI brackets zero.
        assert ci.point_improvement == pytest.approx(0.0, abs=1e-9)
        assert ci.lower <= 0.0 <= ci.upper

    def test_ci_brackets_the_point_estimate_for_noisy_arms(self):
        # A non-degenerate bootstrap distribution must contain its own center.
        ci = surrogate_valid_loss_ci(_SAME_CANDIDATE, _SAME_SURROGATE)
        assert ci.lower <= ci.point_improvement <= ci.upper


class TestHonorsGoalSection7Honesty:
    def test_significance_is_separate_from_materiality(self):
        # A significant but tiny lead: the CI excludes zero (constant arms ->
        # point gap of 0.001), yet the lead is below a 0.01 material margin.
        # GOAL §7: "統計的有意性と実用的集中度を区別" — significant SURPASSES
        # must not auto-count as a §4 win when the effect is immaterial.
        ci = surrogate_valid_loss_ci(
            (1.0, 1.0, 1.0),
            (1.001, 1.001, 1.001),
            material_margin=0.01,
        )
        assert ci.significance_verdict == SURPASSES
        assert ci.significant_surpasses is True
        assert ci.is_material is False
        assert ci.passes is False

    def test_default_margin_makes_passes_mirror_significance(self):
        # Default material_margin=0.0: a positive, significant lead is material
        # by construction, so passes == significant_surpasses (no hidden
        # materiality tightening at the default).
        ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE)
        assert ci.material_margin == 0.0
        assert ci.is_material is True
        assert ci.passes == ci.significant_surpasses

    def test_negative_material_margin_rejected(self):
        with pytest.raises(ValueError, match="material_margin"):
            surrogate_valid_loss_ci(
                _BETTER_CANDIDATE, _BETTER_SURROGATE, material_margin=-0.01
            )

    def test_difference_is_taken_against_the_surrogate_null(self):
        # GOAL §7 "すべての指標にランダム帰無基準を併記": the improvement is
        # mean(surrogate) - mean(candidate), so a bare candidate number never
        # stands alone — it is always relative to the random-order control.
        ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE)
        assert ci.point_improvement == pytest.approx(
            ci.surrogate_mean - ci.candidate_mean
        )


class TestThinEvidence:
    def test_one_element_arm_is_flagged_thin(self):
        # An n=1 candidate resamples to a constant: the bootstrap cannot capture
        # its run-to-run variance, so the CI is too narrow to anchor a verdict.
        ci = surrogate_valid_loss_ci((1.0,), _BETTER_SURROGATE)
        assert ci.is_thin_evidence is True
        assert ci.n_candidate == 1

    def test_below_threshold_surrogate_arm_is_flagged_thin(self):
        ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, (2.0, 2.0))
        assert ci.is_thin_evidence is True
        assert ci.n_surrogate == 2 < MIN_SAMPLE_FOR_BOOTSTRAP

    def test_both_arms_above_threshold_are_not_thin(self):
        ci = surrogate_valid_loss_ci(
            (1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0)
        )
        assert ci.is_thin_evidence is False

    def test_default_bootstrap_resample_count_is_substantial(self):
        # GOAL §4 wants a significance statement, not a 2-draw anecdote; the
        # default resample count must put Monte-Carlo noise well under the
        # resolved differences.
        assert DEFAULT_N_BOOTSTRAP >= 1000


class TestReproducibility:
    def test_same_seed_gives_identical_ci(self):
        # Noisy arms so the CI bounds are non-trivial (not a constant point).
        a = bootstrap_difference_ci(_SAME_CANDIDATE, _SAME_SURROGATE, seed=7)
        b = bootstrap_difference_ci(_SAME_CANDIDATE, _SAME_SURROGATE, seed=7)
        assert a == b  # byte-identical (point, lower, upper)

    def test_ci_is_stable_across_reasonable_seeds(self):
        # The verdict (not the exact bounds) is the durable quantity: across
        # independent seeds a clearly-better candidate stays SURPASSES.
        for seed in (0, 1, 42, 100):
            ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE, seed=seed)
            assert ci.significance_verdict == SURPASSES

    def test_ci_respects_confidence_level_width(self):
        # A wider confidence level yields a (weakly) wider interval: 99% must
        # bracket at least as much as 90% on the same noisy sample.
        _, lo90, hi90 = bootstrap_difference_ci(
            _SAME_CANDIDATE, _SAME_SURROGATE, confidence=0.90, seed=3
        )
        _, lo99, hi99 = bootstrap_difference_ci(
            _SAME_CANDIDATE, _SAME_SURROGATE, confidence=0.99, seed=3
        )
        assert lo99 <= lo90 + 1e-12
        assert hi99 + 1e-12 >= hi90


class TestValidation:
    def test_empty_candidate_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            surrogate_valid_loss_ci((), _BETTER_SURROGATE)

    def test_empty_surrogate_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            surrogate_valid_loss_ci(_BETTER_CANDIDATE, ())

    def test_bootstrap_difference_ci_empty_arm_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            bootstrap_difference_ci((), _BETTER_SURROGATE)

    def test_bad_confidence_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_difference_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE, confidence=1.5)

    def test_zero_n_bootstrap_rejected(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            bootstrap_difference_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE, n_bootstrap=0)


class TestNonFiniteLossRejected:
    """A NaN/inf sample must fail loud at the verdict chokepoint.

    Closed corruption: a diverged training arm (gradient explosion) records a
    NaN/inf valid_loss. Without this guard it reaches the bootstrap math, where
    ``numpy.percentile`` over a NaN-poisoned ``improvements`` array returns NaN
    and ``nan > 0`` / ``nan < 0`` are both ``False`` — so the verdict silently
    falls through to ``TIES``, the *same* label both genuine full-budget §4
    verdicts carry (corrupt-but-green, GOAL §7). The guard sits at
    :func:`surrogate_valid_loss_ci` — the single entry every producer feeds
    (proxy ``run_ci``, 9B ``run_ci_9b``, committed-deposit replay) — complementing
    the formation-chokepoint guard ``756ea96`` (which keeps NaN out of a
    *committed deposit artifact* but never sees the proxy/9B live-arm paths).
    """

    def test_nan_in_candidate_rejected(self):
        # Without the guard this returns verdict == TIES silently (see the
        # math-layer proof at the end of this class).
        with pytest.raises(ValueError, match="non-finite"):
            surrogate_valid_loss_ci([1.5, float("nan"), 1.6], _BETTER_SURROGATE)

    def test_nan_in_surrogate_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            surrogate_valid_loss_ci(_BETTER_CANDIDATE, [1.7, float("nan"), 1.9])

    def test_positive_inf_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            surrogate_valid_loss_ci([1.5, float("inf")], _BETTER_SURROGATE)

    def test_negative_inf_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            surrogate_valid_loss_ci(_BETTER_CANDIDATE, [1.7, float("-inf")])

    def test_nan_string_coercion_rejected(self):
        # float("nan") accepts the string form; a deposit reloaded from JSON
        # could carry it if the formation guard (756ea96) were ever bypassed,
        # so the coercion path must be caught too.
        with pytest.raises(ValueError, match="non-finite"):
            surrogate_valid_loss_ci([1.5, "nan"], _BETTER_SURROGATE)

    def test_error_names_the_diverged_arm(self):
        # The operator must see WHICH arm diverged to drop or re-run it.
        with pytest.raises(ValueError, match="surrogate"):
            surrogate_valid_loss_ci(_BETTER_CANDIDATE, [float("nan")])

    def test_finite_samples_are_unaffected(self):
        # The guard is transparent for valid input: a clearly-better finite
        # candidate still clears the bar with the same verdict (byte-identical
        # path — both committed full-budget deposits replay unchanged).
        ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE)
        assert ci.significance_verdict == SURPASSES

    def test_corruption_mechanism_nan_silently_ties_at_math_layer(self):
        # PROOF of the bug this verdict-entry guard closes. Feed NaN to the raw
        # math layer (bootstrap_difference_ci, which this fix intentionally does
        # NOT guard — surrogate_valid_loss_ci does). numpy.percentile over a
        # NaN-poisoned improvements array returns NaN, and NaN compares False to
        # zero in BOTH directions, so the verdict logic a NaN CI yields is TIES
        # — silently indistinguishable from a genuine null. This is exactly why
        # the guard must sit at the verdict entry every producer feeds, and it
        # is mutation-stable: it exercises the unguarded math layer, so it
        # cannot pass by accident of the guard being present or absent.
        point, lower, upper = bootstrap_difference_ci(
            [1.5, float("nan"), 1.6], _BETTER_SURROGATE
        )
        assert math.isnan(point)
        assert math.isnan(lower)
        assert math.isnan(upper)
        verdict = SURPASSES if lower > 0.0 else (UNDERSHOOTS if upper < 0.0 else TIES)
        assert verdict == TIES  # the silent corruption the guard prevents


class TestFormatter:
    def test_is_deterministic_and_carries_verdict_and_ci(self):
        ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE)
        text = format_surrogate_valid_loss_ci(ci)
        assert text == format_surrogate_valid_loss_ci(ci)
        assert ci.significance_verdict in text
        assert "valid_loss_axis" in text
        assert "improvement" in text
        assert "material" in text
        assert "bootstrap" in text

    def test_thin_evidence_label_is_stated_plainly(self):
        ci = surrogate_valid_loss_ci((1.0,), _BETTER_SURROGATE)
        text = format_surrogate_valid_loss_ci(ci)
        # A thin SURPASSES must read as THIN_EVIDENCE, not as a confirmed win.
        assert "THIN_EVIDENCE" in text
        assert "do not read" in text

    def test_no_thin_note_when_arms_are_sufficient(self):
        ci = surrogate_valid_loss_ci(
            (1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0)
        )
        assert "THIN_EVIDENCE" not in format_surrogate_valid_loss_ci(ci)


class TestFrozenAndTyped:
    def test_result_is_frozen(self):
        ci = surrogate_valid_loss_ci(_BETTER_CANDIDATE, _BETTER_SURROGATE)
        with pytest.raises((AttributeError, Exception)):
            ci.lower = 999.0  # type: ignore[misc]

    def test_low_level_ci_returns_floats(self):
        point, lower, upper = bootstrap_difference_ci(
            _BETTER_CANDIDATE, _BETTER_SURROGATE
        )
        assert isinstance(point, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)


# Smoke-test that the verdict labels are the *same* objects the structural gate
# emits — the bootstrap layer promotes (not renames) the surrogate-exceedance
# verdict, so the vocabulary is shared by import, not duplicated.
def test_verdict_labels_are_imported_from_the_structural_gate():
    from src.tg_lora import freeze_surrogate_ci as ci_mod
    from src.tg_lora import freeze_surrogate_gate as gate_mod

    assert ci_mod.SURPASSES is gate_mod.SURPASSES
    assert ci_mod.TIES is gate_mod.TIES
    assert ci_mod.UNDERSHOOTS is gate_mod.UNDERSHOOTS
    # numpy is the only numeric dependency, and it is an existing repo dep.
    assert isinstance(np.ndarray, type)
