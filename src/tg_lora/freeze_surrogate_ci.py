"""GOAL §4 valid_loss-difference bootstrap CI: is a candidate schedule's quality
lead over the random-order surrogate *significant*, or could sample noise alone
explain it?

GOAL §4's statistical brake ("統計の歯止め") names the check the valid_loss axis
must pass before a freeze schedule's quality retention is credited::

    "valid_loss 差はブートストラップ CI で評価"
    "ランダム順フリーズ（サロゲート）を超えた削減・性能だけを有効と認定"
    "backward FLOPs がランダム順フリーズ対照を有意に超えて削減できる"

:mod:`src.tg_lora.freeze_surrogate_gate` already gives the *structural* verdict:
:func:`~src.tg_lora.freeze_surrogate_gate.surrogate_exceedance` compares a
candidate's numbers against the seeded surrogate distribution and returns the
graduated ``SURPASSES`` / ``TIES`` / ``UNDERSHOOTS`` label. That verdict is
*deterministic* — it clears even the best surrogate arm, the conservative
direction for the exact FLOPs axis. What it cannot do is say whether a *noisy*
valid_loss lead (a handful of GPU seeds, classification C) is real or sampling
luck, because one candidate number against a few-seed surrogate is one
anecdote, not a significance statement.

This module is the missing statistical layer that fills GOAL §4's
"ブートストラップ CI" line and bridges
:mod:`~src.tg_lora.freeze_surrogate_gate` (structure + seed management) to
:mod:`scripts.evaluate_paper_gates` (the G0–G4 acceptance gate): a percentile
bootstrap confidence interval on the *candidate-vs-surrogate valid_loss
difference* that promotes the structural verdict to a *significance-graded* one.
It is pure Python + ``numpy`` (an existing dependency); no GPU, no model. The
moment a GPU run deposits a candidate valid_loss sample and a surrogate
valid_loss sample (GOAL §4 "各条件は複数シードで回す"), this helper turns them
into the §4 null judgement immediately — the same ``SURPASSES`` / ``TIES`` /
``UNDERSHOOTS`` labels, now earned against resampling noise rather than a single
comparison.

The statistic. The difference is signed so *positive* = the candidate retains
quality better than the random-order surrogate (the candidate's valid_loss is
lower, and lower is better)::

    point_improvement = mean(surrogate_valid_loss) - mean(candidate_valid_loss)

A two-sample independent bootstrap resamples each arm with replacement and
takes the difference of resampled means each draw; the central
``confidence`` percentile interval of those draws is the CI. A candidate
``SURPASSES`` when the CI sits entirely above zero (the lead survives
resampling), ``UNDERSHOOTS`` when entirely below, else ``TIES`` (honestly
indistinguishable from the random control under resampling).

Honesty (GOAL §7 鉄則) — this layer keeps the two traps §7 warns about apart:

* **Significance ≠ practical importance** (§7 "統計的有意性と実用的集中度を
  区別"): :attr:`SurrogateValidLossCI.significance_verdict` says whether the CI
  excludes zero; a *separate* :attr:`~SurrogateValidLossCI.is_material` flag says
  whether the point lead clears a caller-set margin. A significant-but-tiny lead
  reads as ``SURPASSES`` yet ``is_material=False`` — the same discipline the
  2.6σ-but-useless ZO precedent enforces, so a paper never claims a §4 win on
  significance alone.
* **Always pair with the random null** (§7 "すべての指標にランダム帰無基準を
  併記"): the surrogate *is* the null the difference is taken against, so the CI
  is already on the candidate-vs-random axis, never a bare candidate number.
* **Thin sample honesty**: below :data:`MIN_SAMPLE_FOR_BOOTSTRAP` in *either*
  arm the bootstrap cannot capture that arm's run-to-run variance (a one-element
  arm resamples to a constant), so the result is flagged
  :attr:`~SurrogateValidLossCI.is_thin_evidence` and must not be read as a
  confirmed significance statement — the same sample bar
  :data:`freeze_cost.MIN_SAMPLE_FOR_CONFIDENCE_BAND` applies to a confidence
  band.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Reuse the exact verdict labels the structural gate emits, so the bootstrap
# layer *promotes* (does not rename) the surrogate-exceedance verdict to
# significance-graded — the same vocabulary, now earned against resampling
# noise. Importing them also makes the freeze_surrogate_gate → CI → gate bridge
# a literal module-to-module link rather than a parallel naming scheme.
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

# Default resample count. A bootstrap CI is a Monte-Carlo estimate; 10k
# resamples puts the percentile-endpoint Monte-Carlo noise well below the
# differences this layer is asked to resolve, while staying sub-millisecond on
# the small samples a few GPU seeds produce.
DEFAULT_N_BOOTSTRAP: int = 10_000

# Default two-sided confidence level (0.95 → the central 95% percentile
# interval on the resampled differences).
DEFAULT_CONFIDENCE: float = 0.95

# Default RNG seed so a CI is byte-reproducible. GOAL §4 demands the
# significance statement be reproducible across runs; the seed pins the
# resampling exactly (same samples + same seed → identical CI).
DEFAULT_CI_SEED: int = 0

# Minimum per-arm sample size for a *non-thin* bootstrap. With one observation
# the resampled mean is constant, so that arm contributes zero variance and the
# CI is too narrow; below this the result is flagged thin evidence. Matches
# freeze_cost.MIN_SAMPLE_FOR_CONFIDENCE_BAND (the same sample bar the §6.3
# confidence band applies — one honesty rule across both statistics layers).
MIN_SAMPLE_FOR_BOOTSTRAP: int = 3


@dataclass(frozen=True)
class SurrogateValidLossCI:
    """Bootstrap CI on the candidate-vs-surrogate valid_loss difference.

    The difference is signed so *positive* = the candidate retains quality
    better than the random-order surrogate (the candidate's valid_loss is
    lower): :attr:`point_improvement` = ``mean(surrogate) − mean(candidate)``.
    A candidate clears GOAL §4's "性能" bar only when that improvement is both
    *significant* (the CI excludes zero) and *material* (the point lead clears a
    caller-set margin) — the two §7 axes kept distinct here rather than
    collapsed into one boolean.

    :attr:`significance_verdict` is the bootstrap promotion of the structural
    :func:`~src.tg_lora.freeze_surrogate_gate.surrogate_exceedance` verdict:
    ``SURPASSES`` when the CI sits entirely above zero, ``UNDERSHOOTS`` when
    entirely below, else ``TIES``. :attr:`is_thin_evidence` flags when an arm
    was too small for the bootstrap to capture its variance — a thin
    ``SURPASSES`` is recorded for the audit but must not be read as confirmed
    (the formatter says so plainly).
    """

    candidate_mean: float
    surrogate_mean: float
    lower: float
    upper: float
    confidence: float
    n_candidate: int
    n_surrogate: int
    n_bootstrap: int
    seed: int
    material_margin: float
    significance_verdict: str
    is_thin_evidence: bool

    @property
    def point_improvement(self) -> float:
        """``mean(surrogate) − mean(candidate)`` — positive = candidate better.

        The effect size on the §4 valid_loss axis: how much lower the
        candidate's valid_loss sits than the random-order surrogate's, on
        average. Distinct from :attr:`significance_verdict` (a CI-vs-zero call)
        and gated separately by :attr:`is_material` — the §7 separation of
        statistical significance from practical concentration.
        """
        return self.surrogate_mean - self.candidate_mean

    @property
    def significant_surpasses(self) -> bool:
        """CI excludes zero on the candidate's side (the statistical axis)."""
        return self.significance_verdict == SURPASSES

    @property
    def is_material(self) -> bool:
        """Point lead clears the material margin (the practical axis, §7).

        A statistically significant but immaterial lead (small
        :attr:`point_improvement` clearing zero yet below ``material_margin``)
        reads ``significant_surpasses=True`` and ``is_material=False`` — exactly
        the 2.6σ-but-useless case §7 refuses to let a paper claim as a win.
        """
        return self.point_improvement >= self.material_margin

    @property
    def passes(self) -> bool:
        """The §4 bar: significant *and* material.

        Both §7 axes must clear: the lead must survive resampling
        (:attr:`significant_surpasses`) *and* be large enough to matter
        (:attr:`is_material`). :attr:`is_thin_evidence` is reported alongside,
        not folded in — matching how :class:`freeze_cost.ConfidenceBand` keeps
        its thin-evidence flag orthogonal to the verdict, so the audit (not the
        boolean) carries the "do not read as confirmed" honesty.
        """
        return self.significant_surpasses and self.is_material


def bootstrap_difference_ci(
    candidate_losses: Sequence[float],
    surrogate_losses: Sequence[float],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_CI_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on ``mean(surrogate) − mean(candidate)``.

    The two-sample independent bootstrap GOAL §4 names for the valid_loss axis:
    each of ``n_bootstrap`` draws resamples ``candidate_losses`` and
    ``surrogate_losses`` *independently* with replacement and records the
    difference of resampled means (positive = candidate better). The central
    ``confidence`` percentile interval of those draws is the CI; the point
    estimate is the observed difference of means.

    Independent (not paired) resampling is the general case: candidate and
    surrogate need not share a seed axis or sample size, which matches how the
    §4 multi-seed control is actually produced (a candidate run over its seeds
    vs. the surrogate distribution over the surrogate seeds). The returned
    ``(point, lower, upper)`` triple is all in valid_loss units.

    Determinism is total for a fixed ``seed`` (``numpy.default_rng``); the same
    samples and seed reproduce the CI bit-for-bit, which is what makes the §4
    significance statement reproducible across runs.
    """
    candidate = np.asarray(_require_non_empty(candidate_losses), dtype=float)
    surrogate = np.asarray(_require_non_empty(surrogate_losses), dtype=float)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    # Draw all resample indices at once: idx arrays of shape (n_bootstrap, n).
    # Indexing gathers the resampled samples; the mean over axis 1 is each
    # draw's resampled mean. Vectorized so 10k draws on a handful of seeds is
    # sub-millisecond and free of a Python-level loop.
    cand_idx = rng.integers(0, candidate.size, size=(n_bootstrap, candidate.size))
    surr_idx = rng.integers(0, surrogate.size, size=(n_bootstrap, surrogate.size))
    improvements = surrogate[surr_idx].mean(axis=1) - candidate[cand_idx].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(improvements, 100.0 * alpha / 2.0))
    upper = float(np.percentile(improvements, 100.0 * (1.0 - alpha / 2.0)))
    point = float(surrogate.mean() - candidate.mean())
    return point, lower, upper


def surrogate_valid_loss_ci(
    candidate_valid_losses: Sequence[float],
    surrogate_valid_losses: Sequence[float],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_CI_SEED,
    material_margin: float = 0.0,
) -> SurrogateValidLossCI:
    """Significance-graded GOAL §4 verdict on the candidate-vs-surrogate
    valid_loss difference.

    The bootstrap layer that fills GOAL §4's "valid_loss 差はブートストラップ
    CI で評価" line. Given a candidate valid_loss sample and a surrogate
    (random-order freeze) valid_loss sample — the two quantities a GPU run
    deposits and :func:`surrogate_exceedance` already consumes structurally —
    it returns the central ``confidence`` bootstrap CI on
    ``mean(surrogate) − mean(candidate)`` and the verdict that promotes the
    surrogate-exceedance label to significance-graded: ``SURPASSES`` if the CI
    is entirely above zero, ``UNDERSHOOTS`` if entirely below, else ``TIES``.

    Parameters
    ----------
    candidate_valid_losses / surrogate_valid_losses:
        One valid_loss per seed for each arm (GOAL §4 "各条件は複数シードで
        回す"). Lower is better. The surrogate *is* the §4 random-order null the
        candidate's lead is judged against.
    n_bootstrap / confidence / seed:
        Bootstrap draw count, two-sided confidence level, and the RNG seed that
        pins the CI for reproducibility.
    material_margin:
        Minimum :attr:`~SurrogateValidLossCI.point_improvement` (valid_loss
        units) for :attr:`~SurrogateValidLossCI.is_material`. The §7 separation
        of significance from practical importance: default ``0.0`` makes the
        material axis a pure "lead is positive" check (so ``passes`` mirrors
        ``significant_surpasses``); set it to a baseline-derived threshold
        (GOAL §4: "閾値はベースライン分散から決定") to require a material, not
        merely statistically detectable, lead.
    """
    if material_margin < 0.0:
        raise ValueError(f"material_margin must be non-negative, got {material_margin}")

    candidate = tuple(float(v) for v in candidate_valid_losses)
    surrogate = tuple(float(v) for v in surrogate_valid_losses)
    _require_non_empty(candidate)
    _require_non_empty(surrogate)

    point, lower, upper = bootstrap_difference_ci(
        candidate,
        surrogate,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    if lower > 0.0:
        verdict = SURPASSES
    elif upper < 0.0:
        verdict = UNDERSHOOTS
    else:
        verdict = TIES

    return SurrogateValidLossCI(
        candidate_mean=float(np.mean(candidate)),
        surrogate_mean=float(np.mean(surrogate)),
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_candidate=len(candidate),
        n_surrogate=len(surrogate),
        n_bootstrap=n_bootstrap,
        seed=seed,
        material_margin=material_margin,
        significance_verdict=verdict,
        is_thin_evidence=(
            len(candidate) < MIN_SAMPLE_FOR_BOOTSTRAP
            or len(surrogate) < MIN_SAMPLE_FOR_BOOTSTRAP
        ),
    )


def format_surrogate_valid_loss_ci(ci: SurrogateValidLossCI) -> str:
    """Render the GOAL §4 bootstrap valid_loss-CI verdict with full provenance.

    A compact, deterministic audit block (the layer a gate or a reader of the
    experiment's significance report inspects): the significance verdict, the
    candidate-vs-surrogate means and the bootstrap CI on their difference, the
    §7 materiality flag against the caller-set margin, and the thin-evidence
    label. The raw point improvement (effect size) stays visible alongside the
    CI so a reader sees the lead the verdict was drawn from, not just the label
    — and when the sample was too thin for the bootstrap to capture an arm's
    variance, the text states plainly that the verdict must not be read as
    confirmed.
    """
    lines = [
        f"surrogate_valid_loss_ci: {ci.significance_verdict} "
        f"(passes={ci.passes})",
        f"  valid_loss_axis: candidate_mean={ci.candidate_mean:.6f} vs "
        f"surrogate_mean={ci.surrogate_mean:.6f} "
        f"n_candidate={ci.n_candidate} n_surrogate={ci.n_surrogate}",
        f"  improvement: point={ci.point_improvement:.6f} "
        f"ci[{ci.confidence:.0%}]=[{ci.lower:.6f}, {ci.upper:.6f}]",
        f"  material: is_material={ci.is_material} "
        f"(margin={ci.material_margin:.6f})",
        f"  bootstrap: n={ci.n_bootstrap} seed={ci.seed}",
    ]
    if ci.is_thin_evidence:
        # A thin sample cannot anchor a significance statement: an arm below
        # MIN_SAMPLE_FOR_BOOTSTRAP resamples to a near-constant, so the CI
        # misses that arm's run-to-run variance and reads narrower than the
        # truth. Flag it so a thin SURPASSES is never read as a confirmed win.
        lines.append(
            "  note: THIN_EVIDENCE — an arm has fewer than "
            f"{MIN_SAMPLE_FOR_BOOTSTRAP} seeds; do not read this verdict as "
            "confirmed (the bootstrap cannot capture that arm's variance)"
        )
    return "\n".join(lines)


def _require_non_empty(values: Sequence[float]) -> Sequence[float]:
    """Reject an empty arm: a bootstrap needs at least one observation per arm."""
    if len(values) < 1:
        raise ValueError(
            "a bootstrap CI requires at least one valid_loss per arm; "
            "got an empty candidate or surrogate sample"
        )
    return values


__all__ = [
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_CI_SEED",
    "MIN_SAMPLE_FOR_BOOTSTRAP",
    "SurrogateValidLossCI",
    "bootstrap_difference_ci",
    "surrogate_valid_loss_ci",
    "format_surrogate_valid_loss_ci",
]
