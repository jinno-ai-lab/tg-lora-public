"""GOAL §4 surrogate-exceedance gate: is a candidate schedule real, or could a
random freeze order match it?

GOAL §4 makes the random-order freeze (design Phase 2 control-(ii)) the null any
real schedule must beat before its reduction or quality retention is claimed:

    "ランダム順フリーズ（サロゲート）を超えた削減・性能だけを有効と認定"
    "「計算が減った」「性能が保てた」も対照を超えて初めて主張"
    "backward FLOPs がランダム順フリーズ対照を有意に超えて削減できる"

:func:`src.tg_lora.freeze_schedule.random_freeze_order` already provides the
reproducible surrogate generator and documents that a shuffled order flows
through the *identical* planner/accountant path as a real schedule (no separate
random branch) — so the candidate-vs-surrogate comparison is apples-to-apples.
What was missing was the *judgement*: one function that compares a candidate
schedule's reduction against the distribution of random-order surrogates over
GOAL §4's required multiple seeds ("各条件は複数シードで回す") and says whether
the candidate clears the bar. This module is that judgement.

Two axes, matching GOAL §4 line 242 (both "compute decreased" and "performance
maintained" must exceed the control):

* **FLOPs axis** — ``1 − progressive/full`` (GOAL §5). Exact, model-free, CPU.
  The candidate's reduction is compared against the surrogate reduction each
  seeded random order realizes at the same ``(depth, start, spacing)``.
* **valid_loss axis** — quality retention. GPU-dependent (classification C), so
  the gate threads it structurally: a GPU run deposits ``candidate_valid_loss``
  and ``surrogate_valid_losses`` and the verdict tightens, but the call and the
  FLOPs-axis judgement are unchanged when those are absent (flagged
  ``valid_loss_unverified``).

Honesty (the keystone): on a *homogeneous* stack (uniform per-layer cost) the
FLOPs reduction depends only on ``(depth, timing)``, never order, so a real
schedule's reduction equals every surrogate's and the verdict is correctly
``TIES`` — you cannot beat random by reordering when order is irrelevant. Making
that a verdict (not a hidden assumption) is exactly the §4 honesty the gate
exists to enforce; a discriminating ``SURPASSES`` needs non-uniform per-layer
costs (GOAL §1.5/§8: DeltaNet vs Attention), which is the regime the Phase 2
sweep is designed to surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from src.tg_lora.freeze_frontier import FrontierSpec, evaluate_schedule
from src.tg_lora.freeze_schedule import random_freeze_order

# Graduated verdict labels. A bare boolean cannot carry the §4 honesty gradation
# once a finite seed sample is involved: "exceeds random" is SURPASSES only when
# the candidate clears even the luckiest random arm, not when it merely looks
# better than one. TIES is the honest "indistinguishable from random" outcome
# (the verdict every homogeneous-stack comparison correctly returns).
SURPASSES = "SURPASSES"
TIES = "TIES"
UNDERSHOOTS = "UNDERSHOOTS"

# Default seeds for the surrogate distribution. GOAL §4 "各条件は複数シードで
# 回す" requires the random control to be a distribution, not one anecdote; this
# carries five independent reproducible orderings by default.
DEFAULT_SURROGATE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Margin (absolute reduction units) a candidate must clear above the best
# surrogate to count as SURPASSES. Reductions are exact FLOP ratios, so a
# statistical dead-heat (candidate == best surrogate) must read as TIES, not a
# win — a tiny epsilon enforces "有意に超えて" without floating-point noise
# flipping an exact tie. Bump it to require a material lead.
DEFAULT_EXCEEDANCE_MARGIN: float = 1e-9

_RANK: dict[str, int] = {UNDERSHOOTS: 0, TIES: 1, SURPASSES: 2}


def _axis_verdict(
    candidate: float,
    surrogates: Sequence[float],
    *,
    lower_is_better: bool,
    margin: float,
) -> str:
    """Graduated verdict of ``candidate`` vs the surrogate distribution.

    ``SURPASSES`` requires clearing the *best* surrogate by ``margin``: a
    candidate must beat even the luckiest random arm, which is the conservative
    direction (more seeds can only make this harder, never easier). ``lower_is_
    better`` flips the comparison for the valid_loss axis (less degradation is
    better) versus the FLOPs axis (more reduction is better).
    """
    if lower_is_better:
        if candidate < min(surrogates) - margin:
            return SURPASSES
        if candidate > max(surrogates) + margin:
            return UNDERSHOOTS
        return TIES
    if candidate > max(surrogates) + margin:
        return SURPASSES
    if candidate < min(surrogates) - margin:
        return UNDERSHOOTS
    return TIES


def _surrogate_flops_reduction(
    spec: FrontierSpec, depth: int, level: int, seed: int
) -> float:
    """One seeded random-order surrogate's FLOPs reduction, via the prod path.

    Reuses :func:`evaluate_schedule` (the planner→accountant production glue) so
    the surrogate flows through the identical code path as a real schedule: the
    random order is injected as a ``convergence_order`` request at the same
    ``(depth, start, spacing, num_epochs)``. No separate random branch exists
    (design §5.3), which is what makes candidate-vs-surrogate apples-to-apples.
    """
    order = random_freeze_order(spec.active_layer_indices, seed)
    surrogate_spec = replace(
        spec,
        policies=("convergence_order",),
        convergence_order=order,
        stability_epoch=None,
    )
    return evaluate_schedule(surrogate_spec, "convergence_order", depth, level).reduction_rate


@dataclass(frozen=True)
class SurrogateExceedance:
    """Whether a candidate schedule clears the GOAL §4 random-order surrogate.

    The verdict is graduated on each axis (FLOPs exact/CPU; valid_loss
    GPU-dependent). ``overall_verdict`` is the conservative conjunction —
    SURPASSES only when *every measured* axis SURPASSES — so a full §4 claim
    ("compute decreased *and* performance maintained, both beyond the control")
    cannot be made on one axis while the other is unmeasured or merely ties.

    ``valid_loss_unverified`` is the honesty flag for the classification-C axis:
    when no GPU run has deposited valid_loss numbers, the overall verdict is the
    FLOPs-axis verdict alone, and the audit must say so rather than imply a
    two-axis claim. On a homogeneous stack ``flops_verdict`` is ``TIES`` by
    construction (order cannot matter) — the verdict, not a hidden assumption.
    """

    flops_verdict: str
    valid_loss_verdict: str | None
    valid_loss_unverified: bool
    overall_verdict: str
    candidate_flops_reduction: float
    surrogate_flops_reductions: tuple[float, ...]
    candidate_valid_loss: float | None
    surrogate_valid_losses: tuple[float, ...] | None
    policy: str
    depth: int
    level: int
    seeds: tuple[int, ...]
    margin: float

    @property
    def passes(self) -> bool:
        """True only when the candidate SURPASSES the surrogate on every measured axis."""
        return self.overall_verdict == SURPASSES


def surrogate_exceedance(
    spec: FrontierSpec,
    policy: str,
    depth: int,
    level: int = 1,
    *,
    seeds: Sequence[int] = DEFAULT_SURROGATE_SEEDS,
    margin: float = DEFAULT_EXCEEDANCE_MARGIN,
    candidate_valid_loss: float | None = None,
    surrogate_valid_losses: Sequence[float] | None = None,
) -> SurrogateExceedance:
    """Judge whether a candidate schedule exceeds the GOAL §4 random surrogate.

    Measures the candidate's FLOPs reduction via :func:`evaluate_schedule` (the
    planner→accountant production glue) and compares it against the reduction
    each seeded ``random_freeze_order`` realizes at the same
    ``(depth, start, spacing)`` — the surrogate-null distribution GOAL §4
    demands any real schedule beat. A candidate ``SURPASSES`` only if it clears
    the best surrogate by ``margin``; on a homogeneous stack that is impossible
    (order is irrelevant) so the verdict is honestly ``TIES``.

    The valid_loss axis is threaded structurally: supply
    ``candidate_valid_loss`` and ``surrogate_valid_losses`` (one per seed) to
    judge quality retention (lower degradation is better), and the overall
    verdict then requires *both* axes to SURPASS. Leave them ``None`` (the
    classification-C default) and the verdict is the FLOPs-axis one alone,
    flagged ``valid_loss_unverified`` — the §4 claim stands on one axis, honestly
    labelled, until a GPU run deposits the other.

    Parameters
    ----------
    spec:
        The Phase 2 frontier spec the candidate and every surrogate share
        (same layers, costs, ``num_epochs``, ``start_epoch``, ``spacing``).
    policy / depth / level:
        The candidate schedule to judge (reused verbatim by
        :func:`evaluate_schedule`).
    seeds:
        Seeds for the surrogate distribution (GOAL §4 multi-seed). The surrogate
        count equals ``len(seeds)``.
    margin:
        Absolute reduction lead a candidate needs over the best surrogate to
        SURPASSES (defaults to a tie-breaking epsilon).
    candidate_valid_loss / surrogate_valid_losses:
        Optional GPU-measured quality numbers (lower is better). The surrogate
        sample length should match ``seeds`` when supplied.
    """
    seeds_tuple = tuple(seeds)
    if not seeds_tuple:
        raise ValueError("seeds must be non-empty — the surrogate is a distribution, not one anecdote")
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    candidate_reduction = evaluate_schedule(spec, policy, depth, level).reduction_rate
    surrogate_reductions = tuple(
        _surrogate_flops_reduction(spec, depth, level, seed) for seed in seeds_tuple
    )

    flops_verdict = _axis_verdict(
        candidate_reduction, surrogate_reductions, lower_is_better=False, margin=margin
    )

    if candidate_valid_loss is None or surrogate_valid_losses is None:
        valid_loss_verdict: str | None = None
        valid_loss_unverified = True
        overall = flops_verdict
    else:
        surr_vl = tuple(surrogate_valid_losses)
        valid_loss_verdict = _axis_verdict(
            candidate_valid_loss, surr_vl, lower_is_better=True, margin=margin
        )
        valid_loss_unverified = False
        # Conservative conjunction: SURPASSES only if every measured axis does.
        overall = min((flops_verdict, valid_loss_verdict), key=lambda v: _RANK[v])

    return SurrogateExceedance(
        flops_verdict=flops_verdict,
        valid_loss_verdict=valid_loss_verdict,
        valid_loss_unverified=valid_loss_unverified,
        overall_verdict=overall,
        candidate_flops_reduction=candidate_reduction,
        surrogate_flops_reductions=surrogate_reductions,
        candidate_valid_loss=candidate_valid_loss,
        surrogate_valid_losses=tuple(surrogate_valid_losses)
        if surrogate_valid_losses is not None
        else None,
        policy=policy,
        depth=depth,
        level=level,
        seeds=seeds_tuple,
        margin=margin,
    )


def format_surrogate_exceedance(exceedance: SurrogateExceedance) -> str:
    """Render the GOAL §4 surrogate-exceedance verdict with full provenance.

    A compact, deterministic audit block: the overall verdict, each axis's
    verdict, the candidate reduction against the surrogate distribution (min/mean
    /max over the seeded sample), and the valid_loss honesty flag. The raw
    surrogate sample stays visible so a reader sees the distribution the
    candidate was judged against, not just the verdict — the same transparency
    the §7 proxy verdict carries. ``valid_loss_unverified`` is stated plainly so
    the audit never reads a one-axis verdict as a full §4 claim.
    """
    surrogates = exceedance.surrogate_flops_reductions
    s_min = min(surrogates) if surrogates else 0.0
    s_mean = sum(surrogates) / len(surrogates) if surrogates else 0.0
    s_max = max(surrogates) if surrogates else 0.0
    lines = [
        f"surrogate_exceedance: {exceedance.overall_verdict} "
        f"(passes={exceedance.passes})",
        f"  flops_axis: {exceedance.flops_verdict} "
        f"candidate={exceedance.candidate_flops_reduction:.6f} vs "
        f"surrogate[min={s_min:.6f} mean={s_mean:.6f} max={s_max:.6f}] "
        f"n={len(surrogates)}",
    ]
    if exceedance.valid_loss_verdict is None:
        lines.append(
            "  valid_loss_axis: UNVERIFIED (GPU-dependent, classification C) — "
            "verdict stands on the FLOPs axis alone"
        )
    else:
        vl = exceedance.surrogate_valid_losses or ()
        lines.append(
            f"  valid_loss_axis: {exceedance.valid_loss_verdict} "
            f"candidate={exceedance.candidate_valid_loss:.6f} vs "
            f"surrogate[min={min(vl) if vl else 0.0:.6f} "
            f"max={max(vl) if vl else 0.0:.6f}] n={len(vl)}"
        )
    lines.append(
        f"  schedule: policy={exceedance.policy} depth={exceedance.depth} "
        f"level={exceedance.level} margin={exceedance.margin:.2e}"
    )
    return "\n".join(lines)
