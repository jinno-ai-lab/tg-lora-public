"""The GOAL §4 citation-gate honesty primitives — torch-free, single source of truth.

A 9B §4 verdict is citable as the *complete* result ONLY when it clears four
honesty axes: target-scale (not a proxy), full-budget (reached the config's
``max_steps``), non-thin (enough seeds for the bootstrap to capture variance),
and generalization regime (the candidate generalized rather than memorized).
These axes are computed by the *producer*
(:func:`scripts.run_freeze_validloss_ci_9b.result_to_json`) when it stamps
``citable_as_full_section4_verdict`` on a deposit, AND re-derived by the
*consumer* (:mod:`scripts.replay_freeze_validloss_ci`) when it re-judges a
stored deposit GPU-free. Keeping the per-axis logic in this one torch-free leaf
means the two never compute citability from divergent thresholds or rules
(SYSTEM_CONSTITUTION Rule #3 single source of truth): a regime threshold or the
reduced-budget rule tuned here changes producer and consumer at once, so a
committed deposit's stored boolean and the replay's re-derived verdict cannot
drift apart by definition.

The leaf is pure stdlib (``math`` only) — no torch, no numpy — so the GPU-free
replay imports it without pulling the producer's training stack, and the
producer imports it without a second copy of the gate.

Honesty (GOAL §7 鉄則) — the regime axis is the one a bare scale/budget gate
would miss. In the memorization regime (few train examples × many epochs) the
adapter drives train cross-entropy toward 0 and the held-out valid_loss is
dominated by the frozen base, so a "SURPASSES" read off a memorized model is an
artifact, not the §4 question. :func:`classify_regime` makes that regime
machine-readable from the candidate arm's recorded train CE and valid_loss, and
:func:`full_section4_verdict_gate` refuses to open the citation gate on anything
but a verified generalization run — the conservative call (UNKNOWN regime never
opens the gate), so a deposit recorded before the diagnostic existed can never
be over-cited.
"""

from __future__ import annotations

import math

REGIME_GENERALIZATION = "generalization"
REGIME_MEMORIZATION = "memorization"
REGIME_OVERFIT = "overfit"
REGIME_UNKNOWN = "unknown"

# Grounded in the committed 9B deposits, not picked from thin air:
#   * generalization-regime candidate arms (freeze_validloss_ci_9b_generalization
#     / _baseline) have final_ce_train_loss ≈ 1.507 with valid ≈ 1.515 → gap ≈ 0.008;
#   * the full-backprop BASELINE arm overfits with final_ce 0.77 ≪ valid 1.54 → gap 0.77;
#   * memorization-regime arms (8 train × 20 step) collapse train CE toward 0.
# So a train-CE floor of 0.5 separates memorization (~0) from generalization
# (~1.5) with margin, and a train-valid gap threshold of 0.5 separates
# generalization (~0.01) from overfit (~0.77) with margin.
_MEMORIZATION_TRAIN_CE_FLOOR = 0.5
_OVERFIT_GAP_THRESHOLD = 0.5


def classify_regime(final_ce_train_loss, valid_loss):
    """Classify a run's training regime from the candidate arm's train/valid CE.

    ``final_ce_train_loss`` is the candidate arm's mean full cross-entropy over
    the *train* set under the final adapter; ``valid_loss`` is the candidate
    arm's mean held-out valid_loss (:attr:`freeze_surrogate_ci.SurrogateValidLossCI.candidate_mean`).
    Returns one of the :data:`REGIME_*` constants. Anything missing or
    non-finite (e.g. a deposit recorded before the ``final_ce_train_loss``
    diagnostic existed) classifies as :data:`REGIME_UNKNOWN` — the conservative
    call, which never opens the full-§4 citation gate on a regime it cannot
    verify.
    """
    try:
        ce = float(final_ce_train_loss)
        vl = float(valid_loss)
    except (TypeError, ValueError):
        return REGIME_UNKNOWN
    if not (math.isfinite(ce) and math.isfinite(vl)):
        return REGIME_UNKNOWN
    if ce < _MEMORIZATION_TRAIN_CE_FLOOR:
        return REGIME_MEMORIZATION
    if (vl - ce) > _OVERFIT_GAP_THRESHOLD:
        return REGIME_OVERFIT
    return REGIME_GENERALIZATION


def is_reduced_budget(total_steps: int, max_steps: int) -> bool:
    """A run is reduced-budget unless it trained for the config's full
    ``max_steps`` (the §4 verdict's intended training length).

    Keeps the budget axis *honest*: a hardcoded ``reduced_budget=True`` would
    silently lie about a future full-length run, and the citation gate
    (:func:`full_section4_verdict_gate`) that keys off it would stay permanently
    closed no matter how long a run trained. With this, a run that reaches
    ``max_steps`` clears the axis (and a non-thin, target-scale, generalizing one
    becomes citable). ``max_steps <= 0`` (absent / unparsed config) is treated as
    reduced — the conservative call, never silently promoting a run.
    """
    if max_steps <= 0:
        return True
    return total_steps < max_steps


def full_section4_verdict_gate(
    *, proxy_scale: bool, reduced_budget: bool, is_thin_evidence: bool, regime: str
) -> bool:
    """The 4-conjunct citation gate, as one source of truth.

    A run is citable as the COMPLETE §4 verdict ONLY when it clears all four
    axes: target-scale (not a proxy), full-budget (reached config ``max_steps``,
    not reduced), non-thin (enough seeds for the bootstrap to capture variance),
    AND in the generalization regime (the candidate generalized rather than
    memorized — see :func:`classify_regime`). The producer's serializer and the
    deposit self-consistency test (``TestDepositGateSelfConsistency``) both call
    this, and the GPU-free replay re-derives the same axes from a deposit's
    stored artifacts via this leaf — so a future conjunct added here is the ONE
    place it must change, and any committed deposit whose stored boolean predates
    the change is flagged by that test (and, on the replay side, by the
    ``CITATION_LABEL_STALE`` cross-check). The private ``src.data`` quality filter
    is a further axis this gate cannot see on the mirror; it is noted in the
    report, never silently assumed away.
    """
    return (
        (not proxy_scale)
        and (not reduced_budget)
        and (not is_thin_evidence)
        and (regime == REGIME_GENERALIZATION)
    )


__all__ = [
    "REGIME_GENERALIZATION",
    "REGIME_MEMORIZATION",
    "REGIME_OVERFIT",
    "REGIME_UNKNOWN",
    "classify_regime",
    "is_reduced_budget",
    "full_section4_verdict_gate",
]
