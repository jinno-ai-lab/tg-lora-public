"""Unit tests for the §4 citation-gate honesty leaf (``freeze_verdict_honesty``).

Why this file exists. ``classify_regime`` / ``is_reduced_budget`` /
``full_section4_verdict_gate`` are the **single source of truth** (SYSTEM_CONSTITUTION
Rule #3) for the 9B §4 verdict's citability: the producer
(``scripts.run_freeze_validloss_ci_9b``) imports them ``as _classify_regime`` etc.
when it stamps ``citable_as_full_section4_verdict`` on a deposit, and the GPU-free
replay re-derives the same axes from them — so the producer's stored boolean and
the replay's re-judged verdict cannot drift apart by construction. They are a
pure-stdlib (``math`` only) leaf with no torch / numpy / ``src.data`` dependency,
exactly so the replay can import them without the training stack.

They are already exercised exhaustively in ``tests/test_run_freeze_validloss_ci_9b.py``
— **but that file is un-importable on the public mirror**: its module-level
``import scripts.run_freeze_validloss_ci_9b`` pulls the private ``src.data``
pipeline, so the whole file errors at collection time and NONE of those
assertions execute here. Measured coverage of the leaf on this checkout was
therefore **0%** (every line), meaning a flipped threshold or a dropped conjunct
would pass this mirror's CI silently.

This file is the **src.data-free, torch-free runnable twin**: it imports the leaf
directly and pins the exact thresholds / conjunction the §4 SHIP verdict rests
on, mutation-proven at each branch and boundary. It is NOT a second copy of the
gate logic — it tests the canonical function the producer itself calls.
"""

from __future__ import annotations

import pytest

from src.tg_lora.freeze_verdict_honesty import (
    REGIME_GENERALIZATION,
    REGIME_MEMORIZATION,
    REGIME_OVERFIT,
    REGIME_UNKNOWN,
    __all__ as LEAF_ALL,
    classify_regime,
    full_section4_verdict_gate,
    is_reduced_budget,
)

# Grounded constants: the leaf's own thresholds, pinned here so a silent edit to
# either side of the inequality is caught.
_MEMORIZATION_TRAIN_CE_FLOOR = 0.5
_OVERFIT_GAP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """The exported regime vocabulary is exactly the four labels the gate speaks."""

    def test_regime_constants_are_the_four_distinct_labels(self):
        labels = {REGIME_GENERALIZATION, REGIME_MEMORIZATION, REGIME_OVERFIT, REGIME_UNKNOWN}
        assert labels == {"generalization", "memorization", "overfit", "unknown"}

    def test_all_lists_exactly_the_public_api(self):
        # A conjunct added to full_section4_verdict_gate without being re-derived by
        # the producer's serializer would silently desync the two; __all__ is the
        # contract that the producer + replay import, so pin it.
        assert set(LEAF_ALL) == {
            "REGIME_GENERALIZATION",
            "REGIME_MEMORIZATION",
            "REGIME_OVERFIT",
            "REGIME_UNKNOWN",
            "classify_regime",
            "is_reduced_budget",
            "full_section4_verdict_gate",
        }


# ---------------------------------------------------------------------------
# classify_regime — the one axis a bare scale/budget gate would miss
# ---------------------------------------------------------------------------


class TestClassifyRegime:
    """Discretize the candidate arm's train/valid CE into the regime vocabulary.

    Grounded in the committed 9B deposits (see the leaf docstring):
      * generalization arms: final_ce_train ≈ 1.507, valid ≈ 1.515 → gap ≈ 0.008;
      * the full-backprop BASELINE overfits: final_ce 0.77 ≪ valid 1.54 → gap 0.77;
      * memorization arms (8 train × 20 step): train CE collapses toward 0.
    """

    @pytest.mark.parametrize(
        "ce, vl, expected",
        [
            # generalization: train CE well above the floor, train≈valid
            (1.507, 1.515, REGIME_GENERALIZATION),
            # memorization: train CE collapsed below the floor
            (0.001, 1.6, REGIME_MEMORIZATION),
            (0.0, 1.0, REGIME_MEMORIZATION),
            # overfit: train CE above the floor BUT a large train-valid gap
            (0.77, 1.54, REGIME_OVERFIT),
        ],
    )
    def test_regime_classification(self, ce, vl, expected):
        assert classify_regime(ce, vl) == expected

    @pytest.mark.parametrize(
        "ce, vl",
        [
            (None, 1.5),        # TypeError → UNKNOWN (deposit predates the diagnostic)
            ("", 1.5),          # ValueError → UNKNOWN
            ("not-a-number", 2.0),
        ],
    )
    def test_non_numeric_inputs_are_unknown(self, ce, vl):
        # The conservative call: a deposit recorded before final_ce_train_loss
        # existed must NEVER open the full-§4 citation gate.
        assert classify_regime(ce, vl) == REGIME_UNKNOWN

    @pytest.mark.parametrize(
        "ce, vl",
        [
            (float("nan"), 1.5),
            (1.5, float("inf")),
            (float("inf"), 1.5),
        ],
    )
    def test_non_finite_inputs_are_unknown(self, ce, vl):
        assert classify_regime(ce, vl) == REGIME_UNKNOWN

    def test_memorization_floor_is_strict_less_than(self):
        # Boundary that pins the `<` operator at the train-CE floor: a candidate
        # AT the floor (0.5) is NOT memorization; one just below (0.499) is.
        # Mutation `<` → `<=` flips exactly these two.
        assert classify_regime(0.499, 1.0) == REGIME_MEMORIZATION
        assert classify_regime(_MEMORIZATION_TRAIN_CE_FLOOR, 1.0) == REGIME_GENERALIZATION

    def test_overfit_gap_threshold_is_strict_greater_than(self):
        # Boundary that pins the `>` operator at the overfit gap: a gap exactly at
        # the threshold (0.5) is generalization; one just over (0.5001) is overfit.
        # Mutation `>` → `>=` flips exactly these two.
        assert classify_regime(1.0, 1.5) == REGIME_GENERALIZATION          # gap == 0.5
        assert classify_regime(1.0, 1.0 + _OVERFIT_GAP_THRESHOLD + 1e-4) == REGIME_OVERFIT


# ---------------------------------------------------------------------------
# is_reduced_budget — the honest (budget-driven) reduced-budget flag
# ---------------------------------------------------------------------------


class TestIsReducedBudget:
    """A run is reduced-budget unless it trained for the config's full max_steps.

    A hardcoded ``reduced_budget=True`` would silently lie about a future
    full-length run and keep the citation gate permanently closed; deriving it
    from ``total_steps`` vs ``max_steps`` is what lets a full run clear the axis.
    """

    @pytest.mark.parametrize(
        "total_steps, max_steps, expected",
        [
            (40, 1500, True),     # stopped early → reduced
            (1500, 1500, False),  # trained the full intended length → NOT reduced
            (2000, 1500, False),  # trained past it → NOT reduced
        ],
    )
    def test_budget_classification(self, total_steps, max_steps, expected):
        assert is_reduced_budget(total_steps, max_steps) is expected

    def test_equality_boundary_pins_strict_less_than(self):
        # Pins the `<` in ``total_steps < max_steps``: equal-length is full budget.
        # Mutation `<` → `<=` would mark a full run reduced.
        assert is_reduced_budget(total_steps=1500, max_steps=1500) is False
        assert is_reduced_budget(total_steps=1499, max_steps=1500) is True

    @pytest.mark.parametrize("max_steps", [0, -1, -999])
    def test_absent_max_steps_is_conservatively_reduced(self, max_steps):
        # ``max_steps <= 0`` (absent / unparsed config) is treated as reduced —
        # never silently promote a run whose intended length is unknown.
        # Mutation ``<= 0`` → ``< 0`` would un-promote the max_steps == 0 case.
        assert is_reduced_budget(total_steps=1500, max_steps=max_steps) is True


# ---------------------------------------------------------------------------
# full_section4_verdict_gate — the 4-conjunct citation gate
# ---------------------------------------------------------------------------


class TestFullSection4VerdictGate:
    """Citable as the COMPLETE §4 verdict ONLY when all four axes clear.

    target-scale (not proxy) AND full-budget (not reduced) AND non-thin (enough
    seeds) AND generalization regime. Each axis is the one place a future conjunct
    must change; these tests mutation-prove that dropping any single ``and``
    clause re-opens the gate on exactly the run that axis exists to block.
    """

    CLEARING = dict(
        proxy_scale=False,
        reduced_budget=False,
        is_thin_evidence=False,
        regime=REGIME_GENERALIZATION,
    )

    def test_all_axes_clear_opens_the_gate(self):
        assert full_section4_verdict_gate(**self.CLEARING) is True

    @pytest.mark.parametrize(
        "override",
        [
            {"proxy_scale": True},
            {"reduced_budget": True},
            {"is_thin_evidence": True},
            {"regime": REGIME_MEMORIZATION},
            {"regime": REGIME_OVERFIT},
            {"regime": REGIME_UNKNOWN},
        ],
    )
    def test_any_single_failed_axis_closes_the_gate(self, override):
        # Each blocked axis, alone, flips True → False. Mutation-proven: removing
        # the matching ``and (not ...)`` / ``and regime == GENERALIZATION`` clause
        # turns exactly its row green-where-it-should-be-red.
        blocked = {**self.CLEARING, **override}
        assert full_section4_verdict_gate(**blocked) is False

    def test_unknown_regime_never_opens_the_gate(self):
        # The conservative call called out in the docstring: an unverifiable
        # regime (legacy deposit, missing diagnostic) can never be over-cited.
        assert (
            full_section4_verdict_gate(
                proxy_scale=False,
                reduced_budget=False,
                is_thin_evidence=False,
                regime=REGIME_UNKNOWN,
            )
            is False
        )
