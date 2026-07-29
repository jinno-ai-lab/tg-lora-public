"""Regression guard pinning the MEASURED fidelity delta of the 555cae4 fix.

``src/eval/eval_json_extraction.extract_json`` is the TG-LoRA headline quality
metric (perplexity's replacement). Commit 555cae4 replaced its greedy
``\\{.*\\}`` lenient fallback with a balanced-brace scan so a JSON object
followed by explanatory prose containing a ``}`` (a routine model output) no
longer over-captures, fails ``json.loads``, and scores ``valid=0``.

The fix's correctness was pinned structurally in ``test_eval_json_extraction``.
These tests pin its BEHAVIORAL IMPACT — the count of routine model outputs the
fix moves from invalid to valid — converting the fix from plausible to
MEASURED (the principle AI-Hub feedback bullet 2 asks for; chunker is phantom
in this checkout, so the principle is applied to the real json_extraction fix).

The pinned number (4 rescued of 9 routine outputs, 0 regressed) is the
high-leverage evidence: if a future change to ``extract_json`` regresses
extraction, ``rescued_count`` drops below 4 (or ``regressed`` grows) and these
tests fail loud.
"""

from __future__ import annotations

from src.eval.eval_json_extraction import extract_json
from scripts.measure_extraction_fidelity_delta import (
    ROUTINE_CORPUS,
    _legacy_greedy_extract,
    measure_fidelity_delta,
)

# The MEASURED delta, recorded 2026-07-29 by running
# `python scripts/measure_extraction_fidelity_delta.py` against ROUTINE_CORPUS.
# 9 routine model outputs: balanced-brace rescues 4 that the pre-555cae4 greedy
# regex scored valid=0; 0 regressions. This is the count feedback bullet 2 asks
# to pin so the next iteration knows fidelity moved outputs, not just a
# structural invariant.
PINNED_RESCUED_COUNT = 4
PINNED_CORPUS_SIZE = 9


def test_legacy_extractor_is_byte_faithful_to_prefix_on_clean_json():
    """The control baseline must agree with the current extractor on clean JSON
    (both strict-valid) so the delta isolates the lenient-path change, not the
    shared strict path."""
    for label, text in ROUTINE_CORPUS:
        if label == "clean_json":
            cur_obj, cur_strict = extract_json(text)
            leg_obj, leg_strict = _legacy_greedy_extract(text)
            assert cur_obj == leg_obj and cur_obj is not None
            assert cur_strict is True and leg_strict is True


def test_balanced_rescues_more_routine_outputs_than_greedy():
    result = measure_fidelity_delta()
    assert result["candidate_valid"] > result["baseline_valid"], (
        f"balanced ({result['candidate_valid']}) must beat greedy "
        f"({result['baseline_valid']}) on routine outputs — if not, the 555cae4 "
        "fix stopped moving outputs."
    )


def test_rescued_count_pinned_at_measured_value():
    """The headline number: how many routine outputs the fix rescued."""
    result = measure_fidelity_delta()
    assert result["total"] == PINNED_CORPUS_SIZE
    assert result["rescued_count"] == PINNED_RESCUED_COUNT, (
        f"rescued_count moved {PINNED_RESCUED_COUNT} -> {result['rescued_count']}; "
        f"rescued patterns: {result['rescued']}. If this increased, a new routine "
        "pattern was added (update the pin deliberately). If it decreased, "
        "extract_json REGRESSED — a routine output now scores valid=0."
    )
    # The four routine patterns the greedy regex over-captured on.
    assert set(result["rescued"]) == {
        "json_plus_note_with_brace",
        "fenced_json_plus_note",
        "two_records_concatenated",
        "json_plus_trailing_jsonish_ref",
    }


def test_balanced_never_regresses_below_greedy():
    """The fix must not lose any case the greedy path handled (zero regressions)."""
    result = measure_fidelity_delta()
    assert result["regressed_count"] == 0, (
        f"extract_json regressed on routine outputs the greedy path handled: "
        f"{result['regressed']}"
    )


def test_canonical_observed_failure_legacy_invalid_balanced_valid():
    """The exact 555cae4 observed failure: JSON + a note containing '}'.

    Greedy over-captures into the trailing prose -> json.loads fails -> valid=0.
    Balanced returns the object -> valid=1. This is the load-bearing regression
    pinned at the unit level too; asserted here at the corpus level for parity.
    """
    text = next(t for lbl, t in ROUTINE_CORPUS if lbl == "json_plus_note_with_brace")
    leg_obj, _ = _legacy_greedy_extract(text)
    cur_obj, cur_strict = extract_json(text)
    assert leg_obj is None, "greedy should over-capture and fail on JSON + brace-prose"
    assert cur_obj is not None and cur_strict is False, (
        "balanced should leniently rescue the JSON object from brace-prose"
    )
    assert cur_obj.get("type") == "meeting"


def test_guard_detects_regression_to_greedy():
    """If extract_json were reverted to the greedy behavior, the delta collapses.

    Swapping the candidate for the legacy baseline (simulating a revert of
    555cae4) must drive rescued_count to 0 — proving this guard would catch a
    real regression, not pass vacuously.
    """
    result = measure_fidelity_delta(candidate=_legacy_greedy_extract)
    assert result["rescued_count"] == 0
    assert result["candidate_valid"] == result["baseline_valid"]


def test_corpus_covers_nontrivial_routine_patterns():
    """Guard against the corpus silently shrinking to only easy cases."""
    labels = {lbl for lbl, _ in ROUTINE_CORPUS}
    assert "clean_json" in labels
    assert "json_plus_note_with_brace" in labels  # the 555cae4 observed failure
    assert "brace_in_string_value" in labels      # no regression on braces-in-string
    assert "nested_object" in labels              # nested objects handled
    assert "fenced_json" in labels                # markdown fences handled
