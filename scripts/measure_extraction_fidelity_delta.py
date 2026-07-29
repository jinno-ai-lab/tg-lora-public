#!/usr/bin/env python
"""Measure the extraction-fidelity delta of the balanced-brace fix (555cae4).

WHY THIS EXISTS
---------------
``src/eval/eval_json_extraction.extract_json`` is the TG-LoRA headline quality
metric (perplexity's replacement — does the model emit CORRECT structured
output?), wired into the base-model go/no-go gate, both generation-eval
wrappers, and ``config_schema``. Commit 555cae4 replaced its lenient fallback
— a greedy ``\\{.*\\}`` regex (DOTALL) — with a left-to-right balanced-brace
scan, because the greedy regex over-captured a perfect JSON object followed by
explanatory prose that contained a ``}`` (a routine model output: JSON + a
one-line note), failed ``json.loads``, and scored ``valid=0`` (every headline
metric zeroed). A false-negative on the headline metric silently understates
model quality.

That fix was proven correct on ONE observed failure case plus structural
contract tests (555cae4 + ``tests/test_eval_json_extraction.py``). What was
NOT measured: on a corpus of ROUTINE realistic model outputs, HOW MANY does
the balanced scan rescue that the greedy regex scored invalid? That count is
the high-leverage evidence that the fix moved *outputs* (not merely satisfied
a structural invariant) — the number that converts the fix from plausible to
measured.

This tool reproduces the EXACT pre-555cae4 greedy extractor as the "before"
baseline, runs both extractors over a deterministic corpus of routine model
outputs, and reports each extractor's valid count plus the COUNT of rescued
outputs (valid under balanced, invalid under greedy) and any regressed outputs
(valid under greedy, invalid under balanced — must be zero).

NOTE ON SCOPE
~~~~~~~~~~~~~
AI-Hub feedback bullet 2 named a "chunker content-fidelity fix" and asked for a
before/after label-diff count. No ``chunker`` / ``ChunkerConfig`` /
``detect_language`` symbol exists in this checkout (grep-empty — private
AI-Hub infra). This tool applies bullet 2's MEASUREMENT PRINCIPLE to the REAL
content-fidelity fix present here (the 555cae4 JSON-extraction change), which
is the actual load-bearing fidelity surface for the headline metric.

GPU-free, deterministic, offline. Usage::

    python scripts/measure_extraction_fidelity_delta.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

# Allow running as a standalone CLI (``python scripts/measure_extraction_fidelity_delta.py``): a bare
# script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the repo root
# importable so ``src.*`` resolves to THIS repo's own copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.eval_json_extraction import extract_json  # noqa: E402

Extractor = Callable[[str], "tuple[dict | None, bool]"]

# Canonical typed records matching src/eval/eval_json_extraction.SCHEMA_FIELDS.
_MEETING = {
    "type": "meeting",
    "attendee": "Alice Tanaka",
    "date": "2026-07-29",
    "start": "10:00",
    "end": "10:45",
    "location": "Conference Room A",
    "priority": "high",
    "duration_minutes": 45,
}
_PERSON = {
    "type": "person",
    "name": "Bob Sato",
    "role": "Engineer",
    "department": "Research",
    "contact": "bob@example.com",
}
_MEETING_JSON = json.dumps(_MEETING)
_PERSON_JSON = json.dumps(_PERSON)


def _legacy_greedy_extract(text: str) -> "tuple[dict | None, bool]":
    """Faithful reproduction of the PRE-555cae4 ``extract_json`` lenient path.

    This is the "before" baseline for the fidelity-delta measurement — the exact
    greedy ``\\{.*\\}`` (DOTALL) behavior replaced by the balanced-brace scan.
    Kept here (not in the production eval path) as the experimental control; it
    is exercised by ``tests/test_extraction_fidelity_delta.py`` so the delta is
    measured against the real prior behavior, not a paraphrase of it.

    Verified byte-faithful to the removed block in 555cae4 (strict whole-text
    parse, then fence-strip + ``re.search(r"\\{.*\\}", ..., re.DOTALL)``).
    """
    stripped = text.strip()
    for tok in ("<|im_end|>", "</s>", "<|eot_id|>"):
        if stripped.endswith(tok):
            stripped = stripped[: -len(tok)].strip()

    # Strict: whole text is the JSON (unchanged by 555cae4).
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass

    # Legacy lenient: greedy {.*} over fence-stripped text (REPLACED by 555cae4).
    fenced = re.sub(r"```(?:json)?\s*", "", stripped)
    fence_cleaned = fenced.replace("```", "")
    match = re.search(r"\{.*\}", fence_cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj, False
        except json.JSONDecodeError:
            pass
    return None, False


# A deterministic corpus of ROUTINE realistic model outputs (NOT adversarial
# corner cases — these are the everyday ways a model emits a typed JSON record:
# clean JSON, JSON + an explanatory note, fenced code blocks, a nested
# sub-object, a brace inside a string value, an array wrapper, two records
# concatenated, and a trailing JSON-ish reference). Each entry's label names the
# routine pattern so the measured delta is auditable per-pattern.
ROUTINE_CORPUS: list[tuple[str, str]] = [
    ("clean_json", _MEETING_JSON),
    ("json_plus_note_with_brace", _MEETING_JSON + " Note: agenda is in {docs}."),
    ("fenced_json", "```json\n" + _MEETING_JSON + "\n```"),
    ("fenced_json_plus_note",
     "```json\n" + _MEETING_JSON + "\n```\nExplanation: duration is in {minutes}."),
    ("nested_object", json.dumps({**_MEETING, "meta": {"calendar": "work"}})),
    ("brace_in_string_value",
     json.dumps({**_MEETING, "location": "Bldg {A}"})),
    ("array_wrapped", "[" + _MEETING_JSON + "]"),
    ("two_records_concatenated", _MEETING_JSON + " " + _PERSON_JSON),
    ("json_plus_trailing_jsonish_ref",
     _MEETING_JSON + ' Reference: {"id": "mtg-7"}.'),
]


def _valid(extractor: Extractor, text: str) -> bool:
    obj, _ = extractor(text)
    return obj is not None


def measure_fidelity_delta(
    corpus: list[tuple[str, str]] = ROUTINE_CORPUS,
    *,
    candidate: Extractor = extract_json,
    baseline: Extractor = _legacy_greedy_extract,
) -> dict:
    """Run ``candidate`` vs ``baseline`` over ``corpus``; return the measured delta.

    ``candidate`` defaults to the CURRENT balanced-brace ``extract_json``;
    ``baseline`` defaults to the pre-555cae4 greedy extractor. Returns a dict
    with per-extractor valid counts, the list of rescued labels (valid under
    candidate, invalid under baseline), and any regressed labels (the fix must
    not lose a case the greedy path handled — ``regressed`` must be empty).

    The pinned ``rescued_count`` is the number feedback bullet 2 asks to record:
    how many routine outputs the fix moved from ``valid=0`` to ``valid=1``.
    """
    per_entry = []
    rescued: list[str] = []
    regressed: list[str] = []
    cand_valid_n = 0
    base_valid_n = 0
    for label, text in corpus:
        c_valid = _valid(candidate, text)
        b_valid = _valid(baseline, text)
        cand_valid_n += int(c_valid)
        base_valid_n += int(b_valid)
        if c_valid and not b_valid:
            rescued.append(label)
        elif b_valid and not c_valid:
            regressed.append(label)
        per_entry.append({"label": label, "candidate_valid": c_valid,
                          "baseline_valid": b_valid})
    return {
        "total": len(corpus),
        "candidate_valid": cand_valid_n,
        "baseline_valid": base_valid_n,
        "rescued_count": len(rescued),
        "rescued": rescued,
        "regressed_count": len(regressed),
        "regressed": regressed,
        "per_entry": per_entry,
    }


def _format_report(result: dict) -> str:
    rescued_lbl = ", ".join(result["rescued"]) if result["rescued"] else "(none)"
    regressed_lbl = ", ".join(result["regressed"]) if result["regressed"] else "(none)"
    lines = [
        "JSON-Extraction Fidelity Delta (555cae4 balanced-brace vs pre-fix greedy)",
        "=" * 72,
        f"corpus size               : {result['total']} routine model outputs",
        f"candidate (balanced) valid: {result['candidate_valid']}/{result['total']}",
        f"baseline (greedy) valid   : {result['baseline_valid']}/{result['total']}",
        f"RESCUED (greedy=0->bal=1) : {result['rescued_count']}  [{rescued_lbl}]",
        f"regressed (bal lost)      : {result['regressed_count']}  [{regressed_lbl}]",
        "",
        "per-pattern:",
    ]
    for e in result["per_entry"]:
        lines.append(
            f"  {e['label']:30s} candidate={int(e['candidate_valid'])} "
            f"baseline(greedy)={int(e['baseline_valid'])}"
        )
    lines.append("")
    lines.append("")
    lines.append(
        "Under the greedy extractor every rescued output scored valid=0 (ALL"
    )
    lines.append(
        "headline metrics zeroed); the balanced scan recovers them."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path to write the text report")
    args = ap.parse_args()

    result = measure_fidelity_delta()
    report = _format_report(result)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
