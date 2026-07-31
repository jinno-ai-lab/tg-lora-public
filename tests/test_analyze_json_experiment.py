"""Tests for ``scripts/analyze_json_experiment.py`` JSONL loading.

Pins the non-dict-after-json.loads crash-class guard on ``load_json_eval``: a
line that is valid JSON but NOT a dict (a bare scalar / array / string from a
corrupt or hand-edited ``json_eval_log.jsonl``) must be skipped, not crash the
JSON-extraction efficiency analysis. Same crash class the sibling readers already
harden (``compare_runs.load_run`` / ``run_query.parse_jsonl`` / ``run_metrics``
/ the §4 verdict ledger readers); this reader was the one the sweep missed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_json_experiment import backwards_to_target, load_json_eval


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _record(cycle: int, combined: float, backwards: int) -> dict:
    return {"cycle": cycle, "combined": combined, "field_f1": combined,
            "full_backward_passes": backwards}


def test_load_json_eval_returns_empty_when_log_absent(tmp_path: Path):
    """No json_eval_log.jsonl ⇒ empty list (the condition's analysis warns)."""
    assert load_json_eval(tmp_path) == []


def test_load_json_eval_skips_non_dict_lines(tmp_path: Path):
    """A valid-JSON-but-non-dict line (array/scalar from a corrupt or hand-edited
    json_eval_log.jsonl) is skipped, not crashed.

    Without the ``isinstance(rec, dict)`` guard every downstream access raised
    on the first non-dict line — ``r.get(metric)`` / ``last.get('cycle')`` with
    AttributeError, and the plot loop's direct subscript ``r[xkey]`` with
    TypeError — aborting the whole analysis. Mutation-proven: drop the guard and
    this raises on the ``[1, 2, 3]`` line. Same non-dict-after-json.loads crash
    class the sibling readers guard (``compare_runs.load_run`` /
    ``run_query.parse_jsonl`` / ``run_metrics`` / the §4 verdict ledger readers).
    """
    recs = [_record(0, 0.4, 10), _record(1, 0.6, 20), _record(2, 0.8, 30)]
    # Interleave non-dict lines (array, bare scalar) among valid dict records.
    lines = [
        json.dumps(recs[0]),
        "[1, 2, 3]",  # valid JSON, but a list -> r.get / r[xkey] would raise
        json.dumps(recs[1]),
        "42",         # valid JSON, but an int -> r.get would raise
        json.dumps(recs[2]),
    ]
    _write_jsonl(tmp_path / "json_eval_log.jsonl", lines)

    rows = load_json_eval(tmp_path)

    # The two non-dict lines are dropped; the three dict records survive intact.
    assert [r["cycle"] for r in rows] == [0, 1, 2]
    assert all(isinstance(r, dict) for r in rows)


def test_load_json_eval_skips_various_non_dict_shapes(tmp_path: Path):
    """Array, bool, null, and bare-string lines are all valid JSON, non-dict."""
    lines = [
        json.dumps(_record(0, 0.4, 10)),
        "[1, 2, 3]",
        "true",
        "null",
        '"a bare string"',
        json.dumps(_record(1, 0.7, 25)),
    ]
    _write_jsonl(tmp_path / "json_eval_log.jsonl", lines)
    assert [r["cycle"] for r in load_json_eval(tmp_path)] == [0, 1]


def test_load_json_eval_invalid_json_still_raises(tmp_path: Path):
    """Genuine invalid-JSON lines still raise (corruption is surfaced, not
    swallowed) — mirrors the sibling readers' discipline."""
    _write_jsonl(tmp_path / "json_eval_log.jsonl",
                 [json.dumps(_record(0, 0.4, 10)), "not json at all"])
    with pytest.raises(json.JSONDecodeError):
        load_json_eval(tmp_path)


def test_backwards_to_target_consumes_only_surviving_dicts(tmp_path: Path):
    """End-to-end: the analysis helpers see only the dict records that survived
    the guard, so a non-dict line between two valid ones cannot skew the
    backwards-to-target scan."""
    _write_jsonl(
        tmp_path / "json_eval_log.jsonl",
        [json.dumps(_record(0, 0.4, 10)), "null", json.dumps(_record(1, 0.8, 30))],
    )
    rows = load_json_eval(tmp_path)
    assert [r["cycle"] for r in rows] == [0, 1]  # the null line was dropped
    assert backwards_to_target(rows, "combined", 0.8) == 30
