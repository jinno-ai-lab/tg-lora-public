"""Floor guard: every committed JSON / JSONL fixture under ``tests/fixtures/``
parses as valid JSON through the SAME strict ``json.loads`` a judge, operator, or
downstream consumer uses — the parser whose ``Expecting property name enclosed in
double quotes`` (and its siblings: trailing comma, single-quoted/unquoted key,
truncated document, trailing garbage) the ``judge_invalid_json`` rejection
surfaces.

Why this is a *separate* guard, not "already covered"
-----------------------------------------------------
The §4-verdict deposit fixtures are this repo's most load-bearing emitted
artifacts — citable full-budget verdicts a judge/operator reads directly. Yet
*parse validity* was never a first-class invariant. The only fixture-walk that
parsed them (``_deposits_with_label`` in ``test_replay_freeze_validloss_ci.py``)
did so behind an ``except Exception: continue`` that SILENTLY dropped any file
that failed to parse, so a corrupt committed deposit (trailing comma,
single-quoted key, a hand-edit, a merge-conflict marker, a truncated harvest) was
excluded from its own staleness check and shipped GREEN — surfacing only later as
a downstream ``judge_invalid_json`` against an *already-merged* artifact. Several
fixtures (``section4_decision_snapshot_*``, ``prefix_break_even_canonical_summary``,
``probe_9b_memory_frontier``, the ``advise_loop`` JSONL) were parsed by no walk
at all. This floor makes "every committed artifact is valid JSON" structural: it
names the exact file (and, for JSONL, the exact line) on failure, independent of
any one reader.

Scope is deliberately NARROW — parse validity only. Dict-after-parse access is the
separate ``test_json_loads_dict_guard`` axis; replay / label-staleness invariants
stay in their own suite. This guard adds the one missing layer: the bytes a
consumer hands to ``json.loads`` must parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _discover_fixtures() -> list[Path]:
    """Every ``*.json`` / ``*.jsonl`` under ``tests/fixtures/`` (recursive)."""
    return sorted(
        p for pat in ("*.json", "*.jsonl") for p in _FIXTURES_DIR.rglob(pat) if p.is_file()
    )


_FIXTURES = _discover_fixtures()


def test_fixture_inventory_is_nonempty() -> None:
    # Guard against a typo in the glob / a relocated fixtures dir that would make
    # the parametrized cases silently collapse to zero — the "assert any_checked"
    # pattern the repo uses elsewhere. If fixtures ever move, this fails loud
    # instead of the parse floor passing vacuously on an empty set.
    assert _FIXTURES, (
        f"no *.json/*.jsonl found under {_FIXTURES_DIR} — the parse floor would "
        f"pass vacuously; update the discovery path"
    )


@pytest.mark.parametrize(
    "fixture",
    _FIXTURES,
    ids=[p.relative_to(_FIXTURES_DIR).as_posix() for p in _FIXTURES],
)
def test_committed_fixture_parses_as_valid_json(fixture: Path) -> None:
    rel = fixture.relative_to(_FIXTURES_DIR).as_posix()
    text = fixture.read_text(encoding="utf-8")

    if fixture.suffix == ".jsonl":
        # Newline-delimited: every non-blank line is its own JSON document (a
        # trailing blank line is normal for the producer's append; only a
        # non-blank line that fails to parse is corruption). Report every bad
        # line at once so a multi-line corruption isn't fixed one report at a time.
        bad: list[str] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                bad.append(f"line {lineno}: {exc}")
        assert not bad, (
            f"{rel} is corrupt JSONL — {len(bad)} unparseable line(s), a "
            f"consumer's json.loads would reject (the judge_invalid_json mode):\n  "
            + "\n  ".join(bad)
        )
        return

    # Single-document JSON: json.loads rejects trailing commas, single-quoted /
    # unquoted keys, trailing garbage, and truncation — the exact judge_invalid_json
    # failure modes — raising JSONDecodeError with the cause.
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"{rel} is corrupt JSON — a judge/operator json.loads would reject it "
            f"(the judge_invalid_json failure mode): {exc}"
        )
