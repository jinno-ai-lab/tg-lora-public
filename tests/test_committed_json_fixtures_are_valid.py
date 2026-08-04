"""Floor guard: every committed load-bearing JSON / JSONL artifact parses as
valid JSON through the SAME strict ``json.loads`` a judge, operator, or
downstream consumer uses — the parser whose ``Expecting property name enclosed in
double quotes`` (and its siblings: trailing comma, single-quoted/unquoted key,
truncated document, trailing garbage) the ``judge_invalid_json`` rejection
surfaces.

Two scopes, one shared parse check
----------------------------------
1. ``tests/fixtures/`` — discovered recursively. The citable full-budget §4-verdict
   deposits, decision snapshots, ledgers, run-logs: the load-bearing emitted
   artifacts a judge/operator reads directly (the original ``a8b212c`` floor).
2. A curated set of load-bearing committed artifacts *outside* ``tests/fixtures/``
   that a consumer reads but NO other test validity-parses — the fixtures-only
   floor's scope blind spot. Headline: ``section4_landed_decision.json`` (the
   LANDED §4 SHIP verdict a judge reads directly; referenced only as a *write*
   target in ``section4_operator_decision.py``, parsed by no test), plus the
   downstream-eval datasets under ``data/downstream/`` that
   ``scripts/eval_downstream.py`` reads line-by-line and
   ``test_eval_downstream.py`` checks for *existence* only.

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
at all. The same was true of ``section4_landed_decision.json`` at the repo root
until this scope was added. This floor makes "every load-bearing committed
artifact is valid JSON" structural: it names the exact file (and, for JSONL, the
exact line) on failure, independent of any one reader.

Scope is deliberately NARROW — parse validity only. Dict-after-parse access is the
separate ``test_json_loads_dict_guard`` axis; replay / label-staleness invariants
stay in their own suite. This guard adds the one missing layer: the bytes a
consumer hands to ``json.loads`` must parse.

``baselines/velocity_ops.json`` is INTENTIONALLY EXCLUDED from scope #2: it
already carries a richer validity gate in
``test_benchmark_velocity_ops.py::test_baseline_file_valid_json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _assert_artifact_parses_strictly(rel: str, path: Path) -> None:
    """Assert a committed JSON/JSONL artifact parses through strict ``json.loads``.

    Single source of truth for the parse-validity check both the fixtures scope
    and the repo-root load-bearing-artifact scope use. ``rel`` is the path string
    embedded in failure messages (relative to the fixtures dir or the repo root,
    as appropriate) so CI points at the corrupt artifact, not at reader logic.
    """
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".jsonl":
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


def _discover_fixtures() -> list[Path]:
    """Every ``*.json`` / ``*.jsonl`` under ``tests/fixtures/`` (recursive)."""
    return sorted(
        p for pat in ("*.json", "*.jsonl") for p in _FIXTURES_DIR.rglob(pat) if p.is_file()
    )


_FIXTURES = _discover_fixtures()

# Load-bearing committed JSON/JSONL OUTSIDE ``tests/fixtures/`` that a consumer
# (script / judge / operator) reads but NO other test validity-parses. Each is a
# documented blind spot of the fixtures-only floor: a corrupt byte here (trailing
# comma from a hand-edit, a merge-conflict marker, a truncated harvest) would
# ship GREEN and surface later as ``judge_invalid_json`` against an
# already-merged artifact. Repo-root-relative posix paths, resolved against
# ``_REPO_ROOT`` so the test is robust to the CWD pytest is invoked from.
_REPO_LOAD_BEARING_ARTIFACTS = [
    "section4_landed_decision.json",  # LANDED §4 SHIP verdict; judge reads directly; parsed by no test
    "data/downstream/format_json.jsonl",  # eval dataset; read by scripts/eval_downstream.py
    "data/downstream/jp_capability.jsonl",  # eval dataset; read by scripts/eval_downstream.py
]

_REPO_ARTIFACTS = [_REPO_ROOT / rel for rel in _REPO_LOAD_BEARING_ARTIFACTS]


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
    _assert_artifact_parses_strictly(rel, fixture)


def test_repo_load_bearing_artifact_inventory_is_complete() -> None:
    # Anti-vacuous-pass guard for the curated scope #2 list (the same
    # "assert any_checked" discipline as the fixtures inventory above). A missing
    # file here means a load-bearing artifact was deleted or moved without
    # updating the list — fail loud with the exact path(s), don't silently pass
    # the parametrized cases on a broken set or collapse them to zero.
    missing = [
        rel for rel, path in zip(_REPO_LOAD_BEARING_ARTIFACTS, _REPO_ARTIFACTS) if not path.exists()
    ]
    assert _REPO_LOAD_BEARING_ARTIFACTS, (
        "the curated repo-artifact list is empty — the parse floor would pass "
        "vacuously; re-add the load-bearing committed JSON/JSONL outside tests/fixtures/"
    )
    assert not missing, (
        f"load-bearing committed artifact(s) missing from repo root: {missing} — "
        f"either restore them or update _REPO_LOAD_BEARING_ARTIFACTS"
    )


@pytest.mark.parametrize(
    "artifact",
    _REPO_ARTIFACTS,
    ids=_REPO_LOAD_BEARING_ARTIFACTS,
)
def test_committed_repo_artifact_parses_as_valid_json(artifact: Path) -> None:
    rel = artifact.relative_to(_REPO_ROOT).as_posix()
    _assert_artifact_parses_strictly(rel, artifact)
