"""CI-enforced structural pin of the .aihub-session.json untrack fix.

CONTEXT
-------
The AI-Hub make-run runner drops a transient session artifact (``.aihub-session.json``,
plus ``*.aihub-session*`` siblings) into the repo root: a copy of the LLM invocation
prompt with embedded repo docs. It is runtime harness metadata, not source. The
2026-08-18 value-gate rejection was caused by make-run's own commit sweep staging it
(``chore(make-run): commit 1 remaining change(s)``) — a no-behavior-change commit of
a leftover session file.

The fix (re-landed per fresh sibling; first landed as 8b1f88c on the 20260818-044025
sibling) is structural, mirroring the a431945 doc-spine precedent
(``tests/test_doc_spine_untracked.py``): untrack the artifact (``git rm --cached``,
working copy stays for the runner) and gitignore the whole family so ``git add -A``
can never stage it again — rather than relying on per-commit cleanup to catch it.

The ``/*session*.json`` pattern is ROOT-ANCHORED on purpose: nested paths such as
``tests/fixtures/*session*.json`` stay committable (only runner droppings in the
repo root are suppressed).

GAP THIS MODULE CLOSES
----------------------
The gitignore policy alone regresses silently — a dropped entry, or a renamed runner
artifact outside the patterns, re-opens the exact rejection. This module turns the
hygiene into a CI invariant:

  * the artifact is NOT in the tracked set (``git ls-files``),
  * the three ``.gitignore`` family entries are present, AND
  * git reports the artifact AND a family variant as actively ignored
    (``git check-ignore``).

Mutation-proven RED pre-fix: on the rejected HEAD (0a61c8e) the artifact is tracked,
so the first assertion fails before the fix is applied.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_ARTIFACT = ".aihub-session.json"
FAMILY_VARIANT = ".aihub-session-cycle0002.json"  # sibling the runner may emit next
GITIGNORE_ENTRIES = (SESSION_ARTIFACT, "*.aihub-session*", "/*session*.json")


def _git(*args: str) -> str:
    """Run git in the real repo root; return stdout (stripped).

    Non-zero git here is a test-harness error (not a guard signal) — these
    read-only queries never fail on a healthy checkout.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
    )
    proc.check_returncode()
    return proc.stdout.strip()


def test_aihub_session_artifact_is_not_tracked() -> None:
    """The runner session artifact must stay OUT of the tracked set.

    This is the load-bearing half of the fix: while the file is untracked and
    ignored, committing it is structurally impossible and the value gate can
    never again reject an iteration for harness-metadata churn.
    """
    tracked = _git("ls-files")
    assert SESSION_ARTIFACT not in tracked.splitlines(), (
        f"{SESSION_ARTIFACT} is tracked — a make-run commit sweep staged the "
        "runner's transient session file (the 2026-08-18 rejection cause). "
        "Untrack it (git rm --cached, keep the working copy) and keep the "
        ".gitignore family entries."
    )


def test_gitignore_family_entries_present() -> None:
    """All three .gitignore family entries must be present.

    The exact artifact name alone is not enough: the runner emits dated/cycled
    siblings (``*.aihub-session*``) and may rename the artifact entirely — the
    root-anchored ``/*session*.json`` catches that class while leaving nested
    fixture paths committable.
    """
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [entry for entry in GITIGNORE_ENTRIES if entry not in gitignore]
    assert not missing, (
        f".gitignore is missing the aihub-session family entries: {missing}. "
        "Restore them or a runner artifact rename re-opens the churn."
    )


def test_session_artifact_family_is_ignored() -> None:
    """git must report the artifact AND a not-yet-emitted family variant as ignored.

    ``check-ignore`` exits 0 iff a path matches an ignore rule — independent of
    whether the file currently exists on disk. Checking a variant name the runner
    has NOT emitted yet proves the family pattern (not just the exact filename)
    is live, catching malformed or negated patterns the string-match assertion
    above cannot.
    """
    for path in (SESSION_ARTIFACT, FAMILY_VARIANT):
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", path],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"{path} is not git-ignored — `git add -A` would stage it. Restore "
            "the .aihub-session family entries in .gitignore."
        )


def test_nested_session_named_fixture_stays_committable() -> None:
    """The root anchor must NOT suppress nested legit paths (escape-valve pin).

    ``tests/fixtures/`` holds committed JSON witnesses; if someone over-broadens
    the pattern to an unanchored ``*session*.json``, a legitimate nested fixture
    would become silently uncommittable. This pins the anchor's scope.
    """
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "tests/fixtures/runner_session_witness.json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        "tests/fixtures/runner_session_witness.json is matched by the session "
        "ignore patterns — the root anchor was lost/over-broadened and nested "
        "fixtures are no longer committable."
    )
