"""CI-enforced structural pin of the doc-spine untrack fix (commit a431945).

CONTEXT
-------
AI-Hub make-run feedback (this cycle, bullets 2-4) describes the
``specs/_doc_spine.yml`` oscillation: a "normalize spine manifest" step strips
the manifest's ``references:`` block to ``[]`` while make-run regenerates the
full 418-line form — 27 commits of pure add/remove churn with zero behavioral
change. Commit a431945 fixed it *upstream* (the feedback's own point 3:
"stop emitting normalize commits entirely") by untracking + gitignoring the
generated manifest: the canonical doc structure lives in the tracked ``specs/``
markdown source (validated by ``scripts/check_spine_anchors.py``, which does
NOT read this manifest), and make-run still regenerates it on disk for runtime
spine tooling — it is simply no longer committed.

The feedback's referenced commits (``f2773d6``, ``dfd78bc``, ``e1fcacd``,
``0f83b0e``) are NOT in this public mirror's history (they name the private
repo's timeline, where the same kernel lingered longer). The equivalent guard
concept here is
``tests/test_substantive_diff.py::test_guard_range_flags_cross_commit_oscillation``,
whose own docstring documents WHY a per-commit guard cannot catch this churn
(each commit reads ``substantive``; only a ``--range`` spanning both would).
The robust fix is therefore to make the file UN-committable, not to detect the
oscillation after the fact.

GAP THIS MODULE CLOSES
----------------------
a431945 relied on a single ``.gitignore`` policy line. Nothing structurally
prevents a regression — a dropped ``.gitignore`` entry, a careless
``git add -f specs/_doc_spine.yml``, or an automation change — from
re-tracking the manifest and resurrecting the 27-commit oscillation. This
module turns the untrack into a CI-enforced invariant:

  * the manifest is NOT in the tracked set (``git ls-files``),
  * the ``.gitignore`` entry that protects it is present, AND
  * git reports the path as actively ignored (``git check-ignore``).

Mutation-proven: ``git add -f specs/_doc_spine.yml`` reds the first assertion;
deleting the ``.gitignore`` line reds the second and third.

LEGITIMATE ESCAPE VALVE
-----------------------
A genuine future decision to re-track the manifest — e.g. the upstream
``~/instructions/spine.py`` generator is patched to emit a stable canonical
form AND make-run stops regenerating it — would remove the ``.gitignore``
entry and this guard together. That is a visible, reviewed act, the opposite
of silent re-oscillation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "specs/_doc_spine.yml"


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


def test_doc_spine_manifest_is_not_tracked() -> None:
    """The generated manifest must stay OUT of the tracked set.

    This is the load-bearing half of the a431945 fix: while the file is
    untracked the normalize<->make-run oscillation is *structurally
    impossible* — neither step can commit it. A regression that re-tracks it
    (explicit ``git add -f``, or a dropped ``.gitignore`` entry followed by
    ``git add -A``) resurrects the 27-commit churn and must fail CI here.
    """
    tracked = _git("ls-files").splitlines()
    assert MANIFEST not in tracked, (
        f"{MANIFEST} is tracked again — the a431945 untrack fix regressed. "
        "This generated manifest oscillated across 27 commits; untrack it "
        "(git rm --cached) and keep it gitignored, do not re-commit."
    )


def test_doc_spine_gitignore_entry_present() -> None:
    """The ``.gitignore`` policy line that protects the manifest must be present.

    Pins the policy half of the fix so a dropped entry is caught loudly BEFORE
    a subsequent ``git add -A`` silently re-tracks the regenerated manifest.
    """
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert MANIFEST in gitignore, (
        f"{MANIFEST} is no longer in .gitignore — the protection that keeps "
        "make-run's `git add -A` from re-tracking the generated manifest is "
        "gone. Restore the entry (see the 'generated reverse-index' comment)."
    )


def test_doc_spine_manifest_is_ignored() -> None:
    """git must report the manifest path as actively ignored.

    ``check-ignore`` exits 0 iff a path matches an ignore rule — independent of
    whether the file currently exists on disk. A non-zero exit here means the
    manifest could be re-added by ``git add -A`` (e.g. the ``.gitignore`` line
    was renamed, malformed, or cancelled by a ``!`` negation pattern), so this
    catches protection gaps the string-match assertion above cannot.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", MANIFEST],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"{MANIFEST} is not git-ignored — `git add -A` would re-track it. "
        "Restore the .gitignore entry that matches this path."
    )
