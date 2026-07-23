"""Tests for the commit-hygiene guard (scripts/check_substantive_diff.py).

Covers two layers:
  * the pure verdict function over synthetic unified diffs (fast, no git), and
  * end-to-end git integration in a throwaway repo -- the exact churn the AI-Hub
    feedback named (an empty / whitespace-only commit) must be flagged, a real
    edit must pass.

The pure-function tests are mutation-killable: neutralising the whitespace-ignored
branch or the content-line counter REDs the corresponding assertion.
"""

import subprocess
import sys
from pathlib import Path

from scripts.check_substantive_diff import (
    EXIT_EMPTY,
    EXIT_NORMALIZATION_ONLY,
    EXIT_SUBSTANTIVE,
    content_line_counts,
    substantive_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_substantive_diff.py"

# A minimal real-looking unified diff hunk so the counter exercises its header
# exclusions, not just the +/- lines.
_HUNK = "--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n"


# --------------------------------------------------------------------------- #
# Pure-function unit tests (no git)                                           #
# --------------------------------------------------------------------------- #


def test_content_line_counts_excludes_structural_headers():
    diff = (
        "diff --git a/note.md b/note.md\n"
        "index 1234567..7654321 100644\n"
        "--- a/note.md\n"
        "+++ b/note.md\n"
        "@@ -1,2 +1,2 @@\n"
        " context line\n"
        "-old line\n"
        "+new line\n"
        "\\ No newline at end of file\n"
    )
    added, removed = content_line_counts(diff)
    assert (added, removed) == (1, 1)


def test_verdict_empty_when_no_content_lines():
    assert substantive_verdict("", "") == "empty"
    # Structural-only diff (headers, no +/- content) is also empty.
    structural_only = (
        "diff --git a/f b/f\nindex 1..2\n--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n"
    )
    assert substantive_verdict(structural_only, structural_only) == "empty"


def test_verdict_normalization_only_when_ws_ignored_has_no_content():
    # Raw carries a content line, but the -w diff carries none -> whitespace-only.
    raw = _HUNK + "-hello world\n+hello   world\n"
    assert substantive_verdict(raw, "") == "normalization-only"


def test_verdict_substantive_when_ws_ignored_keeps_content():
    raw = _HUNK + "-old\n+new\n"
    ws = _HUNK + "-old\n+new\n"
    assert substantive_verdict(raw, ws) == "substantive"


def test_verdict_pure_addition_of_real_line_is_substantive():
    raw = _HUNK + " keep\n+added real content\n"
    ws = _HUNK + " keep\n+added real content\n"
    assert substantive_verdict(raw, ws) == "substantive"


# --------------------------------------------------------------------------- #
# End-to-end git integration (the feedback's actual churn scenario)           #
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _new_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "tester")


def _run_guard(repo: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the guard with cwd=repo (so its internal `git diff` targets the repo)."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )


def test_guard_flags_normalization_only_commit(tmp_path):
    """A commit whose only change is whitespace/blank-line reflow is flagged --
    the exact 'net diff is zero' churn the feedback complained about."""
    repo = tmp_path
    _new_repo(repo)
    note = repo / "note.md"
    note.write_text("hello world\nfoo bar\n")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-q", "-m", "baseline")
    # Normalization-only: same words, reflowed spacing + a blank line.
    note.write_text("hello   world\n\nfoo   bar\n")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-q", "-m", "reflow")

    proc = _run_guard(repo)
    assert proc.returncode == EXIT_NORMALIZATION_ONLY, proc.stderr


def test_guard_passes_substantive_commit(tmp_path):
    repo = tmp_path
    _new_repo(repo)
    note = repo / "note.md"
    note.write_text("hello world\n")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-q", "-m", "baseline")
    note.write_text("hello world\nNEW CONTENT LINE\n")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-q", "-m", "add real content")

    proc = _run_guard(repo)
    assert proc.returncode == EXIT_SUBSTANTIVE, proc.stderr


def test_guard_flags_empty_staged_check(tmp_path):
    """Pre-commit use (--staged): with nothing staged, the commit would be empty."""
    repo = tmp_path
    _new_repo(repo)
    note = repo / "note.md"
    note.write_text("hello\n")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-q", "-m", "baseline")
    # Nothing staged now.
    proc = _run_guard(repo, "--staged")
    assert proc.returncode == EXIT_EMPTY, proc.stderr


def test_guard_flags_staged_normalization_only(tmp_path):
    repo = tmp_path
    _new_repo(repo)
    note = repo / "note.md"
    note.write_text("hello world\n")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-q", "-m", "baseline")
    note.write_text("hello   world\n")  # whitespace-only change, staged not committed
    _git(repo, "add", "note.md")

    proc = _run_guard(repo, "--staged")
    assert proc.returncode == EXIT_NORMALIZATION_ONLY, proc.stderr


def test_main_returns_substantive_for_this_repos_last_commit():
    """The guard must not false-positive on a normal substantive commit: this
    repo's own HEAD (a real fix) must read substantive. A regression here would
    mean the guard rejects legitimate work."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--range", "HEAD^..HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == EXIT_SUBSTANTIVE, proc.stderr
