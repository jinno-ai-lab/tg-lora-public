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
# Binary-file change (no +/- text body) must read substantive, not empty       #
# --------------------------------------------------------------------------- #
# Regression guard for the prepare-commit-msg hook: git emits
# `Binary files ... differ` with NO +/- content lines, so the line counter
# formerly read (0, 0) -> verdict "empty" -> exit 3 -> the hook REJECTED a
# legitimate binary commit (e.g. a committed .pt/.safetensors fixture). A binary
# delta is substantive by definition.


def test_content_line_counts_counts_binary_marker_as_content():
    # git's exact format for a newly-added binary file.
    binary_add = (
        "diff --git a/weights.pt b/weights.pt\n"
        "new file mode 100644\n"
        "index 0000000..5bcf3dd\n"
        "Binary files /dev/null and b/weights.pt differ\n"
    )
    assert content_line_counts(binary_add) == (1, 0)


def test_verdict_substantive_for_binary_only_diff():
    # Both the raw and the -w diff carry the binary marker (whitespace flags do
    # not strip it), so the verdict must be substantive, not empty.
    binary = "Binary files a/weights.pt and b/weights.pt differ\n"
    assert substantive_verdict(binary, binary) == "substantive"
    # And an add via /dev/null is substantive too.
    binary_add = "Binary files /dev/null and b/weights.pt differ\n"
    assert substantive_verdict(binary_add, binary_add) == "substantive"


def test_verdict_substantive_when_binary_accompanies_text_reflow():
    # A commit that reflows whitespace in a text file AND changes a binary file
    # is substantive: the binary marker survives in the -w diff, so the verdict
    # must not collapse to normalization-only on the text reflow.
    raw = _HUNK + "-hello world\n+hello   world\n" + "Binary files a/x and b/x differ\n"
    ws = "Binary files a/x and b/x differ\n"
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


def test_guard_passes_staged_binary_add(tmp_path):
    """Pre-commit use (--staged): a newly staged BINARY file (no +/- text body,
    git emits `Binary files ... differ`) is substantive -- it must NOT be
    rejected as empty. Without the binary-marker branch this returned EXIT_EMPTY
    (reproduced), which would make the prepare-commit-msg hook block a legitimate
    fixture / artifact commit."""
    repo = tmp_path
    _new_repo(repo)
    note = repo / "note.md"
    note.write_text("hello\n")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-q", "-m", "baseline")
    # Real non-UTF8 bytes so git classifies the file as binary.
    (repo / "weights.pt").write_bytes(b"\x00\x01\x02\xff\xfe-substantive")
    _git(repo, "add", "weights.pt")

    proc = _run_guard(repo, "--staged")
    assert proc.returncode == EXIT_SUBSTANTIVE, proc.stderr


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


# --------------------------------------------------------------------------- #
# Cross-commit oscillation (range-mode catches net-zero across a range)       #
# --------------------------------------------------------------------------- #
# AI-Hub make-run feedback item #3: commit A "normalizes" (collapses a file,
# real line removals) and commit B "make-run" re-expands it byte-for-byte to
# the original tree. Across the range START..B the net diff is ZERO -- `git
# diff --quiet START B` exits 0 -- yet NEITHER commit is empty /
# normalization-only *individually*: A removes real lines, B adds them back.
# So the make-run loop's per-commit view (`HEAD^..HEAD` on each) reads both as
# `substantive` and the oscillation slips through unflagged.
#
# The guard's `--range` mode is the correct tool for cross-commit churn: given
# the revspec that SPANS the oscillation it compares START..END trees, finds the
# net delta empty, and returns `empty`. This regression test pins that property
# so a refactor of range-mode handling (or the empty-verdict branch) cannot
# silently regress the ONE detection path the make-run loop must adopt to catch
# this churn (by invoking the checker `--range <start>..<end>` rather than
# per-commit). It does NOT add a new guard -- it pins an existing, named,
# previously-untested branch. Empirically verified against a throwaway repo
# before this test was written.


def test_guard_range_flags_cross_commit_oscillation(tmp_path):
    """Commit A collapses a file, commit B re-expands it to the byte-identical
    original tree -- the feedback's exact `git diff --quiet` net-zero churn.

    Per-commit (`HEAD^..HEAD` on B) the guard reads `substantive`: B's
    re-expansion is a real addition, so the make-run loop's per-commit view
    does NOT flag the oscillation. But `--range START..B` (spanning both
    commits) compares identical trees and returns `empty` -- the recipe the
    make-run loop must adopt to catch cross-commit churn once invoked
    per-range."""
    repo = tmp_path
    _new_repo(repo)
    spine = repo / "spine.yml"
    spine.write_text("a: 1\nb: 2\nc: 3\n")
    _git(repo, "add", "spine.yml")
    _git(repo, "commit", "-q", "-m", "baseline (expanded tree)")
    start = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Commit A: collapse (remove real lines). Substantive when checked alone.
    spine.write_text("a: 1\n")
    _git(repo, "add", "spine.yml")
    _git(repo, "commit", "-q", "-m", "A: normalize/collapse (-2 lines)")

    # Commit B: re-expand to the BYTE-IDENTICAL baseline tree. Substantive
    # alone (a real addition), but net-zero across start..B.
    spine.write_text("a: 1\nb: 2\nc: 3\n")
    _git(repo, "add", "spine.yml")
    _git(repo, "commit", "-q", "-m", "B: re-expand to original")
    b = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Per-commit view (the make-run loop's current behaviour): B reads
    # substantive, so the oscillation is NOT flagged commit-by-commit.
    proc_per_commit = _run_guard(repo, "--range", "HEAD^..HEAD")
    assert proc_per_commit.returncode == EXIT_SUBSTANTIVE, proc_per_commit.stderr

    # Range view spanning the oscillation: identical START..B trees -> the net
    # diff is empty -> flagged. This is the regression this test pins.
    proc_range = _run_guard(repo, "--range", f"{start}..{b}")
    assert proc_range.returncode == EXIT_EMPTY, proc_range.stderr
