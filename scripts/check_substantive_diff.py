#!/usr/bin/env python3
"""Commit-hygiene guard: flag empty / normalization-only commits.

AI-Hub make-run feedback (TASK-0192): a commit whose diff is *empty* or
*normalization-only* (whitespace / blank-line / line-ending reflow only -- zero
substantive content delta) delivers no value and only churns the commit log.
This guard classifies a diff range so such commits can be **skipped or squashed
before they are judged**, instead of being sent to the value judge as if they
were real work.

Verdict logic mirrors git's own definition of a whitespace-only change:

  * ``empty``              -- the raw diff has no content lines (nothing staged /
                             no delta across the range).
  * ``normalization-only`` -- the raw diff has content lines, but
                             ``git diff <range> -w --ignore-blank-lines`` has
                             none -- every change was whitespace / blank-line /
                             reflow.
  * ``substantive``        -- the whitespace-ignoring diff still carries content
                             lines (a real edit).

The default range is the last commit (``HEAD^..HEAD``) -- the post-commit check
the value judge would run over "the commit I just made". Pass ``--range A..B``
for an arbitrary pair, or ``--staged`` for a pre-commit check of the index.

Exit codes are distinct so a caller can choose skip (empty) vs squash
(normalization-only) vs accept (substantive)::

    0  substantive
    3  empty
    4  normalization-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# Distinct non-zero codes (1 is reserved for unexpected errors; 2 for argparse).
EXIT_SUBSTANTIVE = 0
EXIT_EMPTY = 3
EXIT_NORMALIZATION_ONLY = 4


def content_line_counts(diff_text: str) -> tuple[int, int]:
    """Count added/removed *content* lines in a unified diff.

    File headers (``+++``/``---``), hunk headers (``@@``) and the ``diff`` /
    ``index`` meta-lines are excluded, so a diff that merely touches many files
    without a real edit still reads as zero content lines. ``\\ No newline`` and
    context lines are also ignored.
    """
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        # Skip unified-diff structural lines (file headers, hunk headers, the
        # `diff --git`/`index`/`new file`/`deleted file`/`similarity` banners,
        # and the `\ No newline at end of file` marker).
        if (
            line.startswith("+++")
            or line.startswith("---")
            or line.startswith("@@")
            or line.startswith("diff ")
            or line.startswith("index ")
            or line.startswith("new file")
            or line.startswith("deleted file")
            or line.startswith("similarity ")
            or line.startswith("rename ")
            or line.startswith("copy ")
            or line.startswith("old mode")
            or line.startswith("new mode")
            or line.startswith("\\")
        ):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def substantive_verdict(raw_diff: str, ws_ignored_diff: str) -> str:
    """Classify a diff as ``empty`` | ``normalization-only`` | ``substantive``.

    Pure function over the two diff texts (raw and whitespace-ignored); the CLI
    wrapper obtains them from git. ``ws_ignored_diff`` should be the output of
    ``git diff <range> -w --ignore-blank-lines``: if it carries no content lines
    while the raw diff does, every change was whitespace / blank-line / reflow.
    """
    raw_added, raw_removed = content_line_counts(raw_diff)
    if raw_added == 0 and raw_removed == 0:
        return "empty"
    ws_added, ws_removed = content_line_counts(ws_ignored_diff)
    if ws_added == 0 and ws_removed == 0:
        return "normalization-only"
    return "substantive"


def _run_git_diff(revspec: str, *, ignore_ws: bool) -> str:
    """Run ``git diff <revspec>`` (optionally whitespace-ignoring), return stdout.

    ``git diff`` exits 0 when there is no diff and 1 when there is one; both are
    success here. ``revspec`` is either a ``A..B`` range or the literal
    ``--cached`` for the staged index.
    """
    args = ["git", "diff", revspec]
    if ignore_ws:
        args += ["-w", "--ignore-blank-lines"]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"git diff {revspec!r} failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--range",
        default="HEAD^..HEAD",
        help="git revspec to diff (default: HEAD^..HEAD -- the last commit)",
    )
    group.add_argument(
        "--staged",
        action="store_true",
        help="check the staged index instead of a revspec (pre-commit use)",
    )
    args = parser.parse_args(argv)

    revspec = "--cached" if args.staged else args.range
    raw = _run_git_diff(revspec, ignore_ws=False)
    ws_ignored = _run_git_diff(revspec, ignore_ws=True)
    verdict = substantive_verdict(raw, ws_ignored)

    if verdict == "substantive":
        print(f"substantive: {revspec} has a real content delta")
        return EXIT_SUBSTANTIVE
    print(
        f"{verdict}: {revspec} has NO substantive content delta -- skip or squash "
        f"before judging",
        file=sys.stderr,
    )
    return EXIT_EMPTY if verdict == "empty" else EXIT_NORMALIZATION_ONLY


if __name__ == "__main__":
    raise SystemExit(main())
