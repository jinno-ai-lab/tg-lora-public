#!/usr/bin/env python
"""Validate the integrity of ``<!-- spine:anchor -->`` blocks in spec/doc markdown.

Every spec and TASK file carries a *spine anchor* block declaring its place in
the documentation spine, e.g.::

    <!-- spine:anchor:begin -->
    > **Spine anchor**: [TG-LoRA アーキテクチャ設計](../architecture.md)
    >
    > - parent: `tg-lora/architecture.md`
    > - status: `canonical_child`
    <!-- spine:anchor:end -->

The block restates the parent document **twice**: once as a human-readable
markdown link (file-relative) and once as a backtick ``parent:`` value (a
logical path: ``tg-lora/<x>`` resolves under ``specs/``; anything else is
repo-root-relative). A *re-point* commit that updates one reference but not the
other — or a parent doc that gets renamed/deleted — silently rots the spine.
Nothing in the repo previously caught this; the rot was only ever found
case-by-case in manual promotion reviews.

This validator encodes the spine-anchor contract and fails loudly on drift:

* **shape** — every ``begin`` has a matching ``end`` and the block carries a
  ``Spine anchor`` link line plus ``parent:`` and ``status:`` fields;
* **link resolves** — the markdown link target, resolved relative to the file,
  points at an existing file;
* **parent resolves** — the ``parent:`` logical path resolves to an existing
  file;
* **consistency** — the link target and the ``parent:`` path resolve to the
  *same* file (the two references agree — the drift the re-point commits
  exposed).

Usage::

    # validate the whole repo (specs/ + docs/)
    python scripts/check_spine_anchors.py

    # validate a subset
    python scripts/check_spine_anchors.py specs/tg-lora/tasks/TASK-0148.md

Exit status is non-zero if any anchor violates the contract, so the validator
can gate in CI / a Makefile target. GOAL §7: surface drift instead of letting
provenance metadata rot silently.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as a standalone CLI (``python scripts/check_spine_anchors.py``):
# a bare script invocation puts ``scripts/`` — not the repo root — on sys.path,
# so make the repo root importable without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

_BEGIN = "spine:anchor:begin"
_END = "spine:anchor:end"
_LINK_RE = re.compile(r"\*\*Spine anchor\*\*:\s*\[[^\]]*\]\(([^)]+)\)")
_FIELD_RE_TEMPLATE = r"{field}:\s*`?([^`\n]+?)`?\s*$"
_PARENT_RE = re.compile(_FIELD_RE_TEMPLATE.format(field="parent"), re.MULTILINE)
_STATUS_RE = re.compile(_FIELD_RE_TEMPLATE.format(field="status"), re.MULTILINE)

# ``tg-lora/<x>`` is the spine's logical namespace for ``specs/tg-lora/<x>``.
_SPINE_NAMESPACE = "tg-lora/"


@dataclass(frozen=True)
class AnchorIssue:
    """A single spine-anchor contract violation."""

    file: Path
    kind: str
    detail: str

    def render(self) -> str:
        # Best-effort repo-relative display; fall back to the raw path when the
        # file lives outside the repo (e.g. a ``--root`` external tree) rather
        # than crashing on ``relative_to``.
        try:
            rel = self.file.relative_to(REPO_ROOT)
        except ValueError:
            rel = self.file
        return f"{rel}: {self.kind}: {self.detail}"


def resolve_parent_path(value: str, root: Path) -> list[Path]:
    """Candidate filesystem paths for a ``parent:`` logical value.

    ``tg-lora/<x>`` resolves under ``specs/``; any other value is treated as
    repo-root-relative. Several candidates are returned so the caller can accept
    the first that exists; this keeps resolution robust to the small historical
    variation in how parents were written.
    """
    value = value.strip()
    candidates: list[Path] = []
    if value.startswith(_SPINE_NAMESPACE):
        candidates.append(root / "specs" / value)
    candidates.append(root / value)
    return candidates


def resolve_link_path(link: str, containing_file: Path) -> Path:
    """Resolve a markdown link target relative to its containing file."""
    return (containing_file.parent / link).resolve()


def iter_anchor_blocks(text: str):
    """Yield ``(begin_pos, block_text)`` for each balanced anchor block.

    Each block spans from a ``begin`` marker up to (but not including) the next
    ``end`` marker. A ``begin`` with no following ``end`` yields a block whose
    text lacks the ``end`` marker so the shape check can flag it.
    """
    begins = [m.start() for m in re.finditer(re.escape(_BEGIN), text)]
    for start in begins:
        end = text.find(_END, start)
        if end == -1:
            yield start, text[start:]
        else:
            yield start, text[start:end]


def validate_file(path: Path, root: Path) -> list[AnchorIssue]:
    """Return all spine-anchor contract violations found in ``path``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    issues: list[AnchorIssue] = []
    n_begin = text.count(_BEGIN)
    n_end = text.count(_END)
    if n_begin != n_end:
        issues.append(
            AnchorIssue(
                path,
                "unbalanced",
                f"{n_begin} begin marker(s) but {n_end} end marker(s)",
            )
        )
        # Still inspect whatever balanced blocks we can so a single typo doesn't
        # mask every other anchor in the file.
    for _start, block in iter_anchor_blocks(text):
        link = _LINK_RE.search(block)
        parent = _PARENT_RE.search(block)
        status = _STATUS_RE.search(block)
        if link is None:
            issues.append(
                AnchorIssue(
                    path, "no_link", "missing '**Spine anchor**: [..](..)' line"
                )
            )
            continue
        if parent is None:
            issues.append(AnchorIssue(path, "no_parent", "missing 'parent:' field"))
            continue
        if status is None:
            issues.append(AnchorIssue(path, "no_status", "missing 'status:' field"))
            continue

        link_target = resolve_link_path(link.group(1), path)
        if not link_target.exists():
            issues.append(
                AnchorIssue(
                    path,
                    "link_missing",
                    f"link target '{link.group(1)}' does not resolve",
                )
            )
            continue
        parent_value = parent.group(1).strip()
        parent_candidates = resolve_parent_path(parent_value, root)
        parent_target = next((p for p in parent_candidates if p.exists()), None)
        if parent_target is None:
            tried = ", ".join(str(c.relative_to(root)) for c in parent_candidates)
            issues.append(
                AnchorIssue(
                    path,
                    "parent_missing",
                    f"parent '{parent_value}' does not resolve (tried {tried})",
                )
            )
            continue
        if link_target != parent_target.resolve():
            issues.append(
                AnchorIssue(
                    path,
                    "link_parent_mismatch",
                    f"link resolves to '{link_target}' but parent resolves to '{parent_target.resolve()}'",
                )
            )
    return issues


def collect_markdown_files(paths: list[Path]) -> list[Path]:
    """Expand the given paths (files and/or directories) into markdown files."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.is_file():
            files.append(p)
    # de-duplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            unique.append(f)
    return unique


def validate_tree(root: Path, paths: list[Path]) -> tuple[list[AnchorIssue], int]:
    """Validate all anchor blocks under ``paths`` rooted at ``root``.

    Returns ``(issues, total_blocks)``.
    """
    issues: list[AnchorIssue] = []
    total = 0
    for f in collect_markdown_files(paths):
        text = f.read_text(encoding="utf-8", errors="replace")
        total += text.count(_BEGIN)
        issues.extend(validate_file(f, root))
    return issues, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate spine-anchor integrity in spec/doc markdown."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="Markdown files/directories to check (default: specs/ and docs/).",
    )
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repository root for resolving logical parent paths (default: repo root).",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress per-issue output."
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    targets = (
        [Path(p) for p in args.paths] if args.paths else [root / "specs", root / "docs"]
    )
    # Re-anchor directory hints that were given relative to cwd onto the repo root
    # when they don't exist as-is (lets callers say "specs" from anywhere).
    targets = [
        t if t.exists() else (root / t.name if (root / t.name).exists() else t)
        for t in targets
    ]

    issues, total = validate_tree(root, targets)
    if issues:
        if not args.quiet:
            for issue in sorted(issues, key=lambda i: (str(i.file), i.kind, i.detail)):
                print(issue.render(), file=sys.stderr)
        print(
            f"FAIL: {len(issues)} spine-anchor issue(s) across {total} block(s).",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {total} spine-anchor block(s) verified, no drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
