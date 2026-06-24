"""Tests for the spine-anchor integrity validator (``scripts/check_spine_anchors.py``).

The spine-anchor contract is the documentation-spine provenance metadata that
every spec/TASK file carries. It restates the parent document twice — as a
markdown link (file-relative) and as a ``parent:`` logical path — and nothing
previously verified the two agreed or even resolved. These tests pin the
contract so a future re-point / rename / deletion that rots an anchor fails CI
instead of rotting silently (GOAL §7: surface drift).

Two layers:

* **unit** — each drift mode (mismatch / dangling parent / dangling link /
  malformed shape / unbalanced) is caught against synthetic ``tmp_path`` trees,
  and a clean tree produces zero issues;
* **integration** — the validator is run as a CLI against the real repo and must
  report the live anchor set as drift-free (the regression guard).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_spine_anchors.py"

# Import the validator's pure functions. The script bootstraps the repo root onto
# sys.path itself, but tests import via the repo-root pythonpath configured in
# pyproject so the import path mirrors the other script tests.
sys.path.insert(0, str(ROOT / "scripts"))
import check_spine_anchors as cs  # noqa: E402

sys.path.pop(0)


def _anchor(link: str, parent: str, status: str = "canonical_child") -> str:
    return (
        "<!-- spine:anchor:begin -->\n"
        f"> **Spine anchor**: [Parent]({link})\n"
        ">\n"
        f"> - parent: `{parent}`\n"
        f"> - status: `{status}`\n"
        "<!-- spine:anchor:end -->\n"
    )


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A minimal self-consistent spine tree rooted at ``tmp_path``."""
    _write(tmp_path / "parent.md", "# parent\n")
    _write(
        tmp_path / "child.md",
        "# child\n\n" + _anchor(link="parent.md", parent="parent.md"),
    )
    return tmp_path


class TestPureResolvers:
    def test_parent_logical_namespace_maps_under_specs(self, tmp_path: Path):
        cands = cs.resolve_parent_path("tg-lora/architecture.md", tmp_path)
        assert (tmp_path / "specs" / "tg-lora" / "architecture.md") in cands

    def test_parent_root_relative_value(self, tmp_path: Path):
        cands = cs.resolve_parent_path("docs/GOAL.md", tmp_path)
        assert (tmp_path / "docs" / "GOAL.md") in cands

    def test_link_resolves_file_relative(self, tmp_path: Path):
        child = tmp_path / "specs" / "child.md"
        link_target = cs.resolve_link_path("../parent.md", child)
        assert link_target == (tmp_path / "parent.md").resolve()


class TestCleanTree:
    def test_clean_anchor_has_no_issues(self, repo: Path):
        issues, total = cs.validate_tree(repo, [repo])
        assert issues == []
        assert total == 1

    def test_clean_tree_with_namespace_parent(self, tmp_path: Path):
        # Mirror the real repo: architecture.md (parent under specs namespace)
        # anchored to docs/GOAL.md, and a TASK anchored to architecture.md.
        _write(tmp_path / "docs" / "GOAL.md", "# goal\n")
        _write(
            tmp_path / "specs" / "tg-lora" / "architecture.md",
            "# arch\n\n" + _anchor(link="../../docs/GOAL.md", parent="docs/GOAL.md"),
        )
        _write(
            tmp_path / "specs" / "tg-lora" / "tasks" / "TASK-0001.md",
            "# task\n\n"
            + _anchor(link="../architecture.md", parent="tg-lora/architecture.md"),
        )
        issues, total = cs.validate_tree(tmp_path, [tmp_path])
        assert issues == [], [i.render() for i in issues]
        assert total == 2


class TestDriftDetection:
    """Each drift mode the validator exists to catch."""

    def test_repoint_mismatch_is_caught(self, tmp_path: Path):
        # The link and the parent value point at *different* files — exactly the
        # state a one-sided re-point commit leaves behind.
        _write(tmp_path / "old_parent.md", "# old\n")
        _write(tmp_path / "new_parent.md", "# new\n")
        _write(
            tmp_path / "child.md",
            _anchor(link="new_parent.md", parent="old_parent.md"),
        )
        issues, _ = cs.validate_tree(tmp_path, [tmp_path])
        kinds = {i.kind for i in issues}
        assert "link_parent_mismatch" in kinds, [i.render() for i in issues]

    def test_dangling_parent_is_caught(self, tmp_path: Path):
        _write(tmp_path / "child.md", _anchor(link="parent.md", parent="ghost.md"))
        # link target must exist so we reach the parent check
        _write(tmp_path / "parent.md", "# p\n")
        issues, _ = cs.validate_tree(tmp_path, [tmp_path])
        kinds = {i.kind for i in issues}
        assert "parent_missing" in kinds

    def test_dangling_link_is_caught(self, tmp_path: Path):
        _write(tmp_path / "child.md", _anchor(link="ghost.md", parent="parent.md"))
        _write(tmp_path / "parent.md", "# p\n")
        issues, _ = cs.validate_tree(tmp_path, [tmp_path])
        kinds = {i.kind for i in issues}
        assert "link_missing" in kinds

    def test_missing_parent_field_is_caught(self, tmp_path: Path):
        body = (
            "<!-- spine:anchor:begin -->\n"
            "> **Spine anchor**: [P](parent.md)\n"
            ">\n"
            "> - status: `canonical_child`\n"
            "<!-- spine:anchor:end -->\n"
        )
        _write(tmp_path / "child.md", body)
        _write(tmp_path / "parent.md", "# p\n")
        issues, _ = cs.validate_tree(tmp_path, [tmp_path])
        assert any(i.kind == "no_parent" for i in issues)

    def test_missing_status_field_is_caught(self, tmp_path: Path):
        body = (
            "<!-- spine:anchor:begin -->\n"
            "> **Spine anchor**: [P](parent.md)\n"
            ">\n"
            "> - parent: `parent.md`\n"
            "<!-- spine:anchor:end -->\n"
        )
        _write(tmp_path / "child.md", body)
        _write(tmp_path / "parent.md", "# p\n")
        issues, _ = cs.validate_tree(tmp_path, [tmp_path])
        assert any(i.kind == "no_status" for i in issues)

    def test_missing_link_line_is_caught(self, tmp_path: Path):
        body = (
            "<!-- spine:anchor:begin -->\n"
            "> - parent: `parent.md`\n"
            "> - status: `canonical_child`\n"
            "<!-- spine:anchor:end -->\n"
        )
        _write(tmp_path / "child.md", body)
        _write(tmp_path / "parent.md", "# p\n")
        issues, _ = cs.validate_tree(tmp_path, [tmp_path])
        assert any(i.kind == "no_link" for i in issues)

    def test_unbalanced_begin_without_end_is_caught(self, tmp_path: Path):
        body = (
            "<!-- spine:anchor:begin -->\n"
            + _anchor("parent.md", "parent.md")
            .replace("<!-- spine:anchor:begin -->\n", "")
            .split("<!-- spine:anchor:end -->")[0]
        )
        _write(tmp_path / "child.md", body)
        _write(tmp_path / "parent.md", "# p\n")
        issues, _ = cs.validate_tree(tmp_path, [tmp_path])
        assert any(i.kind == "unbalanced" for i in issues)


class TestCLIIntegration:
    """Run the validator as a CLI against the real repo — the regression guard."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )

    def test_real_repo_is_drift_free(self):
        r = self._run()
        assert r.returncode == 0, r.stderr
        assert "verified, no drift" in r.stdout
        # Pin that a real, non-trivial anchor set was scanned — not an exact
        # count, which would brittle-fail as the spec set legitimately grows.
        m = re.search(r"(\d+) spine-anchor block\(s\) verified", r.stdout)
        assert m, r.stdout
        assert int(m.group(1)) >= 158

    def test_failure_exit_code_on_drift(self, tmp_path: Path):
        _write(tmp_path / "child.md", _anchor(link="ghost.md", parent="ghost.md"))
        r = self._run(str(tmp_path), "--root", str(tmp_path))
        assert r.returncode == 1
        assert "FAIL" in r.stderr
        assert "link_missing" in r.stderr

    def test_quiet_suppresses_detail(self, tmp_path: Path):
        _write(tmp_path / "child.md", _anchor(link="ghost.md", parent="ghost.md"))
        r = self._run(str(tmp_path), "--root", str(tmp_path), "--quiet")
        assert r.returncode == 1
        assert "FAIL" in r.stderr
        # Summary line only; no per-issue lines.
        assert "link_missing" not in r.stderr
