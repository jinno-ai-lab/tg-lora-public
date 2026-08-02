"""Tests for the prepare-commit-msg hook (scripts/hooks/prepare-commit-msg).

The hook is the *structural* answer to AI-Hub make-run feedback bullet 4: the
prior commit-hygiene guard (`make check-commit-substantive`) was opt-in, so the
make-run runner's bare `git commit` bypassed it and empty commits reached main.
This hook makes the SAME check (scripts/check_substantive_diff.py) fire
automatically at commit time.

These tests mirror `make install-hooks` exactly (symlink the checked-in hook
into a throwaway repo's hooks dir) and assert the four behaviours the hook
promises in its header:

  * substantive staged commit        -> allowed
  * empty staged commit (--allow-empty) -> rejected (the make-run anti-pattern)
  * merge / squash / amend source     -> exempt (no staged delta by design)
  * TG_LORA_ALLOW_EMPTY_COMMIT=1     -> allowed (operator escape hatch)

They are end-to-end (real `git commit` in a temp repo with the real hook
installed), so a regression in the hook's checker-location resolution, source
exemption, or exit-code mapping will RED them.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "scripts" / "hooks" / "prepare-commit-msg"


def _git(
    repo: Path, *args: str, env: dict | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _new_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.email", "t@example.com").check_returncode()
    _git(repo, "config", "user.name", "tester").check_returncode()
    # Disable any template/default hooks so only OUR hook is in play, and pin
    # the default branch so `git commit` works without -b on older/newer gits.
    _git(repo, "config", "init.defaultBranch", "main").check_returncode()


def _install_hook(repo: Path) -> None:
    """Mirror `make install-hooks`: symlink the checked-in hook into the repo's
    hooks dir so readlink -f resolves the checker to this repo's scripts/."""
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    # Clear any default sample hooks so they cannot interfere.
    for sample in hooks_dir.glob("*.sample"):
        sample.unlink()
    link = hooks_dir / "prepare-commit-msg"
    link.symlink_to(HOOK)


def _seed_baseline(repo: Path) -> None:
    """One committed file so subsequent empty commits are unambiguous."""
    f = repo / "note.md"
    f.write_text("baseline\n")
    _git(repo, "add", "note.md").check_returncode()
    _git(repo, "commit", "-q", "-m", "baseline").check_returncode()


def test_hook_allows_substantive_commit(tmp_path):
    """A real staged edit must pass the hook and create the commit."""
    repo = tmp_path
    _new_repo(repo)
    _install_hook(repo)
    _seed_baseline(repo)

    (repo / "note.md").write_text("baseline\nNEW SUBSTANTIVE CONTENT\n")
    _git(repo, "add", "note.md").check_returncode()

    proc = _git(repo, "commit", "-q", "-m", "real edit")
    assert proc.returncode == 0, proc.stderr
    # The commit actually landed.
    log = _git(repo, "log", "--oneline")
    assert "real edit" in log.stdout, log.stdout


def test_hook_rejects_empty_commit(tmp_path):
    """Nothing staged + --allow-empty is the exact make-run meta-commit pattern;
    the hook must reject it (non-zero) and NOT create the commit."""
    repo = tmp_path
    _new_repo(repo)
    _install_hook(repo)
    _seed_baseline(repo)

    proc = _git(repo, "commit", "--allow-empty", "-m", "chore(make-run): no-op")
    assert proc.returncode != 0, "empty commit was allowed -- hook failed to fire"
    assert "non-substantive" in proc.stderr, proc.stderr
    # HEAD must still be the baseline commit only.
    log = _git(repo, "log", "--oneline")
    assert "no-op" not in log.stdout, log.stdout


def test_hook_rejects_whitespace_only_commit(tmp_path):
    """A staged change that is pure whitespace/reflow (normalization-only) is
    the other named churn class -- also rejected."""
    repo = tmp_path
    _new_repo(repo)
    _install_hook(repo)
    _seed_baseline(repo)

    (repo / "note.md").write_text("baseline   \n")  # trailing-space reflow only
    _git(repo, "add", "note.md").check_returncode()

    proc = _git(repo, "commit", "-q", "-m", "reflow only")
    assert proc.returncode != 0, proc.stderr
    assert "non-substantive" in proc.stderr, proc.stderr


def test_hook_escape_hatch_allows_empty_commit(tmp_path):
    """TG_LORA_ALLOW_EMPTY_COMMIT=1 is the documented operator escape hatch."""
    repo = tmp_path
    _new_repo(repo)
    _install_hook(repo)
    _seed_baseline(repo)

    proc = subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-m", "deliberate empty"],
        capture_output=True,
        text=True,
        env={**os.environ, "TG_LORA_ALLOW_EMPTY_COMMIT": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    log = _git(repo, "log", "--oneline")
    assert "deliberate empty" in log.stdout, log.stdout


def test_hook_exempts_merge_source_directly(tmp_path):
    """A merge commit carries no staged delta relative to its first parent; the
    `$2 == merge` source exemption must let it through. Invoked directly so the
    test does not depend on a real merge's index shape."""
    repo = tmp_path
    _new_repo(repo)
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    msg_file = repo / "COMMIT_EDITMSG"
    msg_file.write_text("merge commit\n")

    proc = subprocess.run(
        ["bash", str(HOOK), str(msg_file), "merge"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_hook_rejects_empty_when_invoked_directly(tmp_path):
    """Direct invocation with no staged content (source=message) rejects -- this
    pins the checker-location resolution + verdict->exit mapping without a full
    `git commit`, so a future refactor that breaks readlink/rev-parse discovery
    is caught even if `git commit` itself were to change behaviour."""
    repo = tmp_path
    _new_repo(repo)
    _seed_baseline(repo)
    msg_file = repo / "COMMIT_EDITMSG"
    msg_file.write_text("x\n")

    proc = subprocess.run(
        ["bash", str(HOOK), str(msg_file), "message"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "non-substantive" in proc.stderr, proc.stderr


def test_hook_is_present_and_executable():
    """Guard the deliverable itself exists and is runnable (CI runs on Linux)."""
    assert HOOK.is_file(), f"hook missing at {HOOK}"
    assert os.access(HOOK, os.X_OK), f"hook not executable: {HOOK}"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
