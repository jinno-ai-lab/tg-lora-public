"""Tests for the git-hook DEPLOYMENT path (scripts/install_hooks.sh), not the
hook's behaviour (that lives in tests/test_commit_hook.py).

Why these tests exist (AI-Hub feedback, policy 0.3.15, bullet 4 follow-up):
the feedback's named pathology -- "a guard that passes due to its own presence
has not stopped the pathology" / a guard that "silently skip[s] enforcement" --
applies directly to this repo's endorsed highest-leverage item, the
prepare-commit-msg empty-commit guard. That guard can fire ONLY if it is
installed, and nothing previously verified installation:

  * tests/test_commit_hook.py mirrors `make install-hooks` by hand-rolling its
    OWN symlink, so it proved "IF installed, the hook fires" while leaving the
    real deployment recipe untested.
  * The hook was not wired into any standard setup path -- `make install` did
    not call `install-hooks`, so every fresh clone / linked worktree (including
    the make-run execution path the guard was built to protect) shipped INERT.

These tests close that hole by exercising the REAL install script and the
Makefile wiring:

  * every checked-in hook deploys to the target dir
  * each lands as a RESOLVING symlink to this repo's checked-in source (no
    dangling link -> no silent inert guard)
  * the deployed prepare-commit-msg guard is ACTIVE (rejects an empty commit
    when invoked THROUGH the installed symlink, proving the full
    symlink -> hook -> checker chain is live)
  * install is idempotent (re-run replaces, no duplicates)
  * `make install-hooks` honors a HOOKS_DIR override (end-to-end recipe test)
  * `make install` wires in `install-hooks` (the standard setup deploys it)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "install_hooks.sh"
HOOKS_SRC = REPO_ROOT / "scripts" / "hooks"
MAKEFILE = REPO_ROOT / "Makefile"


def _checked_in_hooks() -> list[str]:
    return sorted(p.name for p in HOOKS_SRC.iterdir() if p.is_file())


def _run_script(target_dir: Path) -> subprocess.CompletedProcess:
    """Run the REAL install script, redirecting HOOKS_DIR to a sandbox so the
    developer's live hooks dir is never mutated by the test."""
    target_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HOOKS_DIR": str(target_dir)}
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
    )


def _seed_repo(repo: Path) -> None:
    """A throwaway git repo with one committed file, so an empty commit later is
    unambiguous."""
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@e.com"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "tester"], check=True
    )
    (repo / "note.md").write_text("baseline\n")
    subprocess.run(["git", "-C", str(repo), "add", "note.md"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "baseline"], check=True
    )


def test_script_is_present_and_executable():
    assert SCRIPT.is_file(), f"install script missing at {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"install script not executable: {SCRIPT}"


def test_installs_every_checked_in_hook(tmp_path):
    expected = _checked_in_hooks()
    assert expected, "no checked-in hooks found under scripts/hooks/"
    proc = _run_script(tmp_path / "hooks")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    installed = sorted(p.name for p in (tmp_path / "hooks").iterdir())
    assert installed == expected, (
        f"deployment mismatch: installed {installed}, expected {expected} -- "
        "a checked-in hook is not being deployed"
    )


def test_installed_hooks_are_resolving_symlinks(tmp_path):
    """A symlink that dangles is an inert guard. Each deployed hook must resolve
    to THIS repo's checked-in source, not a stale/missing copy."""
    proc = _run_script(tmp_path / "hooks")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    hooks_dir = tmp_path / "hooks"
    deployed = [p for p in hooks_dir.iterdir()]
    assert deployed, "no hooks were deployed"
    for link in deployed:
        assert link.is_symlink(), f"{link.name} is not a symlink"
        resolved = link.resolve()
        assert resolved.is_file(), (
            f"{link.name} -> {resolved} dangles -- the guard would be inert"
        )
        assert resolved == (HOOKS_SRC / link.name).resolve(), (
            f"{link.name} resolves to {resolved}, not the checked-in source"
        )


def test_deployed_prepare_commit_msg_guard_is_active(tmp_path):
    """End-to-end THROUGH the installed symlink (not the source file): the
    deployed guard must reject an empty staged delta. A symlink that resolves
    but whose checker is unreachable would still be inert -- this proves the
    full chain (symlink -> hook -> check_substantive_diff.py) is live, which the
    source-file tests in test_commit_hook.py cannot."""
    proc = _run_script(tmp_path / "hooks")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    installed = tmp_path / "hooks" / "prepare-commit-msg"
    assert installed.exists(), "prepare-commit-msg was not deployed"

    repo = tmp_path / "repo"
    _seed_repo(repo)
    msg = repo / "COMMIT_EDITMSG"
    msg.write_text("would-be empty commit\n")
    # source=message with no staged delta is the empty-commit class the guard
    # exists to stop. Invoke the INSTALLED symlink directly (its readlink resolves
    # the checker to this repo's scripts/).
    res = subprocess.run(
        ["bash", str(installed.resolve()), str(msg), "message"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, (
        "deployed prepare-commit-msg allowed an empty commit -- the guard is inert"
    )
    assert "non-substantive" in res.stderr, res.stderr


def test_install_is_idempotent(tmp_path):
    """Re-running must not error or leave duplicates (ln -sf replaces in place)."""
    hd = tmp_path / "hooks"
    first = _run_script(hd)
    assert first.returncode == 0, first.stdout + first.stderr
    second = _run_script(hd)
    assert second.returncode == 0, second.stdout + second.stderr
    names = sorted(p.name for p in hd.iterdir())
    assert names == _checked_in_hooks(), (
        f"idempotent re-run changed the deployed set: {names}"
    )


def test_make_install_hooks_deploys_to_sandbox(tmp_path):
    """The Makefile `install-hooks` target must invoke the real script and honor
    a HOOKS_DIR override, so the recipe (Makefile -> script -> symlinks) is
    verified end-to-end without touching the developer's live hooks dir."""
    sandbox = tmp_path / "hooks"
    env = {**os.environ, "HOOKS_DIR": str(sandbox)}
    proc = subprocess.run(
        ["make", "install-hooks"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    deployed = sorted(p.name for p in sandbox.iterdir()) if sandbox.exists() else []
    assert deployed == _checked_in_hooks(), (
        f"make install-hooks did not deploy all hooks via the script: {deployed}"
    )


def test_make_install_wires_in_install_hooks():
    """The deployment GAP this iteration closes: the standard `make install` setup
    must deploy the guard. Before this, only a manual `make install-hooks` did, so
    fresh clones/worktrees shipped inert. Structural assertion (mirrors
    tests/test_ci_gate_wiring.py) so a future edit that drops the wiring is
    caught -- a guard that is never auto-deployed is exactly the silent
    fail-open the feedback warns about."""
    text = MAKEFILE.read_text()
    # Isolate the `install:` target body (up to the next blank line). The first
    # `install:` in the Makefile is the setup target; `install-hooks:` does not
    # contain the `install:` substring (the char after `install` is `-`).
    body = text.split("install:", 1)[1].split("\n\n", 1)[0]
    assert "install-hooks" in body, (
        "`make install` does not deploy git hooks -- the prepare-commit-msg guard "
        "is inert in any clone/worktree that only runs the standard setup. Add "
        "install-hooks as a prerequisite/step of the install target."
    )


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
