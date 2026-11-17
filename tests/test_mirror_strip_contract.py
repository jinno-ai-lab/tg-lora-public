"""Prevent-recurrence guards for the public-mirror strip contract.

The 2026-10-13 incident: leftover-working-tree sweeps re-landed the private
``src/data`` pipeline into this public mirror twice (559d47b, then e4434a3),
the mirror-contract pin went permanently red on main, and the ``-x``
full-suite test job stranded every auto-merge PR for a month until f42af86
removed the files again and scoped CI to the canary.

The fix has two halves, and each previously had a silent-failure mode:

- detection (the pin test) existed but ran nowhere in CI, so a re-landing
  after the canary scoping would not fail CI at all;
- prevention-at-add-time did not exist — ``src/data`` was not gitignored, so
  any ``git add -A`` sweep could pick it up again.

These tests pin the prevention stack itself so it cannot silently rot:

1. ``src/data`` stays gitignored (sweeps cannot re-track it);
2. the CI test job invokes the file holding the mirror pin;
3. no coverage floor comes back into CI (the unattainable ``--cov-fail-under=95``
   is what kept base main permanently red and muted the first incident).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _effective_ignore_patterns() -> list[str]:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def test_gitignore_blocks_src_data_relanding() -> None:
    """``src/data`` must stay gitignored so leftover-working-tree sweeps
    (``git add -A`` — the e4434a3 path) cannot re-track the private pipeline.
    Mutation pin: removing the ``/src/data/`` line flips this red."""
    patterns = _effective_ignore_patterns()
    assert "/src/data/" in patterns or "src/data/" in patterns


def test_ci_test_job_runs_the_mirror_pin_file() -> None:
    """The CI test job must invoke ``tests/test_agent_check_status_gpu.py`` —
    the file holding ``test_src_data_pipeline_present_false_on_public_mirror``.
    After f42af86 scoped the job to the canary only, a ``src/data`` re-landing
    would no longer fail CI anywhere; this invocation is the post-landing
    detection. Mutation pin: dropping the path from the workflow flips this
    red."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "pytest tests/test_cli_help_smoke.py" in workflow
    assert "tests/test_agent_check_status_gpu.py" in workflow


def test_ci_does_not_run_an_unattainable_coverage_floor() -> None:
    """The first incident hid for a month because the test job also demanded
    ``--cov-fail-under=95`` — unattainable on the stripped tree, so base main
    was permanently red regardless and one more red cause changed nothing
    visible. Pin that a coverage floor stays out of this mirror's CI."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "--cov-fail-under" not in workflow
