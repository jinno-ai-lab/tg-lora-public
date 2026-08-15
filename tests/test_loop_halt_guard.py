"""Tests for ``scripts/loop_halt_guard.py``.

The guard makes "stop" structural for the §4 / MS-008 axis: when the axis is
``awaiting_ratification`` and no external trigger has arrived, the loop must
SKIP (produce nothing) instead of emitting another halt-doc. These tests prove:

  * the guard HALTS on the live repo right now (``test_halt_now_on_real_repo``),
  * the recorded baseline is honest (``test_real_repo_baseline_is_honest``),
  * each of the three triggers clears SKIP (mutation-proof),
  * committing the guard itself does NOT self-unblock
    (``test_guard_commit_does_not_self_unblock``) — this resolves the feedback
    paradox: "a genuine halt emits no commit, yet this commit contradicts the
    halt". The guard's own files trip no trigger, so its commit is consistent
    with halting, unlike a halt-doc.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.loop_halt_guard import (
    AWAITING,
    EXIT_PROCEED,
    EXIT_SKIP,
    compute_witnesses,
    decide_skip,
    main,
    should_skip,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_GLOB = "freeze_validloss_ci_9b*.json"


def _seed_nine_b(repo: Path, n: int) -> None:
    fixtures = repo / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (fixtures / f"freeze_validloss_ci_9b_seed{i}.json").write_text("{}", encoding="utf-8")


def _write_state(
    repo: Path,
    *,
    status: str = AWAITING,
    baseline_count: int = 0,
    new_ms_axis: str | None = None,
) -> dict:
    state = {
        "schema_version": 1,
        "axis": "section4-9b-verdict",
        "status": status,
        "baseline": {
            "nine_b_deposit_count": baseline_count,
            "measured_at_commit": "test-seed",
            "glob": f"tests/fixtures/{FIXTURES_GLOB}",
        },
        "operator_signals": {"new_ms_axis_opened": new_ms_axis},
    }
    (repo / "loop_axis_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    return state


# ── pure decision core ───────────────────────────────────────────────────────


def test_decide_skip_awaiting_with_no_trigger_skips():
    decision = decide_skip(AWAITING, {"9b_leg_fired": False,
                                      "closeout_approved": False,
                                      "new_ms_axis_opened": False})
    assert decision.skip is True
    assert decision.active_triggers == []
    assert decision.status == AWAITING


def test_decide_skip_non_awaiting_status_proceeds():
    decision = decide_skip("ready", {"9b_leg_fired": False,
                                     "closeout_approved": False,
                                     "new_ms_axis_opened": False})
    assert decision.skip is False
    assert decision.status == "ready"


@pytest.mark.parametrize(
    "trigger",
    ["9b_leg_fired", "closeout_approved", "new_ms_axis_opened"],
)
def test_decide_skip_any_single_trigger_clears_skip(trigger):
    witnesses = {"9b_leg_fired": False, "closeout_approved": False,
                 "new_ms_axis_opened": False}
    witnesses[trigger] = True
    decision = decide_skip(AWAITING, witnesses)
    assert decision.skip is False
    assert decision.active_triggers == [trigger]


# ── live-repo halt proof + baseline honesty ──────────────────────────────────


def test_halt_now_on_real_repo():
    """The guard MUST halt the loop on the live repo as it stands today."""
    decision = should_skip(REPO_ROOT)
    assert decision.skip is True, (
        f"guard did not halt on the live repo: {decision.as_dict()}"
    )
    assert decision.status == AWAITING
    assert decision.active_triggers == []


def test_real_repo_baseline_is_honest():
    """The tracker's recorded baseline must equal the live fixture count.

    A baseline that over-counts would hide a real 9B run; under-count would
    false-fire. This pins the recorded value to ground truth.
    """
    state = json.loads((REPO_ROOT / "loop_axis_state.json").read_text(encoding="utf-8"))
    recorded = state["baseline"]["nine_b_deposit_count"]
    actual = len(list((REPO_ROOT / "tests" / "fixtures").glob(FIXTURES_GLOB)))
    assert recorded == actual, (
        f"loop_axis_state.json baseline.nine_b_deposit_count={recorded} but the "
        f"live repo has {actual} matching fixtures — the baseline is stale"
    )


def test_live_repo_witnesses_all_false():
    """The three triggers must all be inactive on the live repo right now."""
    state = json.loads((REPO_ROOT / "loop_axis_state.json").read_text(encoding="utf-8"))
    witnesses = compute_witnesses(REPO_ROOT, state)
    assert witnesses == {
        "9b_leg_fired": False,
        "closeout_approved": False,
        "new_ms_axis_opened": False,
    }


# ── entrypoint wiring contracts ──────────────────────────────────────────────
#
# The guard exists only in the Makefile. The goaldev agent picks work from the
# entrypoint docs (PURPOSE.md / AGENTS.md), so those docs must route every
# future iteration through ``make loop-halt-check`` before touching the
# awaiting_ratification axis. These contracts prevent the silent drift of that
# wiring — if either reference disappears, the structural "stop" becomes
# unreachable from the agent's read path again.


def test_purpose_entrypoint_routes_through_halt_check():
    """PURPOSE.md must reference the pre-flight and the state file."""
    text = (REPO_ROOT / "PURPOSE.md").read_text(encoding="utf-8")
    assert "make loop-halt-check" in text, (
        "PURPOSE.md no longer routes readers through `make loop-halt-check` — "
        "the halt guard is unreachable from the goaldev entrypoint"
    )
    assert "loop_axis_state.json" in text, (
        "PURPOSE.md no longer points at loop_axis_state.json — readers cannot "
        "find the halt state's source of truth"
    )


def test_agents_entrypoint_routes_through_halt_check():
    """AGENTS.md must reference the pre-flight and the state file."""
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "make loop-halt-check" in text, (
        "AGENTS.md no longer routes readers through `make loop-halt-check` — "
        "the halt guard is unreachable from the goaldev entrypoint"
    )
    assert "loop_axis_state.json" in text, (
        "AGENTS.md no longer points at loop_axis_state.json — readers cannot "
        "find the halt state's source of truth"
    )


# ── permissive defaults ──────────────────────────────────────────────────────


def test_no_tracker_is_permissive(tmp_path):
    """No loop_axis_state.json -> the guard must not block unrelated work."""
    assert should_skip(tmp_path).skip is False


def test_non_awaiting_status_is_permissive(tmp_path):
    _write_state(tmp_path, status="ready")
    assert should_skip(tmp_path).skip is False


# ── trigger mutation proofs ──────────────────────────────────────────────────


def test_9b_leg_trigger_clears_skip(tmp_path):
    _seed_nine_b(tmp_path, n=3)
    _write_state(tmp_path, baseline_count=3)
    assert should_skip(tmp_path).skip is True  # exactly at baseline -> still halt

    _seed_nine_b(tmp_path, n=4)  # mutation: a new 9B run fired
    decision = should_skip(tmp_path)
    assert decision.skip is False
    assert decision.active_triggers == ["9b_leg_fired"]


def test_closeout_trigger_clears_skip(tmp_path):
    _seed_nine_b(tmp_path, n=1)
    _write_state(tmp_path, baseline_count=1)
    assert should_skip(tmp_path).skip is True

    closeout = tmp_path / "reports" / "close-the-loop"
    closeout.mkdir(parents=True)
    (closeout / "close_the_loop_funnel_go_nogo_20260808.json").write_text("{}", encoding="utf-8")
    decision = should_skip(tmp_path)
    assert decision.skip is False
    assert decision.active_triggers == ["closeout_approved"]


def test_new_ms_axis_trigger_clears_skip(tmp_path):
    _seed_nine_b(tmp_path, n=1)
    _write_state(tmp_path, baseline_count=1, new_ms_axis=None)
    assert should_skip(tmp_path).skip is True

    _write_state(tmp_path, baseline_count=1, new_ms_axis="MS-009")  # operator flip
    decision = should_skip(tmp_path)
    assert decision.skip is False
    assert decision.active_triggers == ["new_ms_axis_opened"]


def test_guard_commit_does_not_self_unblock(tmp_path):
    """The paradox resolver: committing the guard must NOT clear the halt.

    Simulates the guard's own commit (adding this script + tracker + tests) on
    top of a baseline-matching repo. None of those files is a 9B deposit, a
    closeout anchor, or an operator axis-open flip — so should_skip stays True.
    This is exactly the property a halt-doc lacks: its own existence contradicts
    the halt; the guard's does not.
    """
    _seed_nine_b(tmp_path, n=2)
    _write_state(tmp_path, baseline_count=2)
    assert should_skip(tmp_path).skip is True

    # Simulate the guard commit landing: add the guard's own files to the repo.
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "loop_halt_guard.py").write_text("# guard", encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_loop_halt_guard.py").write_text("# tests", encoding="utf-8")

    decision = should_skip(tmp_path)
    assert decision.skip is True, (
        "guard commit self-unblocked the halt — the paradox would reproduce"
    )
    assert decision.active_triggers == []


# ── CLI exit codes ───────────────────────────────────────────────────────────


def test_cli_returns_skip_exit_on_live_repo():
    assert main(["--repo-root", str(REPO_ROOT)]) == EXIT_SKIP


def test_cli_returns_proceed_exit_when_not_awaiting(tmp_path):
    _write_state(tmp_path, status="ready")
    assert main(["--repo-root", str(tmp_path)]) == EXIT_PROCEED


def test_cli_json_output_is_valid(tmp_path, capsys):
    _write_state(tmp_path, status=AWAITING, baseline_count=0)
    rc = main(["--repo-root", str(tmp_path), "--json"])
    assert rc == EXIT_SKIP
    out = json.loads(capsys.readouterr().out)
    assert out["skip"] is True
    assert out["status"] == AWAITING


def test_cli_make_target_reaches_guard_without_venv(tmp_path):
    """``make loop-halt-check`` must reach the guard where .venv is absent.

    The entrypoint contracts above route every iteration through this make
    target, but AI Hub executes the repo from fresh worktrees that have no
    ``.venv`` — there the default ``PYTHON_VENV`` (``.venv/bin/python``) used
    to die with exit 127, and GNU make cannot propagate the guard's exit 77
    anyway (a failing recipe line collapses to make exit 2), so SKIP and a
    broken pre-flight were indistinguishable at the make surface. The target
    now (a) falls back to any working stdlib interpreter and (b) translates:
    a guard verdict of 0/77 is echoed as an explicit ``verdict rc=N`` line with
    make exiting 0; only a broken pre-flight exits non-zero. Simulate a
    venv-less checkout (point VENV at an absent path, clear any PYTHON_VENV
    override) and assert the live repo's SKIP verdict is still reached.
    """
    env = {**os.environ, "VENV": str(tmp_path / "absent-venv")}
    env.pop("PYTHON_VENV", None)
    proc = subprocess.run(
        ["make", "loop-halt-check"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"make loop-halt-check exited {proc.returncode} in a venv-less "
        f"checkout — the pre-flight is unreachable or broken exactly where "
        f"the loop runs:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "verdict rc=77" in proc.stdout, (
        f"the live repo's SKIP verdict did not surface through make:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "BROKEN" not in proc.stdout + proc.stderr
