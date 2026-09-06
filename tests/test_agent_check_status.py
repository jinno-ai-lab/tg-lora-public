"""Pin the fail-loud contract of ``scripts/agent_check_status.py``'s summary
reader.

``agent_check_status.py`` is the Makefile status target (``make`` ~line 664)
an operator runs to see dataset integrity, recent experiment runs, and a
"what to do next" recommendation. Stage 2 reads the most-recent suite's
``aggregate_summary.json``.

It used to load that summary behind a broad ``except Exception`` that printed a
buried ``[-]`` line and returned ``None`` — so a CORRUPT summary was
indistinguishable from a MISSING one, and stage 3 then told the operator to
"Run the 3-seed paper-memory suite" (``make paper-memory``): the suite had
already run, its summary was damaged, and the advice burned GPU re-running it
without ever naming the corrupt file. That is the same silent-death class the
sibling readers (``scripts/best_run_reader.py`` et al.) were promoted out of.

These tests pin the promotion to a loud, machine-distinguishable failure: a
corrupt summary must exit non-zero with a non-empty stderr that names the file
and the JSON cause, and must NOT emit the misleading "run the suite"
recommendation. Each is invoked as a real subprocess with ``cwd=tmp_path`` —
the exact CWD-globbing path the Makefile target takes — so the contract is
verified end-to-end. Reverting the reader to the old silent-swallow turns
``test_corrupt_summary_exits_loud_not_silent`` red (exit 0 + empty stderr under
the swallow, AND the misleading "run the suite" recommendation emitted), which
is the mutation the guard exists to prevent.

Stage 3 additionally pins the LOOP-HALT AWARENESS contract: when the §4/MS-008
axis is ``awaiting_ratification`` with no witnessed trigger (a
``loop_axis_state.json`` in the invocation cwd — the guard CLI's own repo-root
convention), the "what to do next" stage must surface the guard's SKIP verdict
and the operator's 3-trigger unblock set INSTEAD of recommending ``make
prepare-data`` / ``make paper-memory`` — those are axis work the halt suspends,
and the auto-diagnostic must not route the operator (or a goaldev agent) back
into blocked work. Removing the halt gate in ``evaluate_and_suggest`` turns
``test_halt_skip_supplants_blocked_recommendations`` red (the blocked
``make prepare-data`` recommendation reappears and the SKIP verdict vanishes).
The not-SKIP paths stay pinned to the LEGACY recommendation flow so the gate
cannot over-reach: a non-awaiting status and an absent tracker both keep
``make prepare-data``.

The same halt verdict now gates the TRAILING 9B Lever Readiness block: under
SKIP the block must not print at all (a second "advance the 9B lever" decision
menu contradicts the "produce nothing" verdict the operator just read), while
PROCEED / no-tracker keep it. The non-SKIP pin drives the block through a FAKE
``nvidia-smi`` injected via PATH so the block is proven to run to its GPU-probe
conclusion hermetically (CI has no nvidia-smi; dev boxes may hold a real GPU).
Removing the gate in ``main`` turns
``test_halt_skip_suppresses_9b_lever_readiness_block`` red — the readiness
block reappears under the SKIP verdict despite the fake probe succeeding.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "agent_check_status.py"

_SUITE = "runs/paper_memory_suite_test"
_SUMMARY = f"{_SUITE}/aggregate_summary.json"

# check_datasets() requires these files at-or-above these line counts before it
# reports data_ok=True; without them stage 3 short-circuits to "run data
# preparation" and never reaches the summary-dependent recommendation, so the
# silent-swallow's misleading "run the suite" path would be hidden. Seed them so
# the headline test exercises the real corruption -> wrong-recommendation path.
_DATA_MIN_LINES = {"train.jsonl": 4500, "valid_quick.jsonl": 450, "test.jsonl": 450}


def _run(
    cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _make_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, n in _DATA_MIN_LINES.items():
        (data_dir / name).write_text("x\n" * n, encoding="utf-8")


def _make_suite(tmp_path: Path, summary_content: str) -> Path:
    summary = tmp_path / _SUMMARY
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summary_content, encoding="utf-8")
    return summary


def test_corrupt_summary_exits_loud_not_silent(tmp_path: Path) -> None:
    """The headline fix: a corrupt aggregate_summary.json must NOT degrade to
    the misleading 'run the suite' recommendation. Under the old broad
    ``except Exception`` this returned None -> (with data_ok=True) stage 3 said
    'Run the 3-seed paper-memory suite' with exit 0 and empty stderr. Now it
    must exit non-zero with a non-empty stderr that names the file + JSON cause,
    and must NOT emit the re-run advice."""
    _make_data(tmp_path)
    # Trailing comma — the exact 'Expecting property name enclosed in double
    # quotes' JSON failure mode (the prior iteration's judge_invalid_json).
    _make_suite(tmp_path, '{"seeds": 3,}')
    result = _run(tmp_path)
    assert result.returncode == 2, result.stderr
    assert result.stderr.strip(), "corrupt summary died silently (empty stderr)"
    assert "corrupt" in result.stderr.lower()
    assert "JSON" in result.stderr
    assert "aggregate_summary.json" in result.stderr
    # The misleading re-run recommendation must NOT be emitted: the suite ran,
    # the summary is damaged, not absent.
    assert "Run the 3-seed paper-memory suite" not in result.stdout


def test_valid_summary_loads_and_exits_zero(tmp_path: Path) -> None:
    _make_data(tmp_path)
    _make_suite(tmp_path, '{"seeds": 3, "best_valid_loss": 1.05}')
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "[+] Loaded aggregate_summary.json" in result.stdout


def test_missing_summary_does_not_crash(tmp_path: Path) -> None:
    # A suite dir with no aggregate_summary.json: the reader must fall through
    # to None (unchanged) and exit 0 — corruption is loud, absence is not.
    suite = tmp_path / _SUITE
    suite.mkdir(parents=True)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "aggregate_summary.json not found" in result.stdout


def _write_halt_tracker(
    tmp_path: Path, status: str, witness: str | None = None
) -> None:
    """Seed a ``loop_axis_state.json`` in the invocation cwd with the essential
    guard fields (status / baseline / operator_signals), mirroring the shape of
    the repo-root tracker the guard CLI reads.

    ``witness`` optionally fires one of the guard's three external triggers in
    the SAME cwd the subprocess will glob — the real witness artifacts the
    guard's ``compute_witnesses`` reads (a 9B deposit fixture over baseline, a
    close-the-loop anchor, or the operator axis-open signal)."""
    tracker = {
        "schema_version": 1,
        "axis": "section4-9b-verdict",
        "status": status,
        "baseline": {"nine_b_deposit_count": 10},
        "operator_signals": {"new_ms_axis_opened": None},
    }
    if witness == "9b_deposit":
        # 1 committed deposit fixture vs baseline 0 -> count > baseline fires.
        tracker["baseline"] = {"nine_b_deposit_count": 0}
    elif witness == "new_ms_axis":
        tracker["operator_signals"]["new_ms_axis_opened"] = True
    (tmp_path / "loop_axis_state.json").write_text(
        json.dumps(tracker), encoding="utf-8"
    )
    if witness == "9b_deposit":
        fixtures = tmp_path / "tests" / "fixtures"
        fixtures.mkdir(parents=True, exist_ok=True)
        (fixtures / "freeze_validloss_ci_9b_witnessed.json").write_text(
            "{}", encoding="utf-8"
        )
    elif witness == "closeout":
        closeout = tmp_path / "reports" / "close-the-loop"
        closeout.mkdir(parents=True, exist_ok=True)
        (closeout / "close_the_loop_funnel_go_nogo_witnessed.json").write_text(
            "{}", encoding="utf-8"
        )


def test_halt_skip_supplants_blocked_recommendations(tmp_path: Path) -> None:
    """awaiting_ratification + no witnessed trigger: stage 3 must surface the
    SKIP verdict and the operator unblock set, and must NOT recommend the
    axis work the halt suspends (prepare-data / paper-memory)."""
    _write_halt_tracker(tmp_path, "awaiting_ratification")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "SKIP (halt" in result.stdout
    assert "awaiting_ratification" in result.stdout
    assert "Operator decision" in result.stdout
    assert "make loop-halt-check" in result.stdout
    # Pin the RECOMMENDATION form ("Command: make …"), not the bare target
    # name — the SKIP block itself legitimately names the suspended targets.
    assert "Command: make prepare-data" not in result.stdout
    assert "Command: make paper-memory" not in result.stdout


def test_halt_proceed_9b_deposit_trigger_surfaces_minimal_step(tmp_path: Path) -> None:
    """awaiting_ratification + the 9b_leg_fired witness (a committed deposit
    count over baseline): the guard says PROCEED, and stage 3 must surface the
    verdict, the FIRED trigger, and that axis's minimal step (the Cat-C 9B
    PRODUCTION-baseline comparison) — not the generic prepare-data menu. This
    is the D-2 misroute: the trigger fired, but without this block the operator
    cannot tell WHICH axis's minimal step to run."""
    _write_halt_tracker(tmp_path, "awaiting_ratification", witness="9b_deposit")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "PROCEED" in result.stdout
    assert "9b_leg_fired" in result.stdout
    assert "PRODUCTION baseline" in result.stdout
    assert "Command: make prepare-data" not in result.stdout


def test_halt_proceed_closeout_trigger_surfaces_minimal_step(tmp_path: Path) -> None:
    """awaiting_ratification + the closeout_approved witness (a close-the-loop
    anchor on mirror): PROCEED must surface the fired trigger and the closeout
    minimal step (finalize the MS-008 publishable-negative closeout), keeping
    the operator off the generic milestone menu."""
    _write_halt_tracker(tmp_path, "awaiting_ratification", witness="closeout")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "PROCEED" in result.stdout
    assert "closeout_approved" in result.stdout
    assert "publishable-negative closeout" in result.stdout
    assert "Command: make prepare-data" not in result.stdout


def test_halt_proceed_new_ms_axis_trigger_surfaces_minimal_step(
    tmp_path: Path,
) -> None:
    """awaiting_ratification + the new_ms_axis_opened witness (operator flipped
    the axis-open signal): PROCEED must surface the fired trigger and point at
    the NEW axis's minimal step only — the whole point of the signal is to
    redirect work away from the old milestone flow."""
    _write_halt_tracker(tmp_path, "awaiting_ratification", witness="new_ms_axis")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "PROCEED" in result.stdout
    assert "new_ms_axis_opened" in result.stdout
    assert "new axis" in result.stdout
    assert "Command: make prepare-data" not in result.stdout


def test_halt_not_active_keeps_legacy_recommendations(tmp_path: Path) -> None:
    # status != awaiting_ratification (the guard returns PROCEED): the halt
    # gate must not alter stage 3's legacy recommendation flow.
    _write_halt_tracker(tmp_path, "ratified")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Command: make prepare-data" in result.stdout
    assert "SKIP (halt" not in result.stdout


def test_no_halt_tracker_keeps_legacy_recommendations(tmp_path: Path) -> None:
    # No loop_axis_state.json in cwd: the guard is permissive, stage 3 keeps
    # its legacy flow — the status check never bricks on a missing tracker.
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Command: make prepare-data" in result.stdout
    assert "SKIP (halt" not in result.stdout


def _fake_nvidia_smi_env(tmp_path: Path, apps_csv: str) -> dict[str, str]:
    """Env whose PATH fronts a fake ``nvidia-smi`` emitting ``apps_csv``.

    ``report_gpu_availability`` probes the GPU via ``nvidia-smi`` resolved from
    PATH; faking it keeps the pins hermetic (CI has no nvidia-smi; dev boxes
    may hold a real sibling-project GPU) and lets the SKIP pin prove the block
    is SUPPRESSED by the gate, not by a dead probe — under the fake, an
    ungated run would print the block and the holder lines."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / "nvidia-smi"
    tool.write_text(
        "#!/bin/sh\ncat <<'EOF'\n" + apps_csv + "\nEOF\n", encoding="utf-8"
    )
    tool.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }


def test_halt_skip_suppresses_9b_lever_readiness_block(tmp_path: Path) -> None:
    """awaiting_ratification + no witnessed trigger: the trailing 9B Lever
    Readiness block must NOT print after the SKIP verdict — the operator just
    read "produce nothing" plus the 3-trigger unblock set, and a second
    "advance the 9B lever" decision menu contradicts it. The fake nvidia-smi
    (probe WOULD succeed) proves suppression is the halt gate's doing."""
    _write_halt_tracker(tmp_path, "awaiting_ratification")
    env = _fake_nvidia_smi_env(tmp_path, "4321, llama-server, 9000 MiB")
    result = _run(tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    assert "SKIP (halt" in result.stdout
    assert "=== 9B Lever Readiness" not in result.stdout
    assert "GPU is held by" not in result.stdout


def test_halt_not_active_keeps_9b_lever_readiness_block(tmp_path: Path) -> None:
    """Non-awaiting status (guard returns PROCEED): the gate must not
    over-reach — the 9B readiness block keeps printing through to its GPU-probe
    conclusion (holder line pinned via the fake nvidia-smi)."""
    _write_halt_tracker(tmp_path, "ratified")
    env = _fake_nvidia_smi_env(tmp_path, "4321, llama-server, 9000 MiB")
    result = _run(tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    assert "SKIP (halt" not in result.stdout
    assert "=== 9B Lever Readiness" in result.stdout
    assert "GPU is held by 1 compute app(s)" in result.stdout
    assert "4321" in result.stdout


def test_cli_make_check_status_reachable_without_venv(tmp_path: Path) -> None:
    """E2E: ``make check-status`` must stay reachable in venv-less worktrees.

    The halt-awareness contract above lives behind the Makefile target, but AI
    Hub executes this repo from fresh worktrees that have no ``.venv`` — there
    the default ``PYTHON_VENV`` (``.venv/bin/python``) died with exit 127, so
    the diagnostic that surfaces the SKIP verdict was unreachable exactly where
    agents consult it, and the ticket's acceptance grep
    (``make check-status | grep -E 'SKIP|awaiting_ratification|Operator
    decision'``) matched nothing because the recipe never reached the script.
    The recipe now uses the same interpreter fallback as ``loop-halt-check``
    (PYTHON_VENV if executable, else $(PYTHON), else python3 — the script is
    pure stdlib). Simulate a venv-less checkout (point VENV at an absent path,
    clear any PYTHON_VENV override) and assert the live repo's SKIP surfacing
    still reaches stdout through make.
    """
    env = {**os.environ, "VENV": str(tmp_path / "absent-venv")}
    env.pop("PYTHON_VENV", None)
    proc = subprocess.run(
        ["make", "check-status"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"make check-status exited {proc.returncode} in a venv-less checkout — "
        f"the halt-aware diagnostic is unreachable exactly where the loop "
        f"runs:\n{proc.stdout}\n{proc.stderr}"
    )
    for needle in ("SKIP", "awaiting_ratification", "Operator decision"):
        assert needle in proc.stdout, (
            f"the live repo's halt-state surfacing lost {needle!r} through "
            f"make:\n{proc.stdout}\n{proc.stderr}"
        )
