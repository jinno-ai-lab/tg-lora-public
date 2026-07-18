"""Self-retrying launcher for the full-budget 9B §4 verdict.

``make freeze-validloss-ci-9b-full`` is the path to the first
``citable_as_full_section4_verdict=True`` deposit (``total_steps=1500``,
generalization regime, candidate+surrogate+baseline at 3 seeds/arm — ~hours of
9B GPU on a shared 12 GB card). Two pieces already make the multi-hour run
survivable: the worker (``scripts.run_freeze_validloss_ci_9b``) defers a held GPU
with exit ``75`` (EX_TEMPFAIL — "retry later") and banks each completed arm in the
resumable ``--ledger`` so an interruption does not restart from zero.

The gap this launcher closes is documented in the Makefile: GNU make flattens
EVERY non-zero recipe exit to make-exit 2, so a poll loop around ``make`` cannot
tell tempfail (75, retry) from CUDA-down (2, fatal) or torn-ledger (3, re-run) —
they all look identical. ``launch_freeze_ci_9b_full`` therefore bypasses make for
the worker step: it subprocess-invokes the module *directly* (``-m
scripts.run_freeze_validloss_ci_9b``) so the exit codes survive, branches on them,
retries 75/3 with backoff, stops on 0/1/2, and is bounded so a persistently-held
GPU cannot spin forever. Completed arms stay banked in the ``--ledger`` across
free-GPU windows.

Worker exit-code contract (``scripts/run_freeze_validloss_ci_9b.py``, lines
153-160)::

    0  success (deposit written)           -> DONE
    1  unexpected error (uncaught exc.)    -> FATAL (operator: investigate, re-run)
    2  CUDA unavailable                    -> FATAL (operator: not retryable here)
    3  IncompleteResumeError (torn ledger) -> RETRY (re-run fills the gap)
    75 GPU free-memory tempfail OR run-time CUDA OOM (contention) -> RETRY (wait for a free-GPU window)
    -N worker killed by signal N           -> RETRY (transient host event; see below)
    other positive                         -> FATAL (do not infinite-loop on the unknown)

    ``-N`` (signal-kill): on POSIX a NEGATIVE returncode means *exclusively* "the
    child was terminated by signal N" — it is never "the subprocess never started"
    (a failed exec returns a POSITIVE 127; a missing interpreter raises
    FileNotFoundError in the launcher before any returncode exists). So ``-9`` is
    always "the worker was SIGKILLed" — on a contended shared host (the exact
    scenario this launcher exists for) that is the OOM-killer terminating the
    5.5 GiB+ 9B worker, a TRANSIENT event where the ``--ledger`` has banked arms
    and a re-fire on the next window completes the run. Treating it as FATAL
    strands the multi-hour run on one contended-host kill — the same class as the
    run-time-OOM-as-FATAL bug the ``is_cuda_oom`` tempfail (4afc5e9) closed. The
    retry is bounded by ``--max-attempts`` / ``--deadline-seconds`` so a genuinely
    broken worker surfaces ``RETRIES_EXHAUSTED`` instead of spinning forever.

Run detached, e.g.::

    nohup env PYTHON_VENV=/torch/venv/bin/python \\
        python -m scripts.launch_freeze_ci_9b_full -- \\
        --seq-len 1024 --total-steps 1500 ... --ledger runs/full_ledger.jsonl \\
        --json --output tests/fixtures/freeze_validloss_ci_9b_full.json \\
        >runs/full_launcher.log 2>&1 &

or via the ``freeze-validloss-ci-9b-full-bg`` Make target.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

logger = logging.getLogger(__name__)

# The worker module run as a subprocess (NOT via make, so exit codes survive).
MODULE = "scripts.run_freeze_validloss_ci_9b"

# Worker exit-code contract — mirrored here as named constants so the launcher's
# branch logic reads by intent, not by magic number. See the docstring above.
EXIT_DONE = 0
EXIT_UNEXPECTED = 1
EXIT_CUDA_DOWN = 2
EXIT_INCOMPLETE_RESUME = 3
EXIT_GPU_TEMPFAIL = 75  # sysexits.h EX_TEMPFAIL — re-exported for clarity


class Action(str, Enum):
    """What the launcher does after one worker invocation."""

    DONE = "done"      # worker succeeded — stop, deposit is written
    RETRY = "retry"    # worker deferred (tempfail / torn ledger) — back off, re-run
    FATAL = "fatal"    # worker hit a non-retriable error — stop, surface the code


class Outcome(str, Enum):
    """Why the loop terminated."""

    DONE = "done"                                  # worker returned 0
    FATAL = "fatal"                                # worker returned 1/2/unknown
    RETRIES_EXHAUSTED = "retries_exhausted"        # hit --max-attempts while retriable
    DEADLINE_EXCEEDED = "deadline_exceeded"        # hit --deadline-seconds while retriable


@dataclass(frozen=True)
class Decision:
    action: Action
    kind: str            # success | tempfail | resume | cuda_down | unexpected | unknown
    sleep_seconds: float


@dataclass(frozen=True)
class LoopResult:
    outcome: Outcome
    last_code: int
    attempts: int


def classify_exit_code(code: int, *, resume_sleep: float, tempfail_sleep: float) -> Decision:
    """Map a worker exit code to a retry decision.

    Pure: no GPU, no torch, no subprocess — the entire branch policy is
    expressed here so it can be unit-pinned. ``resume_sleep`` (exit 3, re-run
    fills the gap *now*) is deliberately shorter than ``tempfail_sleep`` (exit
    75, wait for a *free-GPU window*).

    A NEGATIVE ``code`` is a signal-kill (POSIX: the child was terminated by
    signal ``-code``; e.g. ``-9`` = SIGKILL). On a contended shared host that is
    the OOM-killer reclaiming the 9B worker mid-window — a *transient* event
    where the ``--ledger`` has banked arms and a re-fire completes the run — so
    it classifies as RETRY with ``tempfail_sleep`` (wait for a stable /
    free-GPU window), the same branch exit 75 takes. A POSITIVE code outside the
    contract (e.g. 127 = exec-failure) stays FATAL — it is not a host-imposed
    kill and retrying it would loop on a non-transient failure. See the module
    docstring for the full rationale (sibling of the 4afc5e9 run-time-OOM fix).
    """
    if code == EXIT_DONE:
        return Decision(Action.DONE, "success", 0.0)
    if code == EXIT_GPU_TEMPFAIL:
        return Decision(Action.RETRY, "tempfail", tempfail_sleep)
    if code == EXIT_INCOMPLETE_RESUME:
        return Decision(Action.RETRY, "resume", resume_sleep)
    if code == EXIT_CUDA_DOWN:
        return Decision(Action.FATAL, "cuda_down", 0.0)
    if code == EXIT_UNEXPECTED:
        return Decision(Action.FATAL, "unexpected", 0.0)
    if code < 0:
        # Signal-kill: transient host event (OOM-killer / host SIGTERM), NOT a
        # logic error — retry the next window; the --ledger banks progress. The
        # bound (max_attempts / deadline_seconds) prevents infinite spinning on
        # a genuinely-broken worker.
        return Decision(Action.RETRY, "signal", tempfail_sleep)
    return Decision(Action.FATAL, "unknown", 0.0)


def run_loop(
    interpreter: str,
    module_argv: list[str],
    *,
    max_attempts: int,
    deadline_seconds: float | None,
    resume_sleep: float,
    tempfail_sleep: float,
    sleep_fn=None,
    now_fn=None,
    runner=None,
) -> LoopResult:
    """Invoke the worker module directly until it succeeds, dies, or a cap fires.

    Defaults for ``sleep_fn`` / ``now_fn`` / ``runner`` are resolved *lazily*
    (looked up on the ``time`` / ``subprocess`` modules at call time) so tests
    can monkeypatch ``launcher.time.sleep`` etc. and have main()'s path see the
    patch — capturing the real callables as signature defaults would freeze them
    at import time and defeat that.
    """
    if sleep_fn is None:
        sleep_fn = time.sleep
    if now_fn is None:
        now_fn = time.monotonic
    if runner is None:
        runner = subprocess.run

    start = now_fn()
    attempts = 0
    last_code = -1
    while True:
        attempts += 1
        argv = [interpreter, "-m", MODULE, *module_argv]
        logger.info("attempt %d: launching %s", attempts, argv)
        proc = runner(argv)
        last_code = int(getattr(proc, "returncode", 1))
        decision = classify_exit_code(
            last_code, resume_sleep=resume_sleep, tempfail_sleep=tempfail_sleep
        )
        logger.info(
            "attempt %d: worker exit %d -> %s (%s)",
            attempts, last_code, decision.action.value, decision.kind,
        )

        if decision.action is Action.DONE:
            return LoopResult(Outcome.DONE, last_code, attempts)
        if decision.action is Action.FATAL:
            logger.error(
                "worker exit %d is fatal (%s) — stopping. Completed arms stay "
                "banked in the --ledger; investigate and re-run.",
                last_code, decision.kind,
            )
            return LoopResult(Outcome.FATAL, last_code, attempts)

        # Retriable (75 tempfail / 3 resume). Bound the loop before sleeping.
        if attempts >= max_attempts:
            logger.warning(
                "retries exhausted at %d attempts (worker still exit %d). "
                "Re-run this launcher on the next free-GPU window; the --ledger "
                "keeps what completed.", attempts, last_code,
            )
            return LoopResult(Outcome.RETRIES_EXHAUSTED, last_code, attempts)
        if deadline_seconds and (now_fn() - start) >= deadline_seconds:
            logger.warning(
                "deadline (%.0fs) exceeded after %d attempts (worker still exit "
                "%d). Re-run on the next free-GPU window; the --ledger keeps what "
                "completed.", deadline_seconds, attempts, last_code,
            )
            return LoopResult(Outcome.DEADLINE_EXCEEDED, last_code, attempts)

        logger.info(
            "backing off %.0fs (%s) before re-running — ledger banks progress",
            decision.sleep_seconds, decision.kind,
        )
        sleep_fn(decision.sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="launch_freeze_ci_9b_full",
        description=(
            "Self-retrying launcher for the full-budget 9B §4 verdict. Invokes "
            "scripts.run_freeze_validloss_ci_9b directly (bypassing make so exit "
            "codes survive), retries GPU-tempfail (75) and torn-ledger (3), "
            "stops on success (0) or a fatal (1/2). Pass the worker's own flags "
            "after `--`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--python", default=os.environ.get("PYTHON_VENV", sys.executable),
        help="interpreter for the WORKER subprocess (default: $PYTHON_VENV, else sys.executable).",
    )
    p.add_argument(
        "--max-attempts", type=int, default=600,
        help="cap on worker invocations — bounds the loop so a persistently-held GPU cannot spin forever.",
    )
    p.add_argument(
        "--deadline-seconds", type=float, default=12.0 * 3600,
        help="wall-clock cap measured from launch. 0 disables.",
    )
    p.add_argument(
        "--tempfail-sleep", type=float, default=120.0,
        help="backoff seconds after a GPU-tempfail (exit 75) — a busy card, retry later.",
    )
    p.add_argument(
        "--resume-sleep", type=float, default=30.0,
        help="backoff seconds after an incomplete-resume (exit 3) — re-run fills the gap now.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the assembled worker command and exit 0 without launching.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s launch_freeze_ci_9b_full %(levelname)s %(message)s",
    )
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # Split the launcher's own options from the worker's flags at the first `--`.
    # argparse REMAINDER is finicky with flag-shaped positionals; splitting on the
    # explicit `--` separator is robust and lets the worker flags pass through
    # verbatim (including their own `--ledger`, `--json`, `--output`, ...).
    if "--" in argv:
        sep = argv.index("--")
        launcher_argv, module_argv = argv[:sep], argv[sep + 1:]
    else:
        launcher_argv, module_argv = argv, []

    args = build_parser().parse_args(launcher_argv)
    module_argv = [str(x) for x in module_argv]
    if not module_argv:
        # No worker flags = a no-op that would launch the worker with all defaults
        # (20-step reduced-budget run, NOT the full verdict). Refuse rather than
        # silently run the wrong experiment.
        build_parser().error(
            "no worker flags after `--` — pass the full-budget flags, e.g. "
            "launch_freeze_ci_9b_full -- --seq-len 1024 --total-steps 1500 ... "
            "--ledger runs/full_ledger.jsonl --json --output OUT.json"
        )

    worker_cmd = [args.python, "-m", MODULE, *module_argv]
    if args.dry_run:
        print("DRY-RUN worker command: " + " ".join(worker_cmd))
        print(
            f"retry policy: max_attempts={args.max_attempts}, "
            f"deadline={args.deadline_seconds:.0f}s, "
            f"tempfail_sleep={args.tempfail_sleep}s, resume_sleep={args.resume_sleep}s"
        )
        return 0

    logger.info("launching full-budget 9B verdict; worker=%s", worker_cmd)
    result = run_loop(
        args.python,
        module_argv,
        max_attempts=args.max_attempts,
        deadline_seconds=args.deadline_seconds or None,
        resume_sleep=args.resume_sleep,
        tempfail_sleep=args.tempfail_sleep,
    )
    logger.info(
        "loop ended: outcome=%s last_worker_exit=%d attempts=%d",
        result.outcome.value, result.last_code, result.attempts,
    )
    # Mirror the worker's terminal exit code: 0 on DONE; the worker's own 1/2 on a
    # fatal; 75 (tempfail) if the GPU never freed within the caps — the honest
    # "still not done, retry me later" signal for an outer scheduler.
    return result.last_code


if __name__ == "__main__":
    sys.exit(main())
