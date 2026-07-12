"""GPU-free tests for the self-retrying full-budget 9B launcher.

``make freeze-validloss-ci-9b-full`` is the path to the first
``citable_as_full_section4_verdict=True`` deposit, but it is hours of 9B GPU on a
shared 12 GB card that a concurrent run routinely preempts. The worker script
already defers a held card with exit 75 (EX_TEMPFAIL) and banks completed arms in
the resumable ``--ledger``. The gap this launcher closes is documented in the
Makefile: GNU make flattens EVERY non-zero recipe exit to make-exit 2, so a poll
loop around ``make`` cannot tell tempfail (75, retry) from CUDA-down (2, fatal)
or torn-ledger (3, re-run). ``scripts/launch_freeze_ci_9b_full.py`` bypasses make
for the worker step — it subprocess-invokes the module directly so the exit codes
survive, retries 75/3 with backoff, stops on 0/1/2, and lets the ledger bank arms
across free-GPU windows.

These tests pin the pure decision logic (``classify_exit_code``) and the retry
loop (``run_loop``) with an injected fake runner / clock / sleep — no GPU, no
torch, no subprocess. ``main``'s ``--dry-run`` is exercised to prove the worker
command is assembled from PYTHON_VENV without launching anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import scripts.launch_freeze_ci_9b_full as launcher
from scripts.launch_freeze_ci_9b_full import (
    Action,
    Outcome,
    classify_exit_code,
    main,
    run_loop,
)


def _proc(code: int) -> SimpleNamespace:
    """A minimal stand-in for subprocess.CompletedProcess."""
    return SimpleNamespace(returncode=code)


class _FakeRunner:
    """Pops successive return codes; records every invocation."""

    def __init__(self, codes: list[int]):
        self._codes = list(codes)
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # noqa: ARG002
        self.calls.append(list(argv))
        if not self._codes:
            raise AssertionError("fake runner exhausted — loop did not stop")
        return _proc(self._codes.pop(0))


class _Clock:
    """A controllable monotonic clock for deadline tests."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestClassifyExitCode:
    def test_done(self):
        d = classify_exit_code(0, resume_sleep=30, tempfail_sleep=120)
        assert d.action is Action.DONE
        assert d.kind == "success"
        assert d.sleep_seconds == 0.0

    def test_tempfail_is_retriable_with_tempfail_sleep(self):
        d = classify_exit_code(75, resume_sleep=30, tempfail_sleep=120)
        assert d.action is Action.RETRY
        assert d.kind == "tempfail"
        assert d.sleep_seconds == 120

    def test_incomplete_resume_is_retriable_with_resume_sleep(self):
        # exit 3 (IncompleteResumeError) — re-run fills the gap, not "wait for GPU",
        # so it uses the short resume sleep, not the long tempfail sleep.
        d = classify_exit_code(3, resume_sleep=30, tempfail_sleep=120)
        assert d.action is Action.RETRY
        assert d.kind == "resume"
        assert d.sleep_seconds == 30

    def test_cuda_down_is_fatal(self):
        d = classify_exit_code(2, resume_sleep=30, tempfail_sleep=120)
        assert d.action is Action.FATAL
        assert d.kind == "cuda_down"
        assert d.sleep_seconds == 0.0

    def test_unexpected_is_fatal(self):
        d = classify_exit_code(1, resume_sleep=30, tempfail_sleep=120)
        assert d.action is Action.FATAL
        assert d.kind == "unexpected"

    def test_unknown_nonzero_is_fatal(self):
        # An exit code outside the documented contract must NOT be silently retried
        # (that would infinite-loop on an unforeseen failure mode).
        d = classify_exit_code(99, resume_sleep=30, tempfail_sleep=120)
        assert d.action is Action.FATAL
        assert d.kind == "unknown"

    def test_negative_signal_is_fatal(self):
        # A subprocess that never started (returncode -N) is not a tempfail.
        d = classify_exit_code(-9, resume_sleep=30, tempfail_sleep=120)
        assert d.action is Action.FATAL


class TestRunLoop:
    def _loop(self, runner, *, clock=None, sleep=None, **kw):
        defaults = dict(
            interpreter="/fake/python",
            module_argv=["--seq-len", "1024"],
            max_attempts=10,
            deadline_seconds=None,
            resume_sleep=30,
            tempfail_sleep=120,
        )
        defaults.update(kw)
        return run_loop(sleep_fn=sleep or (lambda _s: None), now_fn=clock or _Clock(), runner=runner, **defaults)

    def test_immediate_done_no_sleep(self):
        runner = _FakeRunner([0])
        sleeps: list[float] = []
        result = self._loop(runner, sleep=sleeps.append)
        assert result.outcome is Outcome.DONE
        assert result.last_code == 0
        assert result.attempts == 1
        assert sleeps == []  # success on first try — never sleeps
        assert len(runner.calls) == 1

    def test_tempfail_then_done(self):
        runner = _FakeRunner([75, 0])
        sleeps: list[float] = []
        result = self._loop(runner, sleep=sleeps.append)
        assert result.outcome is Outcome.DONE
        assert result.attempts == 2
        assert sleeps == [120]  # one tempfail backoff before the success

    def test_resume_then_done(self):
        # exit 3 (torn ledger) uses the SHORT resume sleep, not the long tempfail one.
        runner = _FakeRunner([3, 0])
        sleeps: list[float] = []
        result = self._loop(runner, sleep=sleeps.append)
        assert result.outcome is Outcome.DONE
        assert result.attempts == 2
        assert sleeps == [30]

    def test_cuda_down_is_fatal_immediately(self):
        runner = _FakeRunner([2])
        sleeps: list[float] = []
        result = self._loop(runner, sleep=sleeps.append)
        assert result.outcome is Outcome.FATAL
        assert result.last_code == 2
        assert result.attempts == 1
        assert sleeps == []  # does NOT retry a CUDA-down — operator action needed
        assert len(runner.calls) == 1

    def test_unexpected_is_fatal_immediately(self):
        runner = _FakeRunner([1])
        result = self._loop(runner, sleep=lambda _s: None)
        assert result.outcome is Outcome.FATAL
        assert result.attempts == 1
        assert len(runner.calls) == 1

    def test_unknown_is_fatal_immediately(self):
        runner = _FakeRunner([42])
        result = self._loop(runner, sleep=lambda _s: None)
        assert result.outcome is Outcome.FATAL
        assert result.last_code == 42

    def test_retries_exhausted_under_persistent_tempfail(self):
        # GPU held by a concurrent run for the whole window: bound the loop so it
        # cannot spin forever. max_attempts=3 → 3 tries, 2 backoff sleeps.
        runner = _FakeRunner([75, 75, 75])
        sleeps: list[float] = []
        result = self._loop(runner, max_attempts=3, sleep=sleeps.append)
        assert result.outcome is Outcome.RETRIES_EXHAUSTED
        assert result.last_code == 75
        assert result.attempts == 3
        assert sleeps == [120, 120]

    def test_deadline_exceeded_under_persistent_tempfail(self):
        runner = _FakeRunner([75, 75, 75])
        clock = _Clock()
        sleeps: list[float] = []

        # A sleep that advances the clock past the deadline on the second backoff.
        def sleep_and_advance(s: float):
            sleeps.append(s)
            clock.advance(s)

        result = run_loop(
            interpreter="/fake/python",
            module_argv=["--seq-len", "1024"],
            max_attempts=10,
            deadline_seconds=240,
            resume_sleep=30,
            tempfail_sleep=120,
            sleep_fn=sleep_and_advance,
            now_fn=clock,
            runner=runner,
        )
        assert result.outcome is Outcome.DEADLINE_EXCEEDED
        assert result.last_code == 75
        # After the first 120s backoff the elapsed time (120) is still < 240, so it
        # retries; after the second 120s backoff elapsed (240) >= 240 → stop.
        assert sleeps == [120, 120]

    def test_worker_invoked_directly_not_via_make(self):
        # The whole point: the module is run as a subprocess under the interpreter
        # (`-m scripts.run_freeze_validloss_ci_9b`), never through `make`, so the
        # exit codes are not flattened to 2.
        runner = _FakeRunner([0])
        self._loop(runner, interpreter="/torch/venv/bin/python")
        argv = runner.calls[0]
        assert argv[0] == "/torch/venv/bin/python"
        assert argv[1] == "-m"
        assert argv[2] == "scripts.run_freeze_validloss_ci_9b"
        assert "make" not in argv

    def test_module_argv_passed_through_verbatim(self):
        runner = _FakeRunner([0])
        flags = ["--seq-len", "1024", "--total-steps", "1500", "--ledger", "runs/x.jsonl"]
        self._loop(runner, module_argv=flags)
        assert runner.calls[0][3:] == flags


class TestMainDryRun:
    def test_dry_run_assembles_command_from_python_venv_and_does_not_launch(self, monkeypatch, capsys):
        # PYTHON_VENV selects the worker interpreter; --dry-run must print the
        # exact worker command and return 0 WITHOUT spawning a subprocess.
        monkeypatch.setenv("PYTHON_VENV", "/torch/venv/bin/python")
        spawned: list[list[str]] = []
        monkeypatch.setattr(launcher.subprocess, "run", lambda argv, **kw: spawned.append(argv) or _proc(0))

        rc = main([
            "--dry-run",
            "--",  # everything after this is the worker's own flags
            "--seq-len", "1024",
            "--total-steps", "1500",
            "--ledger", "runs/x.jsonl",
        ])

        assert rc == 0
        assert spawned == []  # dry-run never launches
        out = capsys.readouterr().out
        assert "/torch/venv/bin/python" in out
        assert "-m" in out
        assert "scripts.run_freeze_validloss_ci_9b" in out
        assert "--total-steps" in out and "1500" in out
        assert "--ledger" in out and "runs/x.jsonl" in out

    def test_main_returns_zero_on_done(self, monkeypatch):
        monkeypatch.setenv("PYTHON_VENV", "/torch/venv/bin/python")
        monkeypatch.setattr(launcher.subprocess, "run", lambda argv, **kw: _proc(0))
        rc = main(["--max-attempts", "3", "--", "--seq-len", "1024"])
        assert rc == 0

    def test_main_mirrors_worker_exit_code_on_fatal(self, monkeypatch):
        # On a fatal worker exit the launcher surfaces the worker's own code (2 for
        # CUDA-down) so an outer scheduler can distinguish it from a tempfail.
        monkeypatch.setenv("PYTHON_VENV", "/torch/venv/bin/python")
        monkeypatch.setattr(launcher.subprocess, "run", lambda argv, **kw: _proc(2))
        rc = main(["--max-attempts", "3", "--", "--seq-len", "1024"])
        assert rc == 2

    def test_main_mirrors_tempfail_on_retries_exhausted(self, monkeypatch):
        # If the GPU never frees within the attempt budget, the launcher exits 75
        # itself — the honest "still not done, retry me later" signal for an outer
        # scheduler/cron, NOT a generic failure code.
        monkeypatch.setenv("PYTHON_VENV", "/torch/venv/bin/python")
        monkeypatch.setattr(launcher.subprocess, "run", lambda argv, **kw: _proc(75))
        monkeypatch.setattr(launcher.time, "sleep", lambda _s: None)
        rc = main(["--max-attempts", "2", "--tempfail-sleep", "0", "--", "--seq-len", "1024"])
        assert rc == 75

    def test_missing_worker_flags_errors_cleanly(self, monkeypatch, capsys):
        monkeypatch.setenv("PYTHON_VENV", "/torch/venv/bin/python")
        # argparse error() exits 2; no worker flags after `--` must be caught.
        try:
            main(["--dry-run", "--"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit for missing worker flags")
