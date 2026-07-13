"""GPU-free tests for the free-memory pre-flight guard in
``scripts/run_freeze_validloss_ci_9b.py``.

The full-budget §4 verdict (``make freeze-validloss-ci-9b-full``) is hours of 9B
GPU on a shared 12 GB card that is routinely preempted by concurrent runs. The
resumable ``--ledger`` (TASK-0158) banks each arm as it completes, but without
this guard a launch onto an already-busy card still OOM-crashes minutes into the
FIRST arm before any arm completes — so the ledger never gets a chance to bank
anything and the workflow burns a full OOM traceback each poll. The guard reads
``torch.cuda.mem_get_info`` once up front and, when another big process is
clearly holding the card, exits ``EXIT_GPU_TEMPFAIL`` (75, EX_TEMPFAIL — "retry
later") so a poll-loop / resumable-ledger workflow simply re-runs on the next
free-GPU window instead of churning OOM crashes.

These tests pin the guard's pure logic without CUDA: the fail-open contract
(unreadable memory / floor disabled → defer None), the floor semantics (enough
free → None; below floor → reason citing exit 75), the exported constants that
the poll-loop contract depends on, and the main() integration — on a busy card
main() returns 75 and never reaches ``run_ci_9b`` (so no model load, no OOM).

They also pin the sibling output-writability guard (``output_paths_writable``),
which catches the *dead-CWD trap*: a run launched from a worktree that has since
been removed trains fine but its relative ``--output``/``--ledger`` resolve to a
deleted directory, so the deposit never lands. That guard is FATAL (exit 1) and
checked before the GPU gates so it is never mislabeled as the retryable tempfail
(75) the launcher would loop on forever.
"""

from __future__ import annotations

import pytest
import torch

import scripts.run_freeze_validloss_ci_9b as mod
from scripts.run_freeze_validloss_ci_9b import (
    DEFAULT_MIN_FREE_GIB,
    EXIT_GPU_TEMPFAIL,
    gpu_free_mib,
    gpu_free_memory_deferred,
    output_paths_writable,
)

_GIB = 1024 * 1024 * 1024
_MIB = 1024 * 1024


def _mem(monkeypatch, *, available: bool, free_gib: float | None):
    """Stub torch.cuda's availability + mem_get_info on the real torch module.

    Patches at ``torch.cuda`` (where the production code looks the attributes
    up), so the module's own ``import torch`` binding sees the stubs too.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    if free_gib is None:
        def _boom():
            raise RuntimeError("simulated unreadable mem_get_info")
        monkeypatch.setattr(torch.cuda, "mem_get_info", _boom)
    else:
        free_bytes = int(free_gib * _GIB)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free_bytes, 24 * _GIB))


class TestConstants:
    def test_exit_code_is_sysexits_tempfail(self):
        # sysexits.h EX_TEMPFAIL == 75 ("temp failure; retry later"). A polling
        # loop / make-target branches on this exact value — pin it.
        assert EXIT_GPU_TEMPFAIL == 75

    def test_default_floor_is_calibrated(self):
        # 10 GiB: sits below the seq1024 suffix-only peak (~11.2 GB, probe
        # da4fa4f) so it does NOT spuriously fire on an idle 12 GB card, but
        # well above the ~4 GB free when a concurrent ~8 GB run holds it.
        assert DEFAULT_MIN_FREE_GIB == 10.0


class TestGpuFreeMib:
    def test_returns_free_mib_when_available(self, monkeypatch):
        _mem(monkeypatch, available=True, free_gib=10.0)
        assert gpu_free_mib() == int(10.0 * 1024)

    def test_none_when_cuda_unavailable(self, monkeypatch):
        _mem(monkeypatch, available=False, free_gib=10.0)
        assert gpu_free_mib() is None

    def test_none_when_mem_get_info_raises(self, monkeypatch):
        # Fail open on cards/drivers where mem_get_info is unavailable.
        _mem(monkeypatch, available=True, free_gib=None)
        assert gpu_free_mib() is None


class TestGpuFreeMemoryDeferred:
    def test_disabled_floor_never_defers(self, monkeypatch):
        _mem(monkeypatch, available=True, free_gib=0.0)
        # --min-free-gib 0 is the explicit opt-out.
        assert gpu_free_memory_deferred(0.0) is None

    def test_enough_free_proceeds(self, monkeypatch):
        _mem(monkeypatch, available=True, free_gib=12.0)
        assert gpu_free_memory_deferred(10.0) is None

    def test_below_floor_defers_with_reason(self, monkeypatch):
        _mem(monkeypatch, available=True, free_gib=4.0)
        reason = gpu_free_memory_deferred(10.0)
        assert reason is not None
        # The reason must name the floor, the measured free, and the exit code
        # so the polling operator / make-target knows it is a retryable tempfail.
        assert "4096 MiB" in reason       # 4 GiB free, in MiB
        assert "Insufficient free GPU memory" in reason
        assert str(EXIT_GPU_TEMPFAIL) in reason
        assert "retry" in reason.lower()

    def test_boundary_exact_floor_proceeds(self, monkeypatch):
        # free == floor (10.0 GiB) is NOT below → proceed (strict <).
        _mem(monkeypatch, available=True, free_gib=10.0)
        assert gpu_free_memory_deferred(10.0) is None

    def test_fail_open_when_free_unreadable(self, monkeypatch):
        # If we cannot read free memory we must NOT defer — the guard exists to
        # spare OOM churn, not to block runs on cards where the metric is
        # unavailable.
        _mem(monkeypatch, available=True, free_gib=None)
        assert gpu_free_memory_deferred(10.0) is None


class TestMainIntegration:
    def test_busy_gpu_returns_tempfail_without_loading_model(self, monkeypatch):
        """The critical path: on a busy card main() exits 75 BEFORE the config
        is loaded or the model arm is trained — so no OOM crash, no wasted
        arm. A poll-loop re-runs and the --ledger keeps what it banked."""
        _mem(monkeypatch, available=True, free_gib=4.0)

        called = {"run_ci_9b": False}

        def _fail_if_called(**kw):  # noqa: ARG001
            called["run_ci_9b"] = True
            raise AssertionError("run_ci_9b must not be reached on a deferred GPU")

        # Also prove the guard fires before OmegaConf.load — patch it to fail
        # too, so the test cannot pass by loading a real config and reaching
        # run_ci_9b through the happy path.
        def _fail_if_loaded(*a, **k):  # noqa: ARG001
            raise AssertionError("OmegaConf.load must not be reached on a deferred GPU")

        monkeypatch.setattr(mod, "run_ci_9b", _fail_if_called)
        monkeypatch.setattr(mod.OmegaConf, "load", _fail_if_loaded)

        rc = mod.main(["--min-free-gib", "10"])

        assert rc == EXIT_GPU_TEMPFAIL
        assert called["run_ci_9b"] is False

    def test_no_cuda_still_returns_two_before_floor_check(self, monkeypatch):
        # The pre-existing "no CUDA" branch (exit 2) must take precedence over
        # the floor check — a CPU-only host reports no free GPU memory either
        # way, and the distinct exit code tells the operator the problem is
        # "no GPU", not "busy GPU".
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # mem_get_info would raise on a CPU host; ensure the floor check is
        # never reached (else it would fail-open and mask the no-CUDA exit 2).
        def _boom():
            raise AssertionError("mem_get_info must not be called when CUDA is absent")
        monkeypatch.setattr(torch.cuda, "mem_get_info", _boom)
        assert mod.main(["--min-free-gib", "10"]) == 2


class TestOutputPathsWritable:
    """The dead-CWD trap: a run launched from a since-removed worktree cannot
    write its relative --output/--ledger, so the deposit never lands. The guard
    refuses to burn GPU up front. ``Path.is_dir`` / ``os.access`` return False on
    a removed CWD (ENOENT is swallowed), so a missing-parent path exercises the
    exact branch a dead CWD hits — these tests pin that branch without rmdir'ing
    the process CWD (which would leave the test process in a broken state)."""

    def test_no_output_no_ledger_is_a_noop(self):
        # main() with neither flag (the existing pre-flight tests) must be
        # unaffected — the guard only fires on a given path.
        assert output_paths_writable(None, None) is None

    def test_writable_output_and_ledger_proceed(self, tmp_path):
        assert output_paths_writable(
            str(tmp_path / "out.json"), str(tmp_path / "ledger.jsonl")
        ) is None

    def test_bare_filename_in_live_cwd_proceeds(self, tmp_path, monkeypatch):
        # A bare filename's parent is "." — must read as the (live) CWD and pass.
        monkeypatch.chdir(tmp_path)
        assert output_paths_writable("out.json", None) is None

    def test_missing_output_parent_is_fatal(self, tmp_path):
        reason = output_paths_writable(
            str(tmp_path / "no_such_dir" / "out.json"), None
        )
        assert reason is not None
        assert "not writable" in reason
        assert "dead-CWD" in reason
        assert "fatal" in reason
        # The offending flag + path must be named so the operator can act.
        assert "--output" in reason

    def test_missing_ledger_parent_is_fatal(self, tmp_path):
        reason = output_paths_writable(
            None, str(tmp_path / "no_such_dir" / "ledger.jsonl")
        )
        assert reason is not None
        assert "--ledger" in reason
        assert "dead-CWD" in reason

    def test_output_reported_when_both_broken(self, tmp_path):
        # Both broken: the first-checked flag (--output) is the one surfaced.
        reason = output_paths_writable(
            str(tmp_path / "missing_out" / "out.json"),
            str(tmp_path / "missing_ledger" / "ledger.jsonl"),
        )
        assert reason is not None
        assert "--output" in reason


class TestOutputGuardMainIntegration:
    def _boom_if(self, name):
        def _boom(*a, **k):  # noqa: ARG001
            raise AssertionError(f"{name} must not be reached on an unwritable output")
        return _boom

    def test_dead_cwd_aborts_before_training(self, monkeypatch, tmp_path):
        """The dead-CWD promise: a removed worktree is FATAL (exit 1) BEFORE the
        config is loaded or an arm trained — so no GPU is burned on a result that
        can't be persisted. ``cwd_is_alive()`` is the robust signal (a recreated
        ``runs/`` subdir would fool ``parent.is_dir()``, but the CWD path itself
        is never recreated); stubbed False here to exercise the exact branch a
        host that deleted the worktree hits. The GPU stubs are 'fine', so the
        only thing that can stop the run is the dead-CWD gate."""
        _mem(monkeypatch, available=True, free_gib=12.0)

        monkeypatch.setattr(mod, "cwd_is_alive", lambda: False)
        monkeypatch.setattr(mod, "run_ci_9b", self._boom_if("run_ci_9b"))
        monkeypatch.setattr(mod.OmegaConf, "load", self._boom_if("OmegaConf.load"))

        rc = mod.main([
            "--min-free-gib", "0",
            "--output", str(tmp_path / "out.json"),
        ])

        assert rc == 1  # FATAL, not the retryable tempfail (75)

    def test_dead_cwd_not_masked_by_busy_gpu(self, monkeypatch, tmp_path):
        """Precedence (the load-bearing one): a dead CWD is FATAL (1) EVEN on a
        busy card. It must NOT surface as the retryable tempfail (75), or the
        self-retrying launcher would re-spawn the worker into the same dead CWD
        forever. The dead-CWD gate therefore runs before the GPU-memory gate."""
        _mem(monkeypatch, available=True, free_gib=4.0)  # below the 10 GiB floor

        monkeypatch.setattr(mod, "cwd_is_alive", lambda: False)
        monkeypatch.setattr(mod, "run_ci_9b", self._boom_if("run_ci_9b"))
        monkeypatch.setattr(mod.OmegaConf, "load", self._boom_if("OmegaConf.load"))

        rc = mod.main([
            "--min-free-gib", "10",  # the floor WOULD defer (75) if reached first
            "--output", str(tmp_path / "out.json"),
        ])

        assert rc == 1
        assert rc != EXIT_GPU_TEMPFAIL

    def test_live_cwd_missing_runs_dir_is_created_not_fatal(self, monkeypatch, tmp_path):
        """A fresh worktree (live CWD, no ``runs/``) must NOT be mistaken for the
        dead-CWD trap: ``append_arm_to_ledger`` already ``mkdir()``s at write
        time, so a missing parent is a trivial setup step the pre-flight now
        performs. This is the fix for the fresh-worktree FATAL that blocked
        ``make freeze-validloss-ci-9b-full-bg`` outright (the worker aborted on a
        ``runs/`` it would have created anyway). Proves main() reaches
        ``run_ci_9b`` (past every pre-flight) and created the dir."""
        _mem(monkeypatch, available=True, free_gib=12.0)
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "runs").exists()

        class _GotPastPreflight(RuntimeError):
            pass

        def _past(**kw):  # noqa: ARG001
            raise _GotPastPreflight("reached run_ci_9b — pre-flight passed")

        monkeypatch.setattr(mod, "run_ci_9b", _past)
        # Config lives under the repo root, not tmp_path; the pre-flight under
        # test is upstream of the load, so stub the load to a dummy cfg.
        monkeypatch.setattr(mod.OmegaConf, "load", lambda *a, **k: {})  # noqa: ARG001

        with pytest.raises(_GotPastPreflight):
            mod.main([
                "--min-free-gib", "0",
                "--ledger", "runs/sub/ledger.jsonl",
                "--output", "runs/sub/out.json",
            ])

        # The fresh-worktree fix: the parent dirs were created, not rejected.
        assert (tmp_path / "runs" / "sub").is_dir()


class TestCwdIsAlive:
    """``cwd_is_alive`` is the robust mid-run dead-CWD signal: ``Path.cwd()``
    raises once the worktree is unlinked, and a recreated ``runs/`` subdir cannot
    fool it (it checks the CWD path itself). Pinned live/deleted so the per-arm
    and startup gates can't silently degrade to always-True."""

    def test_true_on_live_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert mod.cwd_is_alive() is True

    def test_false_after_cwd_removed(self, tmp_path, monkeypatch):
        # Sit in a subdir of tmp_path, unlink it, assert the death is detected —
        # without leaving the test process in a broken CWD (monkeypatch restores
        # the original live CWD on teardown).
        nest = tmp_path / "nest"
        nest.mkdir()
        monkeypatch.chdir(nest)
        nest.rmdir()
        assert mod.cwd_is_alive() is False


class TestOutputsAreAbsolute:
    """The escape hatch: absolute ``--output``/``--ledger`` survive the run's CWD
    being removed (writes never resolve against CWD), so the per-arm dead-CWD
    check is skipped for them — the robust fire for a worktree-recycling host."""

    def test_none_is_unconstrained(self):
        assert mod._outputs_are_absolute(None, None) is True

    def test_relative_output_is_false(self):
        assert mod._outputs_are_absolute("runs/out.json", None) is False

    def test_relative_ledger_is_false(self):
        assert mod._outputs_are_absolute(None, "runs/x.jsonl") is False

    def test_absolute_is_true(self):
        assert mod._outputs_are_absolute("/abs/out.json", "/abs/x.jsonl") is True

    def test_mixed_absolute_and_relative_is_false(self):
        # One relative path makes the whole run CWD-dependent.
        assert mod._outputs_are_absolute("/abs/out.json", "runs/x.jsonl") is False


class TestPerArmDeadCwdCheck:
    """The mid-run dead-CWD trap: a worktree removed BETWEEN arms is caught at
    the next arm boundary for relative paths (bounding wasted GPU to <=1 arm),
    and skipped for absolute paths (they outlive the CWD). Exercises
    ``_collect_arms`` directly with a stub runner — no GPU, no model."""

    @staticmethod
    def _spec(role, index):
        return {"role": role, "index": index}

    def test_relative_paths_raise_before_first_arm(self, monkeypatch):
        # CWD dies before any arm runs: the check fires at the first non-cached
        # arm, so the GPU runner is never reached.
        monkeypatch.setattr(mod, "cwd_is_alive", lambda: False)
        monkeypatch.setattr(mod, "load_ledger", lambda path, fp: {})  # noqa: ARG001

        def _boom(spec):  # noqa: ARG001
            raise AssertionError("runner must not be reached; cwd died before arm 0")

        with pytest.raises(mod.OutputPathDiedDuringRun):
            mod._collect_arms(
                [self._spec("candidate", 0), self._spec("candidate", 1)],
                _boom,
                ledger_path="runs/x.jsonl",
                output="runs/out.json",
            )

    def test_absolute_output_survives_cwd_death(self, monkeypatch):
        # Absolute --output: writes don't depend on CWD, so a dead CWD is
        # harmless and the run proceeds (per-arm check skipped).
        monkeypatch.setattr(mod, "cwd_is_alive", lambda: False)
        specs = [self._spec("candidate", 0), self._spec("surrogate", 0)]

        collected, n_resumed = mod._collect_arms(
            specs, lambda s: (0.5, {}), output="/abs/out.json"
        )  # ledger_path=None -> no load/append

        assert len(collected["candidate"]) == 1
        assert len(collected["surrogate"]) == 1
        assert n_resumed == 0

    def test_live_cwd_never_raises_even_for_relative(self, monkeypatch):
        # CWD alive -> the per-arm check is a no-op; relative paths are fine.
        monkeypatch.setattr(mod, "cwd_is_alive", lambda: True)
        monkeypatch.setattr(mod, "load_ledger", lambda path, fp: {})  # noqa: ARG001
        monkeypatch.setattr(mod, "append_arm_to_ledger", lambda *a, **k: None)  # noqa: ARG001

        collected, _ = mod._collect_arms(
            [self._spec("candidate", 0)],
            lambda s: (0.5, {}),
            ledger_path="runs/x.jsonl",
            output="runs/out.json",
        )
        assert len(collected["candidate"]) == 1
