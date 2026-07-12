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
"""

from __future__ import annotations

import torch

import scripts.run_freeze_validloss_ci_9b as mod
from scripts.run_freeze_validloss_ci_9b import (
    DEFAULT_MIN_FREE_GIB,
    EXIT_GPU_TEMPFAIL,
    gpu_free_mib,
    gpu_free_memory_deferred,
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
