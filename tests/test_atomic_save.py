"""SIGINT / SystemExit safety of the atomic-save helper.

The three site-level fault classes — ``TestAtomicCheckpointSave``
(``tests/test_checkpoint.py``), ``TestAtomicTrajectoryArtifactSave``
(``tests/test_trajectory_delta_artifact.py``), and
``TestAtomicPrefixFeatureCacheSave`` (``tests/test_prefix_feature_shard.py``) —
each prove a mid-commit ``OSError`` never leaves a torn destination: no partial
file published, the prior still-loadable file intact, the orphan temp cleaned.
They all inject ``OSError``, which is an ``Exception`` subclass.

``_atomic_torch_save``'s ``except BaseException`` clause exists for a *different*
interrupt class that ``except Exception`` would MISS: ``KeyboardInterrupt``
(SIGINT — Ctrl-C during a multi-hundred-MB save) and ``SystemExit``, both of
which inherit from ``BaseException`` directly, not ``Exception``. A future
maintainer narrowing the clause to ``except Exception`` (a routine "be more
specific" linter nudge) would silently drop orphan-temp cleanup on a Ctrl-C
landed mid-save — exactly the interrupt the run-feedback review named
("SIGINTs mid-checkpoint ... resumes to confirm the now-atomic checkpoints load
cleanly"). No site-level test would catch that regression, because every one of
them injects an ``Exception`` subclass.

This module pins that pathway directly at the helper, which is the SINGLE publish
point for every on-disk artifact (locked by
``test_no_bare_torch_save_in_src``). Because all five routed sites — the two
``training_state.pt`` writers, the trajectory-delta artifact writer, and both
prefix-feature cache writers — funnel through this one function, a direct test
here covers all of them at once and is stable across shifts in the resume /
training loop (the feedback itself notes the resume seam moved L1336 -> L1374,
so a loop-level SIGINT test there would be brittle to that drift).

Why not a literal 9B multi-seed SIGINT + resume run: this public mirror excludes
the private ``src.data`` pipeline, so ``train_tg_lora.py`` cannot start on real
data here (see ``test_resume_state_integration.py``'s import shim and the
known-unavailable ``src.data`` canary xfail in ``test_cli_help_smoke.py``). The
end-to-end "resume loads the atomic checkpoint cleanly through the real resume
path" proof already exists there
(``test_all_run_wide_state_survives_fault_resume``); this module adds the
SIGINT-specific leg of the atomic guarantee the feedback asked for, at the
strongest fidelity this mirror allows. The contract pinned here is what makes
that real resume safe to attempt after a Ctrl-C: a SIGINT mid-publish leaves the
destination either fully reflecting the new state or still at its prior,
loadable value — never torn — *and* removes the orphan temp a bare torch.save
would have left behind.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.utils.atomic_save import _atomic_torch_save


class TestAtomicSaveBaseExceptionInterrupts:
    """``KeyboardInterrupt`` (SIGINT) and ``SystemExit`` are ``BaseException``
    subclasses that the helper's ``except BaseException`` clause is the ONLY
    thing that catches. Pin that a mid-publish interrupt of this class:

      * leaves the destination intact (prior value still loadable, or, on a fresh
        save, no destination published at all),
      * cleans up the orphaned PID-suffixed temp (the load-bearing justification
        for ``BaseException`` over ``Exception`` — without it, every Ctrl-C
        during a long save litters the checkpoint dir with a ``.tmp.<pid>`` file),
      * re-raises the ORIGINAL interrupt type, so the process terminates on
        Ctrl-C rather than the interrupt being swallowed.

    A regression narrowing ``except BaseException`` -> ``except Exception`` keeps
    the destination safe (rename atomicity holds regardless of the handler) but
    DROPS the orphan-temp cleanup, so the ``glob`` assertions below are what fail
    first and loudest.
    """

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_prior_destination_survives_mid_publish_interrupt(
        self, tmp_path, monkeypatch, interrupt
    ):
        # Publish a valid v1 blob (real os.replace), then interrupt at the commit
        # boundary while overwriting it with a different v2.
        path = tmp_path / "artifact.pt"
        _atomic_torch_save({"v": 1}, path)
        assert path.exists()
        assert torch.load(path) == {"v": 1}

        def _boom(_src, _dst):
            raise interrupt("simulated interrupt mid-publish")

        monkeypatch.setattr(os, "replace", _boom)

        with pytest.raises(interrupt):
            _atomic_torch_save({"v": 2}, path)

        # The prior, still-loadable destination is intact with the OLD value —
        # the torn (v2) state was never published. A regressed ``except Exception``
        # would re-raise correctly (destination still safe via rename atomicity)
        # but LEAVE the orphan temp; the glob below is the assertion that
        # ``except BaseException`` specifically protects.
        assert torch.load(path) == {"v": 1}
        assert not list(tmp_path.glob("artifact.pt.tmp.*"))

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_fresh_save_mid_publish_interrupt_publishes_nothing(
        self, tmp_path, monkeypatch, interrupt
    ):
        def _boom(_src, _dst):
            raise interrupt("simulated interrupt mid-publish")

        monkeypatch.setattr(os, "replace", _boom)

        path = tmp_path / "fresh.pt"
        with pytest.raises(interrupt):
            _atomic_torch_save({"v": 1}, path)

        assert not path.exists()
        assert not list(tmp_path.glob("fresh.pt.tmp.*"))

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_interrupt_during_dump_cleans_orphan_temp(
        self, tmp_path, monkeypatch, interrupt
    ):
        # Interrupt torch.save itself mid-dump (before os.replace ever runs). A
        # bare ``torch.save(blob, path)`` to the destination would truncate the
        # real artifact here; routing through the helper means only the temp is
        # ever at risk, and the ``except BaseException`` branch removes even
        # that. The mocked torch.save mimics a partial write landing in the temp
        # before the interrupt fires, so the cleanup branch is observable.
        path = tmp_path / "dump.pt"

        def _partial_then_interrupt(_obj, tmp, *_args, **_kwargs):
            Path(tmp).write_bytes(b"\x80\x02PARTIAL")  # a torn dump, not valid pickle
            raise interrupt("simulated interrupt mid-dump")

        monkeypatch.setattr(torch, "save", _partial_then_interrupt)

        with pytest.raises(interrupt):
            _atomic_torch_save({"v": 2}, path)

        # The destination was never created, and the orphan temp the interrupted
        # dump left behind was cleaned up by the BaseException branch.
        assert not path.exists()
        assert not list(tmp_path.glob("dump.pt.tmp.*"))
