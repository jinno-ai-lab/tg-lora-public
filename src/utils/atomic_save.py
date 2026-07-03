"""Atomic ``torch.save`` — the sole publish point for every on-disk artifact.

A bare ``torch.save(blob, path)`` truncates *path* before writing, so an
interruption mid-dump (OOM kill, SIGINT during a multi-hundred-MB state write)
leaves a half-written destination that breaks the next load — silently losing
the run (for ``training_state.pt``) or corrupting a large analysis artifact
(``trajectory_delta_artifacts/*.pt``) or a multi-GB prefix-feature cache that
is expensive to rebuild. That directly undermines the resume guarantee the
12-site persistence axis went to the trouble of capturing.

This module is deliberately a **zero-dependency leaf** (stdlib + torch only):
the data layer (:mod:`src.tg_lora.prefix_feature_cache`) and the analysis
artifact layer (:mod:`src.training.trajectory_delta_artifact`) must be able to
persist atomically WITHOUT dragging in the heavyweight training-resume stack
(:mod:`src.utils.checkpoint`, which imports the whole controller graph). The
training-resume module re-imports the helper from here.

The single ``torch.save`` call below is the **only** ``torch.save`` permitted
anywhere in ``src/`` — pinned by
``tests.test_src_static_guards.test_no_bare_torch_save_in_src``. Every on-disk
artifact that must survive an interruption routes through this helper so a
mid-commit fault can never leave a torn destination again.
"""

import os
from pathlib import Path

import torch


def _atomic_torch_save(blob, path: Path) -> None:
    """Persist *blob* to *path* atomically so a load never sees a partial file.

    Writing to a PID-suffixed temp file in the SAME directory and renaming it
    into place is atomic on POSIX (same filesystem), so the destination either
    fully reflects the new state or is left at its prior, still-loadable value —
    never torn. The temp name is PID-suffixed so a temp orphaned by a prior
    crashed run is not silently reused; any orphan is cleaned up on the failure
    path. ``os.replace`` is the sole publish point — it is the seam the
    interruption-injection tests monkeypatch to prove no partial destination
    survives a mid-commit fault.
    """
    path = Path(path)
    tmp_path = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        torch.save(blob, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        # Never publish a partial checkpoint: if the rename did not land, the
        # prior file (if any) must remain intact. Best-effort-remove the orphan
        # temp so a crashed run does not litter the checkpoint directory.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
