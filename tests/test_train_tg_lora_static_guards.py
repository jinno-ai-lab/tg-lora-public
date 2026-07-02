"""Static guard: the TG-LoRA training entry point must not reference undefined names.

Pins two previously-latent ``NameError`` defects in
``src/training/train_tg_lora.py`` that ``ruff`` surfaced as F821 (undefined name):

1. **Fault checkpoint lost dynfreeze state on OOM/CUDA fault.**
   ``_save_fault_checkpoint`` referenced ``dynfreeze`` without receiving it as a
   parameter (the controller lives in the caller's scope, not module globals).
   On a fault the ``TrainingState(...)`` construction raised ``NameError`` at the
   ``dynfreeze_state=...`` line, which the surrounding broad ``except Exception``
   swallowed and logged — so ``training_state.pt`` was *silently never written*
   and fault-resume lost cycle/velocity/delta_tracker/controller/dynfreeze state.
   This sat in configs that are actually used (``dynfreeze_enabled: true`` in
   ``configs/9b_tg_lora_m10_dynfreeze.yaml`` etc.); it only stayed hidden because
   faults are rare.

2. **Enabling progressive freeze crashed the loop instantly.**
   ``ProgressiveFreezeController`` was instantiated (≈line 1200) *before* its
   only import, which lived lazily inside the unrelated ``enable_psa`` block.
   Any run with ``progressive_freeze_enabled: true`` raised ``NameError`` before
   training began. It stayed latent because no committed config enables it and
   the progressive-freeze unit tests construct the controller directly rather
   than through ``train_tg_lora()``.

Both share one root cause — the freeze controllers were lazily imported only
inside conditionals — so the fix hoists them to module top and threads
``dynfreeze`` into ``_save_fault_checkpoint``. This test locks the entry point to
**zero F821** so neither defect (nor any new undefined-name) can return.

The guard runs ``ruff`` (the detector that surfaced both bugs) on the file via
subprocess. It deliberately does **not** import ``train_tg_lora``: that module's
line-1 ``from src.data.build_seed_dataset import load_dataset`` depends on the
private pipeline absent from this public mirror, so importing it here would both
fail and (once cached in ``sys.modules``) perturb the ~130 pre-existing
src.data-blocked tests. Running ``ruff`` on the file path sidesteps both.
Skips when ``ruff`` is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src" / "training" / "train_tg_lora.py"


def test_training_entry_point_has_no_undefined_names() -> None:
    """``train_tg_lora.py`` must be free of F821 (undefined-name) defects.

    Guards the dynfreeze fault-checkpoint NameError and the progressive-freeze
    enable NameError: if either fix regresses, the offending name reappears as
    F821 and this assertion fails.
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the F821 guard statically")

    assert TARGET.is_file(), f"training entry point not found at {TARGET}"

    proc = _run_ruff(TARGET, select="F821")
    assert proc.returncode == 0, (
        "src/training/train_tg_lora.py must contain zero undefined-name (F821) "
        "defects — a non-zero count signals a regression of the dynfreeze "
        "fault-checkpoint NameError or the progressive-freeze enable NameError. "
        "ruff output:\n" + (proc.stdout + proc.stderr).strip()
    )


def _run_ruff(target: Path, select: str | None) -> subprocess.CompletedProcess[str]:
    ruff = shutil.which("ruff")
    cmd = [ruff, "check"]
    if select is not None:
        cmd += ["--select", select]
    cmd.append(str(target))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_training_entry_point_is_ruff_clean() -> None:
    """``train_tg_lora.py`` must be ``ruff``-clean (zero findings, all rules).

    The F821 guard above pins the two historical ``NameError`` defects. This
    guard pins the *whole* file's cleanliness — every rule, not just F821 — so
    the audit's standing claim that the main training entry point is lint-clean
    (it was long described as carrying "2 pre-existing errors (F841 + E741)"
    that had in fact drifted to a single E741, with the F841 a write-only local
    pyflakes could no longer see once ``train_tg_lora`` started calling
    ``locals()``) is enforced by the test suite rather than by prose. A future
    regression — an ambiguous-name comprehension variable, an unused import, a
    re-introduced dead local — fails CI here instead of silently restating a
    stale lint count in PURPOSE.md.

    Like the F821 guard this runs ``ruff`` on the file path (the module's
    line-1 ``src.data`` import is unimportable in this public mirror) and skips
    when ``ruff`` is absent.
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the cleanliness guard")

    assert TARGET.is_file(), f"training entry point not found at {TARGET}"
    proc = _run_ruff(TARGET, select=None)
    assert proc.returncode == 0, (
        "src/training/train_tg_lora.py must be ruff-clean (zero findings across "
        "all rules) — a non-zero count regresses the lint-clean invariant and "
        "makes the audit's standing claim false. ruff output:\n"
        + (proc.stdout + proc.stderr).strip()
    )


def test_progressive_freeze_resume_state_is_wired() -> None:
    """The progressive-freeze resume-state fix must stay wired end-to-end.

    ``ProgressiveFreezeController`` was given ``state_dict`` /
    ``load_state_dict`` / ``refreeze_loaded_layers`` (tested directly in
    ``tests/test_progressive_freeze.py``) and ``TrainingState`` gained a
    ``progressive_freeze_state`` field (round-tripped in
    ``tests/test_checkpoint.py``). This guard pins the *wiring* in the
    src.data-blocked training entry point (unimportable here, so verified by
    source string rather than by running it): the fault + periodic save sites
    must serialize the frozen set, the fault call must thread the controller,
    and the resume path must restore + refreeze it. A future edit that drops any
    leg reopens the resume-state-loss gap (frozen layers silently re-train after
    a fault; run-footer provenance reports only post-fault freezes) and fails
    this assertion.
    """
    assert TARGET.is_file(), f"training entry point not found at {TARGET}"
    source = TARGET.read_text(encoding="utf-8")

    # The fault-checkpoint helper receives the controller as a parameter (it
    # lives in the caller's scope, like ``dynfreeze``).
    assert "progressive_freeze: ProgressiveFreezeController | None," in source, (
        "_save_fault_checkpoint must thread progressive_freeze as a parameter"
    )
    # Both save sites (fault + periodic) serialize the frozen set.
    assert source.count("progressive_freeze.state_dict()") == 2, (
        "both TrainingState save sites (fault + periodic) must serialize "
        "progressive_freeze.state_dict()"
    )
    # The fault call passes the controller through.
    assert "progressive_freeze=progressive_freeze," in source, (
        "the _save_fault_checkpoint call must pass progressive_freeze through"
    )
    # The resume path restores the set and re-applies requires_grad on the
    # freshly adapter-loaded model (safetensors does not carry requires_grad).
    assert "refreeze_loaded_layers(model)" in source, (
        "the resume path must call progressive_freeze.refreeze_loaded_layers"
    )
    assert "ts.progressive_freeze_state" in source, (
        "the resume path must read the persisted progressive_freeze_state"
    )
