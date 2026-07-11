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


def test_regime_detector_resume_state_is_wired() -> None:
    """The PSA regime-detector resume-state fix must stay wired end-to-end.

    ``RegimeDetector`` was given ``state_dict`` / ``load_state_dict`` (tested
    directly in ``tests/test_regime.py``) and ``TrainingState`` gained a
    ``psa_regime_state`` field (round-tripped in ``tests/test_checkpoint.py``).
    This guard pins the *wiring* in the src.data-blocked training entry point
    (unimportable here, so verified by source string rather than by running it):
    the fault + periodic save sites must serialize the detector, the fault call
    must thread the detector, and the resume path must restore it. A future edit
    that drops any leg reopens the resume-state-loss gap (the per-cycle
    ``psa_regime_transitions``, persisted to ``run_metrics.jsonl``, resets to 0
    after a fault/periodic resume) and fails this assertion.
    """
    assert TARGET.is_file(), f"training entry point not found at {TARGET}"
    source = TARGET.read_text(encoding="utf-8")

    # The fault-checkpoint helper receives the detector as a parameter (it lives
    # in the caller's scope, like ``psa_prior``).
    assert "regime_detector: RegimeDetector | None," in source, (
        "_save_fault_checkpoint must thread regime_detector as a parameter"
    )
    # Both save sites (fault + periodic) serialize the detector.
    assert source.count("regime_detector.state_dict()") == 2, (
        "both TrainingState save sites (fault + periodic) must serialize "
        "regime_detector.state_dict()"
    )
    # The fault call passes the detector through.
    assert "regime_detector=regime_detector," in source, (
        "the _save_fault_checkpoint call must pass regime_detector through"
    )
    # The resume path restores the persisted regime state into the in-loop
    # detector constructed in the enable_psa block.
    assert "regime_detector.load_state_dict(restored_training_state.psa_regime_state)" in (
        source
    ), "the resume path must call regime_detector.load_state_dict with the persisted field"


def test_rollback_tolerance_gates_use_relative_metric() -> None:
    """The two in-loop ``rollback_tolerance`` gates must be RELATIVE, not absolute.

    ``rollback_tolerance`` (default ``0.005``) is canonically a *relative* 0.5%
    fraction — pinned magnitude-invariant for ``RandomWalkController.accept`` by
    ``tests/test_accept_property.py`` and for the helper by
    ``tests/test_relative_degradation.py``. Two sites in this entry point
    previously misused the same knob as an *absolute* additive margin
    (``loss_x <= loss_y + rollback_tolerance``), which made the pilot-overshoot
    trigger and the alpha line-search accept drift with the current loss level:
    over-tolerant when loss is small, under-tolerant when loss is large. They
    now route through the shared ``relative_degradation`` helper. This guard
    pins the wiring (unimportable here, so verified by source string): the
    helper must be imported, both call sites must use it, and the historical
    absolute ``+ rollback_tolerance`` patterns must be gone. A future edit that
    inlines the absolute comparison back reopens the scale-drift gap and fails
    this assertion.
    """
    assert TARGET.is_file(), f"training entry point not found at {TARGET}"
    source = TARGET.read_text(encoding="utf-8")

    # The shared helper is imported (alongside the controller it lives next to).
    assert (
        "from src.tg_lora.random_walk_controller import RandomWalkController, relative_degradation"
        in source
    ), "relative_degradation must be imported from random_walk_controller"

    # Both in-loop gates route through the helper against the same tolerance.
    assert (
        source.count("relative_degradation(cycle_state.last_valid_loss, loss_pilot)") >= 1
    ), "the pilot-overshoot trigger must use relative_degradation(last_valid_loss, loss_pilot)"
    assert (
        source.count("relative_degradation(loss_before, loss_new)") >= 1
    ), "the alpha line-search accept must use relative_degradation(loss_before, loss_new)"

    # The historical absolute-misuse patterns must be GONE. ``+ rollback_tolerance``
    # as an additive loss margin is the bug shape; if either returns the gate is
    # scale-dependent again.
    assert source.count("last_valid_loss + controller.rollback_tolerance") == 0, (
        "the pilot trigger must not use absolute 'last_valid_loss + rollback_tolerance' "
        "— rollback_tolerance is a relative fraction (property-tested)"
    )
    assert source.count("loss_before + controller.rollback_tolerance") == 0, (
        "the alpha line-search accept must not use absolute "
        "'loss_before + rollback_tolerance' — rollback_tolerance is a relative fraction"
    )
