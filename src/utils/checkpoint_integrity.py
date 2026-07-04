"""Checkpoint integrity primitives — the load/save diagnosis shared across layers.

Sibling of :mod:`src.utils.atomic_save` (the WRITE-side leaf). That module made
every on-disk ``torch.save`` atomic so the training process can never PUBLISH a
torn destination. This module is the counterpart on the READ side: a torn /
truncated / empty checkpoint that reaches the loader from a source the write
guarantee does NOT govern (a pre-fix checkpoint still in an old run dir, or
external corruption — disk-full during a non-atomic copy/backup, NFS, a manual
edit, ``kill -9`` mid-transfer) is diagnosed as :class:`CheckpointIntegrityError`
with the original loader error CHAINED (``raise ... from exc`` — nothing masked),
instead of crashing resume / analysis with an opaque ``EOFError`` /
``RuntimeError("PytorchStreamReader failed reading zip archive…")`` /
``SafetensorError`` and no actionable message.

This is deliberately a **zero-dependency leaf** (stdlib ``pickle`` only;
``safetensors`` is imported lazily inside :func:`_is_safetensors_corruption` so
this module does not hard-depend on it at import time). The reason is the SAME
layering concern that factored :func:`src.utils.atomic_save._atomic_torch_save`
out of :mod:`src.utils.checkpoint` (commit ``ed26173``): the analysis layer
(:mod:`src.training.trajectory_delta_artifact`, whose ``load_*`` feeds the 8+
Tier-2 ``scripts/offline_*.py`` analysis entrypoints) and the data layer
(:mod:`src.tg_lora.prefix_feature_cache`) must be able to diagnose a torn load
WITHOUT dragging in the heavyweight training-resume stack
(:mod:`src.utils.checkpoint`, which imports the whole controller graph —
``CycleState`` / ``DeltaTracker`` / ``DynFreezeState`` / ``ControllerState`` /
``Velocity``). Inverted layering (analysis → training-resume) couples layers
and pulls unrelated import cost onto a low-level module. So the integrity
primitives live in this leaf, :mod:`src.utils.checkpoint` re-imports them from
here (and re-exports them for backward compatibility —
``from src.utils.checkpoint import CheckpointIntegrityError`` keeps working),
and the analysis/data layers import them directly from here.

A ``RuntimeError`` is matched on MESSAGE (not type) so a genuine deserialization
bug that happens to raise ``RuntimeError`` is NOT masked — it re-raises
unchanged with its original traceback. The torn-file bytes signatures were
captured empirically against torch 2.1.1 + safetensors 0.3.1 and are pinned in
``tests/test_checkpoint_integrity.py`` and
``tests/test_trajectory_delta_artifact.py``.
"""

from __future__ import annotations

import pickle

__all__ = [
    "CheckpointIntegrityError",
    "CheckpointSaveError",
    # Private symbols retained for the legacy ``_``-prefixed import paths used
    # across src/ and tests/ (re-exported by ``src.utils.checkpoint``).
    "_is_torch_load_corruption",
    "_is_safetensors_corruption",
    "_TORCH_ARCHIVE_CORRUPTION_MARKERS",
]


class CheckpointIntegrityError(RuntimeError):
    """A resume checkpoint exists on disk but is torn/truncated and won't load.

    The load-side counterpart to the atomic-save guarantee. The atomic write
    helpers (:func:`src.utils.atomic_save._atomic_torch_save` for single-file
    artifacts and :func:`src.utils.checkpoint._atomic_publish_checkpoint_dir`
    for the staged LoRA adapter dir) make it impossible for the TRAINING PROCESS
    to publish a torn destination — a mid-commit fault leaves the destination
    either fully reflecting the new state or still at its prior, loadable value,
    never torn. So a checkpoint that trips this error could NOT have been torn
    by an in-process fault; it reached the loader from a source the atomic
    guarantee does not govern:

      * a checkpoint written before the atomic helpers landed (commits
        ``510b0d1`` for ``training_state.pt`` and ``620372c`` for the adapter
        dir), still sitting in an old run dir,
      * external corruption — disk-full during a non-atomic copy/backup of the
        run dir, an NFS hiccup, a manual edit, or ``kill -9`` mid-``cp``.

    Resume intentionally does NOT silently fall back to a fresh start: that
    would hide the lost training progress (a GOAL.md honesty break). Instead
    this fails loud, with the original loader error chained (``raise ... from
    exc``), so the operator can delete or restore the corrupt file deliberately.
    Silently restarting would defeat the resume guarantee the 12-site
    persistence axis went to the trouble of capturing.
    """


class CheckpointSaveError(RuntimeError):
    """A checkpoint save completed but produced no loadable artifact.

    The save-side counterpart to :class:`CheckpointIntegrityError`.
    ``save_pretrained`` ran to completion without raising, yet left the staged
    temp dir empty — a model/PEFT misconfiguration or a silently-failing
    backend, not a mid-save fault (which the ``except BaseException`` clause in
    :func:`src.utils.checkpoint.save_checkpoint` already handles by discarding
    the orphan temp).

    The atomic-publish guarantee
    (:func:`src.utils.checkpoint._atomic_publish_checkpoint_dir`) is that the
    destination is *never* left non-loadable: it covers a mid-swap FAULT (the
    prior checkpoint is restored) but not a save that *completes* yet writes
    nothing — publishing an empty temp over a prior good checkpoint would
    replace a loadable destination with an empty one, the one path that still
    violated that contract after the write-side atomicity work. Failing loud
    here (after removing the orphan empty temp) leaves any prior checkpoint
    untouched and stops the caller proceeding to write a ``training_state.pt``
    next to a missing adapter, which would otherwise look like a complete
    checkpoint on resume and crash ``load_adapter_weights`` with a bare
    ``FileNotFoundError``.
    """


# ``torch.load(weights_only=True)`` torn-file signatures, captured empirically
# against torch 2.1.1 on truncated / empty / garbage inputs (pinned in
# ``tests/test_checkpoint_integrity.py``):
#   truncated zip archive -> RuntimeError("PytorchStreamReader failed reading
#                                       zip archive: failed finding central
#                                       directory")
#   <8-byte / non-zip     -> RuntimeError("... not a ZIP archive")
#   empty file            -> EOFError
#   non-pickle garbage    -> pickle.UnpicklingError("Weights only load failed. ...")
# ``RuntimeError`` is matched on MESSAGE (not type) so a genuine deserialization
# bug that happens to raise ``RuntimeError`` is NOT masked — it re-raises
# unchanged with its original traceback.
_TORCH_ARCHIVE_CORRUPTION_MARKERS = (
    "failed reading zip archive",
    "not a ZIP archive",
)


def _is_torch_load_corruption(exc: BaseException) -> bool:
    """True if *exc* is the signature ``torch.load`` raises on a torn / empty /
    garbage file (as opposed to a real deserialization bug)."""
    if isinstance(exc, (EOFError, pickle.UnpicklingError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        return any(marker in msg for marker in _TORCH_ARCHIVE_CORRUPTION_MARKERS)
    return False


def _is_safetensors_corruption(exc: BaseException) -> bool:
    """True if *exc* is the signature ``safetensors.torch.load_file`` raises on
    a torn / garbage ``.safetensors`` file. ``safetensors`` is a rigid format —
    it either loads or the bytes are corrupt / truncated / version-mismatched —
    so a ``SafetensorError`` on a file that exists is corruption, full stop."""
    try:
        from safetensors import SafetensorError
    except ImportError:  # pragma: no cover - safetensors is a core dep; be robust
        return type(exc).__module__.startswith("safetensors")
    return isinstance(exc, SafetensorError)
