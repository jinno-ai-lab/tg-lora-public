"""Dynamic integrity test for the resume-path adapter-restore seam.

The atomic-SAVE axis (:func:`src.utils.atomic_save._atomic_torch_save` for
single-file artifacts and :func:`src.utils.checkpoint._atomic_publish_checkpoint_dir`
for the staged LoRA adapter dir) makes it impossible for the TRAINING PROCESS to
PUBLISH a torn destination. The load-SIDE axis
(:class:`src.utils.checkpoint.CheckpointIntegrityError`, raised by
:func:`~src.utils.checkpoint.load_training_state` and
:func:`~src.utils.checkpoint.load_adapter_weights`) diagnoses a torn checkpoint
that reaches the loader from a source the save guarantee does not govern (a
pre-fix checkpoint still on disk, or external corruption — disk-full during a
non-atomic copy, NFS, ``kill -9`` mid-transfer).

What neither axis pinned until now is the ORDERING invariant at the resume seam
itself: that the adapter integrity load runs BEFORE the apply mutates the model,
so a torn ``adapter_model.safetensors`` raises loud and leaves the model
UNTOUCHED — never half-applied. In the inlined resume block that was only true
by virtue of statement ordering (an untested coincidence a future edit could
silently reorder). Extracting the load+apply into
:func:`src.training.train_tg_lora._restore_adapter_weights` (with an injectable
``apply_state`` so the invariant is testable without the ``peft`` dependency the
public mirror's test env lacks) makes it a named, importable seam. This module
pins the invariant DYNAMICALLY:

  * a torn / garbage adapter raises :class:`CheckpointIntegrityError` AND the
    model is never mutated (``apply_state`` never reached) — the property a
    real SIGINT-mid-save fault, now prevented on the write side, could still
    threaten via a pre-fix or externally-corrupted checkpoint on resume;
  * a clean adapter is applied exactly once with the loaded state dict;
  * a missing adapter dir is a missing-file condition (warn, no raise, no apply),
    preserving the pre-existing contract.

This is the runnable-on-this-mirror analog of the steering feedback's
"SIGINT mid-checkpoint and resume" end-to-end ask: a real 9B multi-seed run is
Category-C here (needs the private ``src.data`` pipeline + >12 GB), so the
resume seam's torn-input behavior is proven at proxy scale through the REAL
:func:`load_adapter_weights` (no mocks of the loader itself) with a recording
apply — the same fail-loud contract the 9B resume path depends on.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test-only import shim (canonical copy in tests/test_resume_state_integration.py)
# ---------------------------------------------------------------------------
# The public mirror excludes the private ``src.data`` pipeline and does not
# install ``peft``; ``src.training.train_tg_lora``'s top-level import chain
# pulls both in, so the module is un-importable here without the shim. Every
# shimmed symbol is ALWAYS mocked in real loop tests and NEVER exercised — they
# only need to resolve so the module imports and the real ``_restore_adapter``
# seam is reachable. See tests/test_resume_state_integration.py for the full
# rationale; this is a minimal copy so this module collects independently.
if "src.data" not in sys.modules:
    sys.modules["src.data"] = types.ModuleType("src.data")
if "src.data.build_seed_dataset" not in sys.modules:
    _shim = types.ModuleType("src.data.build_seed_dataset")
    _shim.load_dataset = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("src.data.load_dataset is private; tests must mock it")
    )
    sys.modules["src.data.build_seed_dataset"] = _shim
if "src.model.load_model" not in sys.modules:
    _load_model_shim = types.ModuleType("src.model.load_model")

    def _unavailable(name):
        def _raise(*_a, **_k):
            raise RuntimeError(
                f"src.model.load_model.{name} needs peft (not installed here); "
                "tests must mock it"
            )

        return _raise

    for _n in ("apply_lora", "get_input_device", "load_base_model", "load_tokenizer"):
        setattr(_load_model_shim, _n, _unavailable(_n))
    sys.modules["src.model.load_model"] = _load_model_shim

from src.training.train_tg_lora import _restore_adapter_weights  # noqa: E402
from src.utils.checkpoint import CheckpointIntegrityError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recording_apply(calls: list, sentinel: object):
    """An apply_state that records (model, loaded-state) without touching peft."""

    def _apply(model, state):
        calls.append((model, state))

    return _apply, sentinel


def _write_clean_adapter(adapter_dir: Path) -> None:
    """Write a real, loadable adapter_model.safetensors into *adapter_dir*."""
    import torch
    from safetensors.torch import save_file

    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"lora_A.weight": torch.tensor([1.0, 2.0, 3.0])},
        str(adapter_dir / "adapter_model.safetensors"),
    )


def _write_torn_adapter(adapter_dir: Path) -> None:
    """Write a truncated adapter_model.safetensors — the dominant torn shape."""
    import torch
    from safetensors.torch import save_file

    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    save_file({"lora_A.weight": torch.tensor([1.0, 2.0, 3.0])}, str(adapter_file))
    full = adapter_file.read_bytes()
    # Cut the header mid-stream → SafetensorError on load.
    adapter_file.write_bytes(full[: len(full) // 2])


def _write_garbage_adapter(adapter_dir: Path) -> None:
    """Write pure garbage — a distinct corruption branch from truncation."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"not a safetensors file")


# ---------------------------------------------------------------------------
# The ordering invariant: torn adapter → fail loud, model NEVER mutated
# ---------------------------------------------------------------------------


class TestTornAdapterFailsBeforeMutation:
    """A torn/garbage adapter raises ``CheckpointIntegrityError`` from the load
    BEFORE ``apply_state`` is reached — so the model is left untouched, never
    half-applied. This is the invariant the inlined resume block only held by
    statement-ordering coincidence."""

    def test_torn_adapter_fails_loud_without_mutating_model(self, tmp_path):
        pytest.importorskip("safetensors")
        adapter_dir = tmp_path / "adapter"
        _write_torn_adapter(adapter_dir)

        calls: list = []
        apply_state, sentinel = _recording_apply(calls, object())

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            _restore_adapter_weights(sentinel, adapter_dir, apply_state=apply_state)

        # The original SafetensorError is chained, not swallowed.
        assert exc_info.value.__cause__ is not None
        assert "safetensors" in type(exc_info.value.__cause__).__module__
        # THE ordering invariant: apply_state was NEVER reached, so a torn
        # adapter cannot leave the model half-applied.
        assert calls == []

    def test_garbage_adapter_fails_loud_without_mutating_model(self, tmp_path):
        pytest.importorskip("safetensors")
        adapter_dir = tmp_path / "adapter"
        _write_garbage_adapter(adapter_dir)

        calls: list = []
        apply_state, sentinel = _recording_apply(calls, object())

        with pytest.raises(CheckpointIntegrityError):
            _restore_adapter_weights(sentinel, adapter_dir, apply_state=apply_state)

        assert calls == []

    def test_integrity_error_message_names_adapter_and_chains(self, tmp_path):
        pytest.importorskip("safetensors")
        adapter_dir = tmp_path / "adapter"
        _write_torn_adapter(adapter_dir)
        apply_state, sentinel = _recording_apply([], object())

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            _restore_adapter_weights(sentinel, adapter_dir, apply_state=apply_state)

        msg = str(exc_info.value)
        assert "torn or corrupt" in msg
        assert "adapter" in msg
        assert str(adapter_dir / "adapter_model.safetensors") in msg


# ---------------------------------------------------------------------------
# Happy path + missing-file contract
# ---------------------------------------------------------------------------


class TestRestoreAdapterHappyAndMissing:
    """A clean adapter is applied exactly once; a missing dir is a missing-file
    condition (warn, no raise, no apply), preserving the pre-existing contract."""

    def test_clean_adapter_applied_once_with_loaded_state(self, tmp_path):
        pytest.importorskip("safetensors")
        adapter_dir = tmp_path / "adapter"
        _write_clean_adapter(adapter_dir)

        calls: list = []
        apply_state, sentinel = _recording_apply(calls, object())

        _restore_adapter_weights(sentinel, adapter_dir, apply_state=apply_state)

        assert len(calls) == 1
        applied_model, applied_state = calls[0]
        # The model passed to apply is exactly the caller's model (no surrogate).
        assert applied_model is sentinel
        # The applied state is the REAL loaded dict, with the saved tensor.
        assert isinstance(applied_state, dict)
        assert "lora_A.weight" in applied_state

    def test_missing_adapter_dir_warns_without_raising_or_applying(self, tmp_path):
        # A nonexistent dir is a missing-file condition, NOT corruption: the
        # pre-existing contract warns and leaves the weights fresh.
        missing_dir = tmp_path / "does_not_exist"
        calls: list = []
        apply_state, sentinel = _recording_apply(calls, object())

        # Must not raise.
        _restore_adapter_weights(sentinel, missing_dir, apply_state=apply_state)

        assert calls == []
