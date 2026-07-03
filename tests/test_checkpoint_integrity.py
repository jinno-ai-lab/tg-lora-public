"""Load-side integrity of the atomic-checkpoint guarantee.

WRITE-side atomicity (``_atomic_torch_save`` for single-file artifacts,
``_atomic_publish_checkpoint_dir`` for the staged LoRA adapter dir) makes it
impossible for the TRAINING PROCESS to publish a torn destination: a mid-commit
fault leaves the destination either fully reflecting the new state or still at
its prior, loadable value — never torn. But a torn checkpoint can still reach
the loader from sources that guarantee does NOT govern:

  * a checkpoint written before the atomic helpers landed (commits ``510b0d1``
    for ``training_state.pt``, ``620372c`` for the adapter dir) still sitting
    in an old run dir,
  * external corruption — disk-full during a non-atomic copy/backup of the run
    dir, an NFS hiccup, a manual edit, ``kill -9`` mid-``cp``.

Without a load-side check, such a file crashes resume with an OPAQUE
``EOFError`` / ``RuntimeError("PytorchStreamReader failed reading zip
archive...")`` / ``SafetensorError`` and no actionable diagnosis. This module
pins the symmetric load-side guarantee:

  * a torn / empty / garbage checkpoint is re-raised as
    :class:`CheckpointIntegrityError` with the original error CHAINED
    (``raise ... from exc`` — nothing is masked),
  * a VALID checkpoint still loads clean (no false positive),
  * a MISSING checkpoint still raises ``FileNotFoundError`` (the existing
    contract, unchanged), and
  * a ``RuntimeError`` that is NOT a corruption signature (a genuine
    deserialization bug) is re-raised UNCHANGED — the message-match discipline
    that keeps real bugs visible.

The torn-file bytes written here reproduce the empirically-captured torch 2.1.1
+ safetensors 0.3.1 signatures documented next to
``_is_torch_load_corruption`` / ``_is_safetensors_corruption`` in
``src/utils/checkpoint.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.checkpoint import (
    CheckpointIntegrityError,
    load_adapter_weights,
    load_training_state,
    save_training_state,
)
from tests.test_fault_recovery import _make_training_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_valid_training_state(tmp_path: Path) -> Path:
    """Save a real, loadable training_state.pt and return its path."""
    state = _make_training_state()  # cycle_offset == 3, fully populated
    path = tmp_path / "training_state.pt"
    save_training_state(state, path)
    return path


# ---------------------------------------------------------------------------
# load_training_state — torn/empty/garbage → CheckpointIntegrityError
# ---------------------------------------------------------------------------


class TestLoadTrainingStateIntegrity:
    """A torn/empty/garbage ``training_state.pt`` is diagnosed, not opaque."""

    def test_truncated_checkpoint_raises_integrity_error(self, tmp_path):
        path = _write_valid_training_state(tmp_path)
        full = path.read_bytes()
        # Cut the zip archive mid-stream — the dominant real-world torn shape
        # (a copy/backup that died partway through a multi-hundred-KB write).
        path.write_bytes(full[: len(full) // 2])

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_training_state(path)

        msg = str(exc_info.value)
        assert "torn or corrupt" in msg
        assert "cannot be loaded for resume" in msg
        # Actionable: names the file and points at the atomic-helper invariant.
        assert str(path) in msg
        assert "atomic-save helper" in msg

    def test_truncated_checkpoint_chains_original_error(self, tmp_path):
        path = _write_valid_training_state(tmp_path)
        full = path.read_bytes()
        path.write_bytes(full[: len(full) // 2])

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_training_state(path)

        # The original loader error is chained, not swallowed — a debugger can
        # still see the underlying EOFError / zip-archive RuntimeError.
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, BaseException)

    def test_empty_checkpoint_raises_integrity_error(self, tmp_path):
        path = tmp_path / "training_state.pt"
        path.write_bytes(b"")  # empty file → torch.load raises EOFError

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_training_state(path)

        assert isinstance(exc_info.value.__cause__, EOFError)

    def test_garbage_checkpoint_raises_integrity_error(self, tmp_path):
        path = tmp_path / "training_state.pt"
        # Non-pickle garbage → pickle.UnpicklingError("Weights only load failed")
        path.write_bytes(b"\x80\x02PARTIAL_TORN_DUMP_NOT_A_PICKLE")

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_training_state(path)

        assert "torn or corrupt" in str(exc_info.value)

    def test_non_zip_first_bytes_raises_integrity_error(self, tmp_path):
        import torch

        # Real torch.save bytes truncated to the 8-byte header → the torch C++
        # zip reader rejects it with RuntimeError("... not a ZIP archive"), a
        # DISTINCT corruption branch from the truncated-half "failed finding
        # central directory" case above. Pins that BOTH zip markers convert.
        src = tmp_path / "src.pt"
        torch.save({"a": torch.tensor([1.0])}, src)
        path = tmp_path / "training_state.pt"
        path.write_bytes(src.read_bytes()[:8])

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_training_state(path)

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "not a ZIP archive" in str(exc_info.value.__cause__) or (
            "zip archive" in str(exc_info.value.__cause__)
        )

    def test_valid_checkpoint_loads_clean(self, tmp_path):
        # No false positive: a real, loadable checkpoint round-trips untouched.
        path = _write_valid_training_state(tmp_path)

        loaded = load_training_state(path)

        assert loaded.cycle_offset == 3  # proves it actually deserialized

    def test_missing_checkpoint_still_raises_file_not_found(self, tmp_path):
        # The existing contract is preserved: a missing file is NOT corruption.
        with pytest.raises(FileNotFoundError):
            load_training_state(tmp_path / "does_not_exist.pt")

    def test_non_corruption_runtime_error_is_not_masked(self, tmp_path, monkeypatch):
        # A RuntimeError that does NOT carry a corruption signature is a genuine
        # deserialization bug — it must surface UNCHANGED, not be relabeled as a
        # torn checkpoint (which would hide the real cause).
        path = tmp_path / "training_state.pt"
        path.write_bytes(b"\x80\x02")  # bytes present so we reach the loader

        def _fake_real_bug(_path):
            raise RuntimeError("a genuine deserialization bug in unrelated code")

        monkeypatch.setattr("src.utils.checkpoint.load_tensor_artifact", _fake_real_bug)

        with pytest.raises(RuntimeError) as exc_info:
            load_training_state(path)

        assert not isinstance(exc_info.value, CheckpointIntegrityError)
        assert "genuine deserialization bug" in str(exc_info.value)

    def test_corruption_signature_runtime_error_is_converted(self, tmp_path, monkeypatch):
        # The flip side: a RuntimeError that DOES carry the zip-archive corruption
        # signature IS relabeled as CheckpointIntegrityError, chained from the
        # original. This pins the message-match discipline end-to-end.
        path = tmp_path / "training_state.pt"
        path.write_bytes(b"\x80\x02")

        signature = RuntimeError(
            "PytorchStreamReader failed reading zip archive: not a ZIP archive"
        )

        def _torn_zip_signature(_path):
            raise signature

        monkeypatch.setattr("src.utils.checkpoint.load_tensor_artifact", _torn_zip_signature)

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_training_state(path)

        assert exc_info.value.__cause__ is signature


# ---------------------------------------------------------------------------
# load_adapter_weights — torn/garbage safetensors → CheckpointIntegrityError
# ---------------------------------------------------------------------------


class TestLoadAdapterWeightsIntegrity:
    """A torn/empty/garbage ``adapter_model.safetensors`` is diagnosed, not
    opaque — the costlier artifact to lose to a torn write."""

    def _make_adapter_dir(self, tmp_path: Path) -> Path:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        return adapter_dir

    def test_torn_safetensors_raises_integrity_error(self, tmp_path):
        pytest.importorskip("safetensors")
        import torch
        from safetensors.torch import save_file

        adapter_dir = self._make_adapter_dir(tmp_path)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        save_file({"lora_A.weight": torch.tensor([1.0, 2.0, 3.0])}, str(adapter_file))
        full = adapter_file.read_bytes()
        # Truncate the header mid-stream → SafetensorError("InvalidHeaderLength")
        adapter_file.write_bytes(full[: len(full) // 2])

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_adapter_weights(adapter_dir)

        msg = str(exc_info.value)
        assert "torn or corrupt" in msg
        assert "adapter" in msg
        # The original SafetensorError is chained, not swallowed.
        assert exc_info.value.__cause__ is not None
        assert "safetensors" in type(exc_info.value.__cause__).__module__

    def test_garbage_safetensors_raises_integrity_error(self, tmp_path):
        pytest.importorskip("safetensors")
        adapter_dir = self._make_adapter_dir(tmp_path)
        # Pure garbage → SafetensorError("HeaderTooSmall")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"not a safetensors file")

        with pytest.raises(CheckpointIntegrityError):
            load_adapter_weights(adapter_dir)

    def test_valid_adapter_loads_clean(self, tmp_path):
        pytest.importorskip("safetensors")
        import torch
        from safetensors.torch import save_file

        adapter_dir = self._make_adapter_dir(tmp_path)
        save_file(
            {"lora_A.weight": torch.tensor([1.0, 2.0, 3.0])},
            str(adapter_dir / "adapter_model.safetensors"),
        )

        state = load_adapter_weights(adapter_dir)

        assert isinstance(state, dict)
        assert "lora_A.weight" in state

    def test_missing_adapter_file_raises_file_not_found(self, tmp_path):
        # Dir exists, adapter_model.safetensors does not → FileNotFoundError
        # (a missing-file condition, NOT corruption) is preserved for the caller.
        pytest.importorskip("safetensors")
        adapter_dir = self._make_adapter_dir(tmp_path)

        with pytest.raises(FileNotFoundError):
            load_adapter_weights(adapter_dir)
