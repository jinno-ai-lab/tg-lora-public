import os
from pathlib import Path

import pytest
import torch

from src.training.trajectory_delta_artifact import (
    build_trajectory_delta_artifact_metadata, load_trajectory_delta_artifact,
    save_trajectory_delta_artifact)
from src.utils.checkpoint import CheckpointIntegrityError


def test_save_and_load_trajectory_delta_artifact(tmp_path: Path):
    delta = {"layer": torch.tensor([1.0, 2.0])}
    metadata = build_trajectory_delta_artifact_metadata(
        mode="baseline",
        anchor_kind="after_optimizer_step",
        trajectory_key="traj-1",
        epoch_batch_plan_key="plan-1",
        batch_plan_manifest="/tmp/batch_plan_manifest.json",
        dataset_key="dataset-1",
        delta_tensors=delta,
        step=3,
        total_backward_passes=24,
        batch_keys=["batch-a"],
        sample_keys=["sample-a", "sample-b"],
    )
    path = save_trajectory_delta_artifact(
        path=tmp_path / "delta.pt",
        metadata=metadata,
        delta_tensors=delta,
    )

    loaded = load_trajectory_delta_artifact(path)
    assert loaded.metadata.trajectory_key == "traj-1"
    assert loaded.metadata.batch_keys == ["batch-a"]
    assert loaded.metadata.sample_keys == ["sample-a", "sample-b"]
    assert loaded.metadata.delta_tensor_count == 1
    assert loaded.metadata.delta_total_norm > 0
    assert torch.allclose(loaded.delta_tensors["layer"], delta["layer"])


def _delta_tensors() -> dict:
    return {"layer": torch.tensor([1.0, 2.0, 3.0])}


def _metadata(trajectory_key: str = "traj-1"):
    return build_trajectory_delta_artifact_metadata(
        mode="baseline",
        anchor_kind="after_optimizer_step",
        trajectory_key=trajectory_key,
        epoch_batch_plan_key="plan-1",
        batch_plan_manifest="/tmp/batch_plan_manifest.json",
        dataset_key="dataset-1",
        delta_tensors=_delta_tensors(),
        step=3,
    )


class TestAtomicTrajectoryArtifactSave:
    """``trajectory_delta_artifacts/*.pt`` is written atomically — a mid-commit
    fault never leaves a torn destination.

    Mirrors ``TestAtomicCheckpointSave`` (test_checkpoint.py). These artifacts
    feed the Tier-2 trajectory-delta analysis; a torn write would silently
    corrupt that dataset at load time. ``os.replace`` is the sole publish point
    the atomic helper uses, so monkeypatching it to raise simulates a fault
    exactly at the commit boundary and locks the contract at the behavior level:

    - a fresh save that faults publishes NO destination file,
    - a faulting overwrite leaves the prior, still-loadable artifact intact,
    - the orphaned PID-suffixed temp is cleaned up either way.

    A regression to a bare ``torch.save(blob, path)`` would truncate the
    destination during serialization and the prior-intact check would then fail.
    """

    def test_fresh_save_fault_creates_no_destination(self, tmp_path, monkeypatch):
        def _boom(src, dst):
            raise OSError("simulated mid-commit fault")

        monkeypatch.setattr(os, "replace", _boom)

        with pytest.raises(OSError):
            save_trajectory_delta_artifact(
                path=tmp_path / "delta.pt",
                metadata=_metadata(),
                delta_tensors=_delta_tensors(),
            )

        # no partial destination was published...
        assert not (tmp_path / "delta.pt").exists()
        # ...and the orphaned PID-suffixed temp was cleaned up
        assert not list(tmp_path.glob("delta.pt.tmp.*"))

    def test_prior_artifact_survives_faulting_overwrite(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "delta.pt"
        save_trajectory_delta_artifact(
            path=path, metadata=_metadata("traj-1"), delta_tensors=_delta_tensors()
        )
        assert path.exists()
        original = load_trajectory_delta_artifact(path)

        def _boom(src, dst):
            raise OSError("simulated mid-commit fault")

        monkeypatch.setattr(os, "replace", _boom)

        with pytest.raises(OSError):
            save_trajectory_delta_artifact(
                path=path, metadata=_metadata("traj-9"), delta_tensors=_delta_tensors()
            )

        # The prior, still-loadable artifact is intact with the OLD key — the
        # torn (traj-9) state was never published. A regressed bare
        # torch.save(path) would have truncated it and this would reload as
        # traj-9 or fail to load entirely.
        assert path.exists()
        reloaded = load_trajectory_delta_artifact(path)
        assert reloaded.metadata.trajectory_key == "traj-1"
        assert reloaded.metadata.trajectory_key == original.metadata.trajectory_key
        assert not list(tmp_path.glob("delta.pt.tmp.*"))


class TestTrajectoryArtifactLoadIntegrity:
    """A torn/empty/garbage trajectory artifact is diagnosed, not opaque.

    Mirrors ``TestLoadTrainingStateIntegrity`` in ``tests/test_checkpoint_integrity.py``
    one-for-one. ``save_trajectory_delta_artifact`` already writes atomically
    (``TestAtomicTrajectoryArtifactSave`` above pins the WRITE side), so a file
    that trips this could NOT have been torn by an in-process fault — it
    predates the atomic helper (commit ``ed26173``) or was corrupted externally.
    Every Tier-2 ``scripts/offline_*.py`` analysis entrypoint reaches the
    dataset through ``load_trajectory_delta_artifact`` (one or two ``.pt`` per
    cycle), so a torn artifact previously aborted a multi-hour offline
    validation midway with an opaque ``EOFError`` / ``RuntimeError`` and no
    actionable message. This class pins the symmetric load-side guarantee:
    corruption → :class:`CheckpointIntegrityError` (chained, not masked), valid
    → loads clean (no false positive), missing → ``FileNotFoundError``
    (unchanged contract), and a non-corruption ``RuntimeError`` re-raised
    UNCHANGED (the message-match discipline that keeps real bugs visible).
    """

    def _write_valid_artifact(self, tmp_path: Path, name: str = "delta.pt") -> Path:
        path = tmp_path / name
        save_trajectory_delta_artifact(
            path=path, metadata=_metadata("traj-1"), delta_tensors=_delta_tensors()
        )
        return path

    def test_truncated_artifact_raises_integrity_error(self, tmp_path):
        path = self._write_valid_artifact(tmp_path)
        full = path.read_bytes()
        # Cut the zip archive mid-stream — the dominant real-world torn shape
        # (a copy/backup that died partway through a multi-hundred-KB write).
        path.write_bytes(full[: len(full) // 2])

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_trajectory_delta_artifact(path)

        msg = str(exc_info.value)
        assert "torn or corrupt" in msg
        assert "cannot be loaded for analysis" in msg
        # Actionable: names the file and points at the atomic-helper invariant.
        assert str(path) in msg
        assert "atomic-save helper" in msg

    def test_truncated_artifact_chains_original_error(self, tmp_path):
        path = self._write_valid_artifact(tmp_path)
        full = path.read_bytes()
        path.write_bytes(full[: len(full) // 2])

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_trajectory_delta_artifact(path)

        # The original loader error is chained, not swallowed — a debugger can
        # still see the underlying EOFError / zip-archive RuntimeError.
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, BaseException)

    def test_empty_artifact_raises_integrity_error(self, tmp_path):
        path = tmp_path / "delta.pt"
        path.write_bytes(b"")  # empty file → torch.load raises EOFError

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_trajectory_delta_artifact(path)

        assert isinstance(exc_info.value.__cause__, EOFError)

    def test_garbage_artifact_raises_integrity_error(self, tmp_path):
        path = tmp_path / "delta.pt"
        # Non-pickle garbage → pickle.UnpicklingError("Weights only load failed")
        path.write_bytes(b"\x80\x02PARTIAL_TORN_DUMP_NOT_A_PICKLE")

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_trajectory_delta_artifact(path)

        assert "torn or corrupt" in str(exc_info.value)

    def test_non_zip_first_bytes_raises_integrity_error(self, tmp_path):
        # Real torch.save bytes truncated to the 8-byte header → the torch C++
        # zip reader rejects it with RuntimeError("... not a ZIP archive"), a
        # DISTINCT corruption branch from the truncated-half "failed finding
        # central directory" case above. Pins that BOTH zip markers convert.
        src = tmp_path / "src.pt"
        torch.save({"a": torch.tensor([1.0])}, src)
        path = tmp_path / "delta.pt"
        path.write_bytes(src.read_bytes()[:8])

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_trajectory_delta_artifact(path)

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "not a ZIP archive" in str(exc_info.value.__cause__) or (
            "zip archive" in str(exc_info.value.__cause__)
        )

    def test_valid_artifact_loads_clean(self, tmp_path):
        # No false positive: a real, loadable artifact round-trips untouched.
        path = self._write_valid_artifact(tmp_path)

        loaded = load_trajectory_delta_artifact(path)

        assert loaded.metadata.trajectory_key == "traj-1"  # proves it deserialized

    def test_missing_artifact_still_raises_file_not_found(self, tmp_path):
        # The existing contract is preserved: a missing file is NOT corruption.
        with pytest.raises(FileNotFoundError):
            load_trajectory_delta_artifact(tmp_path / "does_not_exist.pt")

    def test_non_corruption_runtime_error_is_not_masked(self, tmp_path, monkeypatch):
        # A RuntimeError that does NOT carry a corruption signature is a genuine
        # deserialization bug — it must surface UNCHANGED, not be relabeled as a
        # torn artifact (which would hide the real cause).
        path = tmp_path / "delta.pt"
        path.write_bytes(b"\x80\x02")  # bytes present so we reach the loader

        def _fake_real_bug(_path):
            raise RuntimeError("a genuine deserialization bug in unrelated code")

        monkeypatch.setattr(
            "src.training.trajectory_delta_artifact.load_tensor_artifact",
            _fake_real_bug,
        )

        with pytest.raises(RuntimeError) as exc_info:
            load_trajectory_delta_artifact(path)

        assert not isinstance(exc_info.value, CheckpointIntegrityError)
        assert "genuine deserialization bug" in str(exc_info.value)

    def test_corruption_signature_runtime_error_is_converted(self, tmp_path, monkeypatch):
        # The flip side: a RuntimeError that DOES carry the zip-archive
        # corruption signature IS relabeled as CheckpointIntegrityError,
        # chained from the original. Pins the message-match discipline
        # end-to-end on the trajectory load path.
        path = tmp_path / "delta.pt"
        path.write_bytes(b"\x80\x02")

        signature = RuntimeError(
            "PytorchStreamReader failed reading zip archive: not a ZIP archive"
        )

        def _torn_zip_signature(_path):
            raise signature

        monkeypatch.setattr(
            "src.training.trajectory_delta_artifact.load_tensor_artifact",
            _torn_zip_signature,
        )

        with pytest.raises(CheckpointIntegrityError) as exc_info:
            load_trajectory_delta_artifact(path)

        assert exc_info.value.__cause__ is signature