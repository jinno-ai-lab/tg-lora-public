import os
from pathlib import Path

import pytest
import torch

from src.training.trajectory_delta_artifact import (
    build_trajectory_delta_artifact_metadata, load_trajectory_delta_artifact,
    save_trajectory_delta_artifact)


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