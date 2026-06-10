from pathlib import Path

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