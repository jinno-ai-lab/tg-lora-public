from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.utils.tensor_artifact import load_tensor_artifact


@dataclass(frozen=True)
class TrajectoryDeltaArtifactMetadata:
    format_version: int
    mode: str
    anchor_kind: str
    trajectory_key: str
    epoch_batch_plan_key: str
    batch_plan_manifest: str | None
    dataset_key: str | None
    step: int | None = None
    cycle: int | None = None
    total_backward_passes: int | None = None
    batch_keys: list[str] = field(default_factory=list)
    sample_keys: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    delta_tensor_count: int = 0
    delta_total_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryDeltaArtifactMetadata":
        return cls(**data)


@dataclass(frozen=True)
class TrajectoryDeltaArtifact:
    metadata: TrajectoryDeltaArtifactMetadata
    delta_tensors: dict[str, torch.Tensor]


def summarize_delta_tensors(delta_tensors: dict[str, torch.Tensor]) -> tuple[int, float]:
    tensor_count = len(delta_tensors)
    total_sq = 0.0
    for tensor in delta_tensors.values():
        norm_val = tensor.float().norm().item()
        if not math.isfinite(norm_val):
            continue
        total_sq += norm_val**2
    return tensor_count, total_sq**0.5


def build_trajectory_delta_artifact_metadata(
    *,
    mode: str,
    anchor_kind: str,
    trajectory_key: str,
    epoch_batch_plan_key: str,
    batch_plan_manifest: str | None,
    dataset_key: str | None,
    delta_tensors: dict[str, torch.Tensor],
    step: int | None = None,
    cycle: int | None = None,
    total_backward_passes: int | None = None,
    batch_keys: list[str] | None = None,
    sample_keys: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> TrajectoryDeltaArtifactMetadata:
    tensor_count, total_norm = summarize_delta_tensors(delta_tensors)
    return TrajectoryDeltaArtifactMetadata(
        format_version=1,
        mode=mode,
        anchor_kind=anchor_kind,
        trajectory_key=trajectory_key,
        epoch_batch_plan_key=epoch_batch_plan_key,
        batch_plan_manifest=batch_plan_manifest,
        dataset_key=dataset_key,
        step=step,
        cycle=cycle,
        total_backward_passes=total_backward_passes,
        batch_keys=list(batch_keys or []),
        sample_keys=list(sample_keys or []),
        extra_metadata=dict(extra_metadata or {}),
        delta_tensor_count=tensor_count,
        delta_total_norm=total_norm,
    )


def save_trajectory_delta_artifact(
    *,
    path: str | Path,
    metadata: TrajectoryDeltaArtifactMetadata,
    delta_tensors: dict[str, torch.Tensor],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "metadata": metadata.to_dict(),
        "delta_tensors": {name: tensor.detach().cpu() for name, tensor in delta_tensors.items()},
    }
    torch.save(blob, target)
    return target


def load_trajectory_delta_artifact(path: str | Path) -> TrajectoryDeltaArtifact:
    blob = load_tensor_artifact(path)
    metadata = TrajectoryDeltaArtifactMetadata.from_dict(blob["metadata"])
    delta_tensors = blob["delta_tensors"]
    return TrajectoryDeltaArtifact(metadata=metadata, delta_tensors=delta_tensors)


def artifact_file_name(
    *,
    mode: str,
    anchor_kind: str,
    step: int | None = None,
    cycle: int | None = None,
) -> str:
    if step is not None:
        return f"{mode}_{anchor_kind}_step_{step:06d}.pt"
    if cycle is not None:
        return f"{mode}_{anchor_kind}_cycle_{cycle:06d}.pt"
    return f"{mode}_{anchor_kind}.pt"