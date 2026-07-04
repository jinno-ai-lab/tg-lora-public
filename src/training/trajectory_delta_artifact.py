from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.utils.atomic_save import _atomic_torch_save
from src.utils.checkpoint_integrity import (
    CheckpointIntegrityError,
    _is_torch_load_corruption,
)
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
    _atomic_torch_save(blob, target)
    return target


def load_trajectory_delta_artifact(path: str | Path) -> TrajectoryDeltaArtifact:
    """Load a trajectory-delta artifact, diagnosing a torn file loud.

    The load-side counterpart to :func:`save_trajectory_delta_artifact`'s atomic
    write. The Tier-2 trajectory-delta analysis (every ``scripts/offline_*.py``
    entrypoint + :mod:`src.analysis.extrapolation_predictability`) reaches the
    dataset through THIS function, iterating one or two ``.pt`` files per cycle
    across a whole run. A torn / truncated / empty artifact — from a checkpoint
    that predates the atomic helper (commit ``ed26173``), or external corruption
    (disk-full during a non-atomic copy/backup of the run dir, NFS, a manual
    edit, ``kill -9`` mid-transfer) — used to crash that analysis with an OPAQUE
    ``EOFError`` / ``RuntimeError("PytorchStreamReader failed reading zip
    archive…")`` and no actionable diagnosis, aborting a multi-hour offline
    validation midway. Mirroring :func:`src.utils.checkpoint.load_training_state`
    exactly: the corruption signature is re-raised as
    :class:`CheckpointIntegrityError` with the original error CHAINED
    (``raise ... from exc`` — nothing masked), while a ``RuntimeError`` that is
    NOT a corruption signature (a genuine deserialization bug) is re-raised
    UNCHANGED so it is not masked. The primitives are imported from the
    :mod:`src.utils.checkpoint_integrity` leaf (NOT :mod:`src.utils.checkpoint`)
    so this low-level analysis module does not drag in the training-resume
    controller graph.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory delta artifact not found: {path}")

    try:
        blob = load_tensor_artifact(path)
    except (EOFError, pickle.UnpicklingError, RuntimeError) as exc:
        if _is_torch_load_corruption(exc):
            raise CheckpointIntegrityError(
                f"Trajectory delta artifact at {path} exists but is torn or "
                f"corrupt and cannot be loaded for analysis "
                f"({type(exc).__name__}: {exc}). The atomic-save helper makes it "
                f"impossible for the training process to WRITE a torn artifact, "
                f"so this file predates that helper (commit ed26173) or was "
                f"corrupted externally (disk-full during a non-atomic "
                f"copy/backup, NFS, manual edit, kill -9 mid-transfer). Delete or "
                f"restore it from a known-good artifact before re-running the "
                f"trajectory-delta analysis; the analysis intentionally does NOT "
                f"silently skip a corrupt file, which would hide a gap in the "
                f"Tier-2 dataset."
            ) from exc
        raise

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