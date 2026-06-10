from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from src.training.deterministic_batch_plan import (
    load_deterministic_batch_plan_manifest, resolve_record_for_sample_key,
    resolve_records_for_batch_key)
from src.training.trajectory_delta_artifact import \
    load_trajectory_delta_artifact


@dataclass(frozen=True)
class TrajectoryArtifactSummary:
    path: Path
    metadata: Any


def discover_trajectory_artifacts(run_dir: str | Path) -> list[TrajectoryArtifactSummary]:
    artifact_dir = Path(run_dir) / "trajectory_delta_artifacts"
    if not artifact_dir.is_dir():
        return []

    summaries: list[TrajectoryArtifactSummary] = []
    for path in sorted(artifact_dir.glob("*.pt")):
        artifact = load_trajectory_delta_artifact(path)
        summaries.append(TrajectoryArtifactSummary(path=path, metadata=artifact.metadata))
    return summaries


def _robust_z_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    if mad <= 1e-12:
        scale = max(abs(center), 1e-12)
        return [
            float("inf") if abs(value - center) > scale * 3.0 else 0.0
            for value in values
        ]
    return [0.6745 * (value - center) / mad for value in values]


def _resolve_source_context(summary: TrajectoryArtifactSummary) -> dict[str, Any]:
    metadata = summary.metadata
    manifest_path = metadata.batch_plan_manifest
    if manifest_path is None:
        return {
            "manifest_path": None,
            "dataset_path": None,
            "records": [],
            "source_examples": [],
        }

    manifest = load_deterministic_batch_plan_manifest(manifest_path)
    source_examples: list[dict[str, Any]] = []
    seen_sample_keys: set[str] = set()
    for batch_key in metadata.batch_keys:
        batch_locator = manifest.batch_locator_by_key(batch_key)
        records = resolve_records_for_batch_key(manifest, batch_key)
        if batch_locator is None or records is None:
            continue
        for sample_key, dataset_index, record in zip(
            batch_locator.sample_keys,
            batch_locator.dataset_indices,
            records,
            strict=True,
        ):
            if sample_key in seen_sample_keys:
                continue
            sample_locator = manifest.sample_locator_by_key(sample_key)
            preview = record.get("text") or (
                str(record.get("prompt", "")) + str(record.get("completion", ""))
            )
            source_examples.append(
                {
                    "sample_key": sample_key,
                    "batch_key": batch_key,
                    "dataset_index": dataset_index,
                    "record_id": None if sample_locator is None else sample_locator.record_id,
                    "text_preview": preview.replace("\n", " ")[:120],
                    "record": record,
                }
            )
            seen_sample_keys.add(sample_key)

    for sample_key in metadata.sample_keys:
        if sample_key in seen_sample_keys:
            continue
        record = resolve_record_for_sample_key(manifest, sample_key)
        sample_locator = manifest.sample_locator_by_key(sample_key)
        if record is None or sample_locator is None:
            continue
        preview = record.get("text") or (
            str(record.get("prompt", "")) + str(record.get("completion", ""))
        )
        source_examples.append(
            {
                "sample_key": sample_key,
                "batch_key": None,
                "dataset_index": sample_locator.dataset_index,
                "record_id": sample_locator.record_id,
                "text_preview": preview.replace("\n", " ")[:120],
                "record": record,
            }
        )

    return {
        "manifest_path": manifest_path,
        "dataset_path": manifest.dataset_path,
        "records": [item["record"] for item in source_examples],
        "source_examples": source_examples,
    }


def summarize_trajectory_artifact_anomalies(
    run_dir: str | Path,
    *,
    min_group_size: int = 3,
    robust_z_threshold: float = 3.5,
    max_records: int = 3,
) -> list[dict[str, Any]]:
    summaries = discover_trajectory_artifacts(run_dir)
    grouped: dict[str, list[TrajectoryArtifactSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.metadata.anchor_kind, []).append(summary)

    anomalies: list[dict[str, Any]] = []
    for anchor_kind, group in grouped.items():
        if len(group) < min_group_size:
            continue
        norms = [float(summary.metadata.delta_total_norm) for summary in group]
        scores = _robust_z_scores(norms)
        group_center = median(norms)
        for summary, score in zip(group, scores, strict=True):
            if math.isnan(score) or abs(score) < robust_z_threshold:
                continue
            source_context = _resolve_source_context(summary)
            anomalies.append(
                {
                    "artifact_path": str(summary.path),
                    "anchor_kind": anchor_kind,
                    "cycle": summary.metadata.cycle,
                    "step": summary.metadata.step,
                    "total_backward_passes": summary.metadata.total_backward_passes,
                    "delta_total_norm": float(summary.metadata.delta_total_norm),
                    "robust_z_score": float(score),
                    "group_median_norm": float(group_center),
                    "batch_keys": list(summary.metadata.batch_keys),
                    "sample_keys": list(summary.metadata.sample_keys),
                    "manifest_path": source_context["manifest_path"],
                    "dataset_path": source_context["dataset_path"],
                    "records": source_context["records"][:max_records],
                    "source_examples": source_context["source_examples"][:max_records],
                }
            )

    anomalies.sort(
        key=lambda item: (
            abs(item["robust_z_score"]),
            item.get("cycle") if item.get("cycle") is not None else -1,
            item.get("step") if item.get("step") is not None else -1,
        ),
        reverse=True,
    )
    return anomalies