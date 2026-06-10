import json
from pathlib import Path

import torch

from src.training.deterministic_batch_plan import \
    build_deterministic_batch_plan_manifest
from src.training.trajectory_artifact_anomalies import \
    summarize_trajectory_artifact_anomalies
from src.training.trajectory_delta_artifact import (
    build_trajectory_delta_artifact_metadata, save_trajectory_delta_artifact)


def _write_artifact(
    run_dir: Path,
    *,
    manifest_path: Path,
    dataset_key: str,
    epoch_batch_plan_key: str,
    cycle: int,
    norm_value: float,
    batch_keys: list[str],
    sample_keys: list[str],
):
    delta = {"layer": torch.tensor([norm_value])}
    metadata = build_trajectory_delta_artifact_metadata(
        mode="tg_lora",
        anchor_kind="after_pilot",
        trajectory_key="traj-1",
        epoch_batch_plan_key=epoch_batch_plan_key,
        batch_plan_manifest=str(manifest_path),
        dataset_key=dataset_key,
        delta_tensors=delta,
        cycle=cycle,
        batch_keys=batch_keys,
        sample_keys=sample_keys,
    )
    save_trajectory_delta_artifact(
        path=run_dir / "trajectory_delta_artifacts" / f"cycle_{cycle}.pt",
        metadata=metadata,
        delta_tensors=delta,
    )


def test_summarize_trajectory_artifact_anomalies_resolves_source_records(tmp_path: Path):
    records = [
        {"id": "r1", "text": "alpha"},
        {"id": "r2", "text": "beta"},
    ]
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = build_deterministic_batch_plan_manifest(
        records,
        batch_size=1,
        dataset_path=str(dataset_path),
    )
    manifest_path = tmp_path / "run" / "batch_plan_manifest.json"
    manifest.save(manifest_path)

    run_dir = tmp_path / "run"
    _write_artifact(
        run_dir,
        manifest_path=manifest_path,
        dataset_key=manifest.dataset_key,
        epoch_batch_plan_key=manifest.epoch_batch_plan_key,
        cycle=1,
        norm_value=1.0,
        batch_keys=[manifest.epoch_batch_keys[0]],
        sample_keys=[manifest.sample_keys[0]],
    )
    _write_artifact(
        run_dir,
        manifest_path=manifest_path,
        dataset_key=manifest.dataset_key,
        epoch_batch_plan_key=manifest.epoch_batch_plan_key,
        cycle=2,
        norm_value=1.0,
        batch_keys=[manifest.epoch_batch_keys[1]],
        sample_keys=[manifest.sample_keys[1]],
    )
    _write_artifact(
        run_dir,
        manifest_path=manifest_path,
        dataset_key=manifest.dataset_key,
        epoch_batch_plan_key=manifest.epoch_batch_plan_key,
        cycle=3,
        norm_value=10.0,
        batch_keys=[manifest.epoch_batch_keys[1]],
        sample_keys=[manifest.sample_keys[1]],
    )

    anomalies = summarize_trajectory_artifact_anomalies(run_dir)
    assert len(anomalies) == 1
    assert anomalies[0]["cycle"] == 3
    assert anomalies[0]["records"][0] == records[1]
    assert anomalies[0]["source_examples"][0]["record_id"] == "r2"
    assert anomalies[0]["source_examples"][0]["dataset_index"] == 1
    assert anomalies[0]["source_examples"][0]["text_preview"] == "beta"