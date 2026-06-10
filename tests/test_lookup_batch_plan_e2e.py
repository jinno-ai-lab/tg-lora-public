import json
import subprocess
import sys
from pathlib import Path

from src.training.deterministic_batch_plan import \
    build_deterministic_batch_plan_manifest
from src.training.trajectory_delta_artifact import (
    build_trajectory_delta_artifact_metadata, save_trajectory_delta_artifact)

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "scripts" / "lookup_batch_plan.py"


def test_lookup_batch_plan_resolves_sample_and_artifact(tmp_path: Path):
    records = [{"id": "r1", "text": "alpha"}, {"id": "r2", "text": "beta"}]
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = build_deterministic_batch_plan_manifest(
        records,
        batch_size=2,
        dataset_path=str(dataset_path),
    )
    manifest_path = tmp_path / "batch_plan_manifest.json"
    manifest.save(manifest_path)

    delta = {"layer": __import__("torch").tensor([1.0])}
    metadata = build_trajectory_delta_artifact_metadata(
        mode="tg_lora",
        anchor_kind="after_pilot",
        trajectory_key="traj-1",
        epoch_batch_plan_key=manifest.epoch_batch_plan_key,
        batch_plan_manifest=str(manifest_path),
        dataset_key=manifest.dataset_key,
        delta_tensors=delta,
        cycle=1,
        batch_keys=[manifest.epoch_batch_keys[0]],
        sample_keys=manifest.sample_keys[:2],
    )
    artifact_path = save_trajectory_delta_artifact(
        path=tmp_path / "delta.pt",
        metadata=metadata,
        delta_tensors=delta,
    )

    result = subprocess.run(
        [sys.executable, str(CLI), "--artifact", str(artifact_path)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["artifact"]["metadata"]["anchor_kind"] == "after_pilot"
    assert payload["batch_lookup"]["records"] == records