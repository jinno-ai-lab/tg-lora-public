from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.analysis.extrapolation_predictability import (
    UpdateStep,
    analyze_update_predictability,
    beta_from_window,
    load_update_steps_from_artifacts,
)
from src.training.trajectory_delta_artifact import (
    build_trajectory_delta_artifact_metadata,
    save_trajectory_delta_artifact,
)


def _step(index: int, values: list[float]) -> UpdateStep:
    return UpdateStep(step=index, tensors={"w": torch.tensor(values)})


def test_beta_from_window():
    assert beta_from_window(1) == 0.0
    assert beta_from_window(3) == pytest.approx(2 / 3)
    assert beta_from_window(10) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        beta_from_window(0)


def test_straight_updates_are_predictable():
    updates = [_step(i, [1.0, 0.0]) for i in range(8)]

    report = analyze_update_predictability(
        updates,
        n_values=[1, 2, 5],
        short_window=3,
        long_window=10,
        yes_threshold=0.5,
    )

    for entry in report["per_n"].values():
        assert entry["sample_count"] > 0
        assert entry["predictable"] is True
        assert entry["mean_future_cos_long"] == pytest.approx(1.0)
        assert entry["mean_consistency_cos"] == pytest.approx(1.0)


def test_controls_and_splits_are_reported():
    updates = [_step(i, [1.0, 0.0]) for i in range(5)]
    updates.extend(_step(i, [0.0, 1.0]) for i in range(5, 10))

    report = analyze_update_predictability(
        updates,
        n_values=[2],
        short_window=3,
        long_window=10,
        yes_threshold=0.5,
        control_seed=123,
    )

    entry = report["per_n"]["2"]
    assert entry["mean_random_cos"] is not None
    assert entry["mean_shuffle_cos_long"] is not None
    assert entry["split_by_anchor"]["first_half"]["sample_count"] == 5
    assert entry["split_by_anchor"]["second_half"]["sample_count"] == 3
    assert entry["split_by_anchor"]["first_half"]["mean_future_cos_long"] is not None
    assert entry["split_by_anchor"]["second_half"]["mean_future_cos_long"] is not None


def test_reversing_updates_are_not_predictable_from_old_direction():
    updates = [_step(i, [1.0, 0.0]) for i in range(6)]
    updates.extend(_step(i, [-1.0, 0.0]) for i in range(6, 12))

    report = analyze_update_predictability(
        updates,
        n_values=[3],
        short_window=3,
        long_window=10,
        yes_threshold=0.5,
    )

    entry = report["per_n"]["3"]
    assert entry["sample_count"] == 9
    assert entry["mean_future_cos_long"] < 0.5


def test_load_update_steps_from_cumulative_artifacts(tmp_path: Path):
    artifact_dir = tmp_path / "trajectory_delta_artifacts"
    cumulative = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([3.0, 0.0]),
    ]
    for step, tensor in enumerate(cumulative, start=1):
        delta = {"w": tensor}
        metadata = build_trajectory_delta_artifact_metadata(
            mode="baseline",
            anchor_kind="after_optimizer_step",
            trajectory_key="traj",
            epoch_batch_plan_key="plan",
            batch_plan_manifest=None,
            dataset_key="dataset",
            delta_tensors=delta,
            step=step,
            total_backward_passes=step,
        )
        save_trajectory_delta_artifact(
            path=artifact_dir / f"step_{step}.pt",
            metadata=metadata,
            delta_tensors=delta,
        )

    updates = load_update_steps_from_artifacts(artifact_dir)

    assert [update.step for update in updates] == [1, 2, 3]
    assert all(
        torch.allclose(update.tensors["w"], torch.tensor([1.0, 0.0]))
        for update in updates
    )


def test_cli_writes_json_report(tmp_path: Path):
    artifact_dir = tmp_path / "trajectory_delta_artifacts"
    for step in range(1, 5):
        delta = {"w": torch.tensor([float(step), 0.0])}
        metadata = build_trajectory_delta_artifact_metadata(
            mode="baseline",
            anchor_kind="after_optimizer_step",
            trajectory_key="traj",
            epoch_batch_plan_key="plan",
            batch_plan_manifest=None,
            dataset_key="dataset",
            delta_tensors=delta,
            step=step,
            total_backward_passes=step,
        )
        save_trajectory_delta_artifact(
            path=artifact_dir / f"step_{step}.pt",
            metadata=metadata,
            delta_tensors=delta,
        )

    out = tmp_path / "predictability.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_extrapolation_predictability.py",
            str(artifact_dir),
            "--n-values",
            "1,2",
            "--output",
            str(out),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Extrapolation Predictability" in result.stdout
    assert "Controls" in result.stdout
    report = json.loads(out.read_text())
    assert report["per_n"]["1"]["predictable"] is True
    assert report["per_n"]["1"]["mean_random_cos"] is not None
