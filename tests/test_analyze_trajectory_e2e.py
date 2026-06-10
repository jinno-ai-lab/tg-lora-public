"""E2E integration tests for analyze_trajectory.py CLI (TASK-0123).

Generates synthetic metric data, executes the CLI via subprocess,
and validates report structure, convergence prediction, anomaly detection,
and early-stop recommendations.

Exit code semantics:
  0 — success
  2 — file not found / insufficient data / parse error
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.training.deterministic_batch_plan import \
    build_deterministic_batch_plan_manifest
from src.training.trajectory_delta_artifact import (
    build_trajectory_delta_artifact_metadata, save_trajectory_delta_artifact)

CLI = Path("scripts/analyze_trajectory.py")
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        cwd=str(ROOT),
    )


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _converging_records(n: int = 20) -> list[dict]:
    """Steadily decreasing loss → convergence prediction."""
    return [
        {
            "cycle": i,
            "train_loss": round(2.5 - 0.1 * i + 0.005 * (i % 3), 4),
            "valid_loss": round(2.4 - 0.09 * i + 0.005 * (i % 3), 4),
            "grad_norm": round(max(0.1, 1.0 - 0.04 * i), 4),
            "velocity_magnitude": round(0.1 * (n - i) / n, 4),
        }
        for i in range(n)
    ]


def _diverging_records(n: int = 15) -> list[dict]:
    """Steady decrease followed by a sharp spike → anomaly detection."""
    records = [
        {
            "cycle": i,
            "train_loss": round(2.5 - 0.15 * i, 4),
            "valid_loss": round(2.4 - 0.14 * i, 4),
            "grad_norm": round(max(0.1, 1.0 - 0.05 * i), 4),
        }
        for i in range(n - 1)
    ]
    # Steady losses ~0.3-0.45 range, then spike to 5.0 (z-score > 3)
    records.append({
        "cycle": n - 1,
        "train_loss": 5.0,
        "valid_loss": 5.2,
        "grad_norm": 10.0,
    })
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnalyzeTrajectoryE2E:
    """End-to-end tests exercising analyze_trajectory.py as a subprocess."""

    # -- basic CLI modes ------------------------------------------------------

    def test_e2e_jsonl_input(self, tmp_path: Path):
        """JSONL file input → exit 0 with valid report."""
        jsonl = _write_jsonl(tmp_path / "metrics.jsonl", _converging_records())
        r = _run_cli(str(jsonl))
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Trajectory Analysis" in r.stdout

    def test_e2e_from_losses(self):
        """--from-losses mode → exit 0 with convergence info."""
        r = _run_cli("--from-losses", "2.5,2.3,2.1,1.9,1.8,1.7,1.6,1.55,1.51,1.49")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Convergence:" in r.stdout

    # -- convergence / anomaly detection --------------------------------------

    def test_e2e_converging_data(self, tmp_path: Path):
        """Converging data → positive convergence_rate."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records(20))
        r = _run_cli(str(jsonl), "--output", str(tmp_path / "report.json"))
        assert r.returncode == 0

        report = json.loads((tmp_path / "report.json").read_text())
        conv = report["convergence"]
        assert conv["convergence_rate"] > 0, "Converging data should have positive rate"

    def test_e2e_diverging_data(self, tmp_path: Path):
        """Diverging data → anomaly detection."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _diverging_records(10))
        r = _run_cli(str(jsonl), "--output", str(tmp_path / "report.json"))
        assert r.returncode == 0

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["anomalies"]["detected"], "Diverging data should trigger anomaly"

    # -- error handling -------------------------------------------------------

    def test_e2e_missing_file(self, tmp_path: Path):
        """Non-existent file → exit 2."""
        r = _run_cli(str(tmp_path / "nope.jsonl"))
        assert r.returncode == 2

    def test_e2e_insufficient_data(self, tmp_path: Path):
        """Single data point → exit 2 (need >= 2 points)."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", [{"cycle": 0, "train_loss": 2.0}])
        r = _run_cli(str(jsonl))
        assert r.returncode == 2

    # -- options ---------------------------------------------------------------

    def test_e2e_target_loss_option(self, tmp_path: Path):
        """--target-loss sets convergence target."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records(15))
        r = _run_cli(str(jsonl), "--target-loss", "0.5", "--output", str(tmp_path / "r.json"))
        assert r.returncode == 0

        report = json.loads((tmp_path / "r.json").read_text())
        assert "remaining_steps" in report["convergence"]

    def test_e2e_output_file(self, tmp_path: Path):
        """--output writes JSON report to the specified path."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records(10))
        out = tmp_path / "sub" / "report.json"
        r = _run_cli(str(jsonl), "--output", str(out))
        assert r.returncode == 0
        assert out.exists()

        report = json.loads(out.read_text())
        for key in ("total_points", "convergence", "early_stop", "anomalies",
                     "loss_trend", "volatility"):
            assert key in report, f"Missing field: {key}"

    # -- text output structure -------------------------------------------------

    def test_e2e_text_output_sections(self, tmp_path: Path):
        """Text output contains expected report sections."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records(12))
        r = _run_cli(str(jsonl))
        assert r.returncode == 0
        assert "Convergence:" in r.stdout
        assert "Early Stop Advice:" in r.stdout
        assert "Anomalies:" in r.stdout
        assert "Recommendation:" in r.stdout

    def test_e2e_artifact_anomalies_include_source_records(self, tmp_path: Path):
        jsonl = _write_jsonl(tmp_path / "run_metrics.jsonl", _converging_records(8))
        records = [{"id": "r1", "text": "alpha"}, {"id": "r2", "text": "beta"}]
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
        manifest_path = tmp_path / "batch_plan_manifest.json"
        manifest.save(manifest_path)
        artifact_dir = tmp_path / "trajectory_delta_artifacts"
        artifact_dir.mkdir()
        for cycle, norm_value, sample_index in ((1, 1.0, 0), (2, 1.0, 1), (3, 10.0, 1)):
            delta = {"layer": torch.tensor([norm_value])}
            metadata = build_trajectory_delta_artifact_metadata(
                mode="tg_lora",
                anchor_kind="after_pilot",
                trajectory_key="traj-1",
                epoch_batch_plan_key=manifest.epoch_batch_plan_key,
                batch_plan_manifest=str(manifest_path),
                dataset_key=manifest.dataset_key,
                delta_tensors=delta,
                cycle=cycle,
                batch_keys=[manifest.epoch_batch_keys[sample_index]],
                sample_keys=[manifest.sample_keys[sample_index]],
            )
            save_trajectory_delta_artifact(
                path=artifact_dir / f"cycle_{cycle}.pt",
                metadata=metadata,
                delta_tensors=delta,
            )

        out = tmp_path / "report.json"
        r = _run_cli(str(jsonl), "--output", str(out))
        assert r.returncode == 0
        report = json.loads(out.read_text())
        assert len(report["delta_artifact_anomalies"]) == 1
        assert report["delta_artifact_anomalies"][0]["cycle"] == 3
        assert report["delta_artifact_anomalies"][0]["records"][0] == records[1]
        assert report["delta_artifact_anomalies"][0]["source_examples"][0]["record_id"] == "r2"
        assert report["delta_artifact_anomalies"][0]["source_examples"][0]["dataset_index"] == 1
