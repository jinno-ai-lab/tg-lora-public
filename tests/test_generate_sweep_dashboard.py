"""Tests for scripts/generate_sweep_dashboard.py."""

import json
import subprocess
import sys

import pytest

from scripts.generate_sweep_dashboard import generate_html, load_ranking


def _ranking_data() -> dict:
    return {
        "baseline": {
            "run_id": "accel_no_accel",
            "best_valid_loss": 1.234567,
            "total_backward_passes": 4000,
            "wall_seconds": 300.0,
            "loss_reduction": 0.5,
            "loss_red_per_bp": 1.25e-4,
            "loss_red_per_wall_min": 0.1,
        },
        "best_run": {
            "run_id": "accel_conservative",
            "best_valid_loss": 1.230000,
            "loss_red_per_bp": 1.3e-4,
            "loss_red_per_wall_min": 0.105,
        },
        "pairwise": [
            {
                "run_id": "accel_conservative",
                "decay": 0.3,
                "boost": 1.1,
                "best_valid_loss": 1.230000,
                "delta_vs_baseline": -0.004567,
                "delta_pct": -0.37,
                "efficiency_per_bp": 1.3e-4,
                "loss_red_per_wall_min": 0.105,
            },
            {
                "run_id": "accel_balanced",
                "decay": 0.5,
                "boost": 1.5,
                "best_valid_loss": 1.235000,
                "delta_vs_baseline": 0.000433,
                "delta_pct": 0.04,
                "efficiency_per_bp": 1.2e-4,
                "loss_red_per_wall_min": 0.098,
            },
        ],
        "total_runs": 4,
    }


class TestLoadRanking:
    def test_loads_valid_json(self, tmp_path):
        path = tmp_path / "ranking.json"
        path.write_text(json.dumps(_ranking_data()))
        result = load_ranking(path)
        assert result["total_runs"] == 4

    def test_rejects_non_dict(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            load_ranking(path)


class TestGenerateHtml:
    def test_contains_title(self):
        html = generate_html(_ranking_data())
        assert "Accel Param Sweep Dashboard" in html

    def test_contains_best_config(self):
        html = generate_html(_ranking_data())
        assert "accel_conservative" in html

    def test_contains_pairwise_rows(self):
        html = generate_html(_ranking_data())
        assert "accel_balanced" in html
        assert "0.3" in html
        assert "1.1" in html

    def test_improvement_action(self):
        html = generate_html(_ranking_data())
        assert "Improvement found" in html

    def test_no_improvement_action(self):
        data = _ranking_data()
        data["pairwise"][0]["delta_vs_baseline"] = 0.01
        data["pairwise"][0]["best_valid_loss"] = 1.25
        html = generate_html(data)
        assert "No improvement" in html

    def test_neutral_action(self):
        data = _ranking_data()
        data["pairwise"][0]["delta_vs_baseline"] = 0.0
        html = generate_html(data)
        assert "Neutral" in html

    def test_empty_pairwise(self):
        data = _ranking_data()
        data["pairwise"] = []
        html = generate_html(data)
        assert "Accel Param Sweep Dashboard" in html

    def test_none_values_handled(self):
        data = _ranking_data()
        data["best_run"]["best_valid_loss"] = None
        data["pairwise"][0]["efficiency_per_bp"] = None
        html = generate_html(data)
        assert "N/A" in html


class TestCLI:
    def test_cli_with_ranking_json(self, tmp_path):
        ranking_path = tmp_path / "ranking.json"
        ranking_path.write_text(json.dumps(_ranking_data()))
        output_path = tmp_path / "dashboard.html"
        result = subprocess.run(
            [sys.executable, "scripts/generate_sweep_dashboard.py",
             "--ranking-json", str(ranking_path),
             "--output", str(output_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "accel_conservative" in content

    def test_cli_with_sweep_dir(self, tmp_path):
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        (analysis_dir / "ranking.json").write_text(json.dumps(_ranking_data()))
        result = subprocess.run(
            [sys.executable, "scripts/generate_sweep_dashboard.py", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert (analysis_dir / "dashboard.html").exists()

    def test_cli_missing_input_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_sweep_dashboard.py", "/nonexistent"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_cli_no_args_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_sweep_dashboard.py"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
