"""Tests for scripts/analyze_sensitivity.py — hyperparameter sensitivity analysis."""
from __future__ import annotations

import json

import pytest

from scripts.analyze_sensitivity import (
    _pearson_r,
    compute_correlation_matrix,
    generate_sensitivity_report,
    load_sweep_results,
    rank_sensitivity,
)


class TestPearsonR:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = _pearson_r(x, y)
        assert r == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        r = _pearson_r(x, y)
        assert r == pytest.approx(-1.0, abs=0.001)

    def test_uncorrelated(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, -1.0, 1.0, -1.0, 1.0]
        r = _pearson_r(x, y)
        assert abs(r) < 0.5

    def test_single_element(self):
        r = _pearson_r([1.0], [2.0])
        assert r == 0.0

    def test_zero_variance(self):
        r = _pearson_r([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        assert r == 0.0


class TestComputeCorrelationMatrix:
    def test_basic(self):
        results = [
            {"tg_lora_K": 3, "best_valid_loss": 1.5, "learning_rate": 0.001},
            {"tg_lora_K": 5, "best_valid_loss": 1.2, "learning_rate": 0.001},
            {"tg_lora_K": 7, "best_valid_loss": 0.9, "learning_rate": 0.001},
        ]
        matrix = compute_correlation_matrix(results)
        assert "tg_lora_K" in matrix
        assert "best_valid_loss" in matrix["tg_lora_K"]
        # K increasing, loss decreasing → negative correlation
        assert matrix["tg_lora_K"]["best_valid_loss"] < 0

    def test_explicit_params(self):
        results = [
            {"K": 3, "loss": 1.5},
            {"K": 5, "loss": 1.0},
        ]
        matrix = compute_correlation_matrix(results, params=["K"], metrics=["loss"])
        assert "K" in matrix
        assert "loss" in matrix["K"]

    def test_missing_data_skipped(self):
        results = [
            {"tg_lora_K": 3, "best_valid_loss": 1.5},
            {"best_valid_loss": 1.0},
        ]
        matrix = compute_correlation_matrix(results)
        # Only one pair has K, so correlation should be 0
        assert matrix.get("tg_lora_K", {}).get("best_valid_loss", 0.0) == 0.0


class TestRankSensitivity:
    def test_ranking_order(self):
        correlations = {
            "K": {"loss": 0.9},
            "alpha": {"loss": 0.3},
            "lr": {"loss": 0.1},
        }
        ranked = rank_sensitivity(correlations)
        assert ranked[0][0] == "K"
        assert ranked[-1][0] == "lr"

    def test_empty(self):
        ranked = rank_sensitivity({})
        assert ranked == []


class TestGenerateSensitivityReport:
    def test_full_report(self):
        results = [
            {"tg_lora_K": 3, "best_valid_loss": 1.5, "learning_rate": 0.001},
            {"tg_lora_K": 5, "best_valid_loss": 1.2, "learning_rate": 0.001},
            {"tg_lora_K": 7, "best_valid_loss": 0.9, "learning_rate": 0.001},
        ]
        report = generate_sensitivity_report(results)
        assert report["num_experiments"] == 3
        assert "correlation_matrix" in report
        assert "sensitivity_ranking" in report
        assert len(report["sensitivity_ranking"]) > 0

    def test_output_file(self, tmp_path):
        results = [
            {"tg_lora_K": 3, "best_valid_loss": 1.5},
            {"tg_lora_K": 5, "best_valid_loss": 1.0},
        ]
        out = tmp_path / "report.json"
        generate_sensitivity_report(results, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["num_experiments"] == 2


class TestLoadSweepResults:
    def test_empty_dir(self, tmp_path):
        results = load_sweep_results(str(tmp_path))
        assert results == []

    def test_with_runs(self, tmp_path):
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        header = {"type": "run_header", "tg_lora_K": 5, "learning_rate": 0.001}
        step = {"type": "step", "loss_train": 1.5, "total_backward_passes": 10}
        footer = {"type": "run_footer", "best_valid_loss": 1.2}
        jsonl = run_dir / "run_metrics.jsonl"
        jsonl.write_text(
            json.dumps(header) + "\n" + json.dumps(step) + "\n" + json.dumps(footer) + "\n"
        )
        results = load_sweep_results(str(tmp_path))
        assert len(results) >= 1
        assert results[0].get("best_valid_loss") == 1.2
