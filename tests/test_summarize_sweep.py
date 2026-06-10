"""Tests for scripts/summarize_sweep.py — verifies import health and core functions."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest



# ---------------------------------------------------------------------------
# Import health
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.summarize_sweep")
        assert hasattr(mod, "main")
        assert hasattr(mod, "load_run")


# ---------------------------------------------------------------------------
# --help CLI
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.summarize_sweep", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "sweep" in result.stdout.lower()


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _compute_efficiency — pure function tests
# ---------------------------------------------------------------------------


class TestComputeEfficiency:
    """Focused unit tests for _compute_efficiency numerical correctness."""

    def _make_run(self, *, initial_loss=3.0, best_valid=2.0, total_bp=240, wall_sec=2300):
        return {
            "records": [{"loss_train": initial_loss, "total_backward_passes": total_bp}],
            "footer": {"best_valid_loss": best_valid, "total_wall_seconds": wall_sec},
        }

    def test_basic_computation(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = self._make_run(initial_loss=3.0, best_valid=2.0, total_bp=240, wall_sec=2300)
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] == pytest.approx(1.0)
        assert eff["loss_red_per_bp"] == pytest.approx(1.0 / 240)
        assert eff["loss_red_per_wall_min"] == pytest.approx(1.0 / (2300 / 60))

    def test_zero_loss_reduction(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = self._make_run(initial_loss=2.0, best_valid=2.0)
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] == pytest.approx(0.0)
        assert eff["loss_red_per_bp"] == pytest.approx(0.0)
        assert eff["loss_red_per_wall_min"] == pytest.approx(0.0)

    def test_loss_increase_gives_negative_reduction(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = self._make_run(initial_loss=2.0, best_valid=2.5)
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] == pytest.approx(-0.5)
        assert eff["loss_red_per_bp"] is not None and eff["loss_red_per_bp"] < 0

    def test_zero_backward_passes_gives_none_per_bp(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = self._make_run(total_bp=0)
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] == pytest.approx(1.0)
        assert eff["loss_red_per_bp"] is None
        assert eff["loss_red_per_wall_min"] is not None

    def test_zero_wall_time_gives_none_per_min(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = self._make_run(wall_sec=0)
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] == pytest.approx(1.0)
        assert eff["loss_red_per_bp"] is not None
        assert eff["loss_red_per_wall_min"] is None

    def test_empty_records_gives_none_reduction(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = {"records": [], "footer": {"best_valid_loss": 2.0, "total_wall_seconds": 100}}
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] is None
        assert eff["loss_red_per_bp"] is None
        assert eff["loss_red_per_wall_min"] is None

    def test_missing_best_valid_loss_gives_none(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = {
            "records": [{"loss_train": 3.0, "total_backward_passes": 100}],
            "footer": {"total_wall_seconds": 500},
        }
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] is None

    def test_precision_with_large_bp(self):
        from scripts.summarize_sweep import _compute_efficiency

        run = self._make_run(initial_loss=3.0, best_valid=2.99, total_bp=12000, wall_sec=7200)
        eff = _compute_efficiency(run)
        assert eff["loss_reduction"] == pytest.approx(0.01)
        expected_per_bp = 0.01 / 12000
        assert eff["loss_red_per_bp"] == pytest.approx(expected_per_bp, rel=1e-9)


class TestLoadRun:
    def _write_jsonl(self, path: Path, objects: list[dict]):
        with open(path, "w") as f:
            for obj in objects:
                f.write(json.dumps(obj) + "\n")

    def test_complete_run(self, tmp_path):
        from scripts.summarize_sweep import load_run

        path = tmp_path / "run_metrics.jsonl"
        self._write_jsonl(
            path,
            [
                {"type": "run_header", "model": "test"},
                {"type": "step", "cycle": 1, "loss_train": 2.5, "tg_lora_accepted": True},
                {"type": "step", "cycle": 2, "loss_train": 2.3, "tg_lora_accepted": False},
                {
                    "type": "run_footer",
                    "best_valid_loss": 2.1,
                    "final_train_loss": 2.3,
                    "total_wall_seconds": 120,
                    "tg_lora_summary": {"current_K": 3},
                },
            ],
        )
        result = load_run(path)
        assert result is not None
        assert result["header"]["model"] == "test"
        assert len(result["records"]) == 2
        assert result["footer"]["best_valid_loss"] == 2.1

    def test_missing_footer_returns_none(self, tmp_path):
        from scripts.summarize_sweep import load_run

        path = tmp_path / "run_metrics.jsonl"
        self._write_jsonl(
            path,
            [
                {"type": "run_header", "model": "test"},
                {"type": "step", "cycle": 1},
            ],
        )
        result = load_run(path)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        from scripts.summarize_sweep import load_run

        path = tmp_path / "run_metrics.jsonl"
        path.write_text("")
        result = load_run(path)
        assert result is None

    def test_acceptance_rate_computation(self, tmp_path):
        from scripts.summarize_sweep import load_run

        path = tmp_path / "run_metrics.jsonl"
        self._write_jsonl(
            path,
            [
                {"type": "run_header"},
                {"type": "step", "tg_lora_accepted": True},
                {"type": "step", "tg_lora_accepted": True},
                {"type": "step", "tg_lora_accepted": False},
                {"type": "run_footer", "best_valid_loss": 1.0},
            ],
        )
        result = load_run(path)
        assert result is not None
        accepted = sum(1 for r in result["records"] if r.get("tg_lora_accepted"))
        assert accepted == 2
        assert len(result["records"]) == 3
