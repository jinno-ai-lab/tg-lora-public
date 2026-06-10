"""Tests for scripts/summarize_psa_sweep.py — ablation summary functions."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestImportHealth:
    def test_module_imports(self):
        mod = importlib.import_module("scripts.summarize_psa_sweep")
        assert hasattr(mod, "main")
        assert hasattr(mod, "load_run")
        assert hasattr(mod, "classify_run")
        assert hasattr(mod, "_extract_psa_lt_stats")
        assert hasattr(mod, "_aggregate_lt_stats")


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.summarize_psa_sweep", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestClassifyRun:
    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.summarize_psa_sweep import classify_run
        self.classify = classify_run

    def test_baseline_plain(self):
        assert self.classify("baseline_plain") == "baseline"

    def test_lawa_only(self):
        assert self.classify("lawa_only") == "lawa"

    def test_psa_default(self):
        assert self.classify("psa_default") == "psa_default"

    def test_gamma_sweep(self):
        assert self.classify("gamma_0.5_reset_on") == "psa_gamma"
        assert self.classify("gamma_1.0_reset_off") == "psa_gamma"

    def test_history_sweep(self):
        assert self.classify("history_3") == "psa_history"
        assert self.classify("history_10") == "psa_history"

    def test_interval_sweep(self):
        assert self.classify("interval_1") == "psa_interval"
        assert self.classify("interval_5") == "psa_interval"

    def test_unknown(self):
        assert self.classify("some_random_name") == "unknown"


class TestExtractPsaLtStats:
    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.summarize_psa_sweep import _extract_psa_lt_stats
        self.extract = _extract_psa_lt_stats

    def test_extracts_lt_metrics(self):
        records = [
            {"psa_lt_attention_out_amp_mean": 1.2, "psa_lt_attention_out_prior_stability": 0.9, "other": 5},
            {"psa_lt_attention_out_amp_mean": 1.3, "psa_lt_attention_out_prior_stability": 0.88},
            {"psa_lt_mlp_amp_mean": 0.8, "psa_lt_mlp_amp_std": 0.1},
        ]
        result = self.extract(records)
        assert "attention_out" in result
        assert "amp_mean" in result["attention_out"]
        assert result["attention_out"]["amp_mean"] == [1.2, 1.3]
        assert result["attention_out"]["prior_stability"] == [0.9, 0.88]
        assert "mlp" in result
        assert result["mlp"]["amp_mean"] == [0.8]

    def test_empty_records(self):
        assert self.extract([]) == {}

    def test_no_psa_lt_keys(self):
        assert self.extract([{"loss": 2.5, "cycle": 1}]) == {}


class TestAggregateLtStats:
    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.summarize_psa_sweep import _aggregate_lt_stats
        self.aggregate = _aggregate_lt_stats

    def test_aggregates_mean_and_std(self):
        lt_series = {
            "attention_out": {"amp_mean": [1.2, 1.4, 1.3]},
        }
        result = self.aggregate(lt_series)
        assert "attention_out" in result
        assert result["attention_out"]["amp_mean_mean"] == pytest.approx(1.3)
        # var = population variance, std = sqrt(var)
        expected_var = ((1.2 - 1.3) ** 2 + (1.4 - 1.3) ** 2 + (1.3 - 1.3) ** 2) / 3
        expected_std = expected_var ** 0.5
        assert result["attention_out"]["amp_mean_std"] == pytest.approx(expected_std, abs=0.001)
        assert result["attention_out"]["amp_mean_n"] == 3.0

    def test_single_value_no_std(self):
        lt_series = {"mlp": {"amp_mean": [1.0]}}
        result = self.aggregate(lt_series)
        assert "amp_mean_std" not in result["mlp"]

    def test_empty(self):
        assert self.aggregate({}) == {}


class TestComputeEfficiency:
    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.summarize_psa_sweep import _compute_efficiency
        self.efficiency = _compute_efficiency

    def test_basic(self):
        run = {
            "records": [{"loss_train": 3.0, "total_backward_passes": 100}],
            "footer": {"best_valid_loss": 2.0, "total_wall_seconds": 300},
        }
        eff = self.efficiency(run)
        assert eff["loss_reduction"] == pytest.approx(1.0)
        assert eff["loss_red_per_bp"] == pytest.approx(0.01)
        assert eff["loss_red_per_wall_min"] == pytest.approx(0.2)

    def test_no_footer_best(self):
        run = {
            "records": [{"loss_train": 3.0, "total_backward_passes": 100}],
            "footer": {"total_wall_seconds": 300},
        }
        eff = self.efficiency(run)
        assert eff["loss_reduction"] is None

    def test_empty_records(self):
        run = {"records": [], "footer": {"best_valid_loss": 2.0, "total_wall_seconds": 100}}
        eff = self.efficiency(run)
        assert eff["loss_reduction"] is None


class TestParseConfigName:
    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.summarize_psa_sweep import parse_config_name
        self.parse = parse_config_name

    def test_gamma_reset(self):
        result = self.parse("gamma_0.5_reset_on")
        assert result["gamma"] == pytest.approx(0.5)
        assert result["regime_reset"] is True

    def test_no_match(self):
        result = self.parse("psa_default")
        assert result["gamma"] is None
        assert result["regime_reset"] is None


class TestLoadRun:
    def _write_jsonl(self, path: Path, objects: list[dict]):
        with open(path, "w") as f:
            for obj in objects:
                f.write(json.dumps(obj) + "\n")

    def test_complete_run(self, tmp_path):
        from scripts.summarize_psa_sweep import load_run

        path = tmp_path / "run_metrics.jsonl"
        self._write_jsonl(path, [
            {"type": "run_header"},
            {"type": "step", "cycle": 1, "loss_train": 2.5, "psa_lt_attention_out_amp_mean": 1.2},
            {"type": "run_footer", "best_valid_loss": 2.1, "total_wall_seconds": 60},
        ])
        result = load_run(path)
        assert result is not None
        assert len(result["records"]) == 1
        assert result["records"][0]["psa_lt_attention_out_amp_mean"] == 1.2

    def test_no_footer(self, tmp_path):
        from scripts.summarize_psa_sweep import load_run

        path = tmp_path / "run_metrics.jsonl"
        self._write_jsonl(path, [{"type": "step", "cycle": 1}])
        assert load_run(path) is None


class TestEndToEnd:
    """Integration test with synthetic ablation directory."""

    def _write_run(self, run_dir: Path, name: str, best_vl: float, records: list[dict]):
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "run_metrics.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"type": "run_header", "name": name}) + "\n")
            for rec in records:
                f.write(json.dumps({"type": "step", **rec}) + "\n")
            f.write(json.dumps({
                "type": "run_footer",
                "best_valid_loss": best_vl,
                "total_wall_seconds": 100,
                "tg_lora_summary": {},
            }) + "\n")

    def test_full_ablation_summary(self, tmp_path, capsys):
        self._write_run(tmp_path / "baseline_plain", "baseline_plain", 2.5, [
            {"loss_train": 3.0, "total_backward_passes": 100},
        ])
        self._write_run(tmp_path / "lawa_only", "lawa_only", 2.4, [
            {"loss_train": 3.0, "total_backward_passes": 100, "psa_regime": "stable"},
        ])
        self._write_run(tmp_path / "psa_default", "psa_default", 2.3, [
            {
                "loss_train": 3.0,
                "total_backward_passes": 100,
                "psa_gain_mean": 0.5,
                "psa_regime": "stable",
                "psa_regime_transitions": 2,
                "psa_lt_attention_out_amp_mean": 1.2,
                "psa_lt_attention_out_prior_stability": 0.9,
                "psa_lt_mlp_amp_mean": 0.8,
                "psa_lt_mlp_amp_std": 0.1,
            },
        ])

        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.summarize_psa_sweep", "--sweep-dir", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = result.stdout
        assert "GOAL" in output
        assert "§3.3" in output
        assert "baseline_plain" in output
        assert "lawa_only" in output
        assert "psa_default" in output
        assert "Per-Layer-Type" in output
        assert "out_proj" in output or "PSA WINS" in output

        json_path = tmp_path / "psa_sweep_summary.json"
        assert json_path.exists()
        summary = json.loads(json_path.read_text())
        assert len(summary["runs"]) == 3
