"""Tests for scripts/analyze_benchmark.py — REQ-168 (TC-168-01, TC-168-02)."""

import json
from pathlib import Path

import pytest

from scripts.analyze_benchmark import (
    analyze,
    format_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


BASELINE_DATA = {
    "results": {
        "truthfulqa_mc2": {
            "acc,none": 0.4523,
            "perplexity,none": 12.34,
        },
        "mmlu": {
            "acc_norm,none": 0.5100,
        },
    },
}

TG_LORA_DATA = {
    "results": {
        "truthfulqa_mc2": {
            "acc,none": 0.4519,
            "perplexity,none": 12.10,
        },
        "mmlu": {
            "acc_norm,none": 0.5230,
        },
    },
}


# ---------------------------------------------------------------------------
# TC-168-01: baseline/TG-LoRAメトリクス差分計算
# ---------------------------------------------------------------------------


class TestMetricDeltaComputation:
    """TC-168-01: Each metric's delta is correctly computed."""

    def test_delta_values(self, tmp_path):
        b_path = _write_json(tmp_path, "baseline.json", BASELINE_DATA)
        t_path = _write_json(tmp_path, "tg_lora.json", TG_LORA_DATA)
        deltas = analyze(b_path, t_path)

        tqa_acc = deltas["truthfulqa_mc2/acc,none"]
        assert tqa_acc["baseline"] == pytest.approx(0.4523)
        assert tqa_acc["tg_lora"] == pytest.approx(0.4519)
        assert tqa_acc["delta"] == pytest.approx(0.4519 - 0.4523)

    def test_all_metrics_present(self, tmp_path):
        b_path = _write_json(tmp_path, "baseline.json", BASELINE_DATA)
        t_path = _write_json(tmp_path, "tg_lora.json", TG_LORA_DATA)
        deltas = analyze(b_path, t_path)

        expected_keys = {
            "truthfulqa_mc2/acc,none",
            "truthfulqa_mc2/perplexity,none",
            "mmlu/acc_norm,none",
        }
        assert expected_keys == set(deltas.keys())

    def test_perplexity_delta(self, tmp_path):
        b_path = _write_json(tmp_path, "baseline.json", BASELINE_DATA)
        t_path = _write_json(tmp_path, "tg_lora.json", TG_LORA_DATA)
        deltas = analyze(b_path, t_path)

        ppl = deltas["truthfulqa_mc2/perplexity,none"]
        assert ppl["delta"] == pytest.approx(12.10 - 12.34)

    def test_positive_delta(self, tmp_path):
        b_path = _write_json(tmp_path, "baseline.json", BASELINE_DATA)
        t_path = _write_json(tmp_path, "tg_lora.json", TG_LORA_DATA)
        deltas = analyze(b_path, t_path)

        mmlu = deltas["mmlu/acc_norm,none"]
        assert mmlu["delta"] == pytest.approx(0.5230 - 0.5100)
        assert mmlu["delta"] > 0

    def test_format_report_contains_deltas(self, tmp_path):
        b_path = _write_json(tmp_path, "baseline.json", BASELINE_DATA)
        t_path = _write_json(tmp_path, "tg_lora.json", TG_LORA_DATA)
        deltas = analyze(b_path, t_path)
        report = format_report(deltas)
        assert "delta=" in report
        assert "truthfulqa_mc2" in report


# ---------------------------------------------------------------------------
# TC-168-02: 欠損メトリクス時のエラーハンドリング
# ---------------------------------------------------------------------------


class TestMissingMetricsHandling:
    """TC-168-02: Missing metrics are skipped; available metrics still computed."""

    def test_missing_metric_in_tg_lora(self, tmp_path):
        """TG-LoRA is missing one metric that baseline has."""
        tg_partial = {
            "results": {
                "truthfulqa_mc2": {
                    "acc,none": 0.4519,
                    # perplexity missing
                },
                "mmlu": {
                    "acc_norm,none": 0.5230,
                },
            },
        }
        b_path = _write_json(tmp_path, "baseline.json", BASELINE_DATA)
        t_path = _write_json(tmp_path, "tg_lora.json", tg_partial)
        deltas = analyze(b_path, t_path)

        tqa_ppl = deltas["truthfulqa_mc2/perplexity,none"]
        assert "baseline" in tqa_ppl
        assert "tg_lora" not in tqa_ppl
        assert "delta" not in tqa_ppl

        tqa_acc = deltas["truthfulqa_mc2/acc,none"]
        assert "delta" in tqa_acc

    def test_missing_metric_in_baseline(self, tmp_path):
        """Baseline is missing one metric that TG-LoRA has."""
        baseline_partial = {
            "results": {
                "truthfulqa_mc2": {
                    "acc,none": 0.4523,
                    # perplexity missing
                },
                "mmlu": {
                    "acc_norm,none": 0.5100,
                },
            },
        }
        b_path = _write_json(tmp_path, "baseline.json", baseline_partial)
        t_path = _write_json(tmp_path, "tg_lora.json", TG_LORA_DATA)
        deltas = analyze(b_path, t_path)

        tqa_ppl = deltas["truthfulqa_mc2/perplexity,none"]
        assert "baseline" not in tqa_ppl
        assert "tg_lora" in tqa_ppl
        assert "delta" not in tqa_ppl

    def test_all_metrics_missing_from_one_side(self, tmp_path):
        """TG-LoRA has no metrics — all deltas lack tg_lora key."""
        empty_tg = {"results": {}}
        b_path = _write_json(tmp_path, "baseline.json", BASELINE_DATA)
        t_path = _write_json(tmp_path, "tg_lora.json", empty_tg)
        deltas = analyze(b_path, t_path)

        for entry in deltas.values():
            assert "delta" not in entry
            assert "baseline" in entry
