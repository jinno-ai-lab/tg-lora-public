"""Tests for scripts/export_paper_results.py — paper results export tool."""
from __future__ import annotations

import json

import pytest

from scripts.export_paper_results import (
    export_csv,
    generate_latex_table,
    generate_markdown_table,
    load_aggregate,
)


def _make_summary(tmp_path, per_seed=None):
    data = {
        "per_seed": per_seed or {
            "seed_0": {"loss": 1.0, "accuracy": 0.9},
            "seed_1": {"loss": 1.2, "accuracy": 0.88},
            "seed_2": {"loss": 0.8, "accuracy": 0.92},
        },
        "aggregate": {},
    }
    p = tmp_path / "aggregate_summary.json"
    p.write_text(json.dumps(data))
    return p


class TestLoadAggregate:
    def test_valid_file(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        assert "per_seed" in data

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_aggregate("/nonexistent/path.json")

    def test_invalid_structure(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"other": True}))
        with pytest.raises(ValueError, match="Invalid"):
            load_aggregate(p)

    def test_list_per_seed(self, tmp_path):
        data = {"per_seed": [{"loss": 1.0}, {"loss": 2.0}], "aggregate": {}}
        p = tmp_path / "summary.json"
        p.write_text(json.dumps(data))
        result = load_aggregate(p)
        assert len(result["per_seed"]) == 2


class TestGenerateLatexTable:
    def test_contains_table_env(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        latex = generate_latex_table(data)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\begin{tabular}" in latex
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex

    def test_contains_metric_names(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        latex = generate_latex_table(data)
        assert "accuracy" in latex
        assert "loss" in latex

    def test_contains_ci_columns(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        latex = generate_latex_table(data)
        assert "CI" in latex

    def test_empty_seeds(self, tmp_path):
        data = {"per_seed": {}, "aggregate": {}}
        latex = generate_latex_table(data)
        assert "\\begin{table}" in latex


class TestGenerateMarkdownTable:
    def test_table_format(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        md = generate_markdown_table(data)
        assert "| Metric |" in md
        assert "|---|" in md

    def test_contains_values(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        md = generate_markdown_table(data)
        assert "accuracy" in md
        assert "loss" in md

    def test_empty_seeds(self, tmp_path):
        data = {"per_seed": {}, "aggregate": {}}
        md = generate_markdown_table(data)
        assert "| Metric |" in md


class TestExportCsv:
    def test_creates_file(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        out = tmp_path / "out.csv"
        export_csv(data, out)
        assert out.exists()
        content = out.read_text()
        assert "metric" in content
        assert "loss" in content

    def test_csv_rows(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        out = tmp_path / "out.csv"
        export_csv(data, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) >= 2  # header + at least one data row

    def test_creates_parent_dirs(self, tmp_path):
        p = _make_summary(tmp_path)
        data = load_aggregate(p)
        out = tmp_path / "subdir" / "deep" / "out.csv"
        export_csv(data, out)
        assert out.exists()
