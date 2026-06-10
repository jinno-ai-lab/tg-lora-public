"""Tests for scripts/compare_experiment_configs.py — cross-config experiment comparator."""
from __future__ import annotations

import json


from scripts.compare_experiment_configs import (
    ComparisonMatrix,
    ExperimentSummary,
    build_comparison_matrix,
    discover_experiments,
    format_as_json,
    format_as_markdown,
    rank_experiments,
)


def _make_run(tmp_path, run_id, config=None, metrics=None):
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    header = {"type": "run_header", "run_id": run_id}
    if config:
        header.update(config)

    step = {"type": "step", "loss_train": 1.0, "total_backward_passes": 10}
    footer_data = {"type": "run_footer", "run_id": run_id}
    if metrics:
        footer_data.update(metrics)

    jsonl = run_dir / "run_metrics.jsonl"
    jsonl.write_text(
        json.dumps(header) + "\n" +
        json.dumps(step) + "\n" +
        json.dumps(footer_data) + "\n"
    )
    return run_dir


class TestDiscoverExperiments:
    def test_discovers_runs(self, tmp_path):
        _make_run(tmp_path, "run_a", metrics={"best_valid_loss": 1.2})
        _make_run(tmp_path, "run_b", metrics={"best_valid_loss": 0.9})
        exps = discover_experiments(str(tmp_path))
        assert len(exps) >= 2

    def test_extracts_config(self, tmp_path):
        _make_run(tmp_path, "run_cfg", config={"tg_lora_K": 5, "learning_rate": 0.001})
        exps = discover_experiments(str(tmp_path))
        assert len(exps) >= 1
        exp = exps[0]
        assert exp.config.get("tg_lora_K") == 5

    def test_empty_dir(self, tmp_path):
        exps = discover_experiments(str(tmp_path))
        assert exps == []

    def test_parse_failure_produces_warning(self, tmp_path, monkeypatch):
        """When parse_jsonl fails during step extraction, a warning is recorded."""
        _make_run(tmp_path, "run_ok", metrics={"best_valid_loss": 1.0})

        import scripts.compare_experiment_configs as mod
        monkeypatch.setattr(
            mod, "parse_jsonl",
            lambda path: (_ for _ in ()).throw(ValueError("simulated parse failure")),
        )

        exps = discover_experiments(str(tmp_path))
        assert len(exps) >= 1
        assert any(exp.parse_warnings for exp in exps)
        assert "simulated parse failure" in exps[0].parse_warnings[0]


class TestBuildComparisonMatrix:
    def test_builds_matrix(self):
        exps = [
            ExperimentSummary(
                run_id="a",
                config={"tg_lora_K": 5},
                metrics={"best_valid_loss": 1.2},
            ),
            ExperimentSummary(
                run_id="b",
                config={"tg_lora_K": 3, "learning_rate": 0.001},
                metrics={"best_valid_loss": 0.9, "total_wall_seconds": 100},
            ),
        ]
        matrix = build_comparison_matrix(exps)
        assert len(matrix.experiments) == 2
        assert "tg_lora_K" in matrix.parameters
        assert "best_valid_loss" in matrix.metrics_cols

    def test_empty_experiments(self):
        matrix = build_comparison_matrix([])
        assert matrix.experiments == []
        assert matrix.parameters == []


class TestRankExperiments:
    def test_ranking_by_loss(self):
        exps = [
            ExperimentSummary(run_id="bad", metrics={"best_valid_loss": 2.0}),
            ExperimentSummary(run_id="good", metrics={"best_valid_loss": 0.5}),
            ExperimentSummary(run_id="mid", metrics={"best_valid_loss": 1.0}),
        ]
        matrix = ComparisonMatrix(experiments=exps, parameters=[], metrics_cols=[])
        ranked = rank_experiments(matrix, "best_valid_loss")
        assert ranked[0]["run_id"] == "good"
        assert ranked[0]["rank"] == 1
        assert ranked[-1]["run_id"] == "bad"

    def test_missing_metric_excluded(self):
        exps = [
            ExperimentSummary(run_id="a", metrics={"best_valid_loss": 1.0}),
            ExperimentSummary(run_id="b", metrics={}),
        ]
        matrix = ComparisonMatrix(experiments=exps, parameters=[], metrics_cols=[])
        ranked = rank_experiments(matrix, "best_valid_loss")
        assert len(ranked) == 1
        assert ranked[0]["run_id"] == "a"

    def test_no_valid_experiments(self):
        exps = [
            ExperimentSummary(run_id="a", metrics={}),
        ]
        matrix = ComparisonMatrix(experiments=exps, parameters=[], metrics_cols=[])
        ranked = rank_experiments(matrix, "best_valid_loss")
        assert ranked == []


class TestFormatAsMarkdown:
    def test_basic_table(self):
        exps = [
            ExperimentSummary(
                run_id="run_a",
                config={"tg_lora_K": 5},
                metrics={"best_valid_loss": 1.2},
            ),
            ExperimentSummary(
                run_id="run_b",
                config={"tg_lora_K": 3},
                metrics={"best_valid_loss": 0.9},
            ),
        ]
        matrix = ComparisonMatrix(
            experiments=exps,
            parameters=["tg_lora_K"],
            metrics_cols=["best_valid_loss"],
        )
        md = format_as_markdown(matrix, "best_valid_loss")
        assert "| rank |" in md
        assert "|---|" in md
        assert "run_b" in md  # best one ranked first
        assert "run_a" in md

    def test_empty_experiments(self):
        matrix = ComparisonMatrix(experiments=[], parameters=[], metrics_cols=[])
        md = format_as_markdown(matrix, "best_valid_loss")
        assert "No experiments" in md

    def test_markdown_includes_warnings_section(self):
        exps = [
            ExperimentSummary(
                run_id="bad_run",
                config={"tg_lora_K": 5},
                metrics={"best_valid_loss": 2.0},
                parse_warnings=["Failed to parse data.jsonl: invalid JSON at line 2"],
            ),
        ]
        matrix = ComparisonMatrix(
            experiments=exps,
            parameters=["tg_lora_K"],
            metrics_cols=["best_valid_loss"],
        )
        md = format_as_markdown(matrix, "best_valid_loss")
        assert "Parse Warnings" in md
        assert "data.jsonl" in md


class TestFormatAsJson:
    def test_json_structure(self):
        exps = [
            ExperimentSummary(
                run_id="run_a",
                config={"tg_lora_K": 5},
                metrics={"best_valid_loss": 1.2},
            ),
        ]
        matrix = ComparisonMatrix(
            experiments=exps,
            parameters=["tg_lora_K"],
            metrics_cols=["best_valid_loss"],
        )
        result = format_as_json(matrix, "best_valid_loss")
        assert result["metric"] == "best_valid_loss"
        assert result["num_experiments"] == 1
        assert len(result["ranked_experiments"]) == 1
        assert result["ranked_experiments"][0]["rank"] == 1

    def test_ranking_order(self):
        exps = [
            ExperimentSummary(run_id="a", metrics={"best_valid_loss": 2.0}),
            ExperimentSummary(run_id="b", metrics={"best_valid_loss": 0.5}),
        ]
        matrix = ComparisonMatrix(experiments=exps, parameters=[], metrics_cols=[])
        result = format_as_json(matrix, "best_valid_loss")
        assert result["ranked_experiments"][0]["run_id"] == "b"

    def test_json_includes_warnings(self):
        exps = [
            ExperimentSummary(
                run_id="bad_run",
                metrics={"best_valid_loss": 2.0},
                parse_warnings=["Failed to parse run_metrics.jsonl: bad data"],
            ),
        ]
        matrix = ComparisonMatrix(experiments=exps, parameters=[], metrics_cols=[])
        result = format_as_json(matrix, "best_valid_loss")
        assert "parse_warnings" in result
        assert len(result["parse_warnings"]) == 1
