"""TASK-0129: End-to-end parse_warnings tests with intentionally corrupt JSONL.

Verifies that corrupt JSONL entries are handled gracefully throughout the full
comparison pipeline: gather_runs → render_dashboard → format_json.
Tests both compare_runs and compare_experiment_configs modules.
"""

from pathlib import Path

import orjson

from scripts.compare_runs import (
    format_json,
    gather_runs,
    render_dashboard,
)
from scripts.compare_experiment_configs import (
    ExperimentSummary,
    build_comparison_matrix,
    discover_experiments,
    format_as_json,
    format_as_markdown,
)


def _write_valid_jsonl(run_dir: Path, run_id: str, mode: str = "baseline",
                       best_valid_loss: float = 2.5, final_train_loss: float = 2.6,
                       steps: int = 2) -> Path:
    """Write a well-formed run_metrics.jsonl and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [{"type": "run_header", "run_id": run_id, "mode": mode, "model_name": "test"}]
    for i in range(steps):
        records.append({
            "type": "step", "step": i + 1, "loss_train": final_train_loss + 0.1 * (steps - i),
            "backward_passes": 1, "total_backward_passes": i + 1,
            "elapsed_seconds": 10.0 * (i + 1),
        })
    records.append({
        "type": "run_footer", "total_wall_seconds": 60.0,
        "best_valid_loss": best_valid_loss, "best_valid_step": 1,
        "final_train_loss": final_train_loss,
    })
    path = run_dir / "run_metrics.jsonl"
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
    return path


def _write_corrupt_jsonl(run_dir: Path, run_id: str) -> Path:
    """Write a JSONL with valid header + corrupt middle record + valid footer."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        orjson.dumps({"type": "run_header", "run_id": run_id, "mode": "tg_lora", "model_name": "test"}),
        b'{"type": "step", "loss_train": 3.0, "broke',
        orjson.dumps({"type": "run_footer", "total_wall_seconds": 60.0, "best_valid_loss": 2.8, "best_valid_step": 1, "final_train_loss": 2.9}),
    ]
    path = run_dir / "run_metrics.jsonl"
    path.write_bytes(b"\n".join(lines) + b"\n")
    return path


def _write_partial_jsonl(run_dir: Path, run_id: str) -> Path:
    """Write a JSONL with valid header + step records but NO footer."""
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"type": "run_header", "run_id": run_id, "mode": "baseline", "model_name": "test"},
        {"type": "step", "step": 1, "loss_train": 3.0, "backward_passes": 1, "total_backward_passes": 1},
    ]
    path = run_dir / "run_metrics.jsonl"
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
    return path


class TestCorruptJsonlEndToEnd:
    """Full-pipeline tests: corrupt JSONL → gather_runs → dashboard/JSON."""

    def test_corrupt_run_skipped_by_gather_runs(self, tmp_path):
        """Corrupt JSONL is skipped by gather_runs; only valid runs appear."""
        _write_valid_jsonl(tmp_path / "run_good", "run_good", best_valid_loss=2.0)
        _write_corrupt_jsonl(tmp_path / "run_bad", "run_bad")

        runs = gather_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run_good"
        assert runs[0]["parse_warnings"] == []

    def test_partial_run_listed_without_enrichment(self, tmp_path):
        """Run with header + steps but no footer is listed by list_runs
        (parse_jsonl succeeds) but lacks enrichment fields."""
        _write_valid_jsonl(tmp_path / "run_full", "run_full", best_valid_loss=2.0)
        _write_partial_jsonl(tmp_path / "run_partial", "run_partial")

        runs = gather_runs(tmp_path)
        ids = {r["run_id"] for r in runs}
        assert "run_full" in ids
        assert "run_partial" in ids

        partial = next(r for r in runs if r["run_id"] == "run_partial")
        assert partial.get("best_valid_loss") is None
        assert partial.get("final_train_loss") is None
        assert partial["parse_warnings"] == []

    def test_mixed_clean_and_corrupt_dashboard(self, tmp_path, capsys):
        """Dashboard renders correctly with mix of valid and corrupt runs."""
        _write_valid_jsonl(tmp_path / "run_clean", "run_clean", best_valid_loss=2.0)
        _write_corrupt_jsonl(tmp_path / "run_corrupt", "run_corrupt")

        runs = gather_runs(tmp_path)
        render_dashboard(runs)
        captured = capsys.readouterr()

        assert "run_clean" in captured.out
        assert "Parse Warnings" not in captured.out

    def test_mixed_clean_and_corrupt_json_output(self, tmp_path):
        """JSON output contains valid runs but no parse_warnings when corrupt runs are skipped."""
        _write_valid_jsonl(tmp_path / "run_clean", "run_clean", best_valid_loss=2.0)
        _write_corrupt_jsonl(tmp_path / "run_corrupt", "run_corrupt")

        runs = gather_runs(tmp_path)
        output = format_json(runs)
        parsed = orjson.loads(output)

        assert len(parsed["runs"]) == 1
        assert parsed["runs"][0]["run_id"] == "run_clean"
        assert "parse_warnings" not in parsed

    def test_warnings_surface_when_gather_reparse_fails(self, tmp_path, capsys, monkeypatch):
        """When gather_runs successfully lists a run but fails to re-parse it,
        parse_warnings is populated and surfaces in both dashboard and JSON."""
        _write_valid_jsonl(tmp_path / "run_a", "run_a", best_valid_loss=2.5)

        import scripts.compare_runs as cr

        def failing_parse(path):
            raise ValueError("file changed between reads")

        # Patch cr.parse_jsonl (the direct import used by gather_runs' re-parse).
        # list_runs calls rq.parse_jsonl internally, so it still succeeds.
        monkeypatch.setattr(cr, "parse_jsonl", failing_parse)
        runs = gather_runs(tmp_path)

        assert len(runs) == 1
        assert len(runs[0]["parse_warnings"]) == 1
        assert "file changed between reads" in runs[0]["parse_warnings"][0]

        captured = capsys.readouterr()
        assert "WARNING" in captured.err

        # Dashboard shows the warning panel
        render_dashboard(runs)
        captured = capsys.readouterr()
        assert "Parse Warnings" in captured.out
        assert "file changed between reads" in captured.out

        # JSON output includes parse_warnings
        output = format_json(runs)
        parsed = orjson.loads(output)
        assert "parse_warnings" in parsed
        assert "file changed between reads" in parsed["parse_warnings"][0]

    def test_multiple_corrupt_runs_all_skipped(self, tmp_path):
        """Multiple corrupt runs are all skipped; only clean runs remain."""
        _write_valid_jsonl(tmp_path / "run_ok", "run_ok", best_valid_loss=1.5)
        _write_corrupt_jsonl(tmp_path / "run_bad1", "run_bad1")
        _write_corrupt_jsonl(tmp_path / "run_bad2", "run_bad2")

        runs = gather_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run_ok"


class TestCompareExperimentConfigsCorrupt:
    """parse_warnings handling in compare_experiment_configs with corrupt data."""

    def test_corrupt_jsonl_in_experiment_gathering(self, tmp_path, monkeypatch):
        """Corrupt JSONL during experiment gathering produces parse_warnings."""
        run_dir = tmp_path / "exp_a"
        _write_valid_jsonl(run_dir, "exp_a", mode="tg_lora", best_valid_loss=2.0)

        import scripts.compare_experiment_configs as cec


        def failing_parse(path):
            # First call is from list_runs (succeeds), second from discover_experiments (fails)
            raise ValueError("corrupt data in experiment")

        # Patch the direct import in cec so it fails during re-parse in discover_experiments
        monkeypatch.setattr(cec, "parse_jsonl", failing_parse)
        experiments = discover_experiments(tmp_path)

        assert len(experiments) >= 1
        assert len(experiments[0].parse_warnings) >= 1
        assert "corrupt data" in experiments[0].parse_warnings[0]

    def test_corrupt_experiment_warnings_in_markdown(self):
        """Warnings from corrupt experiments appear in markdown output."""
        exp = ExperimentSummary(
            run_id="corrupt_exp",
            config={"K": 3},
            metrics={"best_valid_loss": 2.5},
            parse_warnings=["Failed to parse run_metrics.jsonl: bad data at line 5"],
        )
        matrix = build_comparison_matrix([exp])
        md = format_as_markdown(matrix)

        assert "Parse Warnings" in md
        assert "bad data at line 5" in md

    def test_corrupt_experiment_warnings_in_json(self, tmp_path):
        """Warnings from corrupt experiments appear in JSON output."""
        exp = ExperimentSummary(
            run_id="corrupt_exp",
            config={"K": 3},
            metrics={"best_valid_loss": 2.5},
            parse_warnings=["Failed to parse: invalid JSON"],
        )
        matrix = build_comparison_matrix([exp])
        result = format_as_json(matrix)

        assert "parse_warnings" in result
        assert "invalid JSON" in result["parse_warnings"][0]

    def test_clean_experiment_no_warnings_in_output(self):
        """Clean experiments produce no parse_warnings in output."""
        exp = ExperimentSummary(
            run_id="clean_exp",
            config={"K": 3},
            metrics={"best_valid_loss": 2.5},
            parse_warnings=[],
        )
        matrix = build_comparison_matrix([exp])

        md = format_as_markdown(matrix)
        assert "Parse Warnings" not in md

        result = format_as_json(matrix)
        assert "parse_warnings" not in result
