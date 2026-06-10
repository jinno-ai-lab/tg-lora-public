"""Tests for scripts/analyze_accel_sweep.py — sweep analysis pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import orjson
import pytest

from scripts.analyze_accel_sweep import (
    analyze_sweep,
    compute_loss_trajectory,
    generate_summary,
    validate_sweep_configs,
    validate_sweep_results,
)

CONFIGS = [
    ("no_accel", 0.99, 1.01),
    ("conservative", 0.3, 1.1),
    ("balanced", 0.5, 1.5),
    ("aggressive", 0.9, 2.0),
]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec) + b"\n")


def _make_run(
    run_id: str,
    decay: float,
    boost: float,
    loss_start: float = 3.0,
    loss_end: float = 2.0,
    cycles: int = 10,
    bp_per_cycle: int = 24,
    wall_per_cycle: float = 230.0,
    accepted_count: int | None = None,
) -> list[dict]:
    header = {
        "type": "run_header",
        "run_id": run_id,
        "mode": "tg_lora",
        "model_name": "Qwen/Qwen3.5-9B",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 1,
        "grad_accumulation": 8,
        "learning_rate": 2e-4,
        "seed": 42,
        "accel_instability_lr_decay": decay,
        "accel_convergence_lr_boost": boost,
    }
    records = [header]
    loss_step = (loss_start - loss_end) / max(cycles, 1)
    if accepted_count is None:
        accepted_count = cycles
    accepted_list = [True] * accepted_count + [False] * (cycles - accepted_count)
    for i in range(cycles):
        loss = loss_start - loss_step * (i + 1)
        records.append({
            "type": "step",
            "step": (i + 1) * bp_per_cycle,
            "cycle": i + 1,
            "loss_train": loss + 0.1,
            "loss_valid": loss,
            "backward_passes": bp_per_cycle,
            "total_backward_passes": (i + 1) * bp_per_cycle,
            "elapsed_seconds": (i + 1) * wall_per_cycle,
            "tg_lora_accepted": accepted_list[i],
            "tg_lora_cosine_sim": 0.5 + 0.03 * i,
            "tg_lora_reduction_rate": 0.1 + 0.005 * i,
            "tg_lora_K": 3,
            "tg_lora_N": 5,
        })
    footer = {
        "type": "run_footer",
        "total_wall_seconds": cycles * wall_per_cycle,
        "best_valid_loss": loss_end,
        "final_train_loss": loss_end + 0.1,
        "best_valid_step": cycles * bp_per_cycle,
        "gpu_peak_mb": 8200,
        "total_cycles": cycles,
    }
    records.append(footer)
    return records


def _write_sweep_dir(
    base: Path,
    configs: list[tuple[str, float, float]],
    *,
    loss_end_fn=None,
) -> Path:
    sweep_dir = base / "sweep"
    for name, decay, boost in configs:
        loss_end = loss_end_fn(name) if loss_end_fn else 2.0
        path = sweep_dir / name / "run_metrics.jsonl"
        _write_jsonl(path, _make_run(name, decay, boost, loss_end=loss_end))
    return sweep_dir


class TestAnalyzeSweep:
    """Test analyze_sweep() with representative synthetic data."""

    @pytest.fixture()
    def sweep_dir(self, tmp_path):
        loss_ends = {
            "no_accel": 2.5,
            "conservative": 2.3,
            "balanced": 2.0,
            "aggressive": 2.2,
        }
        return _write_sweep_dir(
            tmp_path,
            CONFIGS,
            loss_end_fn=lambda n: loss_ends.get(n, 2.0),
        )

    def test_identifies_baseline(self, sweep_dir):
        result = analyze_sweep(sweep_dir)
        assert result["baseline"]["run_id"] == "no_accel"

    def test_identifies_best_run(self, sweep_dir):
        result = analyze_sweep(sweep_dir)
        assert result["best_run"]["run_id"] == "balanced"
        assert result["best_run"]["best_valid_loss"] < result["baseline"]["best_valid_loss"]

    def test_pairwise_count(self, sweep_dir):
        result = analyze_sweep(sweep_dir)
        assert len(result["pairwise"]) == 3

    def test_pairwise_deltas(self, sweep_dir):
        result = analyze_sweep(sweep_dir)
        balanced = next(p for p in result["pairwise"] if p["run_id"] == "balanced")
        assert balanced["delta_vs_baseline"] < 0
        aggressive = next(p for p in result["pairwise"] if p["run_id"] == "aggressive")
        assert aggressive["delta_vs_baseline"] < 0

    def test_efficiency_metrics_populated(self, sweep_dir):
        result = analyze_sweep(sweep_dir)
        for p in result["pairwise"]:
            assert p["efficiency_per_bp"] is not None
            assert p["loss_red_per_wall_min"] is not None

    def test_total_runs(self, sweep_dir):
        result = analyze_sweep(sweep_dir)
        assert result["total_runs"] == 4

    def test_pairwise_sorted_by_loss(self, sweep_dir):
        result = analyze_sweep(sweep_dir)
        losses = [p["best_valid_loss"] for p in result["pairwise"]]
        assert losses == sorted(losses)


class TestGenerateSummary:
    """Test generate_summary() markdown output."""

    @pytest.fixture()
    def analysis(self, tmp_path):
        sweep_dir = _write_sweep_dir(
            tmp_path,
            CONFIGS,
            loss_end_fn=lambda n: {"no_accel": 2.5, "conservative": 2.3, "balanced": 2.0, "aggressive": 2.2}.get(n, 2.0),
        )
        return analyze_sweep(sweep_dir)

    @pytest.mark.parametrize("label,expected", [
        ("best_run", "balanced"),
        ("pairwise_table", "Pairwise Comparison"),
        ("next_actions", "Next Actions"),
        ("ranking", "Ranking"),
    ])
    def test_summary_contains_section(self, analysis, label, expected):
        md = generate_summary(analysis)
        assert expected in md

    def test_summary_pairwise_has_delta(self, analysis):
        md = generate_summary(analysis)
        assert "delta" in md or "Δ" in md

    def test_summary_next_actions_has_improvement_verdict(self, analysis):
        md = generate_summary(analysis)
        assert "Improvement found" in md or "No improvement" in md


class TestAnalyzeSweepCLI:
    """Test the CLI entry point."""

    def test_cli_produces_output_files(self, tmp_path):
        sweep_dir = _write_sweep_dir(
            tmp_path,
            CONFIGS,
            loss_end_fn=lambda n: {"no_accel": 2.5, "conservative": 2.3, "balanced": 2.0, "aggressive": 2.2}.get(n, 2.0),
        )
        result = subprocess.run(
            [sys.executable, "scripts/analyze_accel_sweep.py", str(sweep_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        analysis_dir = sweep_dir / "analysis"
        assert (analysis_dir / "summary.md").exists()
        assert (analysis_dir / "ranking.json").exists()

        summary = (analysis_dir / "summary.md").read_text()
        assert "balanced" in summary

        ranking = json.loads((analysis_dir / "ranking.json").read_text())
        assert ranking["total_runs"] == 4
        assert ranking["best_run"]["run_id"] == "balanced"

    @pytest.mark.parametrize("args", [
        ["/nonexistent/path"],
        [],
    ], ids=["missing_dir", "no_args"])
    def test_cli_error_exit(self, args):
        result = subprocess.run(
            [sys.executable, "scripts/analyze_accel_sweep.py"] + args,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestInProgressRunHandling:
    """Test analyze_accel_sweep with runs that have no footer (still training)."""

    def test_in_progress_run_without_footer(self, tmp_path):
        """Runs without a footer should use step-level metrics for analysis."""
        sweep_dir = tmp_path / "sweep"

        # Completed baseline
        baseline_records = _make_run("no_accel", 0.99, 1.01, loss_start=3.0, loss_end=2.5, cycles=10)
        _write_jsonl(sweep_dir / "no_accel" / "run_metrics.jsonl", baseline_records)

        # In-progress treatment (header + steps only, no footer)
        header = {
            "type": "run_header",
            "run_id": "balanced",
            "mode": "tg_lora",
            "model_name": "Qwen/Qwen3.5-9B",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 2e-4,
            "seed": 42,
            "accel_instability_lr_decay": 0.5,
            "accel_convergence_lr_boost": 1.5,
        }
        in_progress = [header]
        for i in range(5):
            loss = 3.0 - 0.15 * (i + 1)
            in_progress.append({
                "type": "step",
                "step": (i + 1) * 24,
                "cycle": i + 1,
                "loss_train": loss + 0.1,
                "loss_valid": loss,
                "backward_passes": 24,
                "total_backward_passes": (i + 1) * 24,
                "elapsed_seconds": (i + 1) * 230.0,
                "tg_lora_accepted": True,
                "tg_lora_cosine_sim": 0.5,
                "tg_lora_reduction_rate": 0.1,
                "tg_lora_K": 3,
                "tg_lora_N": 5,
            })
        _write_jsonl(sweep_dir / "balanced" / "run_metrics.jsonl", in_progress)

        result = analyze_sweep(sweep_dir)
        assert result["total_runs"] == 2
        # balanced should have best_valid_loss derived from step records
        balanced_pw = next((p for p in result["pairwise"] if p["run_id"] == "balanced"), None)
        assert balanced_pw is not None
        assert balanced_pw["best_valid_loss"] is not None
        assert balanced_pw["best_valid_loss"] < 3.0

    def test_no_improvement_scenario(self, tmp_path):
        """When all treatments are worse than baseline, next actions should reflect this."""
        sweep_dir = _write_sweep_dir(
            tmp_path,
            CONFIGS,
            loss_end_fn=lambda n: {
                "no_accel": 2.0,
                "conservative": 2.5,
                "balanced": 2.3,
                "aggressive": 2.4,
            }.get(n, 2.0),
        )
        result = analyze_sweep(sweep_dir)
        md = generate_summary(result)
        assert "No improvement" in md or "NO IMPROVEMENT" in md

    def test_neutral_scenario(self, tmp_path):
        """When treatments are very close to baseline, analysis should report neutral."""
        sweep_dir = _write_sweep_dir(
            tmp_path,
            CONFIGS,
            loss_end_fn=lambda n: 2.0,
        )
        result = analyze_sweep(sweep_dir)
        assert result["total_runs"] == 4
        generate_summary(result)
        # All equal losses → deltas should be ~0
        for p in result["pairwise"]:
            assert abs(p["delta_vs_baseline"]) < 0.001


class TestAcceptRateAnalysis:
    """Test acceptance rate computation in analyze_sweep."""

    @pytest.mark.parametrize("baseline_acc,treatment_acc,field,expected", [
        (7, 8, ("baseline", "accept_rate"), pytest.approx(0.7)),
        (5, 9, ("pairwise", "accept_rate"), pytest.approx(0.9)),
        (5, 8, ("pairwise", "accept_rate_delta"), pytest.approx(0.3, abs=0.01)),
    ], ids=["baseline_rate", "treatment_rate", "delta_positive"])
    def test_accept_rate_field_from_two_runs(self, tmp_path, baseline_acc, treatment_acc, field, expected):
        sweep_dir = tmp_path / "sweep"
        _write_jsonl(
            sweep_dir / "no_accel" / "run_metrics.jsonl",
            _make_run("no_accel", 0.99, 1.01, cycles=10, accepted_count=baseline_acc),
        )
        _write_jsonl(
            sweep_dir / "balanced" / "run_metrics.jsonl",
            _make_run("balanced", 0.5, 1.5, cycles=10, accepted_count=treatment_acc),
        )
        result = analyze_sweep(sweep_dir)
        src, key = field
        if src == "baseline":
            actual = result["baseline"][key]
        else:
            actual = next(p for p in result["pairwise"] if p["run_id"] == "balanced")[key]
        assert actual == expected

    def test_accept_rate_delta_negative_when_worse(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        _write_jsonl(
            sweep_dir / "no_accel" / "run_metrics.jsonl",
            _make_run("no_accel", 0.99, 1.01, cycles=10, accepted_count=8),
        )
        _write_jsonl(
            sweep_dir / "aggressive" / "run_metrics.jsonl",
            _make_run("aggressive", 0.9, 2.0, cycles=10, accepted_count=4),
        )
        result = analyze_sweep(sweep_dir)
        aggressive = next(p for p in result["pairwise"] if p["run_id"] == "aggressive")
        assert aggressive["accept_rate_delta"] < 0

    def test_best_run_accept_rate(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        _write_jsonl(
            sweep_dir / "no_accel" / "run_metrics.jsonl",
            _make_run("no_accel", 0.99, 1.01, cycles=10, accepted_count=6,
                       loss_end=2.5),
        )
        _write_jsonl(
            sweep_dir / "balanced" / "run_metrics.jsonl",
            _make_run("balanced", 0.5, 1.5, cycles=10, accepted_count=9,
                       loss_end=2.0),
        )
        result = analyze_sweep(sweep_dir)
        assert result["best_run"]["accept_rate"] == pytest.approx(0.9)

    def test_accept_rate_in_summary_markdown(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        _write_jsonl(
            sweep_dir / "no_accel" / "run_metrics.jsonl",
            _make_run("no_accel", 0.99, 1.01, cycles=10, accepted_count=6,
                       loss_end=2.5),
        )
        _write_jsonl(
            sweep_dir / "balanced" / "run_metrics.jsonl",
            _make_run("balanced", 0.5, 1.5, cycles=10, accepted_count=9,
                       loss_end=2.0),
        )
        result = analyze_sweep(sweep_dir)
        md = generate_summary(result)
        assert "Accept%" in md
        assert "Accept rate" in md
        assert "60.0%" in md  # baseline 6/10
        assert "90.0%" in md  # balanced 9/10

    def test_accept_rate_none_when_no_jsonl(self):
        """Runs without _jsonl_path should have None accept_rate."""
        analysis = {
            "best_run": {"run_id": "x", "best_valid_loss": 1.0, "loss_red_per_bp": None, "loss_red_per_wall_min": None, "accept_rate": None},
            "baseline": {"run_id": "x", "best_valid_loss": 1.0, "total_backward_passes": 0, "wall_seconds": 0, "loss_reduction": None, "loss_red_per_bp": None, "loss_red_per_wall_min": None, "accept_rate": None},
            "pairwise": [],
            "total_runs": 0,
        }
        md = generate_summary(analysis)
        assert "N/A" in md

    def test_full_sweep_accept_rates_in_pairwise(self, tmp_path):
        """Full 4-config sweep with varying acceptance rates."""
        sweep_dir = tmp_path / "sweep"
        accept_rates = {"no_accel": 6, "conservative": 7, "balanced": 9, "aggressive": 5}
        for name, decay, boost in CONFIGS:
            _write_jsonl(
                sweep_dir / name / "run_metrics.jsonl",
                _make_run(
                    name, decay, boost,
                    cycles=10,
                    accepted_count=accept_rates[name],
                    loss_end={"no_accel": 2.5, "conservative": 2.3, "balanced": 2.0, "aggressive": 2.2}[name],
                ),
            )
        result = analyze_sweep(sweep_dir)
        for p in result["pairwise"]:
            assert p["accept_rate"] is not None
            assert 0 < p["accept_rate"] <= 1.0
            assert p["accept_rate_delta"] is not None


# ---------------------------------------------------------------------------
# Convergence trajectory analysis
# ---------------------------------------------------------------------------


class TestComputeLossTrajectory:
    """Test compute_loss_trajectory with synthetic step records."""

    def _make_steps(self, losses: list[float]) -> list[dict]:
        records = [{"type": "run_header", "run_id": "test"}]
        for i, loss in enumerate(losses):
            records.append({
                "type": "step",
                "cycle": i + 1,
                "loss_valid": loss,
                "loss_train": loss + 0.1,
            })
        return records

    def test_monotonically_decreasing(self):
        steps = self._make_steps([3.0, 2.8, 2.6, 2.4, 2.2, 2.0])
        result = compute_loss_trajectory(steps)
        assert result["slope"] < 0
        assert result["total_reduction"] == pytest.approx(1.0)
        assert result["convergence_speed"] is not None
        assert result["convergence_speed"] > 0

    def test_flat_losses_no_reduction(self):
        steps = self._make_steps([2.0, 2.0, 2.0, 2.0, 2.0])
        result = compute_loss_trajectory(steps)
        assert result["slope"] == pytest.approx(0.0)
        assert result["total_reduction"] == pytest.approx(0.0)
        assert result["convergence_speed"] is None
        assert result["half_reduction_cycle"] is None

    def test_plateau_detected(self):
        # Loss drops then flattens — need enough flat steps for window=5
        steps = self._make_steps([3.0, 2.5, 2.1, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05])
        result = compute_loss_trajectory(steps)
        assert result["plateau_start"] is not None

    def test_no_plateau_when_still_improving(self):
        steps = self._make_steps([3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
        result = compute_loss_trajectory(steps)
        assert result["plateau_start"] is None

    def test_half_reduction_cycle(self):
        steps = self._make_steps([4.0, 3.5, 3.0, 2.5, 2.0])
        result = compute_loss_trajectory(steps)
        # 4.0 - 2.0 = 2.0 total, target = 4.0 - 1.0 = 3.0, reached at cycle 3
        assert result["half_reduction_cycle"] == 3

    def test_single_step_returns_none(self):
        steps = self._make_steps([3.0])
        result = compute_loss_trajectory(steps)
        assert result["slope"] is None
        assert result["total_reduction"] is None

    def test_nan_losses_skipped(self):
        steps = self._make_steps([3.0, 2.5, float("nan"), 2.0, 1.5])
        result = compute_loss_trajectory(steps)
        assert result["slope"] is not None
        assert result["total_reduction"] == pytest.approx(1.5)

    def test_convergence_speed_fast_start(self):
        # 80% of reduction in first quarter
        steps = self._make_steps([4.0, 2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.01])
        result = compute_loss_trajectory(steps)
        assert result["convergence_speed"] is not None
        assert result["convergence_speed"] > 0.5

    def test_empty_records(self):
        result = compute_loss_trajectory([])
        assert result["slope"] is None


class TestConvergenceInSweep:
    """Test that convergence metrics appear in sweep analysis."""

    def test_sweep_includes_convergence(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        _write_jsonl(
            sweep_dir / "no_accel" / "run_metrics.jsonl",
            _make_run("no_accel", 0.99, 1.01, loss_start=3.0, loss_end=2.5, cycles=10),
        )
        _write_jsonl(
            sweep_dir / "balanced" / "run_metrics.jsonl",
            _make_run("balanced", 0.5, 1.5, loss_start=3.0, loss_end=2.0, cycles=10),
        )
        result = analyze_sweep(sweep_dir)
        assert "convergence" in result["baseline"]
        assert result["baseline"]["convergence"]["slope"] < 0
        for p in result["pairwise"]:
            assert "convergence" in p
            assert p["convergence"]["slope"] < 0

    def test_summary_includes_convergence_section(self, tmp_path):
        sweep_dir = _write_sweep_dir(
            tmp_path,
            CONFIGS,
            loss_end_fn=lambda n: {"no_accel": 2.5, "conservative": 2.3, "balanced": 2.0, "aggressive": 2.2}.get(n, 2.0),
        )
        result = analyze_sweep(sweep_dir)
        md = generate_summary(result)
        assert "Convergence Trajectory" in md
        assert "Slope" in md
        assert "Plateau" in md


# ---------------------------------------------------------------------------
# Sweep result validation
# ---------------------------------------------------------------------------


class TestValidateSweepResults:
    """Test sweep result integrity validation."""

    def test_valid_sweep(self, tmp_path):
        sweep_dir = _write_sweep_dir(tmp_path, CONFIGS)
        result = validate_sweep_results(sweep_dir)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["runs_checked"] == 4

    def test_missing_directory(self):
        result = validate_sweep_results(Path("/nonexistent/path"))
        assert result["valid"] is False
        assert any("not found" in e for e in result["errors"])

    def test_empty_directory(self, tmp_path):
        result = validate_sweep_results(tmp_path)
        assert result["valid"] is False
        assert any("No run_metrics.jsonl" in e for e in result["errors"])

    @pytest.mark.parametrize(
        "loss_field, bad_value, target_cycle",
        [("loss_valid", float("nan"), 3), ("loss_train", float("inf"), 2)],
        ids=["nan_loss_valid", "inf_loss_train"],
    )
    def test_nonfinite_loss_detected(self, tmp_path, loss_field, bad_value, target_cycle):
        sweep_dir = tmp_path / "sweep"
        records = _make_run("test_run", 0.5, 1.5, cycles=5)
        for rec in records:
            if rec.get("type") == "step" and rec.get("cycle") == target_cycle:
                rec[loss_field] = bad_value
        _write_jsonl(sweep_dir / "test_run" / "run_metrics.jsonl", records)
        result = validate_sweep_results(sweep_dir)
        assert result["valid"] is False
        assert any("Non-finite" in e for e in result["errors"])

    def test_missing_footer_warning(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        records = _make_run("test_run", 0.5, 1.5, cycles=5)
        # Remove footer
        records = [r for r in records if r.get("type") != "run_footer"]
        _write_jsonl(sweep_dir / "test_run" / "run_metrics.jsonl", records)
        result = validate_sweep_results(sweep_dir)
        assert result["valid"] is True  # warnings don't invalidate
        assert any("Missing run_footer" in w for w in result["warnings"])

    def test_loss_explosion_warning(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        records = _make_run("test_run", 0.5, 1.5, loss_start=2.0, loss_end=1.5, cycles=10)
        # Inject loss explosion at cycle 5
        for rec in records:
            if rec.get("type") == "step" and rec.get("cycle") == 5:
                rec["loss_train"] = 100.0
        _write_jsonl(sweep_dir / "test_run" / "run_metrics.jsonl", records)
        result = validate_sweep_results(sweep_dir)
        assert any("explosion" in w.lower() for w in result["warnings"])

    def test_corrupted_jsonl(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        path = sweep_dir / "bad_run" / "run_metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json\n")
        result = validate_sweep_results(sweep_dir)
        assert result["valid"] is False
        assert any("bad_run" in e for e in result["errors"])

    def test_run_details_populated(self, tmp_path):
        sweep_dir = _write_sweep_dir(tmp_path, CONFIGS)
        result = validate_sweep_results(sweep_dir)
        assert len(result["run_details"]) == 4
        for name, details in result["run_details"].items():
            assert "errors" in details
            assert "warnings" in details
            assert details["errors"] == []


# ---------------------------------------------------------------------------
# Sweep config validation
# ---------------------------------------------------------------------------


class TestValidateSweepConfigs:
    """Test pre-flight sweep config validation."""

    def test_valid_accel_configs(self):
        paths = [
            Path(f"configs/9b_tg_lora_accel_{name}.yaml")
            for name in ("no_accel", "conservative", "balanced", "aggressive")
        ]
        result = validate_sweep_configs(paths)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_config(self):
        paths = [
            Path("configs/9b_tg_lora_accel_no_accel.yaml"),
            Path("configs/nonexistent.yaml"),
        ]
        result = validate_sweep_configs(paths)
        assert result["valid"] is False
        assert any("not found" in e for e in result["errors"])

    def test_too_few_configs(self):
        result = validate_sweep_configs([Path("configs/9b_tg_lora_accel_no_accel.yaml")])
        assert result["valid"] is False
        assert any("at least 2" in e for e in result["errors"])

    def test_controlled_variable_mismatch(self, tmp_path):
        # Create a modified copy of a valid config with different K_initial

        ref_yaml = Path("configs/9b_tg_lora_accel_no_accel.yaml").read_text()
        # Replace K_initial to create a mismatch
        bad_yaml = ref_yaml.replace("K_initial: 3", "K_initial: 99")
        bad_cfg_path = tmp_path / "bad_config.yaml"
        bad_cfg_path.write_text(bad_yaml)

        result = validate_sweep_configs([
            Path("configs/9b_tg_lora_accel_no_accel.yaml"),
            bad_cfg_path,
        ])
        assert result["valid"] is False
        assert any("K_initial" in e for e in result["errors"])

    def test_identical_decay_boost_warning(self, tmp_path):
        # Copy the no_accel config to create a duplicate
        ref_yaml = Path("configs/9b_tg_lora_accel_no_accel.yaml").read_text()
        dupe_path = tmp_path / "dupe.yaml"
        dupe_path.write_text(ref_yaml)

        result = validate_sweep_configs([
            Path("configs/9b_tg_lora_accel_no_accel.yaml"),
            dupe_path,
        ])
        assert any("identical decay/boost" in w.lower() or "Duplicate" in w for w in result["warnings"])


class TestNonFiniteBaseline:
    """Verify pairwise delta_pct is 0 when baseline loss is non-finite."""

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_nonfinite_baseline_loss_produces_zero_delta_pct(self, tmp_path, bad_value):
        loss_ends = {
            "no_accel": bad_value,
            "balanced": 2.0,
        }
        sweep_dir = _write_sweep_dir(
            tmp_path,
            [("no_accel", 0.99, 1.01), ("balanced", 0.5, 1.5)],
            loss_end_fn=lambda n: loss_ends.get(n, 2.0),
        )
        result = analyze_sweep(sweep_dir)
        for p in result["pairwise"]:
            assert p["delta_pct"] == 0
