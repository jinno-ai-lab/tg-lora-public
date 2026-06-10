"""End-to-end integration test for the accel param sweep pipeline (TASK-0094).

Validates the complete sweep workflow using synthetic run metrics:
  1. All 4 accel configs parse and have correct isolation (only decay/boost differ)
  2. run_accel_sweep.sh passes bash syntax check
  3. Synthetic 4-config sweep data flows through pairwise comparison
  4. Dashboard generation from sweep directory
  5. Summarize sweep analysis
  6. Best config identification (loss reduction / backward pass)
  7. Efficiency metric computation (loss_red/bp, loss_red/wall-min)
"""

import subprocess
import sys
from pathlib import Path

import orjson
import pytest

from scripts.compare_runs import (
    build_comparison_table,
    find_best_run,
    gather_runs,
    generate_report,
    load_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    accepted_count: int = 7,
    wall_per_cycle: float = 230.0,
    bp_per_cycle: int = 24,
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
            "tg_lora_alpha": 0.3,
            "tg_lora_beta": 0.8,
            "tg_lora_lr": 5e-4,
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


# ---------------------------------------------------------------------------
# 1. Config parsing + isolation
# ---------------------------------------------------------------------------


class TestSweepConfigIntegration:
    """Validate all 4 accel configs parse and only vary on decay/boost."""

    def test_all_four_configs_exist(self):
        from src.training.config_schema import load_and_validate_config

        expected = {
            "no_accel": (0.99, 1.01),
            "conservative": (0.3, 1.1),
            "balanced": (0.5, 1.5),
            "aggressive": (0.9, 2.0),
        }
        for name, (exp_decay, exp_boost) in expected.items():
            path = Path(f"configs/9b_tg_lora_accel_{name}.yaml")
            assert path.exists(), f"Missing config: {path}"
            cfg = load_and_validate_config(path)
            assert cfg.tg_lora.accel_instability_lr_decay == pytest.approx(exp_decay)
            assert cfg.tg_lora.accel_convergence_lr_boost == pytest.approx(exp_boost)

    def test_controlled_variables_identical(self):
        from src.training.config_schema import load_and_validate_config

        configs = {}
        for name in ("no_accel", "conservative", "balanced", "aggressive"):
            path = Path(f"configs/9b_tg_lora_accel_{name}.yaml")
            configs[name] = load_and_validate_config(path)

        base = configs["no_accel"]
        for name, cfg in configs.items():
            if name == "no_accel":
                continue
            assert cfg.model.name_or_path == base.model.name_or_path
            assert cfg.lora.r == base.lora.r
            assert cfg.lora.alpha == base.lora.alpha
            assert cfg.training.max_cycles == base.training.max_cycles
            assert cfg.training.learning_rate == base.training.learning_rate
            assert cfg.tg_lora.K_initial == base.tg_lora.K_initial
            assert cfg.tg_lora.N_initial == base.tg_lora.N_initial
            assert cfg.tg_lora.alpha_initial == base.tg_lora.alpha_initial
            assert cfg.tg_lora.beta_initial == base.tg_lora.beta_initial
            assert cfg.tg_lora.enable_random_walk == base.tg_lora.enable_random_walk is False
            assert cfg.tg_lora.lr_explore_prob == 0.0


# ---------------------------------------------------------------------------
# 2. Sweep script validation
# ---------------------------------------------------------------------------


class TestSweepScript:
    """Validate run_accel_sweep.sh passes syntax check and has expected structure."""

    def test_bash_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", "scripts/run_accel_sweep.sh"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_script_references_all_four_configs(self):
        content = Path("scripts/run_accel_sweep.sh").read_text()
        for name in ("no_accel", "conservative", "balanced", "aggressive"):
            assert f"9b_tg_lora_accel_{name}.yaml" in content, (
                f"Missing config reference: {name}"
            )

    def test_script_invokes_compare_runs(self):
        content = Path("scripts/run_accel_sweep.sh").read_text()
        assert "compare_runs.py" in content
        assert "--baseline" in content or "baseline" in content


# ---------------------------------------------------------------------------
# 3. End-to-end sweep pipeline with synthetic data
# ---------------------------------------------------------------------------


class TestSweepPipelineEndToEnd:
    """Full pipeline: synthetic data → pairwise → dashboard → summarize."""

    @pytest.fixture()
    def sweep_dir(self, tmp_path):
        """Create a complete sweep directory with differentiated results."""
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

    def test_pairwise_reports_generate_for_all_treatments(self, sweep_dir):
        baseline_path = sweep_dir / "no_accel" / "run_metrics.jsonl"
        assert baseline_path.exists()
        b_h, b_r, b_f = load_run(baseline_path)

        for name in ("conservative", "balanced", "aggressive"):
            t_path = sweep_dir / name / "run_metrics.jsonl"
            t_h, t_r, t_f = load_run(t_path)
            report = generate_report(b_h, b_r, b_f, t_h, t_r, t_f)
            assert "Efficiency Comparison" in report
            assert "Loss Red. / 100 backward" in report
            assert "Loss Red. / wall-minute" in report

    def test_dashboard_gather_finds_all_runs(self, sweep_dir):
        runs = gather_runs(sweep_dir)
        assert len(runs) == 4
        ids = {r["run_id"] for r in runs}
        assert ids == {"no_accel", "conservative", "balanced", "aggressive"}

    def test_find_best_run_identifies_balanced(self, sweep_dir):
        runs = gather_runs(sweep_dir)
        best = find_best_run(runs)
        assert best is not None
        assert best["run_id"] == "balanced"

    def test_comparison_table_ranks_correctly(self, sweep_dir):
        runs = gather_runs(sweep_dir)
        rows = build_comparison_table(runs)
        assert len(rows) == 4
        best_rows = [r for r in rows if r["is_best"]]
        assert len(best_rows) == 1
        assert best_rows[0]["run_id"] == "balanced"
        sorted_rows = sorted(rows, key=lambda r: r["best_valid_loss"] or float("inf"))
        assert sorted_rows[0]["run_id"] == "balanced"

    def test_summarize_sweep_cli(self, sweep_dir, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.summarize_sweep", "--sweep-dir", str(sweep_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"summarize_sweep failed: {result.stderr}"
        assert "balanced" in result.stdout

        summary_path = sweep_dir / "summary.txt"
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "balanced" in content
        assert "conservative" in content
        assert "no_accel" in content
        assert "aggressive" in content

    def test_compare_runs_dashboard_json(self, sweep_dir):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/compare_runs.py",
                "dashboard",
                str(sweep_dir),
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"dashboard failed: {result.stderr}"
        data = orjson.loads(result.stdout)
        assert isinstance(data, dict)
        assert "runs" in data
        assert len(data["runs"]) == 4
        best = [d for d in data["runs"] if d.get("is_best")]
        assert len(best) == 1
        assert best[0]["run_id"] == "balanced"


# ---------------------------------------------------------------------------
# 4. Efficiency metric verification
# ---------------------------------------------------------------------------


class TestEfficiencyMetrics:
    """Verify loss_red/bp and loss_red/wall-minute computation."""

    def test_loss_reduction_per_backward_pass(self, tmp_path):
        baseline_path = tmp_path / "no_accel" / "run_metrics.jsonl"
        _write_jsonl(
            baseline_path,
            _make_run("no_accel", 0.99, 1.01, loss_start=3.0, loss_end=2.5, cycles=10),
        )
        treatment_path = tmp_path / "balanced" / "run_metrics.jsonl"
        _write_jsonl(
            treatment_path,
            _make_run("balanced", 0.5, 1.5, loss_start=3.0, loss_end=2.0, cycles=10),
        )

        b_h, b_r, b_f = load_run(baseline_path)
        t_h, t_r, t_f = load_run(treatment_path)

        b_init = b_r[0]["loss_train"]
        t_init = t_r[0]["loss_train"]
        b_loss_red = b_init - b_f["best_valid_loss"]
        t_loss_red = t_init - t_f["best_valid_loss"]
        b_bp = b_r[-1]["total_backward_passes"]
        t_bp = t_r[-1]["total_backward_passes"]

        b_per_bp = b_loss_red / b_bp
        t_per_bp = t_loss_red / t_bp
        assert t_per_bp > b_per_bp, "Treatment should be more efficient per backward pass"

    def test_loss_reduction_per_wall_minute(self, tmp_path):
        baseline_path = tmp_path / "no_accel" / "run_metrics.jsonl"
        _write_jsonl(
            baseline_path,
            _make_run("no_accel", 0.99, 1.01, loss_start=3.0, loss_end=2.5, cycles=10),
        )
        treatment_path = tmp_path / "balanced" / "run_metrics.jsonl"
        _write_jsonl(
            treatment_path,
            _make_run("balanced", 0.5, 1.5, loss_start=3.0, loss_end=2.0, cycles=10),
        )

        b_h, b_r, b_f = load_run(baseline_path)
        t_h, t_r, t_f = load_run(treatment_path)

        b_init = b_r[0]["loss_train"]
        t_init = t_r[0]["loss_train"]
        b_loss_red = b_init - b_f["best_valid_loss"]
        t_loss_red = t_init - t_f["best_valid_loss"]
        b_wall_min = b_f["total_wall_seconds"] / 60
        t_wall_min = t_f["total_wall_seconds"] / 60

        b_per_min = b_loss_red / b_wall_min
        t_per_min = t_loss_red / t_wall_min
        assert t_per_min > b_per_min, "Treatment should be more efficient per wall-minute"

    def test_report_contains_efficiency_metrics(self, tmp_path):
        baseline_path = tmp_path / "no_accel" / "run_metrics.jsonl"
        _write_jsonl(
            baseline_path,
            _make_run("no_accel", 0.99, 1.01, loss_start=3.0, loss_end=2.5, cycles=10),
        )
        treatment_path = tmp_path / "balanced" / "run_metrics.jsonl"
        _write_jsonl(
            treatment_path,
            _make_run("balanced", 0.5, 1.5, loss_start=3.0, loss_end=2.0, cycles=10),
        )

        b_h, b_r, b_f = load_run(baseline_path)
        t_h, t_r, t_f = load_run(treatment_path)
        report = generate_report(b_h, b_r, b_f, t_h, t_r, t_f)

        assert "Loss Red. / 100 backward" in report
        assert "Loss Red. / wall-minute" in report
        assert "GB-hour" in report


# ---------------------------------------------------------------------------
# 5. Sweep output structure validation
# ---------------------------------------------------------------------------


class TestSweepOutputStructure:
    """Validate the directory structure that run_accel_sweep.sh expects."""

    def test_sweep_output_has_per_config_dirs(self, tmp_path):
        sweep_dir = _write_sweep_dir(tmp_path, CONFIGS)
        for name, _, _ in CONFIGS:
            assert (sweep_dir / name).is_dir(), f"Missing dir for {name}"
            assert (sweep_dir / name / "run_metrics.jsonl").exists(), f"Missing metrics for {name}"

    def test_no_accel_baseline_detectable(self, tmp_path):
        sweep_dir = _write_sweep_dir(tmp_path, CONFIGS)
        baseline_metrics = None
        treatment_dirs = []
        for d in sorted(sweep_dir.iterdir()):
            if not d.is_dir():
                continue
            metrics = d / "run_metrics.jsonl"
            if not metrics.exists():
                continue
            if "no_accel" in d.name:
                baseline_metrics = metrics
            else:
                treatment_dirs.append(d)
        assert baseline_metrics is not None
        assert len(treatment_dirs) == 3

    def test_each_run_has_header_records_footer(self, tmp_path):
        sweep_dir = _write_sweep_dir(tmp_path, CONFIGS)
        for name, _, _ in CONFIGS:
            path = sweep_dir / name / "run_metrics.jsonl"
            header, records, footer = load_run(path)
            assert header["type"] == "run_header"
            assert header["run_id"] == name
            assert len(records) > 0
            assert footer is not None
            assert "best_valid_loss" in footer


# ---------------------------------------------------------------------------
# 6. Efficiency gate assertions
# ---------------------------------------------------------------------------


class TestEfficiencyGates:
    """Regression gate: loss_red_per_wall_min > 0 for all non-trivial configs."""

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

    def test_all_configs_have_positive_loss_red_per_wall_min(self, sweep_dir):
        runs = gather_runs(sweep_dir)
        rows = build_comparison_table(runs)
        for row in rows:
            assert row["loss_red_per_wall_min"] is not None, (
                f"{row['run_id']}: loss_red_per_wall_min should not be None"
            )
            assert row["loss_red_per_wall_min"] > 0, (
                f"{row['run_id']}: loss_red_per_wall_min={row['loss_red_per_wall_min']} must be > 0"
            )

    def test_all_configs_have_positive_loss_red_per_bp(self, sweep_dir):
        runs = gather_runs(sweep_dir)
        rows = build_comparison_table(runs)
        for row in rows:
            assert row["loss_red_per_bp"] is not None, (
                f"{row['run_id']}: loss_red_per_bp should not be None"
            )
            assert row["loss_red_per_bp"] > 0, (
                f"{row['run_id']}: loss_red_per_bp={row['loss_red_per_bp']} must be > 0"
            )

    def test_summarize_sweep_includes_efficiency_columns(self, sweep_dir):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.summarize_sweep", "--sweep-dir", str(sweep_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"summarize_sweep failed: {result.stderr}"
        assert "Red/bp" in result.stdout
        assert "Red/min" in result.stdout

    def test_summarize_sweep_includes_next_actions(self, sweep_dir):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.summarize_sweep", "--sweep-dir", str(sweep_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"summarize_sweep failed: {result.stderr}"
        assert "Next actions" in result.stdout
        assert "IMPROVEMENT" in result.stdout or "NO IMPROVEMENT" in result.stdout or "NEUTRAL" in result.stdout

    def test_summarize_sweep_pairwise_deltas(self, sweep_dir):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.summarize_sweep", "--sweep-dir", str(sweep_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"summarize_sweep failed: {result.stderr}"
        assert "Pairwise" in result.stdout
        assert "delta=" in result.stdout
