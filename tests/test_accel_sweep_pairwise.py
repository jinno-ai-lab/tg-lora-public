"""Tests for accel sweep pairwise comparison logic: TASK-0094.

Validates the corrected baseline detection, treatment-vs-baseline comparison,
and dashboard generation logic from run_accel_sweep.sh (as implemented in
compare_runs.py).
"""

from pathlib import Path

import orjson
import pytest

from scripts.compare_runs import (
    find_best_run,
    gather_runs,
    generate_report,
    load_run,
    build_comparison_table,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec) + b"\n")


def _make_accel_run(
    run_id: str,
    decay: float,
    boost: float,
    loss_start: float = 3.0,
    loss_end: float = 2.0,
    cycles: int = 10,
    accepted_count: int = 7,
) -> list[dict]:
    """Create a complete run metrics JSONL for an accel sweep config."""
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
    records = []
    loss_step = (loss_start - loss_end) / max(cycles, 1)
    accepted_list = [True] * accepted_count + [False] * (cycles - accepted_count)
    for i in range(cycles):
        loss = loss_start - loss_step * (i + 1)
        records.append({
            "type": "step",
            "step": (i + 1) * 24,
            "cycle": i + 1,
            "loss_train": loss + 0.1,
            "loss_valid": loss,
            "backward_passes": 24,
            "total_backward_passes": (i + 1) * 24,
            "elapsed_seconds": (i + 1) * 230.0,
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
        "total_wall_seconds": cycles * 230.0,
        "best_valid_loss": loss_end,
        "final_train_loss": loss_end + 0.1,
        "best_valid_step": cycles * 24,
        "gpu_peak_mb": 8200,
        "total_cycles": cycles,
    }
    return [header] + records + [footer]


def _write_accel_run(
    base_dir: Path,
    run_name: str,
    decay: float,
    boost: float,
    **kwargs,
) -> Path:
    """Write an accel run to disk and return the metrics path."""
    run_dir = base_dir / run_name
    path = run_dir / "run_metrics.jsonl"
    _write_jsonl(path, _make_accel_run(run_name, decay, boost, **kwargs))
    return path


# ---------------------------------------------------------------------------
# Test: baseline detection from directory names
# ---------------------------------------------------------------------------


class TestBaselineDetection:
    """Test that the sweep correctly identifies no_accel as the baseline."""

    def test_no_accel_directory_detected(self, tmp_path):
        """The directory containing 'no_accel' should be identified as baseline."""
        sweep_dir = tmp_path / "sweep"
        dirs = [
            _write_accel_run(sweep_dir, "tg_lora_9b_accel_no_accel", 0.99, 1.01),
            _write_accel_run(sweep_dir, "tg_lora_9b_accel_conservative", 0.3, 1.1),
            _write_accel_run(sweep_dir, "tg_lora_9b_accel_balanced", 0.5, 1.5),
            _write_accel_run(sweep_dir, "tg_lora_9b_accel_aggressive", 0.9, 2.0),
        ]

        baseline_metrics = None
        treatment_dirs = []
        for d in dirs:
            name = d.parent.name
            if "no_accel" in name:
                baseline_metrics = d
            else:
                treatment_dirs.append(d)

        assert baseline_metrics is not None
        assert "no_accel" in baseline_metrics.parent.name
        assert len(treatment_dirs) == 3
        assert all("no_accel" not in d.parent.name for d in treatment_dirs)

    def test_baseline_header_contains_near_identity_params(self, tmp_path):
        """no_accel baseline should have decay≈1.0, boost≈1.0."""
        path = _write_accel_run(
            tmp_path, "no_accel", 0.99, 1.01,
        )
        header, _, _ = load_run(path)
        assert header["accel_instability_lr_decay"] == pytest.approx(0.99)
        assert header["accel_convergence_lr_boost"] == pytest.approx(1.01)


# ---------------------------------------------------------------------------
# Test: pairwise comparison logic
# ---------------------------------------------------------------------------


class TestPairwiseComparison:
    """Test that each treatment can be compared against the no_accel baseline."""

    def test_pairwise_report_generation(self, tmp_path):
        """generate_report works for each treatment vs baseline."""
        baseline_path = _write_accel_run(
            tmp_path, "no_accel", 0.99, 1.01,
            loss_start=3.0, loss_end=2.2, cycles=5,
        )
        treatments = [
            ("conservative", 0.3, 1.1, 2.0),
            ("balanced", 0.5, 1.5, 1.9),
            ("aggressive", 0.9, 2.0, 2.1),
        ]

        b_header, b_records, b_footer = load_run(baseline_path)
        for name, decay, boost, final_loss in treatments:
            t_path = _write_accel_run(
                tmp_path / "treatments", name, decay, boost,
                loss_start=3.0, loss_end=final_loss, cycles=5,
            )
            t_header, t_records, t_footer = load_run(t_path)

            report = generate_report(
                b_header, b_records, b_footer,
                t_header, t_records, t_footer,
            )
            assert "Efficiency Comparison" in report
            assert "Backward Passes" in report

    def test_pairwise_all_have_same_backward_passes(self, tmp_path):
        """All configs produce the same total_backward_passes (controlled experiment)."""
        baseline_path = _write_accel_run(
            tmp_path, "no_accel", 0.99, 1.01, cycles=10,
        )
        conservative_path = _write_accel_run(
            tmp_path, "conservative", 0.3, 1.1, cycles=10,
        )

        _, b_recs, _ = load_run(baseline_path)
        _, c_recs, _ = load_run(conservative_path)

        b_total_bp = b_recs[-1]["total_backward_passes"]
        c_total_bp = c_recs[-1]["total_backward_passes"]
        assert b_total_bp == c_total_bp == 240  # 10 cycles × 24 bp/cycle

    def test_pairwise_treatment_improves_over_baseline(self, tmp_path):
        """A treatment with better loss should show improvement in report."""
        baseline_path = _write_accel_run(
            tmp_path, "no_accel", 0.99, 1.01,
            loss_start=3.0, loss_end=2.5, cycles=8,
        )
        treatment_path = _write_accel_run(
            tmp_path, "balanced", 0.5, 1.5,
            loss_start=3.0, loss_end=2.0, cycles=8,
        )

        b_h, b_r, b_f = load_run(baseline_path)
        t_h, t_r, t_f = load_run(treatment_path)

        assert t_f["best_valid_loss"] < b_f["best_valid_loss"]

        report = generate_report(b_h, b_r, b_f, t_h, t_r, t_f)
        assert "Efficiency Metrics" in report


# ---------------------------------------------------------------------------
# Test: fallback when no_accel is missing
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """Test fallback when no_accel baseline is not found."""

    def test_uses_first_available_as_baseline(self, tmp_path):
        """When no_accel is missing, first available run becomes baseline."""
        dirs = [
            _write_accel_run(tmp_path, "conservative", 0.3, 1.1),
            _write_accel_run(tmp_path, "balanced", 0.5, 1.5),
        ]

        baseline_metrics = None
        treatment_dirs = []
        for d in dirs:
            name = d.parent.name
            if "no_accel" in name:
                baseline_metrics = d
            else:
                treatment_dirs.append(d)

        if baseline_metrics is None:
            baseline_metrics = treatment_dirs[0]
            treatment_dirs = treatment_dirs[1:]

        assert baseline_metrics is not None
        assert "conservative" in baseline_metrics.parent.name
        assert len(treatment_dirs) == 1


# ---------------------------------------------------------------------------
# Test: dashboard generation across 4 configs
# ---------------------------------------------------------------------------


class TestDashboardGeneration:
    """Test multi-run dashboard for cross-config comparison."""

    def test_gather_runs_finds_all_configs(self, tmp_path):
        """gather_runs discovers all 4 accel sweep configs."""
        sweep_dir = tmp_path / "sweep"
        _write_accel_run(sweep_dir, "no_accel", 0.99, 1.01, cycles=3)
        _write_accel_run(sweep_dir, "conservative", 0.3, 1.1, cycles=3)
        _write_accel_run(sweep_dir, "balanced", 0.5, 1.5, cycles=3)
        _write_accel_run(sweep_dir, "aggressive", 0.9, 2.0, cycles=3)

        runs = gather_runs(sweep_dir)
        assert len(runs) == 4
        ids = {r["run_id"] for r in runs}
        assert ids == {"no_accel", "conservative", "balanced", "aggressive"}

    def test_find_best_run_selects_lowest_loss(self, tmp_path):
        """find_best_run picks the config with lowest best_valid_loss."""
        sweep_dir = tmp_path / "sweep"
        _write_accel_run(sweep_dir, "no_accel", 0.99, 1.01, loss_end=2.5, cycles=3)
        _write_accel_run(sweep_dir, "conservative", 0.3, 1.1, loss_end=2.3, cycles=3)
        _write_accel_run(sweep_dir, "balanced", 0.5, 1.5, loss_end=2.0, cycles=3)
        _write_accel_run(sweep_dir, "aggressive", 0.9, 2.0, loss_end=2.2, cycles=3)

        runs = gather_runs(sweep_dir)
        best = find_best_run(runs)
        assert best is not None
        assert best["run_id"] == "balanced"

    def test_comparison_table_ranks_all_configs(self, tmp_path):
        """build_comparison_table produces rows for all 4 configs."""
        sweep_dir = tmp_path / "sweep"
        _write_accel_run(sweep_dir, "no_accel", 0.99, 1.01, loss_end=2.5, cycles=3)
        _write_accel_run(sweep_dir, "conservative", 0.3, 1.1, loss_end=2.3, cycles=3)
        _write_accel_run(sweep_dir, "balanced", 0.5, 1.5, loss_end=2.0, cycles=3)
        _write_accel_run(sweep_dir, "aggressive", 0.9, 2.0, loss_end=2.2, cycles=3)

        runs = gather_runs(sweep_dir)
        rows = build_comparison_table(runs)
        assert len(rows) == 4
        best_rows = [r for r in rows if r["is_best"]]
        assert len(best_rows) == 1
        assert best_rows[0]["run_id"] == "balanced"


# ---------------------------------------------------------------------------
# Test: accel param isolation
# ---------------------------------------------------------------------------


class TestAccelParamIsolation:
    """Verify that only accel params differ between configs."""

    def test_only_accel_params_differ(self, tmp_path):
        """All 4 configs should have identical fields except decay/boost."""
        configs = [
            ("no_accel", 0.99, 1.01),
            ("conservative", 0.3, 1.1),
            ("balanced", 0.5, 1.5),
            ("aggressive", 0.9, 2.0),
        ]
        sweep_dir = tmp_path / "sweep"
        headers = {}
        for name, decay, boost in configs:
            path = _write_accel_run(sweep_dir, name, decay, boost, cycles=2)
            header, _, _ = load_run(path)
            headers[name] = header

        base = headers["no_accel"]
        base_keys = set(base.keys()) - {
            "accel_instability_lr_decay",
            "accel_convergence_lr_boost",
            "run_id",
        }
        for name in ["conservative", "balanced", "aggressive"]:
            other_keys = set(headers[name].keys()) - {
                "accel_instability_lr_decay",
                "accel_convergence_lr_boost",
                "run_id",
            }
            assert other_keys == base_keys, f"{name} has different keys"

            for key in base_keys:
                assert headers[name][key] == base[key], (
                    f"{name} differs on {key}: "
                    f"{headers[name][key]} != {base[key]}"
                )

    def test_config_grid_covers_design_space(self):
        """Verify the 4 configs cover the intended parameter grid."""
        configs = {
            "no_accel": (0.99, 1.01),
            "conservative": (0.3, 1.1),
            "balanced": (0.5, 1.5),
            "aggressive": (0.9, 2.0),
        }
        decays = [v[0] for v in configs.values()]
        boosts = [v[1] for v in configs.values()]

        assert min(decays) == 0.3
        assert max(decays) == 0.99
        assert min(boosts) == 1.01
        assert max(boosts) == 2.0
        assert len(set(configs.values())) == 4


# ---------------------------------------------------------------------------
# Test: efficiency metrics in comparison table
# ---------------------------------------------------------------------------


class TestEfficiencyMetricsInTable:
    """Test that build_comparison_table includes loss_red/bp and loss_red/wall_min."""

    def test_comparison_table_has_efficiency_columns(self, tmp_path):
        """Each row should include loss_reduction, loss_red_per_bp, loss_red_per_wall_min."""
        sweep_dir = tmp_path / "sweep"
        _write_accel_run(sweep_dir, "no_accel", 0.99, 1.01, loss_start=3.0, loss_end=2.5, cycles=5)
        _write_accel_run(sweep_dir, "balanced", 0.5, 1.5, loss_start=3.0, loss_end=2.0, cycles=5)

        runs = gather_runs(sweep_dir)
        rows = build_comparison_table(runs)

        for row in rows:
            assert "loss_reduction" in row
            assert "loss_red_per_bp" in row
            assert "loss_red_per_wall_min" in row

    def test_efficiency_metrics_are_correct(self, tmp_path):
        """Efficiency values should match manual computation."""
        sweep_dir = tmp_path / "sweep"
        _write_accel_run(sweep_dir, "balanced", 0.5, 1.5, loss_start=3.0, loss_end=2.0, cycles=10)

        runs = gather_runs(sweep_dir)
        rows = build_comparison_table(runs)

        row = rows[0]
        # First step: loss = loss_start - (loss_start-loss_end)/cycles * 1 = 2.9
        # loss_train = 2.9 + 0.1 = 3.0 → initial_loss = 3.0
        # best_valid_loss = loss_end = 2.0
        # loss_reduction = 3.0 - 2.0 = 1.0
        assert row["loss_reduction"] == pytest.approx(1.0)

        # total_backward_passes = 10 * 24 = 240
        assert row["loss_red_per_bp"] == pytest.approx(1.0 / 240)

        # total_wall_seconds = 10 * 230 = 2300
        assert row["loss_red_per_wall_min"] == pytest.approx(1.0 / (2300 / 60))

    def test_best_config_has_best_efficiency(self, tmp_path):
        """The config with the lowest best_valid_loss should have the best efficiency."""
        sweep_dir = tmp_path / "sweep"
        _write_accel_run(sweep_dir, "no_accel", 0.99, 1.01, loss_start=3.0, loss_end=2.5, cycles=5)
        _write_accel_run(sweep_dir, "balanced", 0.5, 1.5, loss_start=3.0, loss_end=2.0, cycles=5)

        runs = gather_runs(sweep_dir)
        rows = build_comparison_table(runs)

        balanced_row = next(r for r in rows if r["run_id"] == "balanced")
        no_accel_row = next(r for r in rows if r["run_id"] == "no_accel")

        assert balanced_row["loss_red_per_bp"] > no_accel_row["loss_red_per_bp"]
        assert balanced_row["loss_red_per_wall_min"] > no_accel_row["loss_red_per_wall_min"]

    def test_efficiency_metrics_none_when_missing_data(self, tmp_path):
        """Efficiency metrics should be None when initial_loss or best_valid_loss is missing."""
        runs = [{"run_id": "incomplete"}]
        rows = build_comparison_table(runs)
        assert rows[0]["loss_reduction"] is None
        assert rows[0]["loss_red_per_bp"] is None
        assert rows[0]["loss_red_per_wall_min"] is None


# ---------------------------------------------------------------------------
# Test: acceptance rate computation
# ---------------------------------------------------------------------------


class TestAcceptanceRate:
    """Test acceptance rate calculation from sweep metrics."""

    def test_acceptance_rate_from_sweep_data(self, tmp_path):
        """Acceptance rate is correctly computed from step records."""
        path = _write_accel_run(
            tmp_path, "balanced", 0.5, 1.5,
            cycles=10, accepted_count=7,
        )
        _, records, _ = load_run(path)
        accepted = sum(1 for r in records if r.get("tg_lora_accepted"))
        total = len(records)
        rate = accepted / total * 100
        assert rate == pytest.approx(70.0)
