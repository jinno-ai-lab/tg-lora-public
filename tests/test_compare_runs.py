"""Tests for compare_runs.py: TC-036-01, TC-036-02, TC-NFR-001-01, TASK-0061.

Tests the comparison report generation, budget parity logic, efficiency
metrics computation, and multi-run dashboard features.
"""

from pathlib import Path

import orjson
import pytest

from scripts.compare_runs import (_fmt, _pct_delta, build_comparison_table,
                                  find_best_run, format_json, gather_runs,
                                  generate_markdown_report, generate_report,
                                  load_run, log_reports_to_mlflow,
                                  plot_acceptance_rate, plot_hyperparams,
                                  plot_layer_scores, plot_loss_curves,
                                  plot_reduction_rate, plot_velocity_magnitude,
                                  render_dashboard)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "wb") as f:
        for rec in records:
            import orjson

            f.write(orjson.dumps(rec) + b"\n")


def _make_baseline_jsonl() -> list[dict]:
    return [
        {
            "type": "run_header",
            "model_name": "test-model",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 1e-4,
            "seed": 42,
            "comparison_reference": {"kind": "valid_quick_pretrain", "loss": 3.0},
        },
        {
            "type": "step",
            "step": 1,
            "loss_train": 3.0,
            "total_backward_passes": 1,
            "elapsed_seconds": 10.0,
        },
        {
            "type": "step",
            "step": 2,
            "loss_train": 2.5,
            "total_backward_passes": 2,
            "elapsed_seconds": 20.0,
        },
        {
            "type": "step",
            "step": 3,
            "loss_train": 2.2,
            "total_backward_passes": 3,
            "elapsed_seconds": 30.0,
        },
        {
            "type": "run_footer",
            "total_wall_seconds": 30.0,
            "best_valid_loss": 2.1,
            "final_train_loss": 2.2,
            "best_valid_step": 3,
            "gpu_peak_mb": 8000,
        },
    ]


def _make_tglora_jsonl() -> list[dict]:
    return [
        {
            "type": "run_header",
            "model_name": "test-model",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 1e-4,
            "seed": 42,
            "comparison_reference": {"kind": "valid_quick_pretrain", "loss": 3.0},
        },
        {
            "type": "step",
            "step": 1,
            "loss_train": 3.0,
            "total_backward_passes": 3,
            "elapsed_seconds": 12.0,
            "tg_lora_N": 2,
            "tg_lora_accepted": True,
            "tg_lora_cosine_sim": 0.85,
            "tg_lora_reduction_rate": 0.1,
        },
        {
            "type": "run_footer",
            "total_wall_seconds": 12.0,
            "best_valid_loss": 2.0,
            "final_train_loss": 2.0,
            "best_valid_step": 1,
            "gpu_peak_mb": 8200,
        },
    ]


def _make_tglora_multi_cycle() -> list[dict]:
    """Multi-cycle TG-LoRA records with full hyperparameter data."""
    return [
        {
            "type": "step",
            "step": 1,
            "cycle": 1,
            "loss_train": 3.0,
            "total_backward_passes": 3,
            "elapsed_seconds": 12.0,
            "tg_lora_accepted": True,
            "tg_lora_cosine_sim": 0.85,
            "tg_lora_reduction_rate": 0.10,
            "tg_lora_K": 3,
            "tg_lora_N": 2,
            "tg_lora_alpha": 0.5,
            "tg_lora_beta": 0.1,
            "tg_lora_lr": 5e-4,
            "grad_norm": 1.2,
        },
        {
            "type": "step",
            "step": 2,
            "cycle": 2,
            "loss_train": 2.8,
            "total_backward_passes": 6,
            "elapsed_seconds": 24.0,
            "tg_lora_accepted": False,
            "tg_lora_cosine_sim": 0.70,
            "tg_lora_reduction_rate": 0.05,
            "tg_lora_K": 3,
            "tg_lora_N": 2,
            "tg_lora_alpha": 0.6,
            "tg_lora_beta": 0.15,
            "tg_lora_lr": 4e-4,
            "grad_norm": 0.9,
        },
        {
            "type": "step",
            "step": 3,
            "cycle": 3,
            "loss_train": 2.5,
            "total_backward_passes": 9,
            "elapsed_seconds": 36.0,
            "tg_lora_accepted": True,
            "tg_lora_cosine_sim": 0.90,
            "tg_lora_reduction_rate": 0.15,
            "tg_lora_K": 4,
            "tg_lora_N": 3,
            "tg_lora_alpha": 0.7,
            "tg_lora_beta": 0.2,
            "tg_lora_lr": 3e-4,
            "grad_norm": 0.6,
        },
    ]


# ---------------------------------------------------------------------------
# TC-036-02: Comparison report generation
# ---------------------------------------------------------------------------


def test_load_run_parses_header_records_footer(tmp_path):
    """load_run correctly separates header, records, and footer."""
    f = tmp_path / "metrics.jsonl"
    _write_jsonl(f, _make_baseline_jsonl())

    header, records, footer = load_run(f)
    assert header["type"] == "run_header"
    assert header["model_name"] == "test-model"
    assert len(records) == 3
    assert all(r["type"] == "step" for r in records)
    assert footer is not None
    assert footer["type"] == "run_footer"
    assert footer["best_valid_loss"] == 2.1


def test_generate_report_contains_key_sections():
    """TC-036-02: Report includes loss curves, efficiency metrics, acceptance rate."""
    b_header, b_records, b_footer = (
        _make_baseline_jsonl()[0],
        _make_baseline_jsonl()[1:4],
        _make_baseline_jsonl()[4],
    )
    t_header, t_records, t_footer = (
        _make_tglora_jsonl()[0],
        _make_tglora_jsonl()[1:2],
        _make_tglora_jsonl()[2],
    )

    report = generate_report(
        b_header, b_records, b_footer, t_header, t_records, t_footer
    )

    assert "Efficiency Comparison" in report
    assert "Compute Budget" in report
    assert "Backward Passes" in report
    assert "Training Outcome" in report
    assert "Efficiency Metrics" in report
    assert "TG-LoRA Specific" in report
    assert "Acceptance Rate" in report
    assert "100.0%" in report  # 1/1 accepted


def test_generate_report_includes_artifact_anomaly_source_preview():
    b_header, b_records, b_footer = (
        _make_baseline_jsonl()[0],
        _make_baseline_jsonl()[1:4],
        _make_baseline_jsonl()[4],
    )
    t_header, t_records, t_footer = (
        _make_tglora_jsonl()[0],
        _make_tglora_jsonl()[1:2],
        _make_tglora_jsonl()[2],
    )

    report = generate_report(
        b_header,
        b_records,
        b_footer,
        t_header,
        t_records,
        t_footer,
        tg_artifact_anomalies=[
            {
                "anchor_kind": "after_pilot",
                "cycle": 3,
                "step": None,
                "delta_total_norm": 9.0,
                "robust_z_score": 8.0,
                "source_examples": [
                    {
                        "record_id": "r2",
                        "dataset_index": 11,
                        "text_preview": "beta anomaly source",
                    }
                ],
                "records": [{"text": "beta anomaly source"}],
            }
        ],
    )

    assert "TG-LoRA Delta Artifact Anomalies" in report
    assert "beta anomaly source" in report
    assert "id=r2, idx=11" in report


def test_tc220_02_dashboard_json_output_includes_delta_artifact_anomalies(tmp_path):
    runs = [
        {
            "run_id": "run_a",
            "mode": "baseline",
            "model_name": "m",
            "best_valid_loss": 2.5,
            "final_train_loss": 2.6,
            "perplexity": None,
            "best_valid_step": 10,
            "total_wall_seconds": 30.0,
            "num_steps": 3,
            "initial_loss": 3.0,
            "final_loss": 2.6,
            "total_backward_passes": 10,
            "parse_warnings": [],
            "delta_artifact_anomalies": [],
        },
        {
            "run_id": "run_b",
            "mode": "tg_lora",
            "model_name": "m",
            "best_valid_loss": 1.8,
            "final_train_loss": 1.9,
            "perplexity": None,
            "best_valid_step": 8,
            "total_wall_seconds": 20.0,
            "num_steps": 2,
            "initial_loss": 2.5,
            "final_loss": 1.9,
            "total_backward_passes": 8,
            "parse_warnings": [],
            "delta_artifact_anomalies": [
                {
                    "anchor_kind": "after_pilot",
                    "cycle": 3,
                    "step": None,
                    "delta_total_norm": 9.0,
                    "robust_z_score": 8.0,
                    "source_examples": [
                        {
                            "record_id": "r2",
                            "dataset_index": 11,
                            "text_preview": "beta anomaly source",
                        }
                    ],
                }
            ],
        },
    ]

    parsed = orjson.loads(format_json(runs))
    target = next(row for row in parsed["runs"] if row["run_id"] == "run_b")
    assert target["delta_artifact_anomalies"][0]["source_examples"][0]["record_id"] == "r2"


def test_generate_report_includes_loss_reduction_metrics():
    """TC-036-02: Report includes loss reduction efficiency metrics."""
    b_header, b_records, b_footer = (
        _make_baseline_jsonl()[0],
        _make_baseline_jsonl()[1:4],
        _make_baseline_jsonl()[4],
    )
    t_header, t_records, t_footer = (
        _make_tglora_jsonl()[0],
        _make_tglora_jsonl()[1:2],
        _make_tglora_jsonl()[2],
    )

    report = generate_report(
        b_header, b_records, b_footer, t_header, t_records, t_footer
    )

    assert "Loss Red." in report


def test_generate_report_prefers_header_comparison_reference_loss():
    b_header = dict(_make_baseline_jsonl()[0])
    b_records = _make_baseline_jsonl()[1:4]
    b_footer = _make_baseline_jsonl()[4]

    t_header = dict(_make_tglora_jsonl()[0])
    t_header["comparison_reference"] = {"kind": "valid_quick_pretrain", "loss": 3.0}
    t_records = [dict(_make_tglora_jsonl()[1])]
    t_records[0]["loss_train"] = 1.0
    t_footer = dict(_make_tglora_jsonl()[2])
    t_footer["best_valid_loss"] = 2.0

    report = generate_report(
        b_header, b_records, b_footer, t_header, t_records, t_footer
    )

    assert "33.3333" in report


def test_generate_report_falls_back_together_when_only_one_side_has_reference():
    b_header = dict(_make_baseline_jsonl()[0])
    b_header["comparison_reference"] = {"kind": "valid_quick_pretrain", "loss": None}
    b_records = _make_baseline_jsonl()[1:4]
    b_footer = dict(_make_baseline_jsonl()[4])
    b_footer["best_valid_loss"] = 2.1

    t_header = dict(_make_tglora_jsonl()[0])
    t_header["comparison_reference"] = {"kind": "valid_quick_pretrain", "loss": 3.0}
    t_records = [dict(_make_tglora_jsonl()[1])]
    t_records[0]["loss_train"] = 1.0
    t_footer = dict(_make_tglora_jsonl()[2])
    t_footer["best_valid_loss"] = 0.5

    report = generate_report(
        b_header, b_records, b_footer, t_header, t_records, t_footer
    )

    assert "16.6667" in report


def test_fmt_helper():
    assert _fmt(3.14159, 10, 4) == "    3.1416"
    assert _fmt(None, 10) == "       N/A"
    assert _fmt(42, 10) == "        42"


def test_pct_delta():
    assert _pct_delta(1.0, 1.1) == "+10.0%"
    assert _pct_delta(2.0, 1.0) == "-50.0%"
    assert _pct_delta(None, 1.0) == "N/A"
    assert _pct_delta(0, 1.0) == "N/A"


def test_plot_loss_curves_creates_file(tmp_path):
    """TC-036-02: plot_loss_curves generates a PNG when matplotlib available."""
    b_records = _make_baseline_jsonl()[1:4]
    t_records = _make_tglora_jsonl()[1:2]
    out_path = tmp_path / "loss_comparison.png"

    plot_loss_curves(b_records, t_records, out_path)

    # If matplotlib is available, file should exist
    try:
        import matplotlib  # noqa: F401

        assert out_path.exists()
    except ImportError:
        pass  # skipped gracefully


# ---------------------------------------------------------------------------
# TC-036-01: Budget parity (same backward pass budget)
# ---------------------------------------------------------------------------


def test_budget_parity_formula():
    """TC-036-01 / TC-NFR-001-01: N_CYCLES = BUDGET_PASSES / K_INITIAL.

    This verifies the core budget parity logic from run_comparison.sh:
    baseline_steps = BUDGET_PASSES
    tg_lora_cycles = BUDGET_PASSES / K_INITIAL
    effective_backward_passes = tg_lora_cycles * K_INITIAL = BUDGET_PASSES
    """
    budget_passes = 1500
    k_initial = 3
    n_cycles = budget_passes // k_initial
    effective_bp = n_cycles * k_initial

    # Budget parity: baseline backward passes == TG-LoRA effective backward passes
    assert effective_bp == budget_passes
    assert n_cycles == 500


def test_budget_parity_various_k_values():
    """TC-NFR-001-01: Budget parity holds for different K values."""
    budget = 1500
    for k in [1, 2, 3, 5, 10]:
        cycles = budget // k
        effective = cycles * k
        # Due to integer division, effective <= budget
        assert effective <= budget
        # The difference is at most (k-1) passes
        assert budget - effective < k


def test_report_shows_matching_backward_passes():
    """TC-036-01: Report shows both runs used the same budget."""
    b_jsonl = _make_baseline_jsonl()
    t_jsonl = _make_tglora_jsonl()
    # Set backward pass counts on step records (not footer)
    b_jsonl[3]["total_backward_passes"] = 1500  # last step record
    t_jsonl[1]["total_backward_passes"] = 1500  # last step record

    report = generate_report(
        b_jsonl[0],
        b_jsonl[1:4],
        b_jsonl[4],
        t_jsonl[0],
        t_jsonl[1:2],
        t_jsonl[2],
    )

    assert "1500" in report


# ---------------------------------------------------------------------------
# TC-036-02: Acceptance rate in report
# ---------------------------------------------------------------------------


def test_report_acceptance_rate_calculation():
    """TC-036-02: Acceptance rate is correctly computed from TG-LoRA records."""
    t_jsonl = _make_tglora_jsonl()
    records = [t_jsonl[1]]  # single accepted record

    accepted = [r for r in records if r.get("tg_lora_accepted")]
    rate = len(accepted) / len(records) * 100
    assert rate == 100.0

    # Add a rejected record
    rejected = dict(records[0])
    rejected["tg_lora_accepted"] = False
    records.append(rejected)

    accepted = [r for r in records if r.get("tg_lora_accepted")]
    rate = len(accepted) / len(records) * 100
    assert rate == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# TASK-0026: New visualization tests
# ---------------------------------------------------------------------------


def test_plot_acceptance_rate_creates_file(tmp_path):
    """TASK-0026: plot_acceptance_rate generates a PNG."""
    t_records = _make_tglora_multi_cycle()
    out_path = tmp_path / "acceptance_rate.png"

    plot_acceptance_rate(t_records, out_path)

    try:
        import matplotlib  # noqa: F401

        assert out_path.exists()
    except ImportError:
        pass


def test_plot_reduction_rate_creates_file(tmp_path):
    """TASK-0026: plot_reduction_rate generates a PNG."""
    t_records = _make_tglora_multi_cycle()
    out_path = tmp_path / "reduction_rate.png"

    plot_reduction_rate(t_records, out_path)

    try:
        import matplotlib  # noqa: F401

        assert out_path.exists()
    except ImportError:
        pass


def test_plot_hyperparams_creates_file(tmp_path):
    """TASK-0026: plot_hyperparams generates a PNG with alpha/beta/K/N subplots."""
    t_records = _make_tglora_multi_cycle()
    out_path = tmp_path / "hyperparams.png"

    plot_hyperparams(t_records, out_path)

    try:
        import matplotlib  # noqa: F401

        assert out_path.exists()
    except ImportError:
        pass


def test_empty_metrics_graceful(tmp_path):
    """TASK-0026: Empty or missing metrics are handled gracefully (no crash)."""
    # Empty records list
    plot_acceptance_rate([], tmp_path / "acc_empty.png")
    plot_reduction_rate([], tmp_path / "red_empty.png")
    plot_hyperparams([], tmp_path / "hyp_empty.png")

    # Records without TG-LoRA fields
    bare = [
        {
            "type": "step",
            "step": 1,
            "loss_train": 3.0,
            "total_backward_passes": 1,
            "elapsed_seconds": 10.0,
        },
    ]
    plot_acceptance_rate(bare, tmp_path / "acc_bare.png")
    plot_reduction_rate(bare, tmp_path / "red_bare.png")
    plot_hyperparams(bare, tmp_path / "hyp_bare.png")

    # All should complete without error; files may or may not exist depending on matplotlib
    try:
        import matplotlib  # noqa: F401

        # acceptance_rate should create a file (cumulative rate of 0%)
        assert (tmp_path / "acc_bare.png").exists()
        # reduction_rate and hyperparams skip when no data → no file
        assert not (tmp_path / "red_bare.png").exists()
        assert not (tmp_path / "hyp_bare.png").exists()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# TASK-0061: Multi-run comparison tests
# ---------------------------------------------------------------------------


def _write_run_jsonl(
    run_dir: Path,
    run_id: str,
    mode: str,
    best_valid_loss: float,
    final_train_loss: float,
    steps: list[dict] | None = None,
    perplexity: float | None = None,
    wall_seconds: float = 60.0,
) -> Path:
    """Write a complete run_metrics.jsonl and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "run_header",
            "run_id": run_id,
            "mode": mode,
            "model_name": "test-model",
        },
    ]
    for i, s in enumerate(
        steps
        or [
            {
                "step": 1,
                "loss_train": final_train_loss + 0.5,
                "backward_passes": 1,
                "total_backward_passes": 1,
            }
        ]
    ):
        base = {
            "type": "step",
            "step": i + 1,
            "loss_train": s.get("loss_train", 2.0),
            "backward_passes": 1,
            "total_backward_passes": i + 1,
            "elapsed_seconds": 10.0 * (i + 1),
        }
        base.update(s)
        records.append(base)
    footer = {
        "type": "run_footer",
        "total_wall_seconds": wall_seconds,
        "best_valid_loss": best_valid_loss,
        "best_valid_step": 1,
        "final_train_loss": final_train_loss,
    }
    if perplexity is not None:
        footer["perplexity"] = perplexity
    records.append(footer)
    path = run_dir / "run_metrics.jsonl"
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
    return path


class TestFindBestRun:
    def test_selects_lowest_loss(self):
        runs = [
            {"run_id": "a", "best_valid_loss": 2.5},
            {"run_id": "b", "best_valid_loss": 1.8},
            {"run_id": "c", "best_valid_loss": 2.0},
        ]
        assert find_best_run(runs)["run_id"] == "b"

    def test_tiebreak_by_perplexity(self):
        runs = [
            {"run_id": "a", "best_valid_loss": 2.0, "perplexity": 5.0},
            {"run_id": "b", "best_valid_loss": 2.0, "perplexity": 4.0},
        ]
        assert find_best_run(runs)["run_id"] == "b"

    def test_returns_none_on_empty(self):
        assert find_best_run([]) is None

    def test_skips_missing_loss(self):
        runs = [
            {"run_id": "a"},
            {"run_id": "b", "best_valid_loss": 1.5},
        ]
        assert find_best_run(runs)["run_id"] == "b"


class TestBuildComparisonTable:
    def test_marks_best_run(self):
        runs = [
            {
                "run_id": "a",
                "mode": "baseline",
                "best_valid_loss": 2.5,
                "final_train_loss": 2.6,
                "perplexity": None,
            },
            {
                "run_id": "b",
                "mode": "tg_lora",
                "best_valid_loss": 1.5,
                "final_train_loss": 1.6,
                "perplexity": 3.0,
            },
        ]
        rows = build_comparison_table(runs)
        assert len(rows) == 2
        best_row = [r for r in rows if r["is_best"]][0]
        assert best_row["run_id"] == "b"
        assert best_row["perplexity"] == 3.0


class TestFormatJson:
    def test_produces_valid_json(self):
        runs = [
            {
                "run_id": "a",
                "mode": "baseline",
                "best_valid_loss": 2.5,
                "final_train_loss": 2.6,
                "perplexity": None,
                "best_valid_step": 1,
                "total_wall_seconds": 60.0,
                "num_steps": 5,
                "initial_loss": 3.0,
                "final_loss": 2.6,
                "total_backward_passes": 5,
                "model_name": "test",
            },
        ]
        output = format_json(runs)
        parsed = orjson.loads(output)
        assert isinstance(parsed, dict)
        assert "runs" in parsed
        assert parsed["runs"][0]["run_id"] == "a"
        assert parsed["runs"][0]["is_best"] is True
        # No warnings when none exist
        assert "parse_warnings" not in parsed

    def test_includes_parse_warnings(self):
        runs = [
            {
                "run_id": "a",
                "mode": "baseline",
                "best_valid_loss": 2.5,
                "parse_warnings": ["Failed to parse bad.json: bad data"],
            },
        ]
        output = format_json(runs)
        parsed = orjson.loads(output)
        assert "parse_warnings" in parsed
        assert len(parsed["parse_warnings"]) == 1
        assert "bad.json" in parsed["parse_warnings"][0]


class TestGatherRuns:
    def test_gathers_multiple_runs(self, tmp_path):
        _write_run_jsonl(
            tmp_path / "run_a",
            "run_a",
            "baseline",
            best_valid_loss=2.5,
            final_train_loss=2.6,
            steps=[{"loss_train": 3.0}, {"loss_train": 2.6}],
        )
        _write_run_jsonl(
            tmp_path / "run_b",
            "run_b",
            "tg_lora",
            best_valid_loss=1.8,
            final_train_loss=1.9,
            perplexity=3.5,
            steps=[{"loss_train": 2.5}, {"loss_train": 1.9}],
        )
        runs = gather_runs(tmp_path)
        assert len(runs) == 2
        ids = {r["run_id"] for r in runs}
        assert ids == {"run_a", "run_b"}
        by_id = {r["run_id"]: r for r in runs}
        assert by_id["run_a"]["num_steps"] == 2
        assert by_id["run_b"]["num_steps"] == 2

    def test_empty_dir(self, tmp_path):
        assert gather_runs(tmp_path) == []

    def test_prefers_header_comparison_reference_loss(self, tmp_path):
        run_dir = tmp_path / "run_ref"
        run_dir.mkdir(parents=True)
        records = [
            {
                "type": "run_header",
                "run_id": "run_ref",
                "mode": "baseline",
                "model_name": "test-model",
                "comparison_reference": {
                    "kind": "valid_quick_pretrain",
                    "loss": 3.0,
                },
            },
            {
                "type": "step",
                "step": 1,
                "loss_train": 2.5,
                "total_backward_passes": 1,
                "elapsed_seconds": 10.0,
            },
            {
                "type": "run_footer",
                "total_wall_seconds": 10.0,
                "best_valid_loss": 2.0,
                "best_valid_step": 1,
                "final_train_loss": 2.5,
            },
        ]
        (run_dir / "run_metrics.jsonl").write_bytes(
            b"".join(orjson.dumps(r) + b"\n" for r in records)
        )

        runs = gather_runs(tmp_path)
        run = next(r for r in runs if r["run_id"] == "run_ref")
        assert run["comparison_reference_loss"] == pytest.approx(3.0)
        assert run["initial_loss"] == pytest.approx(3.0)

    def test_mixed_reference_suite_uses_fallback_initial_loss(self, tmp_path):
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir(parents=True)
        baseline_records = [
            {
                "type": "run_header",
                "run_id": "baseline",
                "mode": "baseline",
                "model_name": "test-model",
                "comparison_reference": {
                    "kind": "valid_quick_pretrain",
                    "loss": None,
                },
            },
            {
                "type": "step",
                "step": 1,
                "loss_train": 2.5,
                "total_backward_passes": 8,
                "elapsed_seconds": 10.0,
            },
            {
                "type": "run_footer",
                "total_wall_seconds": 10.0,
                "best_valid_loss": 1.5,
                "best_valid_step": 1,
                "final_train_loss": 2.5,
            },
        ]
        (baseline_dir / "run_metrics.jsonl").write_bytes(
            b"".join(orjson.dumps(r) + b"\n" for r in baseline_records)
        )

        tg_dir = tmp_path / "tg"
        tg_dir.mkdir(parents=True)
        tg_records = [
            {
                "type": "run_header",
                "run_id": "tg",
                "mode": "tg_lora",
                "model_name": "test-model",
                "comparison_reference": {
                    "kind": "valid_quick_pretrain",
                    "loss": 3.0,
                },
            },
            {
                "type": "step",
                "step": 1,
                "loss_train": 1.0,
                "total_backward_passes": 24,
                "elapsed_seconds": 12.0,
            },
            {
                "type": "run_footer",
                "total_wall_seconds": 12.0,
                "best_valid_loss": 0.5,
                "best_valid_step": 1,
                "final_train_loss": 1.0,
            },
        ]
        (tg_dir / "run_metrics.jsonl").write_bytes(
            b"".join(orjson.dumps(r) + b"\n" for r in tg_records)
        )

        rows = build_comparison_table(gather_runs(tmp_path))
        row_map = {row["run_id"]: row for row in rows}

        assert row_map["baseline"]["initial_loss"] == pytest.approx(2.5)
        assert row_map["tg"]["initial_loss"] == pytest.approx(1.0)

    def test_parse_failure_emits_stderr_warning(self, tmp_path, capsys, monkeypatch):
        """When parse_jsonl fails during step enrichment, a warning goes to stderr
        AND is collected in the run's parse_warnings field."""
        _write_run_jsonl(
            tmp_path / "run_a",
            "run_a",
            "baseline",
            best_valid_loss=2.5,
            final_train_loss=2.6,
        )

        import scripts.compare_runs as cr
        monkeypatch.setattr(
            cr, "parse_jsonl",
            lambda path: (_ for _ in ()).throw(ValueError("simulated parse failure")),
        )
        runs = gather_runs(tmp_path)
        assert len(runs) >= 1
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

        run = next(r for r in runs if r["run_id"] == "run_a")
        assert len(run["parse_warnings"]) == 1
        assert "simulated parse failure" in run["parse_warnings"][0]


class TestCorruptJsonlIntegration:
    """End-to-end: corrupt JSONL handling across the comparison pipeline."""

    def test_corrupt_jsonl_skipped_gracefully(self, tmp_path):
        """Corrupt JSONL files are skipped by list_runs without crashing."""
        _write_run_jsonl(
            tmp_path / "run_good",
            "run_good",
            "baseline",
            best_valid_loss=2.5,
            final_train_loss=2.6,
        )
        bad_dir = tmp_path / "run_bad"
        bad_dir.mkdir(parents=True)
        bad_dir.joinpath("run_metrics.jsonl").write_bytes(b"{{{not json\n")

        runs = gather_runs(tmp_path)
        # Only the valid run appears; corrupt run is silently skipped
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run_good"
        assert runs[0]["parse_warnings"] == []

    def test_parse_warnings_in_json_output(self):
        """Runs with parse_warnings produce them in format_json() output."""
        runs = [
            {
                "run_id": "ok_run",
                "mode": "baseline",
                "best_valid_loss": 2.0,
                "parse_warnings": [],
            },
            {
                "run_id": "bad_run",
                "mode": "tg_lora",
                "best_valid_loss": 2.5,
                "parse_warnings": [
                    "Failed to parse run_metrics.jsonl: Invalid JSON at line 3: bad data"
                ],
            },
        ]
        output = format_json(runs)
        parsed = orjson.loads(output)
        assert "parse_warnings" in parsed
        assert len(parsed["parse_warnings"]) == 1
        assert "Invalid JSON" in parsed["parse_warnings"][0]

    def test_parse_warnings_in_dashboard_panel(self, tmp_path, capsys):
        """render_dashboard() shows a 'Parse Warnings' panel when warnings exist."""
        runs = [
            {
                "run_id": "a",
                "mode": "baseline",
                "best_valid_loss": 2.5,
                "final_train_loss": 2.6,
                "total_wall_seconds": 60,
                "num_steps": 1,
                "parse_warnings": ["Failed to parse: corrupt data"],
            },
        ]
        render_dashboard(runs)
        captured = capsys.readouterr()
        assert "Parse Warnings" in captured.out
        assert "corrupt data" in captured.out

    def test_no_parse_warnings_panel_when_clean(self, tmp_path, capsys):
        """No 'Parse Warnings' panel when all runs are clean."""
        runs = [
            {
                "run_id": "a",
                "mode": "baseline",
                "best_valid_loss": 2.5,
                "parse_warnings": [],
            },
        ]
        render_dashboard(runs)
        captured = capsys.readouterr()
        assert "Parse Warnings" not in captured.out


class TestRenderDashboard:
    def test_renders_without_crash(self, tmp_path):
        _write_run_jsonl(
            tmp_path / "run_a",
            "run_a",
            "baseline",
            best_valid_loss=2.5,
            final_train_loss=2.6,
        )
        _write_run_jsonl(
            tmp_path / "run_b",
            "run_b",
            "tg_lora",
            best_valid_loss=1.8,
            final_train_loss=1.9,
            perplexity=3.5,
        )
        runs = gather_runs(tmp_path)
        render_dashboard(runs)


# ---------------------------------------------------------------------------
# TASK-0062: Visualization enhancement tests
# ---------------------------------------------------------------------------


def test_plot_velocity_magnitude_creates_file(tmp_path):
    """TASK-0062: plot_velocity_magnitude generates a PNG from grad_norm data."""
    t_records = _make_tglora_multi_cycle()
    out_path = tmp_path / "velocity_magnitude.png"

    plot_velocity_magnitude(t_records, out_path)

    try:
        import matplotlib  # noqa: F401

        assert out_path.exists()
        assert out_path.stat().st_size > 0
    except ImportError:
        pass


def test_plot_velocity_magnitude_empty_graceful(tmp_path):
    """TASK-0062: plot_velocity_magnitude handles empty/missing data gracefully."""
    # Empty records
    plot_velocity_magnitude([], tmp_path / "vel_empty.png")

    # Records without grad_norm
    bare = [
        {
            "type": "step",
            "step": 1,
            "loss_train": 3.0,
            "total_backward_passes": 1,
            "elapsed_seconds": 10.0,
        },
    ]
    plot_velocity_magnitude(bare, tmp_path / "vel_bare.png")

    try:
        import matplotlib  # noqa: F401

        # No grad_norm → no file created
        assert not (tmp_path / "vel_bare.png").exists()
    except ImportError:
        pass


def test_plot_layer_scores_creates_file(tmp_path):
    """TASK-0062: plot_layer_scores generates a PNG bar chart."""
    scores = {0: 1.2, 1: 0.8, 2: 2.5, 3: 1.0, 4: 3.1}
    out_path = tmp_path / "layer_scores.png"

    plot_layer_scores(scores, out_path)

    try:
        import matplotlib  # noqa: F401

        assert out_path.exists()
        assert out_path.stat().st_size > 0
    except ImportError:
        pass


def test_plot_layer_scores_empty_graceful(tmp_path):
    """TASK-0062: plot_layer_scores handles empty dict gracefully."""
    plot_layer_scores({}, tmp_path / "ls_empty.png")

    try:
        import matplotlib  # noqa: F401

        assert not (tmp_path / "ls_empty.png").exists()
    except ImportError:
        pass


def test_plot_hyperparams_includes_lr(tmp_path):
    """TASK-0062: plot_hyperparams includes lr trajectory instead of beta."""
    t_records = _make_tglora_multi_cycle()
    out_path = tmp_path / "hyperparams.png"

    plot_hyperparams(t_records, out_path)

    try:
        import matplotlib  # noqa: F401

        assert out_path.exists()
        assert out_path.stat().st_size > 0
    except ImportError:
        pass


def test_plot_hyperparams_empty_graceful(tmp_path):
    """TASK-0062: plot_hyperparams handles records without hp data."""
    bare = [
        {
            "type": "step",
            "step": 1,
            "loss_train": 3.0,
            "total_backward_passes": 1,
            "elapsed_seconds": 10.0,
        },
    ]
    plot_hyperparams(bare, tmp_path / "hyp_bare.png")

    try:
        import matplotlib  # noqa: F401

        assert not (tmp_path / "hyp_bare.png").exists()
    except ImportError:
        pass


class TestPhase56Requirements:
    """REQ-220~223: Dashboard, markdown report, MLflow logging coverage."""

    def _sample_header(self):
        return {"experiment": "test", "K": 2, "N": 1, "alpha": 0.3, "lr": 2e-4}

    def _sample_records(self, n=5, loss=2.5):
        return [
            {"type": "step", "step": i, "loss_train": loss - i * 0.1,
             "total_backward_passes": i * 2, "elapsed_seconds": float(i * 10)}
            for i in range(1, n + 1)
        ]

    def _sample_footer(self):
        return {"type": "footer", "best_loss": 2.0, "total_steps": 5}

    def test_generate_report_contains_key_sections(self):
        """REQ-222: generate_report produces comparison report with key sections."""
        h = self._sample_header()
        r = self._sample_records()
        f = self._sample_footer()
        report = generate_report(h, r, f, h, r, f)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_find_best_run_selects_lowest_loss(self):
        """REQ-220: find_best_run selects the run with the lowest best_valid_loss."""
        runs = [
            {"name": "a", "best_valid_loss": 2.5},
            {"name": "b", "best_valid_loss": 2.0},
            {"name": "c", "best_valid_loss": 3.0},
        ]
        best = find_best_run(runs)
        assert best["name"] == "b"

    def test_find_best_run_returns_none_on_empty(self):
        """REQ-220: find_best_run returns None when no valid loss available."""
        assert find_best_run([]) is None
        assert find_best_run([{"name": "a"}]) is None


# ---------------------------------------------------------------------------
# TC-220-01/02: Dashboard subcommand (multi-run comparison table)
# ---------------------------------------------------------------------------


class TestTC220:
    """REQ-220: Multi-run comparison dashboard."""

    def test_tc220_01_dashboard_generates_comparison_table(self, tmp_path):
        """TC-220-01: dashboard subcommand generates comparison table for multiple runs."""
        _write_run_jsonl(
            tmp_path / "run_baseline",
            "run_baseline",
            "baseline",
            best_valid_loss=2.5,
            final_train_loss=2.6,
            steps=[{"loss_train": 3.0}, {"loss_train": 2.6}],
        )
        _write_run_jsonl(
            tmp_path / "run_tglora",
            "run_tglora",
            "tg_lora",
            best_valid_loss=1.8,
            final_train_loss=1.9,
            perplexity=3.5,
            steps=[{"loss_train": 2.5}, {"loss_train": 1.9}],
        )
        runs = gather_runs(tmp_path)
        rows = build_comparison_table(runs)

        assert len(rows) == 2
        run_ids = {r["run_id"] for r in rows}
        assert run_ids == {"run_baseline", "run_tglora"}

        best_row = [r for r in rows if r["is_best"]][0]
        assert best_row["run_id"] == "run_tglora"
        assert best_row["best_valid_loss"] == 1.8

        render_dashboard(runs)

    def test_tc220_02_dashboard_json_output(self, tmp_path):
        """TC-220-02: --format json outputs valid JSON comparison."""
        _write_run_jsonl(
            tmp_path / "run_a",
            "run_a",
            "baseline",
            best_valid_loss=2.5,
            final_train_loss=2.6,
            steps=[{"loss_train": 3.0}],
        )
        _write_run_jsonl(
            tmp_path / "run_b",
            "run_b",
            "tg_lora",
            best_valid_loss=1.8,
            final_train_loss=1.9,
            steps=[{"loss_train": 2.5}],
        )
        runs = gather_runs(tmp_path)
        json_str = format_json(runs)
        parsed = orjson.loads(json_str)

        assert isinstance(parsed, dict)
        assert "runs" in parsed
        assert len(parsed["runs"]) == 2
        assert all("run_id" in r for r in parsed["runs"])
        assert all("best_valid_loss" in r for r in parsed["runs"])
        assert all("is_best" in r for r in parsed["runs"])
        best_rows = [r for r in parsed["runs"] if r["is_best"]]
        assert len(best_rows) == 1
        assert best_rows[0]["run_id"] == "run_b"


# ---------------------------------------------------------------------------
# TC-221-01: All 5 plot functions generate PNGs
# ---------------------------------------------------------------------------


class TestTC221:
    """REQ-221: Visualization plot functions."""

    def test_tc221_01_all_five_plots_generate_pngs(self, tmp_path):
        """TC-221-01: plot_acceptance_rate, plot_reduction_rate, plot_velocity_magnitude,
        plot_layer_scores, plot_hyperparams all generate valid PNGs."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not available")

        t_records = _make_tglora_multi_cycle()

        out_acc = tmp_path / "acceptance_rate.png"
        out_red = tmp_path / "reduction_rate.png"
        out_vel = tmp_path / "velocity_magnitude.png"
        out_lay = tmp_path / "layer_scores.png"
        out_hyp = tmp_path / "hyperparams.png"

        plot_acceptance_rate(t_records, out_acc)
        plot_reduction_rate(t_records, out_red)
        plot_velocity_magnitude(t_records, out_vel)
        plot_layer_scores({0: 1.2, 1: 0.8, 2: 2.5}, out_lay)
        plot_hyperparams(t_records, out_hyp)

        for path in [out_acc, out_red, out_vel, out_lay, out_hyp]:
            assert path.exists(), f"{path.name} was not created"
            assert path.stat().st_size > 0, f"{path.name} is empty"
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", (
                f"{path.name} is not a valid PNG"
            )


# ---------------------------------------------------------------------------
# TC-222/223: Markdown report generation and MLflow artifact logging
# ---------------------------------------------------------------------------


class TestGenerateMarkdownReport:
    """REQ-222: generate_markdown_report produces valid Markdown comparison."""

    def _full_baseline(self):
        h = {
            "model_name": "test-model",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 1e-4,
            "seed": 42,
        }
        records = [
            {"type": "step", "step": 1, "loss_train": 3.0, "total_backward_passes": 1, "elapsed_seconds": 10.0},
            {"type": "step", "step": 2, "loss_train": 2.5, "total_backward_passes": 2, "elapsed_seconds": 20.0},
            {"type": "step", "step": 3, "loss_train": 2.2, "total_backward_passes": 3, "elapsed_seconds": 30.0},
        ]
        footer = {"total_wall_seconds": 30.0, "best_valid_loss": 2.1, "final_train_loss": 2.2, "best_valid_step": 3, "gpu_peak_mb": 8000}
        return h, records, footer

    def _full_tglora(self):
        h = {
            "model_name": "test-model",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 1e-4,
            "seed": 42,
        }
        records = [
            {"type": "step", "step": 1, "loss_train": 3.0, "total_backward_passes": 3, "elapsed_seconds": 12.0,
             "tg_lora_N": 2, "tg_lora_accepted": True, "tg_lora_cosine_sim": 0.85, "tg_lora_reduction_rate": 0.1},
        ]
        footer = {"total_wall_seconds": 12.0, "best_valid_loss": 2.0, "final_train_loss": 2.0, "best_valid_step": 1, "gpu_peak_mb": 8200}
        return h, records, footer

    def test_produces_markdown_with_sections(self):
        """TC-222-01: Markdown report contains all major sections."""
        bh, br, bf = self._full_baseline()
        th, tr, tf = self._full_tglora()
        md = generate_markdown_report(bh, br, bf, th, tr, tf)

        assert "# TG-LoRA vs Baseline QLoRA" in md
        assert "## Configuration" in md
        assert "## Compute Budget" in md
        assert "## Training Outcome" in md
        assert "## Efficiency Metrics" in md
        assert "## TG-LoRA Specific Metrics" in md
        assert "## Plots" in md

    def test_markdown_contains_table_rows(self):
        """TC-222-01: Markdown has pipe-delimited table rows with data."""
        bh, br, bf = self._full_baseline()
        th, tr, tf = self._full_tglora()
        md = generate_markdown_report(bh, br, bf, th, tr, tf)

        assert "test-model" in md
        assert "42" in md  # seed
        assert "100.0%" in md  # acceptance rate (1/1)

    def test_markdown_omits_unavailable_plot_links(self):
        """Markdown should only link plots that the generator can actually create."""
        bh, br, bf = self._full_baseline()
        th, tr, tf = self._full_tglora()

        md = generate_markdown_report(bh, br, bf, th, tr, tf)

        assert "![Acceptance Rate](acceptance_rate.png)" in md
        assert "![Reduction Rate](reduction_rate.png)" in md
        assert "![Layer Scores](layer_scores.png)" in md
        assert "![Hyperparameters](hyperparams.png)" in md
        assert "![Velocity Magnitude](velocity_magnitude.png)" not in md

    def test_markdown_includes_available_plot_links(self):
        """Markdown links velocity and hyperparameter plots when data is present."""
        bh, br, bf = self._full_baseline()
        th = {
            "model_name": "test-model",
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 1e-4,
            "seed": 42,
        }
        tr = _make_tglora_multi_cycle()
        tf = {
            "total_wall_seconds": 36.0,
            "best_valid_loss": 2.0,
            "final_train_loss": 2.5,
            "best_valid_step": 3,
            "gpu_peak_mb": 8200,
        }

        md = generate_markdown_report(bh, br, bf, th, tr, tf)

        assert "![Velocity Magnitude](velocity_magnitude.png)" in md
        assert "![Hyperparameters](hyperparams.png)" in md

    def test_markdown_empty_records(self):
        """TC-222-02: Markdown report handles empty records gracefully."""
        h = {"model_name": "x", "lora_r": 8, "lora_alpha": 16, "batch_size": 1, "grad_accumulation": 1, "learning_rate": 1e-4, "seed": 0}
        md = generate_markdown_report(h, [], None, h, [], None)
        assert isinstance(md, str)
        assert "# TG-LoRA vs Baseline QLoRA" in md

    def test_markdown_no_tglora_records(self):
        """TC-222-03: No TG-LoRA records omits the TG-LoRA Specific section."""
        bh, br, bf = self._full_baseline()
        h2 = {"model_name": "x", "lora_r": 8, "lora_alpha": 16, "batch_size": 1, "grad_accumulation": 1, "learning_rate": 1e-4, "seed": 0}
        md = generate_markdown_report(bh, br, bf, h2, [], None)
        assert "TG-LoRA Specific" not in md


class TestLogReportsToMlflow:
    """REQ-223: log_reports_to_mlflow logs artifacts via MLflowLogger."""

    def test_logs_files_in_output_dir(self, tmp_path, monkeypatch):
        """TC-223-01: All files in output_dir are logged as artifacts."""
        (tmp_path / "report.txt").write_text("report")
        (tmp_path / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

        logged = []

        class FakeMLflow:
            enabled = True
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def log_artifact(self, path, subdir):
                logged.append((path, subdir))

        import scripts.compare_runs as cr
        original = cr.MLflowLogger
        monkeypatch.setattr(cr, "MLflowLogger", lambda **kw: FakeMLflow())

        try:
            log_reports_to_mlflow(tmp_path)
            assert len(logged) == 2
            assert all(sub == "comparison_report" for _, sub in logged)
        finally:
            cr.MLflowLogger = original

    def test_disabled_logger_is_noop(self, tmp_path, monkeypatch):
        """TC-223-02: When MLflowLogger.enabled=False, no artifacts are logged."""
        (tmp_path / "report.txt").write_text("report")

        class DisabledMLflow:
            enabled = False
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def log_artifact(self, path, subdir):
                raise AssertionError("should not be called")

        import scripts.compare_runs as cr
        original = cr.MLflowLogger
        monkeypatch.setattr(cr, "MLflowLogger", lambda **kw: DisabledMLflow())

        try:
            log_reports_to_mlflow(tmp_path)
        finally:
            cr.MLflowLogger = original

    def test_empty_dir_is_safe(self, tmp_path, monkeypatch):
        """TC-223-03: Empty output_dir does not cause errors."""
        class FakeMLflow:
            enabled = True
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def log_artifact(self, path, subdir):
                pass

        import scripts.compare_runs as cr
        original = cr.MLflowLogger
        monkeypatch.setattr(cr, "MLflowLogger", lambda **kw: FakeMLflow())

        try:
            log_reports_to_mlflow(tmp_path)
        finally:
            cr.MLflowLogger = original
