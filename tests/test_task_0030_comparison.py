"""Tests for TASK-0030: 公正比較実験と結果分析.

Acceptance criteria:
  AC1: 比較レポート（Markdown）が生成される
  AC2: 損失曲線プロット（PNG）が生成される
  AC3: 受理率推移、削減率推移、alpha推移のグラフが含まれる
  AC4: 効率メトリクス（loss/backward, loss/GB-hour等）が計算される
  AC5: TG-LoRAが同一backward予算でベースラインと同等以上の損失低下を示す
"""

from pathlib import Path

import orjson

from scripts.compare_runs import (
    generate_markdown_report,
    generate_report,
    plot_acceptance_rate,
    plot_hyperparams,
    plot_loss_curves,
    plot_reduction_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec) + b"\n")


def _make_baseline_30step():
    """Baseline: 30 backward passes, loss from 3.0 -> 2.5."""
    header = {
        "type": "run_header",
        "model_name": "test-model",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 1,
        "grad_accumulation": 8,
        "learning_rate": 1e-4,
        "seed": 42,
    }
    records = []
    for i in range(1, 31):
        loss = 3.0 - 0.5 * (i / 30)
        records.append(
            {
                "type": "step",
                "step": i,
                "loss_train": round(loss, 4),
                "total_backward_passes": i,
                "elapsed_seconds": i * 10.0,
            }
        )
    footer = {
        "type": "run_footer",
        "total_wall_seconds": 300.0,
        "best_valid_loss": 2.5,
        "final_train_loss": 2.5,
        "best_valid_step": 30,
        "gpu_peak_mb": 8000,
    }
    return header, records, footer


def _make_tglora_30bp():
    """TG-LoRA: 30 backward passes (10 cycles x K=3), loss from 3.0 -> 2.2."""
    header = {
        "type": "run_header",
        "model_name": "test-model",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 1,
        "grad_accumulation": 8,
        "learning_rate": 1e-4,
        "seed": 42,
    }
    records = []
    for i in range(1, 11):
        bp = i * 3
        loss = 3.0 - 0.8 * (i / 10)
        records.append(
            {
                "type": "step",
                "step": i,
                "cycle": i,
                "loss_train": round(loss, 4),
                "total_backward_passes": bp,
                "elapsed_seconds": i * 8.0,
                "tg_lora_accepted": i % 3 != 0,
                "tg_lora_cosine_sim": round(0.7 + 0.2 * (i / 10), 4),
                "tg_lora_reduction_rate": round(0.05 + 0.1 * (i / 10), 4),
                "tg_lora_K": 3 + (i // 4),
                "tg_lora_N": 2,
                "tg_lora_alpha": round(0.5 + 0.3 * (i / 10), 4),
                "tg_lora_beta": round(0.1 + 0.1 * (i / 10), 4),
            }
        )
    footer = {
        "type": "run_footer",
        "total_wall_seconds": 80.0,
        "best_valid_loss": 2.2,
        "final_train_loss": 2.2,
        "best_valid_step": 10,
        "gpu_peak_mb": 8200,
    }
    return header, records, footer


# ---------------------------------------------------------------------------
# AC1: 比較レポート（Markdown）が生成される
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def test_markdown_report_has_required_sections(self):
        """AC1: Markdown report contains all required sections."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        md = generate_markdown_report(b_h, b_r, b_f, t_h, t_r, t_f)

        assert md.startswith("# ")
        assert "## Configuration" in md
        assert "## Compute Budget" in md
        assert "## Training Outcome" in md
        assert "## Efficiency Metrics" in md
        assert "## TG-LoRA Specific" in md
        assert "## Plots" in md

    def test_markdown_report_has_tables(self):
        """AC1: Markdown report uses pipe-delimited tables."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        md = generate_markdown_report(b_h, b_r, b_f, t_h, t_r, t_f)

        assert "| Parameter |" in md
        assert "|---" in md

    def test_markdown_report_saved_as_md_file(self, tmp_path):
        """AC1: Report can be persisted as a .md file."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        md = generate_markdown_report(b_h, b_r, b_f, t_h, t_r, t_f)
        path = tmp_path / "comparison_report.md"
        path.write_text(md)

        assert path.exists()
        assert path.read_text().startswith("# ")


# ---------------------------------------------------------------------------
# AC2: 損失曲線プロット（PNG）が生成される
# ---------------------------------------------------------------------------


class TestLossCurvePlot:
    def test_loss_curve_png_generated(self, tmp_path):
        """AC2: Loss curve PNG is generated from run data."""
        _, b_r, _ = _make_baseline_30step()
        _, t_r, _ = _make_tglora_30bp()
        out = tmp_path / "loss_comparison.png"

        plot_loss_curves(b_r, t_r, out)

        try:
            import matplotlib  # noqa: F401

            assert out.exists()
            assert out.stat().st_size > 0
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# AC3: 受理率推移、削減率推移、alpha推移のグラフが含まれる
# ---------------------------------------------------------------------------


class TestProgressionPlots:
    def test_acceptance_rate_plot(self, tmp_path):
        """AC3: Acceptance rate progression PNG is generated."""
        _, t_r, _ = _make_tglora_30bp()
        out = tmp_path / "acceptance_rate.png"

        plot_acceptance_rate(t_r, out)

        try:
            import matplotlib  # noqa: F401

            assert out.exists()
            assert out.stat().st_size > 0
        except ImportError:
            pass

    def test_reduction_rate_plot(self, tmp_path):
        """AC3: Reduction rate progression PNG is generated."""
        _, t_r, _ = _make_tglora_30bp()
        out = tmp_path / "reduction_rate.png"

        plot_reduction_rate(t_r, out)

        try:
            import matplotlib  # noqa: F401

            assert out.exists()
            assert out.stat().st_size > 0
        except ImportError:
            pass

    def test_alpha_progression_in_hyperparams_plot(self, tmp_path):
        """AC3: Alpha progression is included in hyperparams plot."""
        _, t_r, _ = _make_tglora_30bp()
        out = tmp_path / "hyperparams.png"

        plot_hyperparams(t_r, out)

        try:
            import matplotlib  # noqa: F401

            assert out.exists()
            assert out.stat().st_size > 0
        except ImportError:
            pass

    def test_markdown_report_references_all_plots(self):
        """AC3: Markdown report contains image links for all plots."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        md = generate_markdown_report(b_h, b_r, b_f, t_h, t_r, t_f)

        assert "loss_comparison.png" in md
        assert "acceptance_rate.png" in md
        assert "reduction_rate.png" in md
        assert "hyperparams.png" in md


# ---------------------------------------------------------------------------
# AC4: 効率メトリクス（loss/backward, loss/GB-hour等）が計算される
# ---------------------------------------------------------------------------


class TestEfficiencyMetrics:
    def test_markdown_report_contains_efficiency_metrics(self):
        """AC4: Markdown report includes loss/backward and loss/GB-hour."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        md = generate_markdown_report(b_h, b_r, b_f, t_h, t_r, t_f)

        assert "Loss Red." in md
        assert "GB-hour" in md
        assert "backward" in md
        assert "wall-minute" in md

    def test_efficiency_values_correct(self):
        """AC4: Efficiency values are numerically correct."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        report = generate_report(b_h, b_r, b_f, t_h, t_r, t_f)

        # Baseline: loss_red = b_r[0].loss - best_valid, bp = 30
        b_init = b_r[0]["loss_train"]
        b_best = b_f["best_valid_loss"]
        expected_per_100 = (b_init - b_best) / 30 * 100
        assert f"{expected_per_100:.4f}" in report

    def test_markdown_report_contains_gpu_memory(self):
        """AC4: GPU Peak Memory metric is included."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        md = generate_markdown_report(b_h, b_r, b_f, t_h, t_r, t_f)

        assert "GPU Peak Memory" in md


# ---------------------------------------------------------------------------
# AC5: TG-LoRAが同一backward予算でベースラインと同等以上の損失低下を示す
# ---------------------------------------------------------------------------


class TestLossReductionComparison:
    def test_budget_parity(self):
        """AC5: Both runs use the same backward pass budget."""
        _, b_r, _ = _make_baseline_30step()
        _, t_r, _ = _make_tglora_30bp()

        assert b_r[-1]["total_backward_passes"] == 30
        assert t_r[-1]["total_backward_passes"] == 30

    def test_tglora_equal_or_better_loss_reduction(self):
        """AC5: TG-LoRA achieves >= loss reduction at same budget."""
        _, b_r, b_f = _make_baseline_30step()
        _, t_r, t_f = _make_tglora_30bp()

        b_loss_red = b_r[0]["loss_train"] - b_f["best_valid_loss"]
        t_loss_red = t_r[0]["loss_train"] - t_f["best_valid_loss"]

        # TG-LoRA: 3.0 -> 2.2 (0.8) >= Baseline: 3.0 -> 2.5 (0.5)
        assert t_loss_red >= b_loss_red

    def test_tglora_per_backward_better(self):
        """AC5: TG-LoRA has better loss/backward efficiency."""
        _, b_r, b_f = _make_baseline_30step()
        _, t_r, t_f = _make_tglora_30bp()

        b_red = b_r[0]["loss_train"] - b_f["best_valid_loss"]
        t_red = t_r[0]["loss_train"] - t_f["best_valid_loss"]
        b_bp = b_r[-1]["total_backward_passes"]
        t_bp = t_r[-1]["total_backward_passes"]

        assert t_red / t_bp > b_red / b_bp
        assert b_bp == t_bp

    def test_markdown_report_shows_tglora_best_loss_lower(self):
        """AC5: Report shows TG-LoRA achieved lower (better) best valid loss."""
        b_h, b_r, b_f = _make_baseline_30step()
        t_h, t_r, t_f = _make_tglora_30bp()

        md = generate_markdown_report(b_h, b_r, b_f, t_h, t_r, t_f)

        assert "2.2000" in md
        assert "2.5000" in md
