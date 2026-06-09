"""Tests for the performance benchmark script.

Verifies:
- reduction_rate is in [0, 1] range
- TG-LoRA equivalent steps >= standard LoRA steps
- Benchmark runs complete without error
- Report file is generated in correct Markdown format
- Long-running tests marked with @pytest.mark.slow
"""
from __future__ import annotations

import math

import pytest
import torch

from scripts.benchmark import (
    BenchmarkResult,
    SimpleLoRAModel,
    _run_standard_lora,
    _run_tg_lora,
    format_table,
    main,
    run_benchmarks,
)


class TestBenchmarkResult:
    def test_standard_lora_reduction_rate_in_range(self):
        result = _run_standard_lora(cycles=3)
        assert 0.0 <= result.reduction_rate <= 1.0

    def test_tg_lora_reduction_rate_in_range(self):
        result = _run_tg_lora(cycles=3, K=2, N=1)
        assert 0.0 <= result.reduction_rate <= 1.0

    def test_tg_lora_equivalent_steps_gte_standard(self):
        std = _run_standard_lora(cycles=3, K=3)
        tg = _run_tg_lora(cycles=3, K=3, N=2)
        assert tg.equivalent_steps >= std.equivalent_steps

    def test_tg_lora_with_small_N_has_more_equivalent_steps(self):
        std = _run_standard_lora(cycles=3, K=2)
        tg = _run_tg_lora(cycles=3, K=2, N=1)
        assert tg.equivalent_steps >= std.equivalent_steps

    def test_final_loss_is_finite(self):
        result = _run_tg_lora(cycles=3, K=2, N=1)
        assert math.isfinite(result.final_loss)

    def test_acceptance_rate_in_range(self):
        result = _run_tg_lora(cycles=3, K=2, N=1)
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_wall_clock_positive(self):
        result = _run_standard_lora(cycles=2)
        assert result.wall_clock > 0.0


class TestBenchmarkRuns:
    def test_run_standard_completes(self):
        result = _run_standard_lora(cycles=2)
        assert result.setting == "Standard LoRA"
        assert result.equivalent_steps > 0

    def test_run_tg_lora_completes(self):
        result = _run_tg_lora(cycles=2, K=2, N=1)
        assert "TG-LoRA" in result.setting
        assert result.equivalent_steps > 0

    def test_run_multiple_K_N(self):
        configs = [(2, 1), (3, 2), (3, 5)]
        for K, N in configs:
            result = _run_tg_lora(cycles=2, K=K, N=N)
            assert f"K={K},N={N}" in result.setting
            assert result.reduction_rate >= 0.0


class TestFormatTable:
    def test_table_has_header(self):
        results = [_run_standard_lora(cycles=1)]
        table = format_table(results)
        assert "| Setting |" in table
        assert "|---------|" in table

    def test_table_has_all_columns(self):
        results = [_run_standard_lora(cycles=1)]
        table = format_table(results)
        assert "Wall-clock (s)" in table
        assert "Equivalent Steps" in table
        assert "reduction_rate" in table
        assert "Final Loss" in table
        assert "acceptance_rate" in table

    def test_table_standard_shows_dash_for_acceptance(self):
        results = [_run_standard_lora(cycles=1)]
        table = format_table(results)
        lines = table.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("| Standard")]
        assert len(data_lines) == 1
        assert "| - |" in data_lines[0]

    def test_table_tg_lora_shows_acceptance_rate(self):
        results = [_run_tg_lora(cycles=2, K=2, N=1)]
        table = format_table(results)
        lines = table.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("| TG-LoRA")]
        assert len(data_lines) == 1
        assert "%" in data_lines[0]


class TestRunBenchmarks:
    @pytest.mark.slow
    def test_run_benchmarks_returns_all_settings(self):
        results = run_benchmarks(cycles=2)
        settings = [r.setting for r in results]
        assert "Standard LoRA" in settings
        assert any("K=2,N=1" in s for s in settings)
        assert any("K=3,N=2" in s for s in settings)
        assert any("K=3,N=5" in s for s in settings)

    @pytest.mark.slow
    def test_run_benchmarks_all_reduction_rates_valid(self):
        results = run_benchmarks(cycles=2)
        for r in results:
            assert 0.0 <= r.reduction_rate <= 1.0, f"{r.setting}: reduction_rate={r.reduction_rate}"

    @pytest.mark.slow
    def test_run_benchmarks_tg_equivalent_steps_gte_standard(self):
        results = run_benchmarks(cycles=3)
        std_steps = results[0].equivalent_steps
        for r in results[1:]:
            assert r.equivalent_steps >= std_steps, f"{r.setting}: {r.equivalent_steps} < {std_steps}"


class TestMainOutput:
    def test_main_generates_report_file(self, tmp_path):
        output_dir = tmp_path / "reports" / "benchmark"
        main(["--cycles", "2", "--output-dir", str(output_dir)])
        report_path = output_dir / "benchmark_report.md"
        assert report_path.exists()

        content = report_path.read_text()
        assert "| Setting |" in content
        assert "| Standard LoRA |" in content

    def test_main_returns_results(self, tmp_path):
        output_dir = tmp_path / "out"
        results = main(["--cycles", "2", "--output-dir", str(output_dir)])
        assert len(results) == 4
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_report_file_valid_markdown_table(self, tmp_path):
        output_dir = tmp_path / "out"
        main(["--cycles", "2", "--output-dir", str(output_dir)])
        content = (output_dir / "benchmark_report.md").read_text()
        lines = [line for line in content.strip().split("\n") if line.startswith("|")]
        assert len(lines) >= 3  # header + separator + at least 1 data row


class TestSimpleLoRAModel:
    def test_forward_shape(self):
        model = SimpleLoRAModel(num_layers=4, dim=4)
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 4)
