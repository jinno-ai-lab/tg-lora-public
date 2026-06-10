"""Smoke tests for scripts/benchmark_optimizer_lifecycle.py.

Verifies that the benchmark script's core functions produce valid output
when run with a tiny Linear model on CPU.
"""

import torch

from scripts.benchmark_optimizer_lifecycle import (
    _measure_cycle,
    _round_record,
    _run_warmup_cycle,
    _state_summary,
)
from src.training.optimizer_lifecycle import OptimizerLifecycleManager


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


LR = 1e-3
DEVICE = torch.device("cpu")


class TestBenchmarkSmoke:
    def test_benchmark_produces_both_policies(self):
        """Benchmark produces valid results for both recreate and reuse policies."""
        model = _TinyModel()
        lora_params = list(model.parameters())

        recreate_mgr = OptimizerLifecycleManager(
            model, lr=LR, policy="recreate_per_cycle"
        )
        _run_warmup_cycle(recreate_mgr, lora_params, LR, DEVICE)
        recreate = _measure_cycle(recreate_mgr, lora_params, LR, DEVICE)

        reuse_mgr = OptimizerLifecycleManager(
            model, lr=LR, policy="reuse_state_reset_experimental"
        )
        _run_warmup_cycle(reuse_mgr, lora_params, LR, DEVICE)
        reuse = _measure_cycle(reuse_mgr, lora_params, LR, DEVICE)

        # Both records must have the expected keys
        for label, record in [("recreate", recreate), ("reuse", reuse)]:
            assert "prepare_ms" in record, f"{label} missing prepare_ms"
            assert "step_ms" in record, f"{label} missing step_ms"
            assert isinstance(record["prepare_ms"], float)
            assert isinstance(record["step_ms"], float)

    def test_reuse_preserves_state_tensor_pointers(self):
        """reuse_state_reset_experimental preserves state tensor pointers across cycles."""
        model = _TinyModel()
        lora_params = list(model.parameters())

        reuse_mgr = OptimizerLifecycleManager(
            model, lr=LR, policy="reuse_state_reset_experimental"
        )
        warmup_state, _ = _run_warmup_cycle(reuse_mgr, lora_params, LR, DEVICE)

        pointers_before = {
            (id(param), key): value.data_ptr()
            for param, state in warmup_state.items()
            for key, value in state.items()
            if torch.is_tensor(value)
        }

        _measure_cycle(reuse_mgr, lora_params, LR, DEVICE)
        pointers_after = reuse_mgr.state_tensor_pointers()

        assert pointers_before == pointers_after, (
            "State tensor pointers changed between cycles"
        )

    def test_comparison_metrics_structure(self):
        """Comparison delta section has numeric speedup percentages."""
        model = _TinyModel()
        lora_params = list(model.parameters())

        recreate_mgr = OptimizerLifecycleManager(
            model, lr=LR, policy="recreate_per_cycle"
        )
        _run_warmup_cycle(recreate_mgr, lora_params, LR, DEVICE)
        recreate = _measure_cycle(recreate_mgr, lora_params, LR, DEVICE)

        reuse_mgr = OptimizerLifecycleManager(
            model, lr=LR, policy="reuse_state_reset_experimental"
        )
        _run_warmup_cycle(reuse_mgr, lora_params, LR, DEVICE)
        reuse = _measure_cycle(reuse_mgr, lora_params, LR, DEVICE)

        delta = {
            "prepare_ms": recreate["prepare_ms"] - reuse["prepare_ms"],
            "step_ms": recreate["step_ms"] - reuse["step_ms"],
            "step_speedup_pct": 0.0
            if recreate["step_ms"] == 0
            else (recreate["step_ms"] - reuse["step_ms"])
            / recreate["step_ms"]
            * 100.0,
        }
        rounded = _round_record(delta)

        assert isinstance(rounded["prepare_ms"], float)
        assert isinstance(rounded["step_ms"], float)
        assert isinstance(rounded["step_speedup_pct"], float)

    def test_state_summary_returns_expected_keys(self):
        """_state_summary produces expected structure from a real optimizer."""
        model = _TinyModel()
        mgr = OptimizerLifecycleManager(
            model, lr=LR, policy="reuse_state_reset_experimental"
        )
        optimizer = mgr.prepare_for_cycle(LR)
        # Materialize state by stepping
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()

        summary = _state_summary(optimizer)
        assert "state_param_count" in summary
        assert "state_total_mb" in summary
        assert summary["state_param_count"] > 0
