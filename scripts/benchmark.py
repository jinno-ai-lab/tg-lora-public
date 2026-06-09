#!/usr/bin/env python
"""Performance benchmark comparing Standard LoRA vs TG-LoRA training.

Runs standard LoRA (K real steps only) and TG-LoRA (K + N extrapolation) with
identical model / data and reports wall-clock time, reduction_rate, final loss,
acceptance_rate, and equivalent steps.

Usage:
    python scripts/benchmark.py [--cycles 10] [--output-dir reports/benchmark]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.simple_model import SimpleLoRAModel
from tg_lora import (
    CycleState,
    DeltaTracker,
    RandomWalkController,
    RollbackManager,
    Velocity,
    apply_extrapolation,
    select_active_layers,
    snapshot_lora,
)


def _compute_loss(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).sum()


@torch.no_grad()
def _evaluate(model: torch.nn.Module, x: torch.Tensor) -> float:
    return _compute_loss(model, x).item()


def _make_model() -> SimpleLoRAModel:
    return SimpleLoRAModel(num_layers=4, dim=4)


def _make_data(batch_size: int = 4) -> torch.Tensor:
    return torch.randn(batch_size, 4)


@dataclass
class BenchmarkResult:
    setting: str
    wall_clock: float
    equivalent_steps: int
    reduction_rate: float
    final_loss: float
    acceptance_rate: float
    full_backward_passes: int


def _run_standard_lora(cycles: int, lr: float = 1e-3, K: int = 3) -> BenchmarkResult:
    model = _make_model()
    x = _make_data()
    cycle_state = CycleState()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )

    t0 = time.perf_counter()
    for _ in range(cycles):
        for _ in range(K):
            loss = _compute_loss(model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_loss = _evaluate(model, x)

        cycle_state.record_cycle(
            train_loss=train_loss,
            valid_loss=train_loss,
            accepted=True,
            K=K,
            N=0,
            grad_accum=1,
        )
    elapsed = time.perf_counter() - t0

    return BenchmarkResult(
        setting="Standard LoRA",
        wall_clock=elapsed,
        equivalent_steps=cycle_state.full_backward_passes,
        reduction_rate=cycle_state.reduction_rate,
        final_loss=cycle_state.last_train_loss,
        acceptance_rate=1.0,
        full_backward_passes=cycle_state.full_backward_passes,
    )


def _run_tg_lora(cycles: int, K: int, N: int, lr: float = 1e-3) -> BenchmarkResult:
    model = _make_model()
    x = _make_data()
    controller = RandomWalkController(
        K_initial=K,
        N_initial=N,
        alpha_initial=0.3,
        beta_initial=0.8,
        lr_initial=lr,
        active_layer_strategy="last_25_percent",
        relative_update_cap=0.005,
        rollback_tolerance=0.0,
        enable_random_walk=False,
    )
    velocity = Velocity()
    rollback = RollbackManager()
    delta_tracker = DeltaTracker()
    cycle_state = CycleState()

    t0 = time.perf_counter()
    for _ in range(cycles):
        proposal = controller.propose()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=proposal.lr,
        )

        W0 = snapshot_lora(model)
        for _ in range(proposal.K):
            loss = _compute_loss(model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        WK = snapshot_lora(model)
        dW = delta_tracker.compute_and_record(WK, W0, K=proposal.K)
        velocity.update(dW, beta=proposal.beta)

        loss_pilot = _evaluate(model, x)

        rollback.save(model)
        active_names, _ = select_active_layers(model, strategy=proposal.active_layer_strategy)
        apply_extrapolation(
            model=model,
            velocity=velocity.state,
            active_names=active_names,
            alpha_by_name={},
            default_alpha=proposal.alpha,
            n_steps=proposal.N,
            relative_update_cap=proposal.relative_update_cap,
        )

        loss_after = _evaluate(model, x)

        accepted = controller.accept(loss_pilot, loss_after)
        if accepted:
            controller.reward(loss_pilot, loss_after)
            if rollback._history:
                rollback.pop()
        else:
            rollback.rollback(model)
            controller.penalize(loss_pilot, loss_after)
            if rollback._history:
                rollback.pop()

        cycle_state.record_cycle(
            train_loss=loss_pilot,
            valid_loss=loss_after if accepted else loss_pilot,
            accepted=accepted,
            K=proposal.K,
            N=proposal.N if accepted else 0,
            grad_accum=1,
        )
    elapsed = time.perf_counter() - t0

    equivalent = cycle_state.full_backward_passes + cycle_state.speculative_equivalent_backward_passes
    return BenchmarkResult(
        setting=f"TG-LoRA (K={K},N={N})",
        wall_clock=elapsed,
        equivalent_steps=equivalent,
        reduction_rate=cycle_state.reduction_rate,
        final_loss=cycle_state.last_train_loss,
        acceptance_rate=cycle_state.acceptance_rate,
        full_backward_passes=cycle_state.full_backward_passes,
    )


TG_LORA_CONFIGS = [
    (2, 1),
    (3, 2),
    (3, 5),
]


def run_benchmarks(cycles: int = 10) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    std = _run_standard_lora(cycles=cycles)
    results.append(std)

    for K, N in TG_LORA_CONFIGS:
        tg = _run_tg_lora(cycles=cycles, K=K, N=N)
        results.append(tg)

    return results


def format_table(results: list[BenchmarkResult]) -> str:
    header = (
        "| Setting | Wall-clock (s) | Equivalent Steps | reduction_rate | Final Loss | acceptance_rate |\n"
        "|---------|---------------|------------------|---------------|-----------|-----------------|"
    )
    rows: list[str] = []
    for r in results:
        if r.setting == "Standard LoRA":
            acc_str = "-"
        else:
            acc_str = f"{r.acceptance_rate:.0%}"
        rows.append(
            f"| {r.setting} | {r.wall_clock:.4f} | {r.equivalent_steps} | "
            f"{r.reduction_rate:.0%} | {r.final_loss:.6f} | {acc_str} |"
        )
    return header + "\n" + "\n".join(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TG-LoRA performance benchmark")
    parser.add_argument("--cycles", type=int, default=10, help="Number of training cycles per setting")
    parser.add_argument("--output-dir", type=str, default="reports/benchmark", help="Directory for report output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[BenchmarkResult]:
    args = parse_args(argv)
    results = run_benchmarks(cycles=args.cycles)

    table = format_table(results)
    print(table)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.md"
    report_path.write_text(table + "\n")

    return results


if __name__ == "__main__":
    main()
