#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Allow running as a standalone CLI (``python scripts/benchmark_optimizer_lifecycle.py``): a
# bare script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the
# repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.load_model import apply_lora, load_base_model
from src.model.lora_utils import iter_lora_params
from src.training.optimizer_lifecycle import OptimizerLifecycleManager
from src.utils.device import (
    gpu_device_name,
    gpu_empty_cache,
    gpu_memory_allocated_mb,
    gpu_peak_memory_mb,
    gpu_memory_reserved_mb,
    gpu_reset_peak_stats,
    gpu_synchronize,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TG-LoRA optimizer lifecycle policies on the current model/config."
    )
    parser.add_argument(
        "--config",
        default="configs/9b_tg_lora.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def _mem(device: torch.device) -> dict[str, float]:
    alloc = gpu_memory_allocated_mb(device)
    reserved = gpu_memory_reserved_mb(device)
    peak = gpu_peak_memory_mb(device)
    result: dict[str, float] = {}
    if alloc is not None:
        result["allocated_mb"] = alloc
    if reserved is not None:
        result["reserved_mb"] = reserved
    if peak is not None:
        result["peak_mb"] = peak
    return result


def _sync(device: torch.device) -> None:
    gpu_synchronize(device)


def _round_record(record: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, float):
            out[key] = round(value, 3)
        elif isinstance(value, dict):
            out[key] = _round_record(value)
        else:
            out[key] = value
    return out


def _materialize_zero_grads(lora_params: list[torch.nn.Parameter]) -> None:
    for param in lora_params:
        param.grad = torch.zeros_like(param)


def _state_summary(optimizer) -> dict:
    total_bytes = 0
    state_dtypes: dict[str, str] = {}
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                total_bytes += value.numel() * value.element_size()
                state_dtypes.setdefault(key, str(value.dtype))
    return {
        "state_param_count": len(optimizer.state),
        "state_total_mb": total_bytes / 1024**2,
        "state_dtypes": state_dtypes,
    }


def _run_warmup_cycle(manager, lora_params, lr: float, device: torch.device) -> tuple[dict, dict]:
    optimizer = manager.prepare_for_cycle(lr)
    _materialize_zero_grads(lora_params)
    _sync(device)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer.state, _state_summary(optimizer)


def _measure_cycle(manager, lora_params, lr: float, device: torch.device) -> dict:
    if device.type == "cuda":
        gpu_reset_peak_stats(device)
    before = _mem(device)

    _sync(device)
    t0 = time.perf_counter()
    optimizer = manager.prepare_for_cycle(lr)
    _sync(device)
    prepare_ms = (time.perf_counter() - t0) * 1000
    after_prepare = _mem(device)

    _sync(device)
    t1 = time.perf_counter()
    _materialize_zero_grads(lora_params)
    _sync(device)
    grad_ms = (time.perf_counter() - t1) * 1000
    after_grad = _mem(device)

    _sync(device)
    t2 = time.perf_counter()
    optimizer.step()
    _sync(device)
    step_ms = (time.perf_counter() - t2) * 1000
    optimizer.zero_grad(set_to_none=True)
    after_step = _mem(device)

    result = {
        "prepare_ms": prepare_ms,
        "grad_materialization_ms": grad_ms,
        "step_ms": step_ms,
        "memory_before": before,
        "memory_after_prepare": after_prepare,
        "memory_after_grad": after_grad,
        "memory_after_step": after_step,
        "allocated_growth_mb": after_step.get("allocated_mb", 0.0)
        - before.get("allocated_mb", 0.0),
        "reserved_growth_mb": after_step.get("reserved_mb", 0.0)
        - before.get("reserved_mb", 0.0),
    }
    result.update(_state_summary(optimizer))
    return result


def main() -> None:
    args = _parse_args()
    from src.training.config_schema import load_and_validate_config

    load_and_validate_config(args.config)

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg).__name__}")

    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    device = next(model.parameters()).device
    lora_params = [param for _, param in iter_lora_params(model)]
    trainable_params = sum(param.numel() for param in lora_params)

    if device.type == "cuda":
        gc.collect()
        gpu_empty_cache(device)

    recreate_mgr = OptimizerLifecycleManager(
        model,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        policy="recreate_per_cycle",
    )
    _run_warmup_cycle(recreate_mgr, lora_params, cfg.training.learning_rate, device)
    gc.collect()
    recreate = _measure_cycle(recreate_mgr, lora_params, cfg.training.learning_rate, device)

    reuse_mgr = OptimizerLifecycleManager(
        model,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        policy="reuse_state_reset_experimental",
    )
    warmup_state, _ = _run_warmup_cycle(reuse_mgr, lora_params, cfg.training.learning_rate, device)
    pointers_before = {
        (id(param), key): value.data_ptr()
        for param, state in warmup_state.items()
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    reuse = _measure_cycle(reuse_mgr, lora_params, cfg.training.learning_rate, device)
    pointers_after = reuse_mgr.state_tensor_pointers()

    comparison = {
        "config_path": args.config,
        "device": str(device),
        "gpu_name": gpu_device_name(device),
        "trainable_lora_params": trainable_params,
        "lora_param_dtype": str(lora_params[0].dtype) if lora_params else "n/a",
        "recreate_per_cycle": recreate,
        "reuse_state_reset_experimental": reuse,
        "reuse_state_ptrs_preserved": pointers_before == pointers_after,
        "delta": {
            "prepare_ms": recreate["prepare_ms"] - reuse["prepare_ms"],
            "grad_materialization_ms": recreate["grad_materialization_ms"]
            - reuse["grad_materialization_ms"],
            "step_ms": recreate["step_ms"] - reuse["step_ms"],
            "allocated_growth_mb": recreate["allocated_growth_mb"]
            - reuse["allocated_growth_mb"],
            "reserved_growth_mb": recreate["reserved_growth_mb"]
            - reuse["reserved_growth_mb"],
            "step_speedup_pct": 0.0
            if recreate["step_ms"] == 0
            else (recreate["step_ms"] - reuse["step_ms"])
            / recreate["step_ms"]
            * 100.0,
        },
    }
    print(json.dumps(_round_record(comparison), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()