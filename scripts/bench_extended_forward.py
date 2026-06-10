#!/usr/bin/env python
"""Benchmark suffix-forward column extension for alpha-line experiments.

This script intentionally does not import or call the training loop.  It loads
the same 4-bit Qwen/PEFT stack, fabricates cached suffix hidden states, and
times only the suffix decoder layers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf

from src.model.load_model import apply_lora, load_base_model
from src.tg_lora.activation_cache import (
    _get_decoder_layers,
    _get_layer_types,
    _get_model_config,
    _get_rotary_emb,
    forward_suffix_hidden_states,
)
from src.utils.device import resolve_compute_dtype
from src.utils.seed import set_seed


@dataclass
class TimingStats:
    wall_p10_ms: float
    wall_p50_ms: float
    wall_p90_ms: float
    gpu_p10_ms: float
    gpu_p50_ms: float
    gpu_p90_ms: float
    allocated_before_mb: float
    peak_allocated_mb: float
    peak_delta_mb: float


@dataclass
class ConditionResult:
    name: str
    timing: TimingStats
    effective_tokens: int
    estimated_tflops: float | None
    estimated_weight_bandwidth_gbps: float | None


@dataclass
class ModeResult:
    mode: str
    split_layer: int
    num_suffix_layers: int
    concat_axis: str
    one_x: ConditionResult
    two_x_concat: ConditionResult
    two_independent: ConditionResult
    ratio_2x_over_1x: float
    ratio_2x_over_2x1x: float
    decision: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark whether a 2x extended suffix forward is close to 1F "
            "or behaves like two independent suffix forwards."
        )
    )
    parser.add_argument("--config", default="configs/9b_tg_lora.yaml")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--suffix-fraction", type=float, default=0.25)
    parser.add_argument(
        "--modes",
        default="one,all",
        help="Comma-separated layer modes: one,all",
    )
    parser.add_argument(
        "--concat-axis",
        choices=("batch", "sequence"),
        default="batch",
        help=(
            "Axis used for 2x extension. batch keeps two streams independent "
            "without a block-diagonal attention mask; sequence is timing-only "
            "for Qwen3.5 linear-attention layers."
        ),
    )
    parser.add_argument(
        "--stop-after-one-if-no-go",
        action="store_true",
        help="Skip all-suffix measurement when one-layer r >= 1.7.",
    )
    return parser.parse_args()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _get_hidden_size(model: torch.nn.Module) -> int:
    config = _get_model_config(model)
    hidden_size = getattr(config, "hidden_size", None) if config is not None else None
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    decoder_layers = _get_decoder_layers(model)
    first_layer = decoder_layers[0]
    for module in first_layer.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 1:
            return int(weight.numel())
    raise RuntimeError("Could not infer hidden size")


def _condition_inputs(
    hidden: torch.Tensor,
    tangent: torch.Tensor,
    *,
    concat_axis: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if concat_axis == "batch":
        hidden_2x = torch.cat([hidden, tangent], dim=0)
        mask_1x = torch.ones(
            hidden.shape[0],
            hidden.shape[1],
            device=hidden.device,
            dtype=torch.long,
        )
        mask_2x = torch.ones(
            hidden_2x.shape[0],
            hidden_2x.shape[1],
            device=hidden.device,
            dtype=torch.long,
        )
        return mask_1x, mask_2x, hidden_2x.shape[0] * hidden_2x.shape[1]
    hidden_2x = torch.cat([hidden, tangent], dim=1)
    mask_1x = torch.ones(
        hidden.shape[0],
        hidden.shape[1],
        device=hidden.device,
        dtype=torch.long,
    )
    mask_2x = torch.ones(
        hidden_2x.shape[0],
        hidden_2x.shape[1],
        device=hidden.device,
        dtype=torch.long,
    )
    return mask_1x, mask_2x, hidden_2x.shape[0] * hidden_2x.shape[1]


def _make_forward(
    model: torch.nn.Module,
    *,
    split_layer: int,
    decoder_layers: torch.nn.ModuleList,
    rotary_emb: torch.nn.Module | None,
    layer_types: list[str] | None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def _forward(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return forward_suffix_hidden_states(
            model,
            hidden,
            attention_mask,
            split_layer_idx=split_layer,
            device=hidden.device,
            decoder_layers=decoder_layers,
            rotary_emb=rotary_emb,
            layer_types=layer_types,
        )

    return _forward


def _estimate_linear_flops_and_weight_bytes(
    modules: list[torch.nn.Module],
    *,
    tokens: int,
) -> tuple[float, float]:
    flops = 0.0
    weight_bytes = 0.0
    seen_weights: set[int] = set()
    for root in modules:
        for module in root.modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            weight_id = id(weight)
            if weight_id in seen_weights:
                continue
            seen_weights.add(weight_id)
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            flops += 2.0 * tokens * in_features * out_features
            if weight.__class__.__name__ == "Params4bit":
                weight_bytes += weight.numel() * 0.5
            else:
                weight_bytes += weight.numel() * weight.element_size()
    return flops, weight_bytes


def _measure_condition(
    name: str,
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iters: int,
    effective_tokens: int,
    estimated_flops: float,
    estimated_weight_bytes: float,
    device: torch.device,
) -> ConditionResult:
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA timing")

    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        wall_end = time.perf_counter()
        wall_ms.append((wall_end - wall_start) * 1000.0)
        gpu_ms.append(float(start.elapsed_time(end)))

    peak_allocated = torch.cuda.max_memory_allocated(device)
    timing = TimingStats(
        wall_p10_ms=_percentile(wall_ms, 0.10),
        wall_p50_ms=statistics.median(wall_ms),
        wall_p90_ms=_percentile(wall_ms, 0.90),
        gpu_p10_ms=_percentile(gpu_ms, 0.10),
        gpu_p50_ms=statistics.median(gpu_ms),
        gpu_p90_ms=_percentile(gpu_ms, 0.90),
        allocated_before_mb=allocated_before / 1024**2,
        peak_allocated_mb=peak_allocated / 1024**2,
        peak_delta_mb=(peak_allocated - allocated_before) / 1024**2,
    )

    gpu_seconds = timing.gpu_p50_ms / 1000.0
    estimated_tflops = None
    estimated_bandwidth = None
    if gpu_seconds > 0:
        estimated_tflops = estimated_flops / gpu_seconds / 1e12
        estimated_bandwidth = estimated_weight_bytes / gpu_seconds / 1e9

    return ConditionResult(
        name=name,
        timing=timing,
        effective_tokens=effective_tokens,
        estimated_tflops=estimated_tflops,
        estimated_weight_bandwidth_gbps=estimated_bandwidth,
    )


def _decision(ratio: float) -> str:
    if ratio <= 1.3:
        return "go: memory-bound leaning (r <= 1.3)"
    if ratio >= 1.7:
        return "no-go: compute-bound leaning (r >= 1.7)"
    return "borderline: needs additional B/T sweep"


def _benchmark_mode(
    model: torch.nn.Module,
    *,
    mode: str,
    split_layer: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    concat_axis: str,
    warmup: int,
    iters: int,
    seed: int,
) -> ModeResult:
    decoder_layers = _get_decoder_layers(model)
    rotary_emb = _get_rotary_emb(model)
    layer_types = _get_layer_types(model)
    device = next(decoder_layers[split_layer].parameters()).device
    hidden_size = _get_hidden_size(model)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    tangent = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    mask_1x, mask_2x, tokens_2x = _condition_inputs(
        hidden,
        tangent,
        concat_axis=concat_axis,
    )
    if concat_axis == "batch":
        hidden_2x = torch.cat([hidden, tangent], dim=0)
    else:
        hidden_2x = torch.cat([hidden, tangent], dim=1)

    suffix_modules = list(decoder_layers[split_layer:])
    flops_1x, weight_bytes = _estimate_linear_flops_and_weight_bytes(
        suffix_modules,
        tokens=batch_size * seq_len,
    )
    flops_2x, _ = _estimate_linear_flops_and_weight_bytes(
        suffix_modules,
        tokens=tokens_2x,
    )
    forward = _make_forward(
        model,
        split_layer=split_layer,
        decoder_layers=decoder_layers,
        rotary_emb=rotary_emb,
        layer_types=layer_types,
    )

    with torch.inference_mode():
        one_x = _measure_condition(
            "1x suffix forward",
            lambda: forward(hidden, mask_1x),
            warmup=warmup,
            iters=iters,
            effective_tokens=batch_size * seq_len,
            estimated_flops=flops_1x,
            estimated_weight_bytes=weight_bytes,
            device=device,
        )
        two_x = _measure_condition(
            "2x extended forward",
            lambda: forward(hidden_2x, mask_2x),
            warmup=warmup,
            iters=iters,
            effective_tokens=tokens_2x,
            estimated_flops=flops_2x,
            estimated_weight_bytes=weight_bytes,
            device=device,
        )
        two_independent = _measure_condition(
            "two independent 1x forwards",
            lambda: (forward(hidden, mask_1x), forward(tangent, mask_1x)),
            warmup=warmup,
            iters=iters,
            effective_tokens=2 * batch_size * seq_len,
            estimated_flops=2.0 * flops_1x,
            estimated_weight_bytes=2.0 * weight_bytes,
            device=device,
        )

    ratio_2x = two_x.timing.wall_p50_ms / one_x.timing.wall_p50_ms
    ratio_vs_two = two_x.timing.wall_p50_ms / two_independent.timing.wall_p50_ms
    return ModeResult(
        mode=mode,
        split_layer=split_layer,
        num_suffix_layers=len(decoder_layers) - split_layer,
        concat_axis=concat_axis,
        one_x=one_x,
        two_x_concat=two_x,
        two_independent=two_independent,
        ratio_2x_over_1x=ratio_2x,
        ratio_2x_over_2x1x=ratio_vs_two,
        decision=_decision(ratio_2x),
    )


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    metadata: dict[str, object],
    results: list[ModeResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "args": vars(args),
        "metadata": metadata,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "bench_summary.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Extended Suffix Forward Benchmark",
        "",
        "## Metadata",
        "",
        f"- GPU: {metadata.get('gpu_name')}",
        f"- CUDA device: {metadata.get('device')}",
        f"- model: {metadata.get('model_name')}",
        f"- dtype: {metadata.get('dtype')}",
        f"- batch size: {args.batch_size}",
        f"- seq len: {args.seq_len}",
        f"- concat axis: {args.concat_axis}",
        f"- warmup / iters: {args.warmup} / {args.iters}",
        "- estimates: TFLOPS and weight bandwidth are heuristic, linear-module-only estimates",
        "",
        "## Results",
        "",
        "| mode | suffix layers | 1x wall p50 ms | 2x wall p50 ms | 2x1x wall p50 ms | r=2x/1x | 2x/(2x1x) | peak VRAM 2x MB | est TFLOPS 1x | est weight GB/s 1x | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.mode} | "
            f"{result.num_suffix_layers} | "
            f"{_format_float(result.one_x.timing.wall_p50_ms)} | "
            f"{_format_float(result.two_x_concat.timing.wall_p50_ms)} | "
            f"{_format_float(result.two_independent.timing.wall_p50_ms)} | "
            f"{_format_float(result.ratio_2x_over_1x)} | "
            f"{_format_float(result.ratio_2x_over_2x1x)} | "
            f"{_format_float(result.two_x_concat.timing.peak_allocated_mb, 1)} | "
            f"{_format_float(result.one_x.estimated_tflops)} | "
            f"{_format_float(result.one_x.estimated_weight_bandwidth_gbps)} | "
            f"{result.decision} |"
        )

    lines.extend(
        [
            "",
            "## Per-Condition Timing",
            "",
            "| mode | condition | wall p10/p50/p90 ms | gpu p10/p50/p90 ms | peak delta MB |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        for condition in (
            result.one_x,
            result.two_x_concat,
            result.two_independent,
        ):
            timing = condition.timing
            lines.append(
                "| "
                f"{result.mode} | "
                f"{condition.name} | "
                f"{_format_float(timing.wall_p10_ms)}/"
                f"{_format_float(timing.wall_p50_ms)}/"
                f"{_format_float(timing.wall_p90_ms)} | "
                f"{_format_float(timing.gpu_p10_ms)}/"
                f"{_format_float(timing.gpu_p50_ms)}/"
                f"{_format_float(timing.gpu_p90_ms)} | "
                f"{_format_float(timing.peak_delta_mb, 1)} |"
            )

    (output_dir / "bench_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(repo_root / args.config)
    if args.batch_size is None:
        args.batch_size = int(cfg.get("alpha_line", {}).get("b_light", 16))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root
        / f"runs/bench_extended_forward_{datetime.now():%Y%m%d_%H%M%S}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = apply_lora(load_base_model(cfg), cfg)
    model.eval()
    decoder_layers = _get_decoder_layers(model)
    num_layers = len(decoder_layers)
    suffix_layers = max(1, math.ceil(num_layers * args.suffix_fraction))
    all_split = num_layers - suffix_layers
    mode_to_split = {
        "one": num_layers - 1,
        "all": all_split,
    }
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in mode_to_split]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")

    device = next(model.parameters()).device
    dtype = resolve_compute_dtype(device, cfg.model.get("dtype", "bf16"))
    metadata: dict[str, object] = {
        "gpu_name": torch.cuda.get_device_name(device),
        "device": str(device),
        "model_name": cfg.model.name_or_path,
        "dtype": str(dtype),
        "num_layers": num_layers,
        "suffix_fraction": args.suffix_fraction,
        "layer_types": _get_layer_types(model),
    }

    results: list[ModeResult] = []
    for mode in modes:
        result = _benchmark_mode(
            model,
            mode=mode,
            split_layer=mode_to_split[mode],
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dtype=dtype,
            concat_axis=args.concat_axis,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed + len(results),
        )
        results.append(result)
        _write_summary(output_dir, args=args, metadata=metadata, results=results)
        if (
            mode == "one"
            and args.stop_after_one_if_no_go
            and result.ratio_2x_over_1x >= 1.7
        ):
            break

    _write_summary(output_dir, args=args, metadata=metadata, results=results)
    print(output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
