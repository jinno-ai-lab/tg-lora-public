#!/usr/bin/env python
"""Micro-benchmark for velocity EMA update and cap_update in-place operations.

Measures execution time and memory usage over 1000 iterations, outputs JSON.
Follows the benchmark_optimizer_lifecycle.py pattern.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

# Allow running as a standalone CLI (``python scripts/benchmark_velocity_ops.py``): a
# bare script invocation puts ``scripts/`` — not the repo root — on sys.path, so make
# the repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tg_lora.extrapolator import cap_update
from src.tg_lora.velocity import Velocity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark for velocity EMA update and cap_update in-place ops."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations per benchmark (default: 1000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 50 iterations for smoke testing (reduced variance vs old 10)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline JSON file. Exits 1 if any per-iter metric regresses beyond --threshold.",
    )
    parser.add_argument(
        "--save-baseline",
        type=str,
        default=None,
        dest="save_baseline",
        help="Save current results as a baseline JSON file for future comparisons.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Regression threshold %% (default: 20). Exit 1 if current > baseline * (1 + threshold/100).",
    )
    parser.add_argument(
        "--max-cap-overhead-ratio",
        type=float,
        default=None,
        dest="max_cap_overhead_ratio",
        help=(
            "Portable CI gate: exit 1 if cap_update_overhead_ratio "
            "(capped per-iter ms / no-cap per-iter ms, measured in the same run) "
            "exceeds this ceiling. Hardware-normalized — both terms scale together "
            "with host speed, so this catches algorithmic regressions (a redundant "
            "clone/norm or an out-of-place op blowing up the capped path) without "
            "the absolute-time non-portability of --baseline. Default: off."
        ),
    )
    return parser.parse_args()


def _mem_usage_kb() -> float:
    """Return RSS in KB via /proc/self/status if available, else 0."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0.0


def _round_record(record: dict) -> dict:
    out: dict = {}
    for key, value in record.items():
        if isinstance(value, float):
            out[key] = round(value, 3)
        elif isinstance(value, dict):
            out[key] = _round_record(value)
        else:
            out[key] = value
    return out


def benchmark_velocity_ema(iterations: int) -> dict:
    """Benchmark Velocity.update (EMA in-place) over *iterations* calls."""
    vel = Velocity(max_history=iterations)
    keys = [f"layer.{i}.weight" for i in range(4)]
    shape = (64, 64)
    beta = 0.9

    deltas = [
        {k: torch.randn(*shape) for k in keys}
        for _ in range(min(iterations, 128))
    ]

    gc.collect()
    mem_before = _mem_usage_kb()
    t0 = time.perf_counter()

    for i in range(iterations):
        vel.update(deltas[i % len(deltas)], beta)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    mem_after = _mem_usage_kb()

    return {
        "velocity_ema_time_ms": elapsed_ms,
        "velocity_ema_per_iter_ms": elapsed_ms / iterations,
        "velocity_ema_mem_delta_kb": mem_after - mem_before,
        "velocity_ema_iterations": iterations,
    }


def benchmark_cap_update(iterations: int) -> dict:
    """Benchmark cap_update (in-place mul_) over *iterations* calls."""
    shape = (1024, 1024)
    update = torch.randn(*shape)
    ref = torch.randn(*shape)
    max_ratio = 0.01

    gc.collect()
    mem_before = _mem_usage_kb()
    t0 = time.perf_counter()

    for _ in range(iterations):
        # Work on a clone so the benchmark loop is repeatable
        u = update.clone()
        cap_update(u, ref, max_ratio=max_ratio)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    mem_after = _mem_usage_kb()

    # Also measure no-capping path (update already small)
    small_update = torch.randn(*shape) * 1e-8
    t1 = time.perf_counter()
    for _ in range(iterations):
        u = small_update.clone()
        cap_update(u, ref, max_ratio=max_ratio)
    nocap_ms = (time.perf_counter() - t1) * 1000

    # Hardware-normalized overhead of the capping step. Both the capped and
    # no-cap loops clone the same (1024, 1024) tensor and run the two norm
    # reductions; the capped path adds exactly one in-place ``mul_``. Their ratio
    # therefore isolates the marginal cost of the cap arithmetic with the host's
    # raw speed cancelling out — the only quantity in this benchmark that is
    # portable across machines/CI runners (absolute per-iter ms is NOT: see
    # ``--max-cap-overhead-ratio`` vs the local-only ``--baseline``).
    overhead_ratio = elapsed_ms / nocap_ms if nocap_ms > 0 else None

    return {
        "cap_update_time_ms": elapsed_ms,
        "cap_update_per_iter_ms": elapsed_ms / iterations,
        "cap_update_mem_delta_kb": mem_after - mem_before,
        "cap_update_nocap_time_ms": nocap_ms,
        "cap_update_nocap_per_iter_ms": nocap_ms / iterations,
        "cap_update_overhead_ratio": overhead_ratio,
        "cap_update_iterations": iterations,
        "cap_update_tensor_shape": list(shape),
    }


_METRIC_PATHS: list[tuple[str, str]] = [
    ("velocity_ema", "velocity_ema_per_iter_ms"),
    ("cap_update", "cap_update_per_iter_ms"),
    ("cap_update", "cap_update_nocap_per_iter_ms"),
]


def _compare_with_baseline(
    current: dict,
    baseline: dict,
    threshold_pct: float,
) -> list[dict]:
    """Compare *current* results against *baseline* for per-iter metrics.

    Returns a list of regression dicts (empty if none detected).
    """
    regressions: list[dict] = []
    for section, metric in _METRIC_PATHS:
        cur_val = current.get(section, {}).get(metric)
        base_val = baseline.get(section, {}).get(metric)
        if cur_val is None or base_val is None or base_val <= 0:
            continue
        ratio = cur_val / base_val
        limit = 1.0 + threshold_pct / 100.0
        if ratio > limit:
            regressions.append({
                "metric": metric,
                "section": section,
                "baseline_ms": base_val,
                "current_ms": cur_val,
                "ratio": round(ratio, 3),
                "threshold": round(limit, 3),
            })
    return regressions


def main() -> None:
    args = _parse_args()
    iterations = 50 if args.quick else args.iterations

    results = {
        "iterations": iterations,
        "velocity_ema": benchmark_velocity_ema(iterations),
        "cap_update": benchmark_cap_update(iterations),
    }

    if args.save_baseline:
        with open(args.save_baseline, "w") as f:
            json.dump(_round_record(results), f, ensure_ascii=False, indent=2)
            f.write("\n")

    output = _round_record(results)

    if args.baseline:
        try:
            with open(args.baseline) as f:
                baseline = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"ERROR: cannot load baseline: {exc}", file=sys.stderr)
            sys.exit(2)

        regressions = _compare_with_baseline(results, baseline, args.threshold)
        output["baseline_comparison"] = {
            "baseline_file": args.baseline,
            "threshold_pct": args.threshold,
            "regressions": regressions,
            "regressed": len(regressions) > 0,
        }

    # Portable CI gate: the hardware-normalized cap overhead ratio. Computed
    # from the RAW (pre-rounding) per-iter values so the verdict is exact.
    portable_failure = None
    if args.max_cap_overhead_ratio is not None:
        cap = results["cap_update"]
        cap_per = cap["cap_update_per_iter_ms"]
        nocap_per = cap["cap_update_nocap_per_iter_ms"]
        ratio = cap_per / nocap_per if nocap_per > 0 else float("inf")
        ceiling = args.max_cap_overhead_ratio
        passed = ratio <= ceiling
        output["portable_gate"] = {
            "metric": "cap_update_overhead_ratio",
            "ratio": round(ratio, 3),
            "ceiling": ceiling,
            "passed": passed,
        }
        if not passed:
            portable_failure = (ratio, ceiling)

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.baseline and regressions:
        print("\nREGRESSION DETECTED — the following metrics exceeded threshold:",
              file=sys.stderr)
        for r in regressions:
            print(
                f"  {r['metric']}: {r['current_ms']:.4f}ms "
                f"(baseline {r['baseline_ms']:.4f}ms, "
                f"{r['ratio']:.1f}x > {r['threshold']:.1f}x)",
                file=sys.stderr,
            )
        print(
            "  NOTE: --baseline compares ABSOLUTE per-iter ms against a "
            "checked-in baseline and is only portable to the host that recorded "
            "it; use --max-cap-overhead-ratio for the hardware-normalized CI gate.",
            file=sys.stderr,
        )
        sys.exit(1)

    if portable_failure is not None:
        ratio, ceiling = portable_failure
        print(
            "\nPORTABLE GATE FAILED — cap_update overhead ratio exceeded ceiling:",
            file=sys.stderr,
        )
        print(
            f"  cap_update_overhead_ratio: {ratio:.3f} > ceiling {ceiling:.3f} "
            f"(capped per-iter ms / no-cap per-iter ms, same run)",
            file=sys.stderr,
        )
        print(
            "  The capped path adds one in-place mul_ over the shared clone+norm "
            "cost, so this ratio is normally ~1.0-1.3 and hardware-independent; "
            "a value far above ceiling signals an algorithmic regression such as "
            "a redundant clone/norm or an out-of-place op in cap_update.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
