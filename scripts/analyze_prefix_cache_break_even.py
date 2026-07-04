#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a standalone CLI (``python scripts/analyze_prefix_cache_break_even.py``):
# a bare script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the
# repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_json, save_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze when prefix-cache cold-build cost amortizes against warm-run "
            "wall-clock savings. Accepts either a single-run summary.json from "
            "benchmark_prefix_cache.py or the aggregate_summary.json produced by "
            "run_paper_memory_suite.sh."
        )
    )
    parser.add_argument(
        "--paper-summary",
        required=True,
        help="Path to paper-memory summary.json or aggregate_summary.json",
    )
    parser.add_argument(
        "--precompute-summary",
        default="",
        help=(
            "Optional path to parallel_precompute_summary.json. When provided, its "
            "overall wall seconds are treated as the canonical cold-build cost."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path; defaults next to the paper summary",
    )
    # Decision gates — the single consumer that turns the otherwise display-only
    # break-even metrics (break_even_status / break_even_repeated_runs /
    # one_run_total_delta_seconds) into a pass/fail CI verdict.  Every gate is
    # opt-in: with none set the script keeps its pre-gate behavior (always exit 0,
    # metrics stay display-only and the output JSON is byte-identical).
    gates = parser.add_argument_group(
        "decision gates",
        description=(
            "Turn the break-even metrics into a pass/fail decision. When any "
            "enabled gate fails, each failure is printed to stderr and the "
            "process exits with status 1. Without a gate flag the metrics remain "
            "display-only (exit 0)."
        ),
    )
    gates.add_argument(
        "--require-warm-win",
        action="store_true",
        help="Fail unless warm TG beats warm baseline on wall-clock (a prerequisite "
        "for the cold build to amortize at all).",
    )
    gates.add_argument(
        "--max-break-even-runs",
        type=float,
        default=None,
        help="Fail unless the cold build amortizes within this many warm reuses "
        "(break_even_repeated_runs <= N). Boundary-inclusive.",
    )
    gates.add_argument(
        "--require-one-run-win",
        action="store_true",
        help="Fail unless a single run INCLUDING the cold build already beats the "
        "baseline (one_run_total_delta_seconds > 0).",
    )
    return parser.parse_args()


def _extract_from_single_run(summary: dict[str, Any]) -> dict[str, Any]:
    warm_baseline = summary["warm"]["baseline"]["wall_seconds"]
    warm_tg = summary["warm"]["tg_lora"]["wall_seconds"]
    cold_build = summary["cold"]["tg_lora"].get("prefix_feature_cache_total_build_seconds")
    return {
        "summary_type": "single_run",
        "warm_baseline_wall_seconds": warm_baseline,
        "warm_tg_wall_seconds": warm_tg,
        "cold_build_seconds": cold_build,
        "warm_baseline_gpu_peak_mb": summary["warm"]["baseline"].get("gpu_peak_mb"),
        "warm_tg_gpu_peak_mb": summary["warm"]["tg_lora"].get("gpu_peak_mb"),
    }


def _extract_from_aggregate(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    return {
        "summary_type": "aggregate",
        "warm_baseline_wall_seconds": aggregate["warm_baseline_wall_seconds"]["mean"],
        "warm_tg_wall_seconds": aggregate["warm_tg_wall_seconds"]["mean"],
        "cold_build_seconds": aggregate["tg_cache_build_seconds"]["mean"],
        "warm_baseline_gpu_peak_mb": aggregate["warm_baseline_gpu_peak_mb"]["mean"],
        "warm_tg_gpu_peak_mb": aggregate["warm_tg_gpu_peak_mb"]["mean"],
    }


def _load_paper_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    if not isinstance(summary, dict):
        raise ValueError("paper summary must resolve to a JSON object")
    if "cold" in summary and "warm" in summary:
        return _extract_from_single_run(summary)
    if "aggregate" in summary and "per_seed" in summary:
        return _extract_from_aggregate(summary)
    raise ValueError(
        "Unsupported paper summary format. Expected benchmark summary.json or aggregate_summary.json"
    )


def analyze_break_even(paper: dict[str, Any], precompute: dict[str, Any] | None) -> dict[str, Any]:
    warm_baseline = float(paper["warm_baseline_wall_seconds"])
    warm_tg = float(paper["warm_tg_wall_seconds"])
    warm_delta = warm_baseline - warm_tg

    cold_build_seconds = paper.get("cold_build_seconds")
    cold_build_source = "paper_summary"
    if precompute is not None:
        cold_build_seconds = precompute.get("overall_wall_seconds")
        cold_build_source = "parallel_precompute_summary"

    if cold_build_seconds is None:
        raise ValueError("cold_build_seconds is required to analyze break-even")
    cold_build_seconds = float(cold_build_seconds)

    break_even_runs = None
    break_even_status = "no_warm_win"
    if warm_delta > 0:
        break_even_runs = cold_build_seconds / warm_delta
        break_even_status = "warm_win"

    total_one_run_tg = cold_build_seconds + warm_tg
    total_one_run_delta = warm_baseline - total_one_run_tg

    return {
        "summary_type": paper["summary_type"],
        "cold_build_source": cold_build_source,
        "cold_build_seconds": cold_build_seconds,
        "warm_baseline_wall_seconds": warm_baseline,
        "warm_tg_wall_seconds": warm_tg,
        "warm_wall_delta_seconds": warm_delta,
        "warm_baseline_gpu_peak_mb": paper.get("warm_baseline_gpu_peak_mb"),
        "warm_tg_gpu_peak_mb": paper.get("warm_tg_gpu_peak_mb"),
        "one_run_total_tg_seconds_including_cold_build": total_one_run_tg,
        "one_run_total_delta_seconds": total_one_run_delta,
        "break_even_status": break_even_status,
        "break_even_repeated_runs": break_even_runs,
        "interpretation": (
            "Warm TG is already faster than baseline; cold-build amortizes after repeated reuse"
            if warm_delta > 0
            else "Warm TG does not yet beat baseline on wall-clock, so cold-build amortization alone is insufficient"
        ),
    }


def evaluate_gates(
    result: dict[str, Any],
    *,
    require_warm_win: bool,
    max_break_even_runs: float | None,
    require_one_run_win: bool,
) -> list[dict[str, str]]:
    """Turn the break-even metrics into a pass/fail decision.

    This is the single consumer that makes the previously display-only
    ``break_even_status`` / ``break_even_repeated_runs`` /
    ``one_run_total_delta_seconds`` metrics drive a CI exit code instead of
    merely being printed.  Returns one record per failed gate (empty list =
    every enabled gate passed); each record carries ``gate`` (the flag name)
    and ``message`` (human-readable detail for stderr).
    """
    failures: list[dict[str, str]] = []

    if require_warm_win and result["break_even_status"] != "warm_win":
        failures.append(
            {
                "gate": "--require-warm-win",
                "message": (
                    f"break_even_status={result['break_even_status']!r}; warm TG "
                    f"({result['warm_tg_wall_seconds']}s) does not beat warm baseline "
                    f"({result['warm_baseline_wall_seconds']}s), so the prefix cache "
                    "cannot amortize regardless of reuse count."
                ),
            }
        )

    if max_break_even_runs is not None:
        repeated = result["break_even_repeated_runs"]
        if repeated is None:
            failures.append(
                {
                    "gate": "--max-break-even-runs",
                    "message": (
                        "break_even_repeated_runs is None (no warm win); the cold "
                        f"build cannot amortize within {max_break_even_runs} runs."
                    ),
                }
            )
        elif repeated > max_break_even_runs:
            failures.append(
                {
                    "gate": "--max-break-even-runs",
                    "message": (
                        f"break_even_repeated_runs={repeated:.3f} exceeds budget "
                        f"{max_break_even_runs} "
                        f"(cold_build={result['cold_build_seconds']}s / "
                        f"warm_delta={result['warm_wall_delta_seconds']}s)."
                    ),
                }
            )

    if require_one_run_win and float(result["one_run_total_delta_seconds"]) <= 0:
        failures.append(
            {
                "gate": "--require-one-run-win",
                "message": (
                    f"one_run_total_delta_seconds={result['one_run_total_delta_seconds']:.3f} "
                    f"<= 0; a single run including the cold build "
                    f"({result['one_run_total_tg_seconds_including_cold_build']}s) does not "
                    f"come out ahead of the baseline ({result['warm_baseline_wall_seconds']}s)."
                ),
            }
        )

    return failures


def gate_eval_record(
    args: argparse.Namespace, failures: list[dict[str, str]]
) -> dict[str, Any] | None:
    """Build the ``gates`` record for the output JSON.

    Returns ``None`` when no gate was requested so the gate-free output stays
    byte-identical to the pre-gate behavior (existing JSON consumers are
    unaffected).  When any gate is requested the verdict is recorded so the
    deposit artifact carries the decision, not just the numbers.
    """
    if not any(
        (
            args.require_warm_win,
            args.max_break_even_runs is not None,
            args.require_one_run_win,
        )
    ):
        return None

    requested: list[str] = []
    if args.require_warm_win:
        requested.append("--require-warm-win")
    if args.max_break_even_runs is not None:
        requested.append(f"--max-break-even-runs={args.max_break_even_runs}")
    if args.require_one_run_win:
        requested.append("--require-one-run-win")

    return {
        "requested": requested,
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    args = _parse_args()
    paper_summary_path = Path(args.paper_summary)
    paper_summary = _load_paper_summary(paper_summary_path)

    precompute_summary = None
    if args.precompute_summary:
        precompute_summary = load_json(args.precompute_summary)
        if not isinstance(precompute_summary, dict):
            raise ValueError("precompute summary must resolve to a JSON object")

    result = analyze_break_even(paper_summary, precompute_summary)

    # Consume the metrics: evaluate any requested decision gates before writing
    # so the deposit artifact records the verdict, then reflect a gate failure
    # in the process exit code.
    failures = evaluate_gates(
        result,
        require_warm_win=args.require_warm_win,
        max_break_even_runs=args.max_break_even_runs,
        require_one_run_win=args.require_one_run_win,
    )
    gates_record = gate_eval_record(args, failures)
    if gates_record is not None:
        result["gates"] = gates_record

    output_path = (
        Path(args.output)
        if args.output
        else paper_summary_path.with_name(paper_summary_path.stem + "_break_even.json")
    )
    save_json(result, output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Break-even analysis written to {output_path}")

    if failures:
        for failure in failures:
            print(
                f"GATE FAILED [{failure['gate']}]: {failure['message']}",
                file=sys.stderr,
            )
        print(f"{len(failures)} gate(s) failed; exiting non-zero.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()