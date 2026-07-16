#!/usr/bin/env python
"""Run external quality evaluation (G3 Gate) on best models from paper-memory suite.

Reads aggregate_summary.json to identify the best TG and baseline model seeds,
then runs lm-evaluation-harness on both and produces external_eval_results.json
with G3 gate pass/fail determination.

Usage::

    # Full evaluation (requires GPU + lm-eval)
    python scripts/run_paper_external_eval.py runs/.../aggregate_summary.json

    # Analysis mode: evaluate G3 on pre-existing results
    python scripts/run_paper_external_eval.py --analysis-mode \\
        --external-eval runs/.../external_eval_results.json \\
        --output runs/.../g3_report.json

Exit codes: 0 = G3 passed, 1 = G3 failed, 2 = input error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(2)
    return json.loads(p.read_text(encoding="utf-8"))


def find_best_model_paths(summary_path: str | Path) -> dict[str, Any]:
    """Identify best TG and baseline model seeds from aggregate_summary.

    Returns dict with keys: tg_seed, baseline_seed, tg_adapter_path,
    baseline_adapter_path. Legacy tg_model_path / baseline_model_path aliases are
    kept for compatibility.
    """
    summary = _load_json(summary_path)
    summary_dir = Path(summary_path).resolve().parent

    per_seed = summary.get("per_seed", [])
    if not per_seed:
        return {
            "tg_seed": None,
            "baseline_seed": None,
            "tg_adapter_path": None,
            "baseline_adapter_path": None,
            "tg_model_path": None,
            "baseline_model_path": None,
        }

    best_tg_seed = min(per_seed, key=lambda r: r.get("warm_tg_best_valid_loss", float("inf")))
    best_bl_seed = min(per_seed, key=lambda r: r.get("warm_baseline_best_valid_loss", float("inf")))

    tg_seed = best_tg_seed.get("seed")
    bl_seed = best_bl_seed.get("seed")

    def _resolve_adapter(seed: int | None, variant: str) -> str | None:
        if seed is None:
            return None
        candidates = [
            summary_dir / f"seed_{seed}/coldwarm/warm/{variant}/best_model",
            summary_dir / f"seed_{seed}/warm/{variant}/best_model",
            summary_dir / f"seed_{seed}/{variant}/best_model",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    tg_adapter_path = _resolve_adapter(tg_seed, "tg_lora")
    baseline_adapter_path = _resolve_adapter(bl_seed, "baseline")

    return {
        "tg_seed": tg_seed,
        "baseline_seed": bl_seed,
        "tg_adapter_path": tg_adapter_path,
        "baseline_adapter_path": baseline_adapter_path,
        "tg_model_path": tg_adapter_path,
        "baseline_model_path": baseline_adapter_path,
    }


def infer_base_model(summary_path: str | Path, *, tg_seed: int | None, baseline_seed: int | None) -> str | None:
    """Infer the base model from copied paper-suite configs when not passed explicitly."""

    try:
        from omegaconf import OmegaConf
    except ImportError:
        return None

    summary_dir = Path(summary_path).resolve().parent
    candidates: list[Path] = []
    for seed, name in ((tg_seed, "tg_cache.yaml"), (baseline_seed, "baseline.yaml")):
        if seed is None:
            continue
        candidates.append(summary_dir / f"seed_{seed}/configs/{name}")

    for candidate in candidates:
        if not candidate.exists():
            continue
        cfg = OmegaConf.load(candidate)
        model_cfg = cfg.get("model")
        model_name = model_cfg.get("name_or_path") if hasattr(model_cfg, "get") else None
        if model_name:
            return str(model_name)
    return None


def evaluate_g3(
    tg_results: dict[str, float],
    bl_results: dict[str, float],
    *,
    max_aggregate_drop: float = 0.01,
    max_single_task_drop: float = 0.03,
    requested_tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate G3 gate: aggregate mean relative drop < 1%, single task < 3%.

    Compares TG vs baseline scores across tasks. A positive relative drop means
    TG is worse than baseline.

    A requested task that failed to produce a score on one or both sides (e.g.
    an lm-eval task whose primary metric was not recognized, so it never entered
    ``tg_results``/``bl_results``) is recorded in ``dropped_tasks`` with
    ``incomplete=True`` rather than vanishing silently — a gate that PASSES over
    a secretly-truncated task set is dishonest. The threshold verdict
    (``passed``) is still computed over the compared survivors; the downstream
    G3.3 "required tasks present" guard is what turns a drop into a gate FAIL.
    """
    compared_tasks = [t for t in tg_results if t in bl_results and bl_results[t] > 0]

    if not compared_tasks:
        # Still surface what was requested vs. what vanished, so an all-drop is
        # loud rather than reading as an empty-but-clean report.
        dropped = _compute_dropped_tasks(requested_tasks, tg_results, bl_results, [])
        detail = "No common tasks with non-zero baseline scores to compare"
        if dropped:
            detail += f"; dropped: {sorted(dropped)}"
        return {
            "passed": False,
            "aggregate_relative_drop": None,
            "task_drops": {},
            "compared_tasks": [],
            "dropped_tasks": sorted(dropped),
            "incomplete": bool(dropped),
            "detail": detail,
        }

    task_drops: dict[str, float] = {}
    for task in compared_tasks:
        tg_score = tg_results[task]
        bl_score = bl_results[task]
        if bl_score > 0:
            relative_drop = (bl_score - tg_score) / bl_score
            task_drops[task] = float(relative_drop)

    dropped = _compute_dropped_tasks(requested_tasks, tg_results, bl_results, compared_tasks)

    if not task_drops:
        detail = "No valid task comparisons"
        if dropped:
            detail += f"; dropped: {sorted(dropped)}"
        return {
            "passed": False,
            "aggregate_relative_drop": None,
            "task_drops": {},
            "compared_tasks": sorted(compared_tasks),
            "dropped_tasks": sorted(dropped),
            "incomplete": bool(dropped),
            "detail": detail,
        }

    aggregate_drop = float(sum(task_drops.values()) / len(task_drops))

    max_drop = float(max(task_drops.values()))
    aggregate_ok = aggregate_drop < max_aggregate_drop
    single_task_ok = max_drop < max_single_task_drop

    detail = f"Aggregate drop: {aggregate_drop*100:.2f}%, max single-task: {max_drop*100:.2f}%"
    if dropped:
        detail += f"; DROPPED (not compared): {sorted(dropped)}"

    return {
        "passed": bool(aggregate_ok and single_task_ok),
        "aggregate_relative_drop": aggregate_drop,
        "task_drops": task_drops,
        "compared_tasks": sorted(compared_tasks),
        "dropped_tasks": sorted(dropped),
        "incomplete": bool(dropped),
        "max_single_task_drop": max_drop,
        "detail": detail,
    }


def _compute_dropped_tasks(
    requested_tasks: list[str] | None,
    tg_results: dict[str, float],
    bl_results: dict[str, float],
    compared_tasks: list[str],
) -> list[str]:
    """Tasks that were requested (or seen on one side) but never compared.

    When ``requested_tasks`` is given (the normal harvest path), a drop is any
    requested task absent from the compared set — this catches the truly silent
    case where a task produced no score on EITHER side and therefore appears in
    neither results dict. Without the requested list that case is undetectable,
    so we fall back to flagging tasks present on only one side as incomparable.
    """
    compared = set(compared_tasks)
    if requested_tasks is not None:
        return [t for t in requested_tasks if t not in compared]
    seen = set(tg_results) | set(bl_results)
    return [t for t in seen if t not in compared]


def build_external_eval_results(
    *,
    tg_results: dict[str, float],
    baseline_results: dict[str, float],
    base_model: str | None,
    tg_adapter_path: str | None,
    baseline_adapter_path: str | None,
    tasks: list[str],
) -> dict[str, Any]:
    """Build the external_eval_results.json structure.

    ``tasks`` is the list of REQUESTED tasks. The ``comparison`` block records
    ``compared_tasks`` (actually scored on both sides) and ``dropped_tasks``
    (requested but silently unscored) so a reader can tell intent from reality —
    the top-level ``tasks`` field alone would hide a silent drop.
    """
    g3 = evaluate_g3(tg_results, baseline_results, requested_tasks=tasks)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
        "models": {
            "tg": {
                "base_model": base_model,
                "adapter_path": tg_adapter_path,
                "model_path": tg_adapter_path,
                "results": tg_results,
            },
            "baseline": {
                "base_model": base_model,
                "adapter_path": baseline_adapter_path,
                "model_path": baseline_adapter_path,
                "results": baseline_results,
            },
        },
        "comparison": {
            "aggregate_relative_drop": g3["aggregate_relative_drop"],
            "task_relative_drops": g3.get("task_drops", {}),
            "g3_passed": g3["passed"],
            "compared_tasks": g3.get("compared_tasks", []),
            "dropped_tasks": g3.get("dropped_tasks", []),
            "incomplete": g3.get("incomplete", False),
        },
    }


# Primary lm-eval metric per known task (the raw metric name, sans the
# ",none" suffix lm-eval appends). truthfulqa_mc2 reports ``mc2`` and gsm8k
# reports ``exact_match`` — NEITHER is in the acc/acc_norm family, so the old
# hardcoded acc-only key list silently dropped their scores, and the task
# vanished from the G3 report (and from the G3.3 "required tasks" guard, which
# read the requested-task list). Map the required + common tasks explicitly.
_TASK_PRIMARY_METRIC: dict[str, str] = {
    "truthfulqa_mc2": "mc2",
    "truthfulqa_mc1": "mc1",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "piqa": "acc",
    "gsm8k": "exact_match",
    "mmlu": "acc",
}


def _extract_primary_metric(task_name: str, task_data: dict[str, Any]) -> float | None:
    """Extract a task's primary lm-eval score, or ``None`` if unrecognized.

    Returns ``None`` (rather than silently omitting the task) so the caller can
    record the task as dropped via :func:`evaluate_g3`'s ``dropped_tasks``
    instead of letting it vanish from the report. Preference order: the task's
    known primary metric (e.g. ``mc2`` for truthfulqa), then the generic
    acc/acc_norm family for unmapped tasks.
    """
    candidates: list[str] = []
    primary = _TASK_PRIMARY_METRIC.get(task_name)
    if primary is not None:
        candidates.append(f"{primary},none")
        candidates.append(primary)
    candidates.extend(["acc_norm,none", "acc,none", "acc_norm", "acc"])
    for key in candidates:
        value = task_data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _run_lm_eval(
    *,
    base_model: str,
    adapter_path: str,
    tasks: list[str],
    batch_size: str = "auto",
    limit: int | None = None,
) -> dict[str, float]:
    """Run lm-evaluation-harness and extract scores.

    Returns dict mapping task name to primary metric score.
    """
    try:
        import lm_eval  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: lm-eval not installed. Run: pip install lm-eval", file=sys.stderr)
        sys.exit(2)

    preferred_model_args = (
        f"pretrained={base_model},"
        f"peft={adapter_path},"
        "dtype=float16,"
        "load_in_4bit=True"
    )

    try:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=preferred_model_args,
            tasks=tasks,
            batch_size=batch_size,
            limit=limit,
        )
    except TypeError as exc:
        if "load_in_4bit" not in str(exc):
            raise
        print(
            "Warning: lm-eval string-based 4-bit loading failed. Retrying with a preloaded 4-bit PEFT model.",
            file=sys.stderr,
        )
        results = lm_eval.simple_evaluate(
            model=_build_preloaded_4bit_hflm(
                base_model=base_model,
                adapter_path=adapter_path,
                batch_size=batch_size,
            ),
            tasks=tasks,
            batch_size=batch_size,
            limit=limit,
        )

    task_scores: dict[str, float] = {}
    for task_name, task_data in results.get("results", {}).items():
        # Extract the task's primary metric (mc2 for truthfulqa, exact_match for
        # gsm8k, acc/acc_norm otherwise). A task with no recognized metric is
        # NOT added here — evaluate_g3(dropped_tasks=...) surfaces the drop
        # loudly rather than letting the task vanish from the G3 report.
        score = _extract_primary_metric(task_name, task_data)
        if score is not None:
            task_scores[task_name] = score

    return task_scores


def _build_preloaded_4bit_hflm(
    *,
    base_model: str,
    adapter_path: str,
    batch_size: str,
):
    """Build a preloaded 4-bit PEFT-wrapped HFLM instance.

    This bypasses lm-eval's string-based loader path, which currently forwards
    `load_in_4bit` incompatibly on the installed transformers version.
    """
    from lm_eval.models.huggingface import HFLM  # type: ignore[import-untyped]
    from omegaconf import OmegaConf
    from peft import PeftModel

    from src.model.load_model import load_base_model, load_tokenizer

    cfg = OmegaConf.create(
        {
            "model": {
                "name_or_path": base_model,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bf16",
                "bnb_4bit_quant_type": "nf4",
                "dtype": "bf16",
                "device": "cuda",
                "device_map": None,
            },
            "training": {
                "gradient_checkpointing": False,
            },
        }
    )
    model = load_base_model(cfg)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    tokenizer = load_tokenizer(cfg)
    return HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external quality evaluation (G3 Gate)")
    parser.add_argument("summary_json", nargs="?", help="Path to aggregate_summary.json")
    parser.add_argument("--analysis-mode", action="store_true", help="Analyze existing external eval results without running lm-eval")
    parser.add_argument("--external-eval", help="Path to existing external_eval_results.json (for analysis mode)")
    parser.add_argument("--tasks", default="truthfulqa_mc2,arc_easy,hellaswag", help="Comma-separated eval tasks (default: truthfulqa_mc2,arc_easy,hellaswag)")
    parser.add_argument("--batch-size", default="auto", help="lm-eval batch size (default: auto)")
    parser.add_argument("--limit", type=int, help="Limit the number of examples per task (for faster debug/test runs)")
    parser.add_argument("--base-model", help="Base model name_or_path used to load the PEFT adapters")
    parser.add_argument("--output", "-o", help="Output path for external_eval_results.json")
    parser.add_argument("--max-aggregate-drop", type=float, default=0.01, help="G3 max aggregate relative drop (default: 0.01 = 1%%)")
    parser.add_argument("--max-single-task-drop", type=float, default=0.03, help="G3 max single-task relative drop (default: 0.03 = 3%%)")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.analysis_mode:
        if not args.external_eval:
            print("ERROR: --analysis-mode requires --external-eval", file=sys.stderr)
            sys.exit(2)

        eval_data = _load_json(args.external_eval)
        tg_results = eval_data.get("models", {}).get("tg", {}).get("results", {})
        bl_results = eval_data.get("models", {}).get("baseline", {}).get("results", {})
        # Prefer the requested-task list recorded in the file (intent) so a task
        # that silently failed to score is detected as a drop, not hidden.
        requested = eval_data.get("tasks") or tasks

        g3 = evaluate_g3(
            tg_results, bl_results,
            max_aggregate_drop=args.max_aggregate_drop,
            max_single_task_drop=args.max_single_task_drop,
            requested_tasks=requested,
        )

        print("=" * 50)
        print("G3 Gate: External Quality Retention")
        print("=" * 50)
        print(f"Tasks evaluated: {list(g3.get('task_drops', {}).keys())}")
        for task, drop in g3.get("task_drops", {}).items():
            status = "OK" if drop < args.max_single_task_drop else "FAIL"
            print(f"  {task}: relative drop = {drop*100:.2f}% [{status}]")
        agg = g3.get("aggregate_relative_drop")
        if agg is not None:
            print(f"Aggregate mean drop: {agg*100:.2f}%")
        # A dropped task is LOUD: it means a requested task produced no score
        # and would otherwise skew (or vacuously pass) the gate. Surface it as a
        # prominent banner, not just a buried JSON field.
        dropped = g3.get("dropped_tasks", [])
        if dropped:
            print(
                f"\nINCOMPLETE: {len(dropped)} requested task(s) produced no comparable "
                f"score and were dropped: {dropped}",
                file=sys.stderr,
            )
        print(f"\nG3 Result: {'PASS' if g3['passed'] else 'FAIL'}"
              f"{' (INCOMPLETE)' if g3.get('incomplete') else ''}")
        print("=" * 50)

        if args.output:
            output_data = {**eval_data, "g3_analysis": g3}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(output_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"\nResults written to {args.output}")

        sys.exit(0 if g3["passed"] else 1)

    # Full evaluation mode (requires GPU)
    if not args.summary_json:
        print("ERROR: summary_json required when not in --analysis-mode", file=sys.stderr)
        parser.print_help()
        sys.exit(2)

    paths = find_best_model_paths(args.summary_json)

    if not paths["tg_adapter_path"] or not paths["baseline_adapter_path"]:
        print("ERROR: Could not locate adapter paths from aggregate summary", file=sys.stderr)
        sys.exit(2)

    base_model = args.base_model or infer_base_model(
        args.summary_json,
        tg_seed=paths["tg_seed"],
        baseline_seed=paths["baseline_seed"],
    )
    if not base_model:
        print(
            "ERROR: Could not determine base model. Pass --base-model or keep copied configs under seed_*/configs/.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Base model: {base_model}")
    print(f"Best TG adapter: seed {paths['tg_seed']} at {paths['tg_adapter_path']}")
    print(f"Best baseline adapter: seed {paths['baseline_seed']} at {paths['baseline_adapter_path']}")
    print(f"Evaluating tasks: {tasks}")

    tg_results = _run_lm_eval(
        base_model=base_model,
        adapter_path=paths["tg_adapter_path"],
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    bl_results = _run_lm_eval(
        base_model=base_model,
        adapter_path=paths["baseline_adapter_path"],
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    output = build_external_eval_results(
        tg_results=tg_results,
        baseline_results=bl_results,
        base_model=base_model,
        tg_adapter_path=paths["tg_adapter_path"],
        baseline_adapter_path=paths["baseline_adapter_path"],
        tasks=tasks,
    )

    output_path = args.output or str(Path(args.summary_json).parent / "external_eval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nExternal eval results written to {output_path}")

    g3 = evaluate_g3(
        tg_results, bl_results,
        max_aggregate_drop=args.max_aggregate_drop,
        max_single_task_drop=args.max_single_task_drop,
        requested_tasks=tasks,
    )
    status = "PASS" if g3["passed"] else "FAIL"
    print(f"G3 Gate: {status} — {g3.get('detail', '')}")
    if g3.get("incomplete"):
        print(
            f"INCOMPLETE: {len(g3['dropped_tasks'])} requested task(s) produced no "
            f"comparable score and were dropped: {g3['dropped_tasks']}",
            file=sys.stderr,
        )

    sys.exit(0 if g3["passed"] else 1)


if __name__ == "__main__":
    main()
