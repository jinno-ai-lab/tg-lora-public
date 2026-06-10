#!/usr/bin/env python
"""Run external quality evaluation on ALL seeds in the paper-memory suite.

Iterates over seeds 42, 43, 44, loads baseline and TG-LoRA models,
runs lm-eval on ARC-Easy, HellaSwag, and TruthfulQA MC2,
and outputs a unified 3-seed downstream evaluation report.
"""
import sys
from pathlib import Path
import json
from datetime import datetime, timezone

# Add repository root to python path to allow imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from scripts.run_paper_external_eval import _run_lm_eval, infer_base_model  # noqa: E402

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run downstream evaluation on all seeds")
    parser.add_argument("--summary-path", type=str, default=None, help="Path to aggregate_summary.json")
    args = parser.parse_args()

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
    else:
        summary_path = repo_root / "runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json"

    if not summary_path.exists():
        print(f"ERROR: Aggregate summary not found at {summary_path}")
        sys.exit(1)
        
    summary_dir = summary_path.parent
    seeds = [42, 43, 44]
    tasks = ["arc_easy", "hellaswag", "truthfulqa_mc2"]
    
    # Infer base model
    base_model = infer_base_model(summary_path, tg_seed=44, baseline_seed=43) or "Qwen/Qwen3.5-9B"
    print(f"Base model identified: {base_model}")
    
    results = {}
    
    for seed in seeds:
        print("\n==========================================")
        print(f" Evaluating Seed {seed}")
        print("==========================================")
        
        results[str(seed)] = {}
        
        # Locate adapters
        tg_adapter = summary_dir / f"seed_{seed}/coldwarm/warm/tg_lora/best_model"
        bl_adapter = summary_dir / f"seed_{seed}/coldwarm/warm/baseline/best_model"
        
        if not tg_adapter.exists() or not bl_adapter.exists():
            print(f"WARNING: Checkpoints for seed {seed} not found, skipping.")
            continue
            
        print(f"Evaluating TG adapter: {tg_adapter}")
        tg_scores = _run_lm_eval(
            base_model=base_model,
            adapter_path=str(tg_adapter),
            tasks=tasks,
            batch_size="auto",
        )
        print(f"TG Scores: {tg_scores}")
        
        print(f"Evaluating Baseline adapter: {bl_adapter}")
        bl_scores = _run_lm_eval(
            base_model=base_model,
            adapter_path=str(bl_adapter),
            tasks=tasks,
            batch_size="auto",
        )
        print(f"Baseline Scores: {bl_scores}")
        
        results[str(seed)] = {
            "tg": tg_scores,
            "baseline": bl_scores
        }
        
    # Write intermediate results
    output_path = summary_dir / "external_eval_3seeds_raw.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRaw evaluations written to {output_path}")
    
    # Compute aggregate stats
    tg_agg = {t: [] for t in tasks}
    bl_agg = {t: [] for t in tasks}
    
    for seed in results:
        for t in tasks:
            if t in results[seed]["tg"]:
                tg_agg[t].append(results[seed]["tg"][t])
            if t in results[seed]["baseline"]:
                bl_agg[t].append(results[seed]["baseline"][t])
                
    # Calculate means
    final_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
        "base_model": base_model,
        "seeds_evaluated": [int(s) for s in results.keys()],
        "results_per_seed": results,
        "aggregates": {
            "tg": {},
            "baseline": {},
            "relative_drops": {}
        }
    }
    
    import statistics
    
    for t in tasks:
        tg_vals = tg_agg[t]
        bl_vals = bl_agg[t]
        
        tg_mean = statistics.mean(tg_vals) if tg_vals else 0.0
        tg_std = statistics.stdev(tg_vals) if len(tg_vals) > 1 else 0.0
        
        bl_mean = statistics.mean(bl_vals) if bl_vals else 0.0
        bl_std = statistics.stdev(bl_vals) if len(bl_vals) > 1 else 0.0
        
        rel_drop = (bl_mean - tg_mean) / bl_mean if bl_mean > 0 else 0.0
        
        final_report["aggregates"]["tg"][t] = {"mean": tg_mean, "std": tg_std}
        final_report["aggregates"]["baseline"][t] = {"mean": bl_mean, "std": bl_std}
        final_report["aggregates"]["relative_drops"][t] = rel_drop
        
    # Calculate overall aggregate mean drop
    drops = list(final_report["aggregates"]["relative_drops"].values())
    final_report["aggregate_relative_drop"] = sum(drops) / len(drops) if drops else 0.0
    
    summary_output_path = summary_dir / "external_eval_3seeds_summary.json"
    summary_output_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"Summary report written to {summary_output_path}")
    print(f"Overall 3-seed aggregate quality drop: {final_report['aggregate_relative_drop']*100:.4f}%")

if __name__ == "__main__":
    main()
