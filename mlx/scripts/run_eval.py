#!/usr/bin/env python3
"""Run lm-evaluation-harness with MLX backend.

Usage:
    python scripts/run_mlx_eval.py \
        --model .cache/mlx_models/Qwen--Qwen3.5-9B \
        --tasks arc_easy,hellaswag \
        --adapter_path runs/mlx_qlora_baseline_500/adapters.safetensors
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mlx.src.eval.mlx_lm_backend import MLXLMEval  # noqa: E402


def main():
    import argparse

    parser = argparse.ArgumentParser(description="lm-eval with MLX backend")
    parser.add_argument("--model", required=True, help="Path to MLX model")
    parser.add_argument("--adapter_path", default=None, help="Path to MLX LoRA adapter")
    parser.add_argument("--tasks", default="arc_easy,hellaswag,gsm8k,truthfulqa_mc2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="reports/eval/mlx_results.json")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples per task"
    )
    args = parser.parse_args()

    from lm_eval.models import MODEL_MAPPING

    MODEL_MAPPING["mlx"] = MLXLMEval

    # Create model instance directly
    model = MLXLMEval(
        model=args.model,
        adapter_path=args.adapter_path,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    # Run evaluation
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    results = simple_evaluate(
        model=model,
        tasks=args.tasks.split(","),
        task_manager=task_manager,
        limit=args.limit,
    )

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    if "results" in results:
        print("\n" + "=" * 60)
        print("  MLX Evaluation Results")
        print("=" * 60)
        for task_name, metrics in results["results"].items():
            print(f"\n{task_name}:")
            for metric_name, value in metrics.items():
                if not metric_name.startswith("_"):
                    if isinstance(value, float):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
        print("=" * 60)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
