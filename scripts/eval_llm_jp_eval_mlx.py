import argparse
import json
import logging
import time
from pathlib import Path

from datasets import load_dataset
from mlx_lm import generate
from mlx_lm.tuner.utils import load_adapters
from mlx_lm.utils import load as mlx_load
from tqdm import tqdm

from scripts.eval_utils import compute_char_f1
from scripts.jp_eval_formats import format_jcommonsenseqa, format_jnli, format_jsquad

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm-jp-eval-mlx")


def main():
    parser = argparse.ArgumentParser(description="Evaluate JGLUE tasks from llm-jp-eval")
    parser.add_argument("--model-path", type=str, default=".cache/mlx_models/Qwen--Qwen3.5-9B", help="Path to MLX model folder")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to MLX adapter folder")
    parser.add_argument("--max-examples", type=int, default=50, help="Max number of examples per task to evaluate")
    parser.add_argument("--output-dir", type=str, default="reports/llm_jp_eval_mlx", help="Directory to save evaluation reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load MLX Model
    logger.info(f"Loading MLX model from {args.model_path}...")
    model, tokenizer = mlx_load(args.model_path)
    model.freeze()

    if args.adapter_path:
        logger.info(f"Loading MLX adapter from {args.adapter_path}...")
        load_adapters(model, args.adapter_path)

    model.eval()

    # 2. Download and Parse Datasets
    logger.info("Loading datasets from Hugging Face...")
    datasets_raw = {}
    try:
        datasets_raw["jcommonsenseqa"] = load_dataset("sbintuitions/JCommonsenseQA", split="validation")
        logger.info("[OK] JCommonsenseQA loaded.")
    except Exception as e:
        logger.error(f"Failed to load JCommonsenseQA: {e}")

    try:
        datasets_raw["jnli"] = load_dataset("zenless-lab/jnli", split="test")
        logger.info("[OK] JNLI loaded.")
    except Exception as e:
        logger.error(f"Failed to load JNLI: {e}")

    try:
        datasets_raw["jsquad"] = load_dataset("zenless-lab/jsquad", split="test")
        logger.info("[OK] JSQuAD loaded.")
    except Exception as e:
        logger.error(f"Failed to load JSQuAD: {e}")

    results = {}

    # 3. Evaluate each dataset
    for task_name, ds in datasets_raw.items():
        logger.info(f"Evaluating {task_name}...")
        examples = list(ds)
        if args.max_examples:
            examples = examples[:args.max_examples]

        task_results = []
        correct_count = 0
        total_char_f1 = 0.0

        for item in tqdm(examples, desc=f"Evaluating {task_name}"):
            if task_name == "jcommonsenseqa":
                prompt, expected = format_jcommonsenseqa(item)
            elif task_name == "jnli":
                prompt, expected = format_jnli(item)
            elif task_name == "jsquad":
                prompt, expected = format_jsquad(item)
            else:
                continue

            gen_text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=64,
                verbose=False
            ).strip()

            # Metric logic
            is_correct = False
            char_f1 = compute_char_f1(expected, gen_text)
            total_char_f1 += char_f1

            if task_name == "jcommonsenseqa":
                is_correct = expected in gen_text[:5]
            elif task_name == "jnli":
                is_correct = expected in gen_text
            elif task_name == "jsquad":
                is_correct = (gen_text == expected) or (char_f1 >= 0.85)

            if is_correct:
                correct_count += 1

            task_results.append({
                "prompt": prompt,
                "expected": expected,
                "generated": gen_text,
                "is_correct": is_correct,
                "char_f1": char_f1
            })

        accuracy = correct_count / len(examples) if examples else 0.0
        mean_char_f1 = total_char_f1 / len(examples) if examples else 0.0

        results[task_name] = {
            "accuracy": accuracy,
            "mean_char_f1": mean_char_f1,
            "details": task_results
        }
        logger.info(f"Task {task_name} complete: Accuracy={accuracy:.4%}, Mean Char-F1={mean_char_f1:.4f}")

    # 4. Save results
    tag = "base" if not args.adapter_path else Path(args.adapter_path).name
    report_json_path = output_dir / f"report_llm_jp_eval_{tag}.json"
    report_md_path = output_dir / f"report_llm_jp_eval_{tag}.md"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "results": {
            k: {
                "accuracy": v["accuracy"],
                "mean_char_f1": v["mean_char_f1"]
            }
            for k, v in results.items()
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# llm-jp-eval (JGLUE) Evaluation Report ({tag})\n\n")
        f.write(f"- **Evaluation Date**: {report_data['timestamp']}\n")
        f.write(f"- **Base Model**: `{report_data['model_path']}`\n")
        f.write(f"- **Adapter Checkpoint**: `{report_data['adapter_path'] if report_data['adapter_path'] else 'None (Base Model)'}`\n\n")

        f.write("## 1. Summary Metrics\n\n")
        f.write("| Task (Dataset) | Accuracy | Mean Char-F1 |\n")
        f.write("| --- | --- | --- |\n")
        for k, v in results.items():
            f.write(f"| **{k.upper()}** | {v['accuracy']:.2%} | {v['mean_char_f1']:.4f} |\n")

    logger.info(f"Evaluation complete. Summary saved to {report_md_path}")


if __name__ == "__main__":
    main()
