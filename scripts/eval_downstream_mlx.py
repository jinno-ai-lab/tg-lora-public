import argparse
import json
import logging
import time
from pathlib import Path

from mlx_lm import generate
from mlx_lm.tuner.utils import load_adapters
from mlx_lm.utils import load as mlx_load
from tqdm import tqdm

from scripts.eval_utils import check_json_validity, compute_char_f1, load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("downstream-eval-mlx")


def evaluate_dataset_mlx(
    model,
    tokenizer,
    dataset_path: Path,
    task_type: str,
    max_examples: int | None = None,
    max_new_tokens: int = 128
) -> dict:
    records = load_jsonl(str(dataset_path))
    if max_examples:
        records = records[:max_examples]

    eval_results = []

    total = 0
    valid_json_count = 0
    key_compliance_count = 0
    total_char_f1 = 0.0

    for idx, rec in enumerate(tqdm(records, desc=f"MLX Evaluating {task_type}")):
        prompt = rec.get("prompt", "")
        expected = rec.get("completion", rec.get("output", ""))
        if not prompt:
            continue

        # Generate with mlx_lm.generate
        gen_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            verbose=False
        ).strip()

        total += 1

        # Metric evaluation
        char_f1 = compute_char_f1(expected, gen_text)
        total_char_f1 += char_f1

        is_json = False
        has_keys = False

        expected_keys = []
        if task_type == "format_json":
            try:
                exp_parsed = json.loads(expected.strip())
                if isinstance(exp_parsed, dict):
                    expected_keys = list(exp_parsed.keys())
            except json.JSONDecodeError:
                pass

            is_json, has_keys = check_json_validity(gen_text, expected_keys)
            if is_json:
                valid_json_count += 1
            if has_keys:
                key_compliance_count += 1

        eval_results.append({
            "prompt": prompt,
            "expected": expected,
            "generated": gen_text,
            "char_f1": char_f1,
            "is_json": is_json,
            "has_keys": has_keys,
            "expected_keys": expected_keys
        })

    summary = {
        "total": total,
        "mean_char_f1": total_char_f1 / total if total > 0 else 0.0,
    }

    if task_type == "format_json":
        summary.update({
            "json_validity_rate": valid_json_count / total if total > 0 else 0.0,
            "key_compliance_rate": key_compliance_count / total if total > 0 else 0.0,
        })

    return {
        "summary": summary,
        "details": eval_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate downstream tasks for MLX models/adapters")
    parser.add_argument("--model-path", type=str, default=".cache/mlx_models/Qwen--Qwen3.5-9B", help="Path to MLX model folder")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to MLX adapter folder")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples evaluated")
    parser.add_argument("--output-dir", type=str, default="reports/downstream_eval_mlx", help="Directory to save evaluation reports")
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

    # 2. Check Dataset paths
    jp_dataset_path = Path("data/downstream/jp_capability.jsonl")
    json_dataset_path = Path("data/downstream/format_json.jsonl")

    if not jp_dataset_path.exists() or not json_dataset_path.exists():
        raise FileNotFoundError("Downstream datasets must be pre-created under data/downstream/")

    # 3. Evaluate
    logger.info("Evaluating Japanese Capability downstream task...")
    jp_results = evaluate_dataset_mlx(
        model, tokenizer, jp_dataset_path,
        task_type="jp_capability", max_examples=args.max_examples
    )

    logger.info("Evaluating JSON Format Compliance downstream task...")
    json_results = evaluate_dataset_mlx(
        model, tokenizer, json_dataset_path,
        task_type="format_json", max_examples=args.max_examples
    )

    # 4. Save results
    tag = "base" if not args.adapter_path else Path(args.adapter_path).name
    report_json_path = output_dir / f"report_{tag}.json"
    report_md_path = output_dir / f"report_{tag}.md"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "results": {
            "jp_capability": jp_results,
            "format_json": json_results
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Generate Markdown Summary
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# MLX Downstream Evaluation Report ({tag})\n\n")
        f.write(f"- **Evaluation Date**: {report_data['timestamp']}\n")
        f.write(f"- **Base Model**: `{report_data['model_path']}`\n")
        f.write(f"- **Adapter Checkpoint**: `{report_data['adapter_path'] if report_data['adapter_path'] else 'None (Base Model)'}`\n\n")

        f.write("## 1. Summary Metrics\n\n")
        f.write("| Task | Metric | Value |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| **Japanese Capability** | Mean Char-F1 | {jp_results['summary']['mean_char_f1']:.4f} |\n")
        f.write(f"| **JSON Format Compliance** | Mean Char-F1 | {json_results['summary']['mean_char_f1']:.4f} |\n")
        f.write(f"| | JSON Validity Rate | {json_results['summary']['json_validity_rate']:.4%} |\n")
        f.write(f"| | Key Compliance Rate | {json_results['summary']['key_compliance_rate']:.4%} |\n\n")

    logger.info(f"Evaluation complete. Reports saved to {report_md_path}")


if __name__ == "__main__":
    main()
