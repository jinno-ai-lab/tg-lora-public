import argparse
import gc
import json
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval_utils import check_json_validity, compute_char_f1, load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("downstream-eval-pytorch")


def evaluate_dataset_pytorch(
    model,
    tokenizer,
    dataset_path: Path,
    device: torch.device,
    task_type: str,
    max_examples: int | None = None,
    max_new_tokens: int = 128
) -> dict:
    records = load_jsonl(str(dataset_path))
    if max_examples:
        records = records[:max_examples]

    was_training = model.training
    model.eval()

    eval_results = []
    total = 0
    valid_json_count = 0
    key_compliance_count = 0
    total_char_f1 = 0.0

    try:
        for idx, rec in enumerate(tqdm(records, desc=f"PyTorch Evaluating {task_type}")):
            prompt = rec.get("prompt", "")
            expected = rec.get("completion", rec.get("output", ""))
            if not prompt:
                continue

            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_text = tokenizer.decode(
                outputs[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            total += 1
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

            del enc, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    finally:
        if was_training:
            model.train()

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
    parser = argparse.ArgumentParser(description="Evaluate downstream tasks for PyTorch/PEFT models")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Path to base HF model")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to PEFT adapter folder")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on (cuda, mps, cpu)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples evaluated")
    parser.add_argument("--output-dir", type=str, default="reports/downstream_eval_pytorch", help="Directory to save evaluation reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine Device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # 2. Load Model and Tokenizer
    logger.info(f"Loading base model and tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        trust_remote_code=True
    )

    if args.adapter_path:
        logger.info(f"Applying adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model = model.to(device)

    # 3. Check Dataset paths
    jp_dataset_path = Path("data/downstream/jp_capability.jsonl")
    json_dataset_path = Path("data/downstream/format_json.jsonl")

    if not jp_dataset_path.exists() or not json_dataset_path.exists():
        raise FileNotFoundError("Downstream datasets must be pre-created under data/downstream/")

    # 4. Evaluate
    logger.info("Evaluating Japanese Capability downstream task...")
    jp_results = evaluate_dataset_pytorch(
        model, tokenizer, jp_dataset_path, device,
        task_type="jp_capability", max_examples=args.max_examples
    )

    logger.info("Evaluating JSON Format Compliance downstream task...")
    json_results = evaluate_dataset_pytorch(
        model, tokenizer, json_dataset_path, device,
        task_type="format_json", max_examples=args.max_examples
    )

    # 5. Save results
    tag = "base" if not args.adapter_path else Path(args.adapter_path).name
    report_json_path = output_dir / f"report_pytorch_{tag}.json"
    report_md_path = output_dir / f"report_pytorch_{tag}.md"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "device": str(device),
        "results": {
            "jp_capability": jp_results,
            "format_json": json_results
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# PyTorch Downstream Evaluation Report ({tag})\n\n")
        f.write(f"- **Evaluation Date**: {report_data['timestamp']}\n")
        f.write(f"- **Base Model**: `{report_data['model_path']}`\n")
        f.write(f"- **Adapter Checkpoint**: `{report_data['adapter_path'] if report_data['adapter_path'] else 'None (Base Model)'}`\n")
        f.write(f"- **Execution Device**: `{report_data['device']}`\n\n")

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
