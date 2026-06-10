import argparse
import json
import logging
import os
import time
import gc
from pathlib import Path
from difflib import SequenceMatcher

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.load_model import load_base_model, load_tokenizer, apply_lora
from src.model.lora_utils import configure_trainable_lora_scope
from src.training.config_schema import load_validate_and_build_config
from src.utils.io import load_jsonl
from src.utils.device import detect_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("downstream-eval")


def compute_char_f1(ref: str, gen: str) -> float:
    """Compute character-level F1 score as a language-agnostic similarity metric."""
    ref_clean = "".join(ref.split())
    gen_clean = "".join(gen.split())
    if not ref_clean or not gen_clean:
        return 0.0
    matcher = SequenceMatcher(None, ref_clean, gen_clean)
    match_len = sum(triple.size for triple in matcher.get_matching_blocks())
    precision = match_len / len(gen_clean)
    recall = match_len / len(ref_clean)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def check_json_validity(gen: str, expected_keys: list) -> tuple[bool, bool]:
    """Clean markdown JSON wrappers and evaluate JSON structure and key compliance."""
    gen_clean = gen.strip()
    
    # Strip markdown code blocks if any
    if gen_clean.startswith("```json"):
        gen_clean = gen_clean[7:]
    elif gen_clean.startswith("```"):
        gen_clean = gen_clean[3:]
    if gen_clean.endswith("```"):
        gen_clean = gen_clean[:-3]
    gen_clean = gen_clean.strip()
    
    try:
        parsed = json.loads(gen_clean)
        is_valid = True
        has_keys = True
        if expected_keys and isinstance(parsed, dict):
            has_keys = all(k in parsed for k in expected_keys)
        return is_valid, has_keys
    except json.JSONDecodeError:
        # Fallback: try finding first { and last }
        start = gen.find('{')
        end = gen.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(gen[start:end+1])
                is_valid = True
                has_keys = True
                if expected_keys and isinstance(parsed, dict):
                    has_keys = all(k in parsed for k in expected_keys)
                return is_valid, has_keys
            except json.JSONDecodeError:
                pass
        return False, False


def evaluate_dataset(
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
        for idx, rec in enumerate(tqdm(records, desc=f"Evaluating {task_type}")):
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
            
            # Memory cleaning
            del enc, outputs
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
    parser = argparse.ArgumentParser(description="Evaluate downstream tasks for TG-LoRA")
    parser.add_argument("--config", type=str, default="configs/9b_tg_lora.yaml", help="Path to config YAML")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on (cuda, mps, cpu)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples evaluated")
    parser.add_argument("--output-dir", type=str, default="reports/downstream_eval", help="Directory to save evaluation reports")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Config
    _, cfg = load_validate_and_build_config(args.config)
    
    # 2. Determine Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = detect_device()
        
    logger.info(f"Using device: {device}")
    
    # Force disable 4bit quant on non-CUDA (MPS/CPU) to prevent bitsandbytes errors
    if device.type != "cuda":
        logger.warning(f"Device is '{device.type}'. Force disabling load_in_4bit to prevent bitsandbytes import errors.")
        cfg.model.load_in_4bit = False
        
    # 3. Load Model and Tokenizer
    logger.info("Loading tokenizer and base model...")
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    
    if args.adapter_path:
        logger.info(f"Applying adapter from {args.adapter_path}...")
        # Make sure adapter config matches requirements
        model = apply_lora(model, cfg)
        # Load state dict if adapter path is provided
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        logger.info("No adapter path provided. Evaluating BASE model.")
        model = apply_lora(model, cfg)
        
    trainable_lora_scope = cfg.training.get("trainable_lora_scope", "all")
    configure_trainable_lora_scope(model, trainable_lora_scope)
    
    # Ensure model is on correct device
    model = model.to(device)
    
    # 4. Check Dataset paths
    jp_dataset_path = Path("data/downstream/jp_capability.jsonl")
    json_dataset_path = Path("data/downstream/format_json.jsonl")
    
    if not jp_dataset_path.exists() or not json_dataset_path.exists():
        raise FileNotFoundError("Downstream datasets must be pre-created under data/downstream/")
        
    # 5. Evaluate
    logger.info("Evaluating Japanese Capability downstream task...")
    jp_results = evaluate_dataset(
        model, tokenizer, jp_dataset_path, device,
        task_type="jp_capability", max_examples=args.max_examples
    )
    
    logger.info("Evaluating JSON Format Compliance downstream task...")
    json_results = evaluate_dataset(
        model, tokenizer, json_dataset_path, device,
        task_type="format_json", max_examples=args.max_examples
    )
    
    # 6. Save results
    report_json_path = output_dir / "downstream_eval_report.json"
    report_md_path = output_dir / "downstream_eval_report.md"
    
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "adapter_path": args.adapter_path,
        "device": str(device),
        "results": {
            "jp_capability": jp_results,
            "format_json": json_results
        }
    }
    
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
        
    # Generate Markdown Summary
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# TG-LoRA Downstream Evaluation Report\n\n")
        f.write(f"- **Evaluation Date**: {report_data['timestamp']}\n")
        f.write(f"- **Config**: `{report_data['config']}`\n")
        f.write(f"- **Adapter Checkpoint**: `{report_data['adapter_path'] if report_data['adapter_path'] else 'None (Base Model)'}`\n")
        f.write(f"- **Execution Device**: `{report_data['device']}`\n\n")
        
        f.write("## 1. Summary Metrics\n\n")
        f.write("| Task | Metric | Value |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| **Japanese Capability** | Mean Char-F1 | {jp_results['summary']['mean_char_f1']:.4f} |\n")
        f.write(f"| **JSON Format Compliance** | Mean Char-F1 | {json_results['summary']['mean_char_f1']:.4f} |\n")
        f.write(f"| | JSON Validity Rate | {json_results['summary']['json_validity_rate']:.4%} |\n")
        f.write(f"| | Key Compliance Rate | {json_results['summary']['key_compliance_rate']:.4%} |\n\n")
        
        f.write("## 2. Detailed Task Results\n\n")
        f.write("### 2.1 Japanese Capability (Sample outputs)\n\n")
        for i, detail in enumerate(jp_results['details'][:5]):
            f.write(f"#### Sample {i+1} (Char-F1: {detail['char_f1']:.4f})\n")
            f.write(f"**Prompt**:\n> {detail['prompt']}\n\n")
            f.write(f"**Expected Output**:\n> {detail['expected']}\n\n")
            f.write(f"**Generated Output**:\n> {detail['generated']}\n\n")
            f.write("---\n\n")
            
        f.write("### 2.2 JSON Format Compliance (Sample outputs)\n\n")
        for i, detail in enumerate(json_results['details'][:5]):
            f.write(f"#### Sample {i+1} (Valid JSON: {detail['is_json']}, Key Compliance: {detail['has_keys']})\n")
            f.write(f"**Prompt**:\n> {detail['prompt']}\n\n")
            f.write(f"**Expected Output**:\n> {detail['expected']}\n\n")
            f.write(f"**Generated Output**:\n> {detail['generated']}\n\n")
            f.write("---\n\n")
            
    logger.info(f"Evaluation complete. Reports saved to {args.output_dir}")
    print(f"\n=== DOWNSTREAM EVALUATION SUMMARY ===")
    print(f"Japanese Capability - Mean Char-F1: {jp_results['summary']['mean_char_f1']:.4f}")
    print(f"JSON Format Compliance - Mean Char-F1: {json_results['summary']['mean_char_f1']:.4f}")
    print(f"JSON Format Compliance - JSON Validity Rate: {json_results['summary']['json_validity_rate']:.4%}")
    print(f"JSON Format Compliance - Key Compliance Rate: {json_results['summary']['key_compliance_rate']:.4%}")
    print(f"======================================\n")


if __name__ == "__main__":
    main()
