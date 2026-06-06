import argparse
import json
import logging
import os
import time
from pathlib import Path
from difflib import SequenceMatcher

from tqdm import tqdm
from datasets import load_dataset
from mlx_lm.utils import load as mlx_load
from mlx_lm.tuner.utils import load_adapters
from mlx_lm import generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm-jp-eval-mlx")


def compute_char_f1(ref: str, gen: str) -> float:
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


def format_jcommonsenseqa(item) -> tuple[str, str]:
    prompt = f"質問: {item['question']}\n選択肢:\n"
    prompt += f"- 0: {item['choice0']}\n"
    prompt += f"- 1: {item['choice1']}\n"
    prompt += f"- 2: {item['choice2']}\n"
    prompt += f"- 3: {item['choice3']}\n"
    prompt += f"- 4: {item['choice4']}\n"
    prompt += "回答は選択肢の番号（0, 1, 2, 3, 4）のみで答えてください。\n回答: "
    return prompt, str(item['label'])


def format_jnli(item) -> tuple[str, str]:
    premise = item.get('premise', item.get('sentence1', ''))
    hypothesis = item.get('hypothesis', item.get('sentence2', ''))
    prompt = f"前提: {premise}\n仮説: {hypothesis}\n"
    prompt += "前提と仮説の関係は、含意（entailment）、矛盾（contradiction）、中立（neutral）のどれですか？\n"
    prompt += "回答は「含意」、「矛盾」、「中立」のいずれかで答えてください。\n回答: "
    
    label_map = {
        "entailment": "含意",
        "contradiction": "矛盾",
        "neutral": "中立",
        0: "含意",
        1: "矛盾",
        2: "中立"
    }
    lbl = item.get('label', '')
    if isinstance(lbl, int) and 0 <= lbl < 3:
        expected = ["含意", "矛盾", "中立"][lbl]
    else:
        expected = label_map.get(lbl, "中立")
    return prompt, expected


def format_jsquad(item) -> tuple[str, str]:
    context = item.get('context', '')
    question = item.get('question', '')
    prompt = f"文脈: {context}\n質問: {question}\n"
    prompt += "質問に対する回答を文脈から抽出して短く答えてください。\n回答: "
    
    answers = item.get('answers', {})
    expected = ""
    if isinstance(answers, dict):
        texts = answers.get('text', [])
        if texts:
            expected = texts[0]
    elif isinstance(answers, list) and answers:
        first_ans = answers[0]
        if isinstance(first_ans, dict):
            expected = first_ans.get('text', '')
        else:
            expected = str(first_ans)
    return prompt, expected


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
