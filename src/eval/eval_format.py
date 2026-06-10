import json
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils.io import load_jsonl


def eval_format_compliance(
    model,
    tokenizer: AutoTokenizer,
    test_path: str,
    device: str | None = None,
    max_seq_len: int = 512,
    max_new_tokens: int = 256,
) -> dict:
    if device is None:
        from src.utils.device import detect_device
        device = str(detect_device())
    records = load_jsonl(test_path)
    was_training = model.training
    model.eval()

    results = {
        "total": 0,
        "valid_json": 0,
        "has_required_keys": 0,
        "format_scores": [],
    }

    required_keys = _infer_required_keys(records)

    try:
        for rec in tqdm(records, desc="Format eval"):
            prompt = rec.get("prompt", "")
            if not prompt:
                continue

            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            completion = tokenizer.decode(
                outputs[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            results["total"] += 1

            # JSON check
            try:
                parsed = json.loads(completion)
                results["valid_json"] += 1

                if all(k in parsed for k in required_keys):
                    results["has_required_keys"] += 1
            except json.JSONDecodeError:
                pass
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    if results["total"] > 0:
        results["json_rate"] = results["valid_json"] / results["total"]
        results["key_compliance_rate"] = results["has_required_keys"] / results["total"]
    else:
        results["json_rate"] = 0.0
        results["key_compliance_rate"] = 0.0

    return results


def _infer_required_keys(records: list[dict]) -> list[str]:
    key_counts: Counter = Counter()
    for rec in records[:50]:
        target = rec.get("completion", rec.get("output", ""))
        try:
            parsed = json.loads(target)
            if isinstance(parsed, dict):
                key_counts.update(parsed.keys())
        except (json.JSONDecodeError, TypeError):
            pass

    threshold = len(records[:50]) * 0.5
    return [k for k, c in key_counts.items() if c >= threshold]
