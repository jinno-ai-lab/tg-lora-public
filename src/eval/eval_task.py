import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils.io import load_jsonl, save_jsonl


def eval_task_performance(
    model,
    tokenizer: AutoTokenizer,
    test_path: str,
    output_path: str | None = None,
    device: str | None = None,
    max_seq_len: int = 512,
    max_new_tokens: int = 256,
    metric_fn=None,
) -> dict:
    if device is None:
        from src.utils.device import detect_device
        device = str(detect_device())
    records = load_jsonl(test_path)
    was_training = model.training
    model.eval()

    predictions = []
    scores = []

    try:
        for rec in tqdm(records, desc="Task eval"):
            prompt = rec.get("prompt", "")
            expected = rec.get("completion", rec.get("output", ""))

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

            pred = {
                "prompt": prompt,
                "expected": expected,
                "predicted": completion,
            }
            predictions.append(pred)

            if metric_fn:
                score = metric_fn(expected, completion)
                scores.append(score)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    result = {
        "total": len(predictions),
    }

    if scores:
        result["mean_score"] = sum(scores) / len(scores)
        result["scores"] = scores

    if output_path:
        save_jsonl(predictions, output_path)

    return result
