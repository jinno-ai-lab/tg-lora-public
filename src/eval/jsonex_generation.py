"""Generation + scoring runner for the JSON-extraction domain task.

Thin torch-dependent wrapper around the pure scorer in eval_json_extraction.
Shared by the base-model difficulty probe (scripts/eval_base_model_json.py) and
the training loop's periodic gold evaluation (TG-LoRA Guard experiment,
design 10_guard_experiment.md §5.2). Keeping the generation logic in one place
guarantees the probe and the live gold metric score identically.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from src.eval.eval_json_extraction import score_json_extraction

_ASSISTANT_MARKER = "<|im_start|>assistant"
_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>")


def build_prompt_prefix(record: dict) -> str:
    """Reconstruct the prompt up to (and including) the assistant turn header.

    Mirrors scripts/eval_base_model_json.py: everything before ``assistant`` plus
    the ``<|im_start|>assistant\\n`` header, so generation continues the answer.
    """
    text = record.get("text", "")
    if _ASSISTANT_MARKER in text:
        return text.split(_ASSISTANT_MARKER)[0] + _ASSISTANT_MARKER + "\n"
    return text


@torch.no_grad()
def generate_json_completions(
    model,
    tokenizer,
    records: list[dict],
    *,
    max_examples: int | None = None,
    max_new_tokens: int = 128,
    device: Any | None = None,
) -> tuple[list[str], list[dict]]:
    """Greedy-generate completions for JSON-extraction records.

    Returns (predictions, golds); golds are the parsed completion dicts.
    Restores the model's training mode afterward so this is safe to call
    mid-training.
    """
    if device is None:
        device = next(model.parameters()).device
    if max_examples is not None:
        records = records[:max_examples]

    predictions: list[str] = []
    golds: list[dict] = []
    was_training = model.training
    model.eval()
    try:
        for record in records:
            prompt_text = build_prompt_prefix(record)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )
            for tok in _STOP_TOKENS:
                if tok in gen:
                    gen = gen.split(tok)[0]
            predictions.append(gen.strip())
            golds.append(json.loads(record["completion"]))
    finally:
        if was_training:
            model.train()
    return predictions, golds


def evaluate_json_extraction_run(
    model,
    tokenizer,
    records: list[dict],
    *,
    max_examples: int | None = None,
    max_new_tokens: int = 128,
    device: Any | None = None,
) -> dict[str, float]:
    """Generate then score. Returns the aggregate score dict (empty if no records)."""
    predictions, golds = generate_json_completions(
        model,
        tokenizer,
        records,
        max_examples=max_examples,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return score_json_extraction(predictions, golds)
