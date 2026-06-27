"""Batched generation + scoring for the JSON-extraction quality metric.

Used by the TG-LoRA cycle trainer (full-eval checkpoint) and by standalone
analysis scripts. Generation is batched because the hybrid-attention fast path
(fla / causal-conv1d) is not installed, so per-token decode is slow.

Reuses ``src/eval/eval_json_extraction.score_json_extraction`` for scoring.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from src.eval.eval_json_extraction import score_json_extraction

# Tokens that terminate an assistant turn; generation is cut at the first match.
_END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _prompt_from_record(rec: dict) -> str:
    """Everything up to and including the assistant header — the generation prompt."""
    return rec["text"].split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"


def generate_predictions(
    model,
    tokenizer,
    records: list[dict],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 96,
    device: Any = None,
) -> list[str]:
    """Run batched greedy generation and return the decoded predictions.

    Temporarily switches the tokenizer to left-padding (required for batched
    generation) and restores the original side afterwards. Sets model to eval
    mode and restores train mode afterwards.
    """
    if device is None:
        device = next(model.parameters()).device

    prompts = [_prompt_from_record(r) for r in records]

    prev_side = tokenizer.padding_side
    prev_pad = tokenizer.pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    was_training = getattr(model, "training", False)
    model.eval()

    predictions: list[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = out[:, enc["input_ids"].shape[1]:]
            for gen in tokenizer.batch_decode(gen_ids, skip_special_tokens=False):
                for tok in _END_TOKENS:
                    if tok in gen:
                        gen = gen.split(tok)[0]
                predictions.append(gen.strip())
    finally:
        tokenizer.padding_side = prev_side
        tokenizer.pad_token = prev_pad
        if was_training:
            model.train()

    return predictions


def generate_and_score_json(
    model,
    tokenizer,
    records: list[dict],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 96,
    device: Any = None,
    n_preview: int = 0,
) -> dict:
    """Generate predictions for ``records`` and return aggregate JSON scores.

    Each record must have a ``completion`` field holding the gold JSON string.
    Returns a dict of mean metrics (valid, strict_valid, type_correct,
    field_f1, exact_match, combined) plus, if ``n_preview > 0``, a
    ``_preview`` list of ``{prompt, gold, pred}`` dicts for the first n records.
    """
    predictions = generate_predictions(
        model,
        tokenizer,
        records,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    golds = [json.loads(r["completion"]) for r in records]
    scores = score_json_extraction(predictions, golds)
    if n_preview:
        scores["_preview"] = [
            {"prompt": r["prompt"], "gold": r["completion"], "pred": predictions[k]}
            for k, r in enumerate(records[:n_preview])
        ]
    return scores


if __name__ == "__main__":
    # Quick smoke: score the test set with whatever model is given via env, else
    # just validate the gold self-consistency path (no model needed).
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    test_path = Path("data/jsonex_test.jsonl")
    records = [json.loads(line) for line in open(test_path) if line.strip()]
    model_id = os.environ.get("MODEL_ID")
    if not model_id:
        print("Set MODEL_ID to run generation; skipping (no model).")
        sys.exit(0)
    from omegaconf import OmegaConf
    from src.model.load_model import load_base_model, load_tokenizer

    cfg = OmegaConf.create(
        {
            "model": {
                "name_or_path": model_id,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bf16",
                "dtype": "bf16",
            },
            "training": {"gradient_checkpointing": False},
        }
    )
    tok = load_tokenizer(cfg)
    mdl = load_base_model(cfg)
    s = generate_and_score_json(
        mdl, tok, records[:32], batch_size=8, max_new_tokens=96, n_preview=2
    )
    print({k: round(v, 3) for k, v in s.items() if not k.startswith("_")})
    for p in s.get("_preview", []):
        print("NL:  ", p["prompt"])
        print("GOLD:", p["gold"])
        print("PRED:", p["pred"][:200])
