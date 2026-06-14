"""MLX generation + scoring runner for the JSON-extraction gold metric.

Mirror of src/eval/jsonex_generation.py, swapping torch.generate for
mlx_lm.generate. Reuses the pure prompt builder (build_prompt_prefix) and the
pure scorer (score_json_extraction) so the MLX gold metric is identical to the
PyTorch-side metric used by the analyzer.
"""

from __future__ import annotations

import json

from mlx_lm import generate

from src.eval.eval_json_extraction import score_json_extraction

_ASSISTANT_MARKER = "<|im_start|>assistant"
_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>")


def build_prompt_prefix(record: dict) -> str:
    """ChatML prompt up to (and including) the assistant turn header.

    Inlined (rather than imported from src.eval.jsonex_generation) to keep the
    MLX training process free of any torch dependency.
    """
    text = record.get("text", "")
    if _ASSISTANT_MARKER in text:
        return text.split(_ASSISTANT_MARKER)[0] + _ASSISTANT_MARKER + "\n"
    return text


def generate_json_completions_mlx(
    model,
    tokenizer,
    records: list[dict],
    *,
    max_examples: int | None = None,
    max_tokens: int = 128,
) -> tuple[list[str], list[dict]]:
    """Greedy-generate completions for JSON-extraction records (MLX)."""
    if max_examples is not None:
        records = records[:max_examples]

    predictions: list[str] = []
    golds: list[dict] = []
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        for record in records:
            prompt = build_prompt_prefix(record)
            gen = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
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


def evaluate_json_extraction_run_mlx(
    model,
    tokenizer,
    records: list[dict],
    *,
    max_examples: int | None = None,
    max_tokens: int = 128,
) -> dict[str, float]:
    """Generate then score. Returns the aggregate gold_* dict (empty if no records)."""
    predictions, golds = generate_json_completions_mlx(
        model,
        tokenizer,
        records,
        max_examples=max_examples,
        max_tokens=max_tokens,
    )
    return score_json_extraction(predictions, golds)
