"""Run base (untrained) model on JSON-extraction test set to measure task difficulty.

Loads Qwen/Qwen3.5-9B in 4bit via the project's loader, generates completions for
the test prompts, and scores with eval_json_extraction. This is the go/no-go gate
for the experiment: if the base model already produces good JSON, the task is too
easy and there's no headroom to measure a learning curve against.

Generation + scoring reuse ``src.eval.json_generation.generate_and_score_json``
(the same path the TG-LoRA trainer uses), so this gate measures exactly what the
experiment measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.json_generation import generate_and_score_json
from src.model.load_model import load_base_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of test examples")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    test_path = Path("data/jsonex_test.jsonl")
    records = [json.loads(l) for l in open(test_path) if l.strip()][: args.n]
    print(f"Loaded {len(records)} test records", flush=True)

    # 4bit (bitsandbytes) requires CUDA; fall back to bf16 on MPS/CPU so the
    # difficulty gate runs on the dev Mac as well as the CUDA host.
    has_cuda = torch.cuda.is_available()
    cfg = OmegaConf.create(
        {
            "model": {
                "name_or_path": "Qwen/Qwen3.5-9B",
                "load_in_4bit": has_cuda,
                "bnb_4bit_compute_dtype": "bf16",
                "dtype": "bf16",
            },
            "training": {"gradient_checkpointing": False},
        }
    )
    print(f"Loading Qwen/Qwen3.5-9B ({'4bit' if has_cuda else 'bf16 (no CUDA)'})...", flush=True)
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)

    scores = generate_and_score_json(
        model,
        tokenizer,
        records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        n_preview=4,
    )
    preview = scores.pop("_preview", [])

    print(f"\n=== BASE MODEL JSON-EXTRACTION SCORE (test, n={len(records)}) ===")
    for k, v in scores.items():
        print(f"  {k:18s} {v:.3f}")

    for p in preview:
        print(f"\n--- {p['prompt'][:60]}... ---")
        print(f"GOLD: {p['gold']}")
        print(f"GEN:  {p['pred'][:200]}")


if __name__ == "__main__":
    main()
