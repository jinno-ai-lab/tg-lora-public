"""Shared evaluation utilities for TG-LoRA eval scripts.

Provides common functions used across downstream and LLM-JP evaluation
scripts (both PyTorch and MLX variants) to eliminate code duplication.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_char_f1(ref: str, gen: str) -> float:
    """Character-level F1 between reference and generated text."""
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


def check_json_validity(gen: str, expected_keys: list[str]) -> tuple[bool, bool]:
    """Check whether a generated string is valid JSON with expected keys.

    Returns (is_valid_json, has_expected_keys).
    """
    gen_clean = gen.strip()
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
        start = gen.find("{")
        end = gen.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(gen[start : end + 1])
                is_valid = True
                has_keys = True
                if expected_keys and isinstance(parsed, dict):
                    has_keys = all(k in parsed for k in expected_keys)
                return is_valid, has_keys
            except json.JSONDecodeError:
                pass
        return False, False
