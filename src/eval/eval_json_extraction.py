"""Evaluation for the JSON-extraction domain task.

Scores model generations against gold typed-JSON records. Designed as the
quality metric for the TG-LoRA efficiency experiment: instead of perplexity,
we measure whether the model actually produces correct structured output.

Metrics (all in [0, 1]):
  validity_rate     — fraction of outputs that parse as valid JSON
  strict_validity   — parses without any prose/markdown extraction (clean output)
  type_accuracy     — correct "type" field
  field_f1          — mean per-field exact match (string/numeric aware)
  exact_match       — all fields correct
  computed_accuracy — fraction of COMPUTED (arithmetic) fields correct, averaged
                      ONLY over records whose type has one (meeting/transaction).
                      Person records have no computed field and are excluded.
  combined_score    — 0.3*validity + 0.2*type + 0.5*field_f1
"""

from __future__ import annotations

import json
import re
from typing import Any

# Expected fields per record type. meeting/transaction each carry one
# COMPUTED field (duration_minutes, total_cost) that requires arithmetic —
# the lever that keeps a strong base model below ceiling for many cycles.
SCHEMA_FIELDS: dict[str, set[str]] = {
    "meeting": {"type", "attendee", "date", "start", "end", "location",
                "priority", "duration_minutes"},
    "person": {"type", "name", "role", "department", "contact"},
    "transaction": {"type", "item", "quantity", "unit_price", "total_cost",
                    "counterparty"},
}

# Computed (arithmetic) fields per type — the discriminating difficulty lever.
# `computed_accuracy` reports the fraction of these the model gets right.
COMPUTED_FIELDS: dict[str, set[str]] = {
    "meeting": {"duration_minutes"},
    "transaction": {"total_cost"},
}


def extract_json(text: str) -> tuple[dict | None, bool]:
    """Extract a JSON object from text.

    Returns (parsed_dict_or_None, was_strict).
    `was_strict` is True if the entire text (stripped) parsed directly — i.e.,
    the model produced clean JSON with no surrounding prose.
    """
    stripped = text.strip()
    # Remove trailing assistant token if present
    for tok in ("<|im_end|>", "</s>", "<|eot_id|>"):
        if stripped.endswith(tok):
            stripped = stripped[: -len(tok)].strip()

    # Strict: whole text is the JSON
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass

    # Lenient: extract first {...} block (strip markdown fences, prose)
    fenced = re.sub(r"```(?:json)?\s*", "", stripped)
    fence_cleaned = fenced.replace("```", "")
    match = re.search(r"\{.*\}", fence_cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj, False
        except json.JSONDecodeError:
            pass
    return None, False


def _field_match(pred: Any, gold: Any) -> bool:
    """Type-aware field comparison.

    Numeric fields (e.g. total_cost, duration_minutes) compare as numbers with
    a small tolerance; thousands separators / whitespace in a string prediction
    ("432,000") are tolerated so a formatting quirk isn't scored as a math error.
    """
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        try:
            if isinstance(pred, str):
                pred = pred.replace(",", "").strip()
            return abs(float(pred) - float(gold)) < 1e-6
        except (TypeError, ValueError):
            return False
    # Strings: exact match
    return str(pred).strip() == str(gold).strip()


def score_single(prediction: str, gold: dict) -> dict[str, float]:
    """Score a single prediction against gold. Returns per-example metrics.

    ``computed_accuracy`` is None when the gold type has no computed field
    (e.g. person); such examples are excluded from the aggregate so the metric
    reflects only the arithmetic-bearing records.
    """
    obj, was_strict = extract_json(prediction)
    result: dict[str, Any] = {
        "valid": 0.0,
        "strict_valid": 0.0,
        "type_correct": 0.0,
        "field_f1": 0.0,
        "exact_match": 0.0,
        "computed_accuracy": None,
    }
    if obj is not None:
        result["valid"] = 1.0
        if was_strict:
            result["strict_valid"] = 1.0

    gold_type = gold.get("type")
    if obj is not None and obj.get("type") == gold_type:
        result["type_correct"] = 1.0

    if obj is not None:
        fields = SCHEMA_FIELDS.get(gold_type, set(gold.keys()))
        matched = sum(1 for f in fields if f in obj and _field_match(obj.get(f), gold.get(f)))
        result["field_f1"] = matched / len(fields) if fields else 0.0
        result["exact_match"] = 1.0 if matched == len(fields) else 0.0

    # Computed (arithmetic) fields in isolation — the graded signal.
    # Only types WITH a computed field contribute; others stay None (excluded).
    comp = COMPUTED_FIELDS.get(gold_type, set())
    if comp:
        c_matched = sum(
            1 for f in comp
            if obj is not None and f in obj and _field_match(obj.get(f), gold.get(f))
        )
        result["computed_accuracy"] = c_matched / len(comp)

    result["combined"] = (
        0.3 * result["valid"] + 0.2 * result["type_correct"] + 0.5 * result["field_f1"]
    )
    return result


def score_json_extraction(
    predictions: list[str], golds: list[dict]
) -> dict[str, float]:
    """Aggregate scoring across a set of predictions/golds.

    ``computed_accuracy`` is averaged only over records whose type has a
    computed field (meeting/transaction); person records (None) are excluded.
    """
    assert len(predictions) == len(golds), "length mismatch"
    if not predictions:
        return {}
    keys = ["valid", "strict_valid", "type_correct", "field_f1", "exact_match", "combined"]
    totals = {k: 0.0 for k in keys}
    comp_sum = 0.0
    comp_n = 0
    for pred, gold in zip(predictions, golds):
        s = score_single(pred, gold)
        for k in keys:
            totals[k] += s[k]
        ca = s["computed_accuracy"]
        if ca is not None:
            comp_sum += ca
            comp_n += 1
    n = len(predictions)
    out = {k: v / n for k, v in totals.items()}
    out["computed_accuracy"] = (comp_sum / comp_n) if comp_n else 0.0
    return out


if __name__ == "__main__":
    # Self-test: gold completions should score perfectly
    from pathlib import Path

    data_dir = Path("data")
    for split in ["train", "valid", "test"]:
        path = data_dir / f"jsonex_{split}.jsonl"
        if not path.exists():
            continue
        preds, golds = [], []
        for line in open(path):
            r = json.loads(line)
            preds.append(r["completion"])
            golds.append(json.loads(r["completion"]))
        scores = score_json_extraction(preds, golds)
        print(f"{split}: " + "  ".join(f"{k}={v:.3f}" for k, v in scores.items()))
