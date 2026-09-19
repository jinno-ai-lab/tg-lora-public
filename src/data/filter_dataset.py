import logging

from src.data.build_seed_dataset import analyze_record_supervision
from src.utils.io import load_jsonl, save_jsonl

logger = logging.getLogger("tg-lora")


def filter_records(
    records: list[dict],
    *,
    min_length: int = 20,
    max_length: int = 4096,
    min_quality_score: float = 0.0,
    required_fields: list[str] | None = None,
    tokenizer=None,
    max_seq_len: int | None = None,
    train_on_prompt: bool = False,
    min_supervised_tokens: int | None = None,
) -> tuple[list[dict], dict[str, object]]:
    required_fields = required_fields or ["text"]
    token_audit_enabled = tokenizer is not None and max_seq_len is not None

    filtered = []
    supervised_sum = 0
    all_masked_count = 0
    prompt_dominant_count = 0
    below_min_supervised_count = 0
    kept_supervised_min: int | None = None
    kept_supervised_max = 0
    issue_examples: list[dict[str, object]] = []

    for idx, rec in enumerate(records):
        text = rec.get("text", "")
        if len(text) < min_length or len(text) > max_length:
            continue

        if not all(rec.get(f) for f in required_fields):
            continue

        score = rec.get(
            "quality_score", rec.get("provenance", {}).get("quality_score", 1.0)
        )
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_quality_score:
            continue

        if token_audit_enabled:
            supervision = analyze_record_supervision(
                rec,
                tokenizer,
                max_seq_len=max_seq_len,
                train_on_prompt=train_on_prompt,
            )
            supervised_tokens = int(supervision["supervised_tokens"])
            all_masked = bool(supervision["all_masked"])
            prompt_ratio = float(supervision["prompt_token_ratio"])
            all_masked_count += int(all_masked)
            prompt_dominant_count += int(prompt_ratio >= 0.9)
            if min_supervised_tokens is not None and supervised_tokens < min_supervised_tokens:
                below_min_supervised_count += 1
                if len(issue_examples) < 10:
                    issue_examples.append(
                        {
                            "index": idx,
                            "supervised_tokens": supervised_tokens,
                            "prompt_tokens": int(supervision["prompt_tokens"]),
                            "total_tokens": int(supervision["total_tokens"]),
                            "prompt_token_ratio": round(prompt_ratio, 4),
                            "text_preview": text[:160],
                        }
                    )
                continue

            supervised_sum += supervised_tokens
            kept_supervised_min = (
                supervised_tokens
                if kept_supervised_min is None
                else min(kept_supervised_min, supervised_tokens)
            )
            kept_supervised_max = max(kept_supervised_max, supervised_tokens)

        filtered.append(rec)

    summary: dict[str, object] = {
        "input_records": len(records),
        "kept_records": len(filtered),
        "removed_records": len(records) - len(filtered),
        "token_audit_enabled": token_audit_enabled,
    }
    if token_audit_enabled:
        summary.update(
            {
                "max_seq_len": max_seq_len,
                "train_on_prompt": train_on_prompt,
                "min_supervised_tokens": min_supervised_tokens,
                "all_masked_records": all_masked_count,
                "prompt_dominant_records": prompt_dominant_count,
                "removed_below_min_supervised_tokens": below_min_supervised_count,
                "avg_supervised_tokens_kept": (
                    supervised_sum / len(filtered) if filtered else 0.0
                ),
                "min_supervised_tokens_kept": kept_supervised_min or 0,
                "max_supervised_tokens_kept": kept_supervised_max,
                "issue_examples": issue_examples,
            }
        )
    return filtered, summary


def filter_dataset(
    input_path: str,
    output_path: str,
    min_length: int = 20,
    max_length: int = 4096,
    min_quality_score: float = 0.0,
    required_fields: list[str] | None = None,
    tokenizer=None,
    max_seq_len: int | None = None,
    train_on_prompt: bool = False,
    min_supervised_tokens: int | None = None,
) -> dict[str, object]:
    records = load_jsonl(input_path)
    filtered, summary = filter_records(
        records,
        min_length=min_length,
        max_length=max_length,
        min_quality_score=min_quality_score,
        required_fields=required_fields,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_prompt=train_on_prompt,
        min_supervised_tokens=min_supervised_tokens,
    )

    save_jsonl(filtered, output_path)
    logger.info(f"Filter: {len(records)} -> {len(filtered)} records -> {output_path}")
    return summary
