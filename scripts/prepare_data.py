"""Prepare downloaded datasets into train/valid/test splits in JSONL format.

Converts raw datasets into the format expected by the training pipeline:
- text: full prompt+completion concatenated
- prompt: instruction portion
- completion: response portion

Usage:
    python scripts/prepare_data.py [--source dolly|capybara] [--train-size 3000] [--valid-size 300] [--test-size 500]
"""

import argparse
import json
import logging
import random
from pathlib import Path

# NOTE: ``src.data.filter_dataset`` lives in the private data pipeline that is
# stripped from this public mirror. Importing it at module scope makes even
# ``--help`` fail here; it is imported lazily inside ``prepare_dataset`` so the
# argparse ``--help`` path works with only the stdlib (mirrors the existing lazy
# ``load_tokenizer`` import in ``_load_audit_tokenizer``).
logger = logging.getLogger("tg-lora")

CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

CHAT_TEMPLATE_WITH_CONTEXT = """<|im_start|>user
{instruction}

Context: {context}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""


# NOTE: Qwen3.5 uses the same ChatML template as Qwen2.5.
# The model processes <|im_start|>user / <|im_start|>assistant tokens natively.
# For non-thinking mode during SFT, include only the final response in the
# assistant turn (no thinking content).


def format_record(raw: dict) -> dict:
    """Convert raw record to training format."""
    instruction = raw.get("instruction", "")
    context = raw.get("context", "")
    response = raw.get("response", "")

    if context.strip():
        text = CHAT_TEMPLATE_WITH_CONTEXT.format(
            instruction=instruction, context=context, response=response
        )
    else:
        text = CHAT_TEMPLATE.format(instruction=instruction, response=response)

    return {
        "text": text,
        "prompt": instruction,
        "completion": response,
        "category": raw.get("category", "general"),
    }


def load_raw(path: Path) -> list[dict]:
    """Load raw JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records to {path}")


def _load_audit_tokenizer(model_name: str):
    from omegaconf import OmegaConf

    from src.model.load_model import load_tokenizer

    cfg = OmegaConf.create({"model": {"name_or_path": model_name}})
    return load_tokenizer(cfg)


def _write_token_audit_report(report: dict[str, object], output_dir: Path) -> None:
    json_path = output_dir / "token_audit_report.json"
    md_path = output_dir / "token_audit_report.md"

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Token Audit Report",
        "",
        f"- model: {report['model_name']}",
        f"- max_seq_len: {report['max_seq_len']}",
        f"- min_supervised_tokens: {report['min_supervised_tokens']}",
        f"- train_on_prompt: {report['train_on_prompt']}",
        "",
    ]

    for split_name in ("train", "valid", "test"):
        split = report["splits"][split_name]
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- input_records: {split['input_records']}",
                f"- kept_records: {split['kept_records']}",
                f"- removed_below_min_supervised_tokens: {split['removed_below_min_supervised_tokens']}",
                f"- all_masked_records: {split['all_masked_records']}",
                f"- prompt_dominant_records: {split['prompt_dominant_records']}",
                f"- avg_supervised_tokens_kept: {split['avg_supervised_tokens_kept']:.2f}",
                "",
            ]
        )
        issues = split.get("issue_examples", [])
        if issues:
            lines.append("Issue examples:")
            for issue in issues:
                lines.append(
                    f"- idx={issue['index']} supervised={issue['supervised_tokens']} total={issue['total_tokens']} ratio={issue['prompt_token_ratio']}: {issue['text_preview']}"
                )
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved token audit reports to {json_path} and {md_path}")


def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    train_size: int = 3000,
    valid_size: int = 300,
    test_size: int = 500,
    seed: int = 42,
    *,
    audit_model_name: str | None = "Qwen/Qwen3.5-9B",
    audit_max_seq_len: int = 1024,
    min_supervised_tokens: int = 1,
    train_on_prompt: bool = False,
) -> dict[str, object] | None:
    """Prepare train/valid/test splits from raw data."""
    from src.data.filter_dataset import filter_records

    raw_records = load_raw(source_path)
    logger.info(f"Loaded {len(raw_records)} raw records from {source_path}")

    # Format all records
    formatted = [format_record(r) for r in raw_records]

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(formatted)

    total_needed = train_size + valid_size + test_size
    if len(formatted) < total_needed:
        logger.warning(
            f"Only {len(formatted)} records available, "
            f"requested {total_needed}. Using all."
        )
        valid_size = min(valid_size, len(formatted) // 10)
        test_size = min(test_size, len(formatted) // 10)
        train_size = len(formatted) - valid_size - test_size

    valid_records = formatted[:valid_size]
    test_records = formatted[valid_size : valid_size + test_size]
    train_records = formatted[valid_size + test_size : valid_size + test_size + train_size]

    audit_report: dict[str, object] | None = None
    if audit_model_name is not None:
        audit_tokenizer = _load_audit_tokenizer(audit_model_name)
        train_records, train_summary = filter_records(
            train_records,
            min_length=0,
            max_length=10**9,
            min_quality_score=0.0,
            required_fields=["text", "completion"],
            tokenizer=audit_tokenizer,
            max_seq_len=audit_max_seq_len,
            train_on_prompt=train_on_prompt,
            min_supervised_tokens=min_supervised_tokens,
        )
        valid_records, valid_summary = filter_records(
            valid_records,
            min_length=0,
            max_length=10**9,
            min_quality_score=0.0,
            required_fields=["text", "completion"],
            tokenizer=audit_tokenizer,
            max_seq_len=audit_max_seq_len,
            train_on_prompt=train_on_prompt,
            min_supervised_tokens=min_supervised_tokens,
        )
        test_records, test_summary = filter_records(
            test_records,
            min_length=0,
            max_length=10**9,
            min_quality_score=0.0,
            required_fields=["text", "completion"],
            tokenizer=audit_tokenizer,
            max_seq_len=audit_max_seq_len,
            train_on_prompt=train_on_prompt,
            min_supervised_tokens=min_supervised_tokens,
        )
        audit_report = {
            "model_name": audit_model_name,
            "max_seq_len": audit_max_seq_len,
            "min_supervised_tokens": min_supervised_tokens,
            "train_on_prompt": train_on_prompt,
            "splits": {
                "train": train_summary,
                "valid": valid_summary,
                "test": test_summary,
            },
        }
        _write_token_audit_report(audit_report, output_dir)

    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(valid_records, output_dir / "valid_quick.jsonl")
    save_jsonl(test_records, output_dir / "valid_full.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")

    # Create a small gold test set (subset of valid for task eval)
    gold_test = valid_records[:50]
    save_jsonl(gold_test, output_dir / "gold_test.jsonl")

    logger.info(
        f"Split complete: train={len(train_records)}, "
        f"valid_quick={len(valid_records)}, valid_full/test={len(test_records)}, gold_test={len(gold_test)}"
    )
    return audit_report


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--source",
        choices=["dolly", "capybara"],
        default="dolly",
        help="Source dataset to prepare",
    )
    parser.add_argument(
        "--train-size", type=int, default=3000, help="Training set size"
    )
    parser.add_argument(
        "--valid-size", type=int, default=300, help="Validation set size"
    )
    parser.add_argument(
        "--test-size", type=int, default=500, help="Test set size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"), help="Raw data directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"), help="Output directory"
    )
    parser.add_argument(
        "--audit-model-name",
        default="Qwen/Qwen3.5-9B",
        help="Model/tokenizer to use for token-aware supervision audit",
    )
    parser.add_argument(
        "--audit-max-seq-len",
        type=int,
        default=1024,
        help="Sequence length used for token-aware supervision audit",
    )
    parser.add_argument(
        "--min-supervised-tokens",
        type=int,
        default=1,
        help="Minimum supervised tokens required to keep a record during token-aware filtering",
    )
    parser.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="Audit using training.train_on_prompt=true semantics",
    )
    parser.add_argument(
        "--disable-token-audit",
        action="store_true",
        help="Skip tokenizer-based supervision audit and filtering",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    source_map = {
        "dolly": "dolly_15k.jsonl",
        "capybara": "capybara.jsonl",
    }

    source_path = args.raw_dir / source_map[args.source]
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        logger.error("Run `python scripts/download_data.py` first.")
        raise SystemExit(1)

    prepare_dataset(
        source_path=source_path,
        output_dir=args.output_dir,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        audit_model_name=None if args.disable_token_audit else args.audit_model_name,
        audit_max_seq_len=args.audit_max_seq_len,
        min_supervised_tokens=args.min_supervised_tokens,
        train_on_prompt=args.train_on_prompt,
    )


if __name__ == "__main__":
    main()
