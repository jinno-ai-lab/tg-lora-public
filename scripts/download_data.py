"""Download public datasets for initial validation.

Datasets:
- Dolly 15k (databricks/databricks-dolly-15k): instruction-following
- Capybara (LDJnr/Capybara): multi-turn SFT data

Usage:
    python scripts/download_data.py [--dataset dolly|capybara|all] [--output-dir data/raw]
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger("tg-lora")


def download_dolly(output_dir: Path) -> Path:
    """Download Databricks Dolly 15k dataset."""
    logger.info("Downloading databricks/databricks-dolly-15k ...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    out_path = output_dir / "dolly_15k.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            record = {
                "instruction": row["instruction"],
                "context": row["context"],
                "response": row["response"],
                "category": row["category"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Dolly 15k saved: {out_path} ({len(ds)} records)")
    return out_path


def download_capybara(output_dir: Path) -> Path:
    """Download Capybara dataset."""
    logger.info("Downloading LDJnr/Capybara ...")
    ds = load_dataset("LDJnr/Capybara", split="train")

    out_path = output_dir / "capybara.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            conversation = row.get("conversation", [])
            if not conversation:
                continue
            # Extract first turn as instruction/response
            record = {
                "instruction": conversation[0].get("input", ""),
                "response": conversation[0].get("output", ""),
                "source": "capybara",
                "turns": len(conversation),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Capybara saved: {out_path} ({len(ds)} records)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download public datasets")
    parser.add_argument(
        "--dataset",
        choices=["dolly", "capybara", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("dolly", "all"):
        download_dolly(args.output_dir)

    if args.dataset in ("capybara", "all"):
        download_capybara(args.output_dir)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
