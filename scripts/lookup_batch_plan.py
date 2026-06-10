#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.training.deterministic_batch_plan import (
    load_deterministic_batch_plan_manifest, resolve_record_for_sample_key,
    resolve_records_for_batch_key)
from src.training.trajectory_delta_artifact import \
    load_trajectory_delta_artifact


def _resolve_payload(
    *,
    manifest_path: str | Path | None,
    artifact_path: str | Path | None,
    sample_key: str | None,
    batch_key: str | None,
) -> dict[str, Any]:
    artifact = None
    metadata = None
    if artifact_path is not None:
        artifact = load_trajectory_delta_artifact(artifact_path)
        metadata = artifact.metadata
        if manifest_path is None:
            manifest_path = metadata.batch_plan_manifest
        if sample_key is None and batch_key is None:
            if len(metadata.batch_keys) == 1:
                batch_key = metadata.batch_keys[0]
            elif len(metadata.sample_keys) == 1:
                sample_key = metadata.sample_keys[0]

    if manifest_path is None:
        raise ValueError("manifest path is required unless artifact metadata includes it")

    manifest = load_deterministic_batch_plan_manifest(manifest_path)
    payload: dict[str, Any] = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "dataset_path": manifest.dataset_path,
        "epoch_batch_plan_key": manifest.epoch_batch_plan_key,
    }
    if metadata is not None:
        payload["artifact"] = {
            "path": str(Path(artifact_path).resolve()) if artifact_path is not None else None,
            "metadata": metadata.to_dict(),
        }
        if metadata.batch_keys:
            payload["artifact_batch_lookups"] = [
                {
                    "batch_key": key,
                    "locator": (
                        None
                        if manifest.batch_locator_by_key(key) is None
                        else manifest.batch_locator_by_key(key).__dict__
                    ),
                    "records": resolve_records_for_batch_key(manifest, key),
                }
                for key in metadata.batch_keys
            ]
        if metadata.sample_keys:
            payload["artifact_sample_lookups"] = [
                {
                    "sample_key": key,
                    "locator": (
                        None
                        if manifest.sample_locator_by_key(key) is None
                        else manifest.sample_locator_by_key(key).__dict__
                    ),
                    "record": resolve_record_for_sample_key(manifest, key),
                }
                for key in metadata.sample_keys
            ]

    if sample_key is not None:
        locator = manifest.sample_locator_by_key(sample_key)
        payload["sample_lookup"] = {
            "sample_key": sample_key,
            "locator": None if locator is None else locator.__dict__,
            "record": resolve_record_for_sample_key(manifest, sample_key),
        }

    if batch_key is not None:
        batch_locator = manifest.batch_locator_by_key(batch_key)
        payload["batch_lookup"] = {
            "batch_key": batch_key,
            "locator": None if batch_locator is None else batch_locator.__dict__,
            "records": resolve_records_for_batch_key(manifest, batch_key),
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve deterministic batch-plan keys and trajectory delta artifacts back to original dataset records"
    )
    parser.add_argument("--manifest", help="Path to batch_plan_manifest.json")
    parser.add_argument("--artifact", help="Path to a trajectory delta artifact (.pt)")
    parser.add_argument("--sample-key", help="Sample key to resolve")
    parser.add_argument("--batch-key", help="Batch key to resolve")
    parser.add_argument("--output", "-o", help="Optional JSON output path")
    args = parser.parse_args()

    if not args.manifest and not args.artifact:
        parser.error("provide --manifest or --artifact")
    if not args.sample_key and not args.batch_key and not args.artifact:
        parser.error("provide --sample-key or --batch-key when using --manifest only")

    payload = _resolve_payload(
        manifest_path=args.manifest,
        artifact_path=args.artifact,
        sample_key=args.sample_key,
        batch_key=args.batch_key,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)