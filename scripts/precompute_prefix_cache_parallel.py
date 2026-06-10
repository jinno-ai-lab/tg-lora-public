#!/usr/bin/env python
"""Offline multi-GPU prefix-cache precompute.

This intentionally uses one process per GPU with rank-sharded data instead of
DDP collectives. Prefix cache construction is forward-only and embarrassingly
parallel, so NCCL setup and all-reduce synchronization would add brittleness
without improving throughput.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.data import Subset
from transformers import AutoConfig

from src.data.build_seed_dataset import load_dataset
from src.model.load_model import (apply_lora, get_input_device,
                                  load_base_model, load_tokenizer)
from src.tg_lora.prefix_feature_cache import (
    build_prefix_feature_cache_metadata, build_prefix_feature_dataset,
    compute_prefix_feature_shard_ranges, get_prefix_feature_cache_path,
    merge_prefix_feature_cache_shards, resolve_prefix_feature_cache_seed,
    save_prefix_feature_dataset)
from src.training.config_schema import TGLoRAConfig, load_and_validate_config
from src.utils.io import save_json


@dataclass
class ShardTask:
    dataset_label: str
    dataset_path: str
    start_idx: int
    end_idx: int
    shard_path: str


@dataclass
class WorkerConfig:
    config_path: str
    device: str
    batch_size: int
    max_seq_len: int
    split_layer_idx: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    tasks: list[ShardTask]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute prefix-feature caches across multiple GPUs by partitioning "
            "datasets into independent shards and merging them into the canonical cache blob."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml",
        help="TG-LoRA config with prefix_feature_cache_experimental enabled",
    )
    parser.add_argument(
        "--datasets",
        default="auto",
        help="Comma-separated labels from {train,valid_quick,valid_full}; 'auto' uses enabled config flags",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Comma-separated CUDA devices like 'cuda:0,cuda:1' or bare indices '0,1'; 'auto' uses all visible GPUs",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Override final cache directory (defaults to training.prefix_feature_cache_dir)",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Where to write the precompute summary JSON (defaults under cache dir)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the final merged cache already exists",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep intermediate shard files after merge",
    )
    return parser.parse_args()


def _normalize_devices(devices_arg: str) -> list[str]:
    if devices_arg == "auto":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA is required for parallel prefix-cache precompute")
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    devices: list[str] = []
    for raw in devices_arg.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.isdigit():
            devices.append(f"cuda:{token}")
        elif token.startswith("cuda:"):
            devices.append(token)
        else:
            raise ValueError(f"Unsupported device token: {token}")
    if not devices:
        raise ValueError("No CUDA devices were selected")
    return devices


def _selected_labels(cfg: TGLoRAConfig, datasets_arg: str) -> list[str]:
    valid = ["train", "valid_quick", "valid_full"]
    if datasets_arg == "auto":
        labels = [
            label
            for label in valid
            if getattr(cfg.training, f"prefix_feature_cache_{label}")
        ]
    else:
        labels = [label.strip() for label in datasets_arg.split(",") if label.strip()]
    invalid = sorted(set(labels) - set(valid))
    if invalid:
        raise ValueError(f"Unsupported dataset labels: {invalid}")
    return labels


def _resolve_split_layer(cfg: TGLoRAConfig) -> int:
    if not cfg.training.prefix_feature_cache_experimental:
        raise ValueError("parallel precompute requires prefix_feature_cache_experimental=true")
    if cfg.training.trainable_lora_scope != "last_25_percent":
        raise ValueError(
            "parallel precompute currently requires training.trainable_lora_scope=last_25_percent"
        )
    if cfg.lora.dropout != 0.0:
        raise ValueError("parallel precompute requires lora.dropout=0.0 for deterministic prefix features")

    config = AutoConfig.from_pretrained(cfg.model.name_or_path)
    text_config = getattr(config, "text_config", config)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("Unable to resolve num_hidden_layers from model config")
    suffix_layers = max(1, math.ceil(num_layers * 0.25))
    return num_layers - suffix_layers


def _dataset_path(cfg: TGLoRAConfig, label: str) -> str:
    mapping = {
        "train": cfg.data.train_path,
        "valid_quick": cfg.data.valid_quick_path,
        "valid_full": cfg.data.valid_full_path,
    }
    return mapping[label]


def _count_records(path: str) -> int:
    count = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _build_worker_configs(
    cfg: TGLoRAConfig,
    config_path: Path,
    labels: list[str],
    devices: list[str],
    split_layer_idx: int,
    cache_dir: Path,
    shard_root: Path,
    *,
    force_rebuild: bool,
) -> tuple[list[WorkerConfig], dict[str, dict[str, Any]]]:
    per_rank_tasks: list[list[ShardTask]] = [[] for _ in devices]
    dataset_plan: dict[str, dict[str, Any]] = {}

    for label in labels:
        dataset_path = _dataset_path(cfg, label)
        metadata = build_prefix_feature_cache_metadata(
            dataset_path=dataset_path,
            model_name=cfg.model.name_or_path,
            seed=resolve_prefix_feature_cache_seed(
                cfg.experiment.seed,
                share_across_seeds=cfg.training.prefix_feature_cache_share_across_seeds,
            ),
            max_seq_len=cfg.data.max_seq_len,
            split_layer_idx=split_layer_idx,
            lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            lora_target_modules=cfg.lora.target_modules,
            trainable_lora_scope=cfg.training.trainable_lora_scope,
        )
        final_cache_path = get_prefix_feature_cache_path(cache_dir, metadata)
        total_examples = _count_records(dataset_path)
        skip_existing = final_cache_path.exists() and not force_rebuild
        shard_paths: list[str] = []

        if not skip_existing:
            shard_ranges = compute_prefix_feature_shard_ranges(total_examples, len(devices))
            for rank, (start_idx, end_idx) in enumerate(shard_ranges):
                shard_path = shard_root / label / f"rank_{rank}.pt"
                shard_paths.append(str(shard_path))
                per_rank_tasks[rank].append(
                    ShardTask(
                        dataset_label=label,
                        dataset_path=dataset_path,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        shard_path=str(shard_path),
                    )
                )

        dataset_plan[label] = {
            "dataset_path": dataset_path,
            "metadata": metadata,
            "cache_path": str(final_cache_path),
            "total_examples": total_examples,
            "skip_existing": skip_existing,
            "shard_paths": shard_paths,
        }

    worker_cfgs = [
        WorkerConfig(
            config_path=str(config_path),
            device=device,
            batch_size=cfg.training.batch_size,
            max_seq_len=cfg.data.max_seq_len,
            split_layer_idx=split_layer_idx,
            num_workers=int(cfg.training.prefix_feature_cache_num_workers),
            pin_memory=bool(cfg.training.prefix_feature_cache_pin_memory),
            persistent_workers=bool(cfg.training.prefix_feature_cache_persistent_workers),
            prefetch_factor=cfg.training.prefix_feature_cache_prefetch_factor,
            tasks=tasks,
        )
        for device, tasks in zip(devices, per_rank_tasks, strict=True)
    ]
    return worker_cfgs, dataset_plan


def _worker(rank: int, worker_cfgs: list[WorkerConfig]) -> None:
    worker = worker_cfgs[rank]
    if not worker.tasks:
        return

    device = torch.device(worker.device)
    torch.cuda.set_device(device)

    cfg = OmegaConf.load(worker.config_path)
    cfg.model.device = worker.device
    cfg.model.device_map = None

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    input_device = get_input_device(model)

    for task in worker.tasks:
        if task.start_idx >= task.end_idx:
            continue

        dataset = load_dataset(task.dataset_path, tokenizer, worker.max_seq_len)
        subset = Subset(dataset, range(task.start_idx, task.end_idx))
        started = time.perf_counter()
        cached = build_prefix_feature_dataset(
            model,
            subset,
            batch_size=worker.batch_size,
            device=input_device,
            split_layer_idx=worker.split_layer_idx,
            num_workers=worker.num_workers,
            pin_memory=worker.pin_memory,
            persistent_workers=worker.persistent_workers,
            prefetch_factor=worker.prefetch_factor,
        )
        build_seconds = time.perf_counter() - started

        shard_path = Path(task.shard_path)
        shard_metadata = {
            "dataset_label": task.dataset_label,
            "shard_rank": rank,
            "shard_start_idx": task.start_idx,
            "shard_end_idx": task.end_idx,
            "build_seconds": round(build_seconds, 3),
        }
        save_prefix_feature_dataset(cached, shard_path, metadata=shard_metadata)
        save_json(
            {
                "rank": rank,
                "device": worker.device,
                "dataset_label": task.dataset_label,
                "shard_path": str(shard_path),
                "start_idx": task.start_idx,
                "end_idx": task.end_idx,
                "examples": len(cached),
                "cache_gib": round(cached.total_bytes / 1024**3, 6),
                "build_seconds": round(build_seconds, 3),
            },
            shard_path.with_suffix(".json"),
        )

    del model
    torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    validated = load_and_validate_config(config_path)
    if not isinstance(validated, TGLoRAConfig):
        raise ValueError("parallel precompute requires a TG-LoRA config")

    devices = _normalize_devices(args.devices)
    labels = _selected_labels(validated, args.datasets)
    split_layer_idx = _resolve_split_layer(validated)

    cache_dir = Path(args.cache_dir or validated.training.prefix_feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else cache_dir / "parallel_precompute_summary.json"
    )
    shard_root = cache_dir / ".parallel_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    worker_cfgs, dataset_plan = _build_worker_configs(
        validated,
        config_path,
        labels,
        devices,
        split_layer_idx,
        cache_dir,
        shard_root,
        force_rebuild=args.force_rebuild,
    )

    overall_started = time.perf_counter()
    if any(not plan["skip_existing"] for plan in dataset_plan.values()):
        mp.spawn(_worker, args=(worker_cfgs,), nprocs=len(worker_cfgs), join=True)

    dataset_summaries: dict[str, Any] = {}
    for label, plan in dataset_plan.items():
        final_cache_path = Path(plan["cache_path"])
        if plan["skip_existing"]:
            dataset_summaries[label] = {
                "dataset_path": plan["dataset_path"],
                "cache_path": str(final_cache_path),
                "source": "existing",
                "total_examples": plan["total_examples"],
            }
            continue

        shard_paths = [Path(path) for path in plan["shard_paths"] if Path(path).exists()]
        merge_started = time.perf_counter()
        merge_prefix_feature_cache_shards(
            shard_paths,
            final_cache_path,
            metadata=plan["metadata"],
        )
        merge_seconds = time.perf_counter() - merge_started

        shard_details = []
        worker_seconds = []
        total_examples = 0
        total_cache_gib = 0.0
        for shard_path in shard_paths:
            shard_summary = json.loads(shard_path.with_suffix(".json").read_text())
            shard_details.append(shard_summary)
            worker_seconds.append(float(shard_summary["build_seconds"]))
            total_examples += int(shard_summary["examples"])
            total_cache_gib += float(shard_summary["cache_gib"])

        dataset_summaries[label] = {
            "dataset_path": plan["dataset_path"],
            "cache_path": str(final_cache_path),
            "source": "built_parallel",
            "total_examples": total_examples,
            "split_layer_idx": split_layer_idx,
            "devices": devices,
            "num_shards": len(shard_paths),
            "worker_build_seconds": worker_seconds,
            "max_worker_build_seconds": round(max(worker_seconds), 3) if worker_seconds else 0.0,
            "sum_worker_build_seconds": round(sum(worker_seconds), 3),
            "merge_seconds": round(merge_seconds, 3),
            "approx_dataset_wall_seconds": round((max(worker_seconds) if worker_seconds else 0.0) + merge_seconds, 3),
            "approx_parallel_speedup_vs_sequential": round(
                (sum(worker_seconds) / ((max(worker_seconds) if worker_seconds else 0.0) + merge_seconds))
                if worker_seconds and ((max(worker_seconds) if worker_seconds else 0.0) + merge_seconds) > 0
                else 0.0,
                3,
            ),
            "cache_gib": round(total_cache_gib, 6),
            "shards": shard_details,
        }

    overall_seconds = time.perf_counter() - overall_started
    summary = {
        "config": str(config_path),
        "devices": devices,
        "split_layer_idx": split_layer_idx,
        "datasets": dataset_summaries,
        "overall_wall_seconds": round(overall_seconds, 3),
    }
    save_json(summary, summary_path)

    if not args.keep_shards:
        shutil.rmtree(shard_root, ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()