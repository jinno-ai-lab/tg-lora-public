from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.tg_lora.activation_cache import _get_decoder_layers
from src.utils.tensor_artifact import load_tensor_artifact

_PREFIX_FEATURE_CACHE_FORMAT_VERSION = 1


@dataclass
class PrefixFeatureExample:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    split_layer_idx: int
    position_ids: torch.Tensor | None = None


class PrefixFeatureDatasetBase(Dataset):
    @property
    def total_bytes(self) -> int:
        raise NotImplementedError


class PrefixFeatureDataset(PrefixFeatureDatasetBase):
    def __init__(self, examples: list[PrefixFeatureExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        for i, ex in enumerate(examples):
            if not isinstance(ex, PrefixFeatureExample):
                raise TypeError(
                    f"examples[{i}] must be a PrefixFeatureExample, got {type(ex).__name__}"
                )
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self._examples[idx]
        item = {
            "hidden_states": ex.hidden_states,
            "attention_mask": ex.attention_mask,
            "labels": ex.labels,
            "split_layer_idx": torch.tensor(ex.split_layer_idx, dtype=torch.long),
        }
        if ex.position_ids is not None:
            item["position_ids"] = ex.position_ids
        return item

    @property
    def total_bytes(self) -> int:
        total = 0
        for ex in self._examples:
            total += ex.hidden_states.numel() * ex.hidden_states.element_size()
            total += ex.attention_mask.numel() * ex.attention_mask.element_size()
            total += ex.labels.numel() * ex.labels.element_size()
            if ex.position_ids is not None:
                total += ex.position_ids.numel() * ex.position_ids.element_size()
        return total


class MappedPrefixFeatureDataset(PrefixFeatureDatasetBase):
    def __init__(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        split_layer_idx: int,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        if hidden_states is None:
            raise ValueError("hidden_states must not be None")
        if attention_mask is None:
            raise ValueError("attention_mask must not be None")
        if labels is None:
            raise ValueError("labels must not be None")
        if split_layer_idx < 0:
            raise ValueError(f"split_layer_idx must be non-negative, got {split_layer_idx}")
        batch_sizes = {
            "hidden_states": hidden_states.shape[0],
            "attention_mask": attention_mask.shape[0],
            "labels": labels.shape[0],
        }
        unique_sizes = set(batch_sizes.values())
        if len(unique_sizes) > 1:
            raise ValueError(
                f"Batch size mismatch: {', '.join(f'{k}={v}' for k, v in batch_sizes.items())}"
            )
        self._hidden_states = hidden_states
        self._attention_mask = attention_mask
        self._labels = labels
        self._split_layer_idx = split_layer_idx
        self._position_ids = position_ids

    def __len__(self) -> int:
        return int(self._hidden_states.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "hidden_states": self._hidden_states[idx],
            "attention_mask": self._attention_mask[idx],
            "labels": self._labels[idx],
            "split_layer_idx": torch.tensor(self._split_layer_idx, dtype=torch.long),
        }
        if self._position_ids is not None:
            item["position_ids"] = self._position_ids[idx]
        return item

    @property
    def total_bytes(self) -> int:
        total = self._hidden_states.numel() * self._hidden_states.element_size()
        total += self._attention_mask.numel() * self._attention_mask.element_size()
        total += self._labels.numel() * self._labels.element_size()
        if self._position_ids is not None:
            total += self._position_ids.numel() * self._position_ids.element_size()
        return total


def build_prefix_feature_cache_metadata(
    *,
    dataset_path: str,
    model_name: str,
    seed: int,
    max_seq_len: int,
    split_layer_idx: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    trainable_lora_scope: str,
) -> dict[str, Any]:
    data_file = Path(dataset_path)
    resolved_path = str(data_file.resolve()) if data_file.exists() else dataset_path
    size_bytes = data_file.stat().st_size if data_file.exists() else None
    mtime_ns = data_file.stat().st_mtime_ns if data_file.exists() else None
    return {
        "format_version": _PREFIX_FEATURE_CACHE_FORMAT_VERSION,
        "dataset_path": resolved_path,
        "dataset_size_bytes": size_bytes,
        "dataset_mtime_ns": mtime_ns,
        "model_name": model_name,
        "seed": seed,
        "max_seq_len": max_seq_len,
        "split_layer_idx": split_layer_idx,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "trainable_lora_scope": trainable_lora_scope,
    }


def resolve_prefix_feature_cache_seed(seed: int, *, share_across_seeds: bool) -> int:
    return 0 if share_across_seeds else seed


def get_prefix_feature_cache_path(
    cache_dir: str | Path,
    metadata: dict[str, Any],
) -> Path:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    dataset_stem = Path(str(metadata.get("dataset_path", "dataset"))).stem or "dataset"
    return Path(cache_dir) / f"{dataset_stem}_{digest}.pt"


def compute_prefix_feature_shard_ranges(
    total_examples: int,
    shard_count: int,
) -> list[tuple[int, int]]:
    if total_examples < 0:
        raise ValueError(f"total_examples must be >= 0, got {total_examples}")
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if total_examples == 0:
        return []

    active_shard_count = min(total_examples, shard_count)
    base_width, remainder = divmod(total_examples, active_shard_count)
    start = 0
    ranges: list[tuple[int, int]] = []
    for shard_idx in range(active_shard_count):
        width = base_width + (1 if shard_idx < remainder else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def _extract_prefix_feature_storage(
    dataset: PrefixFeatureDatasetBase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
    if isinstance(dataset, PrefixFeatureDataset):
        examples = dataset._examples
        if not examples:
            raise ValueError("Cannot persist an empty PrefixFeatureDataset")
        has_position_ids = all(ex.position_ids is not None for ex in examples)
        if not has_position_ids and any(ex.position_ids is not None for ex in examples):
            raise ValueError("PrefixFeatureDataset contains mixed position_ids presence")
        return (
            torch.stack([ex.hidden_states for ex in examples]),
            torch.stack([ex.attention_mask for ex in examples]),
            torch.stack([ex.labels for ex in examples]),
            int(examples[0].split_layer_idx),
            (
                torch.stack([cast(torch.Tensor, ex.position_ids) for ex in examples])
                if has_position_ids
                else None
            ),
        )

    if isinstance(dataset, MappedPrefixFeatureDataset):
        return (
            dataset._hidden_states,
            dataset._attention_mask,
            dataset._labels,
            dataset._split_layer_idx,
            dataset._position_ids,
        )

    raise TypeError(f"Unsupported prefix feature dataset type: {type(dataset)!r}")


def save_prefix_feature_dataset(
    dataset: PrefixFeatureDatasetBase,
    cache_path: str | Path,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hidden_states, attention_mask, labels, split_layer_idx, position_ids = (
        _extract_prefix_feature_storage(dataset)
    )

    blob = {
        "format_version": _PREFIX_FEATURE_CACHE_FORMAT_VERSION,
        "metadata": metadata,
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "labels": labels,
        "split_layer_idx": split_layer_idx,
        "position_ids": position_ids,
    }
    torch.save(blob, path)


def merge_prefix_feature_cache_shards(
    shard_paths: Sequence[str | Path],
    cache_path: str | Path,
    *,
    metadata: dict[str, Any],
) -> None:
    paths = [Path(path) for path in shard_paths]
    if not paths:
        raise ValueError("shard_paths must not be empty")

    blobs = [load_tensor_artifact(path) for path in paths]
    for blob in blobs:
        if blob.get("format_version") != _PREFIX_FEATURE_CACHE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported prefix feature cache format version: "
                f"{blob.get('format_version')}"
            )

    first = blobs[0]
    split_layer_idx = int(first["split_layer_idx"])
    has_position_ids = first.get("position_ids") is not None

    hidden_states = []
    attention_mask = []
    labels = []
    position_ids = []
    for blob in blobs:
        if int(blob["split_layer_idx"]) != split_layer_idx:
            raise ValueError("All shard caches must have the same split_layer_idx")
        shard_has_position_ids = blob.get("position_ids") is not None
        if shard_has_position_ids != has_position_ids:
            raise ValueError("All shard caches must agree on position_ids presence")
        hidden_states.append(blob["hidden_states"])
        attention_mask.append(blob["attention_mask"])
        labels.append(blob["labels"])
        if has_position_ids:
            position_ids.append(blob["position_ids"])

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged_blob = {
        "format_version": _PREFIX_FEATURE_CACHE_FORMAT_VERSION,
        "metadata": metadata,
        "hidden_states": torch.cat(hidden_states, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
        "labels": torch.cat(labels, dim=0),
        "split_layer_idx": split_layer_idx,
        "position_ids": torch.cat(position_ids, dim=0) if has_position_ids else None,
    }
    torch.save(merged_blob, path)


def load_prefix_feature_dataset(
    cache_path: str | Path,
    *,
    lazy: bool = False,
) -> PrefixFeatureDatasetBase:
    blob = load_tensor_artifact(cache_path, mmap=lazy)
    if blob.get("format_version") != _PREFIX_FEATURE_CACHE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported prefix feature cache format version: "
            f"{blob.get('format_version')}"
        )

    hidden_states = blob["hidden_states"]
    attention_mask = blob["attention_mask"]
    labels = blob["labels"]
    split_layer_idx = int(blob["split_layer_idx"])
    position_ids = blob.get("position_ids")

    if lazy:
        return MappedPrefixFeatureDataset(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels,
            split_layer_idx=split_layer_idx,
            position_ids=position_ids,
        )

    examples: list[PrefixFeatureExample] = []
    for idx in range(hidden_states.shape[0]):
        examples.append(
            PrefixFeatureExample(
                hidden_states=hidden_states[idx].clone(),
                attention_mask=attention_mask[idx].clone(),
                labels=labels[idx].clone(),
                split_layer_idx=split_layer_idx,
                position_ids=(
                    position_ids[idx].clone() if position_ids is not None else None
                ),
            )
        )
    return PrefixFeatureDataset(examples)


def collate_prefix_feature_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {
        "hidden_states": torch.stack([item["hidden_states"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "split_layer_idx": torch.stack([item["split_layer_idx"] for item in batch]),
    }
    if "position_ids" in batch[0]:
        out["position_ids"] = torch.stack([item["position_ids"] for item in batch])
    return out


@torch.no_grad()
def build_prefix_feature_dataset(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device | str,
    split_layer_idx: int,
    max_batches: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> PrefixFeatureDataset:
    """Precompute hidden states entering the suffix split layer for a dataset."""
    decoder_layers = _get_decoder_layers(model)
    if split_layer_idx <= 0 or split_layer_idx >= len(decoder_layers):
        raise ValueError(
            f"split_layer_idx must be within prefix range, got {split_layer_idx} for {len(decoder_layers)} layers"
        )

    dataloader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    collate_fn = getattr(dataset, "collate_fn", None)
    if collate_fn is not None:
        dataloader_kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(dataset, **cast(dict[str, Any], dataloader_kwargs))
    captured: list[torch.Tensor] = []
    examples: list[PrefixFeatureExample] = []

    def _hook_fn(module, args, kwargs):
        del module
        if args:
            captured.append(args[0].detach().cpu())
        elif "hidden_states" in kwargs:
            captured.append(kwargs["hidden_states"].detach().cpu())

    hook = decoder_layers[split_layer_idx].register_forward_pre_hook(
        _hook_fn,
        with_kwargs=True,
    )

    was_training = model.training
    model.eval()
    count = 0
    try:
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            captured.clear()

            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            if not captured:
                raise RuntimeError(
                    "Failed to capture prefix hidden states; split-layer hook did not fire"
                )

            hidden_batch = captured[0]
            position_batch = batch.get("position_ids")
            for row in range(hidden_batch.shape[0]):
                examples.append(
                    PrefixFeatureExample(
                        hidden_states=hidden_batch[row].clone(),
                        attention_mask=batch["attention_mask"][row].detach().cpu().clone(),
                        labels=batch["labels"][row].detach().cpu().clone(),
                        split_layer_idx=split_layer_idx,
                        position_ids=(
                            position_batch[row].detach().cpu().clone()
                            if position_batch is not None
                            else None
                        ),
                    )
                )

            count += 1
            if max_batches is not None and count >= max_batches:
                break
    finally:
        hook.remove()
        if was_training:
            model.train()

    return PrefixFeatureDataset(examples)