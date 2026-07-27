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
from src.utils.atomic_save import _atomic_torch_save
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
    train_on_prompt: bool,
    dtype: str,
    load_in_4bit: bool,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: str,
) -> dict[str, Any]:
    """Cache-identity metadata for a prefix-feature cache shard.

    ``train_on_prompt`` is REQUIRED (not defaulted) precisely because it changes
    the cached content, not just its interpretation: the dataset's ``labels``
    are produced by the loader (``train_on_prompt=False`` masks the prompt with
    ``labels=-100``; ``True`` keeps them), and :func:`build_prefix_feature_dataset`
    copies those labels verbatim into the cached ``PrefixFeatureExample`` (it
    does NOT re-derive them on load). Because the field is applied at load time
    from ``cfg.training.train_on_prompt`` — NOT baked into the dataset file —
    two runs over the same ``dataset_path`` that differ ONLY in
    ``train_on_prompt`` would otherwise share this fingerprint, hit one cache
    path, and silently replay the other run's (wrongly-)masked labels: the
    operator toggling ``train_on_prompt`` trains on stale labels with no signal.
    Omitting it here is a TypeError at every call site on purpose, so a future
    producer path cannot silently default it.

    ``dtype`` / ``load_in_4bit`` / ``bnb_4bit_quant_type`` /
    ``bnb_4bit_compute_dtype`` are REQUIRED for the same reason, on the
    ACTIVATION side of the cache. ``build_prefix_feature_dataset`` captures the
    base model's forward pass at ``split_layer_idx`` (a forward hook), so every
    cached ``hidden_states`` tensor is computed by ``load_base_model(cfg)`` —
    whose precision is set at load time by these four fields
    (``build_bnb_config`` builds the ``BitsAndBytesConfig`` from
    ``load_in_4bit``/``bnb_4bit_quant_type``/``bnb_4bit_compute_dtype``; absent
    4-bit the weights are cast to ``dtype``). They are NOT baked into the
    dataset file and are only proxied by ``model_name`` here, so two runs over
    the same ``dataset_path``/``model_name`` that differ ONLY in quantization
    (4-bit nf4 vs fp16, nf4 vs fp4, bf16 vs fp16 compute) would otherwise share
    one cache path and silently replay the other run's wrong-precision
    activations — the LoRA suffix then trains against a shifted objective with
    no signal. The four fields together fully determine the forward precision
    (some are no-ops depending on ``load_in_4bit``); all four are stamped so the
    enumeration of precision-changing config stays complete. Omitting any is a
    TypeError at every call site on purpose.
    """
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
        "train_on_prompt": bool(train_on_prompt),
        "dtype": str(dtype),
        "load_in_4bit": bool(load_in_4bit),
        "bnb_4bit_quant_type": str(bnb_4bit_quant_type),
        "bnb_4bit_compute_dtype": str(bnb_4bit_compute_dtype),
    }


def resolve_prefix_feature_cache_seed(seed: int, *, share_across_seeds: bool) -> int:
    return 0 if share_across_seeds else seed


def _cfg_field(node: Any, name: str, default: Any) -> Any:
    """Read one config field from EITHER config model this repo runs on.

    ``cfg.training`` is a Pydantic ``TrainingConfig`` in tests (and in every
    producer path that validates via ``config_schema``) and an OmegaConf
    ``DictConfig`` at the Hydra runtime. Attribute access works on both;
    dict-style ``.get()`` only on the latter. ``getattr(node, name, default)``
    catches both missing-attribute errors uniformly (OmegaConf's
    ``ConfigAttributeError`` subclasses ``AttributeError``). An explicit null
    in an OmegaConf YAML resolves to ``None`` where the Pydantic default would
    have applied, so ``None`` is normalized back to ``default`` — otherwise
    ``str(None) == "None"`` / ``bool(None) is False`` would let the two config
    models disagree on the fingerprint for the SAME intended config.
    """
    value = getattr(node, name, default)
    return default if value is None else value


def prefix_feature_cache_metadata_from_config(
    cfg: Any,
    *,
    dataset_path: str,
    split_layer_idx: int,
) -> dict[str, Any]:
    """The single config→fingerprint mapping shared by ALL cache producers.

    The cache producers (``train_tg_lora``, ``async_cache_builder``,
    ``scripts/precompute_prefix_cache_parallel``) historically each inlined
    this mapping with divergent idioms — ``cfg.training.get(...)`` vs
    ``getattr(cfg.training, ...)`` vs a pre-resolved local, and one producer
    read ``prefix_feature_cache_share_across_seeds`` with NO default. The
    fingerprint is the cache's content-addressed identity: producers that read
    the same cfg differently either mint divergent cache paths for identical
    content (silent redundant rebuilds) or — the TASK-0206/0207 collision
    class — agree on a path while disagreeing on what the bytes mean. Routing
    every producer through this one function makes cross-producer agreement
    structural instead of coincidental, and any future label-affecting or
    precision-affecting config field is added in exactly one place.

    ``dataset_path`` and ``split_layer_idx`` stay parameters: the former is
    per-split (train / valid_quick / valid_full), the latter is computed from
    the model's layer structure rather than carried on ``cfg``.
    """
    return build_prefix_feature_cache_metadata(
        dataset_path=dataset_path,
        model_name=str(cfg.model.name_or_path),
        seed=resolve_prefix_feature_cache_seed(
            int(cfg.experiment.seed),
            share_across_seeds=bool(
                _cfg_field(cfg.training, "prefix_feature_cache_share_across_seeds", False)
            ),
        ),
        max_seq_len=int(cfg.data.max_seq_len),
        split_layer_idx=int(split_layer_idx),
        lora_r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        lora_target_modules=str(cfg.lora.target_modules),
        trainable_lora_scope=str(_cfg_field(cfg.training, "trainable_lora_scope", "all")),
        train_on_prompt=bool(_cfg_field(cfg.training, "train_on_prompt", False)),
        dtype=str(cfg.model.dtype),
        load_in_4bit=bool(cfg.model.load_in_4bit),
        bnb_4bit_quant_type=str(cfg.model.bnb_4bit_quant_type),
        bnb_4bit_compute_dtype=str(cfg.model.bnb_4bit_compute_dtype),
    )


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


def _assert_cached_bytes_match_fingerprint(
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    """Fail loud if persisted bytes contradict the fingerprint they are filed under.

    The fingerprint's ``max_seq_len`` is the truncation bound the dataset
    loader applies from ``cfg.data.max_seq_len``; every cached tensor's
    sequence dimension must be at or below it. A producer that persists rows
    LONGER than its own fingerprint's bound (its loader skipped the
    truncation the config declares) files a lie: a later run with the
    correctly-truncated dataset mints the SAME fingerprint, hits this cache
    path, and silently replays untruncated bytes — the TASK-0206/0207
    silent-replay class arriving THROUGH the producer instead of through the
    fingerprint. The field-enumeration guards
    (``TestCacheFingerprintCompleteness``) prove the bound is IN the
    fingerprint; this guard proves the fingerprint is IN the bytes.

    Metadata without ``max_seq_len`` (non-canonical unit-test fixtures only —
    canonical producers always mint it: it is a required kwarg of
    :func:`build_prefix_feature_cache_metadata`) carries no bound to
    contradict, so the check is a no-op there, not a KeyError.
    """
    bound = metadata.get("max_seq_len")
    if bound is None:
        return
    bound = int(bound)
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("attention_mask", attention_mask),
        ("labels", labels),
    ):
        seq_len = tensor.shape[1]
        if seq_len > bound:
            raise ValueError(
                f"cached {name} sequence length {seq_len} exceeds the "
                f"fingerprint's max_seq_len={bound}: the dataset was not "
                "truncated to the config this cache would be filed under — "
                "persisting it would plant a cache a correctly-truncated run "
                "silently replays"
            )


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
    _assert_cached_bytes_match_fingerprint(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        labels=labels,
        metadata=metadata,
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
    _atomic_torch_save(blob, path)


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

    merged_hidden = torch.cat(hidden_states, dim=0)
    merged_attention_mask = torch.cat(attention_mask, dim=0)
    merged_labels = torch.cat(labels, dim=0)
    _assert_cached_bytes_match_fingerprint(
        hidden_states=merged_hidden,
        attention_mask=merged_attention_mask,
        labels=merged_labels,
        metadata=metadata,
    )

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged_blob = {
        "format_version": _PREFIX_FEATURE_CACHE_FORMAT_VERSION,
        "metadata": metadata,
        "hidden_states": merged_hidden,
        "attention_mask": merged_attention_mask,
        "labels": merged_labels,
        "split_layer_idx": split_layer_idx,
        "position_ids": torch.cat(position_ids, dim=0) if has_position_ids else None,
    }
    _atomic_torch_save(merged_blob, path)


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