"""Background prefix feature cache builder for multi-GPU setups.

Loads a second model copy on a background device and builds prefix feature
caches in a daemon thread while training runs on the primary GPU.

Correctness guarantee: PEFT initializes LoRA B matrices to zeros, so at
initialization LoRA(x) = B @ A @ x = 0 and the model output is identical
to the base model. A fresh model copy produces the same prefix features.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from src.model.load_model import apply_lora, get_input_device, load_base_model
from src.tg_lora.prefix_feature_cache import (
    PrefixFeatureDatasetBase, build_prefix_feature_cache_metadata,
    build_prefix_feature_dataset, get_prefix_feature_cache_path,
    load_prefix_feature_dataset, resolve_prefix_feature_cache_seed,
    save_prefix_feature_dataset)

# Type alias for the model factory callable accepted by AsyncCacheBuilder.
# When provided, the builder skips load_base_model/apply_lora and uses the
# factory output directly.  This enables zero-mock smoke testing on CPU.
ModelFactory = Callable[[DictConfig], tuple[nn.Module, torch.device]]

logger = logging.getLogger("tg-lora")


@dataclass
class AsyncCacheBuildResult:
    label: str
    dataset: PrefixFeatureDatasetBase | None
    build_seconds: float
    source: str
    cache_path: Path
    error: Exception | None = None


class AsyncCacheBuilder:
    """Builds prefix feature caches on a background GPU in a daemon thread."""

    def __init__(
        self,
        cfg: DictConfig,
        raw_datasets: dict[str, Dataset],
        cache_loader_kwargs: dict,
        split_layer: int,
        cache_dir: Path,
        force_rebuild: bool,
        trainable_lora_scope: str,
        background_device: str,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if cfg is None:
            raise TypeError("cfg must be a DictConfig, not None")
        if not isinstance(raw_datasets, dict) or len(raw_datasets) == 0:
            raise ValueError("raw_datasets must be a non-empty dict")
        if not isinstance(cache_dir, Path):
            raise TypeError("cache_dir must be a Path")
        if not isinstance(trainable_lora_scope, str) or not trainable_lora_scope:
            raise ValueError("trainable_lora_scope must be a non-empty string")
        _valid_devices = {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "mps"}
        if background_device not in _valid_devices and not background_device.startswith("cuda:"):
            raise ValueError(
                f"background_device must be a valid device string (e.g. 'cpu', 'cuda', 'cuda:0', 'mps'), "
                f"got {background_device!r}"
            )
        if split_layer < 0:
            raise ValueError(f"split_layer must be non-negative, got {split_layer}")
        self._cfg = cfg
        self._raw_datasets = raw_datasets
        self._cache_loader_kwargs = cache_loader_kwargs
        self._split_layer = split_layer
        self._cache_dir = cache_dir
        self._force_rebuild = force_rebuild
        self._trainable_lora_scope = trainable_lora_scope
        self._background_device = background_device
        self._model_factory = model_factory

        self._results: dict[str, AsyncCacheBuildResult] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._failed = False
        self._completed = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="async-cache-builder",
            daemon=True,
        )
        self._thread.start()

    def poll(self) -> bool:
        with self._lock:
            return self._completed

    def get_result(self, label: str) -> AsyncCacheBuildResult | None:
        with self._lock:
            return self._results.get(label)

    def join(self, timeout: float = 300.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._failed

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def _run(self) -> None:
        try:
            t0 = time.perf_counter()
            if self._model_factory is not None:
                model_copy, device = self._model_factory(self._cfg)
            else:
                bg_cfg = OmegaConf.merge(
                    self._cfg, {"model": {"device": self._background_device}}
                )
                model_copy = load_base_model(bg_cfg)
                model_copy = apply_lora(model_copy, bg_cfg)
                device = get_input_device(model_copy)
            model_load_seconds = time.perf_counter() - t0
            logger.info(
                "Async cache builder: model loaded on %s in %.1fs",
                self._background_device,
                model_load_seconds,
            )

            for label, raw_dataset in self._raw_datasets.items():
                dataset_path = self._resolve_dataset_path(label)
                enabled = self._is_enabled(label)
                if not enabled:
                    continue
                result = self._build_one(model_copy, device, label, raw_dataset, dataset_path)
                with self._lock:
                    self._results[label] = result
                if result.error is not None:
                    logger.warning(
                        "Async cache builder: %s failed: %s", label, result.error
                    )

            del model_copy
            from src.utils.device import gpu_empty_cache
            gpu_empty_cache(torch.device(self._background_device))

            with self._lock:
                self._completed = True
            logger.info("Async cache builder: all caches built")

        except Exception as exc:
            logger.error("Async cache builder failed: %s", exc, exc_info=True)
            with self._lock:
                self._failed = True
                self._error = exc
                self._completed = True

    def _resolve_dataset_path(self, label: str) -> str:
        path_map = {
            "valid_quick": str(self._cfg.data.valid_quick_path),
            "valid_full": str(self._cfg.data.valid_full_path),
            "train": str(self._cfg.data.train_path),
        }
        return path_map[label]

    def _is_enabled(self, label: str) -> bool:
        key = f"prefix_feature_cache_{label}"
        return bool(self._cfg.training.get(key, True))

    def _build_one(
        self,
        model: nn.Module,
        device: torch.device,
        label: str,
        raw_dataset: Dataset,
        dataset_path: str,
    ) -> AsyncCacheBuildResult:
        try:
            metadata = build_prefix_feature_cache_metadata(
                dataset_path=dataset_path,
                model_name=str(self._cfg.model.name_or_path),
                seed=resolve_prefix_feature_cache_seed(
                    int(self._cfg.experiment.seed),
                    share_across_seeds=bool(
                        self._cfg.training.get(
                            "prefix_feature_cache_share_across_seeds", False
                        )
                    ),
                ),
                max_seq_len=int(self._cfg.data.max_seq_len),
                split_layer_idx=self._split_layer,
                lora_r=int(self._cfg.lora.r),
                lora_alpha=int(self._cfg.lora.alpha),
                lora_dropout=float(self._cfg.lora.dropout),
                lora_target_modules=str(self._cfg.lora.target_modules),
                trainable_lora_scope=self._trainable_lora_scope,
            )
            cache_path = get_prefix_feature_cache_path(self._cache_dir, metadata)
            lazy_disk = (
                str(self._cfg.training.get("prefix_feature_cache_mode", "reuse"))
                == "one_shot"
            )

            if cache_path.exists() and not self._force_rebuild:
                t0 = time.perf_counter()
                cached = load_prefix_feature_dataset(cache_path, lazy=lazy_disk)
                load_seconds = time.perf_counter() - t0
                logger.info(
                    "Async cache builder: %s loaded from disk in %.1fs",
                    label,
                    load_seconds,
                )
                return AsyncCacheBuildResult(
                    label=label,
                    dataset=cached,
                    build_seconds=0.0,
                    source="disk",
                    cache_path=cache_path,
                )

            t0 = time.perf_counter()
            cached = build_prefix_feature_dataset(
                model,
                raw_dataset,
                batch_size=int(self._cfg.training.batch_size),
                device=device,
                split_layer_idx=self._split_layer,
                num_workers=int(self._cache_loader_kwargs.get("num_workers", 0)),
                pin_memory=bool(self._cache_loader_kwargs.get("pin_memory", False)),
                persistent_workers=bool(
                    self._cache_loader_kwargs.get("persistent_workers", False)
                ),
                prefetch_factor=self._cache_loader_kwargs.get("prefetch_factor"),
            )
            build_seconds = time.perf_counter() - t0

            save_prefix_feature_dataset(cached, cache_path, metadata=metadata)
            if lazy_disk:
                cached = load_prefix_feature_dataset(cache_path, lazy=True)
            logger.info(
                "Async cache builder: %s built in %.1fs", label, build_seconds
            )

            return AsyncCacheBuildResult(
                label=label,
                dataset=cached,
                build_seconds=build_seconds,
                source="built",
                cache_path=cache_path,
            )
        except Exception as exc:
            return AsyncCacheBuildResult(
                label=label,
                dataset=None,
                build_seconds=0.0,
                source="error",
                cache_path=Path(""),
                error=exc,
            )
