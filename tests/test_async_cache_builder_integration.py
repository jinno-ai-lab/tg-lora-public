"""REQ-139: AsyncCacheBuilder full-lifecycle integration tests.

Exercises the complete build → wait → DataLoader swap flow on CPU with a
real (tiny) model, validating the gap between mocked unit tests and actual
runtime behaviour.
"""

from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.tg_lora.prefix_feature_cache import (
    collate_prefix_feature_batch,
)
from src.training.async_cache_builder import AsyncCacheBuilder

from .conftest import TokenDataset, TinyModel


def _make_cfg(tmp_path, split_layer=2):
    return OmegaConf.create({
        "model": {
            "name_or_path": "dummy",
            "device": "cpu",
            "dtype": "float32",
            "load_in_4bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float32",
            "device_map": "auto",
        },
        "data": {
            "train_path": str(tmp_path / "train.jsonl"),
            "valid_quick_path": str(tmp_path / "vq.jsonl"),
            "valid_full_path": str(tmp_path / "vf.jsonl"),
            "max_seq_len": 8,
        },
        "training": {
            "batch_size": 2,
            "prefix_feature_cache_valid_quick": True,
            "prefix_feature_cache_valid_full": True,
        },
        "experiment": {"seed": 42},
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
    })


def _model():
    return TinyModel()


# ---------------------------------------------------------------------------
# REQ-139 / EDGE-167: Full lifecycle
# ---------------------------------------------------------------------------


def test_full_lifecycle_build_wait_load_on_cpu(tmp_path: Path):
    """Build cache with real model → wait → load cached DataLoader → validate batches."""
    cfg = _make_cfg(tmp_path)
    raw_vq = TokenDataset(n=6)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder.start()
        builder.join(timeout=60)

    # (a) Build completed without failure
    assert builder.poll()
    assert not builder.failed

    # (b) Result contains a usable PrefixFeatureDataset
    result = builder.get_result("valid_quick")
    assert result is not None
    assert result.dataset is not None
    assert result.error is None
    assert result.source == "built"
    assert result.build_seconds >= 0
    assert len(result.dataset) == 6

    # (c) DataLoader from cached dataset produces valid batches
    loader = DataLoader(result.dataset, batch_size=2, collate_fn=collate_prefix_feature_batch)
    batches = list(loader)
    assert len(batches) == 3  # 6 samples / batch_size=2

    batch = batches[0]
    assert "hidden_states" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["hidden_states"].shape == (2, 8, 16)  # (batch, seq, hidden)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)

    # (d) Cache file persisted to disk
    assert result.cache_path.exists()
    assert result.cache_path.suffix == ".pt"


def test_poll_and_swap_pattern_simulates_training(tmp_path: Path):
    """Simulate the training-loop poll-and-swap pattern from train_tg_lora.py."""
    cfg = _make_cfg(tmp_path)
    raw_vq = TokenDataset(n=4)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )

        # Phase 1: Start background build, continue "training" with raw dataset
        builder.start()
        raw_loader = DataLoader(raw_vq, batch_size=2)
        raw_batches_consumed = 0
        async_ready = False

        # Consume at least one batch from raw loader (simulating training)
        for batch in raw_loader:
            raw_batches_consumed += 1
            # Poll for completion
            if not async_ready and builder.poll():
                async_ready = True
                break

        assert raw_batches_consumed >= 1

        # Wait if not yet ready
        if not async_ready:
            builder.join(timeout=60)
            async_ready = builder.poll()

        assert async_ready
        assert not builder.failed

        # Phase 2: Swap to cached DataLoader
        result = builder.get_result("valid_quick")
        assert result is not None
        assert result.dataset is not None

        cached_loader = DataLoader(
            result.dataset, batch_size=2, collate_fn=collate_prefix_feature_batch
        )
        cached_batches = list(cached_loader)
        assert len(cached_batches) == 2  # 4 samples / batch_size=2

        # Cached batches have prefix feature structure
        for cb in cached_batches:
            assert "hidden_states" in cb
            assert cb["hidden_states"].isfinite().all()


def test_build_failure_continues_with_raw_dataset(tmp_path: Path):
    """Build failure → builder.failed=True → training continues with raw dataset."""
    cfg = _make_cfg(tmp_path)
    raw_vq = TokenDataset(n=4)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch(
        "src.training.async_cache_builder.load_base_model",
        side_effect=RuntimeError("simulated OOM on cuda:1"),
    ):
        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder.start()
        builder.join(timeout=30)

    # Build failed gracefully
    assert builder.failed
    assert builder.error is not None
    assert "simulated OOM" in str(builder.error)

    # Training can continue with raw dataset
    raw_loader = DataLoader(raw_vq, batch_size=2)
    raw_batches = list(raw_loader)
    assert len(raw_batches) == 2
    for batch in raw_batches:
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] <= 2


def test_disk_cache_reuse_skips_rebuild(tmp_path: Path):
    """First run writes cache to disk; second run loads from disk (source='disk')."""
    cfg = _make_cfg(tmp_path)
    raw_vq = TokenDataset(n=4)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Run 1: Build cache
    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder1 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder1.start()
        builder1.join(timeout=60)

    assert builder1.poll()
    result1 = builder1.get_result("valid_quick")
    assert result1 is not None
    assert result1.source == "built"
    cache_path = result1.cache_path
    assert cache_path.exists()

    # Run 2: Should load from disk without rebuilding
    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder2 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder2.start()
        builder2.join(timeout=60)

    assert builder2.poll()
    result2 = builder2.get_result("valid_quick")
    assert result2 is not None
    assert result2.source == "disk"
    assert result2.dataset is not None
    assert len(result2.dataset) == len(result1.dataset)


def test_concurrent_poll_and_get_result_are_threadsafe(tmp_path: Path):
    """Rapid concurrent poll()/get_result() calls don't crash or deadlock."""
    import threading

    cfg = _make_cfg(tmp_path)
    raw_vq = TokenDataset(n=4)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder.start()

        # Hammer poll/get_result from multiple threads
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def poll_loop():
            barrier.wait()
            for _ in range(50):
                try:
                    builder.poll()
                except Exception as e:
                    errors.append(e)

        def get_result_loop():
            barrier.wait()
            for _ in range(50):
                try:
                    builder.get_result("valid_quick")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=poll_loop),
            threading.Thread(target=poll_loop),
            threading.Thread(target=get_result_loop),
            threading.Thread(target=get_result_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        builder.join(timeout=60)

    assert not errors, f"Thread-safety errors: {errors}"
    assert builder.poll()
    assert not builder.failed
