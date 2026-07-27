"""REQ-139: AsyncCacheBuilder full-lifecycle integration tests.

Exercises the complete build → wait → DataLoader swap flow on CPU with a
real (tiny) model, validating the gap between mocked unit tests and actual
runtime behaviour.
"""

from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.tg_lora.prefix_feature_cache import (
    collate_prefix_feature_batch,
    load_prefix_feature_dataset,
)
from src.training.async_cache_builder import AsyncCacheBuilder

from .conftest import TokenDataset, TinyModel


def _make_cfg(tmp_path, split_layer=2, max_seq_len=8):
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
            "max_seq_len": max_seq_len,
        },
        "training": {
            "batch_size": 2,
            "prefix_feature_cache_valid_quick": True,
            "prefix_feature_cache_valid_full": True,
            "trainable_lora_scope": "last_25_percent",
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


def test_max_seq_len_change_triggers_cache_miss_and_rebuild(tmp_path: Path):
    """TC-132-09 integration: ``cfg.data.max_seq_len`` change → cache miss + rebuild.

    The fingerprint-level proof that ``max_seq_len`` produces a different
    SHA-256 cache path lives in
    ``tests/test_prefix_feature_cache.py::TestCachePathSha256.test_different_max_seq_len_gives_different_path``.
    This test exercises the same property at the AsyncCacheBuilder scale,
    which is the *integration* answer to feedback §3's "実サーバーまたは
    統合テストで同一入力の設定差による cache miss と再推論を確認":

      - Run 1 (max_seq_len=8):   writes cache_8   to disk (source='built').
      - Run 2 (max_seq_len=16):  must NOT reuse cache_8, must rebuild.

    A ``source='disk'`` result on Run 2 would mean the second run silently
    replayed Run 1's truncation-bound content — the same silent-collision
    class closed for ``train_on_prompt`` (TC-132-04) and the model-precision
    fields (TC-132-05/06/07). ``max_seq_len`` has the same shape because
    the HF tokenizer's ``truncation=True, max_length=max_seq_len`` shrinks
    the input token sequence, which (a) changes the cached ``hidden_states``
    (forward pass over the truncated tokens) and (b) shifts where the
    loader's ``labels=-100`` prompt mask lands in the cached ``labels``
    tensor. Both are byte-changing.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw_vq = TokenDataset(n=4)

    # Run 1: build with max_seq_len=8.
    cfg1 = _make_cfg(tmp_path, max_seq_len=8)
    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder1 = AsyncCacheBuilder(
            cfg=cfg1,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            background_device="cpu",
        )
        builder1.start()
        builder1.join(timeout=60)

    assert builder1.poll()
    assert not builder1.failed
    result1 = builder1.get_result("valid_quick")
    assert result1 is not None
    assert result1.source == "built"
    cache_path_8 = result1.cache_path
    assert cache_path_8.exists()

    # Run 2: same setup, only cfg.data.max_seq_len differs (8 → 16).
    cfg2 = _make_cfg(tmp_path, max_seq_len=16)
    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder2 = AsyncCacheBuilder(
            cfg=cfg2,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            background_device="cpu",
        )
        builder2.start()
        builder2.join(timeout=60)

    assert builder2.poll()
    assert not builder2.failed
    result2 = builder2.get_result("valid_quick")
    assert result2 is not None

    # (a) Cache-miss: paths diverge because max_seq_len is in the fingerprint.
    # If a future refactor dropped max_seq_len from
    # build_prefix_feature_cache_metadata, this assertion would fail — that
    # is the mutation proof (silent-collision class, same shape as TC-132-04).
    assert result2.cache_path != cache_path_8, (
        "max_seq_len change must produce a different cache path; got "
        f"identical paths {result2.cache_path!r}. This means max_seq_len "
        "is no longer in the fingerprint — the silent-replay regression "
        "TC-132-09 exists to prevent."
    )

    # (b) Re-inference: source='built' (NOT 'disk'). A 'disk' source here
    # would mean Run 2 silently reused Run 1's truncation-bound content.
    assert result2.source == "built", (
        f"Expected rebuild (source='built') for max_seq_len change; got "
        f"source={result2.source!r}. A 'disk' source means the second run "
        "silently replayed the first run's cached tokens, hidden_states, "
        "and label mask."
    )


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


class _TruncatableDataset(Dataset):
    """Shared raw token rows exposed at a truncation bound.

    Models what the production loader applies from ``cfg.data.max_seq_len``:
    the SAME underlying input rows, truncated to the bound. Two instances
    built from one ``raw`` at different bounds are the feedback-bullet-3
    fixture — same input, config difference only.
    """

    def __init__(self, raw: torch.Tensor, bound: int):
        self.input_ids = raw[:, :bound].contiguous()
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def test_max_seq_len_change_re_infers_config_reflecting_cached_bytes(tmp_path: Path):
    """Feedback bullet 3, empirically: same input + different truncation bound
    → cache miss + re-inference whose PERSISTED bytes reflect the bound.

    TC-132-09 (test_max_seq_len_change_triggers_cache_miss_and_rebuild) pins
    path divergence + source='built'. This test pins the next link: the
    artifacts on disk actually differ — the cached labels of the bound-8 run
    are exactly the bound-16 run's labels truncated to 8, so the truncation
    bound demonstrably changes cached LABEL bytes, not just the fingerprint
    path (the label-affecting rationale, verified at the integration scale).
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw = torch.arange(4 * 16, dtype=torch.long).reshape(4, 16) % 32

    def _run(bound: int):
        cfg = _make_cfg(tmp_path, max_seq_len=bound)
        with (
            patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
            patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
            patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
        ):
            builder = AsyncCacheBuilder(
                cfg=cfg,
                raw_datasets={"valid_quick": _TruncatableDataset(raw, bound)},
                cache_loader_kwargs={},
                split_layer=2,
                cache_dir=cache_dir,
                force_rebuild=False,
                background_device="cpu",
            )
            builder.start()
            builder.join(timeout=60)
        assert builder.poll()
        assert not builder.failed
        result = builder.get_result("valid_quick")
        assert result is not None
        assert result.source == "built"
        return result

    result_8 = _run(8)
    result_16 = _run(16)

    # Cache miss: the fingerprint path diverges (max_seq_len is in it).
    assert result_16.cache_path != result_8.cache_path

    # Re-inference proof at the PERSISTED-artifact level: reload BOTH caches
    # from disk and compare bytes, not the in-memory build results.
    cached_8 = load_prefix_feature_dataset(result_8.cache_path)
    cached_16 = load_prefix_feature_dataset(result_16.cache_path)
    ex_8, ex_16 = cached_8[0], cached_16[0]
    assert ex_8["hidden_states"].shape[0] == 8
    assert ex_16["hidden_states"].shape[0] == 16
    assert ex_8["attention_mask"].shape[0] == 8
    assert ex_16["attention_mask"].shape[0] == 16
    assert ex_8["labels"].shape[0] == 8
    assert ex_16["labels"].shape[0] == 16
    # Same raw input rows: the bound-8 labels ARE the bound-16 labels
    # truncated — the config difference demonstrably changes cached bytes.
    assert torch.equal(ex_8["labels"], ex_16["labels"][:8])


def test_builder_fails_loud_when_dataset_exceeds_fingerprint_bound(tmp_path: Path):
    """A producer feeding rows LONGER than cfg.data.max_seq_len (its loader
    skipped the truncation the config declares) must not plant a cache filed
    under the shorter bound: a later correctly-truncated run would mint the
    SAME fingerprint, hit this cache path, and silently replay untruncated
    bytes. The persistence boundary refuses the write and the builder
    surfaces the per-split failure as source='error'.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw = torch.arange(4 * 16, dtype=torch.long).reshape(4, 16) % 32

    cfg = _make_cfg(tmp_path, max_seq_len=8)  # fingerprint declares bound 8 ...
    with (
        patch("src.training.async_cache_builder.load_base_model", return_value=_model()),
        patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m),
        patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")),
    ):
        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": _TruncatableDataset(raw, 16)},  # ... rows are 16
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            background_device="cpu",
        )
        builder.start()
        builder.join(timeout=60)

    assert builder.poll()  # the loop completes; the failure is per-split:
    result = builder.get_result("valid_quick")
    assert result is not None
    assert result.source == "error"
    assert isinstance(result.error, ValueError)
    assert "max_seq_len" in str(result.error)
    assert list(cache_dir.iterdir()) == [], "a refused write must plant nothing"
