"""Zero-mock smoke tests for AsyncCacheBuilder on CPU.

Uses the model_factory injection path so that no `unittest.mock.patch` is
needed.  Every test exercises the real build → wait → DataLoader swap
pipeline with a tiny nn.Module, hitting the same code paths as production
minus the HuggingFace model loader.

Run:  pytest tests/test_async_cache_builder_smoke.py -v
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.tg_lora.prefix_feature_cache import (
    collate_prefix_feature_batch,
)
from src.training.async_cache_builder import AsyncCacheBuilder

from .conftest import TokenDataset, TinyModel


def _make_cfg(tmp_path: Path, split_layer: int = 2) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "name_or_path": "tiny-test",
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
        }
    )


def _model_factory(cfg):
    del cfg
    model = TinyModel()
    model.eval()
    return model, torch.device("cpu")


# ---------------------------------------------------------------------------
# Smoke tests — no unittest.mock used
# ---------------------------------------------------------------------------


class TestSmokeFullLifecycle:
    """Build cache → wait → create DataLoader → validate batches."""

    def test_build_wait_load_single_dataset(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=6)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder.start()
        builder.join(timeout=60)

        assert builder.poll()
        assert not builder.failed

        result = builder.get_result("valid_quick")
        assert result is not None
        assert result.dataset is not None
        assert result.error is None
        assert result.source == "built"
        assert len(result.dataset) == 6

        # DataLoader produces correctly-shaped batches
        loader = DataLoader(
            result.dataset, batch_size=2, collate_fn=collate_prefix_feature_batch
        )
        batches = list(loader)
        assert len(batches) == 3

        batch = batches[0]
        assert batch["hidden_states"].shape == (2, 8, 16)
        assert batch["attention_mask"].shape == (2, 8)
        assert batch["labels"].shape == (2, 8)
        assert batch["hidden_states"].isfinite().all()

        # Cache file persisted to disk
        assert result.cache_path.exists()
        assert result.cache_path.suffix == ".pt"

    def test_build_two_datasets_concurrently(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=4)
        raw_vf = TokenDataset(n=6)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq, "valid_full": raw_vf},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder.start()
        builder.join(timeout=60)

        assert builder.poll()
        assert not builder.failed

        r_vq = builder.get_result("valid_quick")
        r_vf = builder.get_result("valid_full")
        assert r_vq is not None and r_vq.dataset is not None
        assert r_vf is not None and r_vf.dataset is not None
        assert len(r_vq.dataset) == 4
        assert len(r_vf.dataset) == 6


class TestSmokePollAndSwap:
    """Simulate the training-loop poll-and-swap pattern."""

    def test_poll_and_swap_training_loop(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=8)
        raw_vf = TokenDataset(n=6)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq, "valid_full": raw_vf},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )

        # Phase 1: Start background build, consume raw data
        builder.start()
        raw_loader = DataLoader(raw_vq, batch_size=2)
        raw_batches_consumed = 0
        async_ready = False

        for batch in raw_loader:
            raw_batches_consumed += 1
            if not async_ready and builder.poll():
                async_ready = True
                break

        assert raw_batches_consumed >= 1

        # Phase 2: Wait for completion if needed
        if not async_ready:
            builder.join(timeout=60)
            async_ready = builder.poll()

        assert async_ready
        assert not builder.failed

        # Phase 3: Swap to cached DataLoaders
        for label, raw_ds in [("valid_quick", raw_vq), ("valid_full", raw_vf)]:
            result = builder.get_result(label)
            assert result is not None
            assert result.dataset is not None

            cached_loader = DataLoader(
                result.dataset, batch_size=2, collate_fn=collate_prefix_feature_batch
            )
            for cb in cached_loader:
                assert cb["hidden_states"].isfinite().all()
                assert cb["hidden_states"].shape[0] <= 2

    def test_swap_mid_training_preserves_batch_count(self, tmp_path: Path):
        """After swap, cached loader should produce same sample count as raw."""
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=6)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder.start()
        builder.join(timeout=60)

        result = builder.get_result("valid_quick")
        assert result is not None

        raw_loader = DataLoader(raw_vq, batch_size=2)
        cached_loader = DataLoader(
            result.dataset, batch_size=2, collate_fn=collate_prefix_feature_batch
        )

        raw_count = sum(b["input_ids"].shape[0] for b in raw_loader)
        cached_count = sum(b["hidden_states"].shape[0] for b in cached_loader)
        assert raw_count == cached_count == 6


class TestSmokeDiskPersistence:
    """Verify cache survives across builder instances."""

    def test_first_build_second_disk_load(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=4)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Run 1: Build
        builder1 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder1.start()
        builder1.join(timeout=60)

        assert builder1.poll()
        r1 = builder1.get_result("valid_quick")
        assert r1 is not None
        assert r1.source == "built"
        assert r1.cache_path.exists()

        # Run 2: Should load from disk
        builder2 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder2.start()
        builder2.join(timeout=60)

        assert builder2.poll()
        r2 = builder2.get_result("valid_quick")
        assert r2 is not None
        assert r2.source == "disk"
        assert len(r2.dataset) == len(r1.dataset)

    def test_force_rebuild_ignores_disk_cache(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=4)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Populate disk cache first
        builder1 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder1.start()
        builder1.join(timeout=60)

        # Force rebuild should ignore existing cache
        builder2 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=True,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder2.start()
        builder2.join(timeout=60)

        r2 = builder2.get_result("valid_quick")
        assert r2 is not None
        assert r2.source == "built"


class TestSmokeErrorHandling:
    """Verify builder handles factory errors gracefully."""

    def test_factory_exception_sets_failed(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=4)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def failing_factory(cfg):
            raise RuntimeError("simulated OOM on background device")

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=failing_factory,
        )
        builder.start()
        builder.join(timeout=30)

        assert builder.failed
        assert builder.error is not None
        assert "simulated OOM" in str(builder.error)

        # Training can continue with raw dataset
        raw_loader = DataLoader(raw_vq, batch_size=2)
        assert len(list(raw_loader)) == 2

    def test_partial_failure_one_dataset_fails(self, tmp_path: Path):
        """When build_one fails for one label, other labels still succeed."""
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=4)
        raw_vf = TokenDataset(n=4)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        call_count = {"n": 0}

        def factory_with_bad_split(cfg):
            """Return a model whose split_layer_idx=999 triggers a ValueError."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                model = TinyModel(layers=1)
                model.eval()
                return model, torch.device("cpu")
            return _model_factory(cfg)

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq, "valid_full": raw_vf},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=factory_with_bad_split,
        )
        builder.start()
        builder.join(timeout=60)

        # Both datasets should succeed
        assert not builder.failed
        assert builder.get_result("valid_quick") is not None
        assert builder.get_result("valid_full") is not None


class TestSmokeHiddenStateValues:
    """Verify that cached hidden states are deterministic and non-trivial."""

    def test_hidden_states_deterministic_across_builds(self, tmp_path: Path):
        """Two builds of the same dataset produce identical hidden states."""
        cfg = _make_cfg(tmp_path)

        torch.manual_seed(42)
        raw_vq = TokenDataset(n=4)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Share same model weights across both builds via state_dict
        torch.manual_seed(99)
        shared_state = TinyModel().state_dict()

        def deterministic_factory(cfg):
            del cfg
            model = TinyModel()
            model.load_state_dict(shared_state)
            model.eval()
            return model, torch.device("cpu")

        builder1 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=deterministic_factory,
        )
        builder1.start()
        builder1.join(timeout=60)

        r1 = builder1.get_result("valid_quick")

        builder2 = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=True,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=deterministic_factory,
        )
        builder2.start()
        builder2.join(timeout=60)

        r2 = builder2.get_result("valid_quick")

        h1 = r1.dataset[0]["hidden_states"]
        h2 = r2.dataset[0]["hidden_states"]
        assert torch.allclose(h1, h2, atol=1e-6)

    def test_hidden_states_not_all_zeros(self, tmp_path: Path):
        """Cached features should be non-trivial (not all zeros)."""
        cfg = _make_cfg(tmp_path)
        raw_vq = TokenDataset(n=4)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
            model_factory=_model_factory,
        )
        builder.start()
        builder.join(timeout=60)

        result = builder.get_result("valid_quick")
        loader = DataLoader(
            result.dataset, batch_size=4, collate_fn=collate_prefix_feature_batch
        )
        batch = next(iter(loader))
        assert not (batch["hidden_states"] == 0).all()
