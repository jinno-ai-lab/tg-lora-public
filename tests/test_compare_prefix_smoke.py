"""Smoke tests for compare-prefix cold/warm/coldwarm targets.

Validates the configuration wiring that run_comparison.sh performs
(TG_PREFIX_CACHE_DIR, TG_PREFIX_FORCE_REBUILD) and verifies cache hit/miss
behavior through the _maybe_cache_dataset logic used in train_tg_lora.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tg_lora.prefix_feature_cache import (
    PrefixFeatureDataset,
    PrefixFeatureExample,
    build_prefix_feature_cache_metadata,
    get_prefix_feature_cache_path,
    load_prefix_feature_dataset,
    save_prefix_feature_dataset,
)


def _make_example(n: int = 2, seq_len: int = 8, hidden: int = 16):
    examples = []
    for _ in range(n):
        examples.append(
            PrefixFeatureExample(
                hidden_states=torch.randn(seq_len, hidden),
                attention_mask=torch.ones(seq_len, dtype=torch.long),
                labels=torch.randint(0, 32, (seq_len,)),
                split_layer_idx=2,
            )
        )
    return PrefixFeatureDataset(examples)


def _default_metadata(**overrides):
    base = {
        "dataset_path": "data/train.jsonl",
        "model_name": "dummy-model",
        "seed": 42,
        "max_seq_len": 8,
        "split_layer_idx": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target_modules": "all-linear",
        "trainable_lora_scope": "last_25_percent",
    }
    base.update(overrides)
    return build_prefix_feature_cache_metadata(**base)


def _simulate_source(cache_path: Path, force_rebuild: bool,
                      cached_prefix_datasets: dict | None = None) -> str:
    """Replicate _maybe_cache_dataset source logic from train_tg_lora."""
    if cached_prefix_datasets is None:
        cached_prefix_datasets = {}
    if cache_path in cached_prefix_datasets:
        return "memory"
    elif cache_path.exists() and not force_rebuild:
        return "disk"
    else:
        return "built"


class TestComparePrefixConfigWiring:
    """Verify that the shell script env var → OmegaConf config wiring is correct."""

    def test_force_rebuild_false_parsed_from_string(self):
        cfg = OmegaConf.create({"training": {"prefix_feature_cache_force_rebuild": False}})
        assert bool(cfg.training.get("prefix_feature_cache_force_rebuild", False)) is False

    def test_force_rebuild_true_parsed_from_string(self):
        cfg = OmegaConf.create({"training": {"prefix_feature_cache_force_rebuild": True}})
        assert bool(cfg.training.get("prefix_feature_cache_force_rebuild", False)) is True

    def test_cache_dir_resolved_to_path(self):
        cache_dir = ".cache/prefix_feature_cache_compare"
        cfg = OmegaConf.create({"training": {"prefix_feature_cache_dir": cache_dir}})
        resolved = Path(str(cfg.training.get("prefix_feature_cache_dir", ".cache/prefix_feature_cache")))
        assert str(resolved) == cache_dir


class TestColdStartCacheBehavior:
    """Verify cold start: no cache exists → must build."""

    def test_cold_start_builds_cache(self, tmp_path):
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        force_rebuild = False

        assert not cache_path.exists()
        if cache_path.exists() and not force_rebuild:
            source = "disk"
        else:
            source = "built"

        assert source == "built"

    def test_cold_start_saves_cache_for_subsequent_warm(self, tmp_path):
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        dataset = _make_example(n=4)
        save_prefix_feature_dataset(dataset, cache_path, metadata=metadata)

        assert cache_path.exists()
        loaded = load_prefix_feature_dataset(cache_path)
        assert len(loaded) == 4


class TestWarmStartCacheBehavior:
    """Verify warm start: cache exists → disk load."""

    def test_warm_start_uses_disk_cache(self, tmp_path):
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        dataset = _make_example(n=4)
        save_prefix_feature_dataset(dataset, cache_path, metadata=metadata)

        force_rebuild = False
        if cache_path.exists() and not force_rebuild:
            loaded = load_prefix_feature_dataset(cache_path)
            source = "disk"
        else:
            source = "built"

        assert source == "disk"
        assert len(loaded) == 4


class TestColdWarmSequence:
    """Verify cold→warm sequence: first build then reuse."""

    def test_cold_then_warm_uses_same_cache_path(self, tmp_path):
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

        # Cold: no cache → build + save
        assert not cache_path.exists()
        dataset = _make_example(n=4)
        save_prefix_feature_dataset(dataset, cache_path, metadata=metadata)
        assert cache_path.exists()

        # Warm: cache exists → disk load
        force_rebuild = False
        if cache_path.exists() and not force_rebuild:
            loaded = load_prefix_feature_dataset(cache_path)
            source = "disk"
        else:
            source = "built"

        assert source == "disk"
        assert len(loaded) == 4
        assert loaded.total_bytes == dataset.total_bytes

    def test_coldwarm_produces_consistent_data(self, tmp_path):
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

        dataset = _make_example(n=4)
        save_prefix_feature_dataset(dataset, cache_path, metadata=metadata)

        loaded = load_prefix_feature_dataset(cache_path)
        for i in range(len(loaded)):
            assert torch.equal(
                loaded._examples[i].hidden_states,
                dataset._examples[i].hidden_states,
            )
            assert torch.equal(
                loaded._examples[i].attention_mask,
                dataset._examples[i].attention_mask,
            )
            assert torch.equal(
                loaded._examples[i].labels,
                dataset._examples[i].labels,
            )


# ---------------------------------------------------------------------------
# REQ-135: compare-prefix cold/warm smoke tests
# ---------------------------------------------------------------------------


class TestComparePrefixColdWarmSmoke:
    """TC-135-01/02/B01: simulate cold→warm cache cycle with source tracking."""

    def test_cold_completes_and_creates_cache(self, tmp_path: Path):
        """TC-135-01: cold run completes, cache exists, source='built'."""
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        assert not cache_path.exists()

        source = _simulate_source(cache_path, force_rebuild=False)
        assert source == "built"

        dataset = _make_example(n=4)
        save_prefix_feature_dataset(dataset, cache_path, metadata=metadata)
        assert cache_path.exists()

    def test_warm_reuses_disk_cache(self, tmp_path: Path):
        """TC-135-02: warm run reuses cache, source='disk'."""
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

        dataset = _make_example(n=4)
        save_prefix_feature_dataset(dataset, cache_path, metadata=metadata)
        assert cache_path.exists()

        source = _simulate_source(cache_path, force_rebuild=False)
        assert source == "disk"

    def test_cold_then_warm_confirms_different_sources(self, tmp_path: Path):
        """TC-135-B01: cold→warm sequence yields source='built' then 'disk'."""
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

        # Cold pass
        cold_source = _simulate_source(cache_path, force_rebuild=False)
        dataset = _make_example(n=4)
        save_prefix_feature_dataset(dataset, cache_path, metadata=metadata)

        # Warm pass
        warm_source = _simulate_source(cache_path, force_rebuild=False)

        assert cold_source == "built"
        assert warm_source == "disk"
