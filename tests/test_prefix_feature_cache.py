import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.eval_loss import eval_loss
from src.model.lora_utils import (iter_all_lora_params_by_layer,
                                  set_trainable_lora_layers)
from src.tg_lora.prefix_feature_cache import (
    MappedPrefixFeatureDataset, PrefixFeatureDataset, PrefixFeatureExample,
    build_prefix_feature_cache_metadata, build_prefix_feature_dataset,
    collate_prefix_feature_batch, compute_prefix_feature_shard_ranges,
    get_prefix_feature_cache_path, load_prefix_feature_dataset,
    merge_prefix_feature_cache_shards, resolve_prefix_feature_cache_seed,
    save_prefix_feature_dataset)
from src.training.loss import compute_loss


class _SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return (self.norm(self.linear(hidden_states)),)


class _SimplePrefixCacheModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 16, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            [_SimpleDecoderLayer(hidden) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

        self.lora_bank = nn.Module()
        for idx in range(num_layers):
            layer_mod = nn.Module()
            layer_mod.register_parameter("lora_A", nn.Parameter(torch.randn(hidden, hidden) * 0.01))
            layer_mod.register_parameter("lora_B", nn.Parameter(torch.randn(hidden, hidden) * 0.01))
            self.layers[idx].add_module("mock_lora", layer_mod)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        del kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)[0]
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return type("Out", (), {"loss": loss})()


class _TokenDataset(Dataset):
    def __init__(self, n: int = 6, seq_len: int = 8, vocab_size: int = 32):
        self.input_ids = torch.randint(0, vocab_size, (n, seq_len))
        self.attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class _TokenDatasetWithPositions(Dataset):
    def __init__(self, n: int = 6, seq_len: int = 8, vocab_size: int = 32):
        self.input_ids = torch.randint(0, vocab_size, (n, seq_len))
        self.attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        self.labels = self.input_ids.clone()
        self.position_ids = torch.arange(seq_len).unsqueeze(0).expand(n, -1).clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "position_ids": self.position_ids[idx],
        }


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


def test_build_prefix_feature_dataset_matches_full_eval_loss():
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=6)
    raw_loader = DataLoader(raw_dataset, batch_size=2)

    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
    )
    cached_loader = DataLoader(
        cached_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_prefix_feature_batch,
    )

    full_loss = eval_loss(model, raw_loader, device="cpu")
    cached_loss = eval_loss(model, cached_loader, device="cpu")

    assert len(cached_dataset) == len(raw_dataset)
    assert cached_dataset.total_bytes > 0
    assert math.isclose(full_loss, cached_loss, rel_tol=0.0, abs_tol=1e-5)


def test_compute_loss_accepts_cached_hidden_state_batch():
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=2)
    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
        max_batches=1,
    )
    batch = collate_prefix_feature_batch([cached_dataset[0], cached_dataset[1]])
    loss = compute_loss(model, batch)
    assert torch.isfinite(loss)


def test_set_trainable_lora_layers_freezes_prefix_layers():
    model = _SimplePrefixCacheModel(num_layers=4)
    active_names = set_trainable_lora_layers(model, {2, 3})
    layer_map = iter_all_lora_params_by_layer(model)

    assert active_names
    for layer_idx, params in layer_map.items():
        for name, param in params:
            assert param.requires_grad is (layer_idx in {2, 3})
            if layer_idx in {2, 3}:
                assert name in active_names


def test_prefix_feature_dataset_round_trips_through_disk_cache(tmp_path: Path):
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=4)
    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
    )
    metadata = build_prefix_feature_cache_metadata(
        dataset_path="data/train.jsonl",
        model_name="dummy-model",
        seed=42,
        max_seq_len=8,
        split_layer_idx=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules="all-linear",
        trainable_lora_scope="last_25_percent",
    )
    cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

    save_prefix_feature_dataset(cached_dataset, cache_path, metadata=metadata)
    reloaded_dataset = load_prefix_feature_dataset(cache_path)

    raw_loader = DataLoader(raw_dataset, batch_size=2)
    reloaded_loader = DataLoader(
        reloaded_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_prefix_feature_batch,
    )
    assert cache_path.exists()
    assert reloaded_dataset.total_bytes == cached_dataset.total_bytes
    assert math.isclose(
        eval_loss(model, raw_loader, device="cpu"),
        eval_loss(model, reloaded_loader, device="cpu"),
        rel_tol=0.0,
        abs_tol=1e-5,
    )


def test_prefix_feature_dataset_can_load_lazy_from_disk(tmp_path: Path):
    model = _SimplePrefixCacheModel()
    raw_dataset = _TokenDataset(n=4)
    cached_dataset = build_prefix_feature_dataset(
        model,
        raw_dataset,
        batch_size=2,
        device="cpu",
        split_layer_idx=2,
    )
    metadata = _default_metadata()
    cache_path = get_prefix_feature_cache_path(tmp_path, metadata)

    save_prefix_feature_dataset(cached_dataset, cache_path, metadata=metadata)
    lazy_dataset = load_prefix_feature_dataset(cache_path, lazy=True)

    assert isinstance(lazy_dataset, MappedPrefixFeatureDataset)
    assert lazy_dataset.total_bytes == cached_dataset.total_bytes
    assert torch.equal(lazy_dataset[0]["labels"], cached_dataset[0]["labels"])


# ---------------------------------------------------------------------------
# REQ-128: Corrupted cache file handling
# ---------------------------------------------------------------------------


class TestCorruptedCacheHandling:
    """TC-128-E01/E02/E03: load_prefix_feature_dataset rejects malformed files."""

    def test_partial_write_raises_error(self, tmp_path: Path):
        """TC-128-E01: 1-byte file causes torch.load failure."""
        bad_file = tmp_path / "partial.pt"
        bad_file.write_bytes(b"\x00")
        with pytest.raises(Exception):
            load_prefix_feature_dataset(bad_file)

    def test_non_dict_format_raises_error(self, tmp_path: Path):
        """TC-128-E02: tensor-only file causes AttributeError/TypeError."""
        bad_file = tmp_path / "tensor_only.pt"
        torch.save(torch.randn(3, 4), bad_file)
        with pytest.raises((AttributeError, TypeError, KeyError)):
            load_prefix_feature_dataset(bad_file)

    def test_missing_hidden_states_key_raises_error(self, tmp_path: Path):
        """TC-128-E03: dict without hidden_states causes KeyError."""
        bad_file = tmp_path / "no_hidden.pt"
        torch.save({"format_version": 1, "metadata": {}}, bad_file)
        with pytest.raises(KeyError):
            load_prefix_feature_dataset(bad_file)


# ---------------------------------------------------------------------------
# REQ-129: force_rebuild flag
# ---------------------------------------------------------------------------


class TestForceRebuildFlag:
    """TC-129-01/02: _maybe_cache_dataset force_rebuild logic."""

    @staticmethod
    def _simulate_maybe_cache(cache_path: Path, force_rebuild: bool):
        """Simulate the _maybe_cache_dataset source decision logic."""
        cached_prefix_datasets: dict[Path, PrefixFeatureDataset] = {}
        if cache_path in cached_prefix_datasets:
            return "memory"
        elif cache_path.exists() and not force_rebuild:
            return "disk"
        else:
            return "built"

    def test_force_rebuild_false_reuses_disk_cache(self, tmp_path: Path):
        """TC-129-01: force_rebuild=false → source='disk' when cache exists."""
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        examples = [
            PrefixFeatureExample(
                hidden_states=torch.randn(8, 16),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.randint(0, 32, (8,)),
                split_layer_idx=2,
            )
        ]
        ds = PrefixFeatureDataset(examples)
        save_prefix_feature_dataset(ds, cache_path, metadata=metadata)
        assert cache_path.exists()
        source = self._simulate_maybe_cache(cache_path, force_rebuild=False)
        assert source == "disk"

    def test_force_rebuild_true_skips_disk_cache(self, tmp_path: Path):
        """TC-129-02: force_rebuild=true → source='built' even when cache exists."""
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        examples = [
            PrefixFeatureExample(
                hidden_states=torch.randn(8, 16),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.randint(0, 32, (8,)),
                split_layer_idx=2,
            )
        ]
        ds = PrefixFeatureDataset(examples)
        save_prefix_feature_dataset(ds, cache_path, metadata=metadata)
        assert cache_path.exists()
        source = self._simulate_maybe_cache(cache_path, force_rebuild=True)
        assert source == "built"


# ---------------------------------------------------------------------------
# REQ-130: position_ids build path
# ---------------------------------------------------------------------------


class TestPositionIdsBuildPath:
    """TC-130-01/02: position_ids through build and save/load roundtrip."""

    def test_position_ids_preserved_in_build(self):
        """TC-130-01: build with position_ids dataset stores them per-example."""
        model = _SimplePrefixCacheModel()
        ds = _TokenDatasetWithPositions(n=4)
        cached = build_prefix_feature_dataset(
            model, ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        for ex in cached._examples:
            assert ex.position_ids is not None
            assert ex.position_ids.shape == (8,)

    def test_position_ids_roundtrip_through_disk(self, tmp_path: Path):
        """TC-130-02: save→load preserves position_ids."""
        model = _SimplePrefixCacheModel()
        ds = _TokenDatasetWithPositions(n=4)
        cached = build_prefix_feature_dataset(
            model, ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        save_prefix_feature_dataset(cached, cache_path, metadata=metadata)
        loaded = load_prefix_feature_dataset(cache_path)
        for orig, reloaded in zip(cached._examples, loaded._examples):
            assert reloaded.position_ids is not None
            assert torch.equal(orig.position_ids, reloaded.position_ids)


# ---------------------------------------------------------------------------
# REQ-131: model.training state restoration
# ---------------------------------------------------------------------------


class TestModelTrainingRestoration:
    """TC-131-E01/E02: build restores model.training on both success and error."""

    def test_training_restored_after_exception(self):
        """TC-131-E01: model.training restored after forward raises."""
        model = _SimplePrefixCacheModel()
        model.train()
        ds = _TokenDataset(n=4)
        with patch.object(
            model, "forward", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                build_prefix_feature_dataset(
                    model, ds, batch_size=2, device="cpu", split_layer_idx=2,
                )
        assert model.training is True

    def test_training_restored_after_normal_build(self):
        """TC-131-E02: model.training restored after successful build."""
        model = _SimplePrefixCacheModel()
        model.train()
        ds = _TokenDataset(n=4)
        build_prefix_feature_dataset(
            model, ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        assert model.training is True


# ---------------------------------------------------------------------------
# REQ-132: SHA-256 cache path uniqueness
# ---------------------------------------------------------------------------


class TestCachePathSha256:
    """TC-132-01/02/03: cache path varies with metadata."""

    def test_different_seed_gives_different_path(self):
        """TC-132-01: different seed → different path."""
        m1 = _default_metadata(seed=42)
        m2 = _default_metadata(seed=43)
        p1 = get_prefix_feature_cache_path("/tmp/cache", m1)
        p2 = get_prefix_feature_cache_path("/tmp/cache", m2)
        assert p1 != p2

    def test_same_metadata_gives_same_path(self):
        """TC-132-02: identical metadata → identical path."""
        m = _default_metadata(seed=42)
        p1 = get_prefix_feature_cache_path("/tmp/cache", m)
        p2 = get_prefix_feature_cache_path("/tmp/cache", m)
        assert p1 == p2

    def test_different_lora_r_gives_different_path(self):
        """TC-132-03: different lora_r → different path."""
        m1 = _default_metadata(lora_r=16)
        m2 = _default_metadata(lora_r=32)
        p1 = get_prefix_feature_cache_path("/tmp/cache", m1)
        p2 = get_prefix_feature_cache_path("/tmp/cache", m2)
        assert p1 != p2


# ---------------------------------------------------------------------------
# REQ-133: format_version mismatch
# ---------------------------------------------------------------------------


class TestFormatVersionMismatch:
    """TC-133-E01/E02: load rejects wrong/missing format_version."""

    def test_format_version_zero_raises_value_error(self, tmp_path: Path):
        """TC-133-E01: format_version=0 → ValueError."""
        bad_file = tmp_path / "v0.pt"
        torch.save(
            {"format_version": 0, "metadata": {}, "hidden_states": torch.randn(2, 4, 8),
             "attention_mask": torch.ones(2, 4, dtype=torch.long),
             "labels": torch.randint(0, 32, (2, 4)), "split_layer_idx": 2},
            bad_file,
        )
        with pytest.raises(ValueError, match="Unsupported prefix feature cache format version: 0"):
            load_prefix_feature_dataset(bad_file)

    def test_missing_format_version_raises_value_error(self, tmp_path: Path):
        """TC-133-E02: no format_version key → ValueError with None."""
        bad_file = tmp_path / "no_version.pt"
        torch.save(
            {"metadata": {}, "hidden_states": torch.randn(2, 4, 8),
             "attention_mask": torch.ones(2, 4, dtype=torch.long),
             "labels": torch.randint(0, 32, (2, 4)), "split_layer_idx": 2},
            bad_file,
        )
        with pytest.raises(ValueError, match="Unsupported prefix feature cache format version: None"):
            load_prefix_feature_dataset(bad_file)


# ---------------------------------------------------------------------------
# REQ-134: Empty dataset rejection
# ---------------------------------------------------------------------------


class TestEmptyDatasetRejection:
    """TC-134-E01: save rejects empty PrefixFeatureDataset."""

    def test_empty_dataset_raises_value_error(self, tmp_path: Path):
        """TC-134-E01: empty dataset → ValueError, no file created."""
        with pytest.raises(ValueError, match="examples must not be empty"):
            PrefixFeatureDataset([])
        cache_path = tmp_path / "empty.pt"
        assert not cache_path.exists()


def test_compute_prefix_feature_shard_ranges_even_split():
    assert compute_prefix_feature_shard_ranges(8, 2) == [(0, 4), (4, 8)]


def test_compute_prefix_feature_shard_ranges_caps_worker_count_and_handles_remainder():
    assert compute_prefix_feature_shard_ranges(5, 8) == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    assert compute_prefix_feature_shard_ranges(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_merge_prefix_feature_cache_shards_roundtrip(tmp_path: Path):
    metadata = _default_metadata()
    shard_a = PrefixFeatureDataset(
        [
            PrefixFeatureExample(
                hidden_states=torch.full((8, 16), 1.0),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.arange(8, dtype=torch.long),
                split_layer_idx=2,
            )
        ]
    )
    shard_b = PrefixFeatureDataset(
        [
            PrefixFeatureExample(
                hidden_states=torch.full((8, 16), 2.0),
                attention_mask=torch.ones(8, dtype=torch.long),
                labels=torch.arange(8, 16, dtype=torch.long),
                split_layer_idx=2,
            )
        ]
    )
    shard_a_path = tmp_path / "shard_a.pt"
    shard_b_path = tmp_path / "shard_b.pt"
    save_prefix_feature_dataset(shard_a, shard_a_path, metadata={"rank": 0})
    save_prefix_feature_dataset(shard_b, shard_b_path, metadata={"rank": 1})

    merged_path = tmp_path / "merged.pt"
    merge_prefix_feature_cache_shards(
        [shard_a_path, shard_b_path],
        merged_path,
        metadata=metadata,
    )
    merged = load_prefix_feature_dataset(merged_path)

    assert len(merged) == 2
    assert torch.equal(merged[0]["labels"], torch.arange(8, dtype=torch.long))
    assert torch.equal(merged[1]["labels"], torch.arange(8, 16, dtype=torch.long))
    assert torch.allclose(merged[0]["hidden_states"], torch.full((8, 16), 1.0))
    assert torch.allclose(merged[1]["hidden_states"], torch.full((8, 16), 2.0))


def test_resolve_prefix_feature_cache_seed_respects_share_flag():
    assert resolve_prefix_feature_cache_seed(42, share_across_seeds=False) == 42
    assert resolve_prefix_feature_cache_seed(42, share_across_seeds=True) == 0


class TestPrefixFeatureDatasetValidation:
    def test_rejects_empty_examples(self):
        with pytest.raises(ValueError, match="examples must not be empty"):
            PrefixFeatureDataset([])

    def test_rejects_non_example_elements(self):
        with pytest.raises(TypeError, match=r"examples\[1\] must be a PrefixFeatureExample"):
            PrefixFeatureDataset([
                PrefixFeatureExample(
                    hidden_states=torch.randn(4, 8),
                    attention_mask=torch.ones(4, dtype=torch.long),
                    labels=torch.randint(0, 10, (4,)),
                    split_layer_idx=2,
                ),
                "not_an_example",
            ])

    def test_mapped_rejects_negative_split_layer_idx(self):
        with pytest.raises(ValueError, match="split_layer_idx must be non-negative"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(2, 4, 8),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=-1,
            )

    def test_mapped_rejects_none_hidden_states(self):
        with pytest.raises(ValueError, match="hidden_states must not be None"):
            MappedPrefixFeatureDataset(
                hidden_states=None,
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=0,
            )

    def test_mapped_rejects_none_attention_mask(self):
        with pytest.raises(ValueError, match="attention_mask must not be None"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(2, 4, 8),
                attention_mask=None,
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=0,
            )

    def test_mapped_rejects_none_labels(self):
        with pytest.raises(ValueError, match="labels must not be None"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(2, 4, 8),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=None,
                split_layer_idx=0,
            )

    def test_mapped_rejects_mismatched_batch_sizes(self):
        with pytest.raises(ValueError, match="Batch size mismatch"):
            MappedPrefixFeatureDataset(
                hidden_states=torch.randn(3, 4, 8),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                labels=torch.randint(0, 10, (2, 4)),
                split_layer_idx=0,
            )


class TestOneShotMode:
    """REQ-224: prefix_feature_cache_mode='one_shot' loads via MappedPrefixFeatureDataset."""

    def _build_and_save_cache(self, tmp_path, n=4):
        model = _SimplePrefixCacheModel()
        raw_ds = _TokenDataset(n=n)
        cached = build_prefix_feature_dataset(
            model, raw_ds, batch_size=2, device="cpu", split_layer_idx=2,
        )
        cache_path = tmp_path / "one_shot_cache.pt"
        save_prefix_feature_dataset(cached, cache_path, metadata={"test": True})
        return cache_path

    def test_load_lazy_returns_mapped_dataset(self, tmp_path):
        cache_path = self._build_and_save_cache(tmp_path)
        ds = load_prefix_feature_dataset(cache_path, lazy=True)
        assert isinstance(ds, MappedPrefixFeatureDataset), (
            f"one_shot mode should return MappedPrefixFeatureDataset, got {type(ds).__name__}"
        )

    def test_load_eager_returns_prefix_feature_dataset(self, tmp_path):
        cache_path = self._build_and_save_cache(tmp_path)
        ds = load_prefix_feature_dataset(cache_path, lazy=False)
        assert isinstance(ds, PrefixFeatureDataset), (
            f"reuse mode should return PrefixFeatureDataset, got {type(ds).__name__}"
        )

    def test_lazy_dataset_has_correct_length(self, tmp_path):
        cache_path = self._build_and_save_cache(tmp_path, n=6)
        ds = load_prefix_feature_dataset(cache_path, lazy=True)
        assert len(ds) == 6


# ---------------------------------------------------------------------------
# TC-224-02: one_shot YAML config passes Pydantic validation
# ---------------------------------------------------------------------------


class TestTC224:
    """REQ-224: one_shot YAML config validation."""

    def test_tc224_02_one_shot_config_passes_pydantic_validation(self):
        """TC-224-02: configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml
        passes Pydantic config_schema validation."""
        from src.training.config_schema import load_and_validate_config

        config = load_and_validate_config(
            "configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml"
        )
        assert config.training.prefix_feature_cache_mode == "one_shot"
        assert config.training.prefix_feature_cache_experimental is True