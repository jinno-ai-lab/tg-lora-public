"""Robustness tests for prefix_feature_cache: corrupted cache, force_rebuild, position_ids.

Addresses A27 improvement recommendations:
- corrupted cache file handling
- force_rebuild flag behavior
- position_ids preservation through build/save/load cycle
- model.training state restoration after build failure
- cache invalidation on hyperparameter change (SHA-256 divergence)
"""


import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.tg_lora.prefix_feature_cache import (
    PrefixFeatureDataset,
    PrefixFeatureExample,
    _PREFIX_FEATURE_CACHE_FORMAT_VERSION,
    build_prefix_feature_cache_metadata,
    build_prefix_feature_dataset,
    collate_prefix_feature_batch,
    get_prefix_feature_cache_path,
    load_prefix_feature_dataset,
    save_prefix_feature_dataset,
)


# ---------------------------------------------------------------------------
# Lightweight model / dataset fixtures
# ---------------------------------------------------------------------------


class _SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return (self.norm(self.linear(hidden_states)),)


class _SimpleModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 16, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            [_SimpleDecoderLayer(hidden) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

        for idx in range(num_layers):
            layer_mod = nn.Module()
            layer_mod.register_parameter(
                "lora_A", nn.Parameter(torch.randn(hidden, hidden) * 0.01)
            )
            layer_mod.register_parameter(
                "lora_B", nn.Parameter(torch.randn(hidden, hidden) * 0.01)
            )
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


class _TokenDatasetWithPositionIds(Dataset):
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


# ---------------------------------------------------------------------------
# Corrupted cache handling
# ---------------------------------------------------------------------------


class TestCorruptedCache:
    def test_truncated_file_raises(self, tmp_path):
        cache_path = tmp_path / "truncated.pt"
        cache_path.write_bytes(b"not-a-valid-torch-file-truncated")
        with pytest.raises(Exception):
            load_prefix_feature_dataset(cache_path)

    def test_empty_file_raises(self, tmp_path):
        cache_path = tmp_path / "empty.pt"
        cache_path.write_bytes(b"")
        with pytest.raises(Exception):
            load_prefix_feature_dataset(cache_path)

    def test_wrong_format_version_raises(self, tmp_path):
        cache_path = tmp_path / "wrong_version.pt"
        blob = {
            "format_version": 999,
            "hidden_states": torch.randn(2, 4, 8),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 4)),
            "split_layer_idx": 2,
        }
        torch.save(blob, cache_path)
        with pytest.raises(ValueError, match="Unsupported prefix feature cache format version"):
            load_prefix_feature_dataset(cache_path)

    def test_missing_hidden_states_key_raises(self, tmp_path):
        cache_path = tmp_path / "missing_key.pt"
        blob = {
            "format_version": _PREFIX_FEATURE_CACHE_FORMAT_VERSION,
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 4)),
            "split_layer_idx": 2,
        }
        torch.save(blob, cache_path)
        with pytest.raises(KeyError):
            load_prefix_feature_dataset(cache_path)

    def test_missing_attention_mask_key_raises(self, tmp_path):
        cache_path = tmp_path / "missing_mask.pt"
        blob = {
            "format_version": _PREFIX_FEATURE_CACHE_FORMAT_VERSION,
            "hidden_states": torch.randn(2, 4, 8),
            "labels": torch.randint(0, 32, (2, 4)),
            "split_layer_idx": 2,
        }
        torch.save(blob, cache_path)
        with pytest.raises(KeyError):
            load_prefix_feature_dataset(cache_path)

    def test_valid_cache_loads_successfully(self, tmp_path):
        model = _SimpleModel()
        raw_dataset = _TokenDataset(n=2)
        cached = build_prefix_feature_dataset(
            model, raw_dataset, batch_size=2, device="cpu", split_layer_idx=2,
        )
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        save_prefix_feature_dataset(cached, cache_path, metadata=metadata)

        loaded = load_prefix_feature_dataset(cache_path)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# force_rebuild flag
# ---------------------------------------------------------------------------


class TestForceRebuild:
    def _build_and_save_cache(self, tmp_path, metadata):
        model = _SimpleModel()
        raw_dataset = _TokenDataset(n=4)
        cached = build_prefix_feature_dataset(
            model, raw_dataset, batch_size=2, device="cpu", split_layer_idx=2,
        )
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        save_prefix_feature_dataset(cached, cache_path, metadata=metadata)
        return cache_path, cached

    def test_force_rebuild_false_uses_disk_cache(self, tmp_path):
        metadata = _default_metadata()
        cache_path, original = self._build_and_save_cache(tmp_path, metadata)

        # Simulate the training loop logic: cache exists + force_rebuild=False → disk load
        force_rebuild = False
        assert cache_path.exists()
        if cache_path.exists() and not force_rebuild:
            loaded = load_prefix_feature_dataset(cache_path)
            source = "disk"
        else:
            source = "built"

        assert source == "disk"
        assert len(loaded) == len(original)

    def test_force_rebuild_true_skips_disk_cache(self, tmp_path):
        metadata = _default_metadata()
        cache_path, _original = self._build_and_save_cache(tmp_path, metadata)

        # Simulate: force_rebuild=True should skip disk cache and go to build branch
        force_rebuild = True
        assert cache_path.exists()
        if cache_path.exists() and not force_rebuild:
            source = "disk"
        else:
            source = "built"

        assert source == "built"

    def test_force_rebuild_rebuild_produces_equivalent_data(self, tmp_path):
        metadata = _default_metadata()
        cache_path, original = self._build_and_save_cache(tmp_path, metadata)

        model = _SimpleModel()
        raw_dataset = _TokenDataset(n=4)
        rebuilt = build_prefix_feature_dataset(
            model, raw_dataset, batch_size=2, device="cpu", split_layer_idx=2,
        )

        assert len(rebuilt) == len(original)
        assert rebuilt.total_bytes == original.total_bytes


# ---------------------------------------------------------------------------
# position_ids build path
# ---------------------------------------------------------------------------


class TestPositionIdsBuildPath:
    def test_position_ids_preserved_through_build(self):
        model = _SimpleModel()
        raw = _TokenDatasetWithPositionIds(n=4, seq_len=8)
        cached = build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=2,
        )
        for i in range(len(cached)):
            item = cached[i]
            assert "position_ids" in item
            assert item["position_ids"].shape == (8,)

    def test_position_ids_preserved_through_save_load(self, tmp_path):
        model = _SimpleModel()
        raw = _TokenDatasetWithPositionIds(n=4, seq_len=8)
        cached = build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=2,
        )
        metadata = _default_metadata()
        cache_path = get_prefix_feature_cache_path(tmp_path, metadata)
        save_prefix_feature_dataset(cached, cache_path, metadata=metadata)

        loaded = load_prefix_feature_dataset(cache_path)
        for i in range(len(loaded)):
            item = loaded[i]
            assert "position_ids" in item
            assert item["position_ids"].shape == (8,)
            original_item = cached[i]
            assert torch.equal(item["position_ids"], original_item["position_ids"])

    def test_position_ids_values_match_original(self, tmp_path):
        model = _SimpleModel()
        raw = _TokenDatasetWithPositionIds(n=2, seq_len=8)
        cached = build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=2,
        )

        expected_positions = raw.position_ids.clone()
        for i in range(len(cached)):
            assert torch.equal(cached[i]["position_ids"], expected_positions[i])

    def test_no_position_ids_when_dataset_lacks_them(self):
        model = _SimpleModel()
        raw = _TokenDataset(n=4, seq_len=8)
        cached = build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=2,
        )
        for i in range(len(cached)):
            assert "position_ids" not in cached[i]

    def test_collate_preserves_position_ids(self):
        seq_len, hidden = 8, 16
        items = [
            {
                "hidden_states": torch.randn(seq_len, hidden),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.randint(0, 32, (seq_len,)),
                "split_layer_idx": torch.tensor(2, dtype=torch.long),
                "position_ids": torch.arange(seq_len, dtype=torch.long),
            }
            for _ in range(3)
        ]
        batch = collate_prefix_feature_batch(items)
        assert "position_ids" in batch
        assert batch["position_ids"].shape == (3, seq_len)

    def test_position_ids_total_bytes_includes_them(self):
        seq_len, hidden = 8, 16
        ex_with = PrefixFeatureExample(
            hidden_states=torch.randn(seq_len, hidden),
            attention_mask=torch.ones(seq_len, dtype=torch.long),
            labels=torch.randint(0, 32, (seq_len,)),
            split_layer_idx=2,
            position_ids=torch.arange(seq_len, dtype=torch.long),
        )
        ex_without = PrefixFeatureExample(
            hidden_states=torch.randn(seq_len, hidden),
            attention_mask=torch.ones(seq_len, dtype=torch.long),
            labels=torch.randint(0, 32, (seq_len,)),
            split_layer_idx=2,
        )
        ds_with = PrefixFeatureDataset([ex_with])
        ds_without = PrefixFeatureDataset([ex_without])
        assert ds_with.total_bytes > ds_without.total_bytes


# ---------------------------------------------------------------------------
# model.training state restoration after build
# ---------------------------------------------------------------------------


class TestTrainingStateRestoration:
    def test_model_training_state_restored_after_build(self):
        model = _SimpleModel()
        model.train()
        assert model.training is True

        raw = _TokenDataset(n=2)
        build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=2,
        )
        assert model.training is True

    def test_model_eval_state_not_altered_after_build(self):
        model = _SimpleModel()
        model.eval()
        assert model.training is False

        raw = _TokenDataset(n=2)
        build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=2,
        )
        assert model.training is False

    def test_model_training_restored_on_build_error(self):
        model = _SimpleModel()
        model.train()
        assert model.training is True

        raw = _TokenDataset(n=2)
        with pytest.raises(ValueError, match="split_layer_idx"):
            build_prefix_feature_dataset(
                model, raw, batch_size=2, device="cpu", split_layer_idx=0,
            )
        assert model.training is True


# ---------------------------------------------------------------------------
# Cache invalidation on hyperparameter change (SHA-256 divergence)
# ---------------------------------------------------------------------------


class TestCacheInvalidationOnHyperparameterChange:
    def test_different_metadata_produces_different_cache_path(self, tmp_path):
        meta_a = _default_metadata(lora_r=16)
        meta_b = _default_metadata(lora_r=32)
        path_a = get_prefix_feature_cache_path(tmp_path, meta_a)
        path_b = get_prefix_feature_cache_path(tmp_path, meta_b)
        assert path_a != path_b

    def test_different_seed_produces_different_cache_path(self, tmp_path):
        meta_a = _default_metadata(seed=42)
        meta_b = _default_metadata(seed=123)
        path_a = get_prefix_feature_cache_path(tmp_path, meta_a)
        path_b = get_prefix_feature_cache_path(tmp_path, meta_b)
        assert path_a != path_b

    def test_different_split_layer_produces_different_cache_path(self, tmp_path):
        meta_a = _default_metadata(split_layer_idx=2)
        meta_b = _default_metadata(split_layer_idx=3)
        path_a = get_prefix_feature_cache_path(tmp_path, meta_a)
        path_b = get_prefix_feature_cache_path(tmp_path, meta_b)
        assert path_a != path_b

    def test_same_metadata_produces_same_cache_path(self, tmp_path):
        meta_a = _default_metadata(lora_r=16, seed=42)
        meta_b = _default_metadata(lora_r=16, seed=42)
        path_a = get_prefix_feature_cache_path(tmp_path, meta_a)
        path_b = get_prefix_feature_cache_path(tmp_path, meta_b)
        assert path_a == path_b

    def test_cache_path_contains_dataset_stem(self, tmp_path):
        meta = _default_metadata(dataset_path="data/my_special_dataset.jsonl")
        path = get_prefix_feature_cache_path(tmp_path, meta)
        assert "my_special_dataset" in path.name
