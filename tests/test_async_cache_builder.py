import time

import pytest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.training.async_cache_builder import AsyncCacheBuilder, AsyncCacheBuildResult
from src.tg_lora.prefix_feature_cache import PrefixFeatureDataset


class _SimpleModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 16, layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        del kw
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm(h))
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
    def __init__(self, n: int = 4, seq_len: int = 8, vocab: int = 32):
        self.input_ids = torch.randint(0, vocab, (n, seq_len))
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


def _make_cfg(tmp_path, split_layer=2):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": {"name_or_path": "dummy", "device": "cpu", "dtype": "float32",
                  "load_in_4bit": False, "bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_compute_dtype": "float32", "device_map": "auto"},
        "data": {"train_path": str(tmp_path / "train.jsonl"),
                 "valid_quick_path": str(tmp_path / "vq.jsonl"),
                 "valid_full_path": str(tmp_path / "vf.jsonl"),
                 "max_seq_len": 8},
        "training": {"batch_size": 2, "prefix_feature_cache_valid_quick": True,
                     "prefix_feature_cache_valid_full": True},
        "experiment": {"seed": 42},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": "all-linear"},
    })


def _mock_model():
    model = _SimpleModel()
    model.eval()
    return model


def test_async_build_produces_cached_dataset(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    raw_vq = _TokenDataset(n=4)
    raw_vf = _TokenDataset(n=4)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch("src.training.async_cache_builder.load_base_model", return_value=_mock_model()), \
         patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m), \
         patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")):

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq, "valid_full": raw_vf},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=False,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder.start()
        builder.join(timeout=30)

    assert builder.poll()
    assert not builder.failed

    result_vq = builder.get_result("valid_quick")
    assert result_vq is not None
    assert result_vq.dataset is not None
    assert result_vq.error is None
    assert len(result_vq.dataset) == 4

    result_vf = builder.get_result("valid_full")
    assert result_vf is not None
    assert result_vf.dataset is not None


def test_async_build_disk_hit_skips_build(tmp_path: Path):
    from src.tg_lora.prefix_feature_cache import (
        PrefixFeatureExample, build_prefix_feature_cache_metadata,
        get_prefix_feature_cache_path, save_prefix_feature_dataset,
    )

    cfg = _make_cfg(tmp_path)
    raw_vq = _TokenDataset(n=2)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    metadata = build_prefix_feature_cache_metadata(
        dataset_path=str(tmp_path / "vq.jsonl"),
        model_name="dummy", seed=42, max_seq_len=8,
        split_layer_idx=2, lora_r=4, lora_alpha=8,
        lora_dropout=0.0, lora_target_modules="all-linear",
        trainable_lora_scope="last_25_percent",
    )
    cache_path = get_prefix_feature_cache_path(cache_dir, metadata)

    fake_dataset = PrefixFeatureDataset([
        PrefixFeatureExample(
            hidden_states=torch.randn(8, 16),
            attention_mask=torch.ones(8, dtype=torch.long),
            labels=torch.randint(0, 32, (8,)),
            split_layer_idx=2,
        )
    ])
    save_prefix_feature_dataset(fake_dataset, cache_path, metadata=metadata)

    with patch("src.training.async_cache_builder.load_base_model", return_value=_mock_model()), \
         patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m), \
         patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")):

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

    result = builder.get_result("valid_quick")
    assert result is not None
    assert result.source == "disk"
    assert result.dataset is not None


def test_async_build_error_sets_failed(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    raw_vq = _TokenDataset(n=2)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch("src.training.async_cache_builder.load_base_model", side_effect=RuntimeError("no GPU")):

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

    assert builder.failed
    assert builder.error is not None
    assert "no GPU" in str(builder.error)


def test_async_poll_is_nonblocking(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    raw_vq = _TokenDataset(n=2)
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
    )
    builder.start()

    t0 = time.perf_counter()
    builder.poll()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.1
    builder.join(timeout=30)


def test_async_get_result_nonexistent_label_returns_none(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    raw_vq = _TokenDataset(n=2)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch("src.training.async_cache_builder.load_base_model", return_value=_mock_model()), \
         patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m), \
         patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")):

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

    assert builder.get_result("nonexistent_label") is None


def test_async_force_rebuild_ignores_disk_cache(tmp_path: Path):
    from src.tg_lora.prefix_feature_cache import (
        PrefixFeatureExample, build_prefix_feature_cache_metadata,
        get_prefix_feature_cache_path, save_prefix_feature_dataset,
    )

    cfg = _make_cfg(tmp_path)
    raw_vq = _TokenDataset(n=2)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    metadata = build_prefix_feature_cache_metadata(
        dataset_path=str(tmp_path / "vq.jsonl"),
        model_name="dummy", seed=42, max_seq_len=8,
        split_layer_idx=2, lora_r=4, lora_alpha=8,
        lora_dropout=0.0, lora_target_modules="all-linear",
        trainable_lora_scope="last_25_percent",
    )
    cache_path = get_prefix_feature_cache_path(cache_dir, metadata)
    fake_dataset = PrefixFeatureDataset([
        PrefixFeatureExample(
            hidden_states=torch.randn(8, 16),
            attention_mask=torch.ones(8, dtype=torch.long),
            labels=torch.randint(0, 32, (8,)),
            split_layer_idx=2,
        )
    ])
    save_prefix_feature_dataset(fake_dataset, cache_path, metadata=metadata)

    with patch("src.training.async_cache_builder.load_base_model", return_value=_mock_model()), \
         patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m), \
         patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")):

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=True,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder.start()
        builder.join(timeout=30)

    result = builder.get_result("valid_quick")
    assert result is not None
    assert result.source == "built"


def test_async_partial_failure_continues_other_datasets(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    raw_vq = _TokenDataset(n=2)
    raw_vf = _TokenDataset(n=2)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch("src.training.async_cache_builder.load_base_model", return_value=_mock_model()), \
         patch("src.training.async_cache_builder.apply_lora", side_effect=lambda m, c: m), \
         patch("src.training.async_cache_builder.get_input_device", return_value=torch.device("cpu")), \
         patch.object(AsyncCacheBuilder, "_build_one", autospec=True) as mock_build:

        def side_effect(self_arg, model, device, label, raw_dataset, dataset_path):
            if label == "valid_quick":
                return AsyncCacheBuildResult(
                    label=label, dataset=None, build_seconds=0.0,
                    source="error", cache_path=Path(""), error=RuntimeError("build failed"),
                )
            return AsyncCacheBuildResult(
                label=label, dataset=None, build_seconds=0.1,
                source="built", cache_path=Path("ok"),
            )

        mock_build.side_effect = side_effect

        builder = AsyncCacheBuilder(
            cfg=cfg,
            raw_datasets={"valid_quick": raw_vq, "valid_full": raw_vf},
            cache_loader_kwargs={},
            split_layer=2,
            cache_dir=cache_dir,
            force_rebuild=True,
            trainable_lora_scope="last_25_percent",
            background_device="cpu",
        )
        builder.start()
        builder.join(timeout=30)

    assert not builder.failed
    result_vq = builder.get_result("valid_quick")
    assert result_vq is not None
    assert result_vq.error is not None
    assert result_vq.source == "error"


class TestAsyncCacheBuilderValidation:
    def test_rejects_invalid_device(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        with pytest.raises(ValueError, match="background_device must be a valid device string"):
            AsyncCacheBuilder(
                cfg=cfg,
                raw_datasets={"train": _TokenDataset()},
                cache_loader_kwargs={},
                split_layer=2,
                cache_dir=tmp_path / "cache",
                force_rebuild=False,
                trainable_lora_scope="last_25_percent",
                background_device="invalid",
            )

    def test_rejects_negative_split_layer(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        with pytest.raises(ValueError, match="split_layer must be non-negative"):
            AsyncCacheBuilder(
                cfg=cfg,
                raw_datasets={"train": _TokenDataset()},
                cache_loader_kwargs={},
                split_layer=-1,
                cache_dir=tmp_path / "cache",
                force_rebuild=False,
                trainable_lora_scope="last_25_percent",
                background_device="cpu",
            )

    def test_rejects_none_cfg(self, tmp_path):
        with pytest.raises(TypeError, match="cfg must be a DictConfig, not None"):
            AsyncCacheBuilder(
                cfg=None,
                raw_datasets={"train": _TokenDataset()},
                cache_loader_kwargs={},
                split_layer=2,
                cache_dir=tmp_path / "cache",
                force_rebuild=False,
                trainable_lora_scope="last_25_percent",
                background_device="cpu",
            )

    def test_rejects_empty_raw_datasets(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        with pytest.raises(ValueError, match="raw_datasets must be a non-empty dict"):
            AsyncCacheBuilder(
                cfg=cfg,
                raw_datasets={},
                cache_loader_kwargs={},
                split_layer=2,
                cache_dir=tmp_path / "cache",
                force_rebuild=False,
                trainable_lora_scope="last_25_percent",
                background_device="cpu",
            )

    def test_rejects_invalid_cache_dir(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        with pytest.raises(TypeError, match="cache_dir must be a Path"):
            AsyncCacheBuilder(
                cfg=cfg,
                raw_datasets={"train": _TokenDataset()},
                cache_loader_kwargs={},
                split_layer=2,
                cache_dir="not_a_path",
                force_rebuild=False,
                trainable_lora_scope="last_25_percent",
                background_device="cpu",
            )

    def test_rejects_empty_trainable_lora_scope(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        with pytest.raises(ValueError, match="trainable_lora_scope must be a non-empty string"):
            AsyncCacheBuilder(
                cfg=cfg,
                raw_datasets={"train": _TokenDataset()},
                cache_loader_kwargs={},
                split_layer=2,
                cache_dir=tmp_path / "cache",
                force_rebuild=False,
                trainable_lora_scope="",
                background_device="cpu",
            )
