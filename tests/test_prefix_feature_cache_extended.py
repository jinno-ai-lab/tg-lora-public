"""Extended tests for prefix_feature_cache edge cases (TASK-0073)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.tg_lora.prefix_feature_cache import (
    PrefixFeatureDataset,
    PrefixFeatureExample,
    build_prefix_feature_dataset,
    collate_prefix_feature_batch,
)


# ---------------------------------------------------------------------------
# Lightweight model / dataset fixtures (mirrors test_prefix_feature_cache.py)
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


# ---------------------------------------------------------------------------
# max_batches limit
# ---------------------------------------------------------------------------


def test_max_batches_limits_examples():
    """max_batches=1 should produce at most batch_size examples."""
    model = _SimpleModel(num_layers=4)
    raw = _TokenDataset(n=6, seq_len=8)
    batch_size = 2

    cached = build_prefix_feature_dataset(
        model, raw, batch_size=batch_size, device="cpu",
        split_layer_idx=2, max_batches=1,
    )
    assert len(cached) <= batch_size


# ---------------------------------------------------------------------------
# split_layer_idx boundary
# ---------------------------------------------------------------------------


def test_split_layer_zero_raises():
    """split_layer_idx=0 must raise ValueError."""
    model = _SimpleModel(num_layers=4)
    raw = _TokenDataset(n=2)
    with pytest.raises(ValueError, match="split_layer_idx"):
        build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=0,
        )


def test_split_layer_equal_num_layers_raises():
    """split_layer_idx == num_layers must raise ValueError."""
    model = _SimpleModel(num_layers=4)
    raw = _TokenDataset(n=2)
    with pytest.raises(ValueError, match="split_layer_idx"):
        build_prefix_feature_dataset(
            model, raw, batch_size=2, device="cpu", split_layer_idx=4,
        )


def test_split_layer_first_valid():
    """split_layer_idx=1 (first valid) should succeed."""
    model = _SimpleModel(num_layers=4)
    raw = _TokenDataset(n=2)
    cached = build_prefix_feature_dataset(
        model, raw, batch_size=2, device="cpu", split_layer_idx=1,
    )
    assert len(cached) == 2


def test_split_layer_last_valid():
    """split_layer_idx=num_layers-1 (last valid) should succeed."""
    model = _SimpleModel(num_layers=4)
    raw = _TokenDataset(n=2)
    cached = build_prefix_feature_dataset(
        model, raw, batch_size=2, device="cpu", split_layer_idx=3,
    )
    assert len(cached) == 2


# ---------------------------------------------------------------------------
# collate_prefix_feature_batch without position_ids
# ---------------------------------------------------------------------------


def test_collate_without_position_ids():
    """collate should succeed when items lack position_ids."""
    seq_len, hidden = 8, 16
    items = [
        {
            "hidden_states": torch.randn(seq_len, hidden),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "labels": torch.randint(0, 32, (seq_len,)),
            "split_layer_idx": torch.tensor(2, dtype=torch.long),
        }
        for _ in range(3)
    ]
    batch = collate_prefix_feature_batch(items)
    assert batch["hidden_states"].shape == (3, seq_len, hidden)
    assert "position_ids" not in batch


# ---------------------------------------------------------------------------
# PrefixFeatureDataset.total_bytes
# ---------------------------------------------------------------------------


def _make_example(seq_len: int = 8, hidden: int = 16, vocab: int = 32):
    return PrefixFeatureExample(
        hidden_states=torch.randn(seq_len, hidden),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        labels=torch.randint(0, vocab, (seq_len,)),
        split_layer_idx=2,
    )


def test_total_bytes_positive():
    """total_bytes should be > 0 for a non-empty dataset."""
    ds = PrefixFeatureDataset([_make_example()])
    assert ds.total_bytes > 0


def test_total_bytes_matches_manual_calculation():
    """total_bytes must equal the sum of element_size * numel across all tensors."""
    examples = [_make_example(seq_len=8, hidden=16, vocab=32) for _ in range(3)]
    ds = PrefixFeatureDataset(examples)

    expected = 0
    for ex in examples:
        expected += ex.hidden_states.numel() * ex.hidden_states.element_size()
        expected += ex.attention_mask.numel() * ex.attention_mask.element_size()
        expected += ex.labels.numel() * ex.labels.element_size()
    assert ds.total_bytes == expected


def test_total_bytes_includes_position_ids():
    """total_bytes must account for optional position_ids tensors."""
    seq_len, hidden = 8, 16
    ex = PrefixFeatureExample(
        hidden_states=torch.randn(seq_len, hidden),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        labels=torch.randint(0, 32, (seq_len,)),
        split_layer_idx=2,
        position_ids=torch.arange(seq_len, dtype=torch.long),
    )
    ds = PrefixFeatureDataset([ex])
    expected = (
        ex.hidden_states.numel() * ex.hidden_states.element_size()
        + ex.attention_mask.numel() * ex.attention_mask.element_size()
        + ex.labels.numel() * ex.labels.element_size()
        + ex.position_ids.numel() * ex.position_ids.element_size()
    )
    assert ds.total_bytes == expected
