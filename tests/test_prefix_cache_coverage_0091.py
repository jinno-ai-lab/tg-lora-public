"""TASK-0091: coverage gap closure for prefix_feature_cache.py.

Covers uncovered lines: 117 (mixed position_ids), 243 (hook-failed-to-fire).
Lines 206/208-210 (num_workers>0 DataLoader path) and 219-220 (kwargs hook
fallback) require subprocess-level isolation and are documented as known gaps.
"""

from pathlib import Path

import pytest
import torch

from src.tg_lora.prefix_feature_cache import (
    PrefixFeatureDataset,
    PrefixFeatureExample,
    build_prefix_feature_dataset,
    save_prefix_feature_dataset,
)


def _default_metadata():
    from src.tg_lora.prefix_feature_cache import build_prefix_feature_cache_metadata

    return build_prefix_feature_cache_metadata(
        dataset_path="data/train.jsonl",
        model_name="dummy",
        seed=42,
        max_seq_len=8,
        split_layer_idx=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules="all-linear",
        trainable_lora_scope="last_25_percent",
    )


class TestMixedPositionIds:
    """Cover line 117: save_prefix_feature_dataset rejects mixed position_ids."""

    def test_mixed_position_ids_raises(self, tmp_path: Path):
        ex_with = PrefixFeatureExample(
            hidden_states=torch.randn(4, 8),
            attention_mask=torch.ones(4, dtype=torch.long),
            labels=torch.randint(0, 32, (4,)),
            split_layer_idx=2,
            position_ids=torch.arange(4),
        )
        ex_without = PrefixFeatureExample(
            hidden_states=torch.randn(4, 8),
            attention_mask=torch.ones(4, dtype=torch.long),
            labels=torch.randint(0, 32, (4,)),
            split_layer_idx=2,
            position_ids=None,
        )
        ds = PrefixFeatureDataset([ex_with, ex_without])
        cache_path = tmp_path / "mixed.pt"
        with pytest.raises(ValueError, match="mixed position_ids presence"):
            save_prefix_feature_dataset(ds, cache_path, metadata=_default_metadata())


class TestHookFailedToFire:
    """Cover line 243: RuntimeError when hook does not capture hidden states."""

    def test_no_capture_raises_runtime_error(self):
        from unittest.mock import patch

        from tests.conftest import TinyModel, TokenDataset

        model = TinyModel()
        ds = TokenDataset(n=2)
        # Patch _get_decoder_layers to return layers that won't fire the hook
        # because we use a split_layer_idx that points to a layer that receives
        # only kwargs-based input. We achieve this by patching the hook mechanism.
        with patch(
            "src.tg_lora.prefix_feature_cache._get_decoder_layers",
            return_value=model.layers,
        ):
            # Override model forward to not pass hidden_states through layers[2]
            original_forward = model.forward

            def _forward_no_layer2_fire(*args, **kwargs):
                # Call original but ensure layer at index 2 gets no args
                # This is tricky — we just make the model skip layer 2
                result = original_forward(*args, **kwargs)
                return result

            # Simpler: patch the model forward so it never calls layers[2]
            # but still runs layers[0], layers[1], layers[3]
            def _skip_layer2(input_ids=None, attention_mask=None, labels=None, **kw):
                h = model.embed_tokens(input_ids)
                for i, layer in enumerate(model.layers):
                    if i == 2:
                        continue  # skip the hooked layer
                    h = layer(h)
                logits = model.lm_head(model.norm(h))
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                return type("Out", (), {"loss": loss})()

            model.forward = _skip_layer2

            with pytest.raises(RuntimeError, match="Failed to capture prefix hidden states"):
                build_prefix_feature_dataset(
                    model,
                    ds,
                    batch_size=2,
                    device="cpu",
                    split_layer_idx=2,
                )
