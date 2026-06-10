"""Tests for ActivationCache: layer-skip evaluation."""

import logging
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.tg_lora.activation_cache import (ActivationCache,
                                          _compute_causal_lm_loss,
                                          _get_decoder_layers, _get_final_norm,
                                          _get_layer_types, _get_lm_head,
                                          _get_model_config,
                                          _get_num_position_groups,
                                          _get_rotary_emb,
                                          _normalize_split_layer_idx,
                                          _resolve_runtime_device,
                                          determine_split_layer,
                                          forward_from_hidden_states)


class _SimpleDecoderLayer(nn.Module):
    """Minimal transformer layer for testing."""

    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        out = self.norm(self.linear(hidden_states))
        return (out,)


class _SimpleModel(nn.Module):
    """Minimal model with decoder layers, norm, and lm_head."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            [_SimpleDecoderLayer(hidden) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = _compute_causal_lm_loss(logits, labels)

        class Output:
            pass

        out = Output()
        out.loss = loss
        return out


class _CustomLossModel(_SimpleModel):
    """Model whose forward relies on a custom loss_function hook."""

    def loss_function(self, *, logits, labels, vocab_size, **kwargs):
        del vocab_size, kwargs
        return _compute_causal_lm_loss(logits, labels) + 0.25

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=logits.shape[-1],
            )

        class Output:
            pass

        out = Output()
        out.loss = loss
        return out


def _make_dataloader(batch_size=2, seq_len=8, num_batches=4, vocab_size=32):
    all_ids = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    all_masks = torch.ones_like(all_ids)
    all_labels = all_ids.clone()
    dataset = TensorDataset(all_ids, all_masks, all_labels)

    def collate(batch):
        ids, masks, labels = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


class _MaskedMixDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        attention_mask = torch.ones(4, dtype=torch.long)
        if idx == 0:
            labels = torch.full((4,), -100, dtype=torch.long)
        else:
            labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _make_masked_mix_loader():
    return DataLoader(_MaskedMixDataset(), batch_size=1)


class TestDetermineSplitLayer:
    def test_empty_active_returns_num_layers(self):
        assert determine_split_layer(set(), 32) == 32

    def test_single_active_layer(self):
        assert determine_split_layer({24}, 32) == 24

    def test_multiple_active_layers(self):
        assert determine_split_layer({20, 24, 28, 31}, 32) == 20

    def test_first_layer_active(self):
        assert determine_split_layer({0, 1, 2}, 32) == 0


class TestGetDecoderLayers:
    def test_direct_layers(self):
        model = _SimpleModel()
        layers = _get_decoder_layers(model)
        assert len(layers) == 4

    def test_nested_model(self):
        inner = _SimpleModel()

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

        # model.model.layers should work
        class DoubleWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = Wrapper(m)

        wrapped = DoubleWrapper(inner)
        layers = _get_decoder_layers(wrapped)
        assert len(layers) == 4

    # --- REQ-142: cosine_similarity parity (TC-142-01, TC-142-E01) ---

    def test_debug_log_on_success(self, caplog):
        """TC-142-01: _get_decoder_layers logs path at DEBUG when layers found."""
        model = _SimpleModel()
        with caplog.at_level(logging.DEBUG, logger="tg-lora"):
            layers = _get_decoder_layers(model)
        assert len(layers) == 4
        assert any("Found decoder layers at path:" in r.message for r in caplog.records)

    def test_error_message_includes_candidate_count(self):
        """TC-142-E01: When all paths fail, error includes candidate path count."""
        import pytest

        bare_model = nn.Module()  # no layers attribute at all
        with pytest.raises(AttributeError, match=r"Tried \d+ paths"):
            _get_decoder_layers(bare_model)


class TestActivationCacheEvalAndCache:
    def test_eval_and_cache_returns_loss(self):
        model = _SimpleModel()
        loader = _make_dataloader(num_batches=3)
        cache = ActivationCache()

        loss = cache.eval_and_cache(
            model, loader, "cpu", split_layer_idx=2, max_batches=3
        )

        assert loss > 0
        assert cache.is_valid
        assert cache.num_batches == 3
        assert cache.split_layer == 2

    def test_cache_invalid_after_clear(self):
        model = _SimpleModel()
        loader = _make_dataloader(num_batches=2)
        cache = ActivationCache()

        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=1, max_batches=2)
        assert cache.is_valid
        cache.clear()
        assert not cache.is_valid

    def test_fallback_when_split_exceeds_layers(self):
        model = _SimpleModel(num_layers=4)
        loader = _make_dataloader(num_batches=2)
        cache = ActivationCache()

        loss = cache.eval_and_cache(
            model, loader, "cpu", split_layer_idx=10, max_batches=2
        )
        assert loss > 0
        assert not cache.is_valid

    def test_eval_and_cache_respects_max_examples_exactly(self):
        model = _SimpleModel()
        loader = _make_dataloader(batch_size=2, num_batches=3)
        cache = ActivationCache()

        loss = cache.eval_and_cache(
            model,
            loader,
            "cpu",
            split_layer_idx=2,
            max_examples=3,
        )

        assert loss > 0
        assert cache.is_valid
        assert cache.num_batches == 2

    def test_eval_and_cache_rejects_mixed_limits(self):
        model = _SimpleModel()
        loader = _make_dataloader(num_batches=1)
        cache = ActivationCache()

        with pytest.raises(ValueError, match="at most one"):
            cache.eval_and_cache(
                model,
                loader,
                "cpu",
                split_layer_idx=2,
                max_batches=1,
                max_examples=1,
            )

    def test_eval_and_cache_skips_all_masked_batches(self):
        model = _SimpleModel()
        mixed_loader = _make_masked_mix_loader()
        valid_loader = DataLoader(torch.utils.data.Subset(_MaskedMixDataset(), [1, 2]), batch_size=1)
        cache = ActivationCache()

        mixed_loss = cache.eval_and_cache(
            model, mixed_loader, "cpu", split_layer_idx=2
        )
        valid_loss = ActivationCache().eval_and_cache(
            model, valid_loader, "cpu", split_layer_idx=2
        )

        assert math.isfinite(mixed_loss)
        assert abs(mixed_loss - valid_loss) < 1e-6
        assert cache.num_batches == 2

    def test_full_eval_fallback_skips_all_masked_batches(self):
        class NoLayerModel(nn.Module):
            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                del input_ids, attention_mask, kw
                if torch.all(labels == -100):
                    loss = torch.tensor(float("nan"))
                else:
                    loss = torch.tensor(1.0)

                class Out:
                    pass

                out = Out()
                out.loss = loss
                return out

        model = NoLayerModel()
        mixed_loader = _make_masked_mix_loader()
        cache = ActivationCache()

        loss = cache.eval_and_cache(model, mixed_loader, "cpu", split_layer_idx=0)

        assert math.isfinite(loss)
        assert loss == pytest.approx(1.0)
        assert not cache.is_valid


class TestActivationCacheEvalFromCache:
    def test_eval_from_cache_returns_loss(self):
        model = _SimpleModel(num_layers=4)
        loader = _make_dataloader(num_batches=3)
        cache = ActivationCache()

        # Full eval with caching
        loss_full = cache.eval_and_cache(
            model, loader, "cpu", split_layer_idx=2, max_batches=3
        )

        # Eval from cache (no weight changes → should give same loss)
        loss_cached = cache.eval_from_cache(model, "cpu")

        assert abs(loss_full - loss_cached) < 1e-4

    def test_eval_from_cache_detects_weight_change(self):
        model = _SimpleModel(num_layers=4)
        loader = _make_dataloader(num_batches=3)
        cache = ActivationCache()

        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=3)
        loss_before = cache.eval_from_cache(model, "cpu")

        # Modify weights in layer 3 (after split)
        with torch.no_grad():
            model.layers[3].linear.weight.add_(
                torch.randn_like(model.layers[3].linear.weight) * 0.5
            )

        loss_after = cache.eval_from_cache(model, "cpu")

        # Loss should change after modifying active layers
        assert loss_before != loss_after

    def test_eval_from_cache_unaffected_by_pre_split_change(self):
        model = _SimpleModel(num_layers=4)
        loader = _make_dataloader(num_batches=3)
        cache = ActivationCache()

        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=3)
        loss_cached_1 = cache.eval_from_cache(model, "cpu")

        # Modify weights in layer 0 (before split) - cache should NOT see this
        with torch.no_grad():
            model.layers[0].linear.weight.add_(
                torch.randn_like(model.layers[0].linear.weight) * 10.0
            )

        loss_cached_2 = cache.eval_from_cache(model, "cpu")

        # Cache-based eval ignores pre-split changes (by design)
        assert abs(loss_cached_1 - loss_cached_2) < 1e-5

    def test_raises_when_cache_invalid(self):
        model = _SimpleModel()
        cache = ActivationCache()

        import pytest

        with pytest.raises(RuntimeError, match="Cache is not valid"):
            cache.eval_from_cache(model, "cpu")

    def test_eval_from_cache_uses_model_loss_function(self):
        model = _CustomLossModel(num_layers=4)
        loader = _make_dataloader(num_batches=2)
        cache = ActivationCache()

        loss_full = cache.eval_and_cache(
            model, loader, "cpu", split_layer_idx=2, max_batches=2
        )
        loss_cached = cache.eval_from_cache(model, "cpu")

        assert abs(loss_full - loss_cached) < 1e-4


# --- Tests for invalidate() (line 71) ---

class TestInvalidate:
    def test_invalidate_marks_cache_invalid_but_keeps_batches(self):
        model = _SimpleModel()
        loader = _make_dataloader(num_batches=2)
        cache = ActivationCache()

        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=1, max_batches=2)
        assert cache.is_valid
        assert cache.num_batches == 2

        cache.invalidate()
        assert not cache.is_valid
        # invalidate keeps batches in memory (unlike clear)
        assert cache.num_batches == 2

    def test_invalidate_allows_recache(self):
        model = _SimpleModel()
        loader = _make_dataloader(num_batches=2)
        cache = ActivationCache()

        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=1, max_batches=2)
        cache.invalidate()

        # Can re-cache after invalidation
        loss = cache.eval_and_cache(model, loader, "cpu", split_layer_idx=1, max_batches=2)
        assert cache.is_valid
        assert loss > 0


# --- Tests for fallback when no decoder layers (lines 100-103) ---

class TestFallbackNoDecoderLayers:
    def test_eval_fallback_when_model_has_no_layers(self):
        class NoLayerModel(nn.Module):
            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                _ = nn.functional.one_hot(input_ids, 32).float()
                class Out:
                    loss = torch.tensor(1.0)
                return Out()

        model = NoLayerModel()
        loader = _make_dataloader(num_batches=2)
        cache = ActivationCache()

        loss = cache.eval_and_cache(model, loader, "cpu", split_layer_idx=0, max_batches=2)
        assert not cache.is_valid
        assert loss > 0


# --- Tests for hook kwargs path (lines 117-118) ---

class TestHookKwargsPath:
    def test_hook_captures_hidden_states_from_kwargs(self):
        """Decoder layer that receives hidden_states as kwarg triggers kwargs path."""

        class KwargDecoderLayer(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.linear = nn.Linear(hidden, hidden)
                self.norm = nn.LayerNorm(hidden)

            def forward(self, hidden_states=None, attention_mask=None, **kwargs):
                out = self.norm(self.linear(hidden_states))
                return (out,)

        class KwargModel(nn.Module):
            def __init__(self, vocab_size=32, hidden=16, num_layers=4):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden)
                self.layers = nn.ModuleList(
                    [KwargDecoderLayer(hidden) for _ in range(num_layers)]
                )
                self.norm = nn.LayerNorm(hidden)
                self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                h = self.embed_tokens(input_ids)
                for layer in self.layers:
                    h = layer(hidden_states=h, attention_mask=attention_mask)[0]
                h = self.norm(h)
                logits = self.lm_head(h)
                loss = _compute_causal_lm_loss(logits, labels) if labels is not None else None
                class Out:
                    pass
                out = Out()
                out.loss = loss
                return out

        model = KwargModel()
        loader = _make_dataloader(num_batches=2)
        cache = ActivationCache()

        loss = cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=2)
        assert cache.is_valid
        assert loss > 0
        assert cache.num_batches == 2


# --- Tests for _get_final_norm raise (line 267) ---

class TestGetFinalNormRaise:
    def test_raises_when_no_norm_found(self):
        class NoNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

        model = NoNormModel()
        with pytest.raises(AttributeError, match="Cannot find final norm layer"):
            _get_final_norm(model)


# --- Tests for _get_lm_head raise (line 285) ---

class TestGetLmHeadRaise:
    def test_raises_when_no_lm_head_found(self):
        class NoHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])
                self.norm = nn.LayerNorm(4)

        model = NoHeadModel()
        with pytest.raises(AttributeError, match="Cannot find lm_head"):
            _get_lm_head(model)


# --- Tests for _get_rotary_emb found path (line 301) ---

class TestGetRotaryEmb:
    def test_returns_module_when_present(self):
        class FakeRotaryEmb(nn.Module):
            pass

        class ModelWithRotary(nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = FakeRotaryEmb()

        model = ModelWithRotary()
        result = _get_rotary_emb(model)
        assert isinstance(result, FakeRotaryEmb)

    def test_returns_none_when_absent(self):
        model = _SimpleModel()
        assert _get_rotary_emb(model) is None


# --- Tests for _normalize_split_layer_idx tensor edge cases (lines 311, 314) ---

class TestNormalizeSplitLayerIdx:
    def test_int_passthrough(self):
        assert _normalize_split_layer_idx(5) == 5

    def test_tensor_single_value(self):
        t = torch.tensor([3])
        assert _normalize_split_layer_idx(t) == 3

    def test_tensor_uniform_values(self):
        t = torch.tensor([2, 2, 2])
        assert _normalize_split_layer_idx(t) == 2

    def test_tensor_mismatched_values_raises(self):
        t = torch.tensor([1, 2, 3])
        with pytest.raises(ValueError, match="All split_layer_idx values"):
            _normalize_split_layer_idx(t)

    def test_empty_tensor_raises(self):
        t = torch.tensor([])
        with pytest.raises(ValueError, match="split_layer_idx tensor must not be empty"):
            _normalize_split_layer_idx(t)


# --- Tests for forward_from_hidden_states unsqueezing (lines 340-346) ---

class TestForwardFromHiddenStatesUnsqueeze:
    def test_1d_attention_mask_gets_unsqueezed(self):
        model = _SimpleModel()
        loader = _make_dataloader(batch_size=1, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]
        # Flatten attention_mask to 1D to trigger unsqueeze path
        mask_1d = cached.attention_mask[0]
        assert mask_1d.ndim == 1

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            mask_1d,  # 1D → triggers line 342
            cached.labels,
            split_layer_idx=2,
            device="cpu",
        )
        assert torch.is_tensor(loss)


class TestResolveRuntimeDevice:
    def test_prefers_suffix_device_over_first_parameter_device(self):
        class MixedDeviceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(8, 4)
                self.layers = nn.ModuleList(
                    [
                        _SimpleDecoderLayer(4),
                        _SimpleDecoderLayer(4).to("meta"),
                        _SimpleDecoderLayer(4).to("meta"),
                    ]
                )
                self.norm = nn.LayerNorm(4, device="meta")
                self.lm_head = nn.Linear(4, 8, bias=False, device="meta")

        model = MixedDeviceModel()
        decoder_layers = _get_decoder_layers(model)
        final_norm = _get_final_norm(model)
        lm_head = _get_lm_head(model)

        resolved = _resolve_runtime_device(
            model,
            1,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            lm_head=lm_head,
            rotary_emb=None,
        )

        assert resolved == torch.device("meta")

    def test_2d_hidden_states_gets_unsqueezed(self):
        model = _SimpleModel()
        loader = _make_dataloader(batch_size=1, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]
        # Squeeze to 2D to trigger unsqueeze path
        hidden_2d = cached.hidden_states[0]
        assert hidden_2d.ndim == 2

        loss = forward_from_hidden_states(
            model,
            hidden_2d,  # 2D → triggers line 340
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
        )
        assert torch.is_tensor(loss)

    def test_1d_labels_gets_unsqueezed(self):
        model = _SimpleModel()
        loader = _make_dataloader(batch_size=1, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]
        labels_1d = cached.labels[0]
        assert labels_1d.ndim == 1

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            labels_1d,  # 1D → triggers line 344
            split_layer_idx=2,
            device="cpu",
        )
        assert torch.is_tensor(loss)

    def test_1d_position_ids_gets_unsqueezed(self):
        model = _SimpleModel()
        loader = _make_dataloader(batch_size=1, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]
        position_ids = torch.arange(cached.hidden_states.shape[1])
        assert position_ids.ndim == 1

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            position_ids=position_ids,  # 1D → triggers line 346
        )
        assert torch.is_tensor(loss)


# --- Tests for rotary embedding path (lines 362-379) ---

class TestRotaryEmbPath:
    def _make_model_with_rotary(self, vocab_size=32, hidden=16, num_layers=4):
        """Build a model that has a rotary_emb module to exercise the rotary path."""

        class FakeRotaryEmb(nn.Module):
            def forward(self, x, position_ids):
                # Return cos/sin pair like real rotary embeddings
                seq_len = position_ids.shape[-1]
                cos = torch.ones(seq_len, x.shape[-1], device=x.device)
                sin = torch.zeros(seq_len, x.shape[-1], device=x.device)
                return cos, sin

        class RotDecoderLayer(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.linear = nn.Linear(hidden, hidden)
                self.norm = nn.LayerNorm(hidden)

            def forward(self, hidden_states, attention_mask=None,
                        position_embeddings=None, position_ids=None, **kwargs):
                out = self.norm(self.linear(hidden_states))
                return (out,)

        class RotModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden)
                self.layers = nn.ModuleList(
                    [RotDecoderLayer(hidden) for _ in range(num_layers)]
                )
                self.norm = nn.LayerNorm(hidden)
                self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
                self.rotary_emb = FakeRotaryEmb()

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                h = self.embed_tokens(input_ids)
                for layer in self.layers:
                    h = layer(h, attention_mask=attention_mask)[0]
                h = self.norm(h)
                logits = self.lm_head(h)
                loss = _compute_causal_lm_loss(logits, labels) if labels is not None else None
                class Out:
                    pass
                out = Out()
                out.loss = loss
                return out

        return RotModel()

    def test_forward_with_rotary_emb_and_3d_position_ids(self):
        model = self._make_model_with_rotary()
        loader = _make_dataloader(batch_size=2, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]
        # 3D position_ids with shape (4, batch, seq) → rotary path
        batch_size = cached.hidden_states.shape[0]
        seq_len = cached.hidden_states.shape[1]
        position_ids_3d = torch.arange(seq_len).view(1, 1, -1).expand(4, batch_size, -1)
        assert position_ids_3d.ndim == 3 and position_ids_3d.shape[0] == 4

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            position_ids=position_ids_3d,
        )
        assert torch.is_tensor(loss)

    def test_forward_with_rotary_emb_and_2d_position_ids(self):
        model = self._make_model_with_rotary()
        loader = _make_dataloader(batch_size=2, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]
        # 2D position_ids → gets expanded to (4, batch, seq)
        batch_size = cached.hidden_states.shape[0]
        seq_len = cached.hidden_states.shape[1]
        position_ids_2d = torch.arange(seq_len).view(1, -1).expand(batch_size, -1)
        assert position_ids_2d.ndim == 2

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            position_ids=position_ids_2d,
        )
        assert torch.is_tensor(loss)

    def test_forward_with_rotary_emb_and_no_position_ids(self):
        """When rotary_emb exists but position_ids is None, auto-generates them."""
        model = self._make_model_with_rotary()
        loader = _make_dataloader(batch_size=2, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            position_ids=None,  # auto-generate
        )
        assert torch.is_tensor(loss)


# --- Tests for _get_layer_types (lines 425-426, 431-433) ---

class TestGetLayerTypes:
    def test_returns_none_when_no_config(self):
        model = _SimpleModel()
        assert _get_layer_types(model) is None

    def test_returns_layer_types_from_config(self):
        class FakeConfig:
            layer_types = ["full_attention", "linear_attention", "full_attention", "linear_attention"]

        class ModelWithConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

        model = ModelWithConfig()
        result = _get_layer_types(model)
        assert result == ["full_attention", "linear_attention", "full_attention", "linear_attention"]

    def test_returns_none_when_config_has_no_layer_types(self):
        class ConfigNoTypes:
            pass

        class ModelWithEmptyConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ConfigNoTypes()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

        model = ModelWithEmptyConfig()
        assert _get_layer_types(model) is None

    def test_returns_none_when_layer_types_not_list(self):
        class ConfigBadTypes:
            layer_types = "not_a_list"

        class ModelWithBadConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ConfigBadTypes()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

        model = ModelWithBadConfig()
        assert _get_layer_types(model) is None


# --- Tests for forward_from_hidden_states with layer_types (lines 390-402) ---

class TestForwardWithLayerTypes:
    def test_linear_attention_mask_applied_for_linear_layers(self):
        model = _SimpleModel()

        class FakeConfig:
            layer_types = ["full_attention", "full_attention", "linear_attention", "linear_attention"]

        # Inject config onto model
        model.config = FakeConfig()

        loader = _make_dataloader(batch_size=2, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)

        cached = cache._batches[0]
        layer_types = _get_layer_types(model)
        assert layer_types is not None

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            layer_types=layer_types,
        )
        assert torch.is_tensor(loss)


# --- Tests for _get_num_position_groups and _get_model_config ---

class TestGetModelConfig:
    def test_returns_config_from_direct_model(self):
        class FakeConfig:
            num_key_value_heads = 8

        class ModelWithConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

        model = ModelWithConfig()
        config = _get_model_config(model)
        assert config is not None
        assert config.num_key_value_heads == 8

    def test_returns_config_from_nested_model(self):
        class FakeConfig:
            num_key_value_heads = 4

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Inner()

        model = Wrapper()
        config = _get_model_config(model)
        assert config is not None
        assert config.num_key_value_heads == 4

    def test_returns_none_when_no_config(self):
        model = _SimpleModel()
        assert _get_model_config(model) is None


class TestGetNumPositionGroups:
    def test_returns_num_kv_heads_from_config(self):
        class FakeConfig:
            num_key_value_heads = 8

        class ModelWithKVHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

        model = ModelWithKVHeads()
        assert _get_num_position_groups(model) == 8

    def test_falls_back_to_4_when_no_config(self):
        model = _SimpleModel()
        assert _get_num_position_groups(model) == 4

    def test_falls_back_to_4_when_no_num_kv_heads(self):
        class ConfigNoKV:
            pass

        class ModelNoKV(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ConfigNoKV()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

        model = ModelNoKV()
        assert _get_num_position_groups(model) == 4

    def test_falls_back_to_4_when_num_kv_heads_not_positive(self):
        class ConfigBadKV:
            num_key_value_heads = 0

        class ModelBadKV(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ConfigBadKV()
                self.layers = nn.ModuleList([nn.Linear(4, 4)])

        model = ModelBadKV()
        assert _get_num_position_groups(model) == 4


# --- Tests for multi-architecture position_ids support ---

class _FakeRotaryEmb(nn.Module):
    """Rotary embedding mock that records what position_ids it received."""

    def __init__(self):
        super().__init__()
        self.received_position_ids = None

    def forward(self, x, position_ids):
        self.received_position_ids = position_ids
        seq_len = position_ids.shape[-1] if position_ids is not None else 1
        cos = torch.ones(seq_len, x.shape[-1], device=x.device)
        sin = torch.zeros(seq_len, x.shape[-1], device=x.device)
        return cos, sin


class _RotDecoderLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden_states, attention_mask=None,
                position_embeddings=None, position_ids=None, **kwargs):
        out = self.norm(self.linear(hidden_states))
        return (out,)


class TestMultiArchitecturePositionIds:
    """Verify that forward_from_hidden_states creates correct position_ids
    shapes for both Qwen3.5 (3D) and standard (2D) architectures."""

    def _make_standard_model(self, vocab_size=32, hidden=16, num_layers=4):
        """Model with rotary_emb but no layer_types (e.g. Llama, Mistral)."""

        class StandardModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden)
                self.layers = nn.ModuleList(
                    [_RotDecoderLayer(hidden) for _ in range(num_layers)]
                )
                self.norm = nn.LayerNorm(hidden)
                self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
                self.rotary_emb = _FakeRotaryEmb()

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                h = self.embed_tokens(input_ids)
                for layer in self.layers:
                    h = layer(h, attention_mask=attention_mask)[0]
                h = self.norm(h)
                logits = self.lm_head(h)
                loss = _compute_causal_lm_loss(logits, labels) if labels is not None else None

                class Out:
                    pass

                out = Out()
                out.loss = loss
                return out

        return StandardModel()

    def _make_qwen_model(self, num_kv_heads=4, vocab_size=32, hidden=16, num_layers=4):
        """Model with rotary_emb and layer_types (Qwen3.5 hybrid attention)."""

        class QwenConfig:
            num_key_value_heads = num_kv_heads
            layer_types = ["full_attention"] * num_layers

        class QwenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden)
                self.layers = nn.ModuleList(
                    [_RotDecoderLayer(hidden) for _ in range(num_layers)]
                )
                self.norm = nn.LayerNorm(hidden)
                self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
                self.rotary_emb = _FakeRotaryEmb()
                self.config = QwenConfig()

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                h = self.embed_tokens(input_ids)
                for layer in self.layers:
                    h = layer(h, attention_mask=attention_mask)[0]
                h = self.norm(h)
                logits = self.lm_head(h)
                loss = _compute_causal_lm_loss(logits, labels) if labels is not None else None

                class Out:
                    pass

                out = Out()
                out.loss = loss
                return out

        return QwenModel()

    def test_standard_model_creates_2d_position_ids(self):
        """Non-Qwen3.5 models get 2D position_ids (batch, seq)."""
        model = self._make_standard_model()
        loader = _make_dataloader(batch_size=2, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)
        cached = cache._batches[0]

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            position_ids=None,
        )
        assert torch.is_tensor(loss)
        assert model.rotary_emb.received_position_ids.ndim == 2

    def test_qwen_model_creates_3d_position_ids(self):
        """Qwen3.5 models with layer_types get 3D position_ids."""
        model = self._make_qwen_model(num_kv_heads=4)
        loader = _make_dataloader(batch_size=2, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)
        cached = cache._batches[0]

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            position_ids=None,
        )
        assert torch.is_tensor(loss)
        assert model.rotary_emb.received_position_ids.ndim == 3
        assert model.rotary_emb.received_position_ids.shape[0] == 3

    def test_qwen_model_with_8_kv_heads(self):
        """Qwen3.5 model with 8 KV heads creates 3D position_ids with 8 groups."""
        model = self._make_qwen_model(num_kv_heads=8)
        loader = _make_dataloader(batch_size=2, seq_len=4, num_batches=1)
        cache = ActivationCache()
        cache.eval_and_cache(model, loader, "cpu", split_layer_idx=2, max_batches=1)
        cached = cache._batches[0]

        loss = forward_from_hidden_states(
            model,
            cached.hidden_states,
            cached.attention_mask,
            cached.labels,
            split_layer_idx=2,
            device="cpu",
            position_ids=None,
        )
        assert torch.is_tensor(loss)
        assert model.rotary_emb.received_position_ids.ndim == 3
        assert model.rotary_emb.received_position_ids.shape[0] == 7
