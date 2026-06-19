"""Unit tests for ProgressiveFreezeController (Phase 1 gate)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model.lora_utils import (
    iter_all_lora_params_by_layer,
    set_trainable_lora_layers,
)
from src.tg_lora.activation_cache import _get_decoder_layers
from src.tg_lora.activation_matching import ActivationMatchingLoss
from src.tg_lora.progressive_freeze import ProgressiveFreezeController


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, in_features))


class FakeTransformerModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden: int = 8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = LoRALinear(hidden, hidden)
            layer.self_attn.v_proj = LoRALinear(hidden, hidden)
            self.layers.append(layer)
        self.embed = nn.Parameter(torch.zeros(10, hidden))

    def forward(self, x):
        return x


class TestShouldFreeze:
    def test_before_start_cycle(self):
        ctrl = ProgressiveFreezeController(start_cycle=3, active_layer_indices={2, 3})
        assert not ctrl.should_freeze(0)
        assert not ctrl.should_freeze(2)

    def test_at_start_cycle(self):
        ctrl = ProgressiveFreezeController(start_cycle=3, active_layer_indices={2, 3})
        assert ctrl.should_freeze(3)

    def test_already_frozen(self):
        model = FakeTransformerModel(4)
        set_trainable_lora_layers(model, {2, 3})
        ctrl = ProgressiveFreezeController(start_cycle=1, active_layer_indices={2, 3})
        ctrl.apply_freeze(model)
        assert not ctrl.should_freeze(5)


class TestLayerResolution:
    def test_last_active_with_indices(self):
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            freeze_layer="last_active",
            active_layer_indices={2, 3},
        )
        model = FakeTransformerModel(4)
        assert ctrl._resolve_target_layer(model) == 3

    def test_explicit_integer(self):
        ctrl = ProgressiveFreezeController(start_cycle=1, freeze_layer=2)
        model = FakeTransformerModel(4)
        assert ctrl._resolve_target_layer(model) == 2

    def test_fallback_to_model_layers(self):
        ctrl = ProgressiveFreezeController(start_cycle=1)
        model = FakeTransformerModel(4)
        assert ctrl._resolve_target_layer(model) == 3


class TestApplyFreeze:
    def test_sets_requires_grad_false(self):
        model = FakeTransformerModel(4)
        set_trainable_lora_layers(model, {2, 3})

        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            freeze_layer="last_active",
            active_layer_indices={2, 3},
        )
        result = ctrl.apply_freeze(model)

        assert result.frozen_layer_idx == 3
        assert ctrl.frozen_layer_idx == 3
        assert ctrl.is_frozen

        layer_map = iter_all_lora_params_by_layer(model)
        for _, param in layer_map[3]:
            assert not param.requires_grad

        for _, param in layer_map[2]:
            assert param.requires_grad

    def test_double_freeze_raises(self):
        model = FakeTransformerModel(4)
        set_trainable_lora_layers(model, {3})
        ctrl = ProgressiveFreezeController(start_cycle=1, active_layer_indices={3})
        ctrl.apply_freeze(model)
        with pytest.raises(RuntimeError, match="already frozen"):
            ctrl.apply_freeze(model)

    def test_counts_frozen_params(self):
        model = FakeTransformerModel(4)
        set_trainable_lora_layers(model, {2, 3})
        ctrl = ProgressiveFreezeController(start_cycle=1, active_layer_indices={2, 3})
        result = ctrl.apply_freeze(model)

        layer_map = iter_all_lora_params_by_layer(model)
        assert result.num_frozen_params == len(layer_map[3])


class TestConfigSchema:
    def test_defaults_parse(self):
        from src.training.config_schema import TGLoRAParams

        params = TGLoRAParams(
            K_initial=5,
            K_candidates=[5],
            N_initial=3,
            N_candidates=[3],
            alpha_initial=0.5,
            alpha_min=0.1,
            alpha_max=1.0,
            beta_initial=0.9,
            beta_candidates=[0.9],
            relative_update_cap=0.5,
            active_layer_strategy="last_25_percent",
            progressive_freeze_enabled=True,
            progressive_freeze_start_cycle=5,
        )
        assert params.progressive_freeze_enabled is True
        assert params.progressive_freeze_start_cycle == 5
        assert params.progressive_freeze_layer == "last_active"

    def test_defaults_false(self):
        from src.training.config_schema import TGLoRAParams

        params = TGLoRAParams(
            K_initial=5,
            K_candidates=[5],
            N_initial=3,
            N_candidates=[3],
            alpha_initial=0.5,
            alpha_min=0.1,
            alpha_max=1.0,
            beta_initial=0.9,
            beta_candidates=[0.9],
            relative_update_cap=0.5,
            active_layer_strategy="last_25_percent",
        )
        assert params.progressive_freeze_enabled is False
        assert params.progressive_freeze_start_cycle == 3


# --- cache_xin (xin capture) coverage ---
# The xin cache is the target source for the entire Phase 1/2 mechanism
# (GOAL §1.6.1, docs/design/10_progressive_freezing.md §3.1), so its capture
# contract is locked down here per GOAL §7: verify the mechanism before
# trusting any downstream freeze experiment.


class _ForwardingDecoderLayer(nn.Module):
    """Decoder layer whose forward takes hidden_states as its first positional
    argument, so ProgressiveFreezeController's forward pre-hook captures the
    layer input (xin). LoRA params exist purely for layer-index mapping."""

    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = LoRALinear(hidden, hidden)
        self.self_attn.v_proj = LoRALinear(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        return (self.proj(hidden_states),)


class _ForwardingModel(nn.Module):
    """Model that actually runs its decoder layers in forward, so the target
    layer's pre-hook fires during cache_xin."""

    def __init__(self, num_layers: int = 4, hidden: int = 8, vocab: int = 16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_ForwardingDecoderLayer(hidden) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        del labels
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        out = SimpleNamespace()
        out.loss = None
        out.logits = self.lm_head(h)
        return out


def _make_xin_loader(batch: int = 2, seq: int = 5, vocab: int = 16):
    one = {
        "input_ids": torch.randint(0, vocab, (batch, seq)),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.randint(0, vocab, (batch, seq)),
    }
    return [one, {**one}]  # 2 batches; cache_xin only consumes the first


class TestCacheXin:
    def _make(self, num_layers: int = 4, active: set[int] = {2, 3}):
        model = _ForwardingModel(num_layers=num_layers)
        set_trainable_lora_layers(model, active)
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            freeze_layer="last_active",
            active_layer_indices=active,
        )
        return model, ctrl

    def test_captures_target_layer_input_shape(self):
        model, ctrl = self._make()
        xin_cache, xin_shape = ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        assert xin_shape == (2, 5, 8)
        assert set(xin_cache) == {0}
        assert len(xin_cache[0]) == 1
        assert xin_cache[0][0].shape == (2, 5, 8)

    def test_captures_correct_layer_input(self):
        """Captured xin must equal the actual input to the target layer (3),
        not some other layer's activation — confirms the hook fires on the
        right module and returns the right tensor."""
        model, ctrl = self._make()
        loader = _make_xin_loader()
        ctrl.cache_xin(model, loader, "cpu")
        captured = ctrl._xin_cache[0][0]

        with torch.no_grad():
            h = model.embed_tokens(loader[0]["input_ids"])
            reference = None
            for i, layer in enumerate(model.layers):
                if i == 3:
                    reference = h.clone()
                h = layer(h)[0]
        assert reference is not None
        assert torch.equal(captured, reference)

    def test_captures_no_grad_tensor(self):
        model, ctrl = self._make()
        ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        assert ctrl._xin_cache[0][0].requires_grad is False

    def test_apply_freeze_reports_xin_shape(self):
        model, ctrl = self._make()
        ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        result = ctrl.apply_freeze(model)
        assert result.xin_shape == (2, 5, 8)
        assert ctrl.is_frozen
        assert result.frozen_layer_idx == 3

    def test_training_mode_restored_when_training(self):
        model, ctrl = self._make()
        model.train()
        ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        assert model.training is True

    def test_eval_mode_preserved(self):
        model, ctrl = self._make()
        model.eval()
        ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        assert model.training is False

    def test_forward_pre_hook_removed_after_capture(self):
        model, ctrl = self._make()
        target = _get_decoder_layers(model)[3]
        assert len(target._forward_pre_hooks) == 0
        ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        assert len(target._forward_pre_hooks) == 0

    def test_empty_dataloader_returns_empty(self):
        model, ctrl = self._make()
        xin_cache, xin_shape = ctrl.cache_xin(model, [], "cpu")
        assert xin_cache == {}
        assert xin_shape is None

    def test_clear_xin_cache(self):
        model, ctrl = self._make()
        ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        assert ctrl._xin_cache
        ctrl.clear_xin_cache()
        assert ctrl._xin_cache == {}


# --- compute_local_loss (Level 2 front-layer training signal) coverage ---
# This is the glue that consumes the cached xin: run the front layer forward,
# pair its current output against the cached xin, and return the
# activation-matching loss (GOAL §1.6.1, docs/design/10_progressive_freezing.md
# §8 items 3-4). Verified in isolation before any freeze experiment trusts it.


class TestComputeLocalLoss:
    def _make_frozen(self, num_layers: int = 4, active: set[int] = {2, 3}):
        """Model with layer 3 frozen and its xin cached for batch 0."""
        model = _ForwardingModel(num_layers=num_layers)
        set_trainable_lora_layers(model, active)
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            freeze_layer="last_active",
            active_layer_indices=active,
        )
        loader = _make_xin_loader()
        ctrl.cache_xin(model, loader, "cpu")
        ctrl.apply_freeze(model)
        return model, ctrl, loader

    def test_pairs_current_front_output_vs_cached_xin(self):
        # Immediately after freezing (no weight change) the front layer still
        # emits exactly the cached xin, so the local loss is zero. This proves
        # the glue captures the same quantity cache_xin stored and pairs it
        # correctly (design §3.3: the signal only appears once the front moves).
        model, ctrl, loader = self._make_frozen()
        loss = ctrl.compute_local_loss(model, loader[0], ActivationMatchingLoss())
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_local_loss_drives_front_toward_xin(self):
        # design §3.3: the gap is a genuine learning signal. Gradient descent
        # on the local loss must shrink the front-vs-xin distance — the whole
        # reason Progressive Freezing keeps improving after the suffix is cut.
        model, ctrl, loader = self._make_frozen()
        with torch.no_grad():
            model.layers[2].proj.weight.add_(0.25)  # push front off the xin
        w = model.layers[2].proj.weight
        first, last = None, None
        for _ in range(5):
            model.zero_grad()
            loss = ctrl.compute_local_loss(model, loader[0], ActivationMatchingLoss())
            if first is None:
                first = loss.item()
            last = loss.item()
            loss.backward()
            with torch.no_grad():
                w -= 0.1 * w.grad
        assert last < first

    def test_gradient_flows_into_front_not_frozen(self):
        model, ctrl, loader = self._make_frozen()
        with torch.no_grad():
            model.layers[2].proj.weight.add_(0.25)
        loss = ctrl.compute_local_loss(model, loader[0], ActivationMatchingLoss())
        loss.backward()
        front = model.layers[2].proj.weight
        assert front.grad is not None
        assert front.grad.abs().sum().item() > 0
        # The frozen layer's LoRA params (requires_grad=False) receive nothing.
        frozen = model.layers[3].self_attn.q_proj.lora_A
        assert frozen.grad is None

    def test_attention_mask_passed_through(self):
        # Padding mask must actually reach the loss: a partial mask differs from
        # the all-ones mask once the front is off the cached xin.
        model, ctrl, loader = self._make_frozen()
        with torch.no_grad():
            model.layers[2].proj.weight.add_(0.3)
        full = ctrl.compute_local_loss(model, loader[0], ActivationMatchingLoss())
        partial_batch = {
            **loader[0],
            "attention_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]),
        }
        partial = ctrl.compute_local_loss(
            model, partial_batch, ActivationMatchingLoss()
        )
        assert not torch.allclose(full, partial)

    def test_hook_removed_after_call(self):
        model, ctrl, loader = self._make_frozen()
        target = _get_decoder_layers(model)[3]
        assert len(target._forward_pre_hooks) == 0
        ctrl.compute_local_loss(model, loader[0], ActivationMatchingLoss())
        assert len(target._forward_pre_hooks) == 0

    def test_requires_freeze_first(self):
        model = _ForwardingModel()
        set_trainable_lora_layers(model, {2, 3})
        ctrl = ProgressiveFreezeController(start_cycle=1, active_layer_indices={2, 3})
        ctrl.cache_xin(model, _make_xin_loader(), "cpu")
        with pytest.raises(RuntimeError, match="apply_freeze"):
            ctrl.compute_local_loss(
                model, _make_xin_loader()[0], ActivationMatchingLoss()
            )

    def test_requires_cached_xin(self):
        model = _ForwardingModel()
        set_trainable_lora_layers(model, {2, 3})
        ctrl = ProgressiveFreezeController(start_cycle=1, active_layer_indices={2, 3})
        ctrl.apply_freeze(model)
        with pytest.raises(RuntimeError, match="cache_xin"):
            ctrl.compute_local_loss(
                model, _make_xin_loader()[0], ActivationMatchingLoss()
            )

    def test_unknown_batch_idx_raises(self):
        model, ctrl, loader = self._make_frozen()
        with pytest.raises(RuntimeError, match="batch 7"):
            ctrl.compute_local_loss(
                model, loader[0], ActivationMatchingLoss(), batch_idx=7
            )
