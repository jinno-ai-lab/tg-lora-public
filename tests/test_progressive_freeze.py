"""Unit tests for ProgressiveFreezeController (Phase 1 gate)."""

import torch
import torch.nn as nn
import pytest

from src.model.lora_utils import (
    iter_all_lora_params,
    iter_all_lora_params_by_layer,
    set_trainable_lora_layers,
)
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
            start_cycle=1, freeze_layer="last_active", active_layer_indices={2, 3},
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
            start_cycle=1, freeze_layer="last_active", active_layer_indices={2, 3},
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
