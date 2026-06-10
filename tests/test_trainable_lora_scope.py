"""Integration tests for trainable_lora_scope: index calculation, freeze/thaw, gradient flow."""

import torch
import torch.nn as nn
import pytest

from src.model.lora_utils import (
    configure_trainable_lora_scope,
    get_last_fraction_lora_layer_indices,
    iter_all_lora_params,
    iter_all_lora_params_by_layer,
    set_trainable_lora_layers,
)


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, in_features))


class FakeTransformerModel(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = LoRALinear(8, 8)
            layer.self_attn.v_proj = LoRALinear(8, 8)
            self.layers.append(layer)
        self.embed = nn.Parameter(torch.zeros(10, 8))

    def forward(self, x):
        for layer in self.layers:
            x = x + sum(
                p.view_as(x)
                for _, p in iter_all_lora_params(self)
                if p.requires_grad and p.numel() == x.numel()
            )
        return x


# --- get_last_fraction_lora_layer_indices tests ---


class TestGetLastFractionLoraLayerIndices:
    def test_4layer_fraction_25(self):
        model = FakeTransformerModel(4)
        result = get_last_fraction_lora_layer_indices(model, fraction=0.25)
        assert result == {3}

    def test_8layer_fraction_25(self):
        model = FakeTransformerModel(8)
        result = get_last_fraction_lora_layer_indices(model, fraction=0.25)
        assert result == {6, 7}

    def test_4layer_fraction_50(self):
        model = FakeTransformerModel(4)
        result = get_last_fraction_lora_layer_indices(model, fraction=0.5)
        assert result == {2, 3}

    def test_fraction_100(self):
        model = FakeTransformerModel(4)
        result = get_last_fraction_lora_layer_indices(model, fraction=1.0)
        assert result == {0, 1, 2, 3}

    def test_fraction_near_zero_returns_at_least_one(self):
        model = FakeTransformerModel(4)
        result = get_last_fraction_lora_layer_indices(model, fraction=0.01)
        assert len(result) >= 1
        assert result == {3}

    def test_invalid_fraction_zero(self):
        model = FakeTransformerModel(4)
        with pytest.raises(ValueError, match="fraction must be in"):
            get_last_fraction_lora_layer_indices(model, fraction=0.0)

    def test_invalid_fraction_negative(self):
        model = FakeTransformerModel(4)
        with pytest.raises(ValueError, match="fraction must be in"):
            get_last_fraction_lora_layer_indices(model, fraction=-0.5)

    def test_invalid_fraction_over_one(self):
        model = FakeTransformerModel(4)
        with pytest.raises(ValueError, match="fraction must be in"):
            get_last_fraction_lora_layer_indices(model, fraction=1.5)

    def test_unmapped_lora_params_raise_valueerror(self):
        model = FakeTransformerModel(4)
        model.lm_head_lora_A = nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError, match="not mapped to decoder layers"):
            get_last_fraction_lora_layer_indices(model, fraction=0.5)


# --- set_trainable_lora_layers tests ---


class TestSetTrainableLoraLayers:
    def test_selected_layers_trainable(self):
        model = FakeTransformerModel(4)
        set_trainable_lora_layers(model, {2, 3})
        layer_map = iter_all_lora_params_by_layer(model)
        for idx, params in layer_map.items():
            for _, param in params:
                if idx in {2, 3}:
                    assert param.requires_grad is True

    def test_unselected_layers_frozen(self):
        model = FakeTransformerModel(4)
        set_trainable_lora_layers(model, {2, 3})
        layer_map = iter_all_lora_params_by_layer(model)
        for idx, params in layer_map.items():
            for _, param in params:
                if idx not in {2, 3}:
                    assert param.requires_grad is False

    def test_returns_active_names(self):
        model = FakeTransformerModel(4)
        active_names = set_trainable_lora_layers(model, {2, 3})
        assert len(active_names) > 0
        for name in active_names:
            assert "layers.2." in name or "layers.3." in name
            assert "lora_A" in name or "lora_B" in name

    def test_all_layers_trainable(self):
        model = FakeTransformerModel(4)
        active_names = set_trainable_lora_layers(model, {0, 1, 2, 3})
        layer_map = iter_all_lora_params_by_layer(model)
        total_params = sum(len(ps) for ps in layer_map.values())
        assert len(active_names) == total_params

    def test_empty_set_freezes_all(self):
        model = FakeTransformerModel(4)
        active_names = set_trainable_lora_layers(model, set())
        assert len(active_names) == 0
        for _, p in iter_all_lora_params(model):
            assert p.requires_grad is False


# --- gradient flow tests ---


class TestGradientFlow:
    def _make_model_with_forward(self, num_layers: int = 4):
        """Build a model whose forward path depends on LoRA params."""

        class LoRALinearFwd(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(dim, dim) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(dim, dim))

            def forward(self, x):
                return x + x @ self.lora_A.T @ self.lora_B.T

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                for _ in range(num_layers):
                    layer = nn.Module()
                    layer.self_attn = nn.Module()
                    layer.self_attn.q_proj = LoRALinearFwd(8)
                    self.layers.append(layer)

            def forward(self, x):
                for layer in self.layers:
                    x = layer.self_attn.q_proj(x)
                return x.sum()

        return Model()

    def test_backward_updates_only_trainable(self):
        model = self._make_model_with_forward(4)
        trainable_indices = {2, 3}
        set_trainable_lora_layers(model, trainable_indices)

        x = torch.randn(2, 8)
        loss = model(x)
        loss.backward()

        layer_map = iter_all_lora_params_by_layer(model)
        for idx, params in layer_map.items():
            for name, param in params:
                if idx in trainable_indices:
                    assert param.grad is not None, f"Trainable layer {idx} param {name} has no grad"
                else:
                    assert param.grad is None, f"Frozen layer {idx} param {name} has grad"

    def test_backward_single_layer(self):
        model = self._make_model_with_forward(4)
        set_trainable_lora_layers(model, {3})

        x = torch.randn(2, 8)
        loss = model(x)
        loss.backward()

        layer_map = iter_all_lora_params_by_layer(model)
        for idx, params in layer_map.items():
            for name, param in params:
                if idx == 3:
                    assert param.grad is not None
                else:
                    assert param.grad is None, f"Layer {idx} {name} should have no grad"


# --- configure_trainable_lora_scope integration ---


class TestConfigureTrainableLoraScope:
    def test_scope_all(self):
        model = FakeTransformerModel(4)
        active_names, active_indices = configure_trainable_lora_scope(model, "all")
        assert active_indices == {0, 1, 2, 3}
        assert len(active_names) > 0
        for _, p in iter_all_lora_params(model):
            assert p.requires_grad is True

    def test_scope_last_25_percent(self):
        model = FakeTransformerModel(8)
        active_names, active_indices = configure_trainable_lora_scope(model, "last_25_percent")
        assert active_indices == {6, 7}
        assert len(active_names) > 0

    def test_scope_unsupported(self):
        model = FakeTransformerModel(4)
        with pytest.raises(ValueError, match="Unsupported trainable_lora_scope"):
            configure_trainable_lora_scope(model, "invalid_scope")

    def test_scope_last_25_percent_4layer(self):
        model = FakeTransformerModel(4)
        _, active_indices = configure_trainable_lora_scope(model, "last_25_percent")
        assert active_indices == {3}

    def test_scope_last_25_percent_2layer(self):
        model = FakeTransformerModel(2)
        _, active_indices = configure_trainable_lora_scope(model, "last_25_percent")
        # ceil(2 * 0.25) = 1, so just last layer
        assert active_indices == {1}
