import torch
import torch.nn as nn

from src.model.lora_utils import (
    count_lora_params,
    iter_lora_params,
    iter_lora_params_by_layer,
)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)


class FakeTransformerModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = LoRALinear(8, 8)
            layer.self_attn.v_proj = LoRALinear(8, 8)
            self.layers.append(layer)
        # non-LoRA param that should be excluded
        self.embed = nn.Parameter(torch.zeros(10, 8))


def test_iter_lora_params():
    model = FakeTransformerModel(4)
    params = list(iter_lora_params(model))

    names = [n for n, _ in params]
    # 4 layers x 2 modules (q_proj, v_proj) x 2 params (lora_A, lora_B) = 16
    assert len(names) == 16

    for name, p in params:
        assert "lora_A" in name or "lora_B" in name
        assert p.requires_grad

    # embed should not appear
    assert not any("embed" in n for n in names)


def test_iter_lora_params_excludes_frozen_lora():
    model = FakeTransformerModel(2)
    # Freeze one lora_A param — it should be excluded
    for name, p in model.named_parameters():
        if "layers.0.self_attn.q_proj.lora_A" == name:
            p.requires_grad_(False)
            break

    params = list(iter_lora_params(model))
    names = [n for n, _ in params]
    # 2 layers x 2 modules x 2 params = 8 minus 1 frozen = 7
    assert len(names) == 7
    assert "layers.0.self_attn.q_proj.lora_A" not in names


def test_iter_lora_params_by_layer():
    model = FakeTransformerModel(4)
    layer_map = iter_lora_params_by_layer(model)

    assert set(layer_map.keys()) == {0, 1, 2, 3}

    for idx, params in layer_map.items():
        # Each layer has 2 modules x 2 params = 4
        assert len(params) == 4
        for name, p in params:
            assert f"layers.{idx}." in name
            assert "lora_A" in name or "lora_B" in name


def test_iter_lora_params_by_layer_excludes_non_layer_params():
    # Add a top-level LoRA param (no layers.N. prefix) — should not appear in layer_map
    model = FakeTransformerModel(2)
    model.lm_head_lora_A = nn.Parameter(torch.randn(4, 4))
    model.lm_head_lora_A.requires_grad_(True)

    layer_map = iter_lora_params_by_layer(model)
    assert set(layer_map.keys()) == {0, 1}
    # lm_head_lora_A is yielded by iter_lora_params but not grouped into any layer
    all_lora = list(iter_lora_params(model))
    assert any("lm_head_lora_A" in n for n, _ in all_lora)


def test_count_lora_params():
    model = FakeTransformerModel(4)
    # 4 layers x 2 modules x 2 params (lora_A, lora_B), each param is 8x8=64 elements
    # total = 16 params x 64 elements = 1024
    expected = 4 * 2 * 2 * 8 * 8
    assert count_lora_params(model) == expected


def test_count_lora_params_excludes_non_lora():
    model = FakeTransformerModel(2)
    # embed is 10x8=80 but should not be counted
    lora_count = count_lora_params(model)
    # 2 layers x 2 modules x 2 params x 64 = 512
    assert lora_count == 2 * 2 * 2 * 8 * 8
    # Verify embed (10*8=80) is excluded
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_trainable > lora_count
