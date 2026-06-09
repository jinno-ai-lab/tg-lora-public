import pytest
import torch
import torch.nn as nn

from tg_lora.lora_utils import (
    configure_trainable_lora_scope,
    count_lora_params,
    get_last_fraction_lora_layer_indices,
    get_unmapped_lora_param_names,
    iter_all_lora_params,
    iter_all_lora_params_by_layer,
    iter_lora_params,
    iter_lora_params_by_layer,
    set_all_lora_trainable,
    set_trainable_lora_layers,
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


# ---------------------------------------------------------------------------
# iter_all_lora_params (lines 19-24)
# ---------------------------------------------------------------------------


def test_iter_all_lora_params_includes_frozen():
    model = FakeTransformerModel(2)
    # Freeze all LoRA params
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(False)
    params = list(iter_all_lora_params(model))
    # Should still find all LoRA params even though frozen
    # 2 layers x 2 modules x 2 params = 8
    assert len(params) == 8


def test_iter_all_lora_params_excludes_embed():
    model = FakeTransformerModel(2)
    params = list(iter_all_lora_params(model))
    names = [n for n, _ in params]
    assert not any("embed" in n for n in names)


# ---------------------------------------------------------------------------
# iter_all_lora_params_by_layer (lines 39-48)
# ---------------------------------------------------------------------------


def test_iter_all_lora_params_by_layer():
    model = FakeTransformerModel(4)
    layer_map = iter_all_lora_params_by_layer(model)
    assert set(layer_map.keys()) == {0, 1, 2, 3}
    for idx, params in layer_map.items():
        assert len(params) == 4  # 2 modules x 2 params


def test_iter_all_lora_params_by_layer_includes_frozen():
    model = FakeTransformerModel(2)
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(False)
    layer_map = iter_all_lora_params_by_layer(model)
    assert set(layer_map.keys()) == {0, 1}
    # Should still include all params even if frozen
    total_params = sum(len(v) for v in layer_map.values())
    assert total_params == 8


# ---------------------------------------------------------------------------
# get_unmapped_lora_param_names (lines 51-59)
# ---------------------------------------------------------------------------


def test_get_unmapped_lora_param_names_no_unmapped():
    model = FakeTransformerModel(4)
    unmapped = get_unmapped_lora_param_names(model)
    assert unmapped == []


def test_get_unmapped_lora_param_names_with_top_level():
    model = FakeTransformerModel(2)
    # Add a top-level LoRA param that doesn't match layers.N. pattern
    model.extra_lora_A = nn.Parameter(torch.randn(4, 4))
    model.extra_lora_A.requires_grad_(True)
    unmapped = get_unmapped_lora_param_names(model)
    assert len(unmapped) >= 1
    assert any("extra_lora_A" in n for n in unmapped)


# ---------------------------------------------------------------------------
# set_all_lora_trainable (lines 62-67)
# ---------------------------------------------------------------------------


def test_set_all_lora_trainable():
    model = FakeTransformerModel(3)
    # Freeze everything first
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    active = set_all_lora_trainable(model)
    assert len(active) > 0
    for name, p in iter_all_lora_params(model):
        assert p.requires_grad, f"{name} should be trainable after set_all_lora_trainable"


# ---------------------------------------------------------------------------
# get_last_fraction_lora_layer_indices (lines 70-90)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_get_last_fraction_rejects_invalid_fraction(fraction):
    model = FakeTransformerModel(4)
    with pytest.raises(ValueError, match="fraction must be in"):
        get_last_fraction_lora_layer_indices(model, fraction=fraction)


def test_get_last_fraction_raises_with_unmapped():
    model = FakeTransformerModel(2)
    model.extra_lora_A = nn.Parameter(torch.randn(4, 4))
    model.extra_lora_A.requires_grad_(True)
    with pytest.raises(ValueError, match="not mapped to decoder layers"):
        get_last_fraction_lora_layer_indices(model)


def test_get_last_fraction_raises_on_empty():
    model = nn.Module()
    with pytest.raises(ValueError, match="No LoRA decoder layers found"):
        get_last_fraction_lora_layer_indices(model)


def test_get_last_fraction_one_layer():
    model = FakeTransformerModel(1)
    indices = get_last_fraction_lora_layer_indices(model)
    assert indices == {0}


def test_get_last_fraction_half():
    model = FakeTransformerModel(10)
    indices = get_last_fraction_lora_layer_indices(model, fraction=0.5)
    # ceil(10 * 0.5) = 5 → last 5 layers
    assert indices == {5, 6, 7, 8, 9}


def test_get_last_fraction_full():
    model = FakeTransformerModel(4)
    indices = get_last_fraction_lora_layer_indices(model, fraction=1.0)
    assert indices == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# set_trainable_lora_layers (lines 93-106)
# ---------------------------------------------------------------------------


def test_set_trainable_lora_layers_selective():
    model = FakeTransformerModel(4)
    active = set_trainable_lora_layers(model, {1, 3})
    for name, p in iter_all_lora_params(model):
        if "layers.1." in name or "layers.3." in name:
            assert p.requires_grad, f"{name} should be trainable"
            assert name in active
        else:
            assert not p.requires_grad, f"{name} should NOT be trainable"


def test_set_trainable_lora_layers_empty_set():
    model = FakeTransformerModel(4)
    active = set_trainable_lora_layers(model, set())
    assert active == set()
    # All LoRA params should be frozen
    for name, p in iter_all_lora_params(model):
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# configure_trainable_lora_scope (lines 109-123)
# ---------------------------------------------------------------------------


def test_configure_trainable_lora_scope_all():
    model = FakeTransformerModel(4)
    names, indices = configure_trainable_lora_scope(model, "all")
    assert len(names) > 0
    assert indices == {0, 1, 2, 3}


def test_configure_trainable_lora_scope_last_25_percent():
    model = FakeTransformerModel(12)
    names, indices = configure_trainable_lora_scope(model, "last_25_percent")
    assert indices == {9, 10, 11}
    assert len(names) > 0


def test_configure_trainable_lora_scope_rejects_unknown():
    model = FakeTransformerModel(4)
    with pytest.raises(ValueError, match="Unsupported trainable_lora_scope"):
        configure_trainable_lora_scope(model, "unknown_scope")
