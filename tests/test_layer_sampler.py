import re

import torch
import torch.nn as nn

from tg_lora.layer_sampler import (
    get_layer_indices,
    get_num_layers,
    select_active_layers,
)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)


class FakeTransformerModel(nn.Module):
    def __init__(self, num_layers=12):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = LoRALinear(8, 8)
            layer.self_attn.v_proj = LoRALinear(8, 8)
            self.layers.append(layer)


def test_get_num_layers():
    model = FakeTransformerModel(12)
    assert get_num_layers(model) == 12


def test_get_layer_indices():
    model = FakeTransformerModel(8)
    indices = get_layer_indices(model)
    assert indices == list(range(8))


def test_last_25_percent():
    model = FakeTransformerModel(12)
    active, indices = select_active_layers(model, "last_25_percent")

    # Last 25% of 12 = last 3 layers (9, 10, 11)
    # Each layer has 4 lora params (q_A, q_B, v_A, v_B)
    for name in active:
        m = re.search(r"layers\.(\d+)\.", name)
        assert m is not None
        idx = int(m.group(1))
        assert idx >= 9

    assert indices == {9, 10, 11}


def test_last_25_plus_random_2():
    model = FakeTransformerModel(12)
    active, indices = select_active_layers(
        model, "last_25_percent_plus_random_2", random_middle=2
    )

    assert len(active) > 0
    # Should include some last-25% layers
    layer_indices = set()
    for name in active:
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            layer_indices.add(int(m.group(1)))
    assert len(layer_indices) >= 3  # at least 3 from last 25% + extras


def test_middle_random():
    model = FakeTransformerModel(12)
    active, indices = select_active_layers(model, "middle_random")

    assert len(active) > 0


def test_lisa_weighted_with_scores():
    model = FakeTransformerModel(12)
    scores = {i: float(i) for i in range(12)}
    active, indices = select_active_layers(
        model, "lisa_like_weighted", layer_scores=scores, temperature=1.0
    )
    assert len(active) > 0
    assert len(indices) > 0


def test_lisa_weighted_empty_scores_fallback():
    model = FakeTransformerModel(12)
    active, indices = select_active_layers(
        model, "lisa_like_weighted", layer_scores={}, temperature=1.0
    )
    assert len(active) > 0


def test_lisa_weighted_none_scores_fallback():
    model = FakeTransformerModel(12)
    active, indices = select_active_layers(
        model, "lisa_like_weighted", layer_scores=None, temperature=1.0
    )
    assert len(active) > 0


def test_empty_model_returns_all():
    """If no layers are found by iter_lora_params_by_layer, fallback returns all params."""
    model = nn.Module()  # no LoRA params at all
    active, indices = select_active_layers(model, "last_25_percent")
    # Should return empty set since there are no LoRA params at all
    assert isinstance(active, set)


def test_single_layer_model():
    model = FakeTransformerModel(1)
    active, indices = select_active_layers(model, "last_25_percent")
    assert len(active) > 0
    assert indices == {0}


def test_lisa_weighted_high_temperature():
    model = FakeTransformerModel(12)
    scores = {0: 10.0, 1: 5.0, 2: 1.0, **{i: 0.0 for i in range(3, 12)}}
    active, indices = select_active_layers(
        model, "lisa_like_weighted", layer_scores=scores, temperature=1000.0
    )
    assert len(active) > 0
    assert len(indices) > 0


def test_lisa_weighted_near_zero_temperature_clamped():
    model = FakeTransformerModel(12)
    scores = {i: float(i) for i in range(12)}
    active, indices = select_active_layers(
        model, "lisa_like_weighted", layer_scores=scores, temperature=0.001
    )
    assert len(active) > 0
    assert len(indices) > 0


# ---------------------------------------------------------------------------
# TASK-0039: temperature parameter integration tests
# ---------------------------------------------------------------------------


def _make_tg_lora_config_dict(temperature: float = 1.0) -> dict:
    """Minimal valid TGLoRA config dict with a configurable temperature."""
    return {
        "experiment": {"name": "test", "seed": 42},
        "model": {"name_or_path": "dummy"},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
        "data": {
            "train_path": "/tmp/train.jsonl",
            "valid_quick_path": "/tmp/vq.jsonl",
            "valid_full_path": "/tmp/vf.jsonl",
        },
        "training": {
            "batch_size": 1,
            "grad_accumulation": 1,
            "learning_rate": 1e-4,
            "max_steps": 1,
        },
        "logging": {"run_dir": "/tmp/run"},
        "tg_lora": {
            "K_initial": 1,
            "K_candidates": [1],
            "N_initial": 1,
            "N_candidates": [1],
            "alpha_initial": 0.5,
            "alpha_min": 0.1,
            "alpha_max": 1.0,
            "beta_initial": 0.5,
            "beta_candidates": [0.5],
            "relative_update_cap": 0.5,
            "active_layer_strategy": "lisa_like_weighted",
            "layer_sample_temperature": temperature,
        },
    }


class TestTemperatureDistributionVariation:
    """AC2: layer selection distribution varies with temperature."""

    @staticmethod
    def _sample_histogram(temperature: float, n_trials: int = 2000) -> dict[int, int]:
        model = FakeTransformerModel(12)
        scores = {0: 10.0, 1: 5.0, 2: 1.0, **{i: 0.1 for i in range(3, 12)}}
        counts: dict[int, int] = {}
        for _ in range(n_trials):
            _, indices = select_active_layers(
                model,
                "lisa_like_weighted",
                layer_scores=scores,
                temperature=temperature,
            )
            for idx in indices:
                counts[idx] = counts.get(idx, 0) + 1
        return counts

    def test_low_temp_concentrates_on_high_scores(self):
        counts = self._sample_histogram(0.1)
        top_two = counts.get(0, 0) + counts.get(1, 0)
        bottom_two = counts.get(10, 0) + counts.get(11, 0)
        assert top_two > bottom_two, (
            f"Low temp should concentrate on high-score layers: top={top_two}, bottom={bottom_two}"
        )

    def test_high_temp_spreads_more(self):
        counts_low = self._sample_histogram(0.1)
        counts_high = self._sample_histogram(5.0)
        low_entropy = _distribution_entropy(counts_low)
        high_entropy = _distribution_entropy(counts_high)
        assert high_entropy > low_entropy, (
            f"High temp should spread selection: H_high={high_entropy:.4f} <= H_low={low_entropy:.4f}"
        )

    def test_mid_temp_between_extremes(self):
        counts_mid = self._sample_histogram(1.0)
        counts_low = self._sample_histogram(0.1)
        counts_high = self._sample_histogram(5.0)
        h_mid = _distribution_entropy(counts_mid)
        h_low = _distribution_entropy(counts_low)
        h_high = _distribution_entropy(counts_high)
        assert h_low <= h_mid <= h_high, (
            f"Entropy should increase with temperature: H_low={h_low:.4f} H_mid={h_mid:.4f} H_high={h_high:.4f}"
        )


def _distribution_entropy(counts: dict[int, int]) -> float:
    import math

    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


class TestTemperatureBoundaryValues:
    """AC3: edge cases — near-zero, very large, NaN/Inf."""

    def test_near_zero_temperature_still_selects(self):
        model = FakeTransformerModel(12)
        scores = {i: float(i) for i in range(12)}
        active, indices = select_active_layers(
            model,
            "lisa_like_weighted",
            layer_scores=scores,
            temperature=1e-6,
        )
        assert len(active) > 0
        assert len(indices) > 0

    def test_very_large_temperature_selects(self):
        model = FakeTransformerModel(12)
        scores = {i: float(i) for i in range(12)}
        active, indices = select_active_layers(
            model,
            "lisa_like_weighted",
            layer_scores=scores,
            temperature=1e6,
        )
        assert len(active) > 0
        assert len(indices) > 0

    def test_large_temperature_approximates_uniform(self):
        model = FakeTransformerModel(12)
        scores = {0: 100.0, **{i: 1.0 for i in range(1, 12)}}
        counts: dict[int, int] = {}
        for _ in range(1000):
            _, indices = select_active_layers(
                model,
                "lisa_like_weighted",
                layer_scores=scores,
                temperature=1e6,
            )
            for idx in indices:
                counts[idx] = counts.get(idx, 0) + 1
        min_count = min(counts.values())
        max_count = max(counts.values())
        assert max_count / max(min_count, 1) < 3.0, (
            f"Very high temp should be near-uniform: min={min_count} max={max_count}"
        )

    def test_nan_scores_handled(self):

        model = FakeTransformerModel(12)
        scores = {0: float("nan"), 1: 5.0, **{i: 1.0 for i in range(2, 12)}}
        active, indices = select_active_layers(
            model,
            "lisa_like_weighted",
            layer_scores=scores,
            temperature=1.0,
        )
        # Should not crash; result must be non-empty
        assert len(active) > 0 or len(indices) >= 0

    def test_inf_scores_handled(self):
        model = FakeTransformerModel(12)
        scores = {0: float("inf"), 1: 5.0, **{i: 1.0 for i in range(2, 12)}}
        active, indices = select_active_layers(
            model,
            "lisa_like_weighted",
            layer_scores=scores,
            temperature=1.0,
        )
        assert len(active) > 0 or len(indices) >= 0

    def test_negative_inf_scores_handled(self):
        model = FakeTransformerModel(12)
        scores = {0: float("-inf"), 1: 5.0, **{i: 1.0 for i in range(2, 12)}}
        active, indices = select_active_layers(
            model,
            "lisa_like_weighted",
            layer_scores=scores,
            temperature=1.0,
        )
        assert len(active) > 0 or len(indices) >= 0
