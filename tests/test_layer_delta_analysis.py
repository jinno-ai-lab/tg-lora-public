"""Tests for per-tensor ΔW layer-type analysis (GOAL §4 step 2)."""

import math

import pytest
import torch

from src.tg_lora.layer_delta_analysis import (
    analyze_tensor_deltas,
    compute_direction_stability,
    compute_rank1_dominance,
    group_by_layer_type,
    marchenko_pastur_expected_rank1,
)
from src.tg_lora.layer_type import LayerType, classify_layer_type


# ---------------------------------------------------------------------------
# Layer type classification
# ---------------------------------------------------------------------------

class TestClassifyLayerType:
    def test_out_proj(self):
        assert classify_layer_type("model.layers.5.self_attn.out_proj.lora_A.default.weight") == LayerType.ATTENTION_OUT

    def test_v_proj(self):
        assert classify_layer_type("model.layers.3.self_attn.v_proj.lora_B.default.weight") == LayerType.ATTENTION_V

    def test_attention_other(self):
        assert classify_layer_type("model.layers.7.self_attn.q_proj.lora_A.default.weight") == LayerType.ATTENTION_OTHER
        assert classify_layer_type("model.layers.7.self_attn.k_proj.lora_B.default.weight") == LayerType.ATTENTION_OTHER

    def test_deltanet(self):
        assert classify_layer_type("model.layers.10.dt_proj.lora_A.default.weight") == LayerType.DELTANET
        assert classify_layer_type("model.layers.10.dt_out.lora_B.default.weight") == LayerType.DELTANET

    def test_deltanet_mamba_ssm(self):
        assert classify_layer_type("model.layers.0.mamba.dt_proj.lora_A.weight") == LayerType.DELTANET
        assert classify_layer_type("model.layers.0.ssm.A_log.lora_A.weight") == LayerType.DELTANET

    def test_mlp(self):
        assert classify_layer_type("model.layers.12.mlp.gate_proj.lora_A.default.weight") == LayerType.MLP
        assert classify_layer_type("model.layers.12.mlp.up_proj.lora_B.default.weight") == LayerType.MLP

    def test_unknown(self):
        assert classify_layer_type("model.embed_tokens.lora_A.default.weight") == LayerType.UNKNOWN


# ---------------------------------------------------------------------------
# Rank-1 dominance
# ---------------------------------------------------------------------------

class TestRank1Dominance:
    def test_perfect_rank1(self):
        """A matrix where all rows are multiples of one direction should have rank-1 ~1."""
        torch.manual_seed(42)
        d = 32
        v = torch.randn(d)
        v = v / v.norm()
        # All rows are scaled versions of v
        mat = torch.randn(10, 1) * v.unsqueeze(0)  # [10, d]
        # Actually: outer product
        mat = (torch.randn(10) * 2.0).unsqueeze(1) * v.unsqueeze(0)  # [10, d]
        r1 = compute_rank1_dominance(mat)
        assert r1 > 0.98, f"Expected rank-1 dominance > 0.98 for pure rank-1 matrix, got {r1}"

    def test_random_low_dominance(self):
        """Random matrix should have low rank-1 dominance."""
        torch.manual_seed(0)
        mat = torch.randn(20, 100)
        r1 = compute_rank1_dominance(mat)
        # For 20x100 random, rank-1 dominance should be relatively low
        assert r1 < 0.3, f"Random matrix rank-1 dominance too high: {r1}"

    def test_single_row_returns_zero(self):
        mat = torch.randn(1, 16)
        assert compute_rank1_dominance(mat) == 0.0

    def test_zero_matrix(self):
        mat = torch.zeros(5, 16)
        assert compute_rank1_dominance(mat) == 0.0


# ---------------------------------------------------------------------------
# Direction stability
# ---------------------------------------------------------------------------

class TestDirectionStability:
    def test_stable_direction(self):
        """If first and second halves share the same PC1, stability should be high."""
        torch.manual_seed(42)
        d = 32
        v = torch.randn(d)
        v = v / v.norm()
        # First half: rows along v + noise
        h1 = (torch.randn(5) * 2.0).unsqueeze(1) * v.unsqueeze(0) + torch.randn(5, d) * 0.01
        h2 = (torch.randn(5) * 2.0).unsqueeze(1) * v.unsqueeze(0) + torch.randn(5, d) * 0.01
        mat = torch.cat([h1, h2], dim=0)
        stab = compute_direction_stability(mat)
        assert stab is not None
        assert stab > 0.9, f"Expected high stability for same direction, got {stab}"

    def test_opposite_directions(self):
        """If halves have opposite PC1s, stability should be low (after abs)."""
        torch.manual_seed(42)
        d = 32
        v1 = torch.randn(d)
        v1 = v1 / v1.norm()
        v2 = torch.randn(d)
        v2 = v2 / v2.norm()
        # Make them somewhat different
        v2 = v2 - 0.5 * torch.dot(v2, v1) * v1
        v2 = v2 / v2.norm()
        h1 = (torch.randn(5) * 2.0).unsqueeze(1) * v1.unsqueeze(0) + torch.randn(5, d) * 0.05
        h2 = (torch.randn(5) * 2.0).unsqueeze(1) * v2.unsqueeze(0) + torch.randn(5, d) * 0.05
        mat = torch.cat([h1, h2], dim=0)
        stab = compute_direction_stability(mat)
        assert stab is not None
        assert stab < 0.7, f"Expected low stability for different directions, got {stab}"

    def test_insufficient_data(self):
        mat = torch.randn(3, 16)
        assert compute_direction_stability(mat) is None


# ---------------------------------------------------------------------------
# Marchenko-Pastur null
# ---------------------------------------------------------------------------

class TestMarchenkoPastur:
    def test_returns_positive(self):
        val = marchenko_pastur_expected_rank1(10, 100)
        assert val > 0

    def test_larger_rows_lower_dominance(self):
        """More snapshots should give lower expected rank-1 dominance."""
        v1 = marchenko_pastur_expected_rank1(5, 100)
        v2 = marchenko_pastur_expected_rank1(20, 100)
        assert v2 < v1

    def test_too_small_returns_zero(self):
        assert marchenko_pastur_expected_rank1(1, 100) == 0.0


# ---------------------------------------------------------------------------
# analyze_tensor_deltas
# ---------------------------------------------------------------------------

class TestAnalyzeTensorDeltas:
    def _make_dominant_deltas(
        self, tensor_names: list[str], n_steps: int = 10, dim: int = 16,
    ) -> list[dict[str, torch.Tensor]]:
        """Create synthetic deltas with known dominant direction."""
        torch.manual_seed(42)
        directions = {
            name: torch.randn(dim) for name in tensor_names
        }
        for v in directions.values():
            v.div_(v.norm())

        deltas = []
        for t in range(n_steps):
            d = {}
            for name in tensor_names:
                noise = torch.randn(dim) * 0.1
                d[name] = directions[name] * (t + 1) * 0.5 + noise
            deltas.append(d)
        return deltas

    def test_returns_per_tensor_results(self):
        names = [
            "model.layers.0.self_attn.out_proj.lora_A.default.weight",
            "model.layers.0.mlp.gate_proj.lora_A.default.weight",
        ]
        deltas = self._make_dominant_deltas(names)
        results = analyze_tensor_deltas(deltas)

        assert len(results) == 2
        for name in names:
            assert name in results
            info = results[name]
            assert "rank1_dominance" in info
            assert "direction_stability" in info
            assert "layer_type" in info
            assert "n_snapshots" in info
            assert "rank1_null_expected" in info
            assert "rank1_z" in info
            assert info["n_snapshots"] == 10

    def test_rank1_dominance_high_for_dominant_direction(self):
        names = ["model.layers.0.self_attn.out_proj.lora_A.default.weight"]
        deltas = self._make_dominant_deltas(names, n_steps=20, dim=32)
        results = analyze_tensor_deltas(deltas)
        r1 = results[names[0]]["rank1_dominance"]
        assert r1 > 0.5, f"Dominant direction should give high rank-1, got {r1}"

    def test_layer_type_classification(self):
        names = [
            "model.layers.0.self_attn.out_proj.lora_A.default.weight",
            "model.layers.0.mlp.gate_proj.lora_A.default.weight",
        ]
        deltas = self._make_dominant_deltas(names)
        results = analyze_tensor_deltas(deltas)
        assert results[names[0]]["layer_type"] == "attention_out"
        assert results[names[1]]["layer_type"] == "mlp"

    def test_empty_deltas(self):
        assert analyze_tensor_deltas([]) == {}

    def test_single_step(self):
        deltas = [{"t0": torch.randn(4)}]
        assert analyze_tensor_deltas(deltas) == {}

    def test_tensor_name_filter(self):
        names = ["t0", "t1", "t2"]
        deltas = self._make_dominant_deltas(names)
        results = analyze_tensor_deltas(deltas, tensor_names=["t0", "t2"])
        assert "t0" in results
        assert "t2" in results
        assert "t1" not in results

    def test_z_score_positive_for_dominant(self):
        names = ["t0"]
        deltas = self._make_dominant_deltas(names, n_steps=20, dim=32)
        results = analyze_tensor_deltas(deltas)
        assert results["t0"]["rank1_z"] > 0, "Dominant direction should have positive z-score"


# ---------------------------------------------------------------------------
# group_by_layer_type
# ---------------------------------------------------------------------------

class TestGroupByLayerType:
    def test_groups_correctly(self):
        per_tensor = {
            "model.layers.0.self_attn.out_proj.lora_A.default.weight": {
                "rank1_dominance": 0.9,
                "direction_stability": 0.95,
                "layer_type": "attention_out",
                "n_snapshots": 10,
                "rank1_null_expected": 0.1,
                "rank1_z": 5.0,
            },
            "model.layers.0.mlp.gate_proj.lora_A.default.weight": {
                "rank1_dominance": 0.5,
                "direction_stability": 0.7,
                "layer_type": "mlp",
                "n_snapshots": 10,
                "rank1_null_expected": 0.1,
                "rank1_z": 2.0,
            },
            "model.layers.1.self_attn.out_proj.lora_A.default.weight": {
                "rank1_dominance": 0.85,
                "direction_stability": 0.90,
                "layer_type": "attention_out",
                "n_snapshots": 10,
                "rank1_null_expected": 0.1,
                "rank1_z": 4.5,
            },
        }
        groups = group_by_layer_type(per_tensor)
        assert "attention_out" in groups
        assert "mlp" in groups
        assert groups["attention_out"]["n_tensors"] == 2
        assert groups["mlp"]["n_tensors"] == 1

    def test_aggregation_values(self):
        per_tensor = {
            "t0": {
                "rank1_dominance": 0.8,
                "direction_stability": 0.9,
                "layer_type": "attention_out",
                "n_snapshots": 5,
                "rank1_null_expected": 0.1,
                "rank1_z": 4.0,
            },
            "t1": {
                "rank1_dominance": 0.6,
                "direction_stability": 0.7,
                "layer_type": "attention_out",
                "n_snapshots": 5,
                "rank1_null_expected": 0.1,
                "rank1_z": 3.0,
            },
        }
        groups = group_by_layer_type(per_tensor)
        atn = groups["attention_out"]
        assert atn["rank1_dominance_mean"] == pytest.approx(0.7)
        assert atn["rank1_z_mean"] == pytest.approx(3.5)
        assert atn["direction_stability_mean"] == pytest.approx(0.8)

    def test_none_direction_stability_handled(self):
        per_tensor = {
            "t0": {
                "rank1_dominance": 0.8,
                "direction_stability": None,
                "layer_type": "unknown",
                "n_snapshots": 5,
                "rank1_null_expected": 0.1,
                "rank1_z": 4.0,
            },
            "t1": {
                "rank1_dominance": 0.6,
                "direction_stability": 0.7,
                "layer_type": "unknown",
                "n_snapshots": 5,
                "rank1_null_expected": 0.1,
                "rank1_z": 3.0,
            },
        }
        groups = group_by_layer_type(per_tensor)
        unk = groups["unknown"]
        assert unk["direction_stability_mean"] == pytest.approx(0.7)
        assert unk["n_tensors"] == 2

    def test_empty_input(self):
        assert group_by_layer_type({}) == {}

    def test_end_to_end_with_analysis(self):
        """Full pipeline: create deltas → analyze → group."""
        names = [
            "model.layers.0.self_attn.out_proj.lora_A.default.weight",
            "model.layers.0.self_attn.v_proj.lora_A.default.weight",
            "model.layers.0.mlp.gate_proj.lora_A.default.weight",
        ]
        torch.manual_seed(42)
        # out_proj has strong dominant direction, mlp has weak
        directions = {
            names[0]: torch.randn(16) / torch.randn(16).norm(),
            names[1]: torch.randn(16) / torch.randn(16).norm(),
            names[2]: torch.randn(16) / torch.randn(16).norm(),
        }
        deltas = []
        for t in range(12):
            d = {}
            for name in names:
                noise = torch.randn(16) * 0.1
                # out_proj gets much stronger signal
                strength = 2.0 if "out_proj" in name else 0.3
                d[name] = directions[name] * (t + 1) * strength + noise
            deltas.append(d)

        per_tensor = analyze_tensor_deltas(deltas)
        groups = group_by_layer_type(per_tensor)

        assert "attention_out" in groups
        assert "attention_v" in groups
        assert "mlp" in groups
        # out_proj should have higher rank-1 dominance due to stronger signal
        assert groups["attention_out"]["rank1_dominance_mean"] > groups["mlp"]["rank1_dominance_mean"]
