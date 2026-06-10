import math

import torch
import pytest

from src.tg_lora.metrics import cosine_similarity, total_norm, per_layer_norms


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = {"w": torch.tensor([1.0, 2.0, 3.0])}
        b = {"w": torch.tensor([1.0, 2.0, 3.0])}
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = {"w": torch.tensor([1.0, 0.0])}
        b = {"w": torch.tensor([0.0, 1.0])}
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = {"w": torch.tensor([1.0, 0.0])}
        b = {"w": torch.tensor([-1.0, 0.0])}
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_multiple_keys(self):
        a = {"w1": torch.tensor([1.0]), "w2": torch.tensor([1.0])}
        b = {"w1": torch.tensor([1.0]), "w2": torch.tensor([1.0])}
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_zero_vectors_returns_zero(self):
        a = {"w": torch.zeros(3)}
        b = {"w": torch.tensor([1.0, 2.0, 3.0])}
        with pytest.warns(UserWarning, match="near-zero denominator"):
            assert cosine_similarity(a, b) == 0.0

    def test_multidimensional_tensors(self):
        a = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        b = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        assert cosine_similarity(a, b) == pytest.approx(1.0)


class TestTotalNorm:
    def test_single_tensor(self):
        state = {"w": torch.tensor([3.0, 4.0])}
        assert total_norm(state) == pytest.approx(5.0)

    def test_multiple_tensors(self):
        state = {
            "w1": torch.tensor([3.0, 0.0]),
            "w2": torch.tensor([0.0, 4.0]),
        }
        expected = math.sqrt(9.0 + 16.0)
        assert total_norm(state) == pytest.approx(expected)

    def test_empty_state(self):
        assert total_norm({}) == pytest.approx(0.0)

    def test_nan_tensor_skipped(self):
        state = {
            "good": torch.tensor([3.0, 4.0]),
            "bad": torch.tensor([float("nan"), 1.0]),
        }
        assert total_norm(state) == pytest.approx(5.0)

    def test_inf_tensor_skipped(self):
        state = {"good": torch.tensor([3.0, 4.0]), "bad": torch.tensor([float("inf")])}
        assert total_norm(state) == pytest.approx(5.0)

    def test_all_nonfinite_returns_zero(self):
        state = {"a": torch.tensor([float("nan")]), "b": torch.tensor([float("inf")])}
        assert total_norm(state) == pytest.approx(0.0)


class TestPerLayerNorms:
    def test_extracts_layer_numbers(self):
        state = {
            "model.layers.0.attn.w": torch.tensor([3.0, 4.0]),
            "model.layers.2.ffn.w": torch.tensor([5.0, 12.0]),
        }
        result = per_layer_norms(state)
        assert result["layer_0"] == pytest.approx(5.0)
        assert result["layer_2"] == pytest.approx(13.0)

    def test_aggregates_same_layer(self):
        state = {
            "model.layers.1.attn.q": torch.tensor([3.0]),
            "model.layers.1.attn.k": torch.tensor([4.0]),
        }
        result = per_layer_norms(state)
        assert result["layer_1"] == pytest.approx(5.0)

    def test_non_layer_key_goes_to_other(self):
        state = {"embedding.weight": torch.tensor([3.0, 4.0])}
        result = per_layer_norms(state)
        assert result["layer_other"] == pytest.approx(5.0)

    def test_empty_state(self):
        assert per_layer_norms({}) == {}

    def test_nan_tensor_skipped(self):
        state = {
            "model.layers.0.attn.w": torch.tensor([3.0, 4.0]),
            "model.layers.0.ffn.w": torch.tensor([float("nan"), 1.0]),
        }
        result = per_layer_norms(state)
        assert result["layer_0"] == pytest.approx(5.0)

    def test_inf_tensor_skipped(self):
        state = {
            "model.layers.1.w": torch.tensor([float("inf")]),
            "model.layers.1.v": torch.tensor([5.0, 12.0]),
        }
        result = per_layer_norms(state)
        assert result["layer_1"] == pytest.approx(13.0)


class TestCosineSimilarityKeyMismatch:
    def test_b_missing_key_from_a(self):
        """Regression: cosine_similarity must not raise KeyError when b lacks a key from a."""
        a = {"w1": torch.tensor([1.0, 0.0]), "w2": torch.tensor([0.0, 1.0])}
        b = {"w1": torch.tensor([1.0, 0.0])}
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_a_missing_key_from_b(self):
        """Keys only in b are ignored (by design: iteration is over a)."""
        a = {"w1": torch.tensor([1.0, 0.0])}
        b = {"w1": torch.tensor([1.0, 0.0]), "w2": torch.tensor([0.0, 1.0])}
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_completely_disjoint_keys(self):
        a = {"w1": torch.tensor([1.0, 0.0])}
        b = {"w2": torch.tensor([1.0, 0.0])}
        sim = cosine_similarity(a, b)
        assert sim == 0.0


class TestCosineSimilarityNonFinite:
    def test_nan_tensor_skipped(self):
        a = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([float("nan"), 1.0])}
        b = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([1.0, 0.0])}
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_inf_tensor_skipped(self):
        a = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([float("inf")])}
        b = {"good": torch.tensor([1.0, 0.0]), "bad": torch.tensor([1.0])}
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_all_nonfinite_returns_zero(self):
        a = {"bad1": torch.tensor([float("nan")]), "bad2": torch.tensor([float("inf")])}
        b = {"bad1": torch.tensor([1.0]), "bad2": torch.tensor([1.0])}
        sim = cosine_similarity(a, b)
        assert sim == 0.0

    def test_mixed_finite_and_nan_gives_valid_result(self):
        a = {
            "w1": torch.tensor([1.0, 0.0]),
            "w2": torch.tensor([float("nan")]),
        }
        b = {
            "w1": torch.tensor([0.0, 1.0]),
            "w2": torch.tensor([1.0]),
        }
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_nan_in_both_sides_skipped(self):
        a = {"w": torch.tensor([float("nan"), 1.0])}
        b = {"w": torch.tensor([float("nan"), 1.0])}
        sim = cosine_similarity(a, b)
        assert sim == 0.0
