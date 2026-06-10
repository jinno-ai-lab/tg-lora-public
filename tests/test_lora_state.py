import pytest
import torch

from src.model.lora_utils import iter_lora_params
from src.tg_lora.lora_state import (
    snapshot_lora,
    snapshot_lora_delta,
    apply_delta_snapshot,
    load_lora_snapshot,
    diff_lora,
)

from .conftest import FakeLoRAModel


class TestLoRAState:
    def test_snapshot_and_restore(self):
        model = FakeLoRAModel()
        snap = snapshot_lora(model)

        # Modify params
        for name, p in iter_lora_params(model):
            p.data += torch.ones_like(p)

        # Restore
        load_lora_snapshot(model, snap)

        for name, p in iter_lora_params(model):
            assert torch.allclose(p.cpu(), snap[name])

    def test_diff_lora(self):
        model = FakeLoRAModel()
        before = snapshot_lora(model)

        for name, p in iter_lora_params(model):
            p.data += torch.ones_like(p) * 2.0

        after = snapshot_lora(model)
        delta = diff_lora(after, before, scale=1.0)

        for k in delta:
            expected = torch.ones_like(delta[k]) * 2.0
            assert torch.allclose(delta[k], expected, atol=1e-6)

    def test_diff_lora_with_scale(self):
        model = FakeLoRAModel()
        before = snapshot_lora(model)

        for name, p in iter_lora_params(model):
            p.data += torch.ones_like(p) * 3.0

        after = snapshot_lora(model)
        delta = diff_lora(after, before, scale=1.0 / 3.0)

        for k in delta:
            expected = torch.ones_like(delta[k]) * 1.0
            assert torch.allclose(delta[k], expected, atol=1e-6)

    def test_diff_lora_missing_after_keys(self):
        """Regression: diff_lora must not raise KeyError when after is missing keys from before."""
        before = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        after = {"a": torch.tensor([3.0])}  # "b" missing

        delta = diff_lora(after, before)
        assert "a" in delta
        assert torch.allclose(delta["a"], torch.tensor([2.0]))
        # "b" should be skipped (not in after)
        assert "b" not in delta

    # --- REQ-140: diff_lora fast paths (TC-140-B01, TC-140-B02) ---

    def test_diff_lora_scale_zero_returns_zeros(self):
        """TC-140-B01: scale==0.0 returns all-zero tensors (fast path)."""
        before = {"a": torch.randn(3, 4), "b": torch.randn(2, 5)}
        after = {"a": torch.randn(3, 4), "b": torch.randn(2, 5)}

        delta = diff_lora(after, before, scale=0.0)

        assert set(delta.keys()) == {"a", "b"}
        assert torch.all(delta["a"] == 0)
        assert torch.all(delta["b"] == 0)
        # Must have same shape as inputs
        assert delta["a"].shape == before["a"].shape
        assert delta["b"].shape == before["b"].shape

    def test_diff_lora_scale_one_no_multiplication(self):
        """TC-140-B02: scale==1.0 returns exact after-before (fast path, no multiplication)."""
        before = {"x": torch.tensor([1.0, 2.0, 3.0]), "y": torch.tensor([4.0])}
        after = {"x": torch.tensor([4.0, 5.0, 6.0]), "y": torch.tensor([7.0])}

        delta = diff_lora(after, before, scale=1.0)

        expected = {"x": torch.tensor([3.0, 3.0, 3.0]), "y": torch.tensor([3.0])}
        for k in delta:
            assert torch.allclose(delta[k], expected[k])

    def test_load_lora_snapshot_missing_keys(self):
        """Regression: load_lora_snapshot must not raise KeyError when state is missing model params."""
        model = FakeLoRAModel()
        original = {name: p.clone() for name, p in iter_lora_params(model)}

        # Create a state with only one of the two LoRA params
        all_names = list(original.keys())
        partial_state = {all_names[0]: torch.zeros_like(original[all_names[0]])}

        load_lora_snapshot(model, partial_state)

        # The param in state should be updated
        assert torch.allclose(
            dict(iter_lora_params(model))[all_names[0]],
            partial_state[all_names[0]],
        )
        # The param NOT in state should remain unchanged
        assert torch.allclose(
            dict(iter_lora_params(model))[all_names[1]],
            original[all_names[1]],
        )

    # --- snapshot_lora_delta ---

    def test_snapshot_lora_delta_basic(self):
        """snapshot_lora_delta returns base-relative deltas."""
        model = FakeLoRAModel()
        base = snapshot_lora(model)

        for name, p in iter_lora_params(model):
            p.data += torch.ones_like(p) * 5.0

        delta = snapshot_lora_delta(model, base)

        for name in delta:
            expected = torch.ones_like(delta[name]) * 5.0
            assert torch.allclose(delta[name], expected, atol=1e-6)

    def test_snapshot_lora_delta_roundtrip(self):
        """apply_delta_snapshot restores model to base + delta state."""
        model = FakeLoRAModel()
        base = snapshot_lora(model)

        for name, p in iter_lora_params(model):
            p.data += torch.ones_like(p) * 3.0

        delta = snapshot_lora_delta(model, base)

        # Create fresh model at base state, apply delta, verify matches
        model2 = FakeLoRAModel()
        load_lora_snapshot(model2, base)

        apply_delta_snapshot(model2, base, delta)

        for (n1, p1), (n2, p2) in zip(
            iter_lora_params(model), iter_lora_params(model2)
        ):
            assert n1 == n2
            assert torch.allclose(p1.cpu(), p2.cpu(), atol=1e-6)

    def test_snapshot_lora_delta_rejects_empty_base(self):
        """snapshot_lora_delta raises ValueError when base is empty."""
        model = FakeLoRAModel()
        with pytest.raises(ValueError, match="base snapshot must not be empty"):
            snapshot_lora_delta(model, {})

    def test_snapshot_lora_delta_skips_unknown_keys(self):
        """Params not in base are silently skipped."""
        model = FakeLoRAModel()
        all_names = list(iter_lora_params(model))

        # base with only first param
        base = {all_names[0][0]: all_names[0][1].clone()}

        delta = snapshot_lora_delta(model, base)
        assert len(delta) == 1

    def test_snapshot_lora_delta_zero_change(self):
        """Delta is zero when model hasn't changed from base."""
        model = FakeLoRAModel()
        base = snapshot_lora(model)

        delta = snapshot_lora_delta(model, base)

        for name in delta:
            assert torch.all(delta[name] == 0)

    # --- apply_delta_snapshot ---

    def test_apply_delta_snapshot_partial_restore(self):
        """apply_delta_snapshot skips params missing from base or delta."""
        model = FakeLoRAModel()
        original = {n: p.clone() for n, p in iter_lora_params(model)}
        base = snapshot_lora(model)
        all_names = list(original.keys())

        # Delta with only one param
        partial_delta = {all_names[0]: torch.ones_like(original[all_names[0]]) * 7.0}

        # Modify both params
        for name, p in iter_lora_params(model):
            p.data += torch.ones_like(p) * 100.0

        apply_delta_snapshot(model, base, partial_delta)

        restored = dict(iter_lora_params(model))
        # First param: base + delta
        assert torch.allclose(
            restored[all_names[0]], base[all_names[0]] + partial_delta[all_names[0]], atol=1e-6
        )
        # Second param: unchanged (not in delta)
        assert not torch.allclose(restored[all_names[1]], original[all_names[1]])

    def test_apply_delta_snapshot_dtype_preserved(self):
        """Restored tensors match the model param dtype."""
        model = FakeLoRAModel()
        base = snapshot_lora(model)
        delta = {n: torch.ones_like(p) for n, p in iter_lora_params(model)}

        apply_delta_snapshot(model, base, delta)

        for name, p in iter_lora_params(model):
            expected = base[name] + delta[name]
            assert p.dtype == expected.dtype
