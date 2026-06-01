import pytest
import torch

from tg_lora.lora_utils import iter_lora_params
from tg_lora.extrapolator import apply_extrapolation, cap_update

from conftest import FakeLoRAModel


def test_apply_extrapolation():
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}

    velocity = {}
    active_names = set()
    for name, p in iter_lora_params(model):
        velocity[name] = torch.ones_like(p) * 0.1
        active_names.add(name)

    apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=active_names,
        alpha_by_name={},
        default_alpha=0.3,
        n_steps=5,
        relative_update_cap=100.0,  # high cap so no capping
    )

    for name, p in iter_lora_params(model):
        # cap_update uses p.detach() as ref; for lora_B (zeros), ref_norm is
        # clamped to eps=1e-8 so the effective cap is tiny.  We only verify
        # that the update direction is correct and non-zero for non-zero
        # params (lora_A), and that zero params stay approximately zero.
        diff = p - before[name]
        if before[name].norm() > 1e-6:
            # Non-zero param: raw update should mostly go through
            expected_delta = torch.ones_like(p) * (5 * 0.3 * 0.1)  # n_steps * alpha * v
            assert torch.allclose(diff, expected_delta, atol=1e-3)
        else:
            # Zero param (lora_B): capped update is tiny but non-zero
            assert diff.norm().item() < 1e-3


def test_apply_extrapolation_partial_layers():
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}

    velocity = {}
    all_names = list(before.keys())
    # Only activate lora_A
    active_names = {all_names[0]}

    for name, p in iter_lora_params(model):
        velocity[name] = torch.ones_like(p) * 0.1

    apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=active_names,
        alpha_by_name={},
        default_alpha=1.0,
        n_steps=1,
        relative_update_cap=1.0,
    )

    for name, p in iter_lora_params(model):
        if name in active_names:
            assert not torch.allclose(p, before[name])
        else:
            assert torch.allclose(p, before[name])


def test_cap_update():
    ref = torch.tensor([1.0, 0.0, 0.0])
    update = torch.tensor([10.0, 0.0, 0.0])
    capped = cap_update(update, ref, max_ratio=0.5)

    assert capped.norm().item() <= 0.5 * ref.norm().item() + 1e-6


def test_cap_update_no_cap_needed():
    ref = torch.tensor([1.0, 0.0, 0.0])
    update = torch.tensor([0.001, 0.0, 0.0])
    capped = cap_update(update, ref, max_ratio=0.5)

    assert torch.allclose(capped, update)


def test_apply_extrapolation_skips_missing_velocity_keys():
    """Regression: active_names containing a key absent from velocity must not raise KeyError."""
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}

    all_names = list(before.keys())
    velocity = {all_names[0]: torch.ones_like(before[all_names[0]]) * 0.1}
    # active_names includes a key NOT in velocity
    active_names = set(all_names)

    apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=active_names,
        alpha_by_name={},
        default_alpha=1.0,
        n_steps=1,
        relative_update_cap=100.0,
    )

    # Param with velocity should have changed
    assert not torch.allclose(
        dict(iter_lora_params(model))[all_names[0]], before[all_names[0]]
    )
    # Param without velocity should be unchanged
    assert torch.allclose(
        dict(iter_lora_params(model))[all_names[1]], before[all_names[1]]
    )


def test_cap_update_zero_ref_uses_eps_floor():
    """When ref norm is zero, eps clamp prevents division by zero and caps the update."""
    ref = torch.zeros(4)
    update = torch.tensor([1.0, 2.0, 3.0, 4.0])
    capped = cap_update(update, ref, max_ratio=0.01, eps=1e-8)
    # max_norm = 0.01 * eps = 1e-10, update_norm = sqrt(30) ≈ 5.48
    # capped = update * (1e-10 / 5.48) ≈ tiny
    assert capped.norm().item() < 1e-9


def test_cap_update_zero_update_returns_zero():
    """A zero update should remain zero (not NaN or eps)."""
    ref = torch.tensor([1.0, 0.0, 0.0])
    update = torch.zeros(3)
    capped = cap_update(update, ref, max_ratio=0.5)
    assert torch.allclose(capped, torch.zeros(3))


def test_apply_extrapolation_per_name_alpha_override():
    """alpha_by_name should override default_alpha for specific params."""
    model = FakeLoRAModel()
    all_names = [name for name, _ in iter_lora_params(model)]
    velocity = {
        name: torch.ones_like(dict(iter_lora_params(model))[name]) * 0.1
        for name in all_names
    }

    # Run once with default alpha
    model_a = FakeLoRAModel()
    for name, p in iter_lora_params(model_a):
        p.data.copy_(dict(iter_lora_params(model))[name])
    apply_extrapolation(
        model=model_a,
        velocity=velocity,
        active_names=set(all_names),
        alpha_by_name={},
        default_alpha=1.0,
        n_steps=1,
        relative_update_cap=100.0,
    )
    diff_a = {
        name: (
            dict(iter_lora_params(model_a))[name] - dict(iter_lora_params(model))[name]
        )
        .abs()
        .sum()
        .item()
        for name in all_names
    }

    # Run again with per-name alpha override (2x for first param)
    model_b = FakeLoRAModel()
    for name, p in iter_lora_params(model_b):
        p.data.copy_(dict(iter_lora_params(model))[name])
    apply_extrapolation(
        model=model_b,
        velocity=velocity,
        active_names=set(all_names),
        alpha_by_name={all_names[0]: 2.0},
        default_alpha=1.0,
        n_steps=1,
        relative_update_cap=100.0,
    )
    diff_b = {
        name: (
            dict(iter_lora_params(model_b))[name] - dict(iter_lora_params(model))[name]
        )
        .abs()
        .sum()
        .item()
        for name in all_names
    }

    # Param with override alpha should have ~2x the diff of the same param with default
    assert diff_b[all_names[0]] > diff_a[all_names[0]] * 1.5
    # Other params should be identical
    assert abs(diff_b[all_names[1]] - diff_a[all_names[1]]) < 1e-6


def test_apply_extrapolation_zero_steps_is_noop():
    """n_steps=0 must not modify any parameters."""
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}
    velocity = {name: torch.ones_like(p) * 0.1 for name, p in iter_lora_params(model)}

    apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=set(velocity.keys()),
        alpha_by_name={},
        default_alpha=1.0,
        n_steps=0,
        relative_update_cap=100.0,
    )

    for name, p in iter_lora_params(model):
        assert torch.allclose(p, before[name])


def test_apply_extrapolation_negative_steps_is_noop():
    """n_steps<0 must not modify any parameters."""
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}
    velocity = {name: torch.ones_like(p) * 0.1 for name, p in iter_lora_params(model)}

    apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=set(velocity.keys()),
        alpha_by_name={},
        default_alpha=1.0,
        n_steps=-5,
        relative_update_cap=100.0,
    )

    for name, p in iter_lora_params(model):
        assert torch.allclose(p, before[name])


def test_apply_extrapolation_empty_velocity_is_noop():
    """Empty velocity dict must not modify any parameters."""
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}

    apply_extrapolation(
        model=model,
        velocity={},
        active_names=set(before.keys()),
        alpha_by_name={},
        default_alpha=1.0,
        n_steps=5,
        relative_update_cap=100.0,
    )

    for name, p in iter_lora_params(model):
        assert torch.allclose(p, before[name])


class TestCapUpdateValidation:
    def test_negative_max_ratio_raises(self):
        with pytest.raises(ValueError, match="max_ratio must be positive"):
            cap_update(torch.ones(3), torch.ones(3), max_ratio=-0.1)

    def test_zero_max_ratio_raises(self):
        with pytest.raises(ValueError, match="max_ratio must be positive"):
            cap_update(torch.ones(3), torch.ones(3), max_ratio=0)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            cap_update(torch.ones(3), torch.ones(3), eps=-1e-8)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            cap_update(torch.ones(3), torch.ones(3), eps=0)


# --- TASK-0079: data_ptr / identity preservation tests ---


class TestCapUpdateInPlace:
    """Verify cap_update() returns the same tensor object when applying in-place caps."""

    def test_capping_returns_same_tensor_object(self):
        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([10.0, 0.0, 0.0])
        capped = cap_update(update, ref, max_ratio=0.5)
        assert capped is update

    def test_no_cap_needed_returns_same_tensor_object(self):
        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([0.001, 0.0, 0.0])
        capped = cap_update(update, ref, max_ratio=0.5)
        assert capped is update

    def test_capping_preserves_data_ptr(self):
        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([10.0, 0.0, 0.0])
        ptr_before = update.data_ptr()
        cap_update(update, ref, max_ratio=0.5)
        assert update.data_ptr() == ptr_before

    def test_nan_input_returns_new_tensor(self):
        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([float("nan"), 1.0, 0.0])
        ptr_before = update.data_ptr()
        result = cap_update(update, ref, max_ratio=0.5)
        assert result.data_ptr() != ptr_before

    def test_inf_input_returns_new_tensor(self):
        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([float("inf"), 0.0, 0.0])
        ptr_before = update.data_ptr()
        result = cap_update(update, ref, max_ratio=0.5)
        assert result.data_ptr() != ptr_before

    def test_nan_result_is_zeros(self):
        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([float("nan"), 1.0, 0.0])
        result = cap_update(update, ref, max_ratio=0.5)
        assert torch.allclose(result, torch.zeros(3))

    def test_inf_result_is_zeros(self):
        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([float("inf"), 0.0, 0.0])
        result = cap_update(update, ref, max_ratio=0.5)
        assert torch.allclose(result, torch.zeros(3))


class TestCapUpdateNonFiniteLogging:
    """Verify that cap_update emits a warning when non-finite values are detected."""

    def test_nan_update_emits_warning(self, caplog):
        import logging

        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([float("nan"), 1.0, 0.0])
        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            result = cap_update(update, ref, max_ratio=0.5)
        assert torch.allclose(result, torch.zeros(3))
        assert "non-finite update detected" in caplog.text
        assert "NaN" in caplog.text

    def test_inf_update_emits_warning(self, caplog):
        import logging

        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([float("inf"), 0.0, 0.0])
        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            result = cap_update(update, ref, max_ratio=0.5)
        assert torch.allclose(result, torch.zeros(3))
        assert "non-finite update detected" in caplog.text
        assert "Inf" in caplog.text

    def test_finite_update_no_warning(self, caplog):
        import logging

        ref = torch.tensor([1.0, 0.0, 0.0])
        update = torch.tensor([10.0, 0.0, 0.0])
        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            cap_update(update, ref, max_ratio=0.5)
        assert "non-finite update detected" not in caplog.text

    def test_warning_reports_nan_and_inf_counts(self, caplog):
        import logging

        ref = torch.tensor([1.0, 0.0, 0.0, 0.0])
        update = torch.tensor([float("nan"), float("inf"), 1.0, 0.0])
        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            cap_update(update, ref, max_ratio=0.5)
        assert "1 NaN" in caplog.text
        assert "1 Inf" in caplog.text
