import math

import pytest
import torch

from src.model.lora_utils import iter_lora_params
from src.tg_lora.extrapolator import (
    AlphaLineStepStats,
    ExtrapolationStats,
    ZerothOrderStepStats,
    alpha_line_reconstruct_output,
    alpha_line_step,
    apply_extrapolation,
    cap_update,
    subspace_zeroth_order_step,
)
from src.tg_lora.velocity import OrthonormalBasis

from .conftest import FakeLoRAModel


class AlphaLineLoRALinear(torch.nn.Module):
    def __init__(self, in_features: int = 5, out_features: int = 3, rank: int = 2):
        super().__init__()
        self.base = torch.nn.Linear(in_features, out_features, bias=False)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.1)
        self.lora_B = torch.nn.Parameter(torch.randn(out_features, rank) * 0.1)
        self.scaling = 0.75

    def forward(self, x):
        lora_hidden = torch.nn.functional.linear(x, self.lora_A)
        return self.base(x) + self.scaling * torch.nn.functional.linear(
            lora_hidden, self.lora_B
        )


def test_apply_extrapolation():
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}

    velocity = {}
    active_names = set()
    for name, p in iter_lora_params(model):
        velocity[name] = torch.ones_like(p) * 0.1
        active_names.add(name)

    stats = apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=active_names,
        n_steps=5,
        lr=0.3,
        relative_update_cap=100.0,  # high cap so no capping
    )
    assert isinstance(stats, ExtrapolationStats)
    assert stats.num_tensors == len(active_names)
    assert stats.capped_tensors > 0
    assert 0.0 < stats.global_cap_ratio < 1.0

    for name, p in iter_lora_params(model):
        # cap_update uses p.detach() as ref; for lora_B (zeros), ref_norm is
        # clamped to eps=1e-8 so the effective cap is tiny.  We only verify
        # that the update direction is correct and non-zero for non-zero
        # params (lora_A), and that zero params stay approximately zero.
        diff = p - before[name]
        if before[name].norm() > 1e-6:
            # Non-zero param: raw update should mostly go through
            expected_delta = torch.ones_like(p) * (5 * 0.3 * 0.1)
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
        n_steps=1,
        lr=1.0,
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
        n_steps=1,
        lr=1.0,
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


def test_apply_extrapolation_ignores_legacy_alpha_arguments():
    """alpha arguments are accepted for compatibility but no longer scale updates."""
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
        n_steps=1,
        lr=1.0,
        relative_update_cap=100.0,
        alpha_by_name={},
        default_alpha=1.0,
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
        n_steps=1,
        lr=1.0,
        relative_update_cap=100.0,
        alpha_by_name={all_names[0]: 2.0},
        default_alpha=1.0,
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

    for name in all_names:
        assert abs(diff_b[name] - diff_a[name]) < 1e-6


def test_apply_extrapolation_zero_steps_is_noop():
    """n_steps=0 must not modify any parameters."""
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}
    velocity = {name: torch.ones_like(p) * 0.1 for name, p in iter_lora_params(model)}

    stats = apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=set(velocity.keys()),
        n_steps=0,
        lr=1.0,
        relative_update_cap=100.0,
    )
    assert stats.num_tensors == 0

    for name, p in iter_lora_params(model):
        assert torch.allclose(p, before[name])


def test_apply_extrapolation_negative_steps_is_noop():
    """n_steps<0 must not modify any parameters."""
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}
    velocity = {name: torch.ones_like(p) * 0.1 for name, p in iter_lora_params(model)}

    stats = apply_extrapolation(
        model=model,
        velocity=velocity,
        active_names=set(velocity.keys()),
        n_steps=-5,
        lr=1.0,
        relative_update_cap=100.0,
    )
    assert stats.num_tensors == 0

    for name, p in iter_lora_params(model):
        assert torch.allclose(p, before[name])


def test_alpha_line_reconstruction_matches_full_forward_with_factor_updates():
    torch.manual_seed(0)
    model = AlphaLineLoRALinear().double()
    cached_h = torch.randn(4, 5, dtype=torch.float64)
    alpha = 0.37
    direction = {
        "layer.lora_A": torch.randn_like(model.lora_A) * 0.03,
        "layer.lora_B": torch.randn_like(model.lora_B) * 0.03,
    }

    base_out = model(cached_h).detach()
    reconstructed = alpha_line_reconstruct_output(
        model,
        cached_h,
        direction,
        alpha,
        base_out=base_out,
    )

    with torch.no_grad():
        model.lora_A.add_(direction["layer.lora_A"], alpha=alpha)
        model.lora_B.add_(direction["layer.lora_B"], alpha=alpha)
        full_forward = model(cached_h)
        model.lora_A.add_(direction["layer.lora_A"], alpha=-alpha)
        model.lora_B.add_(direction["layer.lora_B"], alpha=-alpha)

    assert torch.allclose(reconstructed, full_forward, atol=1e-12, rtol=1e-10)


def test_alpha_line_step_updates_scalar_alpha_only():
    torch.manual_seed(1)
    model = AlphaLineLoRALinear().double()
    cached_h = torch.randn(4, 5, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    direction = {
        "lora_A": torch.randn_like(model.lora_A) * 0.02,
        "lora_B": torch.randn_like(model.lora_B) * 0.02,
    }
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }

    alpha_after, stats = alpha_line_step(
        model,
        cached_h,
        direction,
        alpha=0.1,
        alpha_lr=0.5,
        loss_fn=lambda output: torch.nn.functional.mse_loss(output, target),
    )

    assert isinstance(stats, AlphaLineStepStats)
    assert alpha_after == pytest.approx(stats.alpha_after)
    assert stats.grad_alpha != 0.0
    for name, param in model.named_parameters():
        if name in before:
            assert torch.allclose(param, before[name])


def test_alpha_line_step_non_finite_gradient_returns_original_alpha():
    """When loss produces NaN gradient, alpha must remain unchanged."""
    model = AlphaLineLoRALinear().double()
    cached_h = torch.randn(4, 5, dtype=torch.float64)
    direction = {
        "lora_A": torch.randn_like(model.lora_A) * 0.02,
        "lora_B": torch.randn_like(model.lora_B) * 0.02,
    }
    original_alpha = 0.1

    def nan_loss(output):
        return output.mean() * float("nan")

    alpha_after, stats = alpha_line_step(
        model,
        cached_h,
        direction,
        alpha=original_alpha,
        alpha_lr=0.5,
        loss_fn=nan_loss,
    )

    assert alpha_after == original_alpha
    assert math.isnan(stats.grad_alpha) or math.isinf(stats.grad_alpha)


def test_subspace_zeroth_order_step_accepts_quadratic_minimum():
    model = FakeLoRAModel()
    params = dict(iter_lora_params(model))
    name = "linear.lora_A"
    direction = torch.ones_like(params[name])
    direction = direction / direction.norm()
    target = params[name].detach().clone() + 0.25 * direction
    basis = OrthonormalBasis(
        vectors=[{name: direction}],
        dim=1,
        residual_norm=0.0,
        short_norm=1.0,
        long_norm=1.0,
        tau_dim=0.15,
    )

    def loss_closure():
        return ((params[name] - target) ** 2).sum().item()

    stats = subspace_zeroth_order_step(
        model,
        basis,
        {name},
        loss_closure,
        mu_ratio=0.01,
        max_step_ratio=100.0,
    )

    assert isinstance(stats, ZerothOrderStepStats)
    assert stats.accepted
    assert stats.dim == 1
    assert stats.forward_count == 4
    assert loss_closure() < ((params[name].detach() - 0.25 * direction - target) ** 2).sum().item()


def test_subspace_zeroth_order_step_rejects_and_rolls_back():
    model = FakeLoRAModel()
    params = dict(iter_lora_params(model))
    name = "linear.lora_A"
    before = params[name].detach().clone()
    direction = torch.ones_like(params[name])
    direction = direction / direction.norm()
    basis = OrthonormalBasis(
        vectors=[{name: direction}],
        dim=1,
        residual_norm=0.0,
        short_norm=1.0,
        long_norm=1.0,
        tau_dim=0.15,
    )
    losses = iter([1.0, 0.5, 0.0, 2.0])

    def loss_closure():
        return next(losses)

    stats = subspace_zeroth_order_step(
        model,
        basis,
        {name},
        loss_closure,
        mu_ratio=0.01,
        max_step_ratio=100.0,
        tolerance=0.0,
    )

    assert not stats.accepted
    assert stats.rollback_triggered
    assert torch.allclose(params[name], before)


def test_subspace_zeroth_order_step_stops_on_non_descent_primary_g():
    model = FakeLoRAModel()
    params = dict(iter_lora_params(model))
    name = "linear.lora_A"
    before = params[name].detach().clone()
    direction = torch.ones_like(params[name])
    direction = direction / direction.norm()
    basis = OrthonormalBasis(
        vectors=[{name: direction}],
        dim=1,
        residual_norm=0.0,
        short_norm=1.0,
        long_norm=1.0,
        tau_dim=0.15,
    )
    losses = iter([1.0, 1.1, 1.2])

    def loss_closure():
        return next(losses)

    stats = subspace_zeroth_order_step(
        model,
        basis,
        {name},
        loss_closure,
        mu_ratio=0.01,
        max_step_ratio=100.0,
        stop_on_positive_primary_g=True,
    )

    assert not stats.accepted
    assert not stats.rollback_triggered
    assert stats.termination_reason == "primary_g_non_descent"
    assert stats.forward_count == 3
    assert stats.directions[0].g > 0
    assert torch.allclose(params[name], before)


def test_apply_extrapolation_empty_velocity_is_noop():
    """Empty velocity dict must not modify any parameters."""
    model = FakeLoRAModel()
    before = {name: p.clone() for name, p in iter_lora_params(model)}

    stats = apply_extrapolation(
        model=model,
        velocity={},
        active_names=set(before.keys()),
        n_steps=5,
        lr=1.0,
        relative_update_cap=100.0,
    )
    assert stats.num_tensors == 0

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


def test_subspace_m9_fit_step():
    from src.tg_lora.extrapolator import subspace_m9_fit_step
    model = FakeLoRAModel()
    active_names = {name for name, _ in iter_lora_params(model)}
    
    # Create mock trajectory history (3 cycles)
    history = []
    for _ in range(3):
        h_dict = {}
        for name, p in iter_lora_params(model):
            h_dict[name] = torch.randn_like(p) * 0.01
        history.append(h_dict)
        
    dummy_batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    
    # Dummy loss function (MSE of parameter values to make fitting meaningful)
    def dummy_loss(batch):
        tot = 0.0
        for name, p in iter_lora_params(model):
            tot += p.abs().mean().item()
        return tot

    # Call fit step
    m9_delta, stats = subspace_m9_fit_step(
        model=model,
        history=history,
        active_names=active_names,
        batch=dummy_batch,
        loss_fn=dummy_loss,
        selected_N=5,
        fd_epsilon=1e-3,
        fit_lr=0.1,
        fit_steps=2,
    )
    
    # Assertions
    assert isinstance(m9_delta, dict)
    assert set(m9_delta.keys()) == active_names
    for name in active_names:
        assert m9_delta[name].shape == dict(iter_lora_params(model))[name].shape
        
    assert "loss_initial" in stats
    assert "loss_final" in stats
    assert "alpha_fit" in stats
    assert "beta1_fit" in stats
    assert "beta2_fit" in stats
    assert "w_traj" in stats
    assert stats["w_traj"] > 0


def test_subspace_m9_scaling():
    import math
    from src.tg_lora.extrapolator import subspace_m9_fit_step, flatten_tensor_dict
    model = FakeLoRAModel()
    active_names = {name for name, _ in iter_lora_params(model)}
    
    # Create mock trajectory history (3 cycles)
    history = []
    for _ in range(3):
        h_dict = {}
        for name, p in iter_lora_params(model):
            # Use deterministic history
            h_dict[name] = torch.ones_like(p) * 0.01
        history.append(h_dict)
        
    dummy_batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    
    # Dummy loss function
    def dummy_loss(batch):
        return 0.0

    norms = {}
    for N in [1, 5, 10, 20]:
        m9_delta, stats = subspace_m9_fit_step(
            model=model,
            history=history,
            active_names=active_names,
            batch=dummy_batch,
            loss_fn=dummy_loss,
            selected_N=N,
            fd_epsilon=1e-3,
            fit_lr=0.1,
            fit_steps=0,  # 0 steps to keep alpha=1.0 and beta1=beta2=0.0
        )
        flat_delta = flatten_tensor_dict(m9_delta)
        norms[N] = flat_delta.norm().item()
        
    # Norm of speculative update should be exactly linear in N since beta1=beta2=0
    assert math.isclose(norms[5], 5 * norms[1], rel_tol=1e-4)
    assert math.isclose(norms[10], 10 * norms[1], rel_tol=1e-4)
    assert math.isclose(norms[20], 20 * norms[1], rel_tol=1e-4)

