"""Edge case tests for TG-LoRA modules.

Covers NaN/Inf input handling, zero tensor velocity updates, empty LoRA
parameter sets, extreme alpha/beta values, single-layer models, and
non-finite values in velocity extrapolation.
"""
from __future__ import annotations

import math

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
from tg_lora.random_walk_controller import RandomWalkController
from tg_lora.rollback_manager import RollbackManager
from tg_lora.velocity import Velocity

# ---------------------------------------------------------------------------
# Shared model helpers
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Minimal LoRA-enabled linear layer for testing."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)


class FakeTransformerModel(nn.Module):
    """Fake transformer with configurable layers for testing."""

    def __init__(self, num_layers: int = 12) -> None:
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


class FakeLoRAModel(nn.Module):
    """Single-layer model for simple rollback / velocity tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = LoRALinear(4, 4)


# ===========================================================================
# NaN / Inf input handling across all modules
# ===========================================================================


class TestNaNInfVelocity:
    """NaN/Inf inputs to Velocity."""

    def test_nan_delta_first_update(self) -> None:
        v = Velocity()
        delta = {"w": torch.tensor([float("nan"), 1.0])}
        v.update(delta, beta=0.8)
        # State is stored, but magnitude should skip NaN
        assert all(math.isfinite(m) for m in v.magnitudes)

    def test_inf_delta_first_update(self) -> None:
        v = Velocity()
        delta = {"w": torch.tensor([float("inf"), 1.0])}
        v.update(delta, beta=0.8)
        assert all(math.isfinite(m) for m in v.magnitudes)

    def test_neg_inf_delta_first_update(self) -> None:
        v = Velocity()
        delta = {"w": torch.tensor([float("-inf"), 1.0])}
        v.update(delta, beta=0.8)
        assert all(math.isfinite(m) for m in v.magnitudes)


class TestNaNInfRollback:
    """NaN/Inf inputs to RollbackManager."""

    def test_save_with_all_nan_model(self) -> None:
        model = FakeLoRAModel()
        model.linear.lora_A.data.fill_(float("nan"))
        model.linear.lora_B.data.fill_(float("nan"))
        mgr = RollbackManager()
        mgr.save(model)
        for tensor in mgr._history[0].values():
            assert torch.isfinite(tensor).all()

    def test_save_with_all_inf_model(self) -> None:
        model = FakeLoRAModel()
        model.linear.lora_A.data.fill_(float("inf"))
        model.linear.lora_B.data.fill_(float("-inf"))
        mgr = RollbackManager()
        mgr.save(model)
        for tensor in mgr._history[0].values():
            assert torch.isfinite(tensor).all()

    def test_nan_rollback_then_modify_restores_finite(self) -> None:
        model = FakeLoRAModel()
        model.linear.lora_A.data.fill_(float("nan"))
        mgr = RollbackManager()
        mgr.save(model)

        model.linear.lora_A.data.fill_(42.0)
        mgr.rollback(model, 0)
        assert torch.isfinite(model.linear.lora_A.data).all()


class TestNaNInfRandomWalk:
    """NaN/Inf inputs to RandomWalkController."""

    def test_accept_both_nan(self) -> None:
        ctrl = RandomWalkController(
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        assert ctrl.accept(float("nan"), float("nan")) is False

    def test_accept_nan_pilot_inf_after(self) -> None:
        ctrl = RandomWalkController(
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        assert ctrl.accept(float("nan"), float("inf")) is False


# ===========================================================================
# Zero tensor velocity updates
# ===========================================================================


class TestZeroTensorVelocity:
    """Zero tensor deltas in velocity updates."""

    def test_zero_delta_first_update(self) -> None:
        v = Velocity()
        delta = {"w": torch.zeros(4)}
        result = v.update(delta, beta=0.8)
        assert torch.allclose(result["w"], torch.zeros(4))
        assert len(v.magnitudes) == 1
        assert v.magnitudes[0] == 0.0

    def test_zero_delta_after_nonzero(self) -> None:
        v = Velocity()
        v.update({"w": torch.tensor([3.0, 4.0])}, beta=0.8)
        result = v.update({"w": torch.zeros(2)}, beta=0.8)
        # EMA: 0.8 * [3, 4] + 0.2 * [0, 0] = [2.4, 3.2]
        assert torch.allclose(result["w"], torch.tensor([2.4, 3.2]))

    def test_repeated_zero_deltas_converge_to_zero(self) -> None:
        v = Velocity()
        v.update({"w": torch.tensor([10.0])}, beta=0.8)
        for _ in range(100):
            v.update({"w": torch.zeros(1)}, beta=0.8)
        assert abs(v.state["w"].item()) < 1e-4

    def test_magnitude_is_zero_after_zero_updates(self) -> None:
        v = Velocity()
        v.update({"w": torch.zeros(4)}, beta=0.8)
        assert v.magnitudes[0] == 0.0


# ===========================================================================
# Empty LoRA parameter sets
# ===========================================================================


class TestEmptyLoRAParameterSets:
    """Models with no LoRA parameters."""

    def test_iter_lora_params_empty_model(self) -> None:
        model = nn.Module()
        assert list(iter_lora_params(model)) == []

    def test_iter_all_lora_params_empty_model(self) -> None:
        model = nn.Module()
        assert list(iter_all_lora_params(model)) == []

    def test_iter_lora_params_by_layer_empty_model(self) -> None:
        model = nn.Module()
        assert iter_lora_params_by_layer(model) == {}

    def test_iter_all_lora_params_by_layer_empty_model(self) -> None:
        model = nn.Module()
        assert iter_all_lora_params_by_layer(model) == {}

    def test_count_lora_params_empty_model(self) -> None:
        model = nn.Module()
        assert count_lora_params(model) == 0

    def test_get_unmapped_lora_param_names_empty(self) -> None:
        model = nn.Module()
        assert get_unmapped_lora_param_names(model) == []

    def test_get_last_fraction_raises_on_empty(self) -> None:
        model = nn.Module()
        with pytest.raises(ValueError, match="No LoRA decoder layers found"):
            get_last_fraction_lora_layer_indices(model)

    def test_configure_trainable_lora_scope_all_empty(self) -> None:
        model = nn.Module()
        names, indices = configure_trainable_lora_scope(model, "all")
        assert names == set()
        assert indices == set()

    def test_rollback_empty_model(self) -> None:
        model = nn.Module()
        mgr = RollbackManager()
        idx = mgr.save(model)
        assert idx == 0
        mgr.rollback(model, 0)  # should not raise


# ===========================================================================
# Extreme alpha/beta values
# ===========================================================================


class TestExtremeAlphaBeta:
    """Extreme alpha and beta values in RandomWalkController."""

    @pytest.mark.parametrize("beta", [0.0, 1.0])
    def test_beta_at_boundaries(self, beta: float) -> None:
        ctrl = RandomWalkController(
            beta_initial=beta,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        assert ctrl.state.beta == beta

    def test_alpha_at_min(self) -> None:
        ctrl = RandomWalkController(
            alpha_initial=0.01,
            alpha_min=0.01,
            alpha_max=1.5,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        assert ctrl.state.alpha == 0.01

    def test_alpha_at_max(self) -> None:
        ctrl = RandomWalkController(
            alpha_initial=1.5,
            alpha_min=0.03,
            alpha_max=1.5,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        assert ctrl.state.alpha == 1.5

    def test_velocity_with_beta_zero(self) -> None:
        """beta=0.0 means no EMA — new delta replaces state entirely."""
        v = Velocity()
        v.update({"w": torch.tensor([1.0, 2.0])}, beta=0.0)
        result = v.update({"w": torch.tensor([3.0, 4.0])}, beta=0.0)
        # 0.0 * old + (1 - 0.0) * new = new
        assert torch.allclose(result["w"], torch.tensor([3.0, 4.0]))

    def test_velocity_with_beta_one(self) -> None:
        """beta=1.0 means full EMA retention — delta has no effect."""
        v = Velocity()
        v.update({"w": torch.tensor([1.0, 2.0])}, beta=1.0)
        result = v.update({"w": torch.tensor([100.0, 200.0])}, beta=1.0)
        # 1.0 * old + (1 - 1.0) * new = old
        assert torch.allclose(result["w"], torch.tensor([1.0, 2.0]))


# ===========================================================================
# Single-layer model layer selection
# ===========================================================================


class TestSingleLayerModel:
    """Layer selection with a single-layer model."""

    def test_get_last_fraction_single_layer(self) -> None:
        model = FakeTransformerModel(1)
        indices = get_last_fraction_lora_layer_indices(model)
        assert indices == {0}

    def test_configure_trainable_lora_scope_last_25_single_layer(self) -> None:
        model = FakeTransformerModel(1)
        names, indices = configure_trainable_lora_scope(model, "last_25_percent")
        assert indices == {0}
        assert len(names) > 0

    def test_configure_trainable_lora_scope_all_single_layer(self) -> None:
        model = FakeTransformerModel(1)
        names, indices = configure_trainable_lora_scope(model, "all")
        assert indices == {0}
        assert len(names) > 0

    def test_count_lora_params_single_layer(self) -> None:
        model = FakeTransformerModel(1)
        # 1 layer x 2 modules x 2 params x 64 = 256
        expected = 1 * 2 * 2 * 8 * 8
        assert count_lora_params(model) == expected

    def test_get_unmapped_lora_param_names_single_layer(self) -> None:
        model = FakeTransformerModel(1)
        unmapped = get_unmapped_lora_param_names(model)
        assert unmapped == []


# ===========================================================================
# Non-finite values in velocity extrapolation
# ===========================================================================


class TestNonFiniteVelocityExtrapolation:
    """Non-finite magnitude history and trend handling in Velocity."""

    def test_is_magnitude_anomalous_with_all_zero_history(self) -> None:
        v = Velocity()
        for _ in range(5):
            v.update({"w": torch.zeros(2)}, beta=0.8)
        # All magnitudes are 0, std < 1e-12, latest=0, mean*2=0 → not anomalous
        assert not v.is_magnitude_anomalous()

    def test_magnitude_trend_with_single_entry(self) -> None:
        v = Velocity()
        v.update({"w": torch.tensor([1.0])}, beta=0.8)
        assert v.magnitude_trend() == 0.0

    def test_magnitude_acceleration_with_three_constant(self) -> None:
        v = Velocity()
        for _ in range(3):
            v.update({"w": torch.tensor([5.0])}, beta=0.8)
        acc = v.magnitude_acceleration()
        # Constant magnitudes → acceleration ~0
        assert abs(acc) < 1e-6

    def test_cosine_similarity_all_nan_returns_zero(self) -> None:
        v = Velocity()
        v.update({"w": torch.tensor([1.0, 0.0])}, beta=0.8)
        v._state["w"].fill_(float("nan"))
        sim = v.cosine_similarity({"w": torch.tensor([1.0, 0.0])})
        assert sim == 0.0


# ===========================================================================
# LoRA utils: iter_all_lora_params and trainable scope functions
# ===========================================================================


class TestLoraUtilsAllParams:
    """Tests for iter_all_lora_params (includes frozen)."""

    def test_iter_all_lora_params_includes_frozen(self) -> None:
        model = FakeTransformerModel(2)
        # Freeze all LoRA params
        for _, p in model.named_parameters():
            if "lora_A" in str(_) or "lora_B" in str(_):
                p.requires_grad_(False)
        params = list(iter_all_lora_params(model))
        # Should still find all LoRA params even though frozen
        # 2 layers x 2 modules x 2 params = 8
        assert len(params) == 8

    def test_set_all_lora_trainable(self) -> None:
        model = FakeTransformerModel(2)
        # Freeze everything first
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        active = set_all_lora_trainable(model)
        assert len(active) > 0
        # All LoRA params should now be trainable
        for name, p in iter_all_lora_params(model):
            assert p.requires_grad, f"{name} should be trainable"

    def test_set_trainable_lora_layers_selective(self) -> None:
        model = FakeTransformerModel(4)
        # Make only layers 2 and 3 trainable
        active = set_trainable_lora_layers(model, {2, 3})
        for name, p in iter_all_lora_params(model):
            if "layers.2." in name or "layers.3." in name:
                assert p.requires_grad, f"{name} should be trainable"
                assert name in active
            else:
                assert not p.requires_grad, f"{name} should NOT be trainable"
                assert name not in active

    def test_get_unmapped_lora_param_names_with_top_level(self) -> None:
        model = FakeTransformerModel(2)
        # Add a top-level LoRA param that won't match layers.N.
        model.top_lora_A = nn.Parameter(torch.randn(4, 4))
        model.top_lora_A.requires_grad_(True)
        unmapped = get_unmapped_lora_param_names(model)
        assert any("top_lora_A" in n for n in unmapped)

    def test_get_last_fraction_lora_raises_with_unmapped(self) -> None:
        model = FakeTransformerModel(2)
        model.top_lora_A = nn.Parameter(torch.randn(4, 4))
        model.top_lora_A.requires_grad_(True)
        with pytest.raises(ValueError, match="not mapped to decoder layers"):
            get_last_fraction_lora_layer_indices(model)

    @pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
    def test_get_last_fraction_rejects_invalid_fraction(self, fraction: float) -> None:
        model = FakeTransformerModel(4)
        with pytest.raises(ValueError, match="fraction must be in"):
            get_last_fraction_lora_layer_indices(model, fraction=fraction)

    def test_configure_trainable_lora_scope_rejects_unknown(self) -> None:
        model = FakeTransformerModel(4)
        with pytest.raises(ValueError, match="Unsupported trainable_lora_scope"):
            configure_trainable_lora_scope(model, "unknown_scope")

    def test_configure_trainable_lora_scope_last_25_percent(self) -> None:
        model = FakeTransformerModel(12)
        names, indices = configure_trainable_lora_scope(model, "last_25_percent")
        # Last 25% of 12 = last 3 layers
        assert indices == {9, 10, 11}
        assert len(names) > 0


# ===========================================================================
# RollbackManager edge cases
# ===========================================================================


class TestRollbackEdgeCases:
    """Additional rollback edge cases."""

    def test_rollback_manager_rejects_zero_max_history(self) -> None:
        with pytest.raises(ValueError, match="max_history must be positive"):
            RollbackManager(max_history=0)

    def test_rollback_manager_rejects_negative_max_history(self) -> None:
        with pytest.raises(ValueError, match="max_history must be positive"):
            RollbackManager(max_history=-1)

    def test_save_returns_incrementing_index(self) -> None:
        model = FakeLoRAModel()
        mgr = RollbackManager()
        assert mgr.save(model) == 0
        assert mgr.save(model) == 1
        assert mgr.save(model) == 2

    def test_rollback_negative_index(self) -> None:
        model = FakeLoRAModel()
        mgr = RollbackManager()
        mgr.save(model)
        model.linear.lora_A.data.fill_(5.0)
        mgr.save(model)
        mgr.rollback(model, index=-1)
        # Should rollback to the last saved state (A=5.0)
        assert torch.allclose(model.linear.lora_A.data, torch.tensor(5.0).expand_as(model.linear.lora_A.data))

    def test_pop_removes_last_entry(self) -> None:
        model = FakeLoRAModel()
        mgr = RollbackManager()
        mgr.save(model)
        model.linear.lora_A.data.fill_(99.0)
        mgr.save(model)
        assert len(mgr._history) == 2
        mgr.pop()
        assert len(mgr._history) == 1

    def test_clear_empties_history(self) -> None:
        model = FakeLoRAModel()
        mgr = RollbackManager()
        for _ in range(5):
            mgr.save(model)
        mgr.clear()
        assert mgr._history == []


# ===========================================================================
# RandomWalkController edge cases
# ===========================================================================


class TestRandomWalkEdgeCases:
    """Additional random walk controller edge cases."""

    def test_commit_proposal(self) -> None:
        from tg_lora.random_walk_controller import Proposal
        ctrl = RandomWalkController(
            K_initial=3,
            N_initial=5,
            alpha_initial=0.3,
            beta_initial=0.8,
            lr_initial=5e-4,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        proposal = Proposal(
            K=8,
            N=10,
            alpha=1.0,
            beta=0.95,
            lr=8e-4,
            active_layer_strategy="last_25_percent",
            relative_update_cap=0.01,
        )
        ctrl.commit_proposal(proposal)
        assert ctrl.state.K == 8
        assert ctrl.state.N == 10
        assert ctrl.state.alpha == 1.0
        assert ctrl.state.beta == 0.95
        assert ctrl.state.lr == 8e-4
        assert ctrl.state.active_layer_strategy == "last_25_percent"

    def test_accel_deadzone_validation_negative(self) -> None:
        with pytest.raises(ValueError, match="accel_deadzone must be finite and non-negative"):
            RandomWalkController(
                accel_deadzone=-0.1,
                k_explore_prob=0.0,
                n_explore_prob=0.0,
                beta_explore_prob=0.0,
                strategy_explore_prob=0.0,
                lr_explore_prob=0.0,
            )

    def test_accel_deadzone_validation_nan(self) -> None:
        with pytest.raises(ValueError, match="accel_deadzone must be finite and non-negative"):
            RandomWalkController(
                accel_deadzone=float("nan"),
                k_explore_prob=0.0,
                n_explore_prob=0.0,
                beta_explore_prob=0.0,
                strategy_explore_prob=0.0,
                lr_explore_prob=0.0,
            )

    def test_accel_deadzone_validation_inf(self) -> None:
        with pytest.raises(ValueError, match="accel_deadzone must be finite and non-negative"):
            RandomWalkController(
                accel_deadzone=float("inf"),
                k_explore_prob=0.0,
                n_explore_prob=0.0,
                beta_explore_prob=0.0,
                strategy_explore_prob=0.0,
                lr_explore_prob=0.0,
            )

    def test_accept_with_zero_loss(self) -> None:
        """Accept when loss_pilot is 0 — uses max(abs(pilot), 1e-8) fallback."""
        ctrl = RandomWalkController(
            rollback_tolerance=0.005,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        assert ctrl.accept(0.0, 0.0) is True
        # Tiny increase with zero pilot uses 1e-8 denominator
        assert ctrl.accept(0.0, 1e-6) is False  # (1e-6 - 0) / max(0, 1e-8) = 100 > tolerance
