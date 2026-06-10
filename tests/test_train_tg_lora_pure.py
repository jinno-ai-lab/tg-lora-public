"""Unit tests for pure functions extracted from train_tg_lora.py."""

import math

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch.func import functional_call
from transformers import GPT2Config, GPT2LMHeadModel

from src.model.lora_utils import iter_lora_params
from src.tg_lora.activation_cache import (
    forward_from_hidden_states,
    forward_suffix_hidden_states,
)
from src.tg_lora.extrapolator import (
    alpha_line_lora_context,
    alpha_line_loss_cached_first_order,
    alpha_line_loss_cached_zeroth,
    alpha_line_loss_exact,
    compute_alpha_line_base_out_jvp,
)
from src.tg_lora.lora_state import snapshot_lora
from src.training.loss import compute_loss
from src.training.train_tg_lora import (
    _alpha_line_functional_loss,
    _apply_alpha_direction_from_base,
    _compute_pilot_average,
    _decide_accept_rollback,
    _decide_post_extrapolation_eval_policy,
    _evaluate_full_eval_outcome,
    _format_cycle_progress,
    build_training_summary,
    check_lora_params_finite,
    should_run_full_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lora_model():
    """Create a minimal GPT-2 model with LoRA adapters for finite-check tests."""
    cfg = GPT2Config(
        vocab_size=100,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=2,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(cfg)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    return get_peft_model(model, lora_cfg)


def _make_tiny_lm_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _make_tiny_cached_lm_batch(model) -> dict[str, torch.Tensor]:
    batch = _make_tiny_lm_batch()
    transformer = model.base_model.model.transformer
    input_ids = batch["input_ids"]
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
        hidden = transformer.drop(hidden)
    return {
        "hidden_states": hidden.detach(),
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
        "split_layer_idx": torch.tensor(0),
        "position_ids": position_ids,
    }


class _CachedSuffixLossWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        split_layer_idx: int | torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return forward_from_hidden_states(
            self.model,
            hidden_states,
            attention_mask,
            labels,
            split_layer_idx=split_layer_idx,
            position_ids=position_ids,
        )


# ===========================================================================
# should_run_full_eval
# ===========================================================================


class TestShouldRunFullEval:
    def test_multiple_of_interval(self):
        assert should_run_full_eval(10, 5) is True

    def test_zero_cycle(self):
        assert should_run_full_eval(0, 5) is False

    def test_not_multiple(self):
        assert should_run_full_eval(7, 5) is False

    def test_zero_interval_disables(self):
        assert should_run_full_eval(10, 0) is False

    def test_negative_interval_disables(self):
        assert should_run_full_eval(10, -1) is False

    def test_cycle_equals_interval(self):
        assert should_run_full_eval(5, 5) is True


# ===========================================================================
# check_lora_params_finite
# ===========================================================================


class TestCheckLoraParamsFinite:
    def test_all_finite(self):
        model = _make_lora_model()
        ok, detail = check_lora_params_finite(model)
        assert ok is True
        assert detail == ""

    def test_nan_detected(self):
        model = _make_lora_model()
        for name, param in model.named_parameters():
            if "lora" in name:
                param.data.fill_(float("nan"))
                break
        ok, detail = check_lora_params_finite(model)
        assert ok is False
        assert "NaN" in detail

    def test_inf_detected(self):
        model = _make_lora_model()
        for name, param in model.named_parameters():
            if "lora" in name:
                param.data.fill_(float("inf"))
                break
        ok, detail = check_lora_params_finite(model)
        assert ok is False
        assert "Inf" in detail

    def test_reports_first_nonfinite(self):
        model = _make_lora_model()
        lora_names = [n for n, p in model.named_parameters() if "lora" in n]
        assert len(lora_names) >= 2
        for name, param in model.named_parameters():
            if name == lora_names[0]:
                param.data.fill_(float("nan"))
                break
        ok, detail = check_lora_params_finite(model)
        assert ok is False
        assert lora_names[0] in detail


class TestAlphaLineFunctionalPath:
    def test_functional_loss_matches_in_place_alpha_direction(self):
        torch.manual_seed(0)
        model = _make_lora_model()
        model.eval()
        batch = _make_tiny_lm_batch()
        base = snapshot_lora(model)
        active_names = set(base)
        direction = {
            name: torch.full_like(tensor, 0.001) for name, tensor in base.items()
        }
        alpha = torch.tensor(0.2, requires_grad=True)

        functional_loss = _alpha_line_functional_loss(
            model,
            batch,
            base,
            direction,
            alpha,
            active_names=active_names,
        )
        grad = torch.autograd.grad(functional_loss, alpha)[0]

        _apply_alpha_direction_from_base(
            model,
            base,
            direction,
            float(alpha.item()),
            active_names=active_names,
        )
        with torch.no_grad():
            direct_loss = compute_loss(model, batch)

        assert torch.allclose(functional_loss.detach(), direct_loss.detach(), atol=1e-6)
        assert torch.isfinite(grad)
        for _, param in iter_lora_params(model):
            assert param.grad is None

    def test_functional_loss_matches_cached_suffix_in_place_alpha_direction(self):
        torch.manual_seed(0)
        model = _make_lora_model()
        model.eval()
        batch = _make_tiny_cached_lm_batch(model)
        base = snapshot_lora(model)
        active_names = set(base)
        direction = {
            name: torch.full_like(tensor, 0.001) for name, tensor in base.items()
        }
        alpha = torch.tensor(0.2, requires_grad=True)

        functional_loss = _alpha_line_functional_loss(
            model,
            batch,
            base,
            direction,
            alpha,
            active_names=active_names,
        )
        grad = torch.autograd.grad(functional_loss, alpha)[0]

        _apply_alpha_direction_from_base(
            model,
            base,
            direction,
            float(alpha.item()),
            active_names=active_names,
        )
        with torch.no_grad():
            direct_loss = compute_loss(model, batch)

        assert torch.allclose(functional_loss.detach(), direct_loss.detach(), atol=1e-6)
        assert torch.isfinite(grad)
        for _, param in iter_lora_params(model):
            assert param.grad is None

    def test_cached_bracketed_loss_matches_functional_call_reference(self):
        torch.manual_seed(0)
        model = _make_lora_model()
        model.eval()
        batch = _make_tiny_cached_lm_batch(model)
        base = snapshot_lora(model)
        active_names = set(base)
        direction = {
            name: torch.full_like(tensor, 0.001) for name, tensor in base.items()
        }
        alpha = torch.tensor(0.2, requires_grad=True)

        bracketed_loss = _alpha_line_functional_loss(
            model,
            batch,
            base,
            direction,
            alpha,
            active_names=active_names,
        )

        updates = {
            f"model.{name}": tensor.to(dtype=next(model.parameters()).dtype)
            + alpha.to(dtype=next(model.parameters()).dtype)
            * direction[name].to(dtype=next(model.parameters()).dtype)
            for name, tensor in base.items()
        }
        reference_loss = functional_call(
            _CachedSuffixLossWrapper(model),
            updates,
            args=(),
            kwargs={
                "hidden_states": batch["hidden_states"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "split_layer_idx": batch["split_layer_idx"],
                "position_ids": batch["position_ids"],
            },
        )

        assert torch.allclose(
            bracketed_loss.detach(),
            reference_loss.detach(),
            atol=1e-6,
        )

    def test_cached_first_order_jvp_matches_finite_difference_hidden(self):
        torch.manual_seed(0)
        model = _make_lora_model()
        model.eval()
        batch = _make_tiny_cached_lm_batch(model)
        base = snapshot_lora(model)
        active_names = set(base)
        direction = {
            name: torch.randn_like(tensor) * 0.01 for name, tensor in base.items()
        }
        eps = 1e-3

        cache = compute_alpha_line_base_out_jvp(
            model,
            batch,
            base,
            direction,
            active_names=active_names,
        )
        with alpha_line_lora_context(
            model,
            base,
            direction,
            torch.tensor(eps),
            active_names=active_names,
        ):
            hidden_eps = forward_suffix_hidden_states(
                model,
                batch["hidden_states"],
                batch["attention_mask"],
                split_layer_idx=batch["split_layer_idx"],
                position_ids=batch["position_ids"],
            )
        finite_diff = (hidden_eps - cache.hidden_base) / eps

        assert torch.allclose(cache.hidden_jvp, finite_diff, atol=2e-3, rtol=5e-2)

    def test_cached_first_order_loss_is_closer_than_zeroth(self):
        torch.manual_seed(1)
        model = _make_lora_model()
        model.eval()
        batch = _make_tiny_cached_lm_batch(model)
        base = snapshot_lora(model)
        active_names = set(base)
        direction = {
            name: torch.randn_like(tensor) * 0.02 for name, tensor in base.items()
        }
        alpha = torch.tensor(0.2)

        cache = compute_alpha_line_base_out_jvp(
            model,
            batch,
            base,
            direction,
            active_names=active_names,
        )
        exact = alpha_line_loss_exact(
            model,
            batch,
            base,
            direction,
            alpha,
            active_names=active_names,
        )
        first = alpha_line_loss_cached_first_order(model, cache, batch, alpha)
        zeroth = alpha_line_loss_cached_zeroth(model, cache, batch)

        first_err = abs(float((first - exact).detach().item()))
        zeroth_err = abs(float((zeroth - exact).detach().item()))
        assert first_err < zeroth_err


# ===========================================================================
# _compute_pilot_average
# ===========================================================================


class TestComputePilotAverage:
    def test_basic_average(self):
        avg, m = _compute_pilot_average([1.0, 2.0, 3.0], K=4)
        assert avg == pytest.approx(2.0)
        assert m["K"] == 4
        assert m["count"] == 3
        assert m["avg_loss"] == pytest.approx(2.0)
        assert m["min_loss"] == 1.0
        assert m["max_loss"] == 3.0

    def test_empty_list_returns_nan(self):
        avg, m = _compute_pilot_average([], K=3)
        assert math.isnan(avg)
        assert m["count"] == 0

    def test_single_element(self):
        avg, m = _compute_pilot_average([5.5], K=2)
        assert avg == pytest.approx(5.5)
        assert m["min_loss"] == m["max_loss"] == 5.5

    def test_all_nan_returns_nan_with_zero_finite_count(self):
        avg, m = _compute_pilot_average([float("nan"), float("nan")], K=3)
        assert math.isnan(avg)
        assert m["finite_count"] == 0
        assert m["total_count"] == 2

    def test_all_inf_returns_nan_with_zero_finite_count(self):
        avg, m = _compute_pilot_average([float("inf"), float("-inf")], K=3)
        assert math.isnan(avg)
        assert m["finite_count"] == 0
        assert m["total_count"] == 2

    def test_mixed_nan_and_finite_filters_correctly(self):
        avg, m = _compute_pilot_average(
            [1.0, float("nan"), 3.0, float("inf"), 5.0], K=4
        )
        assert avg == pytest.approx(3.0)
        assert m["finite_count"] == 3
        assert m["count"] == 5
        assert m["min_loss"] == 1.0
        assert m["max_loss"] == 5.0

    def test_mixed_inf_and_finite_filters_correctly(self):
        avg, m = _compute_pilot_average([float("inf"), 2.0, float("-inf")], K=2)
        assert avg == pytest.approx(2.0)
        assert m["finite_count"] == 1
        assert m["min_loss"] == m["max_loss"] == 2.0


# ===========================================================================
# _decide_accept_rollback
# ===========================================================================


class TestDecideAcceptRollback:
    def test_improvement(self):
        accepted, reason = _decide_accept_rollback(1.0, 0.8, 0.1)
        assert accepted is True
        assert reason == "improvement"

    def test_exact_equal_is_improvement(self):
        accepted, reason = _decide_accept_rollback(1.0, 1.0, 0.1)
        assert accepted is True
        assert reason == "improvement"

    def test_within_tolerance(self):
        accepted, reason = _decide_accept_rollback(1.0, 1.05, 0.1)
        assert accepted is True
        assert reason == "within_tolerance"

    def test_exceeds_tolerance(self):
        accepted, reason = _decide_accept_rollback(1.0, 1.2, 0.1)
        assert accepted is False
        assert "degradation" in reason

    def test_zero_pilot_loss_denominator_guard(self):
        accepted, reason = _decide_accept_rollback(0.0, 1e-10, 0.1)
        # relative = 1e-10 / max(0.0, 1e-8) = 1e-10 / 1e-8 = 0.01 <= 0.1
        assert accepted is True

    def test_negative_pilot_loss(self):
        accepted, reason = _decide_accept_rollback(-1.0, -0.9, 0.1)
        # loss_after > loss_pilot → not improvement; relative = 0.1/1.0 = 0.1 <= 0.1
        assert accepted is True

    def test_nan_loss_after_rejected(self):
        accepted, _ = _decide_accept_rollback(1.0, float("nan"), 0.1)
        assert accepted is False

    def test_inf_loss_after_rejected(self):
        accepted, _ = _decide_accept_rollback(1.0, float("inf"), 0.1)
        assert accepted is False

    def test_nan_loss_pilot_rejected(self):
        accepted, _ = _decide_accept_rollback(float("nan"), 0.5, 0.1)
        assert accepted is False


# ===========================================================================
# _decide_post_extrapolation_eval_policy
# ===========================================================================


class TestDecidePostExtrapolationEvalPolicy:
    def _policy(self, **overrides):
        kwargs = {
            "consistency": 0.9,
            "selected_N": 5,
            "total_cycles": 5,
            "acceptance_rate": 1.0,
            "velocity_anomalous": False,
            "enabled": True,
            "high_cos": 0.85,
            "mid_cos": 0.70,
            "mid_eval_every": 3,
            "min_cycles": 1,
            "min_acceptance_rate": 0.8,
            "force_eval_N": 20,
        }
        kwargs.update(overrides)
        return _decide_post_extrapolation_eval_policy(**kwargs)

    def test_high_confidence_skips_eval(self):
        should_eval, reason = self._policy(consistency=0.9, selected_N=10)
        assert should_eval is False
        assert reason == "high_confidence"

    def test_force_eval_large_n(self):
        should_eval, reason = self._policy(consistency=0.95, selected_N=20)
        assert should_eval is True
        assert reason == "force_eval_N:20"

    def test_mid_confidence_periodic_eval(self):
        should_eval, reason = self._policy(
            consistency=0.75,
            total_cycles=6,
            selected_N=5,
        )
        assert should_eval is True
        assert reason == "mid_periodic:3"

    def test_mid_confidence_skips_between_periodic_checks(self):
        should_eval, reason = self._policy(
            consistency=0.75,
            total_cycles=5,
            selected_N=5,
        )
        assert should_eval is False
        assert reason == "mid_skip:3"

    def test_low_confidence_evaluates(self):
        should_eval, reason = self._policy(consistency=0.6, selected_N=1)
        assert should_eval is True
        assert reason == "low_confidence"

    def test_warmup_evaluates(self):
        should_eval, reason = self._policy(total_cycles=0, min_cycles=1)
        assert should_eval is True
        assert reason == "warmup"


# ===========================================================================
# _evaluate_full_eval_outcome
# ===========================================================================


class TestEvaluateFullEvalOutcome:
    def test_new_best(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            0.5,
            prev_best=1.0,
            stale_cycles=3,
            patience=5,
            min_cycles=10,
            current_cycle=12,
        )
        assert is_best is True
        assert stop is False
        assert "new_best" in reason

    def test_no_improvement_increments_stale(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            1.5,
            prev_best=1.0,
            stale_cycles=3,
            patience=5,
            min_cycles=10,
            current_cycle=12,
        )
        assert is_best is False
        assert stop is False
        assert "stale=4" in reason

    def test_early_stop_when_patience_exceeded(self):
        is_best, stop, reason = _evaluate_full_eval_outcome(
            1.5,
            prev_best=1.0,
            stale_cycles=4,
            patience=5,
            min_cycles=10,
            current_cycle=15,
        )
        assert is_best is False
        assert stop is True
        assert "early_stop" in reason

    def test_no_stop_before_min_cycles(self):
        _, stop, _ = _evaluate_full_eval_outcome(
            1.5,
            prev_best=1.0,
            stale_cycles=10,
            patience=5,
            min_cycles=20,
            current_cycle=10,
        )
        assert stop is False

    def test_none_patience_disables_early_stop(self):
        _, stop, _ = _evaluate_full_eval_outcome(
            1.5,
            prev_best=1.0,
            stale_cycles=100,
            patience=None,
            min_cycles=0,
            current_cycle=200,
        )
        assert stop is False


# ===========================================================================
# _format_cycle_progress
# ===========================================================================


class TestFormatCycleProgress:
    def test_accepted(self):
        s = _format_cycle_progress(5, 0.42, True, 0.95, 0.10, 8, 3)
        assert "c=5" in s
        assert "Y" in s
        assert "K=8" in s
        assert "N=3" in s

    def test_rejected(self):
        s = _format_cycle_progress(3, 1.23, False, 0.5, -0.05, 4, 2)
        assert "N" in s


# ===========================================================================
# build_training_summary
# ===========================================================================


class TestBuildTrainingSummary:
    def test_merges_all_sources(self):
        class FakeController:
            def summary(self):
                return {"acceptance_rate": 0.75, "total_cycles": 100, "K": 8}

        class FakeCycleState:
            def summary(self):
                return {"acceptance_rate": 0.80, "cycles": 100, "best_loss": 0.5}

        class FakeDeltaTracker:
            def summary(self):
                return {"delta_count": 50}

        result = build_training_summary(
            FakeController(),
            FakeCycleState(),
            FakeDeltaTracker(),
        )
        assert result["controller_acceptance_rate"] == 0.75
        assert result["controller_total_cycles"] == 100
        # cycle_state overwrites acceptance_rate
        assert result["acceptance_rate"] == 0.80
        assert result["delta_count"] == 50

    def test_preserves_controller_keys_on_missing(self):
        class FakeController:
            def summary(self):
                return {"K": 4}

        class FakeCycleState:
            def summary(self):
                return {"cycles": 10}

        class FakeDeltaTracker:
            def summary(self):
                return {}

        result = build_training_summary(
            FakeController(),
            FakeCycleState(),
            FakeDeltaTracker(),
        )
        assert result["controller_acceptance_rate"] == 0.0
        assert result["controller_total_cycles"] == 0
