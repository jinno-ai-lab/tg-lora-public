"""Unit tests for src/training/trainer_loop.py — individual function coverage."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from src.training.trainer_loop import (
    NumericalInstabilityError,
    create_optimizer,
    create_scheduler,
    forward_backward,
    optimizer_step,
)

MAX_SEQ = 32


def _total_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def _tiny_gpt2(seed: int = 42):
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=100,
        n_positions=MAX_SEQ,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(cfg)
    lora_cfg = LoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    return get_peft_model(model, lora_cfg)


def _dummy_batch(vocab_size: int = 100, seq_len: int = MAX_SEQ, bs: int = 2):
    input_ids = torch.randint(0, vocab_size, (bs, seq_len))
    attention_mask = torch.ones(bs, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── create_optimizer ─────────────────────────────────────────────────────────


class TestCreateOptimizer:
    def test_returns_adamw_with_correct_lr(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=3e-4)
        assert isinstance(opt, torch.optim.AdamW)
        for pg in opt.param_groups:
            assert pg["lr"] == 3e-4

    def test_weight_decay_applied(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3, weight_decay=0.05)
        for pg in opt.param_groups:
            assert pg["weight_decay"] == 0.05

    def test_only_trainable_params(self):
        model = _tiny_gpt2()
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        opt = create_optimizer(model, lr=1e-3)
        param_count = sum(p.numel() for pg in opt.param_groups for p in pg["params"])
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == expected
        assert trainable_count > 0


# ── create_scheduler ─────────────────────────────────────────────────────────


class TestCreateScheduler:
    def test_produces_linear_schedule_with_warmup(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        total_steps = 100
        warmup = 10
        sched = create_scheduler(
            opt, num_training_steps=total_steps, warmup_steps=warmup
        )
        assert sched is not None

    def test_warmup_phase_lr_ramps_up(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        total = 100
        warmup = 10
        sched = create_scheduler(opt, num_training_steps=total, warmup_steps=warmup)

        lrs = []
        for _ in range(warmup):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[-1] > lrs[0], "LR should increase during warmup"

    def test_decay_phase_lr_decreases_after_warmup(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        total = 100
        warmup = 5
        sched = create_scheduler(opt, num_training_steps=total, warmup_steps=warmup)

        for _ in range(warmup):
            opt.step()
            sched.step()
        lr_at_end_of_warmup = opt.param_groups[0]["lr"]

        for _ in range(20):
            opt.step()
            sched.step()
        lr_after_decay = opt.param_groups[0]["lr"]

        assert lr_after_decay < lr_at_end_of_warmup, "LR should decrease after warmup"

    def test_zero_warmup(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(opt, num_training_steps=100, warmup_steps=0)
        lr_before = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after < lr_before, "Should decay immediately with no warmup"

    def test_cosine_scheduler_returns_lambda_lr(self):
        from torch.optim.lr_scheduler import LambdaLR

        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(opt, num_training_steps=100, schedule_type="cosine")
        assert isinstance(sched, LambdaLR)

    def test_cosine_scheduler_decays_lr(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(opt, num_training_steps=200, schedule_type="cosine")

        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(50):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])

        assert lrs[-1] < lrs[0], "Cosine schedule should decay LR over time"

    def test_cosine_scheduler_converges_to_zero(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(opt, num_training_steps=100, schedule_type="cosine")
        for _ in range(100):
            opt.step()
            sched.step()
        lr = opt.param_groups[0]["lr"]
        assert lr < 1e-4, f"LR should be near zero after full schedule, got {lr}"

    def test_cosine_with_warmup_ramps_then_decays(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        warmup = 10
        total = 100
        sched = create_scheduler(
            opt,
            num_training_steps=total,
            warmup_steps=warmup,
            schedule_type="cosine",
        )

        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(warmup):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        lr_peak = lrs[-1]

        assert lrs[-1] > lrs[0], "LR should increase during warmup"
        assert lr_peak == pytest.approx(1e-3, rel=0.01), (
            "LR should reach target after warmup"
        )

        for _ in range(30):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])

        assert lrs[-1] < lr_peak, "LR should decay after warmup"

    def test_cosine_warmup_zero_matches_no_warmup(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        _ = create_scheduler(
            opt,
            num_training_steps=100,
            warmup_steps=0,
            schedule_type="cosine",
        )
        lr = opt.param_groups[0]["lr"]
        assert lr == pytest.approx(1e-3, rel=0.01), (
            "With no warmup, LR starts at target"
        )


# ── forward_backward ─────────────────────────────────────────────────────────


class TestForwardBackward:
    def test_returns_finite_loss(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        loss = forward_backward(model, batch)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_loss_scaled_by_grad_accumulation(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        torch.manual_seed(0)
        loss_no_accum = forward_backward(model, batch, grad_accumulation=1)
        model.zero_grad()
        torch.manual_seed(0)
        loss_with_accum = forward_backward(model, batch, grad_accumulation=4)
        assert abs(loss_no_accum - loss_with_accum) < 0.1, (
            f"Returned loss should be similar regardless of accumulation: {loss_no_accum} vs {loss_with_accum}"
        )

    def test_gradients_scaled_by_accumulation(self):
        batch = _dummy_batch()

        model1 = _tiny_gpt2(seed=7)
        torch.manual_seed(0)
        forward_backward(model1, batch, grad_accumulation=1)
        norm1 = _total_grad_norm(model1)

        model2 = _tiny_gpt2(seed=7)
        torch.manual_seed(0)
        forward_backward(model2, batch, grad_accumulation=4)
        norm4 = _total_grad_norm(model2)

        ratio = norm4 / (norm1 + 1e-12)
        assert 0.2 < ratio < 0.4, f"Expected ratio ~0.25, got {ratio:.4f}"

    def test_model_in_train_mode(self):
        model = _tiny_gpt2()
        model.eval()
        batch = _dummy_batch()
        forward_backward(model, batch)
        assert model.training


# ── optimizer_step ───────────────────────────────────────────────────────────


class TestOptimizerStep:
    def _setup(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        batch = _dummy_batch()
        forward_backward(model, batch)
        return model, opt

    def test_gradient_clipping_applied(self):
        model, opt = self._setup()
        max_norm = 0.01
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], float("inf")
        )
        forward_backward(model, _dummy_batch())

        optimizer_step(opt, None, model, max_grad_norm=max_norm)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        assert total_norm <= max_norm * 1.1, (
            f"Gradients should be clipped to <= {max_norm}"
        )

    def test_optimizer_zeroed_after_step(self):
        model, opt = self._setup()
        forward_backward(model, _dummy_batch())
        optimizer_step(opt, None, model, max_grad_norm=1.0)

        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.abs().max().item() == 0.0, (
                    "Gradients should be zeroed after step"
                )

    def test_scheduler_steps(self):
        model, opt = self._setup()
        total = 50
        sched = create_scheduler(opt, num_training_steps=total, warmup_steps=5)
        lr_before = opt.param_groups[0]["lr"]
        forward_backward(model, _dummy_batch())
        optimizer_step(opt, sched, model, max_grad_norm=1.0)
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after != lr_before, "Scheduler should have changed the LR"

    def test_no_scheduler_ok(self):
        model, opt = self._setup()
        forward_backward(model, _dummy_batch())
        optimizer_step(opt, None, model, max_grad_norm=1.0)


# ── optimizer_step: non-finite gradient detection ────────────────────────────
#
# ``clip_grad_norm_`` returns the pre-clip total gradient norm every step. The
# silent-success path it gates: a finite *forward* loss (so ``forward_backward``
# raised nothing) whose *backward* produced Inf/NaN gradients. Before the fix
# the returned norm was discarded, the clip swallowed the bad grads, the run
# kept stepping on corrupted weights, and the process exited 0. The check must
# surface it as NumericalInstabilityError — the signal the fault-recovery /
# instability-exit path (train_tg_lora catch, recover.py / diagnose.py
# classification) keys on. Mirrors TASK-0200's grad_norm divergence check at
# the step seam.


def _first_trainable_grad(model):
    # Pick a param whose grad is *non-zero*: under standard LoRA init lora_B is
    # zeroed, so lora_A receives an all-zero grad (the forward output is
    # constant in lora_A while lora_B == 0) and optimizer.step() leaves it
    # unchanged. lora_B carries the real (non-zero) gradient, so poisoning it
    # both trips the non-finite check AND would move the weight if the step ran.
    for p in model.parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
            return p
    raise AssertionError("no trainable param with a non-zero grad to poison")


class TestOptimizerStepNonFiniteGrad:
    def _setup(self):
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        forward_backward(model, _dummy_batch())
        return model, opt

    def test_nan_grad_raises(self):
        """A NaN gradient must raise, not be silently clipped."""
        model, opt = self._setup()
        _first_trainable_grad(model).grad.data[0] = float("nan")
        with pytest.raises(NumericalInstabilityError, match="non-finite"):
            optimizer_step(opt, None, model, max_grad_norm=1.0)

    def test_inf_grad_raises(self):
        """An Inf gradient must raise, not stall the model into silence."""
        model, opt = self._setup()
        _first_trainable_grad(model).grad.data[0] = float("inf")
        with pytest.raises(NumericalInstabilityError, match="non-finite"):
            optimizer_step(opt, None, model, max_grad_norm=1.0)

    def test_finite_grad_does_not_raise_and_steps(self):
        """Happy path: finite grads step normally — the check must not over-fire."""
        model, opt = self._setup()
        param = _first_trainable_grad(model)
        before = param.detach().clone()
        optimizer_step(opt, None, model, max_grad_norm=1.0)
        assert not torch.equal(param.detach(), before), (
            "optimizer.step() must still apply on finite gradients"
        )

    def test_non_finite_grad_prevents_step(self):
        """The raise must precede optimizer.step(), so corrupted grads never apply."""
        model, opt = self._setup()
        param = _first_trainable_grad(model)
        param.grad.data[0] = float("nan")
        before = param.detach().clone()
        with pytest.raises(NumericalInstabilityError):
            optimizer_step(opt, None, model, max_grad_norm=1.0)
        assert torch.equal(param.detach(), before), (
            "Weights must be unchanged when a non-finite gradient aborts the step"
        )

    def test_neg_inf_grad_raises(self):
        """-Inf gradient is equally non-finite and must raise."""
        model, opt = self._setup()
        _first_trainable_grad(model).grad.data[0] = float("-inf")
        with pytest.raises(NumericalInstabilityError, match="non-finite"):
            optimizer_step(opt, None, model, max_grad_norm=1.0)


# ── NumericalInstabilityError ─────────────────────────────────────────────────


class TestNumericalInstability:
    def test_nan_loss_raises(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        # Patch compute_loss to return NaN
        import src.training.trainer_loop as tl

        original = tl.compute_loss
        nan_loss = torch.tensor(float("nan"), requires_grad=True)
        tl.compute_loss = lambda m, b: nan_loss
        try:
            with pytest.raises(NumericalInstabilityError, match="non-finite"):
                forward_backward(model, batch)
        finally:
            tl.compute_loss = original

    def test_inf_loss_raises(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        import src.training.trainer_loop as tl

        original = tl.compute_loss
        inf_loss = torch.tensor(float("inf"), requires_grad=True)
        tl.compute_loss = lambda m, b: inf_loss
        try:
            with pytest.raises(NumericalInstabilityError, match="non-finite"):
                forward_backward(model, batch)
        finally:
            tl.compute_loss = original

    def test_neg_inf_loss_raises(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        import src.training.trainer_loop as tl

        original = tl.compute_loss
        ninf_loss = torch.tensor(float("-inf"), requires_grad=True)
        tl.compute_loss = lambda m, b: ninf_loss
        try:
            with pytest.raises(NumericalInstabilityError, match="non-finite"):
                forward_backward(model, batch)
        finally:
            tl.compute_loss = original

    def test_finite_loss_passes(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        loss = forward_backward(model, batch)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))


class TestCreateOptimizerValidation:
    def test_no_trainable_params_raises(self):
        """Frozen model with no trainable params should raise ValueError."""
        model = _tiny_gpt2()
        for p in model.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="No trainable parameters"):
            create_optimizer(model, lr=1e-3)


# ── check_lora_params_finite (REQ-056) ────────────────────────────────────


class TestCheckLoraParamsFinite:
    """Tests for post-extrapolation parameter finiteness check."""

    def test_finite_params_return_true(self):
        from src.training.train_tg_lora import check_lora_params_finite

        model = _tiny_gpt2()
        is_finite, detail = check_lora_params_finite(model)
        assert is_finite is True
        assert detail == ""

    def test_nan_params_return_false(self):
        from src.training.train_tg_lora import check_lora_params_finite

        model = _tiny_gpt2()
        # Inject NaN into a LoRA parameter
        for name, param in model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                param.data[0] = float("nan")
                break
        is_finite, detail = check_lora_params_finite(model)
        assert is_finite is False
        assert "NaN" in detail

    def test_inf_params_return_false(self):
        from src.training.train_tg_lora import check_lora_params_finite

        model = _tiny_gpt2()
        # Inject Inf into a LoRA parameter
        for name, param in model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                param.data[0] = float("inf")
                break
        is_finite, detail = check_lora_params_finite(model)
        assert is_finite is False
        assert "Inf" in detail


# ── Integration: schedule_type end-to-end (config → scheduler object) ──────


class TestScheduleTypeIntegration:
    """Verify that schedule_type flows from config schema to scheduler factory."""

    def test_linear_config_produces_linear_schedule(self):
        from src.training.config_schema import TrainingConfig
        from torch.optim.lr_scheduler import LambdaLR, LRScheduler

        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=1,
            learning_rate=1e-4,
            max_steps=100,
            schedule_type="linear",
        )
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(
            opt,
            num_training_steps=cfg.max_steps,
            warmup_steps=0,
            schedule_type=cfg.schedule_type,
        )
        assert isinstance(sched, LRScheduler)
        assert isinstance(sched, LambdaLR)

    def test_cosine_config_produces_cosine_schedule(self):
        from src.training.config_schema import TrainingConfig
        from torch.optim.lr_scheduler import LambdaLR

        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=1,
            learning_rate=1e-4,
            max_steps=100,
            schedule_type="cosine",
        )
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(
            opt,
            num_training_steps=cfg.max_steps,
            warmup_steps=0,
            schedule_type=cfg.schedule_type,
        )
        assert isinstance(sched, LambdaLR)

    def test_cosine_with_warmup_integration(self):
        from src.training.config_schema import TrainingConfig

        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=1,
            learning_rate=1e-4,
            max_steps=100,
            schedule_type="cosine",
            warmup_steps=10,
        )
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(
            opt,
            num_training_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            schedule_type=cfg.schedule_type,
        )
        # Warmup: LR should increase for first 10 steps
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(cfg.warmup_steps):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert lrs[-1] > lrs[0], "LR should ramp up during warmup"
        # Decay: LR should decrease after warmup
        for _ in range(30):
            opt.step()
            sched.step()
        assert opt.param_groups[0]["lr"] < lrs[-1]

    def test_default_config_produces_linear_schedule(self):
        from src.training.config_schema import TrainingConfig
        from torch.optim.lr_scheduler import LambdaLR

        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=1,
            learning_rate=1e-4,
            max_steps=100,
        )
        assert cfg.schedule_type == "linear"
        model = _tiny_gpt2()
        opt = create_optimizer(model, lr=1e-3)
        sched = create_scheduler(
            opt,
            num_training_steps=cfg.max_steps,
            warmup_steps=0,
            schedule_type=cfg.schedule_type,
        )
        assert isinstance(sched, LambdaLR)


class TestForwardBackwardValidation:
    def test_zero_grad_accumulation_raises(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        with pytest.raises(ValueError, match="grad_accumulation must be >= 1"):
            forward_backward(model, batch, grad_accumulation=0)

    def test_negative_grad_accumulation_raises(self):
        model = _tiny_gpt2()
        batch = _dummy_batch()
        with pytest.raises(ValueError, match="grad_accumulation must be >= 1"):
            forward_backward(model, batch, grad_accumulation=-1)
