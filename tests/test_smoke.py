"""Smoke tests: verify training loops run end-to-end with a tiny model."""

import tempfile
from pathlib import Path

import orjson
import pytest
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from src.data.build_seed_dataset import LoraDataset
from src.eval.eval_loss import eval_loss
from src.tg_lora.delta_tracker import compute_mean_delta
from src.tg_lora.extrapolator import apply_extrapolation
from src.tg_lora.layer_sampler import select_active_layers
from src.tg_lora.lora_state import snapshot_lora, load_lora_snapshot
from src.tg_lora.random_walk_controller import RandomWalkController
from src.tg_lora.rollback_manager import RollbackManager
from src.tg_lora.velocity import Velocity
from src.training.trainer_loop import create_optimizer, train_one_step
from src.utils.io import load_jsonl

pytestmark = pytest.mark.slow

# ── Helpers ──────────────────────────────────────────────────────────────────

MAX_SEQ = 32


def _tiny_tokenizer():
    """Tiny tokenizer whose vocab matches the tiny model."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def _tiny_gpt2_with_tokenizer():
    """Minimal GPT-2 causal LM with LoRA, vocab resized to match tokenizer."""
    from transformers import GPT2Config, GPT2LMHeadModel

    tok = _tiny_tokenizer()
    cfg = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=MAX_SEQ,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(cfg)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_cfg)
    return model, tok


def _make_jsonl(path: Path, n: int = 20):
    records = [{"text": f"hello world example number {i} " * 3} for i in range(n)]
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def _make_dataloaders(tmp: Path):
    model, tok = _tiny_gpt2_with_tokenizer()

    train_path = tmp / "train.jsonl"
    valid_path = tmp / "valid.jsonl"
    _make_jsonl(train_path, 20)
    _make_jsonl(valid_path, 8)

    train_ds = LoraDataset(load_jsonl(str(train_path)), tok, MAX_SEQ)
    valid_ds = LoraDataset(load_jsonl(str(valid_path)), tok, MAX_SEQ)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=2, shuffle=False)
    return model, tok, train_loader, valid_loader


# ── Tests ────────────────────────────────────────────────────────────────────


def test_smoke_baseline_training():
    """Baseline QLoRA: 5 steps should run without error and produce finite loss."""
    with tempfile.TemporaryDirectory() as tmp:
        model, tok, train_loader, valid_loader = _make_dataloaders(Path(tmp))
        optimizer = create_optimizer(model, lr=1e-3)

        losses = []
        for batch in train_loader:
            loss = train_one_step(model, batch, optimizer, max_grad_norm=1.0)
            losses.append(loss)
            if len(losses) >= 5:
                break

        assert len(losses) == 5
        assert all(torch.isfinite(torch.tensor(loss_val)) for loss_val in losses)

        valid_loss = eval_loss(model, valid_loader, device="cpu", max_batches=4)
        assert torch.isfinite(torch.tensor(valid_loss))


def test_smoke_tg_lora_one_cycle():
    """TG-LoRA: one full cycle (pilot → snapshot → extrapolate → accept/rollback)."""
    with tempfile.TemporaryDirectory() as tmp:
        model, tok, train_loader, valid_loader = _make_dataloaders(Path(tmp))

        controller = RandomWalkController(
            K_initial=2,
            K_candidates=[2, 3],
            N_initial=3,
            N_candidates=[1, 3],
            alpha_initial=0.3,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        velocity = Velocity()
        rollback_mgr = RollbackManager()

        batch_iter = iter(train_loader)

        # --- Pilot ---
        W0 = snapshot_lora(model)
        optimizer = create_optimizer(model, lr=1e-3)

        pilot_loss_sum = 0.0
        for _ in range(controller.state.K):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)
            pilot_loss_sum += train_one_step(model, batch, optimizer, max_grad_norm=1.0)
        pilot_loss_avg = pilot_loss_sum / controller.state.K
        assert torch.isfinite(torch.tensor(pilot_loss_avg))

        WK = snapshot_lora(model)

        # --- Delta + Velocity ---
        dW = compute_mean_delta(WK, W0, K=controller.state.K)
        velocity.update(
            dW,
            beta=controller.state.beta,
            lr=controller.state.lr,
            K=controller.state.K,
        )

        # --- Rollback save ---
        rollback_mgr.save(model)

        # --- Eval pilot ---
        loss_pilot = eval_loss(model, valid_loader, device="cpu", max_batches=4)

        # --- Extrapolate ---
        proposal = controller.propose()
        active_names, _ = select_active_layers(
            model, strategy="last_25_percent_plus_random_2"
        )

        apply_extrapolation(
            model=model,
            velocity=velocity.state,
            active_names=active_names,
            n_steps=proposal.N,
            lr=proposal.lr,
            relative_update_cap=1.0,  # high cap for tiny model
        )

        # --- Eval after ---
        loss_after = eval_loss(model, valid_loader, device="cpu", max_batches=4)

        # --- Accept / Rollback ---
        if controller.accept(loss_pilot, loss_after):
            controller.reward(loss_pilot, loss_after)
            rollback_mgr.pop()
        else:
            rollback_mgr.rollback(model)
            rollback_mgr.pop()
            controller.penalize(loss_pilot, loss_after)

        # Verify controller state is sane
        s = controller.summary()
        assert s["total_cycles"] == 1
        assert s["acceptance_rate"] >= 0.0


def test_smoke_snapshot_restore_roundtrip():
    """Snapshot → modify → restore → verify params match."""
    model, _ = _tiny_gpt2_with_tokenizer()
    snap = snapshot_lora(model)

    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            p.data += torch.randn_like(p) * 10.0

    load_lora_snapshot(model, snap)

    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            assert torch.allclose(p.cpu(), snap[name]), f"Mismatch in {name}"


def test_smoke_rollback_manager_with_model():
    """RollbackManager correctly restores model after modification."""
    model, _ = _tiny_gpt2_with_tokenizer()
    mgr = RollbackManager()

    mgr.save(model)
    original = snapshot_lora(model)

    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            p.data.zero_()

    mgr.rollback(model)

    restored = snapshot_lora(model)
    for k in original:
        assert torch.allclose(original[k], restored[k]), f"Mismatch in {k}"
