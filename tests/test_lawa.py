"""Tests for LAWA (LAtest-Window Weight Averaging) baseline."""

import pytest
import torch
import torch.nn as nn

from src.tg_lora.weight_averaging import LAWAAverager, evaluate_with_lawa


def _make_model(n_layers=2, hidden=16, rank=2):
    """Create a minimal model with LoRA-like parameters."""
    layers = []
    for i in range(n_layers):
        a = nn.Parameter(torch.randn(rank, hidden) * 0.01)
        b = nn.Parameter(torch.randn(hidden, rank) * 0.01)
        layers.append((f"layers.{i}.self_attn.lora_A.default.weight", a))
        layers.append((f"layers.{i}.self_attn.lora_B.default.weight", b))

    class FakeModel(nn.Module):
        pass

    model = FakeModel()
    for name, param in layers:
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            if not hasattr(obj, p):
                setattr(obj, p, nn.Module())
            obj = getattr(obj, p)
        setattr(obj, parts[-1], param)
        param.requires_grad_(True)
    return model


class TestLAWAAverager:
    def test_record_increments_count(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=3)
        assert avgr.count == 0
        avgr.record(model)
        assert avgr.count == 1
        avgr.record(model)
        assert avgr.count == 2

    def test_ring_buffer_respects_window_size(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=3)
        for _ in range(5):
            avgr.record(model)
        assert avgr.count == 3

    def test_not_ready_below_start_cycle(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=3, start_cycle=10)
        for c in range(5):
            avgr.record(model, cycle=c)
        assert not avgr.is_ready

    def test_ready_after_start_cycle_and_min_snapshots(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=3, start_cycle=2)
        for c in range(5):
            avgr.record(model, cycle=c)
        assert avgr.is_ready

    def test_not_ready_with_fewer_than_two_snapshots(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=5, start_cycle=0)
        avgr.record(model, cycle=0)
        assert not avgr.is_ready

    def test_average_snapshot_none_when_insufficient(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=5)
        avgr.record(model)
        assert avgr.average_snapshot() is None

    def test_average_snapshot_is_arithmetic_mean(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=3, start_cycle=0)

        # Record two distinct states
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(1.0)
        avgr.record(model, cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(3.0)
        avgr.record(model, cycle=1)

        avg = avgr.average_snapshot()
        assert avg is not None
        for name, tensor in avg.items():
            assert torch.allclose(tensor, torch.ones_like(tensor) * 2.0)

    def test_average_uses_latest_window(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=2, start_cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(10.0)
        avgr.record(model, cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(20.0)
        avgr.record(model, cycle=1)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(30.0)
        avgr.record(model, cycle=2)

        avg = avgr.average_snapshot()
        assert avg is not None
        for name, tensor in avg.items():
            assert torch.allclose(tensor, torch.ones_like(tensor) * 25.0)

    def test_latest_returns_most_recent(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=5, start_cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(5.0)
        avgr.record(model, cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(7.0)
        avgr.record(model, cycle=1)

        latest = avgr.latest()
        assert latest is not None
        for name, tensor in latest.items():
            assert torch.allclose(tensor, torch.ones_like(tensor) * 7.0)

    def test_reset_clears_buffer(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=5)
        avgr.record(model)
        avgr.record(model)
        avgr.reset()
        assert avgr.count == 0


class TestEvaluateWithLAWA:
    def test_returns_none_when_not_ready(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=5, start_cycle=0)
        avgr.record(model)  # only 1 snapshot, not ready

        results = []
        def fake_eval(m):
            loss = sum(p.sum().item() for p in m.parameters())
            results.append(loss)
            return loss

        lawa_loss, current_loss = evaluate_with_lawa(model, avgr, fake_eval)
        assert lawa_loss is None
        assert len(results) == 1  # only current eval

    def test_evaluates_both_and_restores(self):
        model = _make_model()
        avgr = LAWAAverager(window_size=3, start_cycle=0)

        # Record two different states
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(1.0)
        avgr.record(model, cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(2.0)
        avgr.record(model, cycle=1)

        # Set to a third state (the "current" for evaluation)
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(3.0)

        original_state = {
            name: p.clone() for name, p in model.named_parameters()
        }

        eval_losses = []
        def fake_eval(m):
            total = sum(p.sum().item() for p in m.parameters())
            eval_losses.append(total)
            return total

        lawa_loss, current_loss = evaluate_with_lawa(model, avgr, fake_eval)

        # Should have evaluated twice: current first, then LAWA
        assert len(eval_losses) == 2
        assert lawa_loss is not None

        # Weights should be restored to original
        for name, p in model.named_parameters():
            assert torch.allclose(p, original_state[name])

    def test_lawa_loss_is_averaged(self):
        """LAWA eval should use averaged weights (mean of 1.0 and 3.0 = 2.0)."""
        model = _make_model()
        avgr = LAWAAverager(window_size=3, start_cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(1.0)
        avgr.record(model, cycle=0)

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(3.0)
        avgr.record(model, cycle=1)

        # Set current to 10 (so we can distinguish)
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.data.fill_(10.0)

        eval_weights = []
        def fake_eval(m):
            # Record the parameter values during eval
            total = 0.0
            for name, p in m.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    total += p.sum().item()
            eval_weights.append(total)
            return total

        lawa_loss, current_loss = evaluate_with_lawa(model, avgr, fake_eval)

        # First eval is current (weight=10), second is LAWA avg (weight=2)
        assert current_loss == eval_weights[0]
        assert lawa_loss == eval_weights[1]
