"""Tests for LAWA (LAtest-Window Weight Averaging) baseline.

GOAL §3.3: PSA must beat LAWA to have value.
"""

import pytest
import torch
import torch.nn as nn

from src.tg_lora.weight_averaging import (
    LAWAAverager,
    averaged_weights_context,
    evaluate_with_lawa,
)


def _make_lora_model():
    """Create a minimal model with LoRA-like named parameters."""

    class LoRALinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Parameter(torch.randn(4, 2))
            self.lora_B = nn.Parameter(torch.randn(2, 4))

        def forward(self, x):
            return x

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([LoRALinear(), LoRALinear()])

        def forward(self, x):
            return x

    model = FakeModel()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# ---------------------------------------------------------------------------
# LAWAAverager — record / is_ready / count
# ---------------------------------------------------------------------------


class TestLAWARecord:
    def test_single_record_not_ready(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        assert avg.count == 1
        assert not avg.is_ready

    def test_two_records_ready(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        avg.record(model, cycle=1)
        assert avg.count == 2
        assert avg.is_ready

    def test_start_cycle_gate(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=10, start_cycle=5)
        for c in range(5):
            avg.record(model, cycle=c)
        assert avg.count == 5
        assert not avg.is_ready  # cycle < start_cycle
        avg.record(model, cycle=5)
        assert avg.is_ready

    def test_window_size_limits_buffer(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=2)
        avg.record(model, cycle=0)
        avg.record(model, cycle=1)
        avg.record(model, cycle=2)
        assert avg.count == 2  # oldest evicted


# ---------------------------------------------------------------------------
# LAWAAverager — average_snapshot
# ---------------------------------------------------------------------------


class TestLAWAAverageSnapshot:
    def test_average_of_identical_weights(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=5)
        avg.record(model, cycle=0)
        avg.record(model, cycle=1)
        snapshot = avg.average_snapshot()
        assert snapshot is not None
        # Averaging identical tensors gives the same tensor
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert torch.allclose(snapshot[name], p.detach().cpu())

    def test_average_of_different_weights(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=5)

        # Record first snapshot
        avg.record(model, cycle=0)

        # Change weights and record second snapshot
        for p in model.parameters():
            p.data += 1.0
        avg.record(model, cycle=1)

        snapshot = avg.average_snapshot()
        assert snapshot is not None

        # Average should be between the two snapshots
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                avg_val = snapshot[name]
                diff = (p.detach().cpu() - avg_val).abs().max()
                assert diff.item() > 0  # not equal to final
                assert diff.item() < 1.0  # not far from final

    def test_average_returns_none_when_insufficient(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=5)
        avg.record(model, cycle=0)
        assert avg.average_snapshot() is None

    def test_average_keys_match_model_params(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=5)
        avg.record(model, cycle=0)
        avg.record(model, cycle=1)
        snapshot = avg.average_snapshot()
        assert snapshot is not None
        lora_names = {
            n for n, p in model.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        }
        assert set(snapshot.keys()) == lora_names


# ---------------------------------------------------------------------------
# LAWAAverager — latest
# ---------------------------------------------------------------------------


class TestLAWALatest:
    def test_latest_returns_last_snapshot(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=5)
        avg.record(model, cycle=0)
        for p in model.parameters():
            p.data.fill_(42.0)
        avg.record(model, cycle=1)

        latest = avg.latest()
        assert latest is not None
        # latest should have the fill_(42) values
        first_key = next(iter(latest))
        assert (latest[first_key] == 42.0).all()

    def test_latest_returns_none_on_empty(self):
        avg = LAWAAverager()
        assert avg.latest() is None


# ---------------------------------------------------------------------------
# LAWAAverager — reset
# ---------------------------------------------------------------------------


class TestLAWAReset:
    def test_reset_clears_buffer(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=5)
        avg.record(model, cycle=0)
        avg.record(model, cycle=1)
        assert avg.count == 2

        avg.reset()
        assert avg.count == 0
        assert not avg.is_ready
        assert avg.latest() is None


# ---------------------------------------------------------------------------
# evaluate_with_lawa
# ---------------------------------------------------------------------------


class TestEvaluateWithLAWA:
    def test_returns_none_when_not_ready(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)

        eval_losses = [1.5]

        def eval_fn(m):
            return eval_losses[0]

        lawa_loss, current_loss = evaluate_with_lawa(model, avg, eval_fn)
        assert lawa_loss is None
        assert current_loss == 1.5

    def test_returns_both_losses_when_ready(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        avg.record(model, cycle=1)

        call_count = [0]

        def eval_fn(m):
            call_count[0] += 1
            return float(call_count[0])

        lawa_loss, current_loss = evaluate_with_lawa(model, avg, eval_fn)
        assert current_loss == 1.0  # first call
        assert lawa_loss == 2.0  # second call

    def test_restores_pre_eval_weights_after_eval(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        for p in model.parameters():
            p.data += 5.0
        avg.record(model, cycle=1)

        # Snapshot weights at call time (original + 5)
        pre_eval = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }

        def eval_fn(m):
            return 1.0

        evaluate_with_lawa(model, avg, eval_fn)

        # Weights should be restored to pre-eval state (not snapshot 1)
        for n, p in model.named_parameters():
            assert torch.allclose(p.detach().cpu(), pre_eval[n].cpu()), (
                f"Weight {n} not restored after evaluate_with_lawa"
            )

    def test_cycle_param_in_record(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3, start_cycle=10)
        # record without explicit cycle — should use internal counter
        avg.record(model)
        assert avg.count == 1


class TestEvaluateWithLAWAPrecomputedLoss:
    def test_precomputed_current_loss_skips_eval_fn(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        avg.record(model, cycle=1)

        eval_calls = [0]

        def eval_fn(m):
            eval_calls[0] += 1
            return float(eval_calls[0])

        # With precomputed loss, eval_fn should only be called once (for LAWA)
        lawa_loss, current_loss = evaluate_with_lawa(
            model, avg, eval_fn, current_loss=0.42,
        )
        assert current_loss == 0.42  # precomputed value used
        assert lawa_loss == 1.0  # single eval_fn call
        assert eval_calls[0] == 1  # not called for current weights

    def test_precomputed_loss_not_ready_returns_none(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)

        def eval_fn(m):
            return 1.0

        lawa_loss, current_loss = evaluate_with_lawa(
            model, avg, eval_fn, current_loss=0.5,
        )
        assert lawa_loss is None
        assert current_loss == 0.5  # precomputed passed through
        # eval_fn should never have been called
        # (we can't directly verify but the precomputed value is returned)


# ---------------------------------------------------------------------------
# averaged_weights_context
# ---------------------------------------------------------------------------


class TestAveragedWeightsContext:
    def test_noop_when_averager_is_none(self):
        """plain/PSA conditions pass None — context must be a pure no-op."""
        model = _make_lora_model()
        pre = {n: p.detach().clone() for n, p in model.named_parameters()}
        with averaged_weights_context(model, None) as active:
            assert active is False
        for n, p in model.named_parameters():
            assert torch.allclose(p.detach().cpu(), pre[n].cpu())

    def test_noop_when_averager_not_ready(self):
        """Before start_cycle / <2 snapshots, fall back to current weights."""
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3, start_cycle=5)
        avg.record(model, cycle=0)  # 1 snapshot, cycle<start -> not ready
        pre = {n: p.detach().clone() for n, p in model.named_parameters()}
        with averaged_weights_context(model, avg) as active:
            assert active is False
        for n, p in model.named_parameters():
            assert torch.allclose(p.detach().cpu(), pre[n].cpu())

    def test_swaps_in_averaged_weights_under_no_grad(self):
        """Regression: the LoRA params are requires_grad=True leaves, so the
        in-place copy_ must run under torch.no_grad(). Without it this raises
        'a leaf Variable that requires grad is being used in an in-place
        operation'. The swap must also load the averaged values."""
        model = _make_lora_model()
        for p in model.parameters():
            assert p.requires_grad  # precondition: leaf, requires grad
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        for p in model.parameters():
            p.data += 1.0
        avg.record(model, cycle=1)  # window = {base, base+1} -> avg = base+0.5
        snapshot = avg.average_snapshot()

        with averaged_weights_context(model, avg) as active:
            assert active is True
            for n, p in model.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    assert torch.allclose(p.detach().cpu(), snapshot[n])

    def test_restores_weights_on_clean_exit(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        for p in model.parameters():
            p.data += 1.0
        avg.record(model, cycle=1)
        pre = {n: p.detach().clone() for n, p in model.named_parameters()}

        with averaged_weights_context(model, avg):
            pass  # weights are averaged inside; must restore on exit

        for n, p in model.named_parameters():
            assert torch.allclose(p.detach().cpu(), pre[n].cpu()), (
                f"{n} not restored after context exit"
            )

    def test_restores_weights_on_exception(self):
        model = _make_lora_model()
        avg = LAWAAverager(window_size=3)
        avg.record(model, cycle=0)
        for p in model.parameters():
            p.data += 1.0
        avg.record(model, cycle=1)
        pre = {n: p.detach().clone() for n, p in model.named_parameters()}

        with pytest.raises(RuntimeError, match="boom"):
            with averaged_weights_context(model, avg):
                raise RuntimeError("boom")

        for n, p in model.named_parameters():
            assert torch.allclose(p.detach().cpu(), pre[n].cpu()), (
                f"{n} not restored after exception in context"
            )
