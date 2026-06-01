import torch
import pytest

from tg_lora.lora_utils import iter_lora_params
from tg_lora.rollback_manager import RollbackManager

from conftest import FakeLoRAModel


def test_rollback_basic():
    model = FakeLoRAModel()
    mgr = RollbackManager()

    idx = mgr.save(model)
    original = {name: p.clone() for name, p in iter_lora_params(model)}

    for name, p in iter_lora_params(model):
        p.data += torch.ones_like(p) * 5.0

    mgr.rollback(model, idx)

    for name, p in iter_lora_params(model):
        assert torch.allclose(p.cpu(), original[name])


def test_rollback_last():
    model = FakeLoRAModel()
    mgr = RollbackManager()

    mgr.save(model)  # state 0
    original = {name: p.clone() for name, p in iter_lora_params(model)}

    for name, p in iter_lora_params(model):
        p.data += torch.ones_like(p)
    mgr.save(model)  # state 1

    for name, p in iter_lora_params(model):
        p.data += torch.ones_like(p) * 10

    mgr.rollback(model)  # rollback to last saved (state 1)

    for name, p in iter_lora_params(model):
        expected = original[name] + 1.0
        assert torch.allclose(p.cpu(), expected, atol=1e-6)


def test_pop_and_clear():
    mgr = RollbackManager()
    assert mgr._history == []
    mgr.pop()  # should not error on empty
    mgr.clear()


def test_rollback_empty_history_raises():
    model = FakeLoRAModel()
    mgr = RollbackManager()
    with pytest.raises(RuntimeError, match="No saved states"):
        mgr.rollback(model)


def test_rollback_out_of_range_raises():
    model = FakeLoRAModel()
    mgr = RollbackManager()
    mgr.save(model)
    mgr.save(model)
    with pytest.raises(IndexError, match="out of range"):
        mgr.rollback(model, index=5)


def test_rollback_negative_out_of_range_raises():
    model = FakeLoRAModel()
    mgr = RollbackManager()
    mgr.save(model)
    with pytest.raises(IndexError, match="out of range"):
        mgr.rollback(model, index=-2)


def test_save_sanitize_nan():
    """NaN in model params must be sanitized to zero in the snapshot."""
    model = FakeLoRAModel()
    model.linear.lora_A.data.fill_(float("nan"))
    mgr = RollbackManager()
    mgr.save(model)

    # After save, snapshot should have zeros instead of NaN
    for name, tensor in mgr._history[0].items():
        assert torch.isfinite(tensor).all(), f"{name} still has non-finite values"
        if "lora_A" in name:
            assert torch.allclose(tensor, torch.zeros_like(tensor))


def test_save_sanitize_inf():
    """Inf values must be clamped to finite bounds in the snapshot."""
    model = FakeLoRAModel()
    model.linear.lora_A.data.fill_(float("inf"))
    model.linear.lora_B.data.fill_(float("-inf"))
    mgr = RollbackManager()
    mgr.save(model)

    for name, tensor in mgr._history[0].items():
        assert torch.isfinite(tensor).all(), f"{name} still has non-finite values"


def test_rollback_restores_sanitized_state():
    """After save sanitizes NaN, rollback should restore finite params."""
    model = FakeLoRAModel()
    mgr = RollbackManager()

    # Corrupt the model, then save — snapshot will be sanitized
    model.linear.lora_A.data.fill_(float("nan"))
    mgr.save(model)

    # Modify further
    model.linear.lora_A.data.fill_(42.0)

    # Rollback should restore sanitized (zeroed) state, not NaN
    mgr.rollback(model, 0)
    assert torch.isfinite(model.linear.lora_A.data).all()


def test_max_history_bounds():
    """History must not exceed max_history entries."""
    model = FakeLoRAModel()
    mgr = RollbackManager(max_history=3)
    for i in range(10):
        mgr.save(model)
    assert len(mgr._history) == 3


def test_max_history_fifo_eviction():
    """Oldest entries are evicted when history exceeds max_history."""
    model = FakeLoRAModel()
    mgr = RollbackManager(max_history=2)

    model.linear.lora_A.data.fill_(1.0)
    mgr.save(model)  # idx 0: A=1

    model.linear.lora_A.data.fill_(2.0)
    mgr.save(model)  # idx 0: A=2 (evicted first entry)

    model.linear.lora_A.data.fill_(3.0)
    mgr.save(model)  # idx 0: A=2, idx 1: A=3

    # Only the last two snapshots should survive
    mgr.rollback(model, 0)
    assert torch.allclose(
        model.linear.lora_A.data, torch.tensor(2.0).expand_as(model.linear.lora_A.data)
    )
