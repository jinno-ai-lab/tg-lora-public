import pytest
import torch

from src.training.optimizer_lifecycle import (
    OptimizerLifecycleManager,
    _zero_optimizer_state_in_place,
)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)


def _materialize_state(optimizer, model):
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


class TestOptimizerLifecycleManager:
    def test_recreate_policy_returns_new_optimizer(self):
        model = _TinyModel()
        mgr = OptimizerLifecycleManager(
            model,
            lr=1e-3,
            weight_decay=0.1,
            policy="recreate_per_cycle",
        )

        opt1 = mgr.prepare_for_cycle(1e-3)
        _materialize_state(opt1, model)
        opt2 = mgr.prepare_for_cycle(2e-3)

        assert opt1 is not opt2
        assert opt2.param_groups[0]["lr"] == 2e-3
        assert len(opt2.state) == 0

    def test_reuse_policy_zeros_state_in_place(self):
        model = _TinyModel()
        mgr = OptimizerLifecycleManager(
            model,
            lr=1e-3,
            weight_decay=0.0,
            policy="reuse_state_reset_experimental",
        )

        opt1 = mgr.prepare_for_cycle(1e-3)
        _materialize_state(opt1, model)

        pointers = {}
        saw_non_zero_state = False
        for param, state in opt1.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    pointers[(id(param), key)] = value.data_ptr()
                    saw_non_zero_state = saw_non_zero_state or bool(
                        torch.count_nonzero(value).item()
                    )

        opt2 = mgr.prepare_for_cycle(2e-3)

        assert saw_non_zero_state
        assert opt1 is opt2
        assert opt2.param_groups[0]["lr"] == 2e-3
        for param, state in opt2.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    assert value.data_ptr() == pointers[(id(param), key)]
                    assert torch.count_nonzero(value).item() == 0

    def test_persistent_policy_preserves_state_in_place(self):
        model = _TinyModel()
        mgr = OptimizerLifecycleManager(
            model,
            lr=1e-3,
            weight_decay=0.0,
            policy="persistent",
        )

        opt1 = mgr.prepare_for_cycle(1e-3)
        _materialize_state(opt1, model)

        pointers = {}
        snapshots = {}
        for param, state in opt1.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    pointers[(id(param), key)] = value.data_ptr()
                    snapshots[(id(param), key)] = value.clone()

        opt2 = mgr.prepare_for_cycle(2e-3)

        assert opt1 is opt2
        assert opt2.param_groups[0]["lr"] == 2e-3
        for param, state in opt2.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state_key = (id(param), key)
                    assert value.data_ptr() == pointers[state_key]
                    assert torch.allclose(value, snapshots[state_key])

    def test_policy_property(self):
        model = _TinyModel()
        for policy in ("reuse_state_reset_experimental", "persistent"):
            mgr = OptimizerLifecycleManager(model, lr=1e-3, policy=policy)
            assert mgr.policy == policy

    def test_state_tensor_pointers_returns_empty_when_no_optimizer(self):
        model = _TinyModel()
        mgr = OptimizerLifecycleManager(
            model, lr=1e-3, policy="reuse_state_reset_experimental"
        )
        # Before first prepare_for_cycle, no optimizer exists
        assert mgr.state_tensor_pointers() == {}

    def test_state_tensor_pointers_returns_data_after_materialize(self):
        model = _TinyModel()
        mgr = OptimizerLifecycleManager(
            model, lr=1e-3, policy="reuse_state_reset_experimental"
        )
        opt = mgr.prepare_for_cycle(1e-3)
        _materialize_state(opt, model)

        ptrs = mgr.state_tensor_pointers()
        assert len(ptrs) > 0
        # All values should be valid data pointers
        for ptr in ptrs.values():
            assert isinstance(ptr, int)
            assert ptr != 0


class TestOptimizerLifecycleManagerValidation:
    def test_rejects_zero_lr(self):
        model = _TinyModel()
        with pytest.raises(ValueError, match="lr must be positive"):
            OptimizerLifecycleManager(model, lr=0.0)

    def test_rejects_negative_lr(self):
        model = _TinyModel()
        with pytest.raises(ValueError, match="lr must be positive"):
            OptimizerLifecycleManager(model, lr=-0.001)

    def test_rejects_negative_weight_decay(self):
        model = _TinyModel()
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            OptimizerLifecycleManager(model, lr=1e-3, weight_decay=-0.01)

    def test_accepts_zero_weight_decay(self):
        model = _TinyModel()
        mgr = OptimizerLifecycleManager(model, lr=1e-3, weight_decay=0.0)
        assert mgr._weight_decay == 0.0

    def test_rejects_none_model(self):
        with pytest.raises(ValueError, match="model must not be None"):
            OptimizerLifecycleManager(None, lr=1e-3)

    def test_rejects_invalid_policy(self):
        model = _TinyModel()
        with pytest.raises(ValueError, match="policy must be one of"):
            OptimizerLifecycleManager(model, lr=1e-3, policy="nonexistent_policy")

    def test_accepts_valid_policies(self):
        model = _TinyModel()
        for policy in (
            "recreate_per_cycle",
            "reuse_state_reset_experimental",
            "persistent",
        ):
            mgr = OptimizerLifecycleManager(model, lr=1e-3, policy=policy)
            assert mgr.policy == policy


class TestZeroOptimizerStateInPlace:
    def test_zeros_tensor_state(self):
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _materialize_state(opt, model)

        _zero_optimizer_state_in_place(opt)
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    assert torch.count_nonzero(value).item() == 0

    def test_zeros_bool_int_float_state(self):
        """_zero_optimizer_state_in_place handles bool, int, float state values."""
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _materialize_state(opt, model)

        # Manually inject non-tensor state types
        param = list(model.parameters())[0]
        opt.state[param]["flag_bool"] = True
        opt.state[param]["step_int"] = 42
        opt.state[param]["rate_float"] = 3.14

        _zero_optimizer_state_in_place(opt)

        assert opt.state[param]["flag_bool"] is False
        assert opt.state[param]["step_int"] == 0
        assert opt.state[param]["rate_float"] == 0.0
