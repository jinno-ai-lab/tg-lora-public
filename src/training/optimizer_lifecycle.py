from __future__ import annotations

from typing import Literal

import torch
from torch.optim import Optimizer

from src.training.trainer_loop import create_optimizer

OptimizerLifecyclePolicy = Literal[
    "recreate_per_cycle",
    "reuse_state_reset_experimental",
    "persistent",
]


def _set_optimizer_hparams(
    optimizer: Optimizer,
    *,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer.defaults["lr"] = lr
    optimizer.defaults["weight_decay"] = weight_decay
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = weight_decay


def _zero_optimizer_state_in_place(optimizer: Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                value.zero_()
            elif isinstance(value, bool):
                state[key] = False
            elif isinstance(value, int):
                state[key] = 0
            elif isinstance(value, float):
                state[key] = 0.0


class OptimizerLifecycleManager:
    """Manage the per-cycle AdamW lifecycle for TG-LoRA.

    The default policy recreates the optimizer each cycle.

    ``reuse_state_reset_experimental`` keeps the same AdamW instance alive and
    zeroes its state tensors in-place between cycles so a fresh-optimizer
    approximation can be tested without reallocating exp_avg / exp_avg_sq.

    ``persistent`` keeps the same AdamW instance and its state across cycles,
    matching baseline AdamW lifecycle as closely as the TG-LoRA cycle structure
    allows.
    """

    _VALID_POLICIES: frozenset[str] = frozenset(
        ("recreate_per_cycle", "reuse_state_reset_experimental", "persistent")
    )

    def __init__(
        self,
        model,
        *,
        lr: float,
        weight_decay: float = 0.0,
        policy: OptimizerLifecyclePolicy = "recreate_per_cycle",
    ) -> None:
        if model is None:
            raise ValueError("model must not be None")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if policy not in self._VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(self._VALID_POLICIES)}, got {policy!r}"
            )
        self._model = model
        self._initial_lr = lr
        self._weight_decay = weight_decay
        self._policy = policy
        self._optimizer: Optimizer | None = None

    @property
    def policy(self) -> OptimizerLifecyclePolicy:
        return self._policy

    def prepare_for_cycle(self, lr: float) -> Optimizer:
        if self._policy == "recreate_per_cycle":
            return create_optimizer(
                self._model,
                lr=lr,
                weight_decay=self._weight_decay,
            )

        optimizer = self._optimizer
        if optimizer is None:
            optimizer = create_optimizer(
                self._model,
                lr=self._initial_lr,
                weight_decay=self._weight_decay,
            )
            self._optimizer = optimizer

        _set_optimizer_hparams(
            optimizer,
            lr=lr,
            weight_decay=self._weight_decay,
        )
        if self._policy == "reuse_state_reset_experimental":
            _zero_optimizer_state_in_place(optimizer)
        optimizer.zero_grad(set_to_none=True)
        return optimizer

    def state_tensor_pointers(self) -> dict[tuple[int, str], int]:
        if self._optimizer is None:
            return {}
        return {
            (id(param), key): value.data_ptr()
            for param, state in self._optimizer.state.items()
            for key, value in state.items()
            if torch.is_tensor(value)
        }
