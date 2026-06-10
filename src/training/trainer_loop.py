from __future__ import annotations

import logging
from typing import Literal

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.training.loss import compute_loss

logger = logging.getLogger("tg-lora")

ScheduleType = Literal["linear", "cosine"]


class NumericalInstabilityError(RuntimeError):
    """Raised when loss becomes NaN or Inf during training."""


def create_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float = 0.0
) -> AdamW:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            "No trainable parameters found. Ensure LoRA adapters are applied "
            "and at least one layer has requires_grad=True."
        )
    return AdamW(trainable, lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    schedule_type: ScheduleType = "linear",
) -> LRScheduler:
    if schedule_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def forward_backward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    grad_accumulation: int = 1,
) -> float:
    if grad_accumulation < 1:
        raise ValueError(
            f"grad_accumulation must be >= 1, got {grad_accumulation}"
        )
    model.train()
    loss = compute_loss(model, batch)
    loss_val = loss.item()
    if not torch.isfinite(loss):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float("inf"),
        ).item()
        raise NumericalInstabilityError(
            f"Loss is {loss_val} (non-finite), grad_norm={grad_norm:.4f}. "
            f"Check learning rate, batch size, and data quality."
        )
    loss = loss / grad_accumulation
    loss.backward()
    return loss_val


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    model: torch.nn.Module,
    max_grad_norm: float = 1.0,
) -> None:
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_grad_norm,
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()


def train_one_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: AdamW,
    scheduler: LRScheduler | None = None,
    max_grad_norm: float = 1.0,
    grad_accumulation: int = 1,
) -> float:
    loss = forward_backward(model, batch, grad_accumulation)
    optimizer_step(optimizer, scheduler, model, max_grad_norm)
    return loss
