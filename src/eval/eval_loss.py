import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.tg_lora.activation_cache import forward_from_hidden_states
from src.training.loss import has_supervised_tokens


class EvalLossResult:
    """Detailed evaluation loss metrics."""

    __slots__ = ("avg_loss", "perplexity", "num_batches", "min_loss", "max_loss")

    def __init__(
        self,
        avg_loss: float,
        num_batches: int,
        min_loss: float,
        max_loss: float,
    ) -> None:
        if not isinstance(num_batches, int) or num_batches < 0:
            raise ValueError(
                f"num_batches must be a non-negative integer, got {num_batches!r}"
            )
        if num_batches > 0:
            if not math.isfinite(avg_loss):
                raise ValueError(
                    f"avg_loss must be a finite number when num_batches > 0, got {avg_loss!r}"
                )
            if not math.isfinite(min_loss):
                raise ValueError(
                    f"min_loss must be a finite number when num_batches > 0, got {min_loss!r}"
                )
            if not math.isfinite(max_loss):
                raise ValueError(
                    f"max_loss must be a finite number when num_batches > 0, got {max_loss!r}"
                )
            if min_loss > max_loss:
                raise ValueError(
                    f"min_loss must not exceed max_loss: {min_loss} > {max_loss}"
                )

        self.avg_loss = avg_loss
        self.num_batches = num_batches
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.perplexity = (
            math.exp(avg_loss)
            if math.isfinite(avg_loss) and avg_loss < 100
            else float("inf")
        )

    def __repr__(self) -> str:
        return (
            f"EvalLossResult(avg_loss={self.avg_loss:.4f}, "
            f"ppl={self.perplexity:.2f}, "
            f"batches={self.num_batches}, "
            f"min={self.min_loss:.4f}, max={self.max_loss:.4f})"
        )


def _infer_batch_size(batch: dict[str, torch.Tensor]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 1


def _truncate_batch(batch: dict[str, torch.Tensor], limit: int) -> dict[str, torch.Tensor]:
    batch_size = _infer_batch_size(batch)
    if limit >= batch_size:
        return batch
    truncated: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            truncated[key] = value[:limit]
        else:
            truncated[key] = value
    return truncated


def _compute_batch_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "hidden_states" in batch:
        return forward_from_hidden_states(
            model,
            batch["hidden_states"],
            batch["attention_mask"],
            batch["labels"],
            split_layer_idx=batch["split_layer_idx"],
            position_ids=batch.get("position_ids"),
        )

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss


@torch.no_grad()
def eval_loss(
    model,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    max_examples: int | None = None,
) -> float:
    if max_batches is not None and max_examples is not None:
        raise ValueError("Specify at most one of max_batches or max_examples")
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        count = 0
        total_examples = 0

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if max_examples is not None:
                remaining = max_examples - total_examples
                if remaining <= 0:
                    break
                batch = _truncate_batch(batch, remaining)

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if not has_supervised_tokens(batch):
                continue

            batch_examples = _infer_batch_size(batch)
            total_loss += _compute_batch_loss(model, batch).item() * batch_examples
            count += 1
            total_examples += batch_examples

            if max_batches and count >= max_batches:
                break
    finally:
        if was_training:
            model.train()

    if count == 0 or total_examples == 0:
        return float("nan")
    return total_loss / total_examples


@torch.no_grad()
def eval_loss_detailed(
    model,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    max_examples: int | None = None,
) -> EvalLossResult:
    """Evaluate model and return detailed loss statistics including perplexity."""
    if max_batches is not None and max_examples is not None:
        raise ValueError("Specify at most one of max_batches or max_examples")
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    batch_losses: list[float] = []
    total_weighted_loss = 0.0
    total_examples = 0
    try:
        count = 0
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if max_examples is not None:
                remaining = max_examples - total_examples
                if remaining <= 0:
                    break
                batch = _truncate_batch(batch, remaining)

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if not has_supervised_tokens(batch):
                continue

            batch_loss = _compute_batch_loss(model, batch).item()
            batch_examples = _infer_batch_size(batch)
            batch_losses.append(batch_loss)
            total_weighted_loss += batch_loss * batch_examples
            total_examples += batch_examples
            count += 1

            if max_batches and count >= max_batches:
                break
    finally:
        if was_training:
            model.train()

    if not batch_losses:
        return EvalLossResult(
            avg_loss=float("nan"),
            num_batches=0,
            min_loss=float("nan"),
            max_loss=float("nan"),
        )

    avg = total_weighted_loss / total_examples
    return EvalLossResult(
        avg_loss=avg,
        num_batches=len(batch_losses),
        min_loss=min(batch_losses),
        max_loss=max(batch_losses),
    )
