import logging

import torch

from src.tg_lora.activation_cache import forward_from_hidden_states

logger = logging.getLogger("tg-lora")

_REQUIRED_KEYS = frozenset({"input_ids", "attention_mask", "labels"})
_CACHED_REQUIRED_KEYS = frozenset(
    {"hidden_states", "attention_mask", "labels", "split_layer_idx"}
)


def has_supervised_tokens(batch: dict[str, torch.Tensor]) -> bool:
    labels = batch.get("labels")
    if not isinstance(labels, torch.Tensor):
        raise KeyError("Batch must contain a tensor 'labels' entry")
    return bool(torch.any(labels != -100).item())


def compute_loss(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    if "hidden_states" in batch:
        missing = _CACHED_REQUIRED_KEYS - batch.keys()
        if missing:
            raise KeyError(
                f"Cached batch is missing required keys: {sorted(missing)}. "
                f"Got: {sorted(batch.keys())}"
            )
        return forward_from_hidden_states(
            model,
            batch["hidden_states"],
            batch["attention_mask"],
            batch["labels"],
            split_layer_idx=batch["split_layer_idx"],
            position_ids=batch.get("position_ids"),
        )

    missing = _REQUIRED_KEYS - batch.keys()
    if missing:
        raise KeyError(
            f"Batch is missing required keys: {sorted(missing)}. "
            f"Got: {sorted(batch.keys())}"
        )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss

