import logging
import random
from typing import Literal

import torch

from src.model.lora_utils import iter_lora_params, iter_lora_params_by_layer

logger = logging.getLogger("tg-lora")


StrategyName = Literal[
    "last_25_percent",
    "last_25_percent_plus_random_2",
    "middle_random",
    "lisa_like_weighted",
]


def get_num_layers(model: torch.nn.Module) -> int:
    layer_map = iter_lora_params_by_layer(model)
    return len(layer_map)


def get_layer_indices(model: torch.nn.Module) -> list[int]:
    return sorted(iter_lora_params_by_layer(model).keys())


def select_active_layers(
    model: torch.nn.Module,
    strategy: StrategyName,
    random_middle: int = 2,
    layer_scores: dict[int, float] | None = None,
    temperature: float = 1.0,
) -> tuple[set[str], set[int]]:
    """Select active layers for extrapolation.

    Strategies:
        last_25_percent: Focus on output layers (best for fine-tuning).
        last_25_percent_plus_random_2: Output layers + random middle (balanced, default).
        middle_random: Random subset from all layers (exploratory).
        lisa_like_weighted: Score-based weighted sampling (requires layer_scores).
    """
    if strategy == "lisa_like_weighted":
        return _lisa_weighted(model, layer_scores or {}, temperature)

    all_names = {name for name, _ in iter_lora_params(model)}
    layer_map = iter_lora_params_by_layer(model)
    sorted_indices = sorted(layer_map.keys())
    num_layers = len(sorted_indices)

    if num_layers == 0:
        logger.warning("No LoRA layers found — returning all parameter names")
        return all_names, set()

    if num_layers == 1:
        return all_names, {sorted_indices[0]}

    target_layers = set()

    if strategy == "last_25_percent":
        start = num_layers - max(1, num_layers // 4)
        target_layers = set(sorted_indices[start:])

    elif strategy == "last_25_percent_plus_random_2":
        start = num_layers - max(1, num_layers // 4)
        target_layers = set(sorted_indices[start:])
        mid_start = num_layers // 4
        mid_end = start
        if mid_end > mid_start:
            extra = random.sample(
                sorted_indices[mid_start:mid_end],
                min(random_middle, mid_end - mid_start),
            )
            target_layers.update(extra)

    elif strategy == "middle_random":
        n = max(2, num_layers // 3)
        target_layers = set(random.sample(sorted_indices, min(n, num_layers)))

    active = set()
    for idx in target_layers:
        if idx in layer_map:
            for name, _ in layer_map[idx]:
                active.add(name)

    return (active if active else all_names, target_layers)


def _lisa_weighted(
    model: torch.nn.Module,
    layer_scores: dict[int, float],
    temperature: float,
) -> tuple[set[str], set[int]]:
    import torch.nn.functional as F

    layer_map = iter_lora_params_by_layer(model)
    indices = sorted(layer_map.keys())
    num_layers = len(indices)

    if num_layers == 0 or not layer_scores:
        names, idxs = select_active_layers(model, "last_25_percent")
        return names, idxs
    scores = torch.tensor(
        [layer_scores.get(i, 0.0) for i in indices],
        dtype=torch.float32,
    )
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    probs = F.softmax(scores / max(temperature, 0.01), dim=0)
    n_select = max(1, num_layers // 4)
    chosen_idx = torch.multinomial(
        probs, min(n_select, len(indices)), replacement=False
    )

    active = set()
    chosen_indices = set()
    for i in chosen_idx.tolist():
        idx = indices[i]
        chosen_indices.add(idx)
        if idx in layer_map:
            for name, _ in layer_map[idx]:
                active.add(name)

    return (
        active if active else {name for name, _ in iter_lora_params(model)},
        chosen_indices,
    )
