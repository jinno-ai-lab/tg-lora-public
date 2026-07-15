import math
import re
from collections.abc import Iterator
from typing import Literal

import torch

TrainableLoraScope = Literal["all", "last_25_percent"]


def iter_lora_params(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            yield name, p


def iter_all_lora_params(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield name, p


def iter_lora_params_by_layer(
    model: torch.nn.Module,
) -> dict[int, list[tuple[str, torch.nn.Parameter]]]:
    layer_map: dict[int, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, p in iter_lora_params(model):
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            idx = int(m.group(1))
            layer_map.setdefault(idx, []).append((name, p))
    return layer_map


def iter_all_lora_params_by_layer(
    model: torch.nn.Module,
) -> dict[int, list[tuple[str, torch.nn.Parameter]]]:
    layer_map: dict[int, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, p in iter_all_lora_params(model):
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            idx = int(m.group(1))
            layer_map.setdefault(idx, []).append((name, p))
    return layer_map


def get_unmapped_lora_param_names(model: torch.nn.Module) -> list[str]:
    mapped_names = {
        name
        for params in iter_all_lora_params_by_layer(model).values()
        for name, _ in params
    }
    return sorted(
        name for name, _ in iter_all_lora_params(model) if name not in mapped_names
    )


def set_all_lora_trainable(model: torch.nn.Module) -> set[str]:
    active_names: set[str] = set()
    for name, param in iter_all_lora_params(model):
        param.requires_grad = True
        active_names.add(name)
    return active_names


def get_last_fraction_lora_layer_indices(
    model: torch.nn.Module,
    fraction: float = 0.25,
) -> set[int]:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    unmapped = get_unmapped_lora_param_names(model)
    if unmapped:
        sample = ", ".join(unmapped[:4])
        raise ValueError(
            "Cannot apply layer-scoped LoRA selection because some adapter params "
            f"are not mapped to decoder layers: {sample}"
        )

    layer_indices = sorted(iter_all_lora_params_by_layer(model).keys())
    if not layer_indices:
        raise ValueError("No LoRA decoder layers found")

    target_count = max(1, math.ceil(len(layer_indices) * fraction))
    return set(layer_indices[-target_count:])


def set_trainable_lora_layers(
    model: torch.nn.Module,
    trainable_layer_indices: set[int],
) -> set[str]:
    """Enable gradients only for LoRA params in the selected decoder layers."""
    active_names: set[str] = set()
    layer_map = iter_all_lora_params_by_layer(model)
    for layer_idx, params in layer_map.items():
        is_trainable = layer_idx in trainable_layer_indices
        for name, param in params:
            param.requires_grad = is_trainable
            if is_trainable:
                active_names.add(name)
    return active_names


def configure_trainable_lora_scope(
    model: torch.nn.Module,
    scope: TrainableLoraScope,
) -> tuple[set[str], set[int]]:
    if scope == "all":
        active_names = set_all_lora_trainable(model)
        active_indices = set(iter_all_lora_params_by_layer(model).keys())
        return active_names, active_indices

    if scope == "last_25_percent":
        active_indices = get_last_fraction_lora_layer_indices(model, fraction=0.25)
        active_names = set_trainable_lora_layers(model, active_indices)
        return active_names, active_indices

    raise ValueError(f"Unsupported trainable_lora_scope: {scope}")


def count_lora_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for _, p in iter_lora_params(model))


def geometric_rank_schedule(n_layers: int, base_rank: int) -> tuple[int, ...]:
    """Per-layer LoRA rank rising geometrically toward the output side.

    The single canonical source of the heterogeneous (per-layer asymmetric)
    rank schedule: a geometric progression ``base_rank ** (i / (n - 1))`` over
    ``n_layers`` entries, so the output-most layer carries ``base_rank``
    (keeping the total adapter budget comparable to a uniform-``base_rank``
    stack) and earlier layers carry progressively less capacity. This realizes
    GOAL §1.5/§8 non-uniform per-layer cost as per-layer adapter *capacity* —
    the specialization signal a uniform-rank stack lacks.

    Constitution Rule #3 (single source of truth): this formula is consumed by
    THREE call sites — the proxy CI harness
    (:func:`scripts.run_freeze_validloss_ci.heterogeneous_ranks`), the
    order-sensitivity diagnosis, and the real-9B target-scale CI harness
    (:func:`scripts.run_freeze_validloss_ci_9b.heterogeneous_ranks_9b`) — so a
    silent drift here would make the proxy apparatus verdict and the 9B target
    verdict test *different* architectures. Both harness wrappers delegate to
    this function; ``tests/test_heterogeneous_rank_single_source.py`` pins the
    delegation and the cross-harness byte-equality.
    """
    if n_layers <= 0:
        return ()
    if n_layers == 1:
        return (base_rank,)
    return tuple(
        max(1, int(round(base_rank ** (i / (n_layers - 1)))))
        for i in range(n_layers)
    )
