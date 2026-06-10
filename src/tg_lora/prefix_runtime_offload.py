from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.tg_lora.activation_cache import _get_decoder_layers


def _find_optional_module(model: nn.Module, candidate_paths: list[str]) -> nn.Module | None:
    for path in candidate_paths:
        obj: Any = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.Module):
                return obj
        except AttributeError:
            continue
    return None


def _get_input_embeddings(model: nn.Module) -> nn.Module | None:
    candidates = [
        "base_model.model.model.embed_tokens",
        "model.model.embed_tokens",
        "model.embed_tokens",
        "embed_tokens",
        "base_model.model.transformer.wte",
        "model.transformer.wte",
        "transformer.wte",
    ]
    return _find_optional_module(model, candidates)


def offload_prefix_runtime_to_cpu(
    model: nn.Module,
    *,
    split_layer_idx: int,
    offload_input_embeddings: bool = True,
) -> dict[str, int | bool]:
    decoder_layers = _get_decoder_layers(model)
    if split_layer_idx < 1 or split_layer_idx > len(decoder_layers):
        raise ValueError(
            f"split_layer_idx must be within [1, {len(decoder_layers)}], got {split_layer_idx}"
        )

    modules_to_offload: list[nn.Module] = []
    input_embeddings = _get_input_embeddings(model) if offload_input_embeddings else None
    if input_embeddings is not None:
        modules_to_offload.append(input_embeddings)

    modules_to_offload.extend(decoder_layers[:split_layer_idx])

    offloaded_parameters = 0
    seen_modules: set[int] = set()
    for module in modules_to_offload:
        module_id = id(module)
        if module_id in seen_modules:
            continue
        seen_modules.add(module_id)
        offloaded_parameters += sum(
            parameter.numel() for parameter in module.parameters(recurse=True)
        )
        module.to("cpu")

    if torch.cuda.is_available():
        from src.utils.device import gpu_empty_cache, detect_device
        gpu_empty_cache(detect_device())

    return {
        "offloaded_prefix_modules": len(seen_modules),
        "offloaded_prefix_parameters": offloaded_parameters,
        "offloaded_prefix_input_embeddings": input_embeddings is not None,
        "split_layer_idx": split_layer_idx,
    }