"""Activation cache for layer-skip evaluation in TG-LoRA.

When extrapolation modifies only a subset of layers (e.g., last 25%),
post-extrapolation eval can skip recomputing unchanged layers by
caching their output hidden states from the pilot eval pass.

This reduces eval FLOPs proportionally to the fraction of unchanged layers.
For a 36-layer model with last 9 layers active, this saves ~75% of eval cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger("tg-lora")


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


def _has_supervised_tokens(batch: dict[str, torch.Tensor]) -> bool:
    labels = batch.get("labels")
    if not isinstance(labels, torch.Tensor):
        raise KeyError("Batch must contain a tensor 'labels' entry")
    return bool(torch.any(labels != -100).item())


@dataclass
class CachedBatch:
    """Stored intermediate state for one eval batch."""

    hidden_states: torch.Tensor  # (batch, seq_len, hidden_dim) on CPU
    attention_mask: torch.Tensor  # original attention_mask
    position_ids: torch.Tensor | None
    labels: torch.Tensor  # for loss computation


class ActivationCache:
    """Caches pre-split-layer hidden states during eval for fast re-evaluation.

    Usage:
        cache = ActivationCache()

        # During pilot eval (full forward):
        pilot_loss = cache.eval_and_cache(model, dataloader, device, split_layer, max_batches)

        # After extrapolation (partial forward from cache):
        extrap_loss = cache.eval_from_cache(model, device)

        # Cleanup:
        cache.clear()
    """

    def __init__(self) -> None:
        self._batches: list[CachedBatch] = []
        self._split_layer_idx: int = 0
        self._valid: bool = False
        self._equivalence_checked: bool = False

    @property
    def is_valid(self) -> bool:
        return self._valid and len(self._batches) > 0

    @property
    def split_layer(self) -> int:
        return self._split_layer_idx

    @property
    def num_batches(self) -> int:
        return len(self._batches)

    def clear(self) -> None:
        self._batches.clear()
        self._valid = False
        self._equivalence_checked = False

    def invalidate(self) -> None:
        """Mark cache as invalid without freeing memory (allows re-cache)."""
        self._valid = False

    @torch.no_grad()
    def eval_and_cache(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device | str,
        split_layer_idx: int,
        max_batches: int | None = None,
        max_examples: int | None = None,
    ) -> float:
        """Run full eval while caching hidden states at split boundary.

        Args:
            model: The PEFT-wrapped causal LM model.
            dataloader: Validation dataloader.
            device: Compute device.
            split_layer_idx: Layer index where cache boundary is placed.
                Hidden states *entering* this layer are cached.
            max_batches: Max batches to evaluate.
            max_examples: Exact maximum number of examples to evaluate.

        Returns:
            Average loss (same as normal eval_loss).
        """
        self.clear()
        self._split_layer_idx = split_layer_idx

        try:
            decoder_layers = _get_decoder_layers(model)
        except AttributeError:
            logger.debug("Cannot find decoder layers, falling back to uncached eval")
            self._valid = False
            return _full_eval_no_cache(
                model,
                dataloader,
                device,
                max_batches=max_batches,
                max_examples=max_examples,
            )

        if split_layer_idx >= len(decoder_layers):
            logger.warning("split_layer_idx=%d >= num_layers=%d, falling back to full eval without caching", split_layer_idx, len(decoder_layers))
            self._valid = False
            return _full_eval_no_cache(
                model,
                dataloader,
                device,
                max_batches=max_batches,
                max_examples=max_examples,
            )

        if max_batches is not None and max_examples is not None:
            raise ValueError("Specify at most one of max_batches or max_examples")

        # Hook to capture hidden_states entering the split layer
        captured: list[torch.Tensor] = []

        def _hook_fn(module, args, kwargs):
            del module
            if args:
                captured.append(args[0].detach().cpu())
            elif "hidden_states" in kwargs:
                captured.append(kwargs["hidden_states"].detach().cpu())

        hook = decoder_layers[split_layer_idx].register_forward_pre_hook(
            _hook_fn, with_kwargs=True
        )

        was_training = model.training
        model.eval()
        total_loss = 0.0
        count = 0
        total_examples = 0

        try:
            for batch in dataloader:
                if max_examples is not None:
                    remaining = max_examples - total_examples
                    if remaining <= 0:
                        break
                    batch = _truncate_batch(batch, remaining)

                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if not _has_supervised_tokens(batch):
                    continue
                captured.clear()

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                batch_examples = _infer_batch_size(batch)
                total_loss += outputs.loss.item() * batch_examples
                count += 1
                total_examples += batch_examples

                # Startup equivalence check: verify cached forward matches
                # full forward on the first cached batch.
                if (
                    count == 1
                    and captured
                    and not self._equivalence_checked
                ):
                    self._equivalence_checked = True
                    cached_loss = forward_from_hidden_states(
                        model,
                        captured[0],
                        batch["attention_mask"],
                        batch["labels"],
                        split_layer_idx=split_layer_idx,
                        device=device,
                        position_ids=batch.get("position_ids", None),
                        decoder_layers=decoder_layers,
                        final_norm=_get_final_norm(model),
                        lm_head=_get_lm_head(model),
                        rotary_emb=_get_rotary_emb(model),
                        layer_types=_get_layer_types(model),
                    ).item()
                    full_loss = outputs.loss.item()
                    if abs(cached_loss - full_loss) > 0.01:
                        logger.warning(
                            "Cache equivalence check FAILED: full_forward=%.6f "
                            "cached_forward=%.6f diff=%.6f — disabling cache",
                            full_loss, cached_loss, abs(cached_loss - full_loss),
                        )
                        self._valid = False
                        hook.remove()
                        return total_loss / total_examples

                if captured:
                    self._batches.append(
                        CachedBatch(
                            hidden_states=captured[0],
                            attention_mask=batch["attention_mask"].cpu(),
                            position_ids=batch.get("position_ids", None),
                            labels=batch["labels"].cpu(),
                        )
                    )

                if max_batches and count >= max_batches:
                    break
        finally:
            hook.remove()
            if was_training:
                model.train()

        self._valid = len(self._batches) == count and count > 0
        if not self._valid:
            logger.warning("Cache incomplete: %d cached vs %d batches", len(self._batches), count)

        return total_loss / total_examples if total_examples > 0 else float("nan")

    @torch.no_grad()
    def eval_from_cache(
        self,
        model: nn.Module,
        device: torch.device | str,
    ) -> float:
        """Evaluate from cached hidden states (partial forward).

        Only runs decoder layers from split_layer onward + final norm + lm_head.
        Saves computation for all layers before split_layer.

        Handles Qwen3.5-style models that require ``position_embeddings``
        (rotary cos/sin) and per-layer-type attention masks.

        Returns:
            Average loss computed from cached states.
        """
        if not self.is_valid:
            raise RuntimeError("Cache is not valid. Call eval_and_cache() first.")

        decoder_layers = _get_decoder_layers(model)
        final_norm = _get_final_norm(model)
        lm_head = _get_lm_head(model)

        # Try to get rotary embedding module for Qwen3.5 / Llama-style models
        rotary_emb = _get_rotary_emb(model)
        layer_types = _get_layer_types(model)

        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_examples = 0

        try:
            for cached in self._batches:
                batch_size = cached.hidden_states.shape[0]
                loss = forward_from_hidden_states(
                    model,
                    cached.hidden_states,
                    cached.attention_mask,
                    cached.labels,
                    split_layer_idx=self._split_layer_idx,
                    device=device,
                    position_ids=cached.position_ids,
                    decoder_layers=decoder_layers,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    rotary_emb=rotary_emb,
                    layer_types=layer_types,
                )
                total_loss += loss.item() * batch_size
                total_examples += batch_size
        finally:
            if was_training:
                model.train()

        return total_loss / total_examples if total_examples > 0 else float("nan")


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Extract decoder layer list from a PEFT-wrapped causal LM."""
    candidates = [
        "base_model.model.model.layers",  # PeftModel(CausalLM)
        "model.model.layers",  # CausalLM without PEFT
        "base_model.model.transformer.h",  # GPT-style
        "model.layers",  # Some architectures
        "layers",  # Direct layer list (test models)
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList):
                logger.debug("Found decoder layers at path: %s", path)
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        f"Cannot find decoder layers in model. Tried {len(candidates)} paths: {candidates}"
    )


def _get_final_norm(model: nn.Module) -> nn.Module:
    """Extract the final layer norm."""
    candidates = [
        "base_model.model.model.norm",
        "model.model.norm",
        "base_model.model.transformer.ln_f",
        "model.norm",
        "norm",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Cannot find final norm layer")


def _get_lm_head(model: nn.Module) -> nn.Module:
    """Extract the language model head."""
    candidates = [
        "base_model.model.lm_head",
        "model.lm_head",
        "lm_head",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Cannot find lm_head")


def _get_rotary_emb(model: nn.Module) -> nn.Module | None:
    """Extract rotary embedding module (Qwen3.5, Llama, etc.)."""
    candidates = [
        "base_model.model.model.rotary_emb",
        "model.model.rotary_emb",
        "model.rotary_emb",
        "rotary_emb",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def _normalize_split_layer_idx(split_layer_idx: int | torch.Tensor) -> int:
    if isinstance(split_layer_idx, torch.Tensor):
        values = split_layer_idx.flatten()
        if values.numel() == 0:
            raise ValueError("split_layer_idx tensor must not be empty")
        first = int(values[0].item())
        if not torch.all(values == first):
            raise ValueError("All split_layer_idx values in a batch must match")
        return first
    return int(split_layer_idx)


def _module_device(module: nn.Module | None) -> torch.device | None:
    if module is None:
        return None
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _resolve_runtime_device(
    model: nn.Module,
    split_idx: int,
    *,
    decoder_layers: nn.ModuleList,
    final_norm: nn.Module | None,
    lm_head: nn.Module | None,
    rotary_emb: nn.Module | None,
) -> torch.device:
    for layer in decoder_layers[split_idx:]:
        layer_device = _module_device(layer)
        if layer_device is not None:
            return layer_device

    for module in (final_norm, lm_head, rotary_emb):
        module_device = _module_device(module)
        if module_device is not None:
            return module_device

    return next(model.parameters()).device


def forward_from_hidden_states(
    model: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    split_layer_idx: int | torch.Tensor,
    device: torch.device | str | None = None,
    position_ids: torch.Tensor | None = None,
    decoder_layers: nn.ModuleList | None = None,
    final_norm: nn.Module | None = None,
    lm_head: nn.Module | None = None,
    rotary_emb: nn.Module | None = None,
    layer_types: list[str] | None = None,
) -> torch.Tensor:
    """Compute causal LM loss from cached hidden states entering a split layer."""
    decoder_layers = decoder_layers or _get_decoder_layers(model)
    hidden = forward_suffix_hidden_states(
        model,
        hidden_states,
        attention_mask,
        split_layer_idx=split_layer_idx,
        device=device,
        position_ids=position_ids,
        decoder_layers=decoder_layers,
        rotary_emb=rotary_emb,
        layer_types=layer_types,
    )
    final_norm = final_norm or _get_final_norm(model)
    lm_head = lm_head or _get_lm_head(model)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)

    logits = lm_head(final_norm(hidden))
    model_config = _get_model_config(model)
    vocab_size = getattr(model_config, "vocab_size", logits.shape[-1])
    return _compute_causal_lm_loss(
        logits,
        labels.to(hidden.device),
        loss_function=_get_loss_function(model),
        vocab_size=vocab_size,
    )


def forward_suffix_hidden_states(
    model: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    split_layer_idx: int | torch.Tensor,
    device: torch.device | str | None = None,
    position_ids: torch.Tensor | None = None,
    decoder_layers: nn.ModuleList | None = None,
    rotary_emb: nn.Module | None = None,
    layer_types: list[str] | None = None,
) -> torch.Tensor:
    """Run decoder layers from cached hidden states and return final hidden states."""
    split_idx = _normalize_split_layer_idx(split_layer_idx)

    decoder_layers = decoder_layers or _get_decoder_layers(model)
    rotary_emb = rotary_emb if rotary_emb is not None else _get_rotary_emb(model)
    layer_types = layer_types if layer_types is not None else _get_layer_types(model)

    if device is None:
        device = _resolve_runtime_device(
            model,
            split_idx,
            decoder_layers=decoder_layers,
            final_norm=None,
            lm_head=None,
            rotary_emb=rotary_emb,
        )

    if hidden_states.ndim == 2:
        hidden_states = hidden_states.unsqueeze(0)
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if position_ids is not None and position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)

    hidden = hidden_states.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device) if position_ids is not None else None

    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
    text_position_ids: torch.Tensor | None = None
    if rotary_emb is not None:
        use_3d = layer_types is not None
        num_groups = _get_num_position_groups(model) if use_3d else 1

        if position_ids is None:
            seq_len = hidden.shape[1]
            if use_3d:
                position_ids = (
                    torch.arange(seq_len, device=device)
                    .view(1, 1, -1)
                    .expand(num_groups, hidden.shape[0], -1)
                )
            else:
                position_ids = (
                    torch.arange(seq_len, device=device)
                    .view(1, -1)
                    .expand(hidden.shape[0], -1)
                )
        elif position_ids.ndim == 2 and use_3d:
            position_ids = position_ids[None, ...].expand(
                num_groups, position_ids.shape[0], -1
            )

        if use_3d and position_ids.ndim == 3 and position_ids.shape[0] == num_groups:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids
            rotary_position_ids = position_ids

        position_embeddings = rotary_emb(hidden, rotary_position_ids)
    else:
        text_position_ids = position_ids

    linear_attn_mask: torch.Tensor | None = attention_mask
    if attention_mask is not None and torch.all(attention_mask == 1):
        linear_attn_mask = None

    for offset, layer in enumerate(decoder_layers[split_idx:]):
        global_idx = split_idx + offset
        if layer_types is not None and global_idx < len(layer_types):
            layer_mask = (
                linear_attn_mask
                if layer_types[global_idx] == "linear_attention"
                else None
            )
        else:
            layer_mask = linear_attn_mask

        kwargs: dict[str, object] = {"attention_mask": layer_mask}
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if text_position_ids is not None:
            kwargs["position_ids"] = text_position_ids

        layer_outputs = layer(hidden, **kwargs)
        hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    return hidden


def _get_layer_types(model: nn.Module) -> list[str] | None:
    """Extract per-layer type list from model config (Qwen3.5 hybrid attention)."""
    config = None
    for path in [
        "base_model.model.model.config",
        "model.model.config",
        "model.config",
        "config",
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            config = obj
            break
        except AttributeError:
            continue

    if config is not None and hasattr(config, "layer_types"):
        layer_types = getattr(config, "layer_types")
        if isinstance(layer_types, (list, tuple)):
            return [str(layer_type) for layer_type in layer_types]
    return None


def _get_model_config(model: nn.Module):
    """Extract model config object, or None if not found."""
    for path in [
        "base_model.model.model.config",
        "model.model.config",
        "model.config",
        "config",
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def _get_loss_function(model: nn.Module):
    """Extract the configured causal-LM loss function when available."""
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        loss_function = getattr(candidate, "loss_function", None)
        if callable(loss_function):
            return loss_function
    return None


def _get_num_position_groups(model: nn.Module) -> int:
    """Detect position-id group count from model config.

    Qwen3.5 hybrid-attention models use 3D position_ids of shape
    ``(num_key_value_heads, batch, seq)``.  The group count equals
    ``config.num_key_value_heads``.  Falls back to 4 when the attribute
    is absent (backward-compatible default).
    """
    config = _get_model_config(model)
    if config is not None:
        num_kv = getattr(config, "num_key_value_heads", None)
        if num_kv is not None and isinstance(num_kv, int) and num_kv > 0:
            return num_kv
    return 4


def _compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_function=None,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Compute causal LM cross-entropy loss (shift by 1)."""
    if loss_function is not None and vocab_size is not None:
        try:
            return loss_function(
                logits=logits.float(),
                labels=labels,
                vocab_size=int(vocab_size),
            )
        except TypeError:
            pass

    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def determine_split_layer(active_layer_indices: set[int], num_layers: int) -> int:
    """Determine the optimal split point for caching.

    The split layer is the first (lowest) active layer index.
    All layers before this are guaranteed unchanged after extrapolation.

    Args:
        active_layer_indices: Set of layer indices modified by extrapolation.
        num_layers: Total number of decoder layers.

    Returns:
        Layer index where cache boundary should be placed.
        Returns num_layers (no caching benefit) if all layers are active.
    """
    if not active_layer_indices:
        return num_layers  # Nothing active, no caching needed
    return min(active_layer_indices)


@torch.no_grad()
def _full_eval_no_cache(
    model: nn.Module,
    dataloader,
    device: torch.device | str,
    max_batches: int | None = None,
    max_examples: int | None = None,
) -> float:
    """Simple full eval without caching (fallback path)."""
    if max_batches is not None and max_examples is not None:
        raise ValueError("Specify at most one of max_batches or max_examples")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    count = 0
    total_examples = 0
    try:
        for batch in dataloader:
            if max_examples is not None:
                remaining = max_examples - total_examples
                if remaining <= 0:
                    break
                batch = _truncate_batch(batch, remaining)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if not _has_supervised_tokens(batch):
                continue

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            batch_examples = _infer_batch_size(batch)
            total_loss += outputs.loss.item() * batch_examples
            count += 1
            total_examples += batch_examples
            if max_batches and count >= max_batches:
                break
    finally:
        if was_training:
            model.train()
    return total_loss / total_examples if total_examples > 0 else float("nan")
