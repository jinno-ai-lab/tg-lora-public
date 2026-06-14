"""Activation cache for layer-skip evaluation in TG-LoRA (MLX port).

Mirrors the contract of src/tg_lora/activation_cache.py (PyTorch):
    pilot_loss, _ = cache.eval_and_cache(model, batch, split_layer_idx)
    for _ in range(N):
        loss = cache.eval_from_cache_with_model(model)
    # prefix forward is computed ONCE; N suffix forwards reuse the cached
    # hidden state at the split boundary.

Framework-specific notes (Qwen3.5 in mlx_lm):
  - Physical model split: we call the prefix/suffix submodules directly
    (embed + decoder_layers[:split] vs decoder_layers[split:]+norm+lm_head).
    MLX exposes the layer list, so this is cleaner than the PyTorch hook
    approach.
  - Masks: Qwen3.5 alternates between full-attention (Attention) and
    linear/GatedDelta (GatedDeltaNet) layers. The full-attention layers
    need a causal mask ("causal" or additive); the linear layers take
    None. We mirror TextModel.__call__: fa_mask = create_attention_mask
    (returns "causal" when no cache and N>1), ssm_mask = create_ssm_mask
    (returns None). Each layer gets mask = ssm_mask if layer.is_linear
    else fa_mask, and cache = None (single forward, no KV/recurrent state).
  - Position: handled inside the attention/gated-delta modules; we do not
    pass position_ids externally.
  - Layer return: DecoderLayer.__call__ returns mx.array (not a tuple).

The INTERFACE (method names, CachedBatch fields, the (prefix cache +
suffix reuse) contract) is intentionally identical to the PyTorch
version (src/tg_lora/activation_cache.py) so a future framework-agnostic
refactor can extract a shared spec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

logger = logging.getLogger("tg-lora")


@dataclass
class CachedBatch:
    """Stored intermediate state for one eval batch (mirrors PyTorch)."""

    hidden_states: mx.array  # (batch, seq_len, hidden_dim)
    attention_mask: mx.array | None
    position_ids: mx.array | None
    labels: mx.array | None


# ---------------------------------------------------------------------------
# Model structure resolution (Qwen3.5 in mlx_lm)
# ---------------------------------------------------------------------------

def _resolve_submodules(model):
    """Resolve (embed_tokens, decoder_layers, norm, lm_head) for MLX causal LM.

    Supports two layouts:
      1. Multimodal Model wrapper (Qwen3.5-9B ships this): embed under
         model.language_model.model.{embed_tokens,layers,norm} and
         lm_head under model.language_model.lm_head.
      2. Flat Qwen3_5ForCausalLM: model.model.{embed_tokens,layers,norm}
         + model.lm_head.
    """
    lang = getattr(model, "language_model", None)
    if lang is not None:
        base = getattr(lang, "model", None)
        embed = getattr(base, "embed_tokens", None) if base is not None else None
        layers = getattr(base, "layers", None) if base is not None else None
        norm = getattr(base, "norm", None) if base is not None else None
        lm_head = getattr(lang, "lm_head", None)
        if all(x is not None for x in (embed, layers, norm, lm_head)):
            return embed, list(layers), norm, lm_head

    base = getattr(model, "model", model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    norm = getattr(base, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if all(x is not None for x in (embed, layers, norm, lm_head)):
        return embed, list(layers), norm, lm_head

    raise AttributeError(
        f"Cannot resolve prefix/suffix submodules for {type(model).__name__}"
    )


# ---------------------------------------------------------------------------
# Prefix / suffix runners (mirror TextModel.__call__)
# ---------------------------------------------------------------------------

def _run_prefix(embed, decoder_layers, split_layer_idx, input_ids):
    """Run embed + decoder_layers[:split_layer_idx] with per-layer causal mask.

    Returns the hidden state at the input of decoder_layers[split_layer_idx].
    """
    if not (0 < split_layer_idx <= len(decoder_layers)):
        raise ValueError(
            f"split_layer_idx must be in [1, {len(decoder_layers)}], got {split_layer_idx}"
        )

    hidden = embed(input_ids.astype(mx.int32))
    if hidden.dtype != mx.float32:
        hidden = hidden.astype(mx.float32)

    fa_mask = create_attention_mask(hidden, cache=None)  # "causal" or None
    ssm_mask = create_ssm_mask(hidden, cache=None)       # None

    for layer in decoder_layers[:split_layer_idx]:
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=None)
        if hidden.dtype != mx.float32:
            hidden = hidden.astype(mx.float32)

    return hidden


def _run_suffix(decoder_layers, norm, lm_head, split_layer_idx, hidden_states,
                labels=None):
    """Run decoder_layers[split_layer_idx:] + norm + lm_head from cached hidden.

    Returns (loss, logits) if labels else (None, logits). Loss is CE with
    label masking (-100) and standard next-token shift.
    """
    hidden = hidden_states
    if hidden.dtype != mx.float32:
        hidden = hidden.astype(mx.float32)

    fa_mask = create_attention_mask(hidden, cache=None)
    ssm_mask = create_ssm_mask(hidden, cache=None)

    for layer in decoder_layers[split_layer_idx:]:
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=None)
        if hidden.dtype != mx.float32:
            hidden = hidden.astype(mx.float32)

    if norm is not None:
        hidden = norm(hidden)
        if hidden.dtype != mx.float32:
            hidden = hidden.astype(mx.float32)

    logits = lm_head(hidden)
    mx.eval(logits)

    if labels is None:
        return None, logits

    logits_for_loss = logits[:, :-1, :].astype(mx.float32)
    targets = labels[:, 1:].astype(mx.int32)
    valid = (targets != -100)
    safe_targets = mx.where(valid, targets, mx.zeros_like(targets))
    log_probs = logits_for_loss - mx.logsumexp(logits_for_loss, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    nll = mx.where(valid, nll, mx.zeros_like(nll))
    n_valid = valid.astype(mx.float32).sum()
    loss = nll.sum() / mx.maximum(n_valid, mx.array(1.0, dtype=mx.float32))
    mx.eval(loss)
    return float(loss), logits


# ---------------------------------------------------------------------------
# ActivationCache (interface mirrors the PyTorch version)
# ---------------------------------------------------------------------------

class ActivationCache:
    """MLX activation cache for layer-skip evaluation (prefix/suffix split)."""

    def __init__(self) -> None:
        self._batches: list[CachedBatch] = []
        self._split_layer_idx: int = 0
        self._valid: bool = False

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

    def invalidate(self) -> None:
        self._valid = False

    def eval_and_cache(self, model, batches, split_layer_idx):
        """Run prefix (cached) + suffix once per batch; accumulate the cache.

        `batches` is an iterable of dicts (input_ids, [attention_mask],
        labels). Mirrors the PyTorch version which takes a dataloader and
        accumulates all batches in one cache. Returns
        (mean_pilot_loss, num_batches_cached).
        """
        self.clear()
        self._split_layer_idx = split_layer_idx

        embed, decoder_layers, norm, lm_head = _resolve_submodules(model)
        was_training = getattr(model, "training", False)
        model.eval()

        pilot_losses: list[float] = []
        try:
            for batch in batches:
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
                labels = batch.get("labels", None)
                hidden = _run_prefix(
                    embed, decoder_layers, split_layer_idx, input_ids
                )
                mx.eval(hidden)
                loss, _ = _run_suffix(
                    decoder_layers, norm, lm_head, split_layer_idx,
                    hidden, labels,
                )
                pilot_losses.append(loss)
                self._batches.append(
                    CachedBatch(
                        hidden_states=hidden,
                        attention_mask=attention_mask,
                        position_ids=None,
                        labels=labels,
                    )
                )
        finally:
            if was_training:
                model.train()

        self._valid = len(self._batches) > 0
        mean_loss = (
            sum(pilot_losses) / len(pilot_losses) if pilot_losses else float("nan")
        )
        return mean_loss, len(self._batches)

    def eval_from_cache_with_model(self, model):
        """Run the suffix from the cached prefix hidden state."""
        if not self.is_valid:
            raise RuntimeError("Cache is not valid. Call eval_and_cache() first.")
        embed, decoder_layers, norm, lm_head = _resolve_submodules(model)
        was_training = getattr(model, "training", False)
        model.eval()
        total_loss = 0.0
        n = 0
        try:
            for cb in self._batches:
                loss, _ = _run_suffix(
                    decoder_layers, norm, lm_head, self._split_layer_idx,
                    cb.hidden_states, cb.labels,
                )
                total_loss += loss
                n += 1
        finally:
            if was_training:
                model.train()
        return total_loss / max(n, 1)
