"""Progressive Freezing Phase 1: single-layer freeze gate controller.

Level 1 freeze: sets requires_grad=False on the target layer's LoRA params.
Activation gradients still flow through the frozen layer during backprop.
The xin cache captures the input activation entering the frozen layer at the
moment of freezing, for later Level 2 activation matching experiments.

See docs/design/10_progressive_freezing.md for the design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.lora_utils import iter_all_lora_params_by_layer
from src.tg_lora.activation_cache import _get_decoder_layers
from src.tg_lora.activation_matching import ActivationMatchingLoss

logger = logging.getLogger("tg-lora")


@dataclass
class FreezeResult:
    frozen_layer_idx: int
    num_frozen_params: int
    frozen_param_names: list[str]
    xin_shape: tuple[int, ...] | None = None


class ProgressiveFreezeController:
    """Manages the single-layer progressive freeze gate (Phase 1).

    Usage in the training cycle loop::

        ctrl = ProgressiveFreezeController(
            start_cycle=3,
            freeze_layer="last_active",
            active_layer_indices={24, 25, 26, 27, 28, 29, 30, 31},
        )
        # After W0 snapshot, before pilot steps:
        if ctrl.should_freeze(cycle):
            _, xin_shape = ctrl.cache_xin(model, valid_loader, device)
            result = ctrl.apply_freeze(model)
    """

    def __init__(
        self,
        start_cycle: int,
        freeze_layer: int | str = "last_active",
        active_layer_indices: set[int] | None = None,
    ) -> None:
        self._start_cycle = start_cycle
        self._freeze_layer_spec = freeze_layer
        self._active_layer_indices = active_layer_indices or set()
        self._is_frozen = False
        self._frozen_layer_idx: int | None = None
        self._xin_cache: dict[int, list[torch.Tensor]] = {}

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def frozen_layer_idx(self) -> int | None:
        return self._frozen_layer_idx

    def should_freeze(self, cycle: int) -> bool:
        return not self._is_frozen and cycle >= self._start_cycle

    def _resolve_target_layer(self, model: nn.Module) -> int:
        if isinstance(self._freeze_layer_spec, int):
            return self._freeze_layer_spec
        if self._active_layer_indices:
            return max(self._active_layer_indices)
        layer_map = iter_all_lora_params_by_layer(model)
        if not layer_map:
            raise ValueError("No LoRA layers found in model")
        return max(layer_map.keys())

    @torch.no_grad()
    def cache_xin(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device | str,
    ) -> tuple[dict[int, list[torch.Tensor]], tuple[int, ...] | None]:
        """Capture xin (layer input activation) via forward pre-hook."""
        target_idx = self._resolve_target_layer(model)
        decoder_layers = _get_decoder_layers(model)
        target_layer = decoder_layers[target_idx]

        captured: list[torch.Tensor] = []

        def _hook_fn(module, args, kwargs):
            del module
            if args:
                captured.append(args[0].detach().cpu())
            elif "hidden_states" in kwargs:
                captured.append(kwargs["hidden_states"].detach().cpu())

        hook = target_layer.register_forward_pre_hook(_hook_fn, with_kwargs=True)
        was_training = model.training
        model.eval()
        xin_cache: dict[int, list[torch.Tensor]] = {}
        xin_shape = None

        try:
            for batch_idx, batch in enumerate(dataloader):
                captured.clear()
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels"),
                )
                if captured:
                    xin_cache[batch_idx] = list(captured)
                    if xin_shape is None:
                        xin_shape = tuple(captured[0].shape)
                break  # Phase 1: 1 batch sufficient for shape verification
        finally:
            hook.remove()
            if was_training:
                model.train()

        self._xin_cache = xin_cache
        return xin_cache, xin_shape

    def apply_freeze(self, model: nn.Module) -> FreezeResult:
        """Freeze the target layer's LoRA parameters (requires_grad=False)."""
        if self._is_frozen:
            raise RuntimeError("Layer already frozen")

        target_idx = self._resolve_target_layer(model)
        layer_map = iter_all_lora_params_by_layer(model)
        frozen_names: list[str] = []

        if target_idx in layer_map:
            for name, param in layer_map[target_idx]:
                param.requires_grad = False
                frozen_names.append(name)

        self._is_frozen = True
        self._frozen_layer_idx = target_idx

        xin_shape = None
        if self._xin_cache and 0 in self._xin_cache:
            xin_shape = tuple(self._xin_cache[0][0].shape)

        result = FreezeResult(
            frozen_layer_idx=target_idx,
            num_frozen_params=len(frozen_names),
            frozen_param_names=frozen_names,
            xin_shape=xin_shape,
        )

        logger.info(
            "Progressive freeze applied: layer=%d params=%d xin_shape=%s",
            target_idx,
            len(frozen_names),
            xin_shape,
        )
        return result

    def compute_local_loss(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: ActivationMatchingLoss,
        *,
        batch_idx: int = 0,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Front-layer activation-matching loss against the cached ``xin``.

        The Level 2 training signal (GOAL §1.6.1, design §8 items 3-4): once
        layer ``X`` is frozen, run the model forward and capture ``X``'s input
        — which is exactly the front layer ``X-1``'s current output. Pair it
        with the cached ``xin`` (the *past* input ``X`` received at freeze time)
        and return ``loss_fn(front_output, xin, mask)``. The scalar carries a
        gradient into the front layer's parameters; the frozen layer's
        parameters receive none (``requires_grad=False`` from
        :meth:`apply_freeze`).

        The gap is non-zero only because ``xin`` is a past observation that the
        still-training front layer no longer emits (design §3.3) — so this is a
        genuine learning signal, not "current output vs itself".

        Parameters
        ----------
        model:
            Model with the target layer already frozen (call
            :meth:`apply_freeze` first).
        batch:
            Batch dict with ``input_ids`` and ``attention_mask``; padded
            positions are masked out of the loss (leak-free, GOAL §3.3).
        loss_fn:
            :class:`ActivationMatchingLoss`. The Phase 1 gate uses the pure-MSE
            default.
        batch_idx:
            Which cached ``xin`` batch to pair against (captured by
            :meth:`cache_xin`). Phase 1 captures batch 0.
        device:
            Device for the forward pass and ``xin`` target alignment.

        Raises
        ------
        RuntimeError
            If the layer is not frozen, no ``xin`` is cached for ``batch_idx``,
            or the forward did not reach the frozen layer.
        """
        if not self._is_frozen:
            raise RuntimeError(
                "compute_local_loss requires apply_freeze() first"
            )
        cached = self._xin_cache.get(batch_idx)
        if not cached:
            raise RuntimeError(
                f"no cached xin for batch {batch_idx}; call cache_xin() first"
            )

        target_idx = self._frozen_layer_idx
        decoder_layers = _get_decoder_layers(model)
        target_layer = decoder_layers[target_idx]

        captured: list[torch.Tensor] = []

        def _hook_fn(module, args, kwargs):
            del module
            # Keep grad + device: unlike cache_xin this is a live capture that
            # must stay on the autograd graph so the loss backprops into the
            # front layer.
            if args:
                captured.append(args[0])
            elif "hidden_states" in kwargs:
                captured.append(kwargs["hidden_states"])

        hook = target_layer.register_forward_pre_hook(_hook_fn, with_kwargs=True)
        try:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            model(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                labels=batch_dev.get("labels"),
            )
        finally:
            hook.remove()

        if not captured:
            raise RuntimeError(
                "forward did not reach the frozen layer; no front-layer "
                "activation captured"
            )

        predicted = captured[0]
        target = cached[0].to(device)
        mask = batch_dev.get("attention_mask")
        if mask is not None:
            mask = mask.to(device=device, dtype=predicted.dtype).unsqueeze(-1)

        return loss_fn(predicted, target, mask=mask)

    def clear_xin_cache(self) -> None:
        self._xin_cache.clear()
