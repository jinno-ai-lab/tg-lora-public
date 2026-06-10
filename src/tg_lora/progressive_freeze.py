"""Progressive Freezing Phase 1: single-layer freeze gate controller.

Level 1 freeze: sets requires_grad=False on the target layer's LoRA params.
Activation gradients still flow through the frozen layer during backprop.
The xin cache captures the input activation entering the frozen layer at the
moment of freezing, for later Level 2 activation matching experiments.

See docs/design/10_progressive_freezing.md for the design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.model.lora_utils import iter_all_lora_params_by_layer
from src.tg_lora.activation_cache import _get_decoder_layers

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

    def clear_xin_cache(self) -> None:
        self._xin_cache.clear()
