"""Progressive Freezing controller (GOAL §1.6, design §4.1).

Two operating modes share one controller:

* **Single-shot gate (Phase 1)** — freeze one target layer
  (:meth:`should_freeze` / :meth:`cache_xin` / :meth:`apply_freeze`). The
  original Level 1 freeze: ``requires_grad=False`` on the target layer's LoRA
  params, with the ``xin`` cache captured at freeze time for the Level 2
  activation-matching local loss.

* **Progressive multi-layer (Phase 2, design §4.1)** — freeze layers
  cumulatively across epochs, one after another, driven by a
  :class:`~src.tg_lora.freeze_schedule.FreezeSchedule`
  (:meth:`layers_due_at` / :meth:`apply_freeze_layer` / :meth:`progress`).
  This is the literal mechanism "Progressive" Freezing is named after: the run
  freezes ``X`` at ``T1``, ``X-1`` at ``T2``, ``X-2`` at ``T3`` ... and the
  frozen set only ever grows (design §4.1). Each frozen layer keeps its own
  ``xin`` cache so :meth:`compute_local_loss` can train each front layer
  against the input *its* frozen successor expected.

Level 1 freeze sets ``requires_grad=False`` on the target layer's LoRA params.
Activation gradients still flow through the frozen layer during backprop. The
``xin`` cache captures the input activation entering the frozen layer at the
moment of freezing, for the Level 2 activation-matching local loss.

See docs/design/10_progressive_freezing.md for the design rationale.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.lora_utils import iter_all_lora_params_by_layer
from src.tg_lora.activation_cache import _get_decoder_layers
from src.tg_lora.activation_matching import (
    ActivationMatchingBreakdown,
    ActivationMatchingLoss,
)
from src.tg_lora.freeze_schedule import (
    FreezeSchedule,
    FreezeScheduleConfig,
    random_freeze_order,
)

logger = logging.getLogger("tg-lora")


def build_freeze_schedule_from_config(
    schedule_cfg: Mapping[str, object] | None,
    active_layer_indices: Iterable[int],
    num_epochs: int,
    *,
    default_start_epoch: int = 0,
) -> FreezeSchedule | None:
    """Resolve a multi-layer freeze-schedule config block, or ``None``.

    Config→schedule glue so ``train_tg_lora`` can construct a multi-layer
    controller from a config block — making the Progressive Freeze mechanism
    (design §4.1) reachable from a real run, the prerequisite for the Tier-2 §4
    order verdict's valid_loss axis (multi-layer ``output_first`` vs
    ``random_order``). Returns ``None`` when ``schedule_cfg`` is absent/empty (the
    trainer keeps its single-shot Phase 1 gate); otherwise resolves the request
    into a :class:`FreezeSchedule` the controller drives via :meth:`progress` /
    :meth:`layers_due_at`. ``FreezeScheduleConfig`` validates the request, so a
    malformed config raises ``ValueError`` at construction. ``num_epochs`` drops
    freezes landing past the run (matches :class:`FreezeCostAccountant`);
    ``default_start_epoch`` is the single-shot start cycle passed when the block
    omits ``start_epoch`` (a depth/rate generalization, not a timing shift).

    ``policy="random_order"`` (with ``seed``) builds the GOAL §4 surrogate-null
    arm: it resolves to ``convergence_order`` carrying a reproducible
    :func:`~src.tg_lora.freeze_schedule.random_freeze_order`, so the surrogate
    flows the identical planner path as a real schedule (no separate random
    branch). With identical ``(max_depth, start_epoch, spacing, num_epochs)``
    the candidate(``output_first``) and surrogate(``random_order``) freeze the
    same layers at the same epochs in a different order — isolating ORDER as
    the sole degree of freedom the Tier-2 §4 order verdict resolves. ``seed``
    is required; an explicit ``convergence_order`` is rejected as contradictory.
    """
    if not schedule_cfg:
        return None
    active = list(active_layer_indices)
    policy = str(schedule_cfg.get("policy", "output_first"))
    # Pre-resolution arm identity. The Tier-2 §4 order verdict's candidate
    # (output_first) and surrogate (random_order) freeze the SAME layers at full
    # depth in a different order, so the run-summary footer needs the REQUESTED
    # policy + seed to tell the arms apart once frozen_layers collides. Capture
    # it here before random_order resolves to convergence_order below.
    requested_policy = policy
    surrogate_seed: int | None = None
    convergence_order = schedule_cfg.get("convergence_order")
    stability_epoch = schedule_cfg.get("stability_epoch")

    # Seeded random-order surrogate: GOAL §4 / §3.1 Phase 2 control-(ii) null
    # baseline, reachable from a config block so a real run can express the
    # candidate(output_first) vs surrogate(random_order) contrast across the
    # multi-seed sweeps the Tier-2 §4 order verdict needs. Resolves to the
    # existing 'convergence_order' policy carrying a reproducible
    # :func:`random_freeze_order` — the surrogate then flows through the
    # IDENTICAL planner/accountant/frontier path as a real schedule (design
    # §5.3: no separate random branch → apples-to-apples candidate-vs-surrogate,
    # with timing held fixed so ORDER is the sole differing degree of freedom).
    if policy == "random_order":
        if convergence_order is not None:
            raise ValueError(
                "policy 'random_order' generates its order from 'seed'; an "
                "explicit 'convergence_order' is contradictory"
            )
        seed = schedule_cfg.get("seed")
        if seed is None:
            raise ValueError(
                "policy 'random_order' requires a 'seed' for a reproducible "
                "surrogate"
            )
        surrogate_seed = int(seed)
        convergence_order = random_freeze_order(active, surrogate_seed)
        policy = "convergence_order"

    config = FreezeScheduleConfig(
        active_layer_indices=active,
        num_epochs=int(num_epochs),
        max_depth=int(schedule_cfg.get("max_depth", len(active))),
        start_epoch=int(schedule_cfg.get("start_epoch", default_start_epoch)),
        spacing=int(schedule_cfg.get("spacing", 1)),
        policy=policy,
        convergence_order=tuple(convergence_order)  # type: ignore[arg-type]
        if convergence_order is not None
        else None,
        stability_epoch=dict(stability_epoch)  # type: ignore[arg-type]
        if stability_epoch is not None
        else None,
    )
    return FreezeSchedule.plan(
        config,
        requested_policy=requested_policy,
        surrogate_seed=surrogate_seed,
    )


@dataclass
class FreezeResult:
    frozen_layer_idx: int
    num_frozen_params: int
    frozen_param_names: list[str]
    xin_shape: tuple[int, ...] | None = None


class ProgressiveFreezeController:
    """Manages the progressive freeze gate (Phase 1 single-shot + Phase 2 multi-layer).

    Single-shot usage in the training cycle loop::

        ctrl = ProgressiveFreezeController(
            start_cycle=3,
            freeze_layer="last_active",
            active_layer_indices={24, 25, 26, 27, 28, 29, 30, 31},
        )
        # After W0 snapshot, before pilot steps:
        if ctrl.should_freeze(cycle):
            _, xin_shape = ctrl.cache_xin(model, valid_loader, device)
            result = ctrl.apply_freeze(model)

    Progressive usage (design §4.1), one controller driving the whole run::

        schedule = FreezeSchedule.plan(FreezeScheduleConfig(...))
        ctrl = ProgressiveFreezeController(
            start_cycle=schedule.config.start_epoch,
            active_layer_indices=set(schedule.config.active_layer_indices),
            schedule=schedule,
        )
        for epoch in range(num_epochs):
            ctrl.progress(model, epoch, valid_loader, device)   # freezes layers due this epoch
            # ... train the still-active front; for any frozen layer X,
            # ctrl.compute_local_loss(model, batch, loss_fn, layer_idx=X) ...

    Both modes share the frozen-set and per-layer ``xin`` state, so a controller
    may also be driven layer-by-layer via :meth:`apply_freeze_layer` without a
    schedule.
    """

    def __init__(
        self,
        start_cycle: int,
        freeze_layer: int | str = "last_active",
        active_layer_indices: set[int] | None = None,
        schedule: FreezeSchedule | None = None,
    ) -> None:
        self._start_cycle = start_cycle
        self._freeze_layer_spec = freeze_layer
        self._active_layer_indices = active_layer_indices or set()
        self._schedule = schedule
        # Cumulative state shared by both modes. The single-shot gate sets one
        # entry; progressive mode grows the set across epochs. ``_last_frozen``
        # backs the ``frozen_layer_idx`` property (single-shot compat + the most
        # recent progressive freeze).
        self._frozen_layers: set[int] = set()
        self._last_frozen_layer: int | None = None
        # Single-shot target store: {batch_idx: [xin]} for the one frozen layer.
        self._xin_cache: dict[int, list[torch.Tensor]] = {}
        # Progressive per-layer target store: {layer_idx: {batch_idx: [xin]}}.
        self._xin_caches: dict[int, dict[int, list[torch.Tensor]]] = {}

    # -- state ---------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """True once any layer has been frozen (single-shot: the one target)."""
        return bool(self._frozen_layers)

    @property
    def frozen_layer_idx(self) -> int | None:
        """Index of the most recently frozen layer (``None`` until first freeze)."""
        return self._last_frozen_layer

    @property
    def frozen_layers(self) -> frozenset[int]:
        """Cumulative set of frozen layer indices (grows monotonically)."""
        return frozenset(self._frozen_layers)

    @property
    def schedule(self) -> FreezeSchedule | None:
        return self._schedule

    # -- single-shot gate (Phase 1) ------------------------------------------

    def should_freeze(self, cycle: int) -> bool:
        """Single-shot gate: due at ``start_cycle`` and not yet frozen.

        Progressive callers use :meth:`layers_due_at` instead — once the first
        layer freezes this returns False, which is correct for the one-shot gate
        but would wrongly halt a progressive loop.
        """
        return not self._frozen_layers and cycle >= self._start_cycle

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
        """Single-shot: capture ``xin`` for the resolved target layer (1 batch)."""
        target_idx = self._resolve_target_layer(model)
        xin_cache, xin_shape = self._capture_xin_for_layer(
            model, target_idx, dataloader, device
        )
        self._xin_cache = xin_cache
        return xin_cache, xin_shape

    def apply_freeze(self, model: nn.Module) -> FreezeResult:
        """Single-shot: freeze the resolved target layer's LoRA params.

        Raises ``RuntimeError`` if that layer is already frozen (idempotency
        guard). Progressive freezing calls :meth:`apply_freeze_layer` with a
        different layer each epoch instead.
        """
        target_idx = self._resolve_target_layer(model)
        if target_idx in self._frozen_layers:
            raise RuntimeError(f"Layer {target_idx} already frozen")
        return self._freeze_layer(model, target_idx, xin_cache=self._xin_cache)

    # -- progressive multi-layer (Phase 2, design §4.1) ----------------------

    def layers_due_at(self, epoch: int) -> list[int]:
        """Layers the schedule freezes at ``epoch`` (ascending, for determinism).

        Requires construction with ``schedule=``. A layer absent from the result
        either never freezes or freezes at a different epoch.
        """
        if self._schedule is None:
            raise RuntimeError(
                "layers_due_at requires a schedule; construct the controller "
                "with schedule=FreezeSchedule.plan(...)"
            )
        due = [
            layer
            for layer, at_epoch in self._schedule.frozen_at_epoch.items()
            if at_epoch == epoch
        ]
        return sorted(due)

    def apply_freeze_layer(
        self,
        model: nn.Module,
        layer_idx: int,
        dataloader,
        device: torch.device | str = "cpu",
    ) -> FreezeResult:
        """Progressive primitive: cache ``xin`` for ``layer_idx`` then freeze it.

        Captures the layer's input activation (its ``xin``) at this moment and
        freezes its LoRA params. Idempotent per layer — re-freezing an already
        frozen layer raises (design §4.1: the frozen set only grows, never
        re-freezes). Distinct layers may be frozen across successive calls.
        """
        if layer_idx in self._frozen_layers:
            raise RuntimeError(f"Layer {layer_idx} already frozen")
        xin_cache, _ = self._capture_xin_for_layer(
            model, layer_idx, dataloader, device
        )
        self._xin_caches[layer_idx] = xin_cache
        return self._freeze_layer(model, layer_idx, xin_cache=xin_cache)

    def progress(
        self,
        model: nn.Module,
        epoch: int,
        dataloader,
        device: torch.device | str = "cpu",
    ) -> list[FreezeResult]:
        """Freeze every layer the schedule names for ``epoch`` (design §4.1).

        Returns one :class:`FreezeResult` per layer frozen this epoch (empty if
        none are due). Safe to call every epoch: layers already frozen are
        skipped by the schedule, and :meth:`apply_freeze_layer` guards the rest.
        """
        return [
            self.apply_freeze_layer(model, layer_idx, dataloader, device)
            for layer_idx in self.layers_due_at(epoch)
        ]

    # -- Level 2 activation-matching local loss ------------------------------

    def compute_local_loss(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: ActivationMatchingLoss,
        *,
        batch_idx: int = 0,
        device: torch.device | str = "cpu",
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Front-layer activation-matching loss against a cached ``xin``.

        The Level 2 training signal (GOAL §1.6.1, design §8 items 3-4): once
        layer ``X`` is frozen, run the model forward and capture ``X``'s input
        — which is exactly the front layer ``X-1``'s current output. Pair it
        with the cached ``xin`` (the *past* input ``X`` received at freeze time)
        and return ``loss_fn(front_output, xin, mask)``. The scalar carries a
        gradient into the front layer's parameters; the frozen layer's
        parameters receive none (``requires_grad=False`` from the freeze).

        The gap is non-zero only because ``xin`` is a past observation that the
        still-training front layer no longer emits (design §3.3) — so this is a
        genuine learning signal, not "current output vs itself".

        Parameters
        ----------
        model:
            Model with the target layer already frozen.
        batch:
            Batch dict with ``input_ids`` and ``attention_mask``; padded
            positions are masked out of the loss (leak-free, GOAL §3.3).
        loss_fn:
            :class:`ActivationMatchingLoss`. The Phase 1 gate uses the pure-MSE
            default.
        batch_idx:
            Which cached ``xin`` batch to pair against. Phase 1 captures batch 0.
        device:
            Device for the forward pass and ``xin`` target alignment.
        layer_idx:
            Which frozen layer's cached ``xin`` to match against. ``None``
            (default) selects the single-shot frozen layer
            (:meth:`apply_freeze`); an index selects a progressively frozen
            layer (:meth:`apply_freeze_layer`). The captured front activation is
            that frozen layer's current input.

        Raises
        ------
        RuntimeError
            If the layer is not frozen, no ``xin`` is cached for it at
            ``batch_idx``, or the forward did not reach the frozen layer.
        """
        predicted, target, mask = self._local_loss_inputs(
            model, batch, batch_idx=batch_idx, device=device, layer_idx=layer_idx
        )
        return loss_fn(predicted, target, mask=mask)

    def local_loss_breakdown(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: ActivationMatchingLoss,
        *,
        batch_idx: int = 0,
        device: torch.device | str = "cpu",
        layer_idx: int | None = None,
    ) -> ActivationMatchingBreakdown:
        """Per-arm breakdown of the boundary local loss — the Phase-3 arms, observed.

        The decomposed counterpart of :meth:`compute_local_loss`: same forward,
        same cached ``xin``, same mask, but returns
        :meth:`ActivationMatchingLoss.breakdown` so each Phase-3 arm's weighted
        contribution (``mse`` / ``cosine`` / ``dist``) is read off the training
        path rather than inferred by calling the scalar loss with and without an
        arm's weight. Use this to log which arm is active (non-zero) at each
        post-freeze step of the Level-2 trajectory — the Phase-3 ablation made
        observable in the training path (GOAL §3.1; design §6.2), the status the
        constitution's §7 honesty rule requires before the arm is trusted
        downstream. ``loss_fn.total`` equals :meth:`compute_local_loss` for the
        same inputs, so the two never disagree.
        """
        predicted, target, mask = self._local_loss_inputs(
            model, batch, batch_idx=batch_idx, device=device, layer_idx=layer_idx
        )
        return loss_fn.breakdown(predicted, target, mask=mask)

    def _local_loss_inputs(
        self,
        model: nn.Module,
        batch: dict,
        *,
        batch_idx: int = 0,
        device: torch.device | str = "cpu",
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Resolve the ``(predicted, target, mask)`` triple the local loss trains on.

        Shared by :meth:`compute_local_loss` (which returns the scalar) and
        :meth:`local_loss_breakdown` (which returns the per-arm decomposition), so
        the two observe an identical forward / cached ``xin`` / mask and can never
        diverge. Raises the same ``RuntimeError`` cases as :meth:`compute_local_loss`.
        """
        if layer_idx is None:
            if self._last_frozen_layer is None:
                raise RuntimeError(
                    "compute_local_loss requires apply_freeze() first"
                )
            target_idx = self._last_frozen_layer
            cached = self._xin_cache.get(batch_idx)
            if not cached:
                raise RuntimeError(
                    f"no cached xin for batch {batch_idx}; call cache_xin() first"
                )
        else:
            if layer_idx not in self._frozen_layers:
                raise RuntimeError(
                    f"layer {layer_idx} is not frozen; "
                    "call apply_freeze_layer() first"
                )
            per_layer = self._xin_caches.get(layer_idx)
            if not per_layer or batch_idx not in per_layer:
                raise RuntimeError(
                    f"no cached xin for layer {layer_idx} batch {batch_idx}; "
                    "call apply_freeze_layer() first"
                )
            target_idx = layer_idx
            cached = per_layer[batch_idx]

        predicted = self._capture_live_layer_input(model, target_idx, batch, device)
        target = cached[0].to(device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device=device, dtype=predicted.dtype).unsqueeze(-1)
        return predicted, target, mask

    # -- private helpers -----------------------------------------------------

    @torch.no_grad()
    def _capture_xin_for_layer(
        self,
        model: nn.Module,
        layer_idx: int,
        dataloader,
        device: torch.device | str,
    ) -> tuple[dict[int, list[torch.Tensor]], tuple[int, ...] | None]:
        """Capture ``xin`` (layer input) for ``layer_idx`` via forward pre-hook.

        Phase 1: one batch is sufficient for shape verification and the local
        loss target (the same data is seen many times in the few-data,
        many-epoch regime this method targets, design §4.3).
        """
        decoder_layers = _get_decoder_layers(model)
        target_layer = decoder_layers[layer_idx]

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

        return xin_cache, xin_shape

    def _capture_live_layer_input(
        self,
        model: nn.Module,
        layer_idx: int,
        batch: dict,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Forward capture of ``layer_idx``'s input, kept on the autograd graph.

        Unlike :meth:`_capture_xin_for_layer` this is a live capture that must
        stay differentiable so the local loss backprops into the front layer.
        """
        decoder_layers = _get_decoder_layers(model)
        target_layer = decoder_layers[layer_idx]

        captured: list[torch.Tensor] = []

        def _hook_fn(module, args, kwargs):
            del module
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

        return captured[0]

    def _freeze_layer(
        self,
        model: nn.Module,
        layer_idx: int,
        *,
        xin_cache: dict[int, list[torch.Tensor]],
    ) -> FreezeResult:
        """Set ``requires_grad=False`` on ``layer_idx``'s LoRA params and record it.

        Shared by single-shot :meth:`apply_freeze` and progressive
        :meth:`apply_freeze_layer`. Callers are responsible for the per-layer
        idempotency guard and for having populated ``xin_cache``.
        """
        layer_map = iter_all_lora_params_by_layer(model)
        frozen_names: list[str] = []
        if layer_idx in layer_map:
            for name, param in layer_map[layer_idx]:
                param.requires_grad = False
                frozen_names.append(name)

        self._frozen_layers.add(layer_idx)
        self._last_frozen_layer = layer_idx

        xin_shape = None
        if xin_cache and 0 in xin_cache:
            xin_shape = tuple(xin_cache[0][0].shape)

        result = FreezeResult(
            frozen_layer_idx=layer_idx,
            num_frozen_params=len(frozen_names),
            frozen_param_names=frozen_names,
            xin_shape=xin_shape,
        )

        logger.info(
            "Progressive freeze applied: layer=%d params=%d xin_shape=%s "
            "cumulative_frozen=%s",
            layer_idx,
            len(frozen_names),
            xin_shape,
            sorted(self._frozen_layers),
        )
        return result

    def clear_xin_cache(self) -> None:
        self._xin_cache.clear()

    # -- resume state (fault / periodic checkpoint) -------------------------

    def state_dict(self) -> dict:
        """Serialize the cumulative freeze state for checkpoint resume.

        The progressive-freeze run's defining state is the cumulative
        ``_frozen_layers`` set (design §4.1: it only ever grows). The training
        loop rebuilds this controller fresh from config on resume, and the LoRA
        adapter weights are restored from safetensors — which carries weights but
        NOT the ``requires_grad=False`` flag the freeze set. Without persisting
        ``_frozen_layers``, a fault-resume rebuilds it empty, the loop's
        ``layers_due_at(cycle)`` gate (which fires only for cycles
        ``>= cycle_offset``, skipping the cycles before the fault) never
        re-freezes the pre-fault layers, and (a) those layers silently re-train —
        undoing the cost reduction that defines Progressive Freezing — and
        (b) the run-summary footer's ``frozen_layers`` (the Tier-2 §4 order-verdict
        arm provenance :func:`progressive_freeze_run_summary` emits) reports only
        post-fault freezes. Sibling resume-state-loss to dynfreeze / LAWA / warmup
        (all already persisted in :class:`~src.utils.checkpoint.TrainingState`).

        The Level-2 activation-matching ``xin`` caches are deliberately NOT
        persisted here: they back the Phase-3 local loss (MS-PF3), a separate
        research axis not on the Tier-2 valid_loss path the freeze-set state
        serves. A resumed Phase-3 run would need to re-cache ``xin`` on its next
        freeze; since the frozen set is restored, the Tier-2 valid_loss axis is
        fully closed by the frozen-set state alone.
        """
        return {
            "frozen_layers": sorted(self._frozen_layers),
            "last_frozen_layer": self._last_frozen_layer,
        }

    def load_state_dict(self, state: dict | None) -> None:
        """Restore cumulative freeze state from a checkpoint.

        Inverse of :meth:`state_dict`. Restores ``_frozen_layers`` /
        ``_last_frozen_layer`` only; the caller must then call
        :meth:`refreeze_loaded_layers` to re-apply ``requires_grad=False`` on the
        freshly adapter-loaded model (safetensors does not carry it). Accepts
        ``None`` (no progressive-freeze state / a pre-fix checkpoint / a disabled
        run) as a no-op, so the resume path mirrors the LAWA / dynfreeze
        ``None``-safe contract rather than needing a separate guard at every
        call site.
        """
        if not state:
            return
        self._frozen_layers = {int(idx) for idx in state.get("frozen_layers", [])}
        last = state.get("last_frozen_layer")
        self._last_frozen_layer = int(last) if last is not None else None

    def refreeze_loaded_layers(self, model: nn.Module) -> list[int]:
        """Re-apply ``requires_grad=False`` to every loaded frozen layer.

        After :meth:`load_state_dict` the controller knows WHICH layers were
        frozen, but the model's LoRA params (just loaded from safetensors) are
        all ``requires_grad=True``. This walks ``_frozen_layers`` and flips each
        layer's LoRA params back to frozen — closing the resume gap the
        safetensors weight load opens. Returns the layers re-frozen (ascending),
        empty when nothing was frozen or the model lacks those layers. Idempotent
        and safe to call on a fresh controller (empty ``_frozen_layers``).
        """
        layer_map = iter_all_lora_params_by_layer(model)
        refrozen: list[int] = []
        for layer_idx in sorted(self._frozen_layers):
            if layer_idx in layer_map:
                for _name, param in layer_map[layer_idx]:
                    param.requires_grad = False
                refrozen.append(layer_idx)
        return refrozen


def progressive_freeze_run_summary(controller: ProgressiveFreezeController) -> dict:
    """Build the run-summary provenance block for a progressive-freeze run.

    The footer persisted by ``train_tg_lora`` must distinguish the Tier-2 §4
    order verdict's candidate (``output_first``) and surrogate (``random_order``)
    arms: at full depth they freeze the SAME layers in a different ORDER, so the
    frozen-layer set is identical and the arm's policy + seed is the only
    machine-readable distinguisher. Before this helper the summary recorded only
    the frozen layers and the start cycle, so a deposited ``run_metrics.jsonl``
    could not tell the arms apart — forcing the hand-labeling transcription hazard
    the deposit CLI removed for ``best_valid_loss``.

    Emits ``policy`` (the requested arm), ``resolved_policy`` (the planner policy
    ``random_order`` collapses to), ``surrogate_seed`` (``None`` for a real arm)
    and ``realized_depth`` only when a multi-layer schedule drove the run; a
    single-shot Phase-1 run reports ``mode="single_shot"`` and omits them so
    legacy footers stay single-shot.
    """
    block: dict = {
        "enabled": True,
        "frozen_layer": controller.frozen_layer_idx,
        "frozen_layers": sorted(controller.frozen_layers),
        "start_cycle": controller._start_cycle,
        "mode": "progressive" if controller.schedule is not None else "single_shot",
    }
    schedule = controller.schedule
    if schedule is not None:
        block["policy"] = schedule.requested_policy
        block["resolved_policy"] = schedule.config.policy
        block["surrogate_seed"] = schedule.surrogate_seed
        block["realized_depth"] = schedule.realized_depth
    return block
