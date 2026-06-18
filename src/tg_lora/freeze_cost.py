"""Exact cost accounting for Progressive Freezing (GOAL §5).

Pure-Python arithmetic that measures how much backward compute and VRAM a
freeze schedule saves versus full backprop. No GPU, no model dependency: the
accounting is exact given per-layer costs; estimating those costs from a real
model is a separate, model-specific (and currently [UNVERIFIED]) step.

GOAL §5 contract
----------------
    progressive_backward_flops = Σ_epoch Σ_{active layers} layer cost
    full_backward_flops        = Σ_epoch Σ_{all layers} layer cost
    reduction_rate             = 1 − progressive / full
    VRAM saved                 = frozen layers' optimizer state
                                 (+ activation-gradient buffers at Level 2)

This is deliberately distinct from ``CycleState.reduction_rate``, which
accounts for the *extrapolation/PSA* path. Freeze savings are a different
quantity (backward work elided by frozen layers), so they live here.

See docs/design/10_progressive_freezing.md §7 for the Level 1 / Level 2 cost
tables this engine implements.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_LEVELS: tuple[int, int] = (1, 2)


@dataclass(frozen=True)
class LayerBackwardCost:
    """Backward-compute and memory cost of one layer, per optimizer step.

    weight_grad_flops
        FLOPs to compute this layer's weight gradient. Skipped once the layer
        is frozen (both Level 1 and Level 2).
    act_grad_flops
        FLOPs to propagate the activation gradient through this layer toward
        earlier layers. Under Level 1 this still runs (the gradient must reach
        the unfrozen front layers); under Level 2 the backward-graph suffix is
        cut, so it is skipped.
    optim_state_bytes
        Optimizer-state bytes (e.g. Adam ``m``, ``v``) freed from VRAM once the
        layer is frozen (Level 1 onward).
    act_grad_bytes
        VRAM holding this layer's activation-gradient buffer during backprop.
        Freed only under Level 2 (the suffix cut stores none for frozen layers).
    """

    weight_grad_flops: float = 0.0
    act_grad_flops: float = 0.0
    optim_state_bytes: int = 0
    act_grad_bytes: int = 0


@dataclass(frozen=True)
class FreezeCostSummary:
    """Bundled accounting result for one schedule at one freeze level."""

    level: int
    full_backward_flops: float
    progressive_backward_flops: float
    reduction_rate: float
    peak_vram_saved_bytes: int


@dataclass
class FreezeCostAccountant:
    """Measures Progressive-Freezing savings versus full backprop (GOAL §5).

    Parameters
    ----------
    layer_costs:
        Per-layer backward cost, keyed by layer index.
    steps_per_epoch:
        Optimizer steps run per epoch (``K * grad_accum``).
    num_epochs:
        Total epochs in the run.
    frozen_at_epoch:
        Layer index -> 0-based epoch at which it becomes frozen and stays
        frozen. A layer absent from the map is never frozen. An entry whose
        epoch is ``>= num_epochs`` never freezes during this run (active for
        all epochs), so it saves nothing.

    A layer frozen at epoch ``f`` is active for epochs ``[0, f)`` and frozen
    for ``[f, num_epochs)``. Under Level 1 its weight-gradient compute is
    skipped while frozen but the activation gradient still propagates; under
    Level 2 both are skipped (backward-graph suffix cut).
    """

    layer_costs: dict[int, LayerBackwardCost]
    steps_per_epoch: int
    num_epochs: int
    frozen_at_epoch: dict[int, int]

    def __post_init__(self) -> None:
        if self.steps_per_epoch < 0:
            raise ValueError(
                f"steps_per_epoch must be non-negative, got {self.steps_per_epoch}"
            )
        if self.num_epochs < 0:
            raise ValueError(f"num_epochs must be non-negative, got {self.num_epochs}")
        for idx, epoch in self.frozen_at_epoch.items():
            if idx not in self.layer_costs:
                raise KeyError(f"frozen_at_epoch references unknown layer index {idx}")
            if epoch < 0:
                raise ValueError(
                    f"frozen_at_epoch[{idx}] must be non-negative, got {epoch}"
                )

    @staticmethod
    def _check_level(level: int) -> None:
        if level not in _VALID_LEVELS:
            raise ValueError(f"level must be one of {_VALID_LEVELS}, got {level}")

    def _active_epochs(self, layer_idx: int) -> int:
        """Epochs (out of ``num_epochs``) the layer stays trainable."""
        freeze_epoch = self.frozen_at_epoch.get(layer_idx)
        if freeze_epoch is None:
            return self.num_epochs
        return max(0, min(freeze_epoch, self.num_epochs))

    def full_backward_flops(self) -> float:
        """Total backward FLOPs if no layer is ever frozen (baseline)."""
        per_step = sum(
            c.weight_grad_flops + c.act_grad_flops for c in self.layer_costs.values()
        )
        return per_step * self.steps_per_epoch * self.num_epochs

    def progressive_backward_flops(self, level: int = 1) -> float:
        """Total backward FLOPs actually paid under the freeze schedule."""
        self._check_level(level)
        total = 0.0
        for idx, cost in self.layer_costs.items():
            active = self._active_epochs(idx)
            frozen = self.num_epochs - active
            active_cost = cost.weight_grad_flops + cost.act_grad_flops
            # Frozen: Level 1 skips weight grad (act grad still flows);
            # Level 2 skips both (suffix cut).
            frozen_cost = cost.act_grad_flops if level == 1 else 0.0
            total += active_cost * active + frozen_cost * frozen
        return total * self.steps_per_epoch

    def reduction_rate(self, level: int = 1) -> float:
        """``1 − progressive / full`` (GOAL §5). Zero when full is zero."""
        full = self.full_backward_flops()
        if full == 0:
            return 0.0
        return 1.0 - self.progressive_backward_flops(level) / full

    def peak_vram_saved_bytes(self, level: int = 1) -> int:
        """Peak VRAM freed by the schedule (GOAL §5).

        Every layer that freezes during the run contributes its optimizer
        state (Level 1 onward). Under Level 2 its activation-gradient buffer
        is also no longer stored. Peak reflects the end-of-schedule state in
        which all scheduled layers are frozen.
        """
        self._check_level(level)
        saved = 0
        for idx, freeze_epoch in self.frozen_at_epoch.items():
            if freeze_epoch >= self.num_epochs:
                continue  # scheduled after the run ends: never frozen here
            cost = self.layer_costs[idx]
            saved += cost.optim_state_bytes
            if level == 2:
                saved += cost.act_grad_bytes
        return saved

    def summary(self, level: int = 1) -> FreezeCostSummary:
        """All GOAL §5 quantities for one freeze level, bundled."""
        self._check_level(level)
        full = self.full_backward_flops()
        progressive = self.progressive_backward_flops(level)
        return FreezeCostSummary(
            level=level,
            full_backward_flops=full,
            progressive_backward_flops=progressive,
            reduction_rate=0.0 if full == 0 else 1.0 - progressive / full,
            peak_vram_saved_bytes=self.peak_vram_saved_bytes(level),
        )
