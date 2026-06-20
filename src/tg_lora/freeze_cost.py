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

# Width envelope over which the *realized* freeze benefit has been validated.
# The accountant's reduction_rate is exact, model-free arithmetic and needs no
# width; but translating that arithmetic into a realized wall-clock / VRAM
# benefit at scale is validated only on proxy models up to this hidden width
# (see docs/design/10_progressive_freezing.md §7 [UNVERIFIED]; the in-vivo
# validation in tests/test_progressive_freeze_invivo.py runs at h=24). The
# acceptance gate (docs/design/10_guard_experiment.md §6.1 / §7) uses
# ``extrapolation_confidence`` to discount a proxy reduction at a larger target
# width rather than silently trusting a proxy number.
PROXY_VALIDATED_MAX_WIDTH: int = 2048


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


@dataclass(frozen=True)
class ExtrapolationConfidence:
    """How far a target model width sits outside the proxy-validated envelope.

    The accountant's ``reduction_rate`` is model-free arithmetic and needs no
    width; this bounds the *confidence* with which a proxy-validated reduction
    may be credited at a larger target width, so an acceptance gate cannot
    silently trust a number measured at ``h <= PROXY_VALIDATED_MAX_WIDTH`` when
    judging a larger run (e.g. a 9B model at h=4096).

    confidence
        1.0 inside the validated envelope, decaying as ``1 / extrapolation_ratio``
        beyond it (0.5 at 2x width, 0.25 at 4x).
    requires_scale_measurement
        True once ``confidence`` drops below the gate's floor — the target is
        then far enough outside the envelope that the gate must refuse to PASS
        on the proxy number alone and require a real CUDA/scale measurement.
    """

    target_width: int
    validated_max_width: int
    extrapolation_ratio: float
    confidence: float
    requires_scale_measurement: bool

    def discount(self, proxy_value: float) -> float:
        """A proxy reduction/VRAM value scaled to its creditable amount."""
        return proxy_value * self.confidence


def extrapolation_confidence(
    target_width: int,
    *,
    validated_max_width: int = PROXY_VALIDATED_MAX_WIDTH,
    scale_measurement_floor: float = 0.5,
) -> ExtrapolationConfidence:
    """Width-extrapolation confidence for a proxy-validated freeze benefit.

    ``confidence = min(1, validated_max_width / target_width)``: 1.0 inside the
    validated envelope, falling off as ``1 / extrapolation_ratio`` beyond it.
    ``requires_scale_measurement`` flips True once ``confidence`` falls to
    ``scale_measurement_floor`` (default 0.5 → 2x width), at which point an
    acceptance gate must not PASS on the proxy reduction alone.

    Rationale: the reduction is exact arithmetic, so it does not widen with
    width — but the *realized* benefit (wall-clock, fixed overheads, quality at
    scale) is only validated on proxies, and crediting it undiminished at a much
    larger width over-trusts a proxy number (see 10_guard_experiment.md §6.1).
    """
    if target_width <= 0:
        raise ValueError(f"target_width must be positive, got {target_width}")
    if validated_max_width <= 0:
        raise ValueError(
            f"validated_max_width must be positive, got {validated_max_width}"
        )
    if not 0.0 < scale_measurement_floor <= 1.0:
        raise ValueError(
            f"scale_measurement_floor must be in (0, 1], got {scale_measurement_floor}"
        )
    ratio = target_width / validated_max_width
    confidence = min(1.0, 1.0 / ratio)
    # Require a scale measurement only when confidence is strictly below the
    # floor; at exactly the floor the gate still PASSes provisionally.
    requires = confidence < scale_measurement_floor
    return ExtrapolationConfidence(
        target_width=target_width,
        validated_max_width=validated_max_width,
        extrapolation_ratio=ratio,
        confidence=confidence,
        requires_scale_measurement=requires,
    )


# A Level-1 (freeze-only, ``requires_grad=False``) schedule realizes ~0 backward
# reduction in vivo: the weight-grad FLOPs the accountant credits as saved never
# become fewer backward traversals, because the activation gradient still runs
# through the frozen layer. Validated empirically by
# tests/test_progressive_freeze_invivo.py ::
# test_accountant_level1_overstates_realizable_savings_in_vivo (Level-1 realized
# reduction is ~0 while the accountant reports > 0); Level 2 (the trio) is
# realized exactly. The speed gate must not credit a Level-1 reduction; this
# ceiling is empirically derived and may be raised if a future in-vivo run (e.g.
# under gradient checkpointing) shows nonzero Level-1 realization.
LEVEL1_REALIZED_REDUCTION_CEILING: float = 0.0


@dataclass(frozen=True)
class RealizedReduction:
    """A freeze reduction corrected for what is empirically realized in vivo.

    The accountant's ``reduction_rate`` is exact FLOP arithmetic, but not every
    credited FLOP becomes a saved backward traversal. Level 1 (freeze-only)
    still propagates the activation gradient through the frozen layer, so the
    weight-grad FLOPs it credits realize ~0 backward reduction; Level 2 (the
    trio: activation_cache + split_layer + dynamic_freeze) cuts the
    backward-graph suffix, so its reduction is realized exactly.
    """

    proxy_reduction: float
    realized_reduction: float

    @property
    def is_realized(self) -> bool:
        """True when the proxy reduction survives the realizability correction."""
        return self.realized_reduction > 0.0


def realizable_reduction(
    accountant: FreezeCostAccountant,
    level: int,
) -> RealizedReduction:
    """Cap a freeze reduction at what is empirically realized in vivo.

    Level-2 reductions pass through unchanged — the trio's suffix cut is
    realized exactly (tests/test_progressive_freeze_invivo.py). Level-1
    reductions are capped at :data:`LEVEL1_REALIZED_REDUCTION_CEILING`: the
    weight-grad FLOPs the accountant credits never become fewer backward
    traversals, so crediting them would over-trust a number the in-vivo suite
    shows to be an overstatement (see 10_guard_experiment.md §6.2).
    """
    proxy = accountant.reduction_rate(level=level)
    realized = LEVEL1_REALIZED_REDUCTION_CEILING if level == 1 else proxy
    return RealizedReduction(proxy_reduction=proxy, realized_reduction=realized)


@dataclass(frozen=True)
class GatedReduction:
    """A freeze reduction corrected for realizability, then discounted by width.

    What an acceptance gate may actually credit at the target width:
    :attr:`effective_reduction` is the *realized* reduction (the Level-1
    overstatement removed by :func:`realizable_reduction`) scaled by
    :attr:`confidence`. :attr:`proxy_reduction` keeps the raw accountant figure
    for transparency — it is *not* what the gate credits. When
    :attr:`requires_scale_measurement` is set, the effective value is provisional
    and the gate must not PASS on it alone.
    """

    proxy_reduction: float
    realized_reduction: float
    confidence: ExtrapolationConfidence
    effective_reduction: float

    @property
    def requires_scale_measurement(self) -> bool:
        return self.confidence.requires_scale_measurement

    def passes(self, threshold: float) -> bool:
        """Gate PASS only if the realized-and-discounted reduction clears
        ``threshold`` and no scale measurement is required (the target is
        inside, or near, the validated envelope). Otherwise the proxy number
        must not PASS the gate."""
        if self.requires_scale_measurement:
            return False
        return self.effective_reduction >= threshold


def gate_reduction(
    accountant: FreezeCostAccountant,
    level: int,
    target_width: int,
    *,
    validated_max_width: int = PROXY_VALIDATED_MAX_WIDTH,
    scale_measurement_floor: float = 0.5,
) -> GatedReduction:
    """Correct the accountant's reduction for realizability, then for width.

    Composes the model-free accountant with the realizability correction
    (:func:`realizable_reduction`, removes the Level-1 overstatement) and the
    width-extrapolation confidence so the Guard acceptance gate
    (10_guard_experiment.md §6.1 / §6.2 / §7) credits a reduction only to the
    extent it is *realized in vivo* and the target width has been validated —
    instead of trusting a proxy number that is either unrealized (Level 1) or
    unvalidated at scale.
    """
    realized = realizable_reduction(accountant, level)
    conf = extrapolation_confidence(
        target_width,
        validated_max_width=validated_max_width,
        scale_measurement_floor=scale_measurement_floor,
    )
    return GatedReduction(
        proxy_reduction=realized.proxy_reduction,
        realized_reduction=realized.realized_reduction,
        confidence=conf,
        effective_reduction=conf.discount(realized.realized_reduction),
    )
