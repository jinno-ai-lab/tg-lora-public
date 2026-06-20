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

import math
from collections.abc import Iterable
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
class Level1RealizationRecord:
    """A measured in-vivo Level-1 realized reduction — the §6.2 ceiling's evidence.

    The default :data:`LEVEL1_REALIZED_REDUCTION_CEILING` (0.0) is the validated
    CPU-proxy in-vivo result: a freeze-only (``requires_grad=False``) schedule
    realizes ~0 backward reduction because the activation gradient still runs
    through the frozen layer. Both the §6.2 design (10_guard_experiment.md §6.2)
    and the ceiling's own comment anticipate that a future in-vivo run — e.g.
    under gradient checkpointing, where the frozen layer's forward recompute is
    skipped — may observe a *nonzero* Level-1 realization the gate should credit
    rather than silently over-trust. This record is the landing zone for that
    measurement: it carries the observed reduction, how many runs observed it,
    and a free-text ``source`` (e.g. ``"9b_cuda_grad_ckpt"``) so any credit it
    unlocks is fully auditable instead of a hand-edited magic constant.

    A record below :data:`MIN_SAMPLE_FOR_CONFIDENCE_BAND` runs is thin evidence:
    it is recorded for the audit but does NOT raise the ceiling (see
    :func:`resolve_level1_ceiling`), so one or two runs of a nonzero number do
    not silently flip a Level-1 FAIL to a PASS — the same honesty bar the §6.3
    band applies to a confidence band.
    """

    observed_reduction: float
    num_runs: int
    source: str = ""

    def __post_init__(self) -> None:
        if self.observed_reduction < 0.0:
            raise ValueError(
                f"observed_reduction must be non-negative, got {self.observed_reduction}"
            )
        if self.num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {self.num_runs}")

    @property
    def is_thin_evidence(self) -> bool:
        """Too few runs to raise the ceiling (matches the §6.3 sample bar)."""
        return self.num_runs < MIN_SAMPLE_FOR_CONFIDENCE_BAND


def resolve_level1_ceiling(
    record: Level1RealizationRecord | None = None,
) -> float:
    """The Level-1 realized-reduction ceiling the §7 gate credits.

    With no record (the default), returns the validated
    :data:`LEVEL1_REALIZED_REDUCTION_CEILING` (0.0): the CPU-proxy in-vivo result
    that holds Level-1 realization at ~0 until a real measurement says
    otherwise. With a record that clears the thin-evidence bar, returns its
    observed reduction so the gate credits the measured realization — the
    "raise the ceiling to recover it" path the §6.2 design names. A thin record
    is recorded (callers keep it for the audit) but does not raise the ceiling:
    one or two reproductions of a nonzero number are not enough to credit a
    Level-1 reduction, for the same reason the §6.3 band refuses to call two
    reproductions a confidence band.
    """
    if record is None or record.is_thin_evidence:
        return LEVEL1_REALIZED_REDUCTION_CEILING
    return record.observed_reduction


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
    *,
    level1_ceiling: float | None = None,
) -> RealizedReduction:
    """Cap a freeze reduction at what is empirically realized in vivo.

    Level-2 reductions pass through unchanged — the trio's suffix cut is
    realized exactly (tests/test_progressive_freeze_invivo.py). Level-1
    reductions are capped at the Level-1 realized-reduction ceiling: the
    weight-grad FLOPs the accountant credits never become fewer backward
    traversals, so crediting them would over-trust a number the in-vivo suite
    shows to be an overstatement (see 10_guard_experiment.md §6.2).

    ``level1_ceiling`` overrides the default
    :data:`LEVEL1_REALIZED_REDUCTION_CEILING` (0.0) with a measured one — the
    resolved value of a :class:`Level1RealizationRecord` via
    :func:`resolve_level1_ceiling`. The realized reduction is the smaller of the
    ceiling and the arithmetic proxy, so a measurement recovers a Level-1
    reduction up to what the arithmetic says is possible, never beyond it.
    ``None`` keeps the validated 0.0 default, so every existing verdict is
    unchanged unless a caller supplies evidence.
    """
    proxy = accountant.reduction_rate(level=level)
    if level1_ceiling is not None and level1_ceiling < 0.0:
        raise ValueError(
            f"level1_ceiling must be non-negative, got {level1_ceiling}"
        )
    if level == 1:
        ceiling = (
            LEVEL1_REALIZED_REDUCTION_CEILING
            if level1_ceiling is None
            else level1_ceiling
        )
        realized = min(ceiling, proxy)
    else:
        realized = proxy
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
    level1_ceiling: float | None = None,
) -> GatedReduction:
    """Correct the accountant's reduction for realizability, then for width.

    Composes the model-free accountant with the realizability correction
    (:func:`realizable_reduction`, removes the Level-1 overstatement) and the
    width-extrapolation confidence so the Guard acceptance gate
    (10_guard_experiment.md §6.1 / §6.2 / §7) credits a reduction only to the
    extent it is *realized in vivo* and the target width has been validated —
    instead of trusting a proxy number that is either unrealized (Level 1) or
    unvalidated at scale.

    ``level1_ceiling`` is threaded straight through to
    :func:`realizable_reduction`, so a measured Level-1 realization (resolved
    from a :class:`Level1RealizationRecord`) propagates into the gate's credited
    reduction. It is a no-op for Level 2.
    """
    realized = realizable_reduction(accountant, level, level1_ceiling=level1_ceiling)
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


# §7 first-gate speed bar (10_guard_experiment.md §7): the Guard headline claim is
# ">=10% wall-clock shortening". When that bar must be judged from proxy FLOP
# accounting (CUDA/scale unavailable), :func:`speed_gate_verdict` credits a
# reduction only as far as it is realized in vivo (§6.2) and validated at the
# target width (§6.1) — never the raw proxy number.
SPEED_GATE_THRESHOLD: float = 0.10

# Graduated verdict labels emitted by :func:`speed_gate_verdict`. A bare boolean
# (``GatedReduction.passes``) cannot express the honesty gradation below, which is
# exactly what stops a gate from silently trusting a proxy number.
VERDICT_PASS = "PASS"
VERDICT_PROVISIONAL_PASS = "PROVISIONAL_PASS"
VERDICT_FAIL = "FAIL"
VERDICT_REQUIRES_SCALE_MEASUREMENT = "REQUIRES_SCALE_MEASUREMENT"


@dataclass(frozen=True)
class SpeedGateVerdict:
    """The §7 first-gate verdict judged from proxy FLOP accounting.

    Wraps :func:`gate_reduction` in the verdict an acceptance gate actually
    emits, with provenance, so the proxy path of the speed gate
    (10_guard_experiment.md §7) is concrete and testable rather than prose. The
    categorization is graduated so the gate never silently trusts a proxy number:

    * :data:`VERDICT_FAIL` — nothing realizable to credit (e.g. a Level-1 freeze,
      whose reduction is ~0 in vivo), or a real reduction below the bar.
    * :data:`VERDICT_REQUIRES_SCALE_MEASUREMENT` — a *real* reduction at a width
      so far outside the validated envelope that no PASS/FAIL may be drawn from
      the proxy; a CUDA/scale run is mandatory.
    * :data:`VERDICT_PROVISIONAL_PASS` — clears the bar after the width discount,
      but at a width only partly validated (e.g. 9B, 2x); PASSes the gate yet is
      explicitly provisional, not a silent full credit of the proxy number.
    * :data:`VERDICT_PASS` — clears the bar at a fully validated width.
    """

    verdict: str
    target_width: int
    proxy_reduction: float
    realized_reduction: float
    effective_reduction: float
    confidence: ExtrapolationConfidence
    threshold: float

    @property
    def requires_scale_measurement(self) -> bool:
        return self.confidence.requires_scale_measurement

    @property
    def passes(self) -> bool:
        """True only for PASS / PROVISIONAL_PASS (consistent with GatedReduction)."""
        return self.verdict in (VERDICT_PASS, VERDICT_PROVISIONAL_PASS)


def speed_gate_verdict(
    accountant: FreezeCostAccountant,
    level: int,
    target_width: int,
    *,
    threshold: float = SPEED_GATE_THRESHOLD,
    validated_max_width: int = PROXY_VALIDATED_MAX_WIDTH,
    scale_measurement_floor: float = 0.5,
    level1_ceiling: float | None = None,
) -> SpeedGateVerdict:
    """Judge the §7 speed gate from proxy FLOP accounting (the CUDA-less path).

    Composes :func:`gate_reduction` (realizability §6.2 + width §6.1) into the
    acceptance verdict. A proxy reduction clears the 10% bar only as far as it is
    *realized* (Level-1 → 0) and *validated* (width-discounted): a 9B target
    (h=4096) that still clears the bar PASSes *provisionally* rather than
    silently, and a 4x-width target (h≥8192) is refused pending a real
    measurement. A Level-1 reduction is a width-independent FAIL because its
    realized reduction is ~0 in vivo regardless of width — *unless* a measured
    Level-1 realization is supplied via ``level1_ceiling`` (resolved from a
    :class:`Level1RealizationRecord`), in which case the gate credits it and the
    verdict may recover from FAIL to PASS / PROVISIONAL_PASS at the measured
    value.
    """
    gated = gate_reduction(
        accountant,
        level,
        target_width,
        validated_max_width=validated_max_width,
        scale_measurement_floor=scale_measurement_floor,
        level1_ceiling=level1_ceiling,
    )
    if gated.realized_reduction <= 0.0:
        # Nothing realizable to credit (Level-1, or a schedule that freezes
        # nothing): FAIL at every width — a realizability failure, not a width one.
        verdict = VERDICT_FAIL
    elif gated.requires_scale_measurement:
        verdict = VERDICT_REQUIRES_SCALE_MEASUREMENT
    elif gated.effective_reduction < threshold:
        verdict = VERDICT_FAIL
    elif gated.confidence.confidence >= 1.0:
        verdict = VERDICT_PASS
    else:
        verdict = VERDICT_PROVISIONAL_PASS
    return SpeedGateVerdict(
        verdict=verdict,
        target_width=target_width,
        proxy_reduction=gated.proxy_reduction,
        realized_reduction=gated.realized_reduction,
        effective_reduction=gated.effective_reduction,
        confidence=gated.confidence,
        threshold=threshold,
    )


def format_speed_gate_verdict(verdict: SpeedGateVerdict) -> str:
    """Render a §7 proxy speed-gate verdict with full provenance.

    A compact, deterministic audit block an acceptance gate (or a reader of the
    experiment's ``gate_decision.txt``) can inspect: the verdict category, the
    target width, the width confidence and the bar, and the raw proxy / realized
    / effective reductions. The raw proxy figure stays visible for transparency,
    while the text makes clear the gate credits only the realized-and-discounted
    ``effective_reduction`` — never the raw proxy. This is the honesty gradation
    a bare ``passes`` boolean cannot carry, made concrete and inspectable rather
    than prose (10_guard_experiment.md §7).
    """
    return (
        f"speed_gate_verdict: {verdict.verdict} (passes={verdict.passes})\n"
        f"  target_width={verdict.target_width} "
        f"confidence={verdict.confidence.confidence:.3f} "
        f"threshold={verdict.threshold:.2f}\n"
        f"  proxy_reduction={verdict.proxy_reduction:.4f} "
        f"realized_reduction={verdict.realized_reduction:.4f} "
        f"effective_reduction={verdict.effective_reduction:.4f}"
    )


# Level-1-vs-Level-2 quantitative comparison (GOAL §5 / §1.6.3 / Phase 3). The
# accountant computes a separate ``reduction_rate`` per freeze level; this
# bundles the two into the comparison the constitution names — Level 1 (the
# established progressive-freeze baseline: weight-grad stop, final loss still
# propagates) versus Level 2 (the activation-matching suffix cut: the advanced
# experiment that additionally stops the activation gradient). The marginal
# reductions — the *extra* arithmetic / realized / effective savings the suffix
# cut buys on top of Level 1 — are the quantity GOAL Phase 3 weighs against
# Level 2's proxy-loss quality risk (§1.6.5). Because Level 2 skips a superset
# of the work Level 1 skips, every marginal delta is non-negative by
# construction.

@dataclass(frozen=True)
class LevelComparison:
    """Level-1 vs Level-2 freeze-savings comparison (GOAL §5, §1.6.3, Phase 3).

    Bundles the §7 verdict for each level over one schedule at one target width,
    plus the marginal reductions Level 2's suffix cut adds on top of the
    established Level 1 baseline. Level 1 (progressive freeze, weight-grad stop)
    is the mature, certain baseline (GOAL §1.6.3: "実装が枯れていて確実");
    Level 2 (the suffix cut / activation-matching trio) is the advanced
    experiment that additionally stops the activation gradient, buying more
    arithmetic savings but at proxy-loss consistency risk ("代理ロスの整合性
    リスクあり"). The ``additional_*`` fields make the extra cut explicit rather
    than two numbers a reader must subtract by hand.

    Under the :data:`LEVEL1_REALIZED_REDUCTION_CEILING`, Level 1 realizes ~0
    backward reduction in vivo at every width, so
    :attr:`additional_realized_reduction` equals Level 2's full realized
    reduction: the suffix cut is the only thing carrying realizable backward
    reduction, which is exactly why GOAL §1.6.3 treats Level 2 as an experiment
    rather than the production path. :attr:`level1_ceiling` records the ceiling
    actually credited (the validated 0.0 default, or a measured value resolved
    from a supplied :class:`Level1RealizationRecord`), so the audit shows when a
    real measurement recovered a Level-1 reduction rather than the baseline ~0.
    :attr:`reproduction_bracket` carries the across-reproduction §6.3 bracket on
    :attr:`additional_realized_reduction` when N A/B reproductions deposit their
    measured headline reductions via a :class:`ReproductionRecord`. ``None`` (the
    default) leaves the headline a point estimate, so the comparison is unchanged
    unless real evidence is supplied — the same landing-point contract as the
    §6.2 ceiling, distinct from it: the ceiling bounds *realizability* (§6.2),
    the bracket bounds *measurement spread* (§6.3).
    """

    target_width: int
    level1: SpeedGateVerdict
    level2: SpeedGateVerdict
    additional_arithmetic_reduction: float
    additional_realized_reduction: float
    additional_effective_reduction: float
    level1_ceiling: float = LEVEL1_REALIZED_REDUCTION_CEILING
    reproduction_bracket: ConfidenceBand | None = None

    @property
    def additional_passes(self) -> bool:
        """Level 2 clears the §7 bar where the Level 1 baseline does not.

        True only when Level 2 PASSes / PROVISIONAL_PASSes and Level 1 does not:
        the case in which the suffix cut's extra realization is what actually
        carries the gate. A Level 2 PASS here means the activation-matching cut
        is the sole thing producing realizable backward reduction at this width
        (Level 1 realizes ~0 under the §6.2 ceiling), so the experiment's extra
        cost and proxy-loss risk are the price of that realized reduction.
        """
        return self.level2.passes and not self.level1.passes


def compare_freeze_levels(
    accountant: FreezeCostAccountant,
    target_width: int,
    *,
    threshold: float = SPEED_GATE_THRESHOLD,
    validated_max_width: int = PROXY_VALIDATED_MAX_WIDTH,
    scale_measurement_floor: float = 0.5,
    level1_record: Level1RealizationRecord | None = None,
    reproduction_record: ReproductionRecord | None = None,
) -> LevelComparison:
    """Quantitative Level-1-vs-Level-2 comparison (GOAL §5 / Phase 3 deliverable).

    Builds the §7 speed-gate verdict for each level over the same schedule and
    target width (so both are judged under the identical §6.1 width bound and
    §6.2 realizability correction), then derives the marginal reductions Level
    2's suffix cut buys on top of Level 1 from the two verdicts' provenance
    fields. Level 2 skips a superset of the work Level 1 skips (weight-grad +
    activation-grad versus weight-grad alone), so every marginal delta is
    non-negative by construction; the comparison surfaces the *extra* realization
    the suffix cut delivers — the quantity the Phase-3 activation-matching
    experiment exists to weigh against its proxy-loss quality risk.

    Both verdicts share the same ``threshold`` / ``validated_max_width`` /
    ``scale_measurement_floor`` so the marginal numbers are a clean
    apples-to-apples delta, not a confound of differing gate settings.

    ``level1_record`` is the researcher-facing entry point for §6.2 evidence: a
    measured in-vivo Level-1 realization (resolved by
    :func:`resolve_level1_ceiling`) that raises the Level-1 ceiling above the
    validated 0.0 default. A non-thin record recovers a Level-1 reduction and
    may flip its verdict from FAIL; a thin record is recorded on the comparison
    but does not raise the ceiling. ``None`` keeps the validated baseline, so
    the comparison is unchanged unless real evidence is supplied.

    ``reproduction_record`` is the researcher-facing entry point for §6.3
    evidence: N measured across-reproduction observations of the headline
    :attr:`LevelComparison.additional_realized_reduction` (resolved by
    :func:`calibrate_reproduction_bracket` into a :class:`ConfidenceBand`). The
    bracket is an uncertainty report on the headline — it does not flip the §7
    verdict (whose honesty the width §6.1, realizability §6.2, and the verdict
    graduation already carry); it only stops the headline being presented as a
    bare point once reproductions exist. A non-thin record calibrates a band;
    a thin record (below :data:`MIN_SAMPLE_FOR_CONFIDENCE_BAND` reproductions) is
    labelled ``THIN_EVIDENCE`` rather than presented as calibrated. ``None``
    leaves the headline a point estimate with byte-identical output.
    """
    level1_ceiling = resolve_level1_ceiling(level1_record)
    level1 = speed_gate_verdict(
        accountant,
        level=1,
        target_width=target_width,
        threshold=threshold,
        validated_max_width=validated_max_width,
        scale_measurement_floor=scale_measurement_floor,
        level1_ceiling=level1_ceiling,
    )
    level2 = speed_gate_verdict(
        accountant,
        level=2,
        target_width=target_width,
        threshold=threshold,
        validated_max_width=validated_max_width,
        scale_measurement_floor=scale_measurement_floor,
        level1_ceiling=level1_ceiling,
    )
    return LevelComparison(
        target_width=target_width,
        level1=level1,
        level2=level2,
        additional_arithmetic_reduction=level2.proxy_reduction
        - level1.proxy_reduction,
        additional_realized_reduction=level2.realized_reduction
        - level1.realized_reduction,
        additional_effective_reduction=level2.effective_reduction
        - level1.effective_reduction,
        level1_ceiling=level1_ceiling,
        reproduction_bracket=calibrate_reproduction_bracket(reproduction_record),
    )


def format_level_comparison(comparison: LevelComparison) -> str:
    """Render the Level-1-vs-Level-2 comparison with the marginal cut explicit.

    The GOAL §5 / Phase 3 deliverable as a compact, deterministic audit block:
    both levels' verdicts and their arithmetic / realized / effective reductions,
    plus the extra reduction Level 2's suffix cut buys over the established
    Level 1 baseline. The ``additional`` line is the quantity the
    activation-matching experiment is run to earn — and because the §6.2 ceiling
    holds Level 1's in-vivo realization at ~0, the additional realized reduction
    is exactly Level 2's full realized reduction: the suffix cut is the only
    thing carrying realizable backward reduction in vivo.
    """
    l1, l2 = comparison.level1, comparison.level2
    lines = [
        f"level_comparison: target_width={comparison.target_width}",
        f"  level1 (progressive freeze): {l1.verdict} "
        f"arith={l1.proxy_reduction:.4f} realized={l1.realized_reduction:.4f} "
        f"effective={l1.effective_reduction:.4f}",
        f"  level2 (suffix cut):         {l2.verdict} "
        f"arith={l2.proxy_reduction:.4f} realized={l2.realized_reduction:.4f} "
        f"effective={l2.effective_reduction:.4f}",
        f"  additional (level2 - level1): "
        f"arith={comparison.additional_arithmetic_reduction:.4f} "
        f"realized={comparison.additional_realized_reduction:.4f} "
        f"effective={comparison.additional_effective_reduction:.4f}",
    ]
    if comparison.level1_ceiling > 0.0:
        # A measured in-vivo record recovered a Level-1 reduction (§6.2): make
        # the raised ceiling explicit in the audit, so the recovered credit is
        # never read as the validated ~0 baseline.
        lines.append(
            f"  level1 ceiling: raised to {comparison.level1_ceiling:.4f} "
            f"by a measured in-vivo record"
        )
    if comparison.reproduction_bracket is not None:
        # §6.3: across-reproduction bracket on the headline additional realized
        # reduction. A thin record (too few reproductions) is labelled
        # THIN_EVIDENCE and shows its count, not dressed as a calibrated bracket
        # — the same honesty rule as the per-level band. Omitted entirely when no
        # record is supplied, so the default output is byte-identical.
        band = comparison.reproduction_bracket
        label = "THIN_EVIDENCE" if band.is_thin_evidence else "calibrated"
        lines.append(
            f"  reproduction_bracket: {band.method} ({label}, n={band.n}) "
            f"lower={band.lower:.4f} upper={band.upper:.4f} "
            f"width={band.width:.4f}"
        )
    return "\n".join(lines)


# §6.3 variance-calibrated confidence band (10_guard_experiment.md §6.3). The §7
# verdict's two bounds (width §6.1, realizability §6.2) graduate a *point*
# reduction. This section records the *measured spread* of the realized
# reduction across cycles/runs and calibrates a band whose width comes from that
# variance — instead of presenting a single number or checking containment
# against a static analytic envelope. The steering feedback named the failure
# mode this retires: "two reproductions of a median is thin evidence to call it
# a confidence band"; :data:`MIN_SAMPLE_FOR_CONFIDENCE_BAND` makes that an
# enforced, auditable rule rather than prose.
MIN_SAMPLE_FOR_CONFIDENCE_BAND: int = 3

# How :func:`calibrate_reduction_band` derives its width from a sample.
CALIBRATION_EMPIRICAL_ENVELOPE: str = "empirical_envelope"
CALIBRATION_NORMAL: str = "normal"


@dataclass(frozen=True)
class ReductionSample:
    """Observed realized-reduction values with their summary statistics.

    Holds the raw observations (one per cycle, per run, or per measurement
    condition) plus the min / max / mean / stddev / N a confidence band is
    calibrated against. Unlike a single point reduction, this records the
    *measured spread*, so a band derived from it has a width grounded in
    observed variance rather than a static analytic envelope
    (10_guard_experiment.md §6.3).

    Reductions are non-negative fractions (a freeze cannot increase backward
    work), so the sample rejects a negative observation.
    """

    observations: tuple[float, ...]

    @classmethod
    def from_values(cls, values: Iterable[float]) -> ReductionSample:
        """Build a sample from any iterable of observed reductions."""
        obs = tuple(float(v) for v in values)
        if any(v < 0.0 for v in obs):
            raise ValueError(f"reductions must be non-negative, got {obs}")
        return cls(observations=obs)

    @classmethod
    def from_runs(cls, *series: Iterable[float]) -> ReductionSample:
        """Build a sample accumulating per-cycle series across runs (§6.3).

        Each positional ``series`` is one run's realized-reduction observations —
        e.g. the list :func:`per_cycle_realized_reductions` returns for that run's
        accountant. They flatten into one sample, so a :class:`ConfidenceBand`
        calibrated from it has its width grounded in measured spread *across
        runs*, not a single run's ramp. This is the across-runs calibration path
        the steering feedback named (``per_cycle_realized_reductions`` is per-run;
        accumulate its output here before building the band). The non-negative
        invariant is enforced over the combined series, as in :meth:`from_values`.
        """
        return cls.from_values(v for run in series for v in run)

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def is_empty(self) -> bool:
        return self.n == 0

    @property
    def is_thin_evidence(self) -> bool:
        """Too few observations to support a calibrated band at all.

        True below :data:`MIN_SAMPLE_FOR_CONFIDENCE_BAND`: a band calibrated
        from one or two reproductions is not a confidence band, so a gate must
        not present it as one (it collapses to the lone observation(s)).
        """
        return self.n < MIN_SAMPLE_FOR_CONFIDENCE_BAND

    @property
    def min(self) -> float:
        return min(self.observations) if self.observations else 0.0

    @property
    def max(self) -> float:
        return max(self.observations) if self.observations else 0.0

    @property
    def mean(self) -> float:
        if not self.observations:
            return 0.0
        return sum(self.observations) / self.n

    @property
    def stddev(self) -> float:
        """Sample standard deviation (ddof=1); 0.0 below two observations."""
        if self.n < 2:
            return 0.0
        mu = self.mean
        variance = sum((v - mu) ** 2 for v in self.observations) / (self.n - 1)
        return math.sqrt(variance)


@dataclass(frozen=True)
class ConfidenceBand:
    """A band over observed reductions whose width is calibrated to their spread.

    :attr:`lower` / :attr:`upper` come from the sample's measured variance
    (the full observed range, or a mean ± z·stddev normal interval), so the
    band width grows with what was actually observed — not a guess.
    :attr:`min_obs` / :attr:`max_obs` / :attr:`stddev` carry the *measured
    spread* the width was calibrated from, so a holder of the band sees the
    full ``min/max/stddev`` provenance (not just the calibrated interval) —
    the steering feedback's "min/max/stddev and N". For the normal method these
    are the raw observed range and its standard deviation, distinct from the
    ``lower`` / ``upper`` interval, so the audit shows both what was calibrated
    and what was measured.
    :attr:`is_thin_evidence` records when the sample was too small to support
    the band, so a gate never calls two reproductions of a median a
    "confidence band" (10_guard_experiment.md §6.3). This is an uncertainty
    report around the realized reduction; it does not flip the §7 verdict,
    whose honesty is already carried by the width (§6.1) and realizability
    (§6.2) bounds.
    """

    lower: float
    upper: float
    center: float
    half_width: float
    min_obs: float
    max_obs: float
    stddev: float
    n: int
    is_thin_evidence: bool
    method: str

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Whether ``value`` falls inside the calibrated band."""
        return self.lower <= value <= self.upper


def calibrate_reduction_band(
    sample: ReductionSample,
    *,
    method: str = CALIBRATION_EMPIRICAL_ENVELOPE,
    z: float = 1.96,
) -> ConfidenceBand:
    """Calibrate a confidence band against the sample's measured spread.

    ``method="empirical_envelope"`` (default) — the band is ``[min, max]`` over
    the observations: the full measured range, non-parametric and honest. Its
    width is exactly the observed spread, never a guess.

    ``method="normal"`` — ``mean ± z·stddev`` (``z=1.96`` ≈ a 95% normal
    interval over the realized reduction). Treats the observations as noisy
    measurements of one underlying reduction; needs ``n >= 2`` for a nonzero
    width. Because it is a symmetric normal interval it may dip below zero for
    a low-mean, high-variance sample — reductions are non-negative, so prefer
    the empirical envelope when that matters.

    Either way a sample smaller than :data:`MIN_SAMPLE_FOR_CONFIDENCE_BAND`
    yields a thin-evidence band (``is_thin_evidence=True``): the statistics are
    still computed for the record, but a gate must not present it as a
    calibrated confidence band — it collapses to the lone observation(s)
    (10_guard_experiment.md §6.3).
    """
    if method not in (CALIBRATION_EMPIRICAL_ENVELOPE, CALIBRATION_NORMAL):
        raise ValueError(
            f"method must be {CALIBRATION_EMPIRICAL_ENVELOPE!r} or "
            f"{CALIBRATION_NORMAL!r}, got {method!r}"
        )
    if sample.is_empty:
        raise ValueError("cannot calibrate a band from an empty sample")
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    if method == CALIBRATION_EMPIRICAL_ENVELOPE:
        lower, upper = sample.min, sample.max
    else:  # CALIBRATION_NORMAL
        half = z * sample.stddev
        lower, upper = sample.mean - half, sample.mean + half
    return ConfidenceBand(
        lower=lower,
        upper=upper,
        center=sample.mean,
        half_width=(upper - lower) / 2.0,
        min_obs=sample.min,
        max_obs=sample.max,
        stddev=sample.stddev,
        n=sample.n,
        is_thin_evidence=sample.is_thin_evidence,
        method=method,
    )


def per_cycle_realized_reductions(
    accountant: FreezeCostAccountant,
    level: int,
    *,
    num_cycles: int | None = None,
) -> list[float]:
    """Realized reduction as the freeze suffix grows, one value per cycle.

    For each cycle ``t`` in ``[0, num_cycles)``, restricts the accountant's
    schedule to the layers already frozen by cycle ``t`` (those with
    ``frozen_at_epoch <= t``) and reports
    ``realizable_reduction(...).realized_reduction`` — the in-vivo-realized
    reduction the observed suffix delivers as of that cycle. The returned
    series is the per-cycle observed spread a :class:`ConfidenceBand` is
    calibrated over (10_guard_experiment.md §6.3): it records how the realized
    reduction actually ramped across the run instead of collapsing it to one
    headline number. Accumulate across runs by concatenating series before
    building the sample.
    """
    # Validate level up front (matches FreezeCostAccountant._check_level) so an
    # invalid level raises even when num_cycles == 0 yields an empty series.
    if level not in _VALID_LEVELS:
        raise ValueError(f"level must be one of {_VALID_LEVELS}, got {level}")
    total = accountant.num_epochs if num_cycles is None else num_cycles
    if total < 0:
        raise ValueError(f"num_cycles must be non-negative, got {total}")
    series: list[float] = []
    for t in range(total):
        truncated = {
            idx: epoch
            for idx, epoch in accountant.frozen_at_epoch.items()
            if epoch <= t
        }
        acc_t = FreezeCostAccountant(
            layer_costs=accountant.layer_costs,
            steps_per_epoch=accountant.steps_per_epoch,
            num_epochs=accountant.num_epochs,
            frozen_at_epoch=truncated,
        )
        series.append(realizable_reduction(acc_t, level).realized_reduction)
    return series


def format_reduction_band(band: ConfidenceBand) -> str:
    """Render a §6.3 variance-calibrated band with full provenance.

    A compact, deterministic audit block: the calibration method, the sample
    size, the calibrated bounds and width, the measured spread the width came
    from, and the thin-evidence verdict. When ``is_thin_evidence`` the label
    states plainly that the band must not be read as a calibrated confidence
    band — the statistics are recorded for the audit, but two reproductions of
    a median do not earn the name.

    When the calibrated ``lower`` dips below zero it is flagged: reductions are
    non-negative (a freeze cannot increase backward work), so a symmetric
    ``normal`` interval on a low-mean, high-variance sample must not be read as
    attaining a negative reduction. The empirical envelope is ``[min, max]`` of
    non-negative observations, so the note only ever arises for the normal
    method and the default (empirical) output stays byte-identical.
    """
    label = "THIN_EVIDENCE" if band.is_thin_evidence else "calibrated"
    lines = [
        f"reduction_band: {band.method} ({label})",
        f"  n={band.n} lower={band.lower:.4f} upper={band.upper:.4f} "
        f"center={band.center:.4f} width={band.width:.4f}",
        f"  measured_spread: min={band.min_obs:.4f} max={band.max_obs:.4f} "
        f"mean={band.center:.4f} stddev={band.stddev:.4f}",
    ]
    if band.lower < 0.0:
        # The symmetric normal interval (mean ± z·stddev) can extend its lower
        # bound below the non-negative reduction floor; flag it so the audit
        # never reads a negative reduction as attainable (10_guard_experiment.md
        # §6.3). Read the effective floor as 0, not the printed negative bound.
        lines.append(
            f"  note: lower={band.lower:.4f} dips below the non-negative "
            f"reduction floor (symmetric normal interval); read as >= 0"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ReproductionRecord:
    """Across-reproduction observations of the A/B headline reduction (§6.3 bracket).

    One observed value per A/B reproduction of the Level-1-vs-Level-2 headline —
    the additional realized reduction (the backward work Level 2's suffix cut
    saves over the Level 1 baseline, measured in a real run). The §7 proxy
    comparison (:func:`compare_freeze_levels`) reports this headline as a point
    derived from model-free arithmetic; this record is the landing zone for the
    *measured spread* across N reproductions, so a future CUDA A/B run can deposit
    its per-run observations and the comparison reports an honest
    across-reproduction bracket instead of a bare point. The bracket does not
    flip the §7 verdict — it is an uncertainty report on the headline, the same
    role :class:`ConfidenceBand` plays for a single level's realized reduction
    (10_guard_experiment.md §6.3); the verdict's honesty is already carried by the
    width (§6.1) and realizability (§6.2) bounds.

    A record below :data:`MIN_SAMPLE_FOR_CONFIDENCE_BAND` reproductions is thin
    evidence: it is recorded for the audit (the count is shown) but the bracket
    is labelled ``THIN_EVIDENCE``, not presented as a calibrated confidence
    interval — two reproductions of a headline number are not a confidence
    interval, for the same reason :class:`ReductionSample` refuses to call one or
    two observations a confidence band. ``None`` (the default on
    :func:`compare_freeze_levels`) leaves the comparison a point estimate, so
    every existing verdict and its rendered output is unchanged unless real
    evidence is supplied — the same landing-point contract as
    :class:`Level1RealizationRecord`, orthogonal to it: the record bounds
    *realizability* (§6.2 ceiling); this record bounds *measurement spread*
    (§6.3 bracket).
    """

    observations: tuple[float, ...]
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.observations) < 1:
            raise ValueError(
                "a ReproductionRecord needs at least one observation; "
                "pass record=None for no evidence"
            )
        if any(v < 0.0 for v in self.observations):
            raise ValueError(
                f"reductions must be non-negative, got {self.observations}"
            )

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def sample(self) -> ReductionSample:
        """The observations as a :class:`ReductionSample` for band calibration."""
        return ReductionSample.from_values(self.observations)

    @property
    def is_thin_evidence(self) -> bool:
        """Too few reproductions to calibrate a bracket (matches the §6.3 bar)."""
        return self.n < MIN_SAMPLE_FOR_CONFIDENCE_BAND


def calibrate_reproduction_bracket(
    record: ReproductionRecord | None,
    *,
    method: str = CALIBRATION_EMPIRICAL_ENVELOPE,
    z: float = 1.96,
) -> ConfidenceBand | None:
    """The across-reproduction §6.3 bracket on the A/B headline, or ``None``.

    Returns ``None`` when there is no record, so :func:`compare_freeze_levels`
    left un-supplied with evidence stays a point estimate whose rendered output
    is byte-identical to before. With a record, the headline's measured spread
    across reproductions calibrates a :class:`ConfidenceBand` by reusing
    :func:`calibrate_reduction_band` over the record's sample: the empirical
    envelope ``[min, max]`` by default, or a ``mean ± z·stddev`` normal interval.

    A thin record still returns a band — but with ``is_thin_evidence=True`` — so
    :func:`format_level_comparison` labels it ``THIN_EVIDENCE`` (and shows the
    count) rather than presenting it as a calibrated bracket. This is the
    across-reproduction "thicken the N=2 bracket" landing point the steering
    feedback named: today's point headline becomes an honest,
    reproduction-counted bracket once a real A/B run deposits its observations.
    """
    if record is None:
        return None
    return calibrate_reduction_band(record.sample, method=method, z=z)


def frozen_at_epoch_from_freeze_log(
    frozen_layers_by_cycle: dict[int, int | set[int]],
) -> dict[int, int]:
    """Earliest epoch each layer froze, from a per-cycle freeze log.

    Accepts ``{cycle: layer_index}`` (one layer) or ``{cycle: {layers}}`` (a
    block), e.g. the guard controller's per-cycle ``guard_block_layers``. A
    layer's freeze epoch is the smallest cycle at which it is observed frozen;
    a layer that never appears is omitted (never frozen). The
    progressive-freeze invariant — a frozen layer stays frozen — is assumed, so
    observing a layer at cycle ``c`` means it is frozen on ``[c, end)``.

    This turns a real run's observed schedule into the ``frozen_at_epoch`` map a
    :class:`FreezeCostAccountant` (and thus :func:`speed_gate_verdict`) consumes,
    so the §7 proxy verdict can be judged from observed data instead of a
    hand-authored map.
    """
    result: dict[int, int] = {}
    for cycle, layers in sorted(frozen_layers_by_cycle.items()):
        layer_set = {layers} if isinstance(layers, int) else set(layers)
        for layer in layer_set:
            existing = result.get(layer)
            if existing is None or cycle < existing:
                result[layer] = cycle
    return result


def uniform_layer_accountant(
    num_layers: int,
    num_epochs: int,
    frozen_at_epoch: dict[int, int],
    *,
    steps_per_epoch: int = 1,
    weight_grad_flops: float = 1.0,
    act_grad_flops: float = 1.0,
) -> FreezeCostAccountant:
    """Homogeneous-stack first-order accountant for the §7 proxy path.

    Builds a :class:`FreezeCostAccountant` whose layers all carry the same
    backward cost — the natural first-order model for a homogeneous transformer
    block stack (the Qwen-9B target is 32 such blocks). Because
    :meth:`FreezeCostAccountant.reduction_rate` is a *ratio*, uniform costs give
    the exact first-order reduction for a schedule; real per-layer costs
    (DeltaNet vs. Attention, GOAL §1.5/§8) are the [UNVERIFIED] model-specific
    refinement, and they do not change the verdict's graduation, which
    :func:`speed_gate_verdict` already discounts for width (§6.1) and caps for
    realizability (§6.2).

    ``frozen_at_epoch`` keys must be layer indices in ``range(num_layers)``;
    layers absent from the map are never frozen. Pair with
    :func:`frozen_at_epoch_from_freeze_log` to build it from a run's observed
    schedule. Optimizer/activation-gradient *bytes* default to 0: the §7 speed
    bar is a FLOP ratio, so VRAM bytes are irrelevant to the verdict.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    layer_costs = {
        i: LayerBackwardCost(
            weight_grad_flops=weight_grad_flops,
            act_grad_flops=act_grad_flops,
        )
        for i in range(num_layers)
    }
    return FreezeCostAccountant(
        layer_costs=layer_costs,
        steps_per_epoch=steps_per_epoch,
        num_epochs=num_epochs,
        frozen_at_epoch=frozen_at_epoch,
    )
