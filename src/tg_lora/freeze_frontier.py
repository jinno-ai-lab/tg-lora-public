"""Phase 2 freeze-frontier predictor (GOAL §3.1 Phase 2 / design §4.1).

The Phase 2 deliverable is the frontier curve of *valid_loss degradation vs
backward-FLOPs reduction* (GOAL §3.1: "後段から1層、2層、…と段階的に深度を
増やし、各深度での valid_loss 劣化と FLOPs 削減をプロット。効率と性能の
フロンティア曲線を描く"). The valid_loss axis needs a GPU run (classification
C); the FLOPs-reduction axis is pure arithmetic. This module builds the
FLOPs-reduction axis *before* any run, by gluing two pieces that already exist
but were not connected in production code:

* :class:`src.tg_lora.freeze_schedule.FreezeSchedule` — turns a
  ``(policy, depth, timing)`` request into a ``frozen_at_epoch`` map.
* :class:`src.tg_lora.freeze_cost.FreezeCostAccountant` — turns a
  ``frozen_at_epoch`` map into GOAL §5 savings.

:func:`frontier` sweeps freeze depth ``0 .. N`` (× policies × levels) and
emits one :class:`FrontierPoint` per realized schedule, so the depth→FLOPs
frontier is one call. The sweep is exact (GOAL §7: verify the mechanism before
trusting a GPU run); per-layer costs still come from the caller.

Monotonicity (a property, not an assumption): deeper freeze only ever removes
backward work, so for a fixed ``(policy, level)`` the reduction_rate is
non-decreasing in depth — see ``test_frontier_reduction_monotonic_in_depth``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tg_lora.freeze_cost import FreezeCostAccountant, LayerBackwardCost
from src.tg_lora.freeze_schedule import (
    VALID_POLICIES,
    FreezeSchedule,
    FreezeScheduleConfig,
)

_VALID_LEVELS: tuple[int, int] = (1, 2)


@dataclass(frozen=True)
class FrontierPoint:
    """One point on the Phase 2 FLOPs-reduction frontier.

    Attributes
    ----------
    depth:
        Realized frozen depth — number of layers that actually freeze during
        this run under this schedule (``len(frozen_at_epoch)``). The
        *requested* depth (``max_depth``) may be larger; layers whose freeze
        lands past ``num_epochs`` are dropped by the planner.
    policy / level:
        Which schedule policy (GOAL §3.1 Phase 2 candidate) and freeze level
        (1 = weight-grad stop, 2 = activation-grad suffix cut, design §2) this
        point belongs to.
    reduction_rate:
        ``1 − progressive / full`` backward FLOPs (GOAL §5). Zero at depth 0.
    progressive_backward_flops / full_backward_flops:
        The two GOAL §5 FLOPs totals whose ratio is ``reduction_rate``.
    peak_vram_saved_bytes:
        Peak VRAM freed by the schedule at this depth (GOAL §5).
    frozen_at_epoch:
        The realized schedule this point was measured from — enough to replay
        the run without re-planning.
    """

    depth: int
    policy: str
    level: int
    reduction_rate: float
    progressive_backward_flops: float
    full_backward_flops: float
    peak_vram_saved_bytes: int
    frozen_at_epoch: dict[int, int]


@dataclass(frozen=True)
class FrontierSpec:
    """Inputs for a Phase 2 frontier sweep.

    ``layer_costs`` / ``steps_per_epoch`` / ``num_epochs`` feed the GOAL §5
    accountant; ``active_layer_indices`` / ``start_epoch`` / ``spacing`` plus
    the policy-specific inputs feed the planner. The accountant and every
    per-depth planner config share the same ``num_epochs``, and every active
    layer must appear in ``layer_costs`` (a frozen layer with no cost is a
    KeyError waiting to happen).
    """

    layer_costs: dict[int, LayerBackwardCost]
    steps_per_epoch: int
    num_epochs: int
    active_layer_indices: tuple[int, ...]
    start_epoch: int
    spacing: int = 1
    policies: tuple[str, ...] = ("output_first",)
    levels: tuple[int, ...] = (1, 2)
    convergence_order: tuple[int, ...] | None = None
    stability_epoch: dict[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.active_layer_indices:
            raise ValueError("active_layer_indices must be non-empty")
        if len(set(self.active_layer_indices)) != len(self.active_layer_indices):
            raise ValueError(
                f"active_layer_indices must be unique, got {list(self.active_layer_indices)}"
            )
        missing = [i for i in self.active_layer_indices if i not in self.layer_costs]
        if missing:
            raise ValueError(f"active layers missing from layer_costs: {missing}")
        if self.steps_per_epoch < 0:
            raise ValueError(
                f"steps_per_epoch must be non-negative, got {self.steps_per_epoch}"
            )
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.spacing < 1:
            raise ValueError(f"spacing must be >= 1, got {self.spacing}")
        if self.start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {self.start_epoch}")

        if not self.policies:
            raise ValueError("policies must be non-empty")
        unknown_policies = [p for p in self.policies if p not in VALID_POLICIES]
        if unknown_policies:
            raise ValueError(
                f"policies must be a subset of {VALID_POLICIES}, got {unknown_policies}"
            )
        if not self.levels:
            raise ValueError("levels must be non-empty")
        bad_levels = [lv for lv in self.levels if lv not in _VALID_LEVELS]
        if bad_levels:
            raise ValueError(
                f"levels must be a subset of {_VALID_LEVELS}, got {bad_levels}"
            )

        # convergence_order must cover the full active set: the sweep requests
        # max_depth up to len(active), and FreezeScheduleConfig rejects a depth
        # larger than the supplied order. Fail fast with one clear message.
        if "convergence_order" in self.policies:
            order = self.convergence_order
            if not order:
                raise ValueError(
                    "policy 'convergence_order' requires spec.convergence_order"
                )
            if len(set(order)) != len(order):
                raise ValueError(f"convergence_order must be unique, got {list(order)}")
            active = set(self.active_layer_indices)
            unknown = [i for i in order if i not in active]
            if unknown:
                raise ValueError(
                    f"convergence_order references non-active layer(s) {unknown}"
                )
            if len(order) < len(active):
                raise ValueError(
                    f"convergence_order length ({len(order)}) must cover the "
                    f"active set ({len(active)}) for a full depth sweep"
                )


def _build_config(spec: FrontierSpec, policy: str, depth: int) -> FreezeScheduleConfig:
    """One per-depth planner request for ``policy``."""
    kwargs: dict = {
        "active_layer_indices": list(spec.active_layer_indices),
        "num_epochs": spec.num_epochs,
        "max_depth": depth,
        "start_epoch": spec.start_epoch,
        "spacing": spec.spacing,
        "policy": policy,
    }
    if policy == "convergence_order":
        kwargs["convergence_order"] = spec.convergence_order
    elif policy == "compromise":
        kwargs["stability_epoch"] = spec.stability_epoch
    return FreezeScheduleConfig(**kwargs)


def _origin(spec: FrontierSpec, policy: str, level: int) -> FrontierPoint:
    """Depth-0 origin: no layer frozen, full backprop, zero savings."""
    full = FreezeCostAccountant(
        layer_costs=spec.layer_costs,
        steps_per_epoch=spec.steps_per_epoch,
        num_epochs=spec.num_epochs,
        frozen_at_epoch={},
    ).full_backward_flops()
    return FrontierPoint(
        depth=0,
        policy=policy,
        level=level,
        reduction_rate=0.0,
        progressive_backward_flops=full,
        full_backward_flops=full,
        peak_vram_saved_bytes=0,
        frozen_at_epoch={},
    )


def evaluate_schedule(
    spec: FrontierSpec, policy: str, depth: int, level: int = 1
) -> FrontierPoint:
    """Plan one schedule and measure its GOAL §5 savings.

    This is the production glue between :class:`FreezeSchedule` and
    :class:`FreezeCostAccountant` (the connection the Phase 2 frontier is
    built from). ``depth`` is the *requested* ``max_depth``; the returned
    point's ``depth`` is the *realized* frozen count, which may be smaller when
    freezes land past ``num_epochs``.
    """
    if level not in _VALID_LEVELS:
        raise ValueError(f"level must be one of {_VALID_LEVELS}, got {level}")
    schedule = FreezeSchedule.plan(_build_config(spec, policy, depth))
    summary = FreezeCostAccountant(
        layer_costs=spec.layer_costs,
        steps_per_epoch=spec.steps_per_epoch,
        num_epochs=spec.num_epochs,
        frozen_at_epoch=schedule.frozen_at_epoch,
    ).summary(level)
    return FrontierPoint(
        depth=schedule.realized_depth,
        policy=policy,
        level=level,
        reduction_rate=summary.reduction_rate,
        progressive_backward_flops=summary.progressive_backward_flops,
        full_backward_flops=summary.full_backward_flops,
        peak_vram_saved_bytes=summary.peak_vram_saved_bytes,
        frozen_at_epoch=dict(schedule.frozen_at_epoch),
    )


def frontier(spec: FrontierSpec) -> list[FrontierPoint]:
    """Sweep freeze depth ``0 .. N`` × ``policies`` × ``levels``.

    Returns one :class:`FrontierPoint` per realized schedule, plus a depth-0
    origin per ``(policy, level)`` series, sorted by ``(policy, level, depth)``
    so each series is a contiguous run a plotter can connect directly. ``N`` is
    ``len(active_layer_indices)`` — the frontier runs all the way to "freeze
    every active layer".
    """
    max_depth = len(spec.active_layer_indices)
    points: list[FrontierPoint] = []
    for policy in spec.policies:
        for level in spec.levels:
            points.append(_origin(spec, policy, level))
            for depth in range(1, max_depth + 1):
                points.append(evaluate_schedule(spec, policy, depth, level))
    return points
