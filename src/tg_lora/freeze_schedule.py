"""Freeze-schedule planner for Progressive Freezing Phase 2 (GOAL §3.1 Phase 2).

GOAL §3.1 Phase 2 / docs/design/10_progressive_freezing.md §4.1 turn "design the
freeze schedule" into a sweep over three degrees of freedom:

    order    — which layer freezes next (3 candidates, design §5.3)
    depth    — how many layers to freeze (1 .. N, the frontier curve)
    timing   — when the first freeze lands and how freezes space out

This planner turns a ``(policy, depth, timing)`` request into the
``frozen_at_epoch: dict[int, int]`` map that
:class:`src.tg_lora.freeze_cost.FreezeCostAccountant` already consumes, so any
candidate schedule's backward-FLOPs / VRAM savings can be predicted *before* a
GPU run. That is what Phase 2's "valid_loss degradation vs FLOPs reduction"
frontier curve is built from.

Policies (the three GOAL §3.1 Phase 2 candidates)
-------------------------------------------------
output_first
    Freeze from the output side inward, one layer at a time (candidate 1).
    Produces a contiguous growing frozen suffix — the most certain compute cut
    (design §5.3, §7). GOAL's prior (strong layer-wise independence, cos≈0) is
    that this degrades least.
convergence_order
    Freeze layers in the order they reached directional stability (candidate 2).
    The caller supplies ``convergence_order`` from the regime/stability analysis.
    Passing a *shuffled* permutation reuses the identical code path as the
    GOAL §7 / design Phase 2 control-(ii) **random-order freeze surrogate** —
    the null baseline any real schedule must beat. No separate random branch.
compromise
    Output-side order, but each layer's freeze is deferred until its stability
    threshold is reached (candidate 3): ``max(nominal_epoch, stability_epoch)``.
    Can produce a non-contiguous suffix when a layer is slow to stabilize
    (design §5.3 notes that breaks the contiguous cut — an explicit trade-off).

All three are pure-Python and model-free; estimating per-layer stability
(the regime/stability analysis input) is a separate, [UNVERIFIED] step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_POLICIES: tuple[str, ...] = ("output_first", "convergence_order", "compromise")


@dataclass(frozen=True)
class FreezeScheduleConfig:
    """A Phase 2 freeze-schedule request.

    Parameters
    ----------
    active_layer_indices:
        Candidate layers eligible for freezing (the trainable LoRA layer set),
        any order — the planner re-sorts per policy. Duplicates are rejected.
    num_epochs:
        Total epochs in the run. A freeze whose epoch lands at ``>= num_epochs``
        never happens during this run and is dropped from the realized schedule
        (matching :class:`FreezeCostAccountant` semantics).
    max_depth:
        Upper bound on how many layers to freeze (the Phase 2 depth sweep).
        Realized depth may be smaller when freezes land past ``num_epochs`` or
        a compromise layer's stability pushes it out of the run.
    start_epoch:
        Epoch at which the first freeze lands (the Phase 2 timing degree of
        freedom — GOAL places it after cycle-6-style phase transition).
    spacing:
        Epochs between successive freezes (``>= 1``).
    policy:
        One of :data:`VALID_POLICIES`.
    convergence_order:
        Required iff ``policy == "convergence_order"``: the stability order.
        Must be unique layers drawn from ``active_layer_indices`` and contain at
        least ``max_depth`` entries.
    stability_epoch:
        Required iff ``policy == "compromise"``: layer -> earliest epoch it may
        freeze. Layers missing from the map get no extra delay (floor 0). Must
        reference only active layers.
    """

    active_layer_indices: list[int]
    num_epochs: int
    max_depth: int
    start_epoch: int
    spacing: int = 1
    policy: str = "output_first"
    convergence_order: tuple[int, ...] | None = None
    stability_epoch: dict[int, int] | None = None

    def __post_init__(self) -> None:
        if self.policy not in VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {VALID_POLICIES}, got {self.policy!r}"
            )

        if len(self.active_layer_indices) != len(set(self.active_layer_indices)):
            raise ValueError(
                f"active_layer_indices must be unique, got {self.active_layer_indices}"
            )
        if not self.active_layer_indices:
            raise ValueError("active_layer_indices must be non-empty")

        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.spacing < 1:
            raise ValueError(f"spacing must be >= 1, got {self.spacing}")
        if self.max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {self.max_depth}")
        if self.start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {self.start_epoch}")

        active_set = set(self.active_layer_indices)

        if self.policy == "convergence_order":
            if self.stability_epoch is not None:
                raise ValueError(
                    "stability_epoch is only meaningful under the 'compromise' "
                    "policy, not 'convergence_order'"
                )
            order = self.convergence_order
            if not order:
                raise ValueError(
                    "convergence_order is required for policy 'convergence_order'"
                )
            if len(set(order)) != len(order):
                raise ValueError(f"convergence_order must be unique, got {list(order)}")
            unknown = [idx for idx in order if idx not in active_set]
            if unknown:
                raise ValueError(
                    f"convergence_order references unknown layer(s) {unknown}"
                )
            if self.max_depth > len(order):
                raise ValueError(
                    f"max_depth ({self.max_depth}) exceeds convergence_order "
                    f"length ({len(order)})"
                )
        elif self.policy == "compromise":
            if self.convergence_order is not None:
                raise ValueError(
                    "convergence_order is only meaningful under the "
                    "'convergence_order' policy, not 'compromise'"
                )
            stab = self.stability_epoch
            if stab is not None:
                unknown = [idx for idx in stab if idx not in active_set]
                if unknown:
                    raise ValueError(
                        f"stability_epoch references unknown layer(s) {unknown}"
                    )
                bad = [e for e in stab.values() if e < 0]
                if bad:
                    raise ValueError(f"stability_epoch values must be >= 0, got {bad}")
        else:  # output_first
            if self.convergence_order is not None:
                raise ValueError(
                    "convergence_order is only meaningful under the "
                    "'convergence_order' policy, not 'output_first'"
                )
            if self.stability_epoch is not None:
                raise ValueError(
                    "stability_epoch is only meaningful under the 'compromise' "
                    "policy, not 'output_first'"
                )

        # Depth upper bound: cannot freeze more layers than exist.
        if self.policy != "convergence_order" and self.max_depth > len(
            self.active_layer_indices
        ):
            raise ValueError(
                f"max_depth ({self.max_depth}) exceeds active layer count "
                f"({len(self.active_layer_indices)})"
            )


@dataclass(frozen=True)
class FreezeSchedule:
    """A realized Phase 2 freeze schedule (the planner's output).

    Attributes
    ----------
    config:
        The request this schedule was planned from.
    frozen_at_epoch:
        Layer index -> 0-based epoch at which it freezes (and stays frozen).
        Only freezes landing within ``[0, num_epochs)`` appear; this dict feeds
        :class:`FreezeCostAccountant` directly. A layer absent here never
        freezes during the run.
    order:
        Realized freeze order as a tuple of layer indices sorted by freeze
        epoch (ascending). Empty when nothing freezes.
    """

    config: FreezeScheduleConfig
    frozen_at_epoch: dict[int, int] = field(default_factory=dict)
    order: tuple[int, ...] = ()

    @property
    def realized_depth(self) -> int:
        """Number of layers that actually freeze during the run."""
        return len(self.frozen_at_epoch)

    @classmethod
    def plan(cls, config: FreezeScheduleConfig) -> "FreezeSchedule":
        """Resolve ``config`` into a realized :class:`FreezeSchedule`.

        The candidate freeze order is fixed per policy, truncated to
        ``max_depth``, then each candidate is assigned a nominal epoch
        ``start_epoch + rank * spacing`` (rank = position in the order). Under
        ``compromise`` that epoch is raised to the layer's stability floor.
        Candidates whose final epoch is ``>= num_epochs`` are dropped — they
        would land after the run ends.
        """
        ordered = _resolve_order(config)
        truncated = ordered[: config.max_depth]
        stability = config.stability_epoch or {}
        is_compromise = config.policy == "compromise"

        frozen: dict[int, int] = {}
        for rank, layer in enumerate(truncated):
            nominal = config.start_epoch + rank * config.spacing
            if is_compromise:
                epoch = max(nominal, stability.get(layer, 0))
            else:
                epoch = nominal
            if epoch < config.num_epochs:
                frozen[layer] = epoch

        order = tuple(sorted(frozen, key=lambda idx: frozen[idx]))
        return cls(config=config, frozen_at_epoch=frozen, order=order)


def _resolve_order(config: FreezeScheduleConfig) -> tuple[int, ...]:
    """Full candidate freeze order for ``config``'s policy (pre-truncation).

    ``output_first`` / ``compromise`` freeze output-side-first (descending
    layer index); ``convergence_order`` uses the caller-supplied stability
    sequence.
    """
    if config.policy == "convergence_order":
        assert config.convergence_order is not None  # validated in __post_init__
        return tuple(config.convergence_order)
    # output_first and compromise both descend from the output side.
    return tuple(sorted(config.active_layer_indices, reverse=True))
