"""GOAL §3.1 Phase 3 loss-ablation harness: switch the activation-matching
local loss between its three ablation arms on CPU, each arm's weighting
config-driven.

GOAL §3.1 Phase 3 (line 166) calls for the loss-function ablation::

    損失関数のアブレーション: MSE 単独 vs MSE+cos vs 分布も合わせる版

:mod:`src.tg_lora.activation_matching` already implements all three loss terms
(MSE / cosine / distribution) and a weighted combiner
(:class:`~src.tg_lora.activation_matching.ActivationMatchingLoss`) that exposes
them as weight slots — so *switching arms is a weight change, not new code*.
What was missing for the Phase-3 Category-A scaffold (GOAL §3.1 Phase 3:
"各 arm の重み付けを config 駆動にする") was:

1. **named arms with pinned canonical weights** — so "the ``mse_cos`` arm" means
   one weight vector, not whatever a caller happened to pass. The ablation is
   reproducible across runs only if the arm is a named, fixed point.
2. **config-driven weighting** — the weights come from a config value
   (:class:`LossArmConfig`), not hardcoded at each freeze call site. The named
   arm is a preset; explicit weight fields override it, so a GPU run re-pins an
   arm's weights without touching code.
3. **a side-by-side harness on CPU** — :func:`run_loss_ablation` runs every arm
   on the same ``(predicted, target)`` pair and returns each arm's scalar loss
   *and* per-term breakdown, so the ablation is *observable before any GPU run*
   (GOAL §7: observe the mechanism before trusting downstream runs).

This module is pure tensor math plus a config dataclass: no model, no GPU. The
Level-1-vs-Level-2 *quantitative* comparison is the Phase-3 GPU run
(classification C, GOAL §3.1 line 167); this harness is the Category-A scaffold
that turns that run into a one-line-per-arm switch when the GPU is free.

Arm naming (the canonical reading of GOAL line 166). MSE is always the base
(Phase-1 gate). The ``mse_cos`` and ``dist`` arms each add exactly one term over
that base — a standard factorial ablation isolating each term's marginal
contribution rather than a cumulative stack, so each of the three terms (none /
cosine / distribution) is individually testable:

* ``mse``     — MSE 単独           → ``(mse=1, cosine=0, dist=0)``
* ``mse_cos`` — MSE+cos            → ``(mse=1, cosine=1, dist=0)``
* ``dist``    — 分布も合わせる版    → ``(mse=1, cosine=0, dist=1)``

A different reading (e.g. the distribution arm stacked cumulatively on
``mse_cos``) is a config override away — :class:`LossArmConfig` exists so the
weighting is a stated, overridable value rather than an unstated assumption.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import torch

from src.tg_lora.activation_matching import (
    ActivationMatchingBreakdown,
    ActivationMatchingLoss,
)


@dataclass(frozen=True)
class LossWeights:
    """The ``(mse, cosine, dist)`` weight triple for one ablation arm.

    A thin, hashable record so a named arm's weighting is a value, not three
    loose floats scattered across a call site. Maps 1:1 onto
    :class:`~src.tg_lora.activation_matching.ActivationMatchingLoss`'s
    ``mse_weight`` / ``cosine_weight`` / ``dist_weight`` via
    :func:`build_matching_loss`.
    """

    mse: float = 1.0
    cosine: float = 0.0
    dist: float = 0.0


#: The three GOAL §3.1 Phase-3 ablation arms, in canonical order. MSE is always
#: the base (Phase-1 gate); the ``mse_cos`` and ``dist`` arms each add exactly
#: one term over it (factorial ablation isolating each term's contribution).
LOSS_ARMS: Mapping[str, LossWeights] = {
    "mse": LossWeights(mse=1.0, cosine=0.0, dist=0.0),  # MSE 単独
    "mse_cos": LossWeights(mse=1.0, cosine=1.0, dist=0.0),  # MSE+cos
    "dist": LossWeights(mse=1.0, cosine=0.0, dist=1.0),  # 分布も合わせる版
}
LOSS_ARM_NAMES: tuple[str, ...] = tuple(LOSS_ARMS)


@dataclass(frozen=True)
class LossArmConfig:
    """One Phase-3 ablation arm, resolved config-driven to a weight triple.

    ``arm`` names a preset in :data:`LOSS_ARMS`; the ``*_weight`` fields, when
    not ``None``, override that preset's slot. If ``arm`` is ``None`` the
    explicit weights *are* the config (defaults to pure MSE). This is the
    "each arm's weighting is config-driven" surface (GOAL §3.1 Phase 3): the
    freeze loop reads arm + overrides from config rather than hardcoding
    :class:`ActivationMatchingLoss` weights.

    Final non-negativity / "at least one positive" validation is delegated to
    :class:`ActivationMatchingLoss` (single source of truth) at build time.
    """

    arm: str | None = "mse"
    mse_weight: float | None = None
    cosine_weight: float | None = None
    dist_weight: float | None = None

    def __post_init__(self) -> None:
        if self.arm is not None and self.arm not in LOSS_ARMS:
            raise ValueError(
                f"unknown ablation arm {self.arm!r}; "
                f"expected one of {LOSS_ARM_NAMES} (or None for raw weights)"
            )

    def resolve_weights(self) -> LossWeights:
        """The arm preset with any explicit overrides applied."""
        base = LOSS_ARMS[self.arm] if self.arm is not None else LossWeights()
        overrides = {
            "mse": self.mse_weight,
            "cosine": self.cosine_weight,
            "dist": self.dist_weight,
        }
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})


def build_matching_loss(config: LossArmConfig) -> ActivationMatchingLoss:
    """Construct the weighted combiner for one config-driven ablation arm.

    The single bridge from :class:`LossArmConfig` (config surface) to
    :class:`ActivationMatchingLoss` (the differentiable training combiner).
    Weight validation (non-negative, at least one positive) happens here, in the
    combiner, so an impossible config is rejected with the same error a
    hand-built combiner would raise.
    """
    w = config.resolve_weights()
    return ActivationMatchingLoss(
        mse_weight=w.mse, cosine_weight=w.cosine, dist_weight=w.dist
    )


def _detach(x: torch.Tensor | float) -> torch.Tensor | float:
    """Detach a tensor (floats pass through) — harness results are for reading."""
    return x.detach() if isinstance(x, torch.Tensor) else x


@dataclass(frozen=True)
class LossArmResult:
    """One arm's outcome in an ablation sweep: weights, scalar loss, breakdown.

    Results are detached: :func:`run_loss_ablation` is an *observation* tool
    (compare arms side-by-side), not a training step. The differentiable path is
    :class:`ActivationMatchingLoss` directly. :attr:`breakdown` is byte-identical
    to what the combiner's ``breakdown()`` would return, only detached, so
    :attr:`loss` == :attr:`breakdown.total` holds exactly.
    """

    arm: str
    weights: LossWeights
    loss: torch.Tensor | float
    breakdown: ActivationMatchingBreakdown

    @property
    def label(self) -> str:
        """A stable human label: the named arm, or the resolved weight triple."""
        if self.arm in LOSS_ARMS:
            return self.arm
        return (
            f"custom(mse={self.weights.mse},cos={self.weights.cosine},"
            f"dist={self.weights.dist})"
        )

    def as_row(self) -> dict[str, float | str]:
        """Flat, loggable row: weights + scalar loss + each term's contribution."""
        return {
            "arm": self.label,
            "mse_weight": self.weights.mse,
            "cosine_weight": self.weights.cosine,
            "dist_weight": self.weights.dist,
            "loss": float(self.loss),
            "mse_term": float(self.breakdown.mse),
            "cosine_term": float(self.breakdown.cosine),
            "dist_term": float(self.breakdown.dist),
        }


@dataclass(frozen=True)
class LossAblationResult:
    """The full Phase-3 ablation sweep over one ``(predicted, target)`` pair."""

    arms: list[LossArmResult]

    def loss_by_arm(self) -> dict[str, torch.Tensor | float]:
        """``{arm_label: scalar_loss}`` — the arm-vs-arm comparison column."""
        return {r.label: r.loss for r in self.arms}

    def as_rows(self) -> list[dict[str, float | str]]:
        """One :meth:`LossArmResult.as_row` per arm — a comparable table."""
        return [r.as_row() for r in self.arms]


def _normalize_arm(arm: LossArmConfig | str) -> LossArmConfig:
    return arm if isinstance(arm, LossArmConfig) else LossArmConfig(arm=arm)


def run_loss_ablation(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    arms: Iterable[LossArmConfig | str] | None = None,
) -> LossAblationResult:
    """Run the Phase-3 loss-ablation arms side-by-side on CPU.

    Each arm in ``arms`` (default: the three GOAL §3.1 canonical arms) is built
    from its config, evaluated on the *same* ``(predicted, target[, mask])``, and
    recorded with its scalar loss and per-term breakdown — so the ablation is
    observable before any GPU run. ``arms`` accepts either
    :class:`LossArmConfig` objects (the config-driven path) or bare arm-name
    strings (convenience shorthand for the named presets).

    The arms share one forward pass worth of input; the difference between arms
    is purely which loss terms are active (GOAL §7: unified, leak-free
    conditions — every arm sees the identical pair). Results are detached.
    """
    if arms is None:
        arms = LOSS_ARM_NAMES
    configs = [_normalize_arm(a) for a in arms]
    if not configs:
        raise ValueError("run_loss_ablation requires at least one arm")

    results: list[LossArmResult] = []
    for cfg in configs:
        w = cfg.resolve_weights()
        loss_fn = build_matching_loss(cfg)
        bd = loss_fn.breakdown(predicted, target, mask=mask)
        detached = ActivationMatchingBreakdown(
            mse=_detach(bd.mse),
            cosine=_detach(bd.cosine),
            dist=_detach(bd.dist),
        )
        label = cfg.arm if cfg.arm in LOSS_ARMS else "custom"
        results.append(
            LossArmResult(arm=label, weights=w, loss=detached.total, breakdown=detached)
        )
    return LossAblationResult(arms=results)


__all__ = [
    "LOSS_ARMS",
    "LOSS_ARM_NAMES",
    "LossWeights",
    "LossArmConfig",
    "LossArmResult",
    "LossAblationResult",
    "build_matching_loss",
    "run_loss_ablation",
]
