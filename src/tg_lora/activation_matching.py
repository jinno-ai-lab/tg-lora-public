"""Activation-matching local loss for Progressive Freezing Level 2.

Once layer ``X`` is frozen, the front layer ``X-1`` is no longer driven by the
final task loss propagating through ``X`` (the backward-graph suffix is cut).
Instead it is trained against a *local* target: ``xin(X)`` — the input
activation that ``X`` was receiving at the moment it froze, cached earlier by
:func:`ProgressiveFreezeController.cache_xin`. The gap between ``X-1``'s
current output and that cached ``xin`` is the learning signal that keeps the
front layers improving after the suffix is cut
(GOAL §1.6.1, docs/design/10_progressive_freezing.md §3, §8 item 3).

GOAL calls for **MSE first** (the Phase 1 gate) with cosine / distribution
matching reserved for the Phase 3 loss ablation
(``MSE`` vs ``MSE+cos`` vs distribution-consistency, GOAL §3.1 Phase 3). The
weighted combiner below exposes those slots so the ablation is a weight
change, not new code.

This module is pure tensor math: no model, no GPU kernel. Its only inputs are
the front layer's current output (``predicted``) and the cached ``xin``
(``target``), paired per data point. Exactness is verifiable by hand-computed
unit tests (GOAL §7: verify the mechanism before trusting downstream runs).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_EPS = 1e-8


def _check_shapes(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must have identical shapes, got "
            f"{tuple(predicted.shape)} vs {tuple(target.shape)}"
        )


def _align_target(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Move cached xin onto the activation's device/precision.

    ``cache_xin`` stores ``xin`` detached on CPU (float32). The front layer's
    output may live on GPU at a different dtype (e.g. bf16 attention path,
    GOAL §1.5). Computing the loss in the activation's working precision keeps
    the gradient on the right device without an explicit caller cast.
    """
    return target.to(device=predicted.device, dtype=predicted.dtype)


def mse_matching_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error between front-layer output and cached ``xin``.

    The Phase 1 local loss (GOAL §3.1 Phase 1: "learn the preceding layer
    against ``xin``, starting with MSE"). Point-wise agreement — the cheapest
    proxy for "the fixed downstream layers still get the input they expect".

    Parameters
    ----------
    predicted:
        Current output of the front layer ``X-1``. Shape ``[N, ...]`` where
        ``N`` is the number of data points and ``...`` is typically
        ``[T, H]`` (token × hidden).
    target:
        Cached ``xin`` for the same ``N`` data points (same shape). Paired
        per data point: ``predicted[i]`` matches ``target[i]``.
    mask:
        Optional weight broadcastable to ``predicted`` (e.g. ``[N, T, 1]`` or
        ``[N, T]``). ``1`` keeps an element, ``0`` drops it. Used to exclude
        padded token positions so padding activations do not corrupt the loss
        (GOAL §7: unified, leak-free evaluation conditions).

    Returns
    -------
    Scalar MSE. Zero iff ``predicted`` already reproduces ``target`` exactly —
    which is precisely the "no learning signal" state (design §3.3): the loss
    only carries signal because ``xin`` is a *past* observation that the
    still-training front layer no longer emits.
    """
    _check_shapes(predicted, target)
    target = _align_target(predicted, target)
    diff_sq = (predicted - target).pow(2)
    if mask is None:
        return diff_sq.mean()
    weight = mask.to(dtype=diff_sq.dtype).broadcast_to(diff_sq.shape)
    denom = weight.sum().clamp_min(_EPS)
    return (diff_sq * weight).sum() / denom


def cosine_matching_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """Directional alignment loss: ``1 − mean cosine`` (Phase 3 ablation).

    Complements MSE: MSE only measures point-wise agreement and misses
    higher-order activation structure (design §6.2). Penalising misaligned
    *directions* can recover task signal that MSE leaves on the table when
    the layer is sensitive to the input distribution rather than its exact
    values. Range ``[0, 2]``; ``0`` means perfect directional agreement.

    Cosine is computed along the last (hidden) axis, then averaged over the
    remaining (data × token) axes, weighted by ``mask`` when given. ``mask``
    broadcasts to ``predicted``'s shape (e.g. ``[N, T, 1]``); its hidden axis
    is collapsed to a per-token weight so it lines up with the per-token
    cosine.
    """
    _check_shapes(predicted, target)
    target = _align_target(predicted, target)
    cos = F.cosine_similarity(predicted, target, dim=-1, eps=eps)
    if mask is None:
        return 1.0 - cos.mean()
    # Mask broadcasts to the full activation shape, then collapses over the
    # hidden axis to match cos (which is per-token, one fewer dim).
    token_weight = mask.to(dtype=cos.dtype).broadcast_to(predicted.shape).mean(dim=-1)
    denom = token_weight.sum().clamp_min(eps)
    return 1.0 - (cos * token_weight).sum() / denom


@dataclass(frozen=True)
class ActivationMatchingLoss:
    """Weighted combination of MSE and cosine matching losses.

    ``mse_weight=1, cosine_weight=0`` is the Phase 1 gate (pure MSE). Raising
    ``cosine_weight`` realises the ``MSE+cos`` arm of the Phase 3 ablation
    without touching the call sites. Distribution-consistency variants remain
    a future Phase 3 addition (would add a ``dist_weight`` slot here).

    Both weights are clamped non-negative and at least one must be positive: a
    zeroed combiner emits no gradient and must be rejected rather than
    silently stalling training (GOAL §7: a metric with no signal is not a
    signal).
    """

    mse_weight: float = 1.0
    cosine_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.mse_weight < 0 or self.cosine_weight < 0:
            raise ValueError(
                "matching weights must be non-negative, got "
                f"mse_weight={self.mse_weight}, cosine_weight={self.cosine_weight}"
            )
        if self.mse_weight + self.cosine_weight <= 0:
            raise ValueError(
                "at least one matching weight must be positive "
                f"(got mse={self.mse_weight}, cosine={self.cosine_weight})"
            )

    def __call__(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.mse_weight * mse_matching_loss(predicted, target, mask=mask)
        if self.cosine_weight > 0:
            loss = loss + self.cosine_weight * cosine_matching_loss(
                predicted, target, mask=mask
            )
        return loss
