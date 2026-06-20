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
matching forming the Phase 3 loss ablation
(``MSE`` vs ``MSE+cos`` vs distribution-consistency, GOAL §3.1 Phase 3). All
three arms are implemented below and the weighted combiner exposes them as
slots, so the ablation is a weight change, not new code.

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


def distribution_matching_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """Distribution-consistency loss: match per-feature mean and covariance.

    The Phase 3 distribution arm (GOAL §3.1 Phase 3: ``分布も合わせる版``;
    design §6.2). MSE measures point-wise agreement and cosine measures
    per-vector direction; both are blind to the *joint* second-order structure
    (covariance, correlation) of the activation batch that a downstream layer
    sensitive to its input *distribution* — not exact values — relies on. This
    loss matches the batch's first two moments: the per-feature mean and the
    per-feature covariance matrix. It is, by construction, **permutation
    invariant** — it matches the batch distribution, not which sample pairs with
    which ``xin`` — so it is the complement to the per-sample MSE, selected when
    the proxy-loss limit design §6.2 warns about is the binding concern.

    The loss is the mean squared gap of the means plus the mean squared
    (Frobenius) gap of the covariance matrices, each averaged over its own axis
    so the magnitude does not scale with hidden size::

        mean_term = mean((mu_pred - mu_tgt)^2)            # over H features
        cov_term  = mean((Sigma_pred - Sigma_tgt)^2)       # over H x H entries
        loss      = mean_term + cov_term

    Zero iff the two batches share their mean and covariance — including the
    trivially matched case (``predicted == target``) and, honestly, any row
    permutation of the target (the distribution is unchanged). Moments are taken
    over the flattened data x token positions (the distribution's samples);
    ``mask`` collapses over the hidden axis (as in :func:`cosine_matching_loss`)
    to a per-position weight so padded positions do not pollute the distribution
    (GOAL §7, leak-free). The covariance is the population estimate (divide by
    the effective sample count), so a single effective row yields zero
    covariance — no NaN.

    This is pure tensor math (no model, no GPU kernel); the H x H covariance is
    the inherent cost of capturing joint structure and lives in the Phase 3
    GPU-experiment regime, like the rest of the Level 2 ablation.
    """
    _check_shapes(predicted, target)
    target = _align_target(predicted, target)
    hidden = predicted.shape[-1]
    p = predicted.reshape(-1, hidden)
    t = target.reshape(-1, hidden)
    if mask is None:
        weight = torch.ones(p.shape[0], device=p.device, dtype=p.dtype)
    else:
        # Mask broadcasts to the full activation shape, then collapses over the
        # hidden axis to a per-position weight (cosine's convention): a [N, T, 1]
        # mask becomes one weight per (data, token) position.
        weight = (
            mask.to(dtype=p.dtype)
            .broadcast_to(predicted.shape)
            .reshape(-1, hidden)
            .mean(dim=-1)
        )
    wsum = weight.sum().clamp_min(eps)

    def _moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = (x * weight.unsqueeze(-1)).sum(dim=0) / wsum
        centered = (x - mu) * weight.unsqueeze(-1)
        cov = centered.t() @ centered / wsum
        return mu, cov

    mu_p, cov_p = _moments(p)
    mu_t, cov_t = _moments(t)
    return (mu_p - mu_t).pow(2).mean() + (cov_p - cov_t).pow(2).mean()


@dataclass(frozen=True)
class ActivationMatchingLoss:
    """Weighted combination of MSE, cosine, and distribution matching losses.

    ``mse_weight=1, cosine_weight=0, dist_weight=0`` is the Phase 1 gate (pure
    MSE). Raising ``cosine_weight`` realises the ``MSE+cos`` arm of the Phase 3
    ablation; raising ``dist_weight`` realises the ``分布も合わせる版``
    distribution-consistency arm (GOAL §3.1 Phase 3, design §6.2) — all without
    touching the call sites: the ablation is a weight change, not new code. The
    distribution arm is permutation-invariant by design, so it complements
    rather than replaces the per-sample MSE.

    All weights are clamped non-negative and at least one must be positive: a
    zeroed combiner emits no gradient and must be rejected rather than
    silently stalling training (GOAL §7: a metric with no signal is not a
    signal).
    """

    mse_weight: float = 1.0
    cosine_weight: float = 0.0
    dist_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.mse_weight < 0 or self.cosine_weight < 0 or self.dist_weight < 0:
            raise ValueError(
                "matching weights must be non-negative, got "
                f"mse_weight={self.mse_weight}, "
                f"cosine_weight={self.cosine_weight}, "
                f"dist_weight={self.dist_weight}"
            )
        if self.mse_weight + self.cosine_weight + self.dist_weight <= 0:
            raise ValueError(
                "at least one matching weight must be positive "
                f"(got mse={self.mse_weight}, cosine={self.cosine_weight}, "
                f"dist={self.dist_weight})"
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
        if self.dist_weight > 0:
            loss = loss + self.dist_weight * distribution_matching_loss(
                predicted, target, mask=mask
            )
        return loss
