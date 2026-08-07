"""§7 random-surrogate null baseline for the PSA per-tensor PC1 prior (GOAL §3.2).

PSA (``src/tg_lora/psa.py``) amplifies each LoRA-tensor gradient along the PC1
prior ``v_PSA`` extracted from the cross-cycle ΔW history::

    G' = G + gamma * <G, v_PSA> * v_PSA

That amplification is only *useful* if ``v_PSA`` captures more of the gradient's
directional energy than a random unit vector would by chance — otherwise PSA
injects gradient noise (the random-prior case) instead of reinforcing a real
direction. GOAL §7 mandates a random null baseline for every metric, and §4's
統計の歯止め recognizes only what beats the surrogate. The §4 arc applied this to
the freeze schedule (random-order surrogate) and ``layer_delta_analysis`` applied
it to rank-1 *eigenvalue* dominance (``rank1_z`` vs Marchenko-Pastur). This leaf
applies the SAME discipline to the PC1 *direction* PSA actually amplifies — the
prerequisite honesty gate before §3.2 PSA can be reactivated as the post-§4
research axis (the terminal verdict's named (C) pivot,
``docs/section4_terminal_verdict.md`` §4; elaborated in
``docs/psa_axis_research_question.md``).

The metric is the mean squared-cosine alignment between a *held-out* set of
gradients and a candidate direction, compared against the same quantity
averaged over random unit directions::

    prior_alignment     = mean_g [ <g, v_PSA>^2 / ||g||^2 ]
    surrogate_alignment = mean_{g, v_rand} [ <g, v_rand>^2 / ||g||^2 ]
    alignment_ratio     = prior_alignment / surrogate_alignment

Under the iid null (deltas AND held-out gradients both isotropic noise),
``alignment_ratio`` concentrates at 1.0 — the prior is no better than random and
PSA is a NULL. When the ΔW history carries a real spike that future gradients
also live along, ``alignment_ratio`` >> 1.0 — the prior carries signal. A high
rank-1 *eigenvalue* dominance (``layer_delta_analysis.rank1_z``) is necessary
but not sufficient: it says a spike exists, not that the spike's direction
aligns with the gradients PSA would amplify. This leaf closes that distinction.
"""

import torch

__all__ = ["prior_vs_surrogate_alignment", "random_unit_directions"]


def random_unit_directions(
    n: int,
    numel: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``n`` independent unit-norm directions on ``R^numel``.

    Mirrors the surrogate-construction pattern of
    ``src/analysis/extrapolation_predictability.py::_random_like_with_norm`` but
    normalizes to unit norm so the random baseline is directly comparable to the
    unit-norm PSA prior ``v_PSA`` (which ``PSAPrior.extract_priors`` renormalizes
    after every blend).
    """
    if n <= 0 or numel <= 0:
        return torch.empty(max(n, 0), max(numel, 0))
    directions = torch.randn(n, numel, generator=generator, dtype=torch.float32)
    norms = directions.norm(dim=1, keepdim=True)
    # Guard a (probability-zero) all-zero row so we never divide by ~0.
    norms = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
    return directions / norms


def prior_vs_surrogate_alignment(
    prior: torch.Tensor,
    grad_samples: torch.Tensor,
    n_surrogate: int = 64,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Measure whether the extracted PC1 prior beats a random-direction surrogate.

    Amplifying along a unit prior ``v`` adds squared energy ``gamma^2 * <g, v>^2``
    to gradient ``g``; the prior's *useful* captured-energy fraction is
    ``<g, v>^2 / ||g||^2``. We compare the prior's mean captured fraction against
    the same fraction averaged over ``n_surrogate`` random unit directions.

    Args:
        prior: 1-D tensor ``[numel]`` — the extracted ``v_PSA`` for one tensor.
            Normalized internally, so a non-unit caller input cannot inflate the
            metric (a measurement footgun).
        grad_samples: ``[N, numel]`` held-out gradients, DISJOINT from the ΔW
            history the prior was extracted from (GOAL §7 "項を共有しない
            ホールドアウト" — testing on the extraction data gives optimistic bias).
        n_surrogate: number of random unit directions averaged for the null.
        generator: optional ``torch.Generator`` for deterministic surrogates.

    Returns:
        ``{"prior_alignment", "surrogate_alignment", "alignment_ratio"}``.
        ``alignment_ratio`` ≈ 1.0 ⇒ prior is a NULL (PSA injects noise);
        ``alignment_ratio`` >> 1.0 ⇒ prior carries gradient-direction signal.
        Returns all zeros when no usable (non-zero-norm) gradient sample is
        provided, so an empty/zero input is a defined no-signal rather than NaN.
    """
    if prior.dim() != 1:
        raise ValueError(f"prior must be 1-D [numel], got shape {tuple(prior.shape)}")
    numel = prior.numel()
    if numel == 0:
        return {"prior_alignment": 0.0, "surrogate_alignment": 0.0, "alignment_ratio": 0.0}

    grads = grad_samples.to(torch.float32)
    if grads.dim() == 1:
        grads = grads.unsqueeze(0)
    if grads.shape[1] != numel:
        raise ValueError(
            f"grad_samples last dim ({grads.shape[1]}) must match prior numel ({numel})"
        )

    v = prior.to(torch.float32)
    v = v / (v.norm() + 1e-12)  # internal normalization: non-unit input can't inflate

    g_sq = (grads * grads).sum(dim=1)  # ||g||^2 per sample, [N]
    usable = g_sq > 1e-18  # drop ~zero-norm samples (undefined alignment)
    if usable.sum() == 0:
        return {"prior_alignment": 0.0, "surrogate_alignment": 0.0, "alignment_ratio": 0.0}
    g_usable = grads[usable]  # [M, numel]
    g_sq_usable = g_sq[usable]  # [M]

    # Prior captured-energy fraction: <g, v>^2 / ||g||^2, averaged over samples.
    proj_prior = (g_usable @ v) ** 2  # [M]
    prior_alignment = float((proj_prior / g_sq_usable).mean())

    # Surrogate: same fraction averaged over n_surrogate random unit directions.
    if n_surrogate > 0:
        surr = random_unit_directions(n_surrogate, numel, generator)  # [S, numel]
        proj_surr = (g_usable @ surr.t()) ** 2  # <g, v_rand>^2 for all (M, S)
        surrogate_alignment = float((proj_surr.mean(dim=1) / g_sq_usable).mean())
    else:
        surrogate_alignment = 0.0

    ratio = prior_alignment / surrogate_alignment if surrogate_alignment > 0.0 else 0.0

    return {
        "prior_alignment": prior_alignment,
        "surrogate_alignment": surrogate_alignment,
        "alignment_ratio": ratio,
    }
