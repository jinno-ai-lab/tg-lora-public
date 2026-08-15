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

from src.tg_lora.psa import _power_iteration_pc1

__all__ = [
    "ALIGNMENT_NULL",
    "ALIGNMENT_SIGNAL",
    "alignment_z_score",
    "decide_alignment_signal",
    "null_alignment_ratio_distribution",
    "prior_vs_surrogate_alignment",
    "random_unit_directions",
]


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


# Decision verdicts for the calibrated §7 go/no-go (see ``decide_alignment_signal``).
ALIGNMENT_SIGNAL = "SIGNAL"
ALIGNMENT_NULL = "NULL"


def null_alignment_ratio_distribution(
    numel: int,
    n_history: int,
    n_grad: int,
    *,
    n_trials: int = 256,
    n_surrogate: int = 128,
    n_extract_iters: int = 20,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Monte-Carlo null distribution of ``alignment_ratio`` under iid noise.

    The metric ``prior_vs_surrogate_alignment`` measures a *single* ratio; to
    turn that number into a §7 go/no-go we need the ratio's distribution under
    the null (prior AND held-out gradients both isotropic noise). A directional
    squared-energy ratio has no closed-form null — unlike the rank-1
    *eigenvalue*, whose null ``layer_delta_analysis`` derives in closed form
    from Marchenko-Pastur (``rank1_z``). This function supplies the
    Monte-Carlo analog: for each trial it runs the FULL production pipeline
    (extract a PC1 prior from iid ΔW via ``_power_iteration_pc1``, measure its
    ratio against iid held-out gradients) and collects the empirical null.

    This is the prerequisite honesty gate ``docs/psa_axis_research_question.md``
    §3/§4 left open: §3 calibrated the null *center* (≈1.0) and the signal
    (>>1.0) but derived no decision boundary, so the 9B live ``alignment_ratio``
    measurement would be a number with no threshold. The returned ``p99`` is
    that threshold (≤1% false-positive rate); ``alignment_z_score`` gives the
    ``rank1_z``-comparable summary.

    Args:
        numel: flattened parameter count of one LoRA tensor (the prior's dim).
        n_history: ΔW snapshot count the prior is extracted from (PSA ring buf).
        n_grad: number of held-out gradient samples (disjoint from history).
        n_trials: Monte-Carlo null samples. Larger tightens ``p99``/``max``.
        n_surrogate: random directions per trial (forwarded to the metric).
        n_extract_iters: power-iteration steps (production default 20).
        generator: optional ``torch.Generator`` — drives EVERY random draw
            (history, extraction seed, gradients, surrogates) so the whole
            distribution is reproducible from this one source.

    Returns:
        ``{"mean", "std", "p99", "max", "n_trials", "numel", "n_history",
        "n_grad"}`` — the calibrated null. ``mean`` ≈ 1.0 confirms the prior is
        no better than random under iid; ``p99`` is the ``decide_alignment_signal``
        boundary; ``max`` is the most conservative (no false positive within the
        sampled null).
    """
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    ratios: list[float] = []
    for _ in range(n_trials):
        mat = torch.randn(n_history, numel, generator=gen, dtype=torch.float32) * 0.1
        guess = torch.randn(numel, generator=gen, dtype=torch.float32)
        prior = _power_iteration_pc1(mat, n_iters=n_extract_iters, initial_guess=guess)
        prior = prior / (prior.norm() + 1e-12)
        grads = torch.randn(n_grad, numel, generator=gen, dtype=torch.float32) * 0.3
        result = prior_vs_surrogate_alignment(
            prior, grads, n_surrogate=n_surrogate, generator=gen
        )
        ratios.append(result["alignment_ratio"])
    samples = torch.tensor(ratios, dtype=torch.float32)
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std(unbiased=False)),
        "p99": float(torch.quantile(samples, 0.99)),
        "max": float(samples.max()),
        "n_trials": int(n_trials),
        "numel": int(numel),
        "n_history": int(n_history),
        "n_grad": int(n_grad),
    }


def alignment_z_score(measured_ratio: float, null: dict[str, float]) -> float:
    """Standardize a measured ``alignment_ratio`` against the calibrated null.

    The ``rank1_z`` analog for the PC1 *direction*: ``z = (ratio - null_mean) /
    null_std``. A real signal clears z ≫ 1; an iid measurement sits at z ≈ 0.
    Returns 0.0 when the null has no spread (degenerate / single-trial null).
    """
    std = float(null.get("std", 0.0) or 0.0)
    if std <= 1e-12:
        return 0.0
    return (float(measured_ratio) - float(null["mean"])) / std


def decide_alignment_signal(
    measured_ratio: float,
    null: dict[str, float],
) -> str:
    """§7 go/no-go: turn one measured ``alignment_ratio`` into a verdict.

    ``SIGNAL`` iff the measured ratio exceeds the 99th percentile of the iid
    null (≤1% false-positive rate — GOAL §7 "ランダム順サロゲートを超えた…
    だけを有効と認定"). Otherwise ``NULL``: the prior is statistically
    indistinguishable from a random direction and PSA would inject gradient
    noise. Conservative by design — PSA reactivates only on unambiguous
    directional signal. For a stricter boundary pass a null computed with more
    trials and compare against ``null["max"]``.
    """
    return (
        ALIGNMENT_SIGNAL
        if float(measured_ratio) > float(null["p99"])
        else ALIGNMENT_NULL
    )
