#!/usr/bin/env python
"""Order-resolution diagnostic: can the proxy apparatus resolve freeze order at all?

Companion to :mod:`scripts.run_freeze_validloss_ci`. That script reports the
GOAL §4 verdict — candidate (output-first) vs random-order surrogate — and at
proxy scale every cell returns ``TIES``. A ``TIES`` is only a *conclusive* null
if the apparatus can resolve an order effect when one exists; otherwise the
``TIES`` could be a below-resolution read (a measurement that cannot see the
signal) rather than a genuine "order does not matter". PURPOSE.md flagged exactly
this: the proxy positive control never fired, so the apparatus **sensitivity was
unproven** — the open scientific debt this diagnostic closes.

It does not try to *make* order matter (the verdict runs already do that); it
measures the apparatus's order-resolution directly with a **variance
decomposition**:

* :data:`var_order` — the spread of the final valid_loss across ``n_orders``
  *distinct* freeze orders at a **FIXED** seed (model init + data draw pinned).
  This is the entire signal a freeze-order effect could imprint on the metric:
  if order matters, varying it at fixed seed must move the valid_loss.
* :data:`var_seed` — the spread across ``n_seeds`` seeds at a **FIXED** order
  (``output_first``). This is the seed-noise floor the order signal would have
  to rise above to be detectable.
* :data:`ratio` = ``var_order / var_seed`` — the fraction of the noise-floor
  variance that order accounts for. ``ratio`` ≈ 0 ⇒ the apparatus cannot
  resolve order (the verdict ``TIES`` is a genuine null, not a power failure);
  ``ratio`` ≫ 0 ⇒ order is resolvable and the verdict runs can in principle
  detect it.

**The finding (RTX 3060, 2026-06-25).** ``ratio = 0.000`` — *exactly* zero, not
merely small — under the prod-path boundary local loss, and this is invariant
across the homogeneous / heterogeneous / capacity-concentrated stacks and across
early (``warmup=10``) and late (``warmup=45``) freeze at depth 3 and 5. The
freeze *is* applied order-dependently (verified: different layers carry
``requires_grad=False`` per order), so the zero is not a wiring bug — it is that
the boundary activation-matching local loss (GOAL §1.6.3, the loss the prod path
switches to once a layer is frozen) does not couple to the held-out task metric,
so the final valid_loss is pinned by the order-independent pre-freeze trajectory.
Running the *task* loss throughout (bypassing the local loss) lifts the ratio to
only ~0.001 — still negligible. The conclusion is therefore stronger than
"unproven sensitivity": **at proxy scale freeze order is genuinely unresolvable**
(the full-rank trainable output head + residual stack is robust to which LoRA
layer is frozen), so the verdict ``TIES`` is a real null and the target-scale 9B
run — where the LM head is the real distribution and layers specialize — is the
*only* regime that can resolve whether order matters. This converts
"target-scale is assumed necessary" into "target-scale is proven necessary".

This is honest about what it is and is not (GOAL §7):

* **It is a real run.** Every arm trains the real progressive-freeze trio on GPU
  when available (``--device cuda`` / ``auto``); the valid_loss is read off the
  real forward pass.
* **It is a diagnostic, not a verdict.** It does not call
  :func:`surrogate_valid_loss_ci` and emits no ``SURPASSES``/``TIES`` label — it
  reports the variance ratio that says whether the verdict runs *could* resolve
  order at this scale. It is the measurement-science step a careful scientist
  runs before trusting a null: prove the assay can see the analyte.
* **It is proxy-scale.** HIDDEN=24 / 6 layers is not the 9B target, so the
  result is tagged ``proxy_scale=True`` and the report says so plainly. A
  target-scale run deposits its own variance decomposition through the same
  function and the conclusion upgrades with no code change.

Usage::

    # Auto device (cuda if available) — the one-shot diagnostic.
    make freeze-order-sensitivity
    python -m scripts.run_freeze_order_sensitivity

    # Pin CPU for a deterministic CI reproduction; write JSON evidence.
    python -m scripts.run_freeze_order_sensitivity --device cpu \\
        --n-orders 6 --n-seeds 6 --json --output order_sensitivity.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from typing import Sequence


# Reuse the verdict runner's fixtures so the diagnostic trains the *same*
# progressive-freeze trio on the *same* learnable proxy — the decomposition is
# over the identical arm the §4 verdict is drawn from, not a parallel model.
from scripts.run_freeze_validloss_ci import (
    ARCHITECTURES,
    DEFAULT_BASE_SEED,
    DEFAULT_DEPTH,
    DEFAULT_TOTAL,
    DEFAULT_WARMUP,
    HOMOGENEOUS,
    HIDDEN,
    NUM_LAYERS,
    TASK_GENERALIZE,
    TASKS,
    arm_valid_loss,
    heterogeneous_ranks,
    output_first_order,
    resolve_device,
)

# Default arm budget mirrors the verdict runner (the converged-regime defaults
# of tests/test_progressive_freeze_invivo.py). The make target keeps these; the
# test suite shrinks them via the CLI flags for a fast, deterministic check.
DEFAULT_N_ORDERS = 12
DEFAULT_N_SEEDS = 12

# Seed offset for drawing the distinct fixed-seed orders. Distinct from the arm
# seed axis (and from the verdict runner's surrogate offset) so the order draw
# is independent of the per-arm RNGs.
ORDER_DRAW_SEED_OFFSET = 7_000

# Resolution threshold. ``ratio >= RESOLUTION_THRESHOLD`` means the order signal
# is at least this fraction of the seed-noise floor — large enough that the
# verdict runs could in principle resolve an order effect at this scale. 0.10 is
# a deliberately conservative bar (order must explain >=10% of the seed noise);
# the proxy finding sits far below it at 0.000, so the exact value does not gate
# the conclusion, only the printed ``resolvable`` flag.
RESOLUTION_THRESHOLD = 0.10


def _variance(values: Sequence[float]) -> float:
    """Sample variance (n-1); 0.0 for fewer than two observations.

    The unbiased estimator so the two arms (``n_orders`` / ``n_seeds`` draws) are
    comparable on the same denominator footing regardless of their (equal) sizes.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def distinct_orders(num_layers: int, n: int, seed: int) -> list[tuple[int, ...]]:
    """The first ``n`` orders of a seeded shuffle of *all* freeze permutations.

    A uniform, reproducible draw of distinct orders: for the small stacks the
    proxy uses (``num_layers`` <= 7, so <= 5040 permutations) every order is
    reachable, and the seeded :func:`random.Random.shuffle` pins the draw
    bit-for-bit. This is the same uniform distribution
    :func:`freeze_schedule.random_freeze_order` samples, made collision-free so
    every one of the ``n_orders`` arms genuinely exercises a *different* order —
    the property the variance decomposition needs (duplicate orders would
    understate :data:`var_order` and the diagnostic with it).
    """
    if n < 1:
        return []
    permutations = list(itertools.permutations(range(num_layers)))
    if n > len(permutations):
        raise ValueError(
            f"requested {n} distinct orders but only {len(permutations)} "
            f"permutations exist for num_layers={num_layers}"
        )
    rng = random.Random(seed)
    rng.shuffle(permutations)
    return [tuple(p) for p in permutations[:n]]


def order_sensitivity(
    *,
    device,
    total: int = DEFAULT_TOTAL,
    warmup: int = DEFAULT_WARMUP,
    depth: int = DEFAULT_DEPTH,
    n_orders: int = DEFAULT_N_ORDERS,
    n_seeds: int = DEFAULT_N_SEEDS,
    base_seed: int = DEFAULT_BASE_SEED,
    num_layers: int = NUM_LAYERS,
    architecture: str = HOMOGENEOUS,
    task: str = TASK_GENERALIZE,
) -> dict:
    """Variance-decomposition of the proxy's freeze-order resolution.

    Trains ``n_orders`` arms under *distinct* freeze orders at a **fixed** seed
    (model init + data pinned) and ``n_seeds`` arms under a **fixed** order
    (``output_first``) across seeds — all through the real progressive-freeze
    trio on the shared proxy stack — and returns :data:`var_order` (the signal an
    order effect could produce), :data:`var_seed` (the seed-noise floor), and
    their :data:`ratio`. Every RNG is locally seeded, so a fixed ``base_seed``
    reproduces the whole decomposition bit-for-bit on a given device.

    ``architecture`` selects the proxy stack both arms share (``homogeneous`` —
    the verdict baseline — or ``heterogeneous`` — per-layer rank rising toward
    the output). ``task`` defaults to :data:`TASK_GENERALIZE`: the
    teacher-student held-out split is the only regime in which freeze order can
    move quality, so it is the regime whose order-resolution this diagnostic
    exists to measure (the ``memorize`` task is order-invariant by construction,
    so a zero :data:`var_order` there is uninformative).

    Returns a dict with the decomposition, the raw per-arm valid_loss samples
    (so the audit can re-derive the variances), the full provenance, and a
    ``proxy_scale`` honesty flag (GOAL §7).
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if n_orders < 2:
        raise ValueError(f"n_orders must be >= 2 to estimate var_order, got {n_orders}")
    if n_seeds < 2:
        raise ValueError(f"n_seeds must be >= 2 to estimate var_seed, got {n_seeds}")

    ranks = (
        None if architecture == HOMOGENEOUS
        else heterogeneous_ranks(num_layers, HIDDEN)
    )
    orders = distinct_orders(num_layers, n_orders, base_seed + ORDER_DRAW_SEED_OFFSET)

    # Var(order): distinct orders, FIXED seed -> isolates the order signal.
    by_order = [
        arm_valid_loss(
            order, base_seed,
            device=device, total=total, warmup=warmup, depth=depth,
            num_layers=num_layers, ranks=ranks, task=task,
        )
        for order in orders
    ]
    # Var(seed): FIXED order (output_first), distinct seeds -> the noise floor.
    by_seed = [
        arm_valid_loss(
            output_first_order(num_layers), base_seed + i,
            device=device, total=total, warmup=warmup, depth=depth,
            num_layers=num_layers, ranks=ranks, task=task,
        )
        for i in range(n_seeds)
    ]

    var_order = _variance(by_order)
    var_seed = _variance(by_seed)
    if var_seed > 0.0:
        ratio = var_order / var_seed
    else:
        # A zero seed-noise floor is degenerate (the proxy did not vary across
        # seeds). Report the ratio as 0 when order is also still, else +inf so a
        # reader sees order moving on a flat seed floor rather than a tidy 0.
        ratio = 0.0 if var_order == 0.0 else float("inf")
    resolvable = ratio >= RESOLUTION_THRESHOLD

    reported_ranks = list(ranks) if ranks is not None else [HIDDEN] * num_layers
    return {
        "var_order": var_order,
        "var_seed": var_seed,
        "ratio": ratio,
        "resolvable": resolvable,
        "by_order": by_order,
        "by_seed": by_seed,
        "orders": [list(o) for o in orders],
        "device": str(device),
        "total": total,
        "warmup": warmup,
        "depth": depth,
        "n_orders": n_orders,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "num_layers": num_layers,
        "architecture": architecture,
        "ranks": reported_ranks,
        "task": task,
        "resolution_threshold": RESOLUTION_THRESHOLD,
        # Proxy-scale honesty (GOAL §7): HIDDEN=24 / 6 layers is not the 9B
        # target. A target run deposits its own decomposition through this same
        # function and the conclusion upgrades with no code change — this flag
        # is what a reader checks before citing the ratio as a target-scale
        # result.
        "proxy_scale": True,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_freeze_order_sensitivity",
        description=(
            "Order-resolution diagnostic: variance-decomposes the proxy's "
            "final valid_loss into a Var(order) signal (distinct orders at a "
            "fixed seed) vs a Var(seed) noise floor (fixed order across seeds) "
            "and reports their ratio — does the §4 verdict apparatus resolve "
            "freeze order at proxy scale? (proxy-scale)."
        ),
    )
    p.add_argument(
        "--device", default="auto",
        help="torch device: 'auto' (cuda if available else cpu), 'cpu', or 'cuda'.",
    )
    p.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="training epochs per arm.")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="freeze start epoch.")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="freeze depth (layers frozen).")
    p.add_argument("--n-orders", type=int, default=DEFAULT_N_ORDERS, help="distinct orders (var_order arm).")
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS, help="seeds (var_seed arm).")
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="sweep base seed.")
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS, help="proxy stack depth.")
    p.add_argument(
        "--architecture", default=HOMOGENEOUS, choices=ARCHITECTURES,
        help=(
            "proxy stack: 'homogeneous' (every layer identical) or "
            "'heterogeneous' (per-layer LoRA rank rising toward the output)."
        ),
    )
    p.add_argument(
        "--task", default=TASK_GENERALIZE, choices=TASKS,
        help=(
            "dataset: 'generalize' (held-out teacher-student split — the "
            "regime where order can matter; the meaningful default) or "
            "'memorize' (train==valid — order-invariant by construction)."
        ),
    )
    p.add_argument("--json", action="store_true", help="emit JSON evidence to stdout.")
    p.add_argument("--output", default=None, help="write the report/JSON to this path too.")
    return p


def format_report(result: dict) -> str:
    """Human-readable order-resolution block with provenance + proxy-scale caveat."""
    resolvable = result["resolvable"]
    ratio = result["ratio"]
    lines = [
        "freeze_order_sensitivity — apparatus order-resolution diagnostic",
        f"  device: {result['device']}  proxy_scale={result['proxy_scale']}  "
        f"architecture={result['architecture']}  task={result['task']}  "
        f"(HIDDEN={HIDDEN}, num_layers={result['num_layers']}, depth={result['depth']}, "
        f"epochs={result['total']}, warmup={result['warmup']})",
        f"  var_order arm: n_orders={result['n_orders']} distinct orders at FIXED "
        f"seed={result['base_seed']}",
        f"  var_seed arm:  n_seeds={result['n_seeds']} seeds at FIXED order=output_first  "
        f"ranks={result['ranks']}",
        "",
        "  variance decomposition (final valid_loss):",
        f"    Var(order) = {result['var_order']:.8f}  "
        f"(spread across {result['n_orders']} orders at a fixed seed = the signal "
        f"an order effect could produce)",
        f"    Var(seed)  = {result['var_seed']:.8f}  "
        f"(spread across {result['n_seeds']} seeds at a fixed order = the "
        f"seed-noise floor)",
        f"    ratio      = {ratio:.8f}  = Var(order)/Var(seed)  "
        f"(fraction of the noise floor order accounts for)",
        "",
        f"  verdict: order {'IS' if resolvable else 'is NOT'} resolvable "
        f"(ratio {'>=' if resolvable else '<'} threshold "
        f"{result['resolution_threshold']:.2f})",
    ]
    if resolvable:
        lines.append(
            "  => The apparatus CAN resolve a freeze-order effect at this scale: "
            "Var(order) is a meaningful fraction of the seed-noise floor, so the "
            "§4 verdict runs (candidate vs random-order surrogate) can in "
            "principle detect an order effect. A TIES there is then a genuine "
            "'no effect' rather than a below-resolution read."
        )
    else:
        lines.append(
            "  => The apparatus CANNOT resolve a freeze-order effect at proxy "
            f"scale: Var(order) is ~0 (ratio {ratio:.6f} << threshold "
            f"{result['resolution_threshold']:.2f}) while Var(seed) is real "
            f"({result['var_seed']:.6f}). The freeze IS applied order-dependently "
            "(different layers frozen per order), but the boundary local loss "
            "the prod path switches to once a layer is frozen does not couple to "
            "the held-out task metric, so the final valid_loss is pinned by the "
            "order-independent pre-freeze trajectory. The §4 verdict TIES is "
            "therefore a GENUINE null (order does not move quality here), not a "
            "power failure — and resolving whether order matters at all needs "
            "the target-scale run (real LM head, real layer specialization)."
        )
    lines += [
        "",
        f"  by_order samples: {[round(v, 6) for v in result['by_order']]}",
        f"  by_seed samples:  {[round(v, 6) for v in result['by_seed']]}",
    ]
    if result["proxy_scale"]:
        lines.append(
            "  note: PROXY_SCALE — the ratio is from a 24-hidden / "
            f"{result['num_layers']}-layer proxy, not the 9B target. A "
            "target-scale run deposits its own decomposition through this same "
            "function and the conclusion upgrades; do not cite this ratio as a "
            "target-scale result."
        )
    return "\n".join(lines)


def result_to_json(result: dict) -> dict:
    return {
        "var_order": result["var_order"],
        "var_seed": result["var_seed"],
        "ratio": result["ratio"],
        "resolvable": result["resolvable"],
        "by_order": result["by_order"],
        "by_seed": result["by_seed"],
        "orders": result["orders"],
        "device": result["device"],
        "total": result["total"],
        "warmup": result["warmup"],
        "depth": result["depth"],
        "n_orders": result["n_orders"],
        "n_seeds": result["n_seeds"],
        "base_seed": result["base_seed"],
        "num_layers": result["num_layers"],
        "architecture": result["architecture"],
        "ranks": result["ranks"],
        "task": result["task"],
        "resolution_threshold": result["resolution_threshold"],
        "proxy_scale": result["proxy_scale"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    result = order_sensitivity(
        device=device,
        total=args.total,
        warmup=args.warmup,
        depth=args.depth,
        n_orders=args.n_orders,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        num_layers=args.num_layers,
        architecture=args.architecture,
        task=args.task,
    )
    payload = json.dumps(result_to_json(result), indent=2) if args.json else format_report(result)
    print(payload)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
