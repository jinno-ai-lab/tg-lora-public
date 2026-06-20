"""Per-layer stability-epoch estimation for the compromise freeze policy.

The ``compromise`` freeze policy (GOAL §3.1 Phase 2 candidate 3, design §4.1 /
§5.3) freezes output-side-first but defers each layer until it has *stabilized*:
``FreezeSchedule`` plans ``frozen_at_epoch = max(nominal, stability_epoch)``,
where ``stability_epoch`` is the earliest epoch a layer is safe to freeze. The
``freeze_schedule`` module docstring flags estimating that map as a separate,
[UNVERIFIED] step — this module *is* that step, closing the loop from an
observed per-layer stability signal to the planner's input.

Producer ↔ consumer
-------------------
* **Producer**: :meth:`src.tg_lora.dynamic_freeze.DynamicFreezeController.compute_r_A`
  emits ``{layer: r_A}`` per epoch, where ``r_A`` is the relative change of the
  LoRA delta-weight Frobenius norm. *Lower = quieter* LoRA movement = stabler
  layer (design §5.3 "安定した層から固める"). Collecting one ``r_A`` per epoch
  per layer gives exactly the ``{layer: [metric, ...]}`` series this function
  reads.
* **Consumer**: :class:`src.tg_lora.freeze_schedule.FreezeScheduleConfig`
  (``policy="compromise"``, ``stability_epoch=...``) and
  :class:`src.tg_lora.freeze_frontier.FrontierSpec` (``stability_epoch=...``).

This is pure Python and model-free — GOAL §7's "verify the mechanism before
trusting a GPU run": the compromise candidate can be planned and costed from
observed stability with no GPU in the loop.

Confirmation rule
-----------------
A layer is *confirmed stable* at epoch ``e`` once the trailing ``patience``
observations are all ``<= threshold`` (a one-off dip does not confirm — the
series must stay quiet for ``patience`` consecutive epochs).
``min_epoch`` forbids trusting stability before a warmup boundary, modeling
GOAL §3.1 / design §5.2's "freeze only after the model has settled". A layer
that never reaches ``patience`` consecutive quiet observations (or only would
before ``min_epoch``) is omitted; under ``compromise`` an omitted layer falls
back to its nominal freeze timing. To instead defer a never-stable layer out of
the run, set its floor to ``num_epochs`` before passing the map to the planner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def estimate_stability_epochs(
    series: Mapping[int, Sequence[float]],
    threshold: float,
    patience: int = 1,
    min_epoch: int = 0,
) -> dict[int, int]:
    """Turn a per-layer stability time-series into compromise-policy floors.

    Parameters
    ----------
    series:
        ``{layer: [metric per epoch]}``, 0-based (epoch index == list index).
        Lower is quieter/stabler, matching
        :meth:`DynamicFreezeController.compute_r_A`'s ``r_A`` semantics. A layer
        with an empty observation list cannot be confirmed and is omitted.
    threshold:
        A metric value ``<= threshold`` counts as one quiet (stable) epoch.
    patience:
        Consecutive quiet epochs required to confirm stability (``>= 1``). A
        single quiet blip below ``patience`` does not confirm.
    min_epoch:
        Earliest epoch at which stability may be declared (``>= 0``). Confirms
        the trailing window ending before ``min_epoch`` are rejected, so a
        layer quiet from epoch 0 is still held until warmup ends.

    Returns
    -------
    dict[int, int]
        ``{layer: earliest confirmed-stable epoch}`` for layers that reached
        ``patience`` consecutive quiet observations at/after ``min_epoch``.
        Layers that never confirmed are absent (→ ``compromise`` nominal
        timing). Iteration order is layer-ascending for deterministic output.

    Raises
    ------
    ValueError
        If ``patience < 1`` or ``min_epoch < 0``.
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if min_epoch < 0:
        raise ValueError(f"min_epoch must be >= 0, got {min_epoch}")

    result: dict[int, int] = {}
    for layer in sorted(series):
        obs = series[layer]
        # The confirmation epoch e must bound a full trailing window of
        # `patience` observations (so e starts at patience-1) and must sit at
        # or after the warmup boundary.
        e = max(min_epoch, patience - 1)
        while e < len(obs):
            window = obs[e - patience + 1 : e + 1]
            if all(value <= threshold for value in window):
                result[layer] = e
                break
            e += 1
    return result
