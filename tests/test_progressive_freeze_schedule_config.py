"""Config-driven multi-layer progressive-freeze schedule construction.

Pins :func:`build_freeze_schedule_from_config` — the config→schedule glue that
lets ``train_tg_lora`` construct a multi-layer controller from a config block,
the prerequisite for the Tier-2 §4 order verdict's valid_loss axis (multi-layer
``output_first`` vs ``random_order``). The controller's native multi-layer mode
is pinned in ``test_progressive_freeze_progressive``; this covers the glue and
that a config-built schedule drives the trainer gate's ``layers_due_at``.
"""

from __future__ import annotations

import pytest

from src.tg_lora.freeze_schedule import FreezeSchedule, random_freeze_order
from src.tg_lora.progressive_freeze import (
    ProgressiveFreezeController,
    build_freeze_schedule_from_config,
)

ACTIVE = [0, 1, 2, 3, 4, 5]  # output side = 5


@pytest.mark.parametrize("absent", [None, {}])
def test_absent_or_empty_config_returns_none(absent):
    # No schedule → None → the trainer keeps its single-shot Phase 1 gate.
    assert build_freeze_schedule_from_config(absent, ACTIVE, 10) is None


@pytest.mark.parametrize(
    "cfg, default_start, expected",
    [
        # output_first descends from the output side; start defaults to start_cycle.
        ({"max_depth": 3}, 1, {5: 1, 4: 2, 3: 3}),
        # explicit start_epoch + spacing control the cadence.
        ({"max_depth": 3, "start_epoch": 2, "spacing": 3}, 0, {5: 2, 4: 5, 3: 8}),
        # a freeze landing at >= num_epochs is dropped (matches the accountant).
        ({"max_depth": 3, "start_epoch": 8, "spacing": 2}, 0, {5: 8}),
    ],
)
def test_output_first_timing(cfg, default_start, expected):
    sched = build_freeze_schedule_from_config(cfg, ACTIVE, 10, default_start_epoch=default_start)
    assert isinstance(sched, FreezeSchedule)
    assert sched.frozen_at_epoch == expected
    assert sched.config.policy == "output_first"


def test_convergence_order_distinct_from_output_first():
    # Tier-2 contrast: a real output_first schedule and a shuffled surrogate
    # (convergence_order) resolve to different sequences from identical timing.
    cand = build_freeze_schedule_from_config({"max_depth": 6, "start_epoch": 0}, ACTIVE, 10)
    surr = build_freeze_schedule_from_config(
        {"policy": "convergence_order", "convergence_order": [0, 3, 5, 1, 4, 2],
         "max_depth": 6, "start_epoch": 0}, ACTIVE, 10,
    )
    assert cand.order == (5, 4, 3, 2, 1, 0)
    assert surr.order == (0, 3, 5, 1, 4, 2)
    assert cand.order != surr.order


def test_compromise_stability_epoch_passes_through():
    sched = build_freeze_schedule_from_config(
        {"policy": "compromise", "max_depth": 3, "start_epoch": 1, "stability_epoch": {4: 6}},
        ACTIVE, 10,
    )
    assert sched.frozen_at_epoch == {5: 1, 4: 6, 3: 3}  # layer 4 raised to its floor


def test_config_built_schedule_drives_controller_gate():
    # The trainer gate calls layers_due_at(cycle); a config-built schedule must
    # drive it identically to a hand-built one. max_depth bounds the candidate
    # list BEFORE epoch assignment, so layers 5,4,3 (not 5,3,1) at epochs 1,3,5.
    sched = build_freeze_schedule_from_config(
        {"max_depth": 3, "start_epoch": 1, "spacing": 2}, ACTIVE, 10,
    )
    ctrl = ProgressiveFreezeController(start_cycle=1, active_layer_indices=set(ACTIVE), schedule=sched)
    assert ctrl.layers_due_at(0) == []
    assert ctrl.layers_due_at(1) == [5]
    assert ctrl.layers_due_at(3) == [4]
    assert ctrl.layers_due_at(5) == [3]


@pytest.mark.parametrize(
    "bad_cfg, match",
    [
        ({"policy": "sideways", "max_depth": 1}, "policy"),
        ({"max_depth": 99}, "max_depth"),
        ({"policy": "convergence_order", "max_depth": 1}, "convergence_order"),
    ],
)
def test_bad_config_raises_value_error(bad_cfg, match):
    # A malformed config fails loudly at construction, never silently no-ops.
    with pytest.raises(ValueError, match=match):
        build_freeze_schedule_from_config(bad_cfg, ACTIVE, 10)


# -- Tier-2 §4 order verdict: config-driven random-order surrogate arm --------
#
# PURPOSE「次の一手 (b)」names the random-order surrogate arm's upstream port as
# the remaining Category-A prerequisite for the Tier-2 §4 order verdict
# (resolve the proxy-scale order-sensitivity ratio=0.000 at 9B target-scale).
# The surrogate-null generator random_freeze_order() existed, but a real run
# could not express candidate(output_first) vs surrogate(random_order) from a
# config block — so the verdict's contrast was unreachable from training. These
# pin the config glue that makes the arm reproducible-from-config.


def test_random_order_surrogate_is_reproducible():
    # GOAL §4 surrogate-null must reproduce across the multi-seed sweeps it
    # demands: identical (layers, seed) → identical realized schedule, and the
    # config path routes through the documented random_freeze_order() generator
    # (the same reproducible permutation), not an ad-hoc RNG.
    a = build_freeze_schedule_from_config(
        {"policy": "random_order", "seed": 42, "max_depth": 3,
         "start_epoch": 1, "spacing": 2}, ACTIVE, 10,
    )
    b = build_freeze_schedule_from_config(
        {"policy": "random_order", "seed": 42, "max_depth": 3,
         "start_epoch": 1, "spacing": 2}, ACTIVE, 10,
    )
    assert a is not None and b is not None
    assert a.frozen_at_epoch == b.frozen_at_epoch
    assert a.order == b.order
    assert a.order == random_freeze_order(ACTIVE, 42)[:3]
    # random_order resolves to the convergence_order policy carrying a seeded
    # random_freeze_order() — no separate planner branch (design §5.3), so the
    # surrogate flows the IDENTICAL planner/accountant path as a real schedule.
    assert a.config.policy == "convergence_order"


def test_random_order_surrogate_shares_timing_with_output_first_candidate():
    # THE Tier-2 §4 apples-to-apples property: candidate(output_first) and
    # surrogate(random_order) with identical (max_depth, start_epoch, spacing,
    # num_epochs) freeze the SAME count of layers at the SAME epochs — only the
    # layer identity (the order) differs. That isolates ORDER as the sole
    # degree of freedom the verdict resolves. Values grounded in the real RNG:
    # seed 42 shuffles ACTIVE to (3,1,2,4,0,5); depth-3 first-3 = (3,1,2) →
    # {3:1, 1:3, 2:5}, vs candidate output_first {5:1, 4:3, 3:5}.
    timing = {"max_depth": 3, "start_epoch": 1, "spacing": 2}
    cand = build_freeze_schedule_from_config(
        {"policy": "output_first", **timing}, ACTIVE, 10,
    )
    surr = build_freeze_schedule_from_config(
        {"policy": "random_order", "seed": 42, **timing}, ACTIVE, 10,
    )
    assert cand is not None and surr is not None
    assert cand.realized_depth == surr.realized_depth == 3
    # Same freeze epochs (timing held fixed)...
    assert sorted(cand.frozen_at_epoch.values()) == [1, 3, 5]
    assert sorted(surr.frozen_at_epoch.values()) == [1, 3, 5]
    # ...different layer identity (the order is the only thing that differs).
    assert set(surr.frozen_at_epoch) == {1, 2, 3}
    assert set(surr.frozen_at_epoch) != set(cand.frozen_at_epoch)
    assert surr.order == (3, 1, 2)


def test_random_order_surrogate_diversifies_across_seeds():
    # A multi-seed sweep needs distinct surrogates: several seeds must yield
    # more than one distinct realized order (a generator collapsing every seed
    # to one permutation could not serve as a null-baseline distribution).
    orders = {
        build_freeze_schedule_from_config(
            {"policy": "random_order", "seed": s, "max_depth": 6}, ACTIVE, 10,
        ).order
        for s in (1, 2, 3, 7, 42, 99)
    }
    assert len(orders) > 1


def test_random_order_requires_seed():
    # A seedless random_order would be non-reproducible — fail loudly, never
    # silently fall back to a global-RNG order.
    with pytest.raises(ValueError, match="seed"):
        build_freeze_schedule_from_config(
            {"policy": "random_order", "max_depth": 3}, ACTIVE, 10,
        )


def test_random_order_rejects_explicit_convergence_order():
    # random_order generates its own order; an explicit convergence_order is
    # contradictory — fail loudly rather than silently ignore one of them.
    with pytest.raises(ValueError, match="convergence_order"):
        build_freeze_schedule_from_config(
            {"policy": "random_order", "seed": 42, "max_depth": 3,
             "convergence_order": [0, 1, 2]}, ACTIVE, 10,
        )


def test_random_order_surrogate_drives_controller_gate():
    # The config-built surrogate threads through the trainer's layers_due_at
    # gate like any schedule. seed 42 depth-3 → layers (3,1,2) at epochs 1,3,5;
    # the frozen set only ever grows (progressive, design §4.1).
    sched = build_freeze_schedule_from_config(
        {"policy": "random_order", "seed": 42, "max_depth": 3,
         "start_epoch": 1, "spacing": 2}, ACTIVE, 10,
    )
    ctrl = ProgressiveFreezeController(
        start_cycle=1, active_layer_indices=set(ACTIVE), schedule=sched,
    )
    assert ctrl.layers_due_at(0) == []
    assert ctrl.layers_due_at(1) == [3]
    assert ctrl.layers_due_at(3) == [1]
    assert ctrl.layers_due_at(5) == [2]
    assert ctrl.layers_due_at(6) == []
