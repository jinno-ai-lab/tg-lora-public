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

from src.tg_lora.freeze_schedule import FreezeSchedule
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
