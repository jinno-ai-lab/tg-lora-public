"""Tests for ``DynamicFreezeController`` §3 (freeze) + §4 (reversible unfreeze).

The PyTorch guard controller (``src/tg_lora/dynamic_freeze.py``) had **zero**
direct coverage of its §4 reversible-release half — ``decide_unfreeze`` /
``apply_unfreeze`` / checkpoint round-trip — even though the MLX sibling port
(``mlx/tests/test_dynfreeze_mlx.py``) tests that path and the trainer wires it
into the live loop (``train_tg_lora.py``). These tests close that gap and pin
the §4 invariant that a released layer *actually re-trains* rather than being
silently re-frozen in the same cycle (the frozen-period ``0.0`` r_A history is
not real quietness). See ``docs/design/10_guard_experiment.md`` §4.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

from src.tg_lora.dynamic_freeze import DynamicFreezeController, DynFreezeState

NUM_LAYERS = 6  # indices 0..5; output side = 5


class _Layer(nn.Module):
    """Minimal decoder layer exposing ``lora_A``/``lora_B`` named params.

    ``lora_B`` starts at zero (standard LoRA init) so a layer with no weight
    movement records r_A ≈ 0 (quiet); scaling ``lora_B`` makes it noisy.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, hidden, bias=False)
        self.lora_A = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.base(x) + x @ self.lora_A.t() @ self.lora_B.t()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(NUM_LAYERS)])


def _build_model() -> _Model:
    model = _Model()
    for p in model.parameters():
        p.requires_grad = True  # all LoRA params trainable by default
    return model


def _set_history(dfc: DynamicFreezeController, values: dict[int, float]) -> None:
    """Seed each layer's r_A window with a flat value (full window)."""
    for li, v in values.items():
        dfc._r_A_history[li] = deque([v] * dfc._window, maxlen=dfc._window)


def _make(**overrides) -> DynamicFreezeController:
    defaults = dict(
        tau=0.02,
        window=4,
        stir_interval=10,
        upstream_activity_factor=1.5,
        all_layer_indices=list(range(NUM_LAYERS)),
    )
    defaults.update(overrides)
    return DynamicFreezeController(**defaults)


# ---------------------------------------------------------------------------
# §3: decide_freeze — output-side contiguous block from quiet layers
# ---------------------------------------------------------------------------


class TestDecideFreeze:
    def test_warmup_returns_empty_before_full_window(self):
        # The controller needs a full window of r_A history before acting.
        dfc = _make(window=4)
        _set_history(dfc, {li: 0.001 for li in range(NUM_LAYERS)})
        assert dfc.decide_freeze(0) == []
        assert dfc.decide_freeze(3) == []

    def test_freezes_output_side_contiguous_quiet_block(self):
        dfc = _make(window=4)
        hist = {li: 0.05 for li in range(NUM_LAYERS)}  # all noisy
        hist.update({5: 0.001, 4: 0.001, 3: 0.001})  # output trio quiet
        _set_history(dfc, hist)
        assert dfc.decide_freeze(4) == [5, 4, 3]

    def test_block_stops_at_first_noisy_layer(self):
        dfc = _make(window=4)
        _set_history(dfc, {5: 0.001, 4: 0.05, 3: 0.001})  # L4 noisy breaks block
        assert dfc.decide_freeze(4) == [5]

    def test_block_extends_contiguously_from_existing_block(self):
        dfc = _make(window=4)
        dfc._frozen_block = [5, 4]
        _set_history(dfc, {li: 0.05 for li in range(NUM_LAYERS)})
        dfc._r_A_history[3] = deque([0.001] * 4, maxlen=4)  # only L3 newly quiet
        assert dfc.decide_freeze(4) == [3]

    def test_existing_block_rejects_non_contiguous_extension(self):
        dfc = _make(window=4)
        dfc._frozen_block = [5, 4]
        # L3 noisy, L2 quiet — quiet layer is NOT adjacent to block → reject all.
        _set_history(dfc, {li: 0.05 for li in range(NUM_LAYERS)})
        dfc._r_A_history[2] = deque([0.001] * 4, maxlen=4)
        assert dfc.decide_freeze(4) == []


# ---------------------------------------------------------------------------
# §4: decide_unfreeze — upstream-only release, two triggers, output protected
# ---------------------------------------------------------------------------


class TestDecideUnfreeze:
    def test_empty_block_releases_nothing(self):
        dfc = _make()
        assert dfc.decide_unfreeze(100) == []

    def test_stir_releases_upstream_end_after_R_cycles(self):
        # §4(a): block held R=10 cycles → release the upstream-most (smallest idx).
        dfc = _make(stir_interval=10)
        dfc._frozen_block = [5, 4, 3]
        dfc._frozen_since_cycle = 0
        assert dfc.decide_unfreeze(10) == [3]  # 10 - 0 >= 10
        assert dfc.decide_unfreeze(9) == []  # 9 - 0 < 10

    def test_stir_never_releases_output_side(self):
        # Output side (L5, largest idx) is never the release target.
        dfc = _make(stir_interval=10)
        dfc._frozen_block = [5, 4, 3]
        dfc._frozen_since_cycle = 0
        released = dfc.decide_unfreeze(20)
        assert released == [3]
        assert 5 not in released

    def test_upstream_activity_releases_upstream_end(self):
        # §4(b): the layer just upstream of the block gets noisy → release.
        dfc = _make(stir_interval=1000)  # disable stir, isolate activity trigger
        dfc._frozen_block = [5, 4, 3]
        dfc._frozen_since_cycle = 0  # well within stir interval
        # Upstream neighbor of block (min([5,4,3])-1 = 2) noisy: 0.05 > τ*1.5=0.03.
        _set_history(dfc, {li: 0.001 for li in range(NUM_LAYERS)})
        dfc._r_A_history[2] = deque([0.05] * 4, maxlen=4)
        assert dfc.decide_unfreeze(4) == [3]

    def test_no_trigger_releases_nothing(self):
        dfc = _make(stir_interval=10)
        dfc._frozen_block = [5, 4, 3]
        dfc._frozen_since_cycle = 100  # frozen 2 cycles at cycle 102 → < 10
        _set_history(dfc, {li: 0.001 for li in range(NUM_LAYERS)})  # upstream quiet
        assert dfc.decide_unfreeze(102) == []


# ---------------------------------------------------------------------------
# §4: apply_unfreeze — mutates requires_grad, block, and (now) the stir timer
# ---------------------------------------------------------------------------


class TestApplyUnfreeze:
    def test_sets_requires_grad_and_shrinks_block(self):
        model = _build_model()
        dfc = _make()
        # apply_freeze builds the block (do not also hand-set _frozen_block).
        dfc.apply_freeze(model, [5, 4, 3], cycle=0)
        assert dfc.apply_unfreeze(model, [3], cycle=10) > 0
        assert dfc.frozen_block == [5, 4]
        # Released layer's params are trainable again.
        from src.model.lora_utils import iter_all_lora_params_by_layer

        for _name, p in iter_all_lora_params_by_layer(model)[3]:
            assert p.requires_grad

    def test_empty_or_missing_block_is_noop(self):
        model = _build_model()
        dfc = _make()
        assert dfc.apply_unfreeze(model, [], cycle=10) == 0
        dfc._frozen_block = [5, 4]
        assert dfc.apply_unfreeze(model, [], cycle=10) == 0


# ---------------------------------------------------------------------------
# THE BUG: §4 release must take effect (not be inert)
# ---------------------------------------------------------------------------


class TestReleaseIsNotInert:
    """A released layer must actually re-train. Before the fix, ``compute_r_A``
    records ``0.0`` for frozen layers, so the immediately-following
    ``decide_freeze`` saw the just-released layer as "quiet" and re-froze it in
    the same cycle — the §4 reversible release was a silent no-op, and the stir
    timer (reset only by ``apply_freeze``) drained the whole block one layer per
    cycle. These two tests capture both facets."""

    def test_released_layer_not_refrozen_same_cycle(self):
        model = _build_model()
        dfc = _make(stir_interval=10)
        dfc.apply_freeze(model, [5, 4, 3], cycle=0)  # block=[5,4,3], frozen_since=0
        # Frozen-period history is all 0.0 (compute_r_A records 0.0 while frozen);
        # the actively-training upstream layers 0-2 are genuinely noisy.
        for li in (5, 4, 3):
            dfc._r_A_history[li] = deque([0.0] * 4, maxlen=4)
        for li in (2, 1, 0):
            dfc._r_A_history[li] = deque([0.05] * 4, maxlen=4)

        cycle = 10  # frozen 10 cycles >= R → stir
        released = dfc.decide_unfreeze(cycle)
        assert released == [3]
        dfc.apply_unfreeze(model, released, cycle=cycle)  # block now [5, 4]

        # Must NOT re-freeze the layer released this same cycle: its 0.0 history
        # is frozen-period artifact, not real quietness.
        refreeze = dfc.decide_freeze(cycle)
        assert refreeze == [], (
            f"released layer L3 re-frozen same cycle ({refreeze}) — "
            "§4 release is inert"
        )

    def test_stir_is_periodic_not_one_layer_per_cycle_drain(self):
        model = _build_model()
        dfc = _make(stir_interval=10)
        dfc._frozen_block = [5, 4, 3]
        dfc._frozen_since_cycle = 0

        cycle = 10  # stir fires, releases upstream end [3]
        dfc.apply_unfreeze(model, dfc.decide_unfreeze(cycle), cycle=cycle)

        # Timer re-armed at release → next cycle has frozen only 1 cycle < R.
        # Pre-fix the timer was never re-armed, so stir fired again → drained [4].
        assert dfc.decide_unfreeze(cycle + 1) == [], (
            "stir drained the block one layer per cycle instead of re-arming"
        )

    def test_released_layer_can_refreeze_after_cooldown_when_settled(self):
        """Reversibility: after a full window of cooldown, a genuinely re-settled
        layer may re-freeze (the block regrows). This confirms the cooldown
        *expires* rather than permanently locking the layer out."""
        model = _build_model()
        dfc = _make(stir_interval=10)
        dfc.apply_freeze(model, [5, 4, 3], cycle=0)  # builds block, frozen_since=0
        # Release L3 at cycle 10.
        dfc.apply_unfreeze(model, dfc.decide_unfreeze(10), cycle=10)

        # Advance past the window cooldown; seed L3 genuinely quiet and the
        # upstream layers noisy so only L3 is the re-freeze candidate.
        dfc._r_A_history[3] = deque([0.001] * 4, maxlen=4)
        for li in (2, 1, 0):
            dfc._r_A_history[li] = deque([0.05] * 4, maxlen=4)
        assert dfc.decide_freeze(10 + dfc._window) == [3]


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    def test_state_dict_round_trips_block_history_and_timer(self):
        dfc = _make(window=4)
        dfc._frozen_block = [5, 4, 3]
        dfc._frozen_since_cycle = 7
        _set_history(dfc, {5: 0.01, 4: 0.02, 3: 0.03})
        dfc._released_at = {2: 5}

        state = dfc.state_dict()
        assert isinstance(state, DynFreezeState)
        assert state.frozen_layer_indices == [5, 4, 3]
        assert state.frozen_since_cycle == 7
        assert state.r_A_history[3] == [0.03, 0.03, 0.03, 0.03]
        assert state.released_at == {2: 5}

        fresh = _make(window=4)
        fresh.load_state_dict(state)
        assert fresh.frozen_block == [5, 4, 3]
        assert fresh._frozen_since_cycle == 7
        assert fresh._released_at == {2: 5}
        assert list(fresh._r_A_history[3]) == [0.03] * 4

    def test_load_state_dict_tolerates_legacy_state_without_released_at(self):
        """Old checkpoints predate the released_at field; loading must not break."""
        dfc = _make()
        legacy = DynFreezeState(
            frozen_layer_indices=[5, 4],
            r_A_history={5: [0.01]},
            frozen_since_cycle=3,
        )
        # Simulate a state object lacking the released_at attribute entirely.
        legacy_dict = {
            k: v for k, v in legacy.__dict__.items() if k != "released_at"
        }
        legacy_obj = DynFreezeState.__new__(DynFreezeState)
        legacy_obj.__dict__.update(legacy_dict)

        dfc.load_state_dict(legacy_obj)
        assert dfc.frozen_block == [5, 4]
        assert dfc._released_at == {}
