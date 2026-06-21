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

from src.model.lora_utils import iter_all_lora_params_by_layer
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


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


def _set_lora_B_identity(model: _Model, idx: int, scale: float) -> None:
    """Set layer ``idx``'s ``lora_B := scale·I`` so ``A_fro = scale·‖A‖_F``.

    Gives a controlled, layer-local ``‖BA‖_F`` magnitude without training — used
    to construct negligible vs legitimately-small series for ``a_mask_ratio``
    tests. ``lora_B`` is square (hidden×hidden) so the identity is exact.
    """
    layer = model.layers[idx]
    with torch.no_grad():
        layer.lora_B.copy_(torch.eye(layer.lora_A.shape[0]) * scale)


def _series_A_fro(model: _Model, idx: int) -> float:
    """Current ``‖B @ A‖_F`` for layer ``idx`` (the same trace-trick ``compute_r_A`` uses)."""
    a = model.layers[idx].lora_A.detach().float()
    b = model.layers[idx].lora_B.detach().float()
    return torch.trace((b.T @ b) @ (a @ a.T)).sqrt().item()


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


# ---------------------------------------------------------------------------
# E2E: real forward/backward/optimizer step — the §4 runtime property no
# timer-state unit test can prove (a released layer actually re-trains)
# ---------------------------------------------------------------------------


class TestEndToEndReversibleRelease:
    """End-to-end behavior under a *real* PyTorch training step.

    Every other test in this file hand-seeds ``_r_A_history`` /
    ``_frozen_block`` / ``_frozen_since_cycle`` and asserts on controller *state*
    (timer values, block contents). None ever runs a backward pass, so the §4
    promise that a released LoRA module *resumes gradient updates* was only ever
    inferred from state, never observed at the weight level. A real
    ``model(x).sum().backward(); optimizer.step()`` is what actually proves it:
    a frozen param receives no grad (the optimizer skips it) while a released
    one accumulates grad and moves. See ``docs/design/10_guard_experiment.md``
    §4 ("解放ロジック … 可逆").
    """

    @staticmethod
    def _train_step(model, optimizer):
        """One real SGD step over whatever is currently trainable."""
        x = torch.randn(4, 8)
        optimizer.zero_grad(set_to_none=True)
        model(x).sum().backward()
        optimizer.step()

    @staticmethod
    def _layer_params(model, idx):
        return dict(iter_all_lora_params_by_layer(model)[idx])

    def test_frozen_holds_then_release_resumes_gradient_updates(self):
        """The headline §4 property, proven at the weight level.

        (1) While frozen, layer 3 receives *no* grad and is bit-identical after a
        real step; (2) the §4(a) stir path releases L3 via the real ``decide``
        code; (3) on the very next real step L3's ``lora_B`` accumulates grad and
        moves (resumes updates), and ``lora_A`` follows once B != 0; (4) the
        still-frozen output-side layer 5 never moves — the freeze is real, not
        just a flag, and so is the release.
        """
        torch.manual_seed(0)
        model = _build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
        dfc = _make(stir_interval=10)

        # Freeze the output-side block via the real apply path; trainable = {0,1,2}.
        dfc.apply_freeze(model, [5, 4, 3], cycle=0)
        assert dfc.frozen_block == [5, 4, 3]

        p3 = self._layer_params(model, 3)  # just-frozen upstream end
        p5 = self._layer_params(model, 5)  # still-frozen output side
        snap3 = {k: v.detach().clone() for k, v in p3.items()}
        snap5_b = p5["layers.5.lora_B"].detach().clone()
        snap5_a = p5["layers.5.lora_A"].detach().clone()

        # --- (1) While frozen: a real step must leave layer 3 untouched. ---
        self._train_step(model, optimizer)
        for name, p in p3.items():
            assert p.grad is None, f"{name} received a grad while frozen"
            assert torch.equal(p, snap3[name]), f"{name} moved while frozen"

        # --- (2) §4(a) stir: real decide path releases the upstream end (L3). ---
        released = dfc.decide_unfreeze(cycle=10)  # 10 - 0 >= R=10
        assert released == [3]
        dfc.apply_unfreeze(model, released, cycle=10)  # requires_grad := True
        assert dfc.frozen_block == [5, 4]
        for p in p3.values():
            assert p.requires_grad

        # --- (3) After release: a real step must drive layer 3 again. ---
        # lora_B sits downstream of A so it gets a nonzero grad at once; lora_A
        # only once B != 0, hence the second step. Both must move.
        self._train_step(model, optimizer)
        assert p3["layers.3.lora_B"].grad is not None
        assert p3["layers.3.lora_B"].grad.abs().sum() > 0
        assert not torch.equal(p3["layers.3.lora_B"], snap3["layers.3.lora_B"]), (
            "released layer's lora_B did not move — §4 release is inert at the "
            "weight level"
        )
        self._train_step(model, optimizer)  # now B != 0 → A gets a real grad too
        assert not torch.equal(p3["layers.3.lora_A"], snap3["layers.3.lora_A"]), (
            "released layer's lora_A did not resume training"
        )

        # --- (4) Contrast: the still-frozen output-side layer 5 never moved. ---
        assert torch.equal(p5["layers.5.lora_B"], snap5_b)
        assert torch.equal(p5["layers.5.lora_A"], snap5_a)

    def test_real_compute_rA_loop_respects_release_cooldown(self):
        """Drive the trainer's real per-cycle order
        (``compute_r_A → decide_unfreeze → apply_unfreeze → decide_freeze →
        apply_freeze``) over real SGD steps — no seeded history.

        Proves (a) ``compute_r_A`` is a genuine signal: layers that actually
        trained report r_A > 0 while frozen ones report exactly 0.0; and (b)
        right after a §4 release the just-released layer's history is empty
        (reads as "quiet"), yet ``decide_freeze`` does *not* re-freeze it — the
        release cooldown, not a hand-seeded fixture, is what holds it for a full
        window. ``a_mask_ratio=0`` disables the early-cycle masking heuristic so
        the r_A signal is uncluttered.
        """
        torch.manual_seed(0)
        model = _build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
        window = 3
        dfc = _make(window=window, stir_interval=window + 1, a_mask_ratio=0.0)

        def cycle_step(cycle):
            self._train_step(model, optimizer)
            return dfc.compute_r_A(model, cycle)

        # Warm-up: train every layer so real r_A history accumulates (all noisy).
        for c in range(window):
            r_A = cycle_step(c)
        assert all(r_A[i] > 0 for i in (0, 1, 2)), (
            f"compute_r_A not reporting real weight movement: {[r_A[i] for i in (0,1,2)]}"
        )

        # Freeze the output block; now only {0,1,2} train (r_A > 0) while the
        # frozen {3,4,5} record exactly 0.0 via compute_r_A's frozen branch.
        dfc.apply_freeze(model, [5, 4, 3], cycle=window)
        r_A = cycle_step(window)
        assert all(r_A[i] == 0.0 for i in (3, 4, 5))
        assert all(r_A[i] > 0 for i in (0, 1, 2))

        # Hold until §4(a) stir fires (frozen >= R cycles) → release L3.
        release_cycle = window + dfc._stir_interval
        for c in range(window + 1, release_cycle + 1):
            cycle_step(c)
        assert dfc.decide_unfreeze(release_cycle) == [3]
        dfc.apply_unfreeze(model, [3], cycle=release_cycle)

        # Trainer order runs decide_freeze in the SAME cycle right after the
        # release. L3's history was just popped (empty → reads quiet), so without
        # the cooldown §3 would re-freeze it here. The cooldown must hold it.
        assert dfc.decide_freeze(release_cycle) == [], (
            "just-released L3 re-frozen in the release cycle against real history "
            "— release cooldown not holding"
        )
        # Still protected one cycle later (1 < window).
        cycle_step(release_cycle + 1)
        assert dfc.decide_freeze(release_cycle + 1) == []

    def test_stir_does_not_fire_before_interval_under_real_steps(self):
        """Negative E2E for the §4(a) stir timer — the false-positive side the
        headline test above leaves open.

        With the block frozen and the §4(b) activity trigger taken out of the
        equation (huge ``upstream_activity_factor``), the stir release must NOT
        fire a single cycle early, and the frozen layers must receive *no*
        gradient across every one of those cycles (the freeze is behaviorally
        stable, not merely a flag ``decide_unfreeze`` declines to clear). The
        on-time release exactly at ``R`` is the positive control proving the
        timer is *not early*, not broken.
        """
        torch.manual_seed(0)
        model = _build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
        # §4(b) threshold := τ · 1e9 → unreachable; isolates the §4(a) timer.
        dfc = _make(stir_interval=10, upstream_activity_factor=1e9)
        dfc.apply_freeze(model, [5, 4, 3], cycle=0)
        assert dfc.frozen_block == [5, 4, 3]

        frozen = {idx: self._layer_params(model, idx) for idx in (3, 4, 5)}
        snaps = {
            idx: {name: p.detach().clone() for name, p in ps.items()}
            for idx, ps in frozen.items()
        }

        # Cycles 1..9: held 1..9 cycles, all < R=10 → stir must stay armed, and
        # the frozen block must be inert at the weight level under real steps.
        for c in range(1, 10):
            self._train_step(model, optimizer)
            dfc.compute_r_A(model, c)  # real signal flows; behavioral, not timer-relevant
            assert dfc.decide_unfreeze(c) == [], (
                f"§4(a) stir fired early at cycle {c} (held {c} < R=10)"
            )
            for idx, ps in frozen.items():
                for name, p in ps.items():
                    assert p.grad is None, f"frozen L{idx} {name} got grad at cycle {c}"
                    assert torch.equal(p, snaps[idx][name]), (
                        f"frozen L{idx} {name} moved at cycle {c} — freeze not holding"
                    )

        # Positive control at the boundary: held exactly R cycles → releases on time.
        assert dfc.decide_unfreeze(10) == [3]


# ---------------------------------------------------------------------------
# Negative tests: the §4 release + a_mask_ratio guards must NOT over-fire on
# legitimate readings. The positive E2E tests prove the guards fire when they
# should; these pin their *false-positive rate* so a future commit cannot
# silently widen a guard (the in-repo analog of "the overclaim is caught but
# not its false-positive rate").
# ---------------------------------------------------------------------------


class TestGuardsDoNotOverFire:
    def test_quiet_upstream_does_not_trigger_activity_release(self):
        """§4(b) negative, isolated: with the stir timer well within its
        interval, a QUIET upstream neighbor (``r_A_window ≤ τ·factor``) must NOT
        trigger a release. The existing combined negative
        (``test_no_trigger_releases_nothing``) suppresses both triggers at once;
        this isolates §4(b) and pairs it with a noisy-upstream sanity check so
        the guard is proven *selective*, not just inert.
        """
        dfc = _make(stir_interval=1000, upstream_activity_factor=1.5)  # isolate §4(b)
        dfc._frozen_block = [5, 4, 3]
        dfc._frozen_since_cycle = 0
        _set_history(dfc, {li: 0.001 for li in range(NUM_LAYERS)})  # all quiet incl. L2
        # Upstream neighbor of the block is L2 = min([5,4,3]) - 1; window 0.001 vs
        # threshold τ·1.5 = 0.02·1.5 = 0.03 → quiet → no release.
        assert dfc.decide_unfreeze(4) == []
        # Sanity: the same setup with L2 noisy DOES release (guard works).
        dfc._r_A_history[2] = deque([0.05] * dfc._window, maxlen=dfc._window)
        assert dfc.decide_unfreeze(4) == [3]

    def test_a_mask_ratio_masks_negligible_series(self):
        """Positive control for the ``a_mask_ratio`` heuristic (``compute_r_A``
        zeroes ``r_A`` for series with ``‖BA‖_F < a_mask_ratio·median``): it DOES
        fire on a genuinely negligible series — even one that is moving."""
        torch.manual_seed(0)
        model = _build_model()
        dfc = _make(a_mask_ratio=0.1)  # default all_layer_indices = 0..5
        for i in (0, 1, 2, 3, 4):
            _set_lora_B_identity(model, i, scale=1.0)  # large → drives the median
        _set_lora_B_identity(model, 5, scale=0.01)  # negligible

        large = _series_A_fro(model, 0)
        tiny = _series_A_fro(model, 5)
        assert tiny < dfc._a_mask_ratio * large  # precondition: below mask threshold

        # Seed median + prior so the mask branch is active and every series is
        # "moving" (prior = half of current). Without the mask, L5 would report
        # a real r_A ≈ 1.0; the mask must zero it.
        dfc._median_A = large
        dfc._prev_A_fro = {
            f"layers.{i}": _series_A_fro(model, i) * 0.5 for i in range(NUM_LAYERS)
        }
        r_A = dfc.compute_r_A(model, 0)
        assert r_A[5] == 0.0, "negligible moving series not masked"
        assert r_A[0] > 0.0, "legitimate large series unexpectedly masked"

    def test_a_mask_ratio_does_not_mask_legitimately_small_moving_series(self):
        """False-positive pin for ``a_mask_ratio``: a series that is SMALL but
        above the mask threshold (``≥ a_mask_ratio·median``) and genuinely moving
        must report its real ``r_A > 0``, NOT be zeroed. A future commit that
        widens the heuristic (e.g. masks anything below the median) would silently
        suppress a legitimately training small series — this test fails first.
        """
        torch.manual_seed(0)
        model = _build_model()
        dfc = _make(a_mask_ratio=0.1)
        for i in (0, 1, 2, 3, 4):
            _set_lora_B_identity(model, i, scale=1.0)
        _set_lora_B_identity(model, 5, scale=0.5)  # small but legitimate (> threshold)

        large = _series_A_fro(model, 0)
        small = _series_A_fro(model, 5)
        assert small >= dfc._a_mask_ratio * large  # precondition: above mask threshold

        dfc._median_A = large
        dfc._prev_A_fro = {
            f"layers.{i}": _series_A_fro(model, i) * 0.5 for i in range(NUM_LAYERS)
        }
        r_A = dfc.compute_r_A(model, 0)
        assert r_A[5] > 0.0, "legitimately small moving series was masked (over-correction)"
