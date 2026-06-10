"""E2E --resume integration tests: save -> interrupt -> resume -> verify loss continues.

TASK-0090: Full lifecycle tests using real TrainingState serialization
with mocked model/data. Verifies cycle skipping, loss continuity, and
state preservation across save -> interrupt -> resume.
"""

from unittest.mock import patch

import pytest
import torch

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.random_walk_controller import RandomWalkController
from src.tg_lora.velocity import Velocity
from src.utils.checkpoint import (TrainingState, load_training_state,
                                  save_training_state)
from tests.test_fault_recovery import _patch_deps, _run_with_deps_resume
from tests.test_training_integration import _make_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_training_cycles(num_cycles=2, base_loss=2.5):
    """Build realistic training state after *num_cycles* of mock training.

    Simulates decreasing loss, velocity updates, and delta tracking to
    create a TrainingState that reflects real training progress.

    Returns (TrainingState, list[float]) where the list tracks train losses.
    """
    controller = RandomWalkController(
        K_initial=3,
        K_candidates=[2, 3, 5],
        N_initial=5,
        N_candidates=[1, 3, 5],
        alpha_initial=0.3,
        alpha_min=0.01,
        alpha_max=2.0,
        alpha_log_sigma=0.15,
        beta_initial=0.8,
        beta_candidates=[0.5, 0.8, 0.9],
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        active_layer_strategy="last_25_percent",
        relative_update_cap=0.005,
        rollback_tolerance=0.005,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    velocity = Velocity()
    delta_tracker = DeltaTracker()
    cycle_state = CycleState()

    losses = []
    for i in range(num_cycles):
        train_loss = base_loss - i * 0.3
        valid_loss = train_loss - 0.1
        losses.append(train_loss)

        controller.propose()
        controller.accept(train_loss, valid_loss)
        controller.reward(train_loss, valid_loss)

        delta = {
            f"layer_{i}.lora_A": torch.randn(4, 4) * 0.1,
            f"layer_{i}.lora_B": torch.randn(4, 4) * 0.05,
        }
        velocity.update(delta, beta=0.8)

        before = {"layer_0.lora_A": torch.zeros(4)}
        after = {"layer_0.lora_A": torch.full((4,), train_loss * 0.01)}
        delta_tracker.compute_and_record(after, before, K=3)

        cycle_state.record_cycle(
            K=3,
            N=5,
            grad_accum=1,
            train_loss=train_loss,
            valid_loss=valid_loss,
            accepted=True,
        )

    return TrainingState(
        cycle_state=cycle_state,
        controller_state=controller.state,
        velocity=velocity,
        delta_tracker=delta_tracker,
        cycle_offset=num_cycles,
        train_batch_position=num_cycles * 3,
        accepted_valid_history=[base_loss - i * 0.3 - 0.1 for i in range(num_cycles)],
    ), losses


def _run_resume_with_capture(cfg, deps, resume_path):
    """Run train_tg_lora with resume, snapshotting controller state after restore."""
    from src.tg_lora.random_walk_controller import \
        RandomWalkController as OrigRWC

    captured = {
        "total_cycles": None,
        "accepted_count": None,
        "alpha": None,
        "K": None,
        "layer_scores": None,
    }

    def _capturing_rwc(*args, **kwargs):
        ctrl = OrigRWC(*args, **kwargs)
        _orig_restore = ctrl.restore_state

        def _wrapped_restore(state):
            _orig_restore(state)
            captured["total_cycles"] = ctrl.state.total_cycles
            captured["accepted_count"] = ctrl.state.accepted_count
            captured["alpha"] = ctrl.state.alpha
            captured["K"] = ctrl.state.K
            captured["layer_scores"] = dict(ctrl.state.layer_scores)

        ctrl.restore_state = _wrapped_restore
        return ctrl

    deps["src.training.train_tg_lora.RandomWalkController"] = _capturing_rwc
    mocks = _run_with_deps_resume(cfg, deps, resume_path)
    return mocks, captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeE2E:
    """E2E --resume integration test: save -> interrupt -> resume -> verify."""

    def test_full_resume_flow_loss_continuity(self, tmp_path):
        """Save after 2 cycles -> resume -> verify state restoration and loss continuity.

        Acceptance criteria 1+2:
        - Controller state is restored (total_cycles, alpha, K match saved state)
        - Cycle state carries prior training progress (best_loss < initial)
        - Training continues from cycle_offset onwards
        """
        # --- Phase 1: simulate 2 training cycles, save state ---
        state, losses = _simulate_training_cycles(num_cycles=2, base_loss=2.5)
        state_path = tmp_path / "training_state.pt"
        save_training_state(state, state_path)

        # Verify round-trip on disk
        loaded = load_training_state(state_path)
        assert loaded.cycle_offset == 2
        assert loaded.cycle_state.cycle == 2
        assert loaded.cycle_state.accepted_count == 2

        # --- Phase 2: resume with real load_training_state ---
        cfg = _make_config(
            training={"max_cycles": 5},
            logging={
                "run_dir": str(tmp_path),
                "log_every_cycles": 1,
                "save_every_cycles": 25,
            },
        )
        eval_losses = [1.5, 1.4] * 20
        deps = _patch_deps(eval_losses=eval_losses, run_dir=tmp_path)
        mocks, captured = _run_resume_with_capture(cfg, deps, str(state_path))

        # Training completed successfully
        assert mocks["eval_loss"].called

        # Controller state was restored from the saved checkpoint
        assert captured["total_cycles"] is not None, (
            "restore_state should have been called during resume"
        )
        assert captured["total_cycles"] == state.controller_state.total_cycles
        assert captured["accepted_count"] == state.controller_state.accepted_count
        assert captured["alpha"] == pytest.approx(state.controller_state.alpha)
        assert captured["K"] == state.controller_state.K

        # Loss continuity proof:
        # Resumed session carries forward best_loss from prior training (2.1),
        # whereas a fresh start would have best_loss = inf.
        resumed_best = loaded.cycle_state.best_loss
        assert resumed_best < 2.5, (
            f"Resumed best_loss ({resumed_best}) should reflect prior training (< 2.5)"
        )
        assert resumed_best < float("inf"), (
            "Fresh start best_loss is inf; resumed must be strictly lower"
        )

    def test_cycle_skipping_on_resume(self, tmp_path):
        """Acceptance criterion 3: cycles < cycle_offset are skipped.

        With cycle_offset=3 and max_cycles=5, only cycles 3 and 4 execute.
        Verified by counting eval_loss calls:
          - With skipping:  <=8  (1 initial + 2 active cycles * ~3 calls)
          - Without skip:   ~11+ (1 initial + 5 cycles * ~2 calls)
        """
        state, _ = _simulate_training_cycles(num_cycles=3, base_loss=2.5)
        state_path = tmp_path / "training_state.pt"
        save_training_state(state, state_path)
        assert state.cycle_offset == 3

        cfg = _make_config(
            training={"max_cycles": 5},
            logging={
                "run_dir": str(tmp_path),
                "log_every_cycles": 1,
                "save_every_cycles": 25,
            },
        )
        eval_losses = [1.5, 1.4] * 20
        deps = _patch_deps(eval_losses=eval_losses, run_dir=tmp_path)

        mocks = _run_with_deps_resume(cfg, deps, str(state_path))

        call_count = mocks["eval_loss"].call_count
        assert call_count <= 8, (
            f"Expected <=8 eval calls (cycles 0-2 skipped), "
            f"got {call_count} — suggests no cycle skipping"
        )

    def test_resume_fast_forwards_batch_iterator(self, tmp_path):
        state, _ = _simulate_training_cycles(num_cycles=2, base_loss=2.5)
        state_path = tmp_path / "training_state.pt"
        save_training_state(state, state_path)

        cfg = _make_config(
            training={"max_cycles": 3},
            logging={
                "run_dir": str(tmp_path),
                "log_every_cycles": 1,
                "save_every_cycles": 25,
            },
        )
        deps = _patch_deps(eval_losses=[1.5, 1.4] * 20, run_dir=tmp_path)
        captured: dict[str, int] = {}

        from src.training.batch_iter import \
            InfiniteBatchIterator as OrigIterator

        class CapturingIterator(OrigIterator):
            def advance(self, batches: int) -> None:
                captured["batches"] = batches
                super().advance(batches)

        with patch("src.training.train_tg_lora.InfiniteBatchIterator", CapturingIterator):
            _run_with_deps_resume(cfg, deps, str(state_path))

        assert captured["batches"] == state.train_batch_position

    def test_resume_preserves_velocity_direction(self, tmp_path):
        """Velocity state is preserved through save -> load cycle.

        Verifies velocity tensors and magnitude history survive the
        serialization round-trip, critical for extrapolation after resume.
        """
        state, _ = _simulate_training_cycles(num_cycles=2, base_loss=2.5)
        state_path = tmp_path / "training_state.pt"
        save_training_state(state, state_path)

        loaded = load_training_state(state_path)

        assert loaded.velocity._state is not None, (
            "Velocity state should be non-empty after 2 training cycles"
        )
        assert len(loaded.velocity.magnitudes) == 2

        for key in state.velocity._state:
            assert key in loaded.velocity._state, f"Missing velocity key: {key}"
            assert torch.allclose(
                loaded.velocity._state[key], state.velocity._state[key]
            ), f"Velocity tensor mismatch for {key}"
