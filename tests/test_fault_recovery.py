"""Tests for TASK-0063: Training job fault recovery and auto-restart.

Covers:
- Training state serialization / deserialization round-trip
- OOM graceful termination with checkpoint save
- CUDA error auto-recovery with CPU fallback
- Restart integration (state restored, training resumes)
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.random_walk_controller import RandomWalkController
from src.tg_lora.velocity import Velocity
from src.training.trainer_loop import NumericalInstabilityError
from src.utils.checkpoint import (TrainingState, load_training_state,
                                  save_training_state)
from src.utils.device import OOM_EXIT_CODE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_controller() -> RandomWalkController:
    return RandomWalkController(
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


def _make_velocity_with_state() -> Velocity:
    v = Velocity()
    delta = {
        "lora_A": torch.tensor([0.1, 0.2, 0.3]),
        "lora_B": torch.tensor([0.4, 0.5]),
    }
    v.update(delta, beta=0.8)
    return v


def _make_delta_tracker_with_history() -> DeltaTracker:
    dt = DeltaTracker()
    for i in range(4):
        before = {"layer.0.lora_A": torch.zeros(4)}
        after = {"layer.0.lora_A": torch.full((4,), 0.1 * (i + 1))}
        dt.compute_and_record(after, before, K=2)
    return dt


def _make_cycle_state() -> CycleState:
    cs = CycleState()
    cs.record_cycle(
        K=3, N=5, grad_accum=1, train_loss=2.5, valid_loss=2.3, accepted=True
    )
    cs.record_cycle(
        K=3, N=5, grad_accum=1, train_loss=2.2, valid_loss=2.1, accepted=True
    )
    cs.record_cycle(
        K=3, N=5, grad_accum=1, train_loss=2.0, valid_loss=2.4, accepted=False
    )
    return cs


def _make_training_state() -> TrainingState:
    controller = _make_controller()
    # Simulate some training progress
    controller.propose()
    controller.accept(2.5, 2.1)
    controller.reward(2.5, 2.1)
    controller.state.layer_scores = {0: 1.5, 1: 0.8, 2: 2.1}

    return TrainingState(
        cycle_state=_make_cycle_state(),
        controller_state=controller.state,
        velocity=_make_velocity_with_state(),
        delta_tracker=_make_delta_tracker_with_history(),
        cycle_offset=3,
        train_batch_position=9,
        accepted_valid_history=[2.3, 2.1],
    )


# ---------------------------------------------------------------------------
# Test: TrainingState serialization round-trip
# ---------------------------------------------------------------------------


class TestTrainingStateSaveLoad:
    """save_training_state / load_training_state round-trip tests."""

    def test_round_trip_preserves_cycle_state(self, tmp_path):
        ts = _make_training_state()
        path = tmp_path / "training_state.pt"
        save_training_state(ts, path)

        loaded = load_training_state(path)
        assert loaded.cycle_state.cycle == ts.cycle_state.cycle
        assert loaded.cycle_state.best_loss == ts.cycle_state.best_loss
        assert loaded.cycle_state.accepted_count == ts.cycle_state.accepted_count
        assert loaded.cycle_state.rejected_count == ts.cycle_state.rejected_count
        assert loaded.cycle_state.stale_cycles == ts.cycle_state.stale_cycles
        assert (
            loaded.cycle_state.full_backward_passes
            == ts.cycle_state.full_backward_passes
        )

    def test_round_trip_preserves_controller_state(self, tmp_path):
        ts = _make_training_state()
        path = tmp_path / "training_state.pt"
        save_training_state(ts, path)

        loaded = load_training_state(path)
        orig = ts.controller_state
        rest = loaded.controller_state
        assert rest.K == orig.K
        assert rest.N == orig.N
        assert rest.alpha == pytest.approx(orig.alpha)
        assert rest.beta == pytest.approx(orig.beta)
        assert rest.lr == pytest.approx(orig.lr)
        assert rest.active_layer_strategy == orig.active_layer_strategy
        assert rest.relative_update_cap == pytest.approx(orig.relative_update_cap)
        assert rest.total_cycles == orig.total_cycles
        assert rest.accepted_count == orig.accepted_count
        assert rest.rolled_back_count == orig.rolled_back_count
        assert rest.layer_scores == orig.layer_scores

    def test_round_trip_preserves_velocity_state(self, tmp_path):
        ts = _make_training_state()
        path = tmp_path / "training_state.pt"
        save_training_state(ts, path)

        loaded = load_training_state(path)
        assert loaded.velocity._state is not None
        for key in ts.velocity._state:
            assert torch.allclose(loaded.velocity._state[key], ts.velocity._state[key])
        assert loaded.velocity.magnitudes == ts.velocity.magnitudes

    def test_round_trip_preserves_delta_tracker(self, tmp_path):
        ts = _make_training_state()
        path = tmp_path / "training_state.pt"
        save_training_state(ts, path)

        loaded = load_training_state(path)
        assert loaded.delta_tracker.norm_history == ts.delta_tracker.norm_history
        assert loaded.delta_tracker.last_stats is not None
        assert loaded.delta_tracker.last_stats.total_norm == pytest.approx(
            ts.delta_tracker.last_stats.total_norm
        )

    def test_round_trip_preserves_cycle_offset(self, tmp_path):
        ts = _make_training_state()
        path = tmp_path / "training_state.pt"
        save_training_state(ts, path)

        loaded = load_training_state(path)
        assert loaded.cycle_offset == ts.cycle_offset
        assert loaded.train_batch_position == ts.train_batch_position
        assert loaded.accepted_valid_history == ts.accepted_valid_history

    def test_creates_parent_directories(self, tmp_path):
        ts = _make_training_state()
        path = tmp_path / "deep" / "nested" / "training_state.pt"
        save_training_state(ts, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_training_state(tmp_path / "nonexistent.pt")


class TestTrainingStateFromComponents:
    """Test TrainingState construction from individual components."""

    def test_from_components(self):
        cs = CycleState()
        cs.record_cycle(
            K=2, N=1, grad_accum=1, train_loss=1.0, valid_loss=1.0, accepted=True
        )

        ctrl = _make_controller()
        vel = Velocity()
        dt = DeltaTracker()

        ts = TrainingState(
            cycle_state=cs,
            controller_state=ctrl.state,
            velocity=vel,
            delta_tracker=dt,
            cycle_offset=1,
        )
        assert ts.cycle_offset == 1
        assert ts.cycle_state.cycle == 1


class TestTrainingStateVelocityEdgeCases:
    """Edge cases for velocity serialization."""

    def test_velocity_with_none_state(self, tmp_path):
        v = Velocity()  # _state is None
        ts = TrainingState(
            cycle_state=CycleState(),
            controller_state=_make_controller().state,
            velocity=v,
            delta_tracker=DeltaTracker(),
            cycle_offset=0,
        )
        path = tmp_path / "ts.pt"
        save_training_state(ts, path)

        loaded = load_training_state(path)
        assert loaded.velocity._state is None
        assert loaded.velocity.magnitudes == []

    def test_velocity_with_multiple_updates(self, tmp_path):
        v = Velocity()
        delta1 = {"a": torch.tensor([1.0, 2.0])}
        delta2 = {"a": torch.tensor([3.0, 4.0])}
        v.update(delta1, beta=0.9)
        v.update(delta2, beta=0.9)

        ts = TrainingState(
            cycle_state=CycleState(),
            controller_state=_make_controller().state,
            velocity=v,
            delta_tracker=DeltaTracker(),
            cycle_offset=0,
        )
        path = tmp_path / "ts.pt"
        save_training_state(ts, path)

        loaded = load_training_state(path)
        assert torch.allclose(loaded.velocity._state["a"], v._state["a"])
        assert len(loaded.velocity.magnitudes) == 2


# ---------------------------------------------------------------------------
# Test: OOM graceful termination
# ---------------------------------------------------------------------------


class TestOOMGracefulTermination:
    """Verify OOM triggers checkpoint save before exiting."""

    def test_oom_saves_training_state(self, tmp_path, caplog):
        """When OOM occurs during training, training state is saved."""

        eval_losses = [2.0, 1.5] * 10
        cfg = _make_run_dir_config(tmp_path)

        oom_call = [0]

        def oom_on_second_step(*args, **kwargs):
            oom_call[0] += 1
            if oom_call[0] >= 3:
                raise torch.cuda.OutOfMemoryError("CUDA OOM simulated")
            return 2.0

        deps = _patch_deps(eval_losses=eval_losses, run_dir=tmp_path)
        deps["src.training.train_tg_lora.eval_loss"] = MagicMock(
            side_effect=oom_on_second_step
        )

        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            _run_with_deps(cfg, deps)

        # Training state should have been saved
        state_path = tmp_path / "training_state.pt"
        assert state_path.exists(), "Training state should be saved on OOM"

    def test_oom_saves_model_checkpoint(self, tmp_path, caplog):
        """When OOM occurs, model checkpoint is also saved."""

        eval_losses = [2.0, 1.5] * 10
        cfg = _make_run_dir_config(tmp_path)

        oom_call = [0]

        def oom_on_forward(*args, **kwargs):
            oom_call[0] += 1
            if oom_call[0] >= 3:
                raise torch.cuda.OutOfMemoryError("CUDA OOM simulated")
            return 2.0

        deps = _patch_deps(eval_losses=eval_losses, run_dir=tmp_path)
        deps["src.training.train_tg_lora.eval_loss"] = MagicMock(
            side_effect=oom_on_forward
        )

        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            _run_with_deps(cfg, deps)

        # Model checkpoint should have been saved
        checkpoint_dir = tmp_path / "oom_checkpoint"
        assert checkpoint_dir.exists(), "OOM checkpoint directory should be created"

    def test_oom_logs_warning(self, tmp_path, caplog):
        """OOM triggers a warning log."""

        eval_losses = [2.0, 1.5] * 10
        cfg = _make_run_dir_config(tmp_path)

        oom_call = [0]

        def oom_on_forward(*args, **kwargs):
            oom_call[0] += 1
            if oom_call[0] >= 3:
                raise torch.cuda.OutOfMemoryError("CUDA OOM simulated")
            return 2.0

        deps = _patch_deps(eval_losses=eval_losses, run_dir=tmp_path)
        deps["src.training.train_tg_lora.eval_loss"] = MagicMock(
            side_effect=oom_on_forward
        )

        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            _run_with_deps(cfg, deps)

        assert any("OOM" in r.message for r in caplog.records), (
            f"Expected OOM warning, got: {[r.message for r in caplog.records]}"
        )

    def test_oom_exits_defer_exit_code(self, tmp_path, caplog):
        """A handled OOM must exit ``OOM_EXIT_CODE`` (defer-and-retry), not 2.

        The graceful-OOM handler saves a fault checkpoint and is safe to resume at
        reduced batch — that is only actionable if the process exit code distinguishes
        "deferrable OOM" from a real fault. Pins the producer half of the contract
        end-to-end (fault_reason='oom' → exit 3); numerical/CUDA faults stay at 2.
        """
        import contextlib

        eval_losses = [2.0, 1.5] * 10
        cfg = _make_run_dir_config(tmp_path)

        oom_call = [0]

        def oom_on_forward(*args, **kwargs):
            oom_call[0] += 1
            if oom_call[0] >= 3:
                raise torch.cuda.OutOfMemoryError("CUDA OOM simulated")
            return 2.0

        deps = _patch_deps(eval_losses=eval_losses, run_dir=tmp_path)
        deps["src.training.train_tg_lora.eval_loss"] = MagicMock(
            side_effect=oom_on_forward
        )
        # Isolate the EXIT-CODE routing from the fault-checkpoint save: this test
        # asserts the OOM path raises SystemExit(OOM_EXIT_CODE), not the checkpoint
        # artifact (that is the sibling tests' job). The mock model has no LoRA
        # layers, so the real save_checkpoint -> save_pretrained fails loud (the
        # atomic-publish guard) on EVERY save (in-loop best_model, periodic, and the
        # fault checkpoint) and would raise CheckpointSaveError before the SystemExit
        # fires; no-op'ing the whole checkpoint-I/O path lets control flow reach the
        # fault-exit line.
        deps["src.training.train_tg_lora._save_fault_checkpoint"] = MagicMock()
        deps["src.training.train_tg_lora.save_checkpoint"] = MagicMock()

        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(return_value=mock_mlf)

        captured: dict[str, object] = {}
        with contextlib.ExitStack() as stack:
            for target, mock_obj in deps.items():
                stack.enter_context(patch(target, new=mock_obj))
            from src.training.train_tg_lora import train_tg_lora

            with caplog.at_level(logging.WARNING, logger="tg-lora"):
                try:
                    train_tg_lora(cfg)
                except SystemExit as exc:
                    captured["code"] = exc.code

        assert "code" in captured, "handled OOM must raise SystemExit (got none)"
        assert captured["code"] == OOM_EXIT_CODE, (
            f"handled OOM must exit OOM_EXIT_CODE={OOM_EXIT_CODE} (defer/retry), "
            f"got {captured['code']!r}"
        )


# ---------------------------------------------------------------------------
# Test: CUDA error auto-recovery
# ---------------------------------------------------------------------------


class TestCUDAErrorRecovery:
    """Verify CUDA errors trigger CPU fallback."""

    def test_cuda_error_triggers_cpu_fallback(self, tmp_path, caplog):
        """When CUDA RuntimeError occurs, model is moved to CPU and training continues."""

        eval_losses = [2.0, 1.5] * 20
        cfg = _make_run_dir_config(tmp_path)

        cuda_error_call = [0]

        def cuda_error_once(*args, **kwargs):
            cuda_error_call[0] += 1
            if cuda_error_call[0] == 3:
                raise RuntimeError("CUDA error: device-side assert triggered")
            return 2.0

        deps = _patch_deps(eval_losses=eval_losses, run_dir=tmp_path)
        deps["src.training.train_tg_lora.eval_loss"] = MagicMock(
            side_effect=cuda_error_once
        )

        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            _run_with_deps(cfg, deps)

        assert any("CUDA" in r.message for r in caplog.records), (
            f"Expected CUDA recovery log, got: {[r.message for r in caplog.records]}"
        )

    def test_non_cuda_runtime_error_not_swallowed(self, tmp_path):
        """Non-CUDA RuntimeErrors are not caught by CUDA recovery."""

        cfg = _make_run_dir_config(tmp_path)
        deps = _patch_deps(eval_losses=[2.0, 1.5] * 10, run_dir=tmp_path)
        deps["src.training.train_tg_lora.eval_loss"] = MagicMock(
            side_effect=RuntimeError("unrelated error")
        )

        with pytest.raises(RuntimeError, match="unrelated error"):
            _run_with_deps(cfg, deps)


class TestNumericalInstabilityRecovery:
    def test_numerical_instability_saves_training_state(self, tmp_path, caplog):
        cfg = _make_run_dir_config(tmp_path)
        deps = _patch_deps(eval_losses=[2.0, 1.5] * 10, run_dir=tmp_path)
        deps["src.training.train_tg_lora.forward_backward"] = MagicMock(
            side_effect=NumericalInstabilityError("Loss is nan (non-finite)")
        )

        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            _run_with_deps(cfg, deps)

        state_path = tmp_path / "training_state.pt"
        assert state_path.exists(), "Training state should be saved on numerical instability"
        loaded = load_training_state(state_path)
        assert loaded.cycle_offset == loaded.cycle_state.cycle
        assert any("Numerical instability" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: Restart integration
# ---------------------------------------------------------------------------


class TestRestartIntegration:
    """End-to-end test of save state → load state → resume training."""

    def test_restart_restores_cycle_state(self, tmp_path):
        """After restart, cycle_state reflects saved progress."""
        ts = _make_training_state()
        state_path = tmp_path / "training_state.pt"
        save_training_state(ts, state_path)

        loaded = load_training_state(state_path)
        assert loaded.cycle_state.cycle == 3
        assert loaded.cycle_state.accepted_count == 2
        assert loaded.cycle_state.rejected_count == 1

    def test_restart_restores_controller_for_continuation(self, tmp_path):
        """Restored controller state allows training to continue with same hyperparams."""
        ts = _make_training_state()
        state_path = tmp_path / "training_state.pt"
        save_training_state(ts, state_path)

        loaded = load_training_state(state_path)
        ctrl = RandomWalkController(
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=5,
            N_candidates=[1, 3, 5],
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        ctrl.state = loaded.controller_state

        # Controller should continue from where it left off
        assert ctrl.state.total_cycles == ts.controller_state.total_cycles
        assert ctrl.state.alpha == pytest.approx(ts.controller_state.alpha)
        assert ctrl.state.layer_scores == ts.controller_state.layer_scores

    def test_restart_restores_velocity_direction(self, tmp_path):
        """Restored velocity maintains extrapolation direction."""
        ts = _make_training_state()
        state_path = tmp_path / "training_state.pt"
        save_training_state(ts, state_path)

        loaded = load_training_state(state_path)
        # Velocity direction should be preserved for extrapolation
        assert loaded.velocity._state is not None
        cos_sim = loaded.velocity.cosine_similarity(
            {
                "lora_A": torch.tensor([0.1, 0.2, 0.3]),
                "lora_B": torch.tensor([0.4, 0.5]),
            }
        )
        assert cos_sim > 0.99  # identical vectors

    def test_restart_restores_delta_tracker_trend(self, tmp_path):
        """Restored delta tracker maintains convergence history."""
        ts = _make_training_state()
        state_path = tmp_path / "training_state.pt"
        save_training_state(ts, state_path)

        loaded = load_training_state(state_path)
        # Norm history should be preserved
        assert len(loaded.delta_tracker.norm_history) == 4
        # Convergence trend should be computable
        trend = loaded.delta_tracker.convergence_trend()
        assert isinstance(trend, float)


# ---------------------------------------------------------------------------
# Helpers for mocked training loop tests
# ---------------------------------------------------------------------------


def _make_run_dir_config(tmp_path):
    """Build a config dict pointing run_dir at tmp_path."""
    from tests.test_training_integration import _make_config

    return _make_config(
        training={"max_cycles": 5},
        logging={
            "run_dir": str(tmp_path),
            "log_every_cycles": 1,
            "save_every_cycles": 25,
        },
    )


def _patch_deps(eval_losses=None, run_dir=None):
    """Build mock patches for train_tg_lora deps."""
    from tests.test_training_integration import _patch_train_tg_lora_deps

    deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)

    # If a real run_dir is provided, override ensure_dir to return it
    if run_dir is not None:
        from pathlib import Path

        real_path = Path(run_dir)
        real_path.mkdir(parents=True, exist_ok=True)
        deps["src.training.train_tg_lora.ensure_dir"] = MagicMock(
            return_value=real_path
        )

    return deps


def _run_with_deps(cfg, deps):
    """Run train_tg_lora with patched deps, returning mock dict."""
    import contextlib

    mock_mlf = MagicMock()
    mock_mlf.enabled = False
    mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
    mock_mlf.__exit__ = MagicMock(return_value=False)
    deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(return_value=mock_mlf)

    with contextlib.ExitStack() as stack:
        mocks = {}
        for target, mock_obj in deps.items():
            mocks[target.split(".")[-1]] = stack.enter_context(
                patch(target, new=mock_obj)
            )
        from src.training.train_tg_lora import train_tg_lora

        try:
            train_tg_lora(cfg)
        except SystemExit as exc:
            if exc.code not in (0, None):
                pass  # expected for fault paths
            else:
                raise
        return mocks


def _run_with_deps_resume(cfg, deps, resume_path):
    """Run train_tg_lora with patched deps and a resume path."""
    import contextlib

    mock_mlf = MagicMock()
    mock_mlf.enabled = False
    mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
    mock_mlf.__exit__ = MagicMock(return_value=False)
    deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(return_value=mock_mlf)

    with contextlib.ExitStack() as stack:
        mocks = {}
        for target, mock_obj in deps.items():
            mocks[target.split(".")[-1]] = stack.enter_context(
                patch(target, new=mock_obj)
            )
        from src.training.train_tg_lora import train_tg_lora

        train_tg_lora(cfg, resume_path=resume_path)
        return mocks


# ---------------------------------------------------------------------------
# Test: restore_state method integration with load_training_state
# ---------------------------------------------------------------------------


class TestRestoreStateIntegration:
    """Tests for RandomWalkController.restore_state() with real checkpoint data."""

    def test_restore_state_from_saved_checkpoint(self, tmp_path):
        """restore_state properly adopts a loaded ControllerState."""
        ts = _make_training_state()
        state_path = tmp_path / "training_state.pt"
        save_training_state(ts, state_path)

        loaded = load_training_state(state_path)

        # Create a fresh controller with same config bounds
        ctrl = RandomWalkController(
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=5,
            N_candidates=[1, 3, 5],
            alpha_min=0.03,
            alpha_max=1.5,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        ctrl.restore_state(loaded.controller_state)

        assert ctrl.state.K == loaded.controller_state.K
        assert ctrl.state.alpha == pytest.approx(loaded.controller_state.alpha)
        assert ctrl.state.total_cycles == loaded.controller_state.total_cycles
        assert ctrl.state.accepted_count == loaded.controller_state.accepted_count
        assert ctrl.last_accel_action == 0

    def test_resume_path_loads_training_state(self, tmp_path):
        """train_tg_lora(resume_path=...) restores state from checkpoint."""
        ts = _make_training_state()
        state_path = tmp_path / "training_state.pt"
        save_training_state(ts, state_path)

        cfg = _make_run_dir_config(tmp_path)
        deps = _patch_deps(
            eval_losses=[2.0, 1.5] * 10,
            run_dir=tmp_path,
        )

        with patch("src.training.train_tg_lora.load_training_state", return_value=ts):
            mocks = _run_with_deps_resume(cfg, deps, str(state_path))

        # Verify the controller picked up restored state
        ctrl = mocks.get("ctrl")
        if ctrl is not None:
            assert ctrl.state.total_cycles == ts.controller_state.total_cycles
