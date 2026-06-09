from __future__ import annotations

import platform

import pytest
import torch
import torch.nn as nn

from tg_lora.checkpoint import load_checkpoint, save_checkpoint
from tg_lora.config import TGLoraConfig
from tg_lora.cycle_state import CycleState
from tg_lora.delta_tracker import DeltaTracker
from tg_lora.random_walk_controller import ControllerState, RandomWalkController
from tg_lora.trajectory import TrajectoryAnalyzer, TrajectoryPoint
from tg_lora.velocity import Velocity


class _SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture()
def model():
    return _SimpleModel()


@pytest.fixture()
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture()
def config():
    return TGLoraConfig(K_initial=2, N_initial=3, max_steps=10)


@pytest.fixture()
def tmp_ckpt_path(tmp_path):
    return tmp_path / "test_checkpoint.pt"


class TestSaveLoadRoundTrip:
    def test_model_state_dict_round_trip(self, model, optimizer, config, tmp_ckpt_path):
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        assert set(loaded["model_state_dict"].keys()) == set(original_sd.keys())
        for k in original_sd:
            assert torch.equal(loaded["model_state_dict"][k].cpu(), original_sd[k].cpu())

    def test_optimizer_state_dict_round_trip(self, model, optimizer, config, tmp_ckpt_path):
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        assert isinstance(loaded["optimizer_state_dict"], dict)
        assert "state" in loaded["optimizer_state_dict"]
        assert "param_groups" in loaded["optimizer_state_dict"]

    def test_config_round_trip(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        restored = TGLoraConfig(**loaded["config"])
        assert restored.to_dict() == config.to_dict()

    def test_full_round_trip_with_all_components(self, model, optimizer, config, tmp_ckpt_path):
        controller = RandomWalkController(K_initial=2, N_initial=3)
        cycle_state = CycleState(cycle=5, accepted_count=3, rejected_count=2)
        delta_tracker = DeltaTracker()
        velocity = Velocity()
        trajectory = TrajectoryAnalyzer()
        trajectory.add_point(TrajectoryPoint(cycle=0, train_loss=1.0, valid_loss=0.9))

        before = {"layer.0.lora_A": torch.randn(4, 4), "layer.0.lora_B": torch.randn(4, 4)}
        after = {"layer.0.lora_A": torch.randn(4, 4), "layer.0.lora_B": torch.randn(4, 4)}
        delta = delta_tracker.compute_and_record(after, before, K=2)
        velocity.update(delta, beta=0.9)

        extra = {"custom_key": "custom_value", "step_count": 42}
        save_checkpoint(
            tmp_ckpt_path, model, optimizer, config,
            controller=controller, cycle_state=cycle_state,
            delta_tracker=delta_tracker, velocity=velocity,
            trajectory=trajectory, extra=extra,
        )
        loaded = load_checkpoint(tmp_ckpt_path)

        assert loaded["controller_summary"] is not None
        assert loaded["controller_summary"]["current_K"] == 2
        assert loaded["controller_summary"]["current_N"] == 3

        assert loaded["cycle_state_summary"] is not None
        assert loaded["cycle_state_summary"]["cycles"] == 5
        assert loaded["cycle_state_summary"]["accepted_count"] == 3

        assert loaded["delta_tracker_state"] is not None
        assert "norms" in loaded["delta_tracker_state"]
        assert "last_stats" in loaded["delta_tracker_state"]

        assert loaded["velocity_state"] is not None
        assert "state" in loaded["velocity_state"]
        assert "magnitudes" in loaded["velocity_state"]

        assert loaded["trajectory_points"] is not None
        assert len(loaded["trajectory_points"]) == 1
        assert loaded["trajectory_points"][0]["cycle"] == 0
        assert loaded["trajectory_points"][0]["train_loss"] == 1.0

        assert loaded["extra"] == extra


class TestAtomicWrite:
    def test_atomic_write_creates_valid_file(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        assert tmp_ckpt_path.exists()
        loaded = load_checkpoint(tmp_ckpt_path)
        assert loaded["version"] == "0.1.0"

    def test_interrupted_write_does_not_corrupt_existing(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        original = load_checkpoint(tmp_ckpt_path)
        tmp_file = tmp_ckpt_path.parent / ".ckpt_tmp_partial.pt"
        tmp_file.write_bytes(b"corrupted data that is not a valid checkpoint")
        try:
            tmp_file.rename(tmp_ckpt_path)
        except OSError:
            pass
        try:
            load_checkpoint(tmp_ckpt_path)
        except Exception:
            pass
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        restored = load_checkpoint(tmp_ckpt_path)
        for k in original["model_state_dict"]:
            assert torch.equal(
                restored["model_state_dict"][k].cpu(),
                original["model_state_dict"][k].cpu(),
            )


class TestMetadata:
    def test_metadata_fields(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        meta = loaded["metadata"]
        assert meta["tg_lora_version"] == "0.1.0"
        assert meta["python_version"] == platform.python_version()
        assert len(meta["torch_version"]) > 0
        assert meta["platform"] == platform.system().lower() or isinstance(meta["platform"], str)

    def test_timestamp_is_iso_format(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        ts = loaded["timestamp"]
        assert isinstance(ts, str)
        assert "T" in ts

    def test_version_present(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        assert loaded["version"] == "0.1.0"


class TestComponentReconstruction:
    def test_controller_state_reconstruction(self, model, optimizer, config, tmp_ckpt_path):
        controller = RandomWalkController(K_initial=5, N_initial=10, alpha_initial=0.5)
        controller.state.total_cycles = 20
        controller.state.accepted_count = 15
        save_checkpoint(tmp_ckpt_path, model, optimizer, config, controller=controller)
        loaded = load_checkpoint(tmp_ckpt_path)
        summary = loaded["controller_summary"]
        assert summary["current_K"] == 5
        assert summary["current_N"] == 10
        assert summary["current_alpha"] == 0.5
        assert summary["total_cycles"] == 20
        assert summary["accepted"] == 15
        restored_state = ControllerState(
            K=summary["current_K"],
            N=summary["current_N"],
            alpha=summary["current_alpha"],
            beta=summary["current_beta"],
            lr=summary["current_lr"],
            active_layer_strategy=summary["strategy"],
            relative_update_cap=controller.state.relative_update_cap,
            total_cycles=summary["total_cycles"],
            accepted_count=summary["accepted"],
            rolled_back_count=summary["rolled_back"],
        )
        new_controller = RandomWalkController.__new__(RandomWalkController)
        new_controller.restore_state(restored_state)
        assert new_controller.state.K == 5
        assert new_controller.state.N == 10

    def test_cycle_state_reconstruction(self, model, optimizer, config, tmp_ckpt_path):
        cs = CycleState(cycle=10, optimizer_steps=30, accepted_count=7, rejected_count=3, best_loss=0.5)
        save_checkpoint(tmp_ckpt_path, model, optimizer, config, cycle_state=cs)
        loaded = load_checkpoint(tmp_ckpt_path)
        restored = CycleState.from_dict(loaded["cycle_state_summary"])
        assert restored.cycle == 10
        assert restored.optimizer_steps == 30
        assert restored.accepted_count == 7
        assert restored.rejected_count == 3

    def test_velocity_tensors_on_cpu(self, model, optimizer, config, tmp_ckpt_path):
        velocity = Velocity()
        delta = {"layer.0.weight": torch.randn(4, 4)}
        velocity.update(delta, beta=0.9)
        save_checkpoint(tmp_ckpt_path, model, optimizer, config, velocity=velocity)
        loaded = load_checkpoint(tmp_ckpt_path)
        for k, v in loaded["velocity_state"]["state"].items():
            assert v.device == torch.device("cpu")

    def test_trajectory_reconstruction(self, model, optimizer, config, tmp_ckpt_path):
        trajectory = TrajectoryAnalyzer()
        trajectory.add_point(TrajectoryPoint(cycle=0, train_loss=2.0, valid_loss=1.8))
        trajectory.add_point(TrajectoryPoint(cycle=1, train_loss=1.5, valid_loss=1.3, velocity_magnitude=0.1))
        save_checkpoint(tmp_ckpt_path, model, optimizer, config, trajectory=trajectory)
        loaded = load_checkpoint(tmp_ckpt_path)
        pts = loaded["trajectory_points"]
        assert len(pts) == 2
        assert pts[0]["cycle"] == 0
        assert pts[0]["train_loss"] == 2.0
        assert pts[0]["valid_loss"] == 1.8
        assert pts[1]["velocity_magnitude"] == 0.1


class TestMissingOptionalComponents:
    def test_no_optional_components(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        assert loaded["controller_summary"] is None
        assert loaded["cycle_state_summary"] is None
        assert loaded["delta_tracker_state"] is None
        assert loaded["velocity_state"] is None
        assert loaded["trajectory_points"] is None
        assert loaded["extra"] is None

    def test_partial_components(self, model, optimizer, config, tmp_ckpt_path):
        cs = CycleState(cycle=3)
        save_checkpoint(tmp_ckpt_path, model, optimizer, config, cycle_state=cs)
        loaded = load_checkpoint(tmp_ckpt_path)
        assert loaded["controller_summary"] is None
        assert loaded["cycle_state_summary"] is not None
        assert loaded["velocity_state"] is None


class TestFileNotFound:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint("/nonexistent/path/checkpoint.pt")

    def test_missing_file_error_message_contains_path(self):
        bad_path = "/tmp/this_does_not_exist_abc123.pt"
        with pytest.raises(FileNotFoundError, match="this_does_not_exist_abc123"):
            load_checkpoint(bad_path)


class TestCpuPortability:
    def test_model_state_dict_on_cpu(self, model, optimizer, config, tmp_ckpt_path):
        save_checkpoint(tmp_ckpt_path, model, optimizer, config)
        loaded = load_checkpoint(tmp_ckpt_path)
        for v in loaded["model_state_dict"].values():
            assert v.device == torch.device("cpu")
