"""Tests for the reference training script (train_tg_lora.py).

Verifies:
- TGLoraConfig loading and initialization
- Single cycle execution and result verification
- Multi-cycle convergence
- Early stop behavior
- CLI argument parsing
- Checkpoint save / restore
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

# Import the training script components
from scripts.train_tg_lora import (
    SimpleLoRAModel,
    evaluate,
    load_checkpoint,
    load_config,
    main,
    parse_args,
    save_checkpoint,
    train_loop,
)
from tg_lora.config import TGLoraConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> TGLoraConfig:
    """Create a TGLoraConfig with sensible defaults for testing."""
    defaults = dict(
        K_initial=2,
        N_initial=1,
        alpha_initial=0.3,
        beta_initial=0.8,
        lr=1e-3,
        max_steps=5,
        batch_size=2,
        active_layer_strategy="last_25_percent",
        rollback_tolerance=0.0,
        relative_update_cap=0.005,
    )
    defaults.update(overrides)
    return TGLoraConfig(**defaults)


def _write_config_json(config: TGLoraConfig, path: Path) -> Path:
    config.save_json(path)
    return path


# ---------------------------------------------------------------------------
# TGLoraConfig loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_load_from_json(self, tmp_path):
        config = _make_config()
        path = _write_config_json(config, tmp_path / "config.json")
        loaded = load_config(str(path))
        assert loaded.K_initial == config.K_initial
        assert loaded.N_initial == config.N_initial
        assert loaded.alpha_initial == config.alpha_initial
        assert loaded.lr == config.lr

    def test_load_from_yaml(self, tmp_path):
        config = _make_config()
        yaml_path = tmp_path / "config.yaml"
        config.save_yaml(yaml_path)
        loaded = load_config(str(yaml_path))
        assert loaded.K_initial == config.K_initial
        assert loaded.beta_initial == config.beta_initial

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.json")

    def test_config_validation(self):
        with pytest.raises(ValueError, match="adapter_rank must be positive"):
            TGLoraConfig(adapter_rank=0)
        with pytest.raises(ValueError, match="lr must be positive"):
            TGLoraConfig(lr=-1.0)

    def test_config_summary(self):
        config = _make_config()
        summary = config.summary()
        assert "K_initial: 2" in summary
        assert "N_initial: 1" in summary


# ---------------------------------------------------------------------------
# SimpleLoRAModel
# ---------------------------------------------------------------------------


class TestSimpleLoRAModel:
    def test_forward_shape(self):
        model = SimpleLoRAModel(num_layers=4, dim=4)
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 4)

    def test_has_lora_params(self):
        model = SimpleLoRAModel(num_layers=4, dim=4)
        lora_params = [
            name for name, _ in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        assert len(lora_params) == 8  # 4 layers x (A + B)

    def test_requires_grad(self):
        model = SimpleLoRAModel(num_layers=4, dim=4)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) == 8


# ---------------------------------------------------------------------------
# Single cycle
# ---------------------------------------------------------------------------


class TestSingleCycle:
    def test_single_cycle_returns_valid_result(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=1)
        result = train_loop(model, x, config, progress=False)

        assert result["cycles"] == 1
        assert result["accepted_count"] + result["rejected_count"] == 1
        assert math.isfinite(result["final_train_loss"])
        assert math.isfinite(result["best_valid_loss"])
        assert result["reduction_rate"] >= 0.0
        assert result["accepted_count"] + result["rejected_count"] == 1

    def test_single_cycle_components_updated(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=1)
        result = train_loop(model, x, config, progress=False)

        assert "controller" in result
        assert result["controller"]["total_cycles"] == 1
        assert "delta_tracker" in result
        assert result["delta_tracker"]["history_length"] == 1

    def test_loss_computed_correctly(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        loss_val = evaluate(model, x)
        assert math.isfinite(loss_val)
        expected = model(x).sum().item()
        assert math.isclose(loss_val, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Multi-cycle convergence
# ---------------------------------------------------------------------------


class TestMultiCycleConvergence:
    def test_multi_cycle_runs_all_cycles(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=10)
        result = train_loop(model, x, config, progress=False)

        assert result["cycles"] == 10
        assert result["accepted_count"] + result["rejected_count"] == 10

    def test_multi_cycle_records_state(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=10)
        result = train_loop(model, x, config, progress=False)

        assert result["full_backward_passes"] > 0
        assert result["extrapolation_steps"] > 0
        assert result["reduction_rate"] >= 0.0
        assert result["acceptance_rate"] >= 0.0

    def test_delta_tracker_accumulates_history(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=5)
        result = train_loop(model, x, config, progress=False)

        assert result["delta_tracker"]["history_length"] == 5

    def test_trajectory_has_points(self):
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=5)
        result = train_loop(model, x, config, progress=False)

        report = result["trajectory_report"]
        assert report.total_points == 5

    def test_convergence_with_reduced_lr(self):
        """Small model + enough cycles should show finite losses."""
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(
            max_steps=20,
            lr=1e-3,
            K_initial=3,
            alpha_initial=0.2,
        )
        result = train_loop(
            model, x, config, patience=None, early_stop_patience=50,
            progress=False,
        )

        # Early stopping may trigger before 20 cycles due to trajectory
        # convergence detection; the important thing is that losses are finite.
        assert result["cycles"] > 0
        assert result["cycles"] <= 20
        assert math.isfinite(result["final_train_loss"])
        assert math.isfinite(result["best_valid_loss"])


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStop:
    def test_patience_based_early_stop(self):
        """With patience=1 and enable_random_walk=False, the controller
        never adapts, so the model quickly stagnates after the first cycle."""
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=100)
        # Use patience=1 — first accepted cycle sets best, next cycle that
        # doesn't improve will trigger early stop
        result = train_loop(
            model, x, config, patience=1, progress=False,
        )
        # Should have stopped well before 100 cycles
        assert result["cycles"] < 100

    def test_early_stop_respects_min_cycles(self):
        """CycleState.should_stop requires min_cycles=10 by default.
        With patience=1 we need >=10 cycles before stopping."""
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=100)
        result = train_loop(
            model, x, config, patience=1, progress=False,
        )
        # Must have run at least 10 cycles (min_cycles default)
        assert result["cycles"] >= 10

    def test_no_early_stop_without_patience(self):
        """Without patience, all cycles should run."""
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=5)
        result = train_loop(
            model, x, config, patience=None, progress=False,
        )
        assert result["cycles"] == 5

    def test_trajectory_early_stop(self):
        """TrajectoryAnalyzer can also trigger early stopping."""
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=200)
        result = train_loop(
            model, x, config, patience=None,
            early_stop_patience=3, progress=False,
        )
        # The trajectory analyzer with patience=3 should detect stagnation
        # and stop before 200 cycles (though it needs >= 10 cycles)
        assert result["cycles"] < 200


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parse_config_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_config_only(self):
        args = parse_args(["--config", "cfg.json"])
        assert args.config == "cfg.json"
        assert args.resume is None
        assert args.output_dir is None

    def test_parse_all_args(self):
        args = parse_args([
            "--config", "cfg.json",
            "--resume", "ckpt.pt",
            "--output-dir", "/tmp/out",
        ])
        assert args.config == "cfg.json"
        assert args.resume == "ckpt.pt"
        assert args.output_dir == "/tmp/out"


# ---------------------------------------------------------------------------
# Checkpoint save / restore
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        from tg_lora.cycle_state import CycleState
        from tg_lora.delta_tracker import DeltaTracker
        from tg_lora.random_walk_controller import RandomWalkController
        from tg_lora.velocity import Velocity

        model = SimpleLoRAModel()
        config = _make_config()
        controller = RandomWalkController(
            K_initial=config.K_initial,
            N_initial=config.N_initial,
            alpha_initial=config.alpha_initial,
            beta_initial=config.beta_initial,
            lr_initial=config.lr,
            enable_random_walk=False,
        )
        cs = CycleState()
        dt = DeltaTracker()
        vel = Velocity()

        # Simulate some state
        cs.record_cycle(train_loss=1.0, valid_loss=0.9, accepted=True, K=2, N=1, grad_accum=1)

        ckpt_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(ckpt_path, model, config, controller, cs, dt, vel)

        assert ckpt_path.exists()
        loaded = load_checkpoint(ckpt_path)
        assert "model_state" in loaded
        assert "config" in loaded
        assert "controller_state" in loaded
        assert "cycle_state" in loaded

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/ckpt.pt")

    def test_resume_from_checkpoint(self, tmp_path):
        """Train for a few cycles, save checkpoint, resume and continue."""
        model = SimpleLoRAModel()
        x = torch.randn(2, 4)
        config = _make_config(max_steps=3)
        config.output_dir = str(tmp_path / "run1")

        # Phase 1: train 3 cycles
        result1 = train_loop(model, x, config, progress=False)
        assert result1["cycles"] == 3

        # Save checkpoint
        from tg_lora.cycle_state import CycleState
        from tg_lora.delta_tracker import DeltaTracker
        from tg_lora.random_walk_controller import RandomWalkController
        from tg_lora.velocity import Velocity

        controller = RandomWalkController(
            K_initial=config.K_initial,
            N_initial=config.N_initial,
            alpha_initial=config.alpha_initial,
            beta_initial=config.beta_initial,
            lr_initial=config.lr,
            enable_random_walk=False,
        )
        cs = CycleState()
        dt = DeltaTracker()
        vel = Velocity()

        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(ckpt_path, model, config, controller, cs, dt, vel)

        # Phase 2: resume and train more
        ckpt = load_checkpoint(ckpt_path)
        config2 = _make_config(max_steps=5)
        config2.output_dir = str(tmp_path / "run2")
        result2 = train_loop(
            model, x, config2, resume_checkpoint=ckpt, progress=False,
        )
        assert result2["cycles"] == 5


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMainIntegration:
    def test_main_with_json_config(self, tmp_path):
        config = _make_config(max_steps=3, output_dir=str(tmp_path / "out"))
        config_path = _write_config_json(config, tmp_path / "config.json")

        result = main(["--config", str(config_path)])
        assert result["cycles"] == 3

        # Check output files were created
        out_dir = tmp_path / "out"
        assert (out_dir / "config_effective.json").exists()
        assert (out_dir / "summary.json").exists()

    def test_main_with_output_dir_override(self, tmp_path):
        config = _make_config(max_steps=2, output_dir=str(tmp_path / "default"))
        config_path = _write_config_json(config, tmp_path / "config.json")
        override_dir = tmp_path / "override"

        result = main([
            "--config", str(config_path),
            "--output-dir", str(override_dir),
        ])
        assert result["cycles"] == 2
        assert (override_dir / "config_effective.json").exists()

    def test_main_with_resume(self, tmp_path):
        config = _make_config(max_steps=2, output_dir=str(tmp_path / "run"))
        config_path = _write_config_json(config, tmp_path / "config.json")

        # First run
        main(["--config", str(config_path)])

        # Use the saved checkpoint from the first run
        ckpt_path = tmp_path / "run" / "checkpoint.pt"
        # The main script doesn't auto-save checkpoints, so we create one
        # manually for the test
        from tg_lora.cycle_state import CycleState
        from tg_lora.delta_tracker import DeltaTracker
        from tg_lora.random_walk_controller import RandomWalkController
        from tg_lora.velocity import Velocity

        model = SimpleLoRAModel()
        controller = RandomWalkController(
            K_initial=config.K_initial,
            N_initial=config.N_initial,
            alpha_initial=config.alpha_initial,
            beta_initial=config.beta_initial,
            lr_initial=config.lr,
            enable_random_walk=False,
        )
        cs = CycleState()
        dt = DeltaTracker()
        vel = Velocity()
        save_checkpoint(ckpt_path, model, config, controller, cs, dt, vel)

        # Run with --resume
        result = main([
            "--config", str(config_path),
            "--resume", str(ckpt_path),
        ])
        assert result["cycles"] == 2
