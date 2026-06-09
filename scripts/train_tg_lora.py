#!/usr/bin/env python
"""Reference training script for TG-LoRA.

Integrates all TG-LoRA components (Velocity, Extrapolator, RandomWalkController,
CycleState, RollbackManager, DeltaTracker, TrajectoryAnalyzer, LayerSampler,
LoRAUtils) into a complete training loop.

Uses SimpleLoRAModel (no HuggingFace/PEFT dependency) so the script can be
tested end-to-end without downloading large models.

Usage:
    python scripts/train_tg_lora.py --config config.json
    python scripts/train_tg_lora.py --config config.yaml
    python scripts/train_tg_lora.py --config config.json --resume checkpoint.pt
    python scripts/train_tg_lora.py --config config.json --output-dir runs/my_run
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from tg_lora.config import TGLoraConfig
from tg_lora.cycle_state import CycleState
from tg_lora.delta_tracker import DeltaTracker
from tg_lora.extrapolator import apply_extrapolation
from tg_lora.layer_sampler import select_active_layers
from tg_lora.lora_state import snapshot_lora
from tg_lora.random_walk_controller import RandomWalkController
from tg_lora.rollback_manager import RollbackManager
from tg_lora.trajectory import TrajectoryAnalyzer, TrajectoryPoint
from tg_lora.velocity import Velocity

logger = logging.getLogger("tg-lora.train")


# ---------------------------------------------------------------------------
# SimpleLoRAModel — lightweight model for testing / demonstration
# ---------------------------------------------------------------------------


class SimpleLoRAModel(nn.Module):
    """Minimal model with LoRA-style parameters for testing.

    Each layer has a ``lora_A`` (trainable) and ``lora_B`` (trainable) pair
    whose product forms the weight matrix used in the forward pass.
    """

    def __init__(self, num_layers: int = 4, dim: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Module()
            layer.self_attn.q_proj.lora_A = nn.Parameter(
                torch.randn(dim, dim) * 0.01
            )
            layer.self_attn.q_proj.lora_B = nn.Parameter(
                torch.zeros(dim, dim)
            )
            layer.self_attn.q_proj.lora_A.requires_grad_(True)
            layer.self_attn.q_proj.lora_B.requires_grad_(True)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            w = layer.self_attn.q_proj.lora_A @ layer.self_attn.q_proj.lora_B
            x = x @ w.T
        return x


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def compute_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Simple sum-of-elements loss for the SimpleLoRAModel."""
    return model(x).sum()


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor) -> float:
    """Return scalar loss without gradients."""
    return compute_loss(model, x).item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_loop(
    model: nn.Module,
    x: torch.Tensor,
    config: TGLoraConfig,
    *,
    max_cycles: int | None = None,
    patience: int | None = None,
    early_stop_patience: int = 5,
    resume_checkpoint: dict | None = None,
    progress: bool = True,
) -> dict:
    """Run the TG-LoRA training loop.

    Parameters
    ----------
    model:
        A model with LoRA-style parameters (e.g. SimpleLoRAModel).
    x:
        Training data tensor.
    config:
        TGLoraConfig controlling hyperparameters.
    max_cycles:
        Override for maximum number of cycles (defaults to
        ``config.max_steps``).
    patience:
        CycleState patience for stale-cycle early stopping.
        ``None`` disables the mechanism.
    early_stop_patience:
        TrajectoryAnalyzer patience for its early-stop advice.
    resume_checkpoint:
        Optional checkpoint dict to restore state from.
    progress:
        Whether to show tqdm progress bar.

    Returns
    -------
    dict with training summary, all component states, and trajectory report.
    """
    if max_cycles is None:
        max_cycles = config.max_steps

    # --- Initialise components ---
    controller = RandomWalkController(
        K_initial=config.K_initial,
        N_initial=config.N_initial,
        alpha_initial=config.alpha_initial,
        beta_initial=config.beta_initial,
        lr_initial=config.lr,
        active_layer_strategy=config.active_layer_strategy,
        relative_update_cap=config.relative_update_cap,
        rollback_tolerance=config.rollback_tolerance,
        enable_random_walk=False,
    )
    velocity = Velocity()
    rollback = RollbackManager()
    delta_tracker = DeltaTracker()
    cycle_state = CycleState()
    trajectory = TrajectoryAnalyzer()

    # --- Optionally restore from checkpoint ---
    if resume_checkpoint is not None:
        _restore_checkpoint(
            model, controller, velocity, delta_tracker,
            cycle_state, trajectory, resume_checkpoint,
        )

    pbar = tqdm(range(max_cycles), desc="TG-LoRA", disable=not progress)

    for cycle_idx in pbar:
        proposal = controller.propose()

        # Create a fresh optimizer each cycle (the controller adapts lr)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=proposal.lr,
        )

        # 1. Snapshot before pilot steps
        W0 = snapshot_lora(model)

        # 2. K real optimizer steps
        for _ in range(proposal.K):
            loss = compute_loss(model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Snapshot after pilot steps, compute delta
        WK = snapshot_lora(model)
        dW = delta_tracker.compute_and_record(WK, W0, K=proposal.K)

        # 4. Update velocity (EMA of deltas)
        velocity.update(dW, beta=proposal.beta)

        # 5. Eval → loss_pilot
        loss_pilot = evaluate(model, x)

        # 6. Save rollback point
        rollback.save(model)

        # 7. Select active layers
        active_names, active_indices = select_active_layers(
            model, strategy=proposal.active_layer_strategy,
        )

        # 8. Apply extrapolation
        apply_extrapolation(
            model=model,
            velocity=velocity.state,
            active_names=active_names,
            alpha_by_name={},
            default_alpha=proposal.alpha,
            n_steps=proposal.N,
            relative_update_cap=proposal.relative_update_cap,
        )

        # 9. Eval → loss_after
        loss_after = evaluate(model, x)

        # 10. Accept or rollback
        accepted = controller.accept(loss_pilot, loss_after)
        if accepted:
            controller.reward(loss_pilot, loss_after)
            controller.commit_proposal(proposal)
            if rollback._history:
                rollback.pop()
        else:
            rollback.rollback(model)
            controller.penalize(loss_pilot, loss_after)
            if rollback._history:
                rollback.pop()

        # 11. Record cycle state
        cycle_state.record_cycle(
            train_loss=loss_pilot,
            valid_loss=loss_after if accepted else loss_pilot,
            accepted=accepted,
            K=proposal.K,
            N=proposal.N if accepted else 0,
            grad_accum=1,
        )

        # 12. Trajectory analysis
        trajectory.add_point(TrajectoryPoint(
            cycle=cycle_state.cycle,
            train_loss=loss_pilot,
            valid_loss=loss_after if accepted else loss_pilot,
            velocity_magnitude=velocity.magnitudes[-1] if velocity.magnitudes else None,
        ))

        # --- Convergence adaptation (controller) ---
        controller.adapt_to_convergence(delta_tracker.convergence_trend())
        controller.adapt_to_acceleration(velocity.magnitude_acceleration())

        # --- Early stopping checks ---
        if cycle_state.should_stop(patience=patience):
            logger.info(
                "Early stopping: stale_cycles=%d >= patience=%d at cycle %d",
                cycle_state.stale_cycles, patience, cycle_state.cycle,
            )
            pbar.close()
            break

        trajectory_report = trajectory.early_stop_advice(patience=early_stop_patience)
        if trajectory_report.should_stop and cycle_state.cycle >= 10:
            logger.info(
                "Trajectory early stop: %s at cycle %d",
                trajectory_report.reason, cycle_state.cycle,
            )
            pbar.close()
            break

        # --- Progress display ---
        pbar.set_postfix({
            "loss": f"{loss_pilot:.4f}",
            "accept": cycle_state.acceptance_rate,
            "K": proposal.K,
            "N": proposal.N if accepted else 0,
        })
    else:
        pbar.close()

    # --- Final report ---
    result = _build_summary(config, cycle_state, controller, delta_tracker,
                            velocity, trajectory)
    return result


# ---------------------------------------------------------------------------
# Checkpoint save / restore
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: TGLoraConfig,
    controller: RandomWalkController,
    cycle_state: CycleState,
    delta_tracker: DeltaTracker,
    velocity: Velocity,
) -> None:
    """Save a training checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": snapshot_lora(model),
        "config": config.to_dict(),
        "controller_state": controller.state.summary(),
        "cycle_state": cycle_state.summary(),
        "delta_tracker": delta_tracker.summary(),
        "velocity_magnitudes": velocity.magnitudes,
    }
    torch.save(ckpt, path)
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _restore_checkpoint(
    model: nn.Module,
    controller: RandomWalkController,
    velocity: Velocity,
    delta_tracker: DeltaTracker,
    cycle_state: CycleState,
    trajectory: TrajectoryAnalyzer,
    ckpt: dict,
) -> None:
    """Restore component states from a checkpoint dict."""
    from tg_lora.lora_state import load_lora_snapshot
    from tg_lora.random_walk_controller import ControllerState

    if "model_state" in ckpt:
        load_lora_snapshot(model, ckpt["model_state"])

    if "controller_state" in ckpt:
        controller.restore_state(ControllerState.from_dict(ckpt["controller_state"]))

    if "cycle_state" in ckpt:
        cs = CycleState.from_dict(ckpt["cycle_state"])
        cycle_state.cycle = cs.cycle
        cycle_state.optimizer_steps = cs.optimizer_steps
        cycle_state.full_backward_passes = cs.full_backward_passes
        cycle_state.extrapolation_steps = cs.extrapolation_steps
        cycle_state.speculative_equivalent_backward_passes = (
            cs.speculative_equivalent_backward_passes
        )
        cycle_state.best_loss = cs.best_loss
        cycle_state.best_step = cs.best_step
        cycle_state.stale_cycles = cs.stale_cycles
        cycle_state.last_train_loss = cs.last_train_loss
        cycle_state.last_valid_loss = cs.last_valid_loss
        cycle_state.accepted_count = cs.accepted_count
        cycle_state.rejected_count = cs.rejected_count


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _build_summary(
    config: TGLoraConfig,
    cycle_state: CycleState,
    controller: RandomWalkController,
    delta_tracker: DeltaTracker,
    velocity: Velocity,
    trajectory: TrajectoryAnalyzer,
) -> dict:
    cs = cycle_state.summary()
    cs["controller"] = controller.summary()
    cs["delta_tracker"] = delta_tracker.summary()
    cs["trajectory_report"] = trajectory.full_report()

    # Compute velocity magnitude from current state
    if velocity.magnitudes:
        cs["velocity_magnitude"] = velocity.magnitudes[-1]
    else:
        cs["velocity_magnitude"] = 0.0

    cs["config"] = config.to_dict()
    return cs


def print_summary(result: dict) -> None:
    """Print a human-readable cycle summary to stdout."""
    print("\n" + "=" * 60)
    print("TG-LoRA Training Summary")
    print("=" * 60)
    print(f"  Total cycles:              {result['cycles']}")
    print(f"  Optimizer steps:           {result['optimizer_steps']}")
    print(f"  Full backward passes:      {result['full_backward_passes']}")
    print(f"  Extrapolation steps:       {result['extrapolation_steps']}")
    print(f"  Reduction rate:            {result['reduction_rate']:.2%}")
    print(f"  Acceptance rate:           {result['acceptance_rate']:.2%}")
    print(f"  Accepted / Rejected:       {result['accepted_count']} / {result['rejected_count']}")
    print(f"  Final train loss:          {result['final_train_loss']:.6f}")
    print(f"  Best valid loss:           {result['best_valid_loss']:.6f}")
    print(f"  Stale cycles:              {result['stale_cycles']}")
    if "controller" in result:
        ctrl = result["controller"]
        print(f"  Final alpha:               {ctrl.get('current_alpha', 'N/A')}")
        print(f"  Final lr:                  {ctrl.get('current_lr', 'N/A')}")
        print(f"  Final K:                   {ctrl.get('current_K', 'N/A')}")
        print(f"  Final N:                   {ctrl.get('current_N', 'N/A')}")
        print(f"  Final beta:                {ctrl.get('current_beta', 'N/A')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TG-LoRA reference training script",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML or JSON config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    return parser.parse_args(argv)


def load_config(path: str) -> TGLoraConfig:
    """Load TGLoraConfig from YAML or JSON based on file extension."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return TGLoraConfig.from_yaml(p)
    return TGLoraConfig.from_json(p)


def main(argv: list[str] | None = None) -> dict:
    """Entry point for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args(argv)
    config = load_config(args.config)

    if args.output_dir is not None:
        config.output_dir = args.output_dir

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config
    config.save_json(output_dir / "config_effective.json")

    logger.info("Config:\n%s", config.summary())

    # --- Build model & data ---
    model = SimpleLoRAModel(num_layers=4, dim=4)
    x = torch.randn(config.batch_size, 4)

    # --- Resume from checkpoint if requested ---
    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_checkpoint(args.resume)
        logger.info("Resumed from checkpoint: %s", args.resume)

    # --- Run training ---
    result = train_loop(
        model=model,
        x=x,
        config=config,
        resume_checkpoint=resume_ckpt,
    )

    # --- Save outputs ---
    print_summary(result)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        # Remove non-serializable trajectory_report for JSON
        json_safe = {k: v for k, v in result.items() if k != "trajectory_report"}
        json.dump(json_safe, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)

    return result


if __name__ == "__main__":
    main()
