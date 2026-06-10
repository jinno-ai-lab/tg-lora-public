import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.random_walk_controller import ControllerState
from src.tg_lora.velocity import Velocity
from src.utils.tensor_artifact import load_tensor_artifact

logger = logging.getLogger(__name__)


def _sanitize_tensors(tensor_dict: dict[str, torch.Tensor], label: str) -> None:
    """Replace NaN/Inf in loaded tensors with zeros and log a warning."""
    bad_keys: list[str] = []
    for k, v in tensor_dict.items():
        if not torch.isfinite(v).all():
            bad_keys.append(k)
            tensor_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if bad_keys:
        logger.warning(
            "Sanitized non-finite values in %s for keys: %s", label, bad_keys
        )


def save_checkpoint(model: torch.nn.Module, tokenizer, save_dir: Path) -> None:
    """Save model and tokenizer to *save_dir* with readback verification.

    Used by both baseline and TG-LoRA trainers for periodic saves and
    best-model persistence.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Readback verification
    if not save_dir.is_dir():
        logger.warning("Checkpoint directory missing after save: %s", save_dir)
    else:
        file_count = sum(1 for _ in save_dir.iterdir())
        if file_count == 0:
            logger.warning("Checkpoint directory is empty after save: %s", save_dir)


@dataclass
class TrainingState:
    """All state needed to resume a TG-LoRA training job from a checkpoint."""

    cycle_state: CycleState
    controller_state: ControllerState
    velocity: Velocity
    delta_tracker: DeltaTracker
    cycle_offset: int = 0
    adapter_checkpoint_dir: str | None = None
    train_batch_position: int = 0
    accepted_valid_history: list[float] | None = None


@dataclass
class BaselineTrainingState:
    """State needed to resume baseline QLoRA training from a checkpoint."""

    global_step: int = 0
    best_loss: float = float("inf")
    best_step: int = 0
    stale_steps: int = 0
    train_batch_position: int = 0
    adapter_checkpoint_dir: str | None = None


def save_baseline_training_state(
    state: BaselineTrainingState,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "global_step": state.global_step,
        "best_loss": state.best_loss,
        "best_step": state.best_step,
        "stale_steps": state.stale_steps,
        "train_batch_position": state.train_batch_position,
        "adapter_checkpoint_dir": state.adapter_checkpoint_dir,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(blob, path)
    logger.info(
        "Saved baseline training state to %s (step %d)", path, state.global_step
    )


def load_baseline_training_state(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline training state not found: {path}")
    blob = load_tensor_artifact(path)
    state = BaselineTrainingState(
        global_step=blob.get("global_step", 0),
        best_loss=blob.get("best_loss", float("inf")),
        best_step=blob.get("best_step", 0),
        stale_steps=blob.get("stale_steps", 0),
        train_batch_position=blob.get("train_batch_position", 0),
        adapter_checkpoint_dir=blob.get("adapter_checkpoint_dir"),
    )
    return {
        "state": state,
        "optimizer_state_dict": blob.get("optimizer_state_dict"),
        "scheduler_state_dict": blob.get("scheduler_state_dict"),
    }


def save_training_state(state: TrainingState, path: Path) -> None:
    """Serialize training state to disk for later recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    velocity_tensors = None
    if state.velocity._state is not None:
        velocity_tensors = {k: v.cpu() for k, v in state.velocity._state.items()}
    velocity_short_tensors = None
    if state.velocity._short_state is not None:
        velocity_short_tensors = {
            k: v.cpu() for k, v in state.velocity._short_state.items()
        }
    velocity_long_tensors = None
    if state.velocity._long_state is not None:
        velocity_long_tensors = {
            k: v.cpu() for k, v in state.velocity._long_state.items()
        }

    delta_tensors = None
    if state.delta_tracker._history:
        delta_tensors = [
            {k: v.cpu() for k, v in h.items()} for h in state.delta_tracker._history
        ]

    blob = {
        "cycle_state": state.cycle_state.summary(),
        "controller_state": state.controller_state.summary(),
        "velocity_tensors": velocity_tensors,
        "velocity_short_tensors": velocity_short_tensors,
        "velocity_long_tensors": velocity_long_tensors,
        "velocity_magnitudes": state.velocity.magnitudes,
        "velocity_max_history": state.velocity._max_history,
        "velocity_beta_short": state.velocity.beta_short,
        "velocity_beta_long": state.velocity.beta_long,
        "velocity_update_count": state.velocity.update_count,
        "delta_tensors": delta_tensors,
        "delta_norm_history": state.delta_tracker.norm_history,
        "delta_max_history": state.delta_tracker._max_history,
        "cycle_offset": state.cycle_offset,
        "adapter_checkpoint_dir": state.adapter_checkpoint_dir,
        "train_batch_position": state.train_batch_position,
        "accepted_valid_history": state.accepted_valid_history,
    }
    torch.save(blob, path)
    logger.info("Saved training state to %s (cycle %d)", path, state.cycle_state.cycle)


def load_training_state(path: Path) -> TrainingState:
    """Load training state from a previously saved checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training state not found: {path}")

    blob = load_tensor_artifact(path)

    # Restore CycleState — from_dict accepts both summary() and legacy checkpoint keys
    cycle_state = CycleState.from_dict(blob["cycle_state"])

    # Restore ControllerState
    controller_state = ControllerState.from_dict(blob["controller_state"])

    # Restore Velocity
    velocity = Velocity(
        max_history=blob.get("velocity_max_history", 100),
        beta_short=blob.get("velocity_beta_short"),
        beta_long=blob.get("velocity_beta_long"),
    )
    velocity_tensors = blob.get("velocity_tensors")
    if velocity_tensors is not None:
        _sanitize_tensors(velocity_tensors, "velocity")
        velocity._state = velocity_tensors
    velocity_short_tensors = blob.get("velocity_short_tensors")
    if velocity_short_tensors is not None:
        _sanitize_tensors(velocity_short_tensors, "velocity_short")
        velocity._short_state = velocity_short_tensors
    velocity_long_tensors = blob.get("velocity_long_tensors")
    if velocity_long_tensors is not None:
        _sanitize_tensors(velocity_long_tensors, "velocity_long")
        velocity._long_state = velocity_long_tensors
    velocity._magnitude_history = blob.get("velocity_magnitudes", [])
    velocity._update_count = blob.get(
        "velocity_update_count",
        len(velocity._magnitude_history),
    )

    # Restore DeltaTracker
    delta_tracker = DeltaTracker(max_history=blob.get("delta_max_history", 100))
    delta_tensors = blob.get("delta_tensors")
    if delta_tensors:
        for i, h in enumerate(delta_tensors):
            _sanitize_tensors(h, f"delta_history[{i}]")
        delta_tracker._history = delta_tensors
    delta_tracker._norm_history = blob.get("delta_norm_history", [])
    # Recompute last_stats from latest history entry
    if delta_tracker._history:
        from src.tg_lora.delta_tracker import _compute_stats

        delta_tracker._last_stats = _compute_stats(delta_tracker._history[-1])

    return TrainingState(
        cycle_state=cycle_state,
        controller_state=controller_state,
        velocity=velocity,
        delta_tracker=delta_tracker,
        cycle_offset=blob.get("cycle_offset", 0),
        adapter_checkpoint_dir=blob.get("adapter_checkpoint_dir"),
        train_batch_position=blob.get("train_batch_position", 0),
        accepted_valid_history=blob.get("accepted_valid_history"),
    )
