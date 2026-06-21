import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.dynamic_freeze import DynFreezeState
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


_CYCLE_DIR_RE = re.compile(r"^checkpoint-cycle-(\d+)$")


def _sorted_cycle_checkpoint_dirs(run_dir: Path) -> list[Path]:
    """Return ``checkpoint-cycle-<N>`` dirs under *run_dir*, oldest cycle first."""
    found: list[tuple[int, Path]] = []
    for p in Path(run_dir).iterdir():
        if p.is_dir():
            m = _CYCLE_DIR_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def prune_checkpoint_cycles(
    run_dir: Path,
    keep_last: int = 0,
    min_free_disk_gb: float = 0.0,
) -> list[Path]:
    """Bound on-disk checkpoint growth by removing old ``checkpoint-cycle-*`` dirs.

    Two independent guards; either may be ``0`` to disable (the safe default,
    preserving today's unbounded behavior for arbitrary configs):

    - ``keep_last``: when > 0, retain only the newest ``keep_last`` checkpoint
      dirs and delete the rest. This is the primary bound against the M10.3
      disk-death class — a run saving every cycle otherwise accumulates one dir
      per cycle forever.
    - ``min_free_disk_gb``: when > 0, if free space on *run_dir*'s filesystem is
      below this floor, delete the oldest checkpoint dirs first until the floor
      is met. This is emergency reclamation: it MAY go below ``keep_last``, but
      never deletes the single newest checkpoint, so one recovery point always
      survives.

    Cycle order is read from the integer suffix of ``checkpoint-cycle-<N>``.
    Returns the dirs removed, oldest-first. Idempotent; safe with no dirs.
    """
    run_dir = Path(run_dir)
    dirs = _sorted_cycle_checkpoint_dirs(run_dir)  # oldest -> newest
    if not dirs:
        return []

    keep_last = max(0, int(keep_last))
    min_free_bytes = max(0.0, float(min_free_disk_gb)) * (1024 ** 3)
    removed: list[Path] = []

    # 1) Count bound: delete everything older than the newest `keep_last`.
    if keep_last > 0:
        cut = len(dirs) - keep_last
        if cut > 0:
            for d in dirs[:cut]:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d)
            dirs = dirs[cut:]

    # 2) Disk floor: prune oldest survivors until the floor is met, but never
    #    below the single newest checkpoint.
    if min_free_bytes > 0:
        while len(dirs) > 1 and shutil.disk_usage(str(run_dir)).free < min_free_bytes:
            oldest = dirs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
            removed.append(oldest)

    return removed


def prune_checkpoint_cycles_from_cfg(cfg, run_dir: Path) -> list[Path]:
    """Read the pruning knobs from ``cfg.logging`` and prune ``checkpoint-cycle-*``.

    The config-driven entry point wired into the periodic-save path of
    ``train_tg_lora``. Extracted from an inline read+call so the config->prune
    coupling is the unit-testable seam: an untested inline block was a silent
    "protection exists but never fires" risk (a renamed key would leave the
    guard inert while every isolated ``prune_checkpoint_cycles`` test still
    passed). Returns the pruned dirs, oldest-first; empty when both knobs are
    off (the safe default).

    ``cfg`` may be an OmegaConf ``DictConfig`` (prod) or a plain dict (tests);
    both expose ``.get``.
    """
    logging_cfg = cfg.get("logging", {}) if cfg is not None else {}
    keep_last = int(logging_cfg.get("keep_last_checkpoints", 0))
    min_free = float(logging_cfg.get("min_free_disk_gb", 0.0))
    if keep_last <= 0 and min_free <= 0.0:
        return []
    return prune_checkpoint_cycles(
        run_dir, keep_last=keep_last, min_free_disk_gb=min_free
    )


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
    dynfreeze_state: DynFreezeState | None = None


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
        "dynfreeze_state": (
            {
                "frozen_layer_indices": state.dynfreeze_state.frozen_layer_indices,
                "r_A_history": state.dynfreeze_state.r_A_history,
                "frozen_since_cycle": state.dynfreeze_state.frozen_since_cycle,
                "prev_A_fro": state.dynfreeze_state.prev_A_fro,
                "median_A": state.dynfreeze_state.median_A,
                "epsilon": state.dynfreeze_state.epsilon,
            }
            if state.dynfreeze_state is not None
            else None
        ),
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

    # Restore DynFreezeState
    dynfreeze_raw = blob.get("dynfreeze_state")
    dynfreeze_state = None
    if dynfreeze_raw is not None:
        dynfreeze_state = DynFreezeState(
            frozen_layer_indices=dynfreeze_raw.get("frozen_layer_indices", []),
            r_A_history=dynfreeze_raw.get("r_A_history", {}),
            frozen_since_cycle=dynfreeze_raw.get("frozen_since_cycle", 0),
            prev_A_fro=dynfreeze_raw.get("prev_A_fro", {}),
            median_A=dynfreeze_raw.get("median_A", 0.0),
            epsilon=dynfreeze_raw.get("epsilon", 1e-6),
        )

    return TrainingState(
        cycle_state=cycle_state,
        controller_state=controller_state,
        velocity=velocity,
        delta_tracker=delta_tracker,
        cycle_offset=blob.get("cycle_offset", 0),
        adapter_checkpoint_dir=blob.get("adapter_checkpoint_dir"),
        train_batch_position=blob.get("train_batch_position", 0),
        accepted_valid_history=blob.get("accepted_valid_history"),
        dynfreeze_state=dynfreeze_state,
    )
