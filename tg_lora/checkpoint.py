from __future__ import annotations

import platform
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

_CHECKPOINT_VERSION = "0.1.0"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: object,
    *,
    controller: object | None = None,
    cycle_state: object | None = None,
    delta_tracker: object | None = None,
    velocity: object | None = None,
    trajectory: object | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_sd = OrderedDict(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )
    optimizer_sd = optimizer.state_dict()

    def _to_cpu_tensors(d: dict) -> dict:
        out: dict = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu()
            elif isinstance(v, dict):
                out[k] = _to_cpu_tensors(v)
            elif isinstance(v, (list, tuple)):
                out[k] = [
                    _to_cpu_tensors(i) if isinstance(i, dict)
                    else (i.detach().cpu() if isinstance(i, torch.Tensor) else i)
                    for i in v
                ]
            else:
                out[k] = v
        return out

    optimizer_sd = _to_cpu_tensors(optimizer_sd)

    ckpt: dict = {
        "version": _CHECKPOINT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model_sd,
        "optimizer_state_dict": optimizer_sd,
        "config": config.to_dict(),
        "controller_summary": controller.summary() if controller is not None else None,
        "cycle_state_summary": cycle_state.summary() if cycle_state is not None else None,
        "delta_tracker_state": _serialize_delta_tracker(delta_tracker) if delta_tracker is not None else None,
        "velocity_state": _serialize_velocity(velocity) if velocity is not None else None,
        "trajectory_points": _serialize_trajectory(trajectory) if trajectory is not None else None,
        "extra": extra,
        "metadata": {
            "tg_lora_version": _CHECKPOINT_VERSION,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": sys.platform,
        },
    }

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".ckpt_tmp_",
        suffix=path.suffix,
    )
    try:
        with open(fd, "wb") as f:
            torch.save(ckpt, f)
        Path(tmp_path).rename(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(ckpt).__name__}")
    _required = {"version", "model_state_dict", "optimizer_state_dict", "config"}
    missing = _required - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {sorted(missing)}")
    return ckpt


def _serialize_delta_tracker(tracker: object) -> dict:
    norms = tracker.norm_history if hasattr(tracker, "norm_history") else []
    last_stats = tracker.last_stats if hasattr(tracker, "last_stats") else None
    stats_dict = {}
    if last_stats is not None:
        stats_dict = {
            "total_norm": last_stats.total_norm,
            "max_component": last_stats.max_component,
            "mean_abs": last_stats.mean_abs,
            "per_layer_norm": dict(last_stats.per_layer_norm),
        }
    return {"norms": list(norms), "last_stats": stats_dict}


def _serialize_velocity(velocity: object) -> dict:
    state = velocity.state if hasattr(velocity, "state") else None
    if state is None:
        return {"state": {}, "magnitudes": []}
    cpu_state = {k: v.detach().cpu() for k, v in state.items()}
    magnitudes = velocity.magnitudes if hasattr(velocity, "magnitudes") else []
    return {"state": cpu_state, "magnitudes": list(magnitudes)}


def _serialize_trajectory(trajectory: object) -> list:
    points = trajectory.points if hasattr(trajectory, "points") else []
    result = []
    for p in points:
        d = {"cycle": p.cycle, "train_loss": p.train_loss}
        if p.valid_loss is not None:
            d["valid_loss"] = p.valid_loss
        if p.grad_norm is not None:
            d["grad_norm"] = p.grad_norm
        if p.velocity_magnitude is not None:
            d["velocity_magnitude"] = p.velocity_magnitude
        result.append(d)
    return result
