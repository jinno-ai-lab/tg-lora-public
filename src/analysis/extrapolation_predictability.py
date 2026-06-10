from __future__ import annotations

from collections.abc import Iterable
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.training.trajectory_delta_artifact import load_trajectory_delta_artifact


@dataclass(frozen=True)
class UpdateStep:
    step: int
    tensors: dict[str, torch.Tensor]


def beta_from_window(window: int) -> float:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if window == 1:
        return 0.0
    return 1.0 - (1.0 / window)


def load_update_steps_from_artifacts(
    artifact_dir: str | Path,
    *,
    anchor_kind: str = "after_optimizer_step",
    delta_mode: str = "cumulative",
) -> list[UpdateStep]:
    """Load saved trajectory artifacts and convert them to update steps.

    Baseline trajectory artifacts are stored as cumulative deltas from the
    initial LoRA snapshot.  ``delta_mode="cumulative"`` converts consecutive
    cumulative deltas into per-artifact updates.  ``delta_mode="direct"`` uses
    each artifact delta as already being one update.
    """
    if delta_mode not in {"cumulative", "direct"}:
        raise ValueError("delta_mode must be 'cumulative' or 'direct'")

    root = Path(artifact_dir)
    if not root.exists():
        raise FileNotFoundError(f"artifact directory not found: {root}")

    loaded = []
    for path in sorted(root.glob("*.pt")):
        artifact = load_trajectory_delta_artifact(path)
        if artifact.metadata.anchor_kind != anchor_kind:
            continue
        order = (
            artifact.metadata.total_backward_passes
            if artifact.metadata.total_backward_passes is not None
            else artifact.metadata.step
            if artifact.metadata.step is not None
            else artifact.metadata.cycle
            if artifact.metadata.cycle is not None
            else len(loaded)
        )
        loaded.append((int(order), path.name, artifact.delta_tensors))

    loaded.sort(key=lambda item: (item[0], item[1]))

    updates: list[UpdateStep] = []
    previous: dict[str, torch.Tensor] | None = None
    for order, _name, delta in loaded:
        current = _clone_tensors(delta)
        if delta_mode == "direct" or previous is None:
            update = current
        else:
            update = _subtract_tensors(current, previous)
        updates.append(UpdateStep(step=order, tensors=update))
        previous = current

    return updates


def analyze_update_predictability(
    updates: list[UpdateStep],
    *,
    n_values: list[int],
    short_window: int = 3,
    long_window: int = 10,
    consistency_thresholds: list[float] | None = None,
    yes_threshold: float = 0.5,
    include_controls: bool = True,
    control_seed: int = 0,
) -> dict[str, Any]:
    if not updates:
        raise ValueError("updates must not be empty")
    if any(n <= 0 for n in n_values):
        raise ValueError("all n_values must be positive")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    thresholds = consistency_thresholds or [0.5, 0.7, 0.9]
    beta_short = beta_from_window(short_window)
    beta_long = beta_from_window(long_window)
    split_index = len(updates) // 2
    control_generator = torch.Generator(device="cpu")
    control_generator.manual_seed(control_seed)
    shuffled_past_short: list[dict[str, torch.Tensor]] | None = None
    shuffled_past_long: list[dict[str, torch.Tensor]] | None = None
    if include_controls:
        shuffled_past_short = _shuffled_past_ema_sequence(
            updates, beta_short, control_generator
        )
        shuffled_past_long = _shuffled_past_ema_sequence(
            updates, beta_long, control_generator
        )

    ema_short: dict[str, torch.Tensor] | None = None
    ema_long: dict[str, torch.Tensor] | None = None
    points_by_n: dict[int, list[dict[str, float]]] = {n: [] for n in n_values}

    for index, update in enumerate(updates):
        ema_short = _ema_update(ema_short, update.tensors, beta_short)
        ema_long = _ema_update(ema_long, update.tensors, beta_long)

        for n_steps in n_values:
            end = index + 1 + n_steps
            if end > len(updates):
                continue
            future = _sum_tensors(step.tensors for step in updates[index + 1 : end])
            future_cos_short = cosine_dicts(future, ema_short)
            future_cos_long = cosine_dicts(future, ema_long)
            consistency_cos = cosine_dicts(ema_short, ema_long)
            long_norm = norm_dict(ema_long)
            norm_ratio = norm_dict(ema_short) / long_norm if long_norm > 1e-12 else 1.0
            point = {
                "anchor_index": float(index),
                "anchor_step": float(update.step),
                "future_cos_short": future_cos_short,
                "future_cos_long": future_cos_long,
                "consistency_cos": consistency_cos,
                "short_long_norm_ratio": norm_ratio,
            }
            if include_controls:
                assert shuffled_past_short is not None
                assert shuffled_past_long is not None
                random_direction = _random_like_with_norm(future, control_generator)
                point["random_cos"] = cosine_dicts(future, random_direction)
                point["shuffle_cos_short"] = cosine_dicts(
                    future,
                    shuffled_past_short[index],
                )
                point["shuffle_cos_long"] = cosine_dicts(
                    future,
                    shuffled_past_long[index],
                )
            points_by_n[n_steps].append(point)

    per_n = {
        str(n_steps): _summarize_points(
            points,
            thresholds=thresholds,
            yes_threshold=yes_threshold,
            split_index=split_index,
        )
        for n_steps, points in points_by_n.items()
    }

    return {
        "update_count": len(updates),
        "short_window": short_window,
        "long_window": long_window,
        "beta_short": beta_short,
        "beta_long": beta_long,
        "yes_threshold": yes_threshold,
        "include_controls": include_controls,
        "control_seed": control_seed,
        "split_index": split_index,
        "n_values": n_values,
        "per_n": per_n,
    }


def cosine_dicts(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for key in left.keys() & right.keys():
        a = left[key].float().flatten()
        b = right[key].float().flatten()
        dot_val = torch.dot(a, b).item()
        left_val = torch.dot(a, a).item()
        right_val = torch.dot(b, b).item()
        if not (
            math.isfinite(dot_val)
            and math.isfinite(left_val)
            and math.isfinite(right_val)
        ):
            continue
        dot += dot_val
        left_sq += left_val
        right_sq += right_val

    denom = math.sqrt(left_sq) * math.sqrt(right_sq)
    if denom <= 1e-12:
        return 0.0
    return dot / denom


def norm_dict(tensors: dict[str, torch.Tensor]) -> float:
    total_sq = 0.0
    for tensor in tensors.values():
        value = tensor.float().norm().item()
        if math.isfinite(value):
            total_sq += value**2
    return math.sqrt(total_sq)


def _summarize_points(
    points: list[dict[str, float]],
    *,
    thresholds: list[float],
    yes_threshold: float,
    split_index: int,
) -> dict[str, Any]:
    if not points:
        return {
            "sample_count": 0,
            "predictable": False,
            "mean_future_cos_long": None,
            "mean_future_cos_short": None,
            "mean_consistency_cos": None,
            "mean_short_long_norm_ratio": None,
            "mean_random_cos": None,
            "mean_shuffle_cos_long": None,
            "mean_shuffle_cos_short": None,
            "gated_by_consistency": {},
            "split_by_anchor": {
                "first_half": _empty_point_summary(),
                "second_half": _empty_point_summary(),
            },
        }

    summary = _point_summary(points, yes_threshold=yes_threshold)
    summary["gated_by_consistency"] = _summarize_gates(points, thresholds)
    summary["split_by_anchor"] = {
        "first_half": _point_summary(
            [point for point in points if point["anchor_index"] < split_index],
            yes_threshold=yes_threshold,
        ),
        "second_half": _point_summary(
            [point for point in points if point["anchor_index"] >= split_index],
            yes_threshold=yes_threshold,
        ),
    }
    return summary


def _summarize_gates(
    points: list[dict[str, float]],
    thresholds: list[float],
) -> dict[str, Any]:
    gated = {}
    for threshold in thresholds:
        selected = [point for point in points if point["consistency_cos"] >= threshold]
        gated[str(threshold)] = {
            "sample_count": len(selected),
            "coverage": len(selected) / len(points),
            "mean_future_cos_long": (
                _mean(point["future_cos_long"] for point in selected)
                if selected
                else None
            ),
            "mean_future_cos_short": (
                _mean(point["future_cos_short"] for point in selected)
                if selected
                else None
            ),
        }
    return gated


def _point_summary(
    points: list[dict[str, float]],
    *,
    yes_threshold: float,
) -> dict[str, Any]:
    if not points:
        return _empty_point_summary()
    mean_future_cos_long = _mean(point["future_cos_long"] for point in points)
    return {
        "sample_count": len(points),
        "predictable": mean_future_cos_long >= yes_threshold,
        "mean_future_cos_long": mean_future_cos_long,
        "mean_future_cos_short": _mean(point["future_cos_short"] for point in points),
        "mean_consistency_cos": _mean(point["consistency_cos"] for point in points),
        "mean_short_long_norm_ratio": _mean(
            point["short_long_norm_ratio"] for point in points
        ),
        "mean_random_cos": _mean_optional(points, "random_cos"),
        "mean_shuffle_cos_long": _mean_optional(points, "shuffle_cos_long"),
        "mean_shuffle_cos_short": _mean_optional(points, "shuffle_cos_short"),
    }


def _empty_point_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "predictable": False,
        "mean_future_cos_long": None,
        "mean_future_cos_short": None,
        "mean_consistency_cos": None,
        "mean_short_long_norm_ratio": None,
        "mean_random_cos": None,
        "mean_shuffle_cos_long": None,
        "mean_shuffle_cos_short": None,
    }


def _ema_update(
    state: dict[str, torch.Tensor] | None,
    update: dict[str, torch.Tensor],
    beta: float,
) -> dict[str, torch.Tensor]:
    if state is None:
        return _clone_tensors(update)
    next_state = _clone_tensors(state)
    for key, tensor in update.items():
        if key not in next_state:
            next_state[key] = tensor.clone()
            continue
        next_state[key].mul_(beta).add_(tensor, alpha=1.0 - beta)
    return next_state


def _shuffled_past_ema_sequence(
    updates: list[UpdateStep],
    beta: float,
    generator: torch.Generator,
) -> list[dict[str, torch.Tensor]]:
    sequence = []
    for index in range(len(updates)):
        shuffled_prefix = _shuffle_updates(updates[: index + 1], generator)
        state: dict[str, torch.Tensor] | None = None
        for update in shuffled_prefix:
            state = _ema_update_in_place(state, update.tensors, beta)
        sequence.append(state if state is not None else {})
    return sequence


def _ema_update_in_place(
    state: dict[str, torch.Tensor] | None,
    update: dict[str, torch.Tensor],
    beta: float,
) -> dict[str, torch.Tensor]:
    if state is None:
        return _clone_tensors(update)
    for key, tensor in update.items():
        if key not in state:
            state[key] = tensor.detach().cpu().clone()
            continue
        state[key].mul_(beta).add_(tensor, alpha=1.0 - beta)
    return state


def _sum_tensors(
    tensor_dicts: Iterable[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    total: dict[str, torch.Tensor] | None = None
    for tensors in tensor_dicts:
        if total is None:
            total = _clone_tensors(tensors)
            continue
        for key, tensor in tensors.items():
            if key not in total:
                total[key] = tensor.clone()
            else:
                total[key].add_(tensor)
    return total or {}


def _subtract_tensors(
    current: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    keys = current.keys() | previous.keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        if key in current and key in previous:
            out[key] = current[key] - previous[key]
        elif key in current:
            out[key] = current[key].clone()
        else:
            out[key] = -previous[key]
    return out


def _clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().clone() for key, tensor in tensors.items()}


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def _mean_optional(points: list[dict[str, float]], key: str) -> float | None:
    values = [point[key] for point in points if key in point]
    if not values:
        return None
    return _mean(values)


def _shuffle_updates(
    updates: list[UpdateStep],
    generator: torch.Generator,
) -> list[UpdateStep]:
    order = torch.randperm(len(updates), generator=generator).tolist()
    return [updates[index] for index in order]


def _random_like_with_norm(
    tensors: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    target_norm = norm_dict(tensors)
    random_tensors = {
        key: torch.randn(
            tensor.shape,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        for key, tensor in tensors.items()
    }
    random_norm = norm_dict(random_tensors)
    if random_norm <= 1e-12:
        return random_tensors
    scale = target_norm / random_norm
    return {key: tensor.mul(scale) for key, tensor in random_tensors.items()}
