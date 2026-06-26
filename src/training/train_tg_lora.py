import json
import logging
import math
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import torch
import numpy as np
from omegaconf import DictConfig
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.build_seed_dataset import load_dataset
from src.eval.eval_loss import eval_loss, eval_loss_detailed
from src.eval.jsonex_generation import evaluate_json_extraction_run
from src.model.load_model import (
    apply_lora,
    get_input_device,
    load_base_model,
    load_tokenizer,
)
from src.model.lora_utils import configure_trainable_lora_scope, iter_lora_params
from src.tg_lora.activation_cache import (
    ActivationCache,
    determine_split_layer,
)
from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.activation_regime import ActivationFingerprintTracker
from src.tg_lora.dynamic_freeze import DynamicFreezeController
from src.tg_lora.weight_averaging import LAWAAverager
from src.tg_lora.psa import PSAPrior
from src.tg_lora.extrapolator import (
    ExtrapolationStats,
    ZerothOrderStepStats,
    apply_extrapolation,
    alpha_line_loss_cached_first_order,
    alpha_line_loss_exact,
    compute_alpha_line_base_out_jvp,
    subspace_zeroth_order_step,
    subspace_m9_fit_step,
)
from src.tg_lora.layer_sampler import get_num_layers, select_active_layers
from src.tg_lora.lora_state import (
    apply_delta_snapshot,
    load_lora_snapshot,
    snapshot_lora,
    snapshot_lora_delta,
)
from src.tg_lora.prefix_feature_cache import (
    PrefixFeatureDatasetBase,
    build_prefix_feature_cache_metadata,
    build_prefix_feature_dataset,
    collate_prefix_feature_batch,
    get_prefix_feature_cache_path,
    load_prefix_feature_dataset,
    resolve_prefix_feature_cache_seed,
    save_prefix_feature_dataset,
)
from src.tg_lora.prefix_runtime_offload import offload_prefix_runtime_to_cpu
from src.tg_lora.progressive_freeze import ProgressiveFreezeController
from src.tg_lora.random_walk_controller import RandomWalkController
from src.tg_lora.rollback_manager import RollbackManager
from src.tg_lora.velocity import Velocity, beta_from_window
from src.training.async_cache_builder import AsyncCacheBuilder
from src.training.batch_iter import InfiniteBatchIterator
from src.training.deterministic_batch_plan import (
    DeterministicBatchSampler,
    build_deterministic_batch_plan_for_dataset,
    build_trajectory_key,
)
from src.training.loss import compute_loss, has_supervised_tokens
from src.training.optimizer_lifecycle import OptimizerLifecycleManager
from src.training.trainer_loop import (
    NumericalInstabilityError,
    forward_backward,
    optimizer_step,
)
from src.training.trajectory_delta_artifact import (
    artifact_file_name,
    build_trajectory_delta_artifact_metadata,
    save_trajectory_delta_artifact,
)
from src.utils.checkpoint import (
    TrainingState,
    load_training_state,
    prune_trajectory_delta_artifacts_from_cfg,
    save_checkpoint,
    save_periodic_cycle_checkpoint,
    save_training_state,
)
from src.utils.logging import ensure_dir
from src.utils.memory import count_parameters
from src.utils.mlflow_logger import MLflowLogger
from src.utils.run_metrics import RunMetrics
from src.utils.seed import set_seed

logger = logging.getLogger("tg-lora")


def _gpu_allocated_mb(device: torch.device | str | None) -> float | None:
    from src.utils.device import gpu_memory_allocated_mb

    return gpu_memory_allocated_mb(device)


def _cached_loader_kwargs(cfg: DictConfig) -> dict[str, object]:
    num_workers = int(cfg.training.get("prefix_feature_cache_num_workers", 0))
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.training.get("prefix_feature_cache_pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            cfg.training.get("prefix_feature_cache_persistent_workers", False)
        )
        prefetch_factor = cfg.training.get("prefix_feature_cache_prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _build_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    cached_loader_kwargs: dict[str, object],
    epoch_batches: list[list[int]] | None = None,
) -> DataLoader:
    kwargs: dict[str, object] = {}
    if epoch_batches is None:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle
    else:
        kwargs["batch_sampler"] = DeterministicBatchSampler(epoch_batches)
    if isinstance(dataset, PrefixFeatureDatasetBase):
        kwargs["collate_fn"] = collate_prefix_feature_batch
        kwargs.update(cached_loader_kwargs)
    else:
        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **cast(dict[str, Any], kwargs))


def should_run_full_eval(cycle: int, full_eval_every: int) -> bool:
    return cycle > 0 and full_eval_every > 0 and cycle % full_eval_every == 0


def check_lora_params_finite(model: torch.nn.Module) -> tuple[bool, str]:
    """Check all LoRA parameters for NaN/Inf after extrapolation.

    Returns (is_finite, detail) where is_finite is True when all params
    are finite and detail describes the first non-finite param found.
    """
    for name, param in iter_lora_params(model):
        if not torch.isfinite(param).all():
            detail = f"{name}: "
            if torch.isnan(param).any():
                detail += "NaN"
            elif torch.isinf(param).any():
                detail += "Inf"
            return False, detail
    return True, ""


def _compute_pilot_average(step_losses: list[float], K: int) -> tuple[float, dict]:
    """Compute pilot cycle average loss and metrics from per-step losses."""
    if not step_losses:
        return float("nan"), {"K": K, "avg_loss": float("nan"), "count": 0}
    finite_losses = [loss for loss in step_losses if math.isfinite(loss)]
    if not finite_losses:
        return float("nan"), {
            "K": K,
            "avg_loss": float("nan"),
            "count": 0,
            "finite_count": 0,
            "total_count": len(step_losses),
        }
    avg = sum(finite_losses) / len(finite_losses)
    metrics = {
        "K": K,
        "avg_loss": avg,
        "count": len(step_losses),
        "finite_count": len(finite_losses),
        "min_loss": min(finite_losses),
        "max_loss": max(finite_losses),
    }
    return avg, metrics


@torch.no_grad()
def _forward_loss_no_grad(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> float:
    was_training = model.training
    model.eval()
    try:
        loss = compute_loss(model, batch)
        return float(loss.detach().item())
    finally:
        if was_training:
            model.train()


def _apply_alpha_direction_from_base(
    model: torch.nn.Module,
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float,
    active_names: set[str] | None = None,
) -> None:
    """Set active LoRA params to ``base + alpha * direction`` in-place."""
    with torch.no_grad():
        for name, param in iter_lora_params(model):
            if active_names is not None and name not in active_names:
                continue
            if name not in base or name not in direction:
                continue
            base_t = base[name].to(device=param.device, dtype=param.dtype)
            dir_t = direction[name].to(device=param.device, dtype=param.dtype)
            param.copy_(base_t + float(alpha) * dir_t)


def _alpha_line_parameter_updates(
    model: torch.nn.Module,
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: torch.Tensor,
    *,
    active_names: set[str] | None = None,
    name_prefix: str = "",
) -> dict[str, torch.Tensor]:
    parameter_updates: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if active_names is not None and name not in active_names:
            continue
        if name not in base or name not in direction:
            continue
        base_t = base[name].to(device=param.device, dtype=param.dtype)
        dir_t = direction[name].to(device=param.device, dtype=param.dtype)
        alpha_t = alpha.to(device=param.device, dtype=param.dtype)
        parameter_updates[f"{name_prefix}{name}"] = base_t + alpha_t * dir_t
    return parameter_updates


def _alpha_line_functional_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: torch.Tensor,
    active_names: set[str] | None = None,
) -> torch.Tensor:
    if "hidden_states" in batch:
        if not _alpha_line_parameter_updates(
            model,
            base,
            direction,
            alpha,
            active_names=active_names,
        ):
            raise RuntimeError(
                "alpha-line direction has no overlap with model parameters"
            )
        was_training = model.training
        model.eval()
        try:
            return alpha_line_loss_exact(
                model,
                batch,
                base,
                direction,
                alpha,
                active_names=active_names,
            )
        finally:
            if was_training:
                model.train()

    parameter_updates = _alpha_line_parameter_updates(
        model,
        base,
        direction,
        alpha,
        active_names=active_names,
    )
    if not parameter_updates:
        raise RuntimeError("alpha-line direction has no overlap with model parameters")
    was_training = model.training
    model.eval()
    try:
        outputs = functional_call(
            model,
            parameter_updates,
            args=(),
            kwargs={
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            },
        )
        return outputs.loss
    finally:
        if was_training:
            model.train()


def _accumulate_lora_grads_cpu(
    model: torch.nn.Module,
    accumulator: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Accumulate current LoRA gradients on CPU for future-work diagnostics."""
    result = accumulator if accumulator is not None else {}
    for name, param in iter_lora_params(model):
        if param.grad is None:
            continue
        grad = param.grad.detach().float().cpu()
        if name in result and result[name].shape == grad.shape:
            result[name].add_(grad)
        else:
            result[name] = grad.clone()
    return result


def _projection_ratio_to_direction(
    gradients: dict[str, torch.Tensor] | None,
    direction: dict[str, torch.Tensor],
    *,
    active_names: set[str] | None = None,
    eps: float = 1e-12,
) -> float | None:
    """Return ``|<g,v>| / (||g|| ||v||)`` over the active LoRA tensors."""
    if not gradients or not direction:
        return None
    dot = 0.0
    grad_norm_sq = 0.0
    direction_norm_sq = 0.0
    for name, grad in gradients.items():
        if active_names is not None and name not in active_names:
            continue
        if name not in direction:
            continue
        grad_f = grad.float().flatten()
        direction_f = direction[name].detach().float().cpu().flatten()
        if grad_f.numel() != direction_f.numel():
            continue
        dot += torch.dot(grad_f, direction_f).item()
        grad_norm_sq += torch.dot(grad_f, grad_f).item()
        direction_norm_sq += torch.dot(direction_f, direction_f).item()
    denom = (grad_norm_sq**0.5) * (direction_norm_sq**0.5)
    if denom <= eps:
        return None
    return abs(dot) / denom


def _next_supervised_batch(
    *,
    batch_iter: InfiniteBatchIterator,
    batch_plan_manifest,
    deterministic_data_order: bool,
    train_batch_position: int,
    max_empty_supervision_skips: int,
    cycle_batch_keys: list[str],
    cycle_sample_keys: list[str],
) -> tuple[dict[str, torch.Tensor], int]:
    empty_supervision_skips = 0
    while True:
        batch = batch_iter.next()
        batch_position = train_batch_position
        train_batch_position += 1
        if has_supervised_tokens(batch):
            break
        empty_supervision_skips += 1
        if empty_supervision_skips > max_empty_supervision_skips:
            raise RuntimeError(
                "Exceeded empty-supervision skip budget while filling alpha-line step"
            )
    if deterministic_data_order:
        batch_locator = batch_plan_manifest.batch_locator_at_position(batch_position)
        cycle_batch_keys.append(batch_locator.batch_key)
        cycle_sample_keys.extend(batch_locator.sample_keys)
    return batch, train_batch_position


def _decide_accept_rollback(
    loss_pilot: float,
    loss_after: float,
    rollback_tolerance: float,
    loss_history: list[float] | None = None,
    temperature: float = 0.0,
) -> tuple[bool, str]:
    """Decide whether to accept extrapolation or roll back.

    Enhanced with two noise-robust mechanisms:
    1. Moving-average baseline: If loss_history is provided, compare loss_after
       against the moving average (not just loss_pilot) to reduce false accepts
       from stochastic eval noise.
    2. Metropolis-Hastings soft accept: If temperature > 0, small degradations
       are accepted probabilistically (exp(-delta/temperature)), preventing
       the algorithm from getting stuck in local minima.

    Returns (accepted, reason).
    """
    # Use moving average as baseline if history is available (noise reduction).
    # The caller is responsible for slicing the desired window size.
    if loss_history:
        baseline = sum(loss_history) / len(loss_history)
    else:
        baseline = loss_pilot

    if loss_after <= baseline:
        return True, "improvement"

    diff = loss_after - baseline
    relative = diff / max(abs(baseline), 1e-8)

    if relative <= rollback_tolerance:
        return True, "within_tolerance"

    # Metropolis-Hastings probabilistic acceptance for borderline cases
    if temperature > 0 and relative < rollback_tolerance * 3:
        import random as _rng

        accept_prob = math.exp(-diff / temperature)
        if _rng.random() < accept_prob:
            return True, f"soft_accept:p={accept_prob:.3f}"

    return False, f"degradation:{relative:.4%}>tol={rollback_tolerance:.4%}"


def _format_cycle_progress(
    cycle: int,
    loss: float,
    accepted: bool,
    cos_sim: float,
    reduction_rate: float,
    K: int,
    N: int,
) -> str:
    """Format a single-line progress string for tqdm / logging."""
    return (
        f"c={cycle} loss={loss:.4f} {'Y' if accepted else 'N'}"
        f" cos={cos_sim:.3f} red={reduction_rate:.1%} K={K} N={N}"
    )


def _cosine_n_threshold_map(raw_thresholds: Any) -> dict[int, float]:
    if raw_thresholds is None:
        return {1: 0.0}
    if not hasattr(raw_thresholds, "items"):
        raise TypeError("cosine_n_selection_thresholds must be a mapping")
    return {int(n_steps): float(c_min) for n_steps, c_min in raw_thresholds.items()}


def _resolve_accept_eval_examples(cfg: DictConfig) -> int | None:
    """Resolve the cheap probe-eval size used for cycle accept/rollback checks."""
    raw = cfg.eval.get("accept_eval_examples", None)
    if raw is None:
        raw = cfg.eval.get("quick_eval_examples", None)
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        return None
    return value


def _decide_post_extrapolation_eval_policy(
    *,
    consistency: float,
    selected_N: int,
    total_cycles: int,
    acceptance_rate: float,
    velocity_anomalous: bool,
    enabled: bool,
    high_cos: float,
    mid_cos: float,
    mid_eval_every: int,
    min_cycles: int,
    min_acceptance_rate: float,
    force_eval_N: int,
) -> tuple[bool, str]:
    """Return ``(should_eval_after_extrapolation, reason)``.

    High-consistency cycles are accepted without the post-extrapolation probe.
    Mid-consistency cycles are sampled periodically.  Low consistency, anomalous
    velocity, warmup cycles, low historical acceptance, and large-N cycles keep
    the rollback probe enabled.
    """
    if not enabled:
        return True, "disabled"
    if force_eval_N > 0 and selected_N >= force_eval_N:
        return True, f"force_eval_N:{selected_N}"
    if velocity_anomalous:
        return True, "velocity_anomalous"
    if total_cycles < min_cycles:
        return True, "warmup"
    if acceptance_rate < min_acceptance_rate:
        return True, "low_acceptance_rate"
    if consistency >= high_cos:
        return False, "high_confidence"
    if consistency >= mid_cos:
        interval = max(1, int(mid_eval_every))
        if total_cycles % interval == 0:
            return True, f"mid_periodic:{interval}"
        return False, f"mid_skip:{interval}"
    return True, "low_confidence"


def _evaluate_full_eval_outcome(
    full_loss: float,
    prev_best: float,
    stale_cycles: int,
    patience: int | None,
    min_cycles: int,
    current_cycle: int,
    min_delta: float = 0.0,
) -> tuple[bool, bool, str]:
    """Determine full-eval outcomes: (is_new_best, should_stop, reason).

    ``min_delta`` is the §5.3 improvement-margin: a loss only counts as a new
    best when it beats ``prev_best`` by strictly more than ``min_delta``
    (Keras-style). Default 0.0 keeps ``full_loss < prev_best`` bit-identical.
    It must match the margin ``CycleState.record_full_eval`` uses so the
    computed ``new_stale`` agrees with the state the loop records.
    """
    is_new_best = prev_best - full_loss > min_delta
    new_stale = 0 if is_new_best else stale_cycles + 1
    should_stop = (
        patience is not None and new_stale >= patience and current_cycle >= min_cycles
    )
    if should_stop:
        reason = f"early_stop:stale={new_stale}>=patience={patience}"
    elif is_new_best:
        reason = f"new_best:{full_loss:.4f}"
    else:
        reason = f"no_improvement:stale={new_stale}"
    return is_new_best, should_stop, reason


def _should_fallback_to_baseline_like(
    *,
    proposal_N: int,
    total_cycles: int,
    acceptance_rate: float,
    pilot_loss: float,
    previous_valid_loss: float,
    acceleration: float,
    velocity_anomalous: bool,
    enabled: bool = False,
    warmup_cycles: int = 5,
    min_acceptance_rate: float = 0.0,
    pilot_margin: float = 0.01,
    max_positive_acceleration: float = 0.02,
) -> tuple[bool, str]:
    """Decide whether to skip extrapolation and keep only the pilot update.

    This approximates "local linearity still holds" with empirical signals:
    recent speculative acceptance, pilot stability against the previous valid
    loss, and velocity magnitude health. When these signals look bad, the
    current cycle behaves more like baseline training by setting N=0.
    """
    if proposal_N <= 0:
        return True, "no_extrapolation_requested"
    if not enabled:
        return False, "disabled"
    if total_cycles < warmup_cycles:
        return False, "warmup"

    stable_pilot = not math.isfinite(
        previous_valid_loss
    ) or pilot_loss <= previous_valid_loss * (1.0 + pilot_margin)

    if velocity_anomalous:
        return True, "velocity_anomaly"
    positive_acceleration_limit = max_positive_acceleration
    if (
        math.isfinite(acceleration)
        and acceleration > positive_acceleration_limit
        and not stable_pilot
    ):
        return True, f"positive_acceleration:{acceleration:.6f}"
    if acceptance_rate < min_acceptance_rate and not stable_pilot:
        return True, "low_acceptance_and_unstable_pilot"
    return False, "linearity_ok"


def _check_and_save_linearity_budget_checkpoint(
    model,
    tokenizer,
    valid_full_loader,
    input_device,
    cycle_state,
    grad_accum,
    triggered_target_steps,
    run_dir,
    logger,
    metrics,
):
    import json
    current_equiv_steps = (cycle_state.full_backward_passes + cycle_state.speculative_equivalent_backward_passes) // grad_accum
    target_steps = [250, 500, 750, 1000, 1250, 1500]
    for target in target_steps:
        if current_equiv_steps >= target and target not in triggered_target_steps:
            triggered_target_steps.add(target)
            logger.info(
                f"[Linearity Budget] Reached equivalent steps {current_equiv_steps} >= target {target}. "
                f"Triggering mandatory checkpoint and full eval."
            )
            # Force evaluate on valid_full
            full_result = eval_loss_detailed(model, valid_full_loader, input_device)
            logger.info(
                f"[Linearity Budget] Target {target} full eval: loss={full_result.avg_loss:.4f} "
                f"ppl={full_result.perplexity:.2f}"
            )
            
            # Save step-aligned full evaluation loss to run_metrics.jsonl
            metrics.record_step(
                step=target,
                cycle=cycle_state.total_cycles,
                loss_train=None,
                loss_valid=full_result.avg_loss,
                backward_passes=0,
                total_backward_passes=cycle_state.full_backward_passes,
                is_step_aligned_full_eval=True,
                aligned_target=target,
            )

            checkpoint_dir = run_dir / f"checkpoint-{target}"
            save_checkpoint(model, tokenizer, checkpoint_dir)
            logger.info(f"[Linearity Budget] Saved target checkpoint to {checkpoint_dir}")

            # Attempt to compare with Baseline at the same target step
            baseline_loss = None
            try:
                baseline_metrics_path = Path(run_dir).parent / "baseline" / "run_metrics.jsonl"
                if baseline_metrics_path.exists():
                    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            step = data.get("step") if data.get("step") is not None else data.get("global_step")
                            if step == target:
                                val_loss = data.get("loss_valid")
                                if val_loss is None:
                                    val_loss = data.get("valid_loss")
                                if val_loss is None:
                                    val_loss = data.get("best_valid_loss")
                                if val_loss is not None:
                                    baseline_loss = val_loss
            except Exception as e:
                logger.warning(f"[Linearity Budget] Failed to load Baseline metrics for comparison: {e}")

            actual_bp = cycle_state.full_backward_passes
            red_rate = cycle_state.reduction_rate
            epoch = target * grad_accum / 4000  # Epoch representation (samples digestion)
            
            if baseline_loss is not None:
                diff = full_result.avg_loss - baseline_loss
                logger.info(
                    f"\n================================================================================\n"
                    f"[Linearity Budget] STEP-ALIGNED COMPARISON AT TARGET STEP {target}:\n"
                    f"  - Epoch equivalent:     {epoch:.2f}\n"
                    f"  - Baseline Valid Loss:  {baseline_loss:.4f}\n"
                    f"  - TG-LoRA Valid Loss:   {full_result.avg_loss:.4f} (Diff: {diff:+.4f})\n"
                    f"  - TG-LoRA Actual BPs:   {actual_bp} (vs Baseline: {target * grad_accum})\n"
                    f"  - Reduction Rate:       {red_rate*100:.2f}%\n"
                    f"================================================================================"
                )
            else:
                logger.info(
                    f"\n================================================================================\n"
                    f"[Linearity Budget] STEP-ALIGNED METRICS AT TARGET STEP {target} (Baseline N/A):\n"
                    f"  - Epoch equivalent:     {epoch:.2f}\n"
                    f"  - TG-LoRA Valid Loss:   {full_result.avg_loss:.4f}\n"
                    f"  - TG-LoRA Actual BPs:   {actual_bp} (vs Baseline: {target * grad_accum})\n"
                    f"  - Reduction Rate:       {red_rate*100:.2f}%\n"
                    f"================================================================================"
                )


def build_training_summary(controller, cycle_state, delta_tracker) -> dict:
    ctrl = controller.summary()
    cs = cycle_state.summary()
    summary = ctrl.copy()
    # Both controller and cycle_state produce "acceptance_rate" and
    # "total_cycles" / "cycles".  Preserve the controller's values
    # under distinct keys before cycle_state.update() overwrites them.
    summary["controller_acceptance_rate"] = ctrl.get("acceptance_rate", 0.0)
    summary["controller_total_cycles"] = ctrl.get("total_cycles", 0)
    summary.update(cs)
    summary.update(delta_tracker.summary())
    return summary


# Names of the run-wide efficiency-accounting counters (GOAL §5 / P3 cost
# accounting) that accumulate across the whole ``train_tg_lora`` run and feed
# the run-end summary. They live as plain function-local tallies, so a
# fault/periodic resume that rebuilds them at zero silently corrupts the run-end
# cost report (validation_forwards_total, cache hit-rate, subspace-ZO / alpha-
# line tallies, future-work projection mean) — the resume-state-loss sibling to
# dynfreeze / best_full_eval / warmup / LAWA / triggered-target / act-regime.
# Snapshotted into ``TrainingState.efficiency_accounting`` and restored on
# resume. Order-independent (a dict); single source of truth shared by the fault
# and periodic save sites.
_EFFICIENCY_ACCOUNTING_KEYS: tuple[str, ...] = (
    "activation_cache_build_count",
    "activation_cache_eligible_count",
    "activation_cache_hit_count",
    "activation_cache_miss_count",
    "pilot_validation_forward_count",
    "post_validation_forward_count",
    "post_extrapolation_eval_count",
    "post_extrapolation_eval_skipped_count",
    "post_extrapolation_eval_skip_reasons",
    "subspace_zo_attempted_steps_total",
    "subspace_zo_accepted_steps_total",
    "subspace_zo_rejected_steps_total",
    "subspace_zo_forward_count_total",
    "subspace_zo_dim1_steps_total",
    "subspace_zo_dim2_steps_total",
    "alpha_line_steps_total",
    "alpha_line_base_recompute_total",
    "alpha_line_v_update_wall_seconds_total",
    "alpha_line_alpha_wall_seconds_total",
    "future_work_projection_ratios",
    "future_work_internal_pair_count",
)


def _snapshot_efficiency_accounting(
    scope: dict[str, object],
) -> dict[str, object]:
    """Snapshot the run-wide efficiency-accounting counters from the caller's
    local scope into a plain dict for ``TrainingState.efficiency_accounting``.

    Only keys present in ``scope`` are captured, so a counter renamed out from
    under ``_EFFICIENCY_ACCOUNTING_KEYS`` is silently omitted rather than written
    as ``None`` (a ``None`` restore would break the live ``+= 1`` sites).
    """
    return {
        key: scope[key] for key in _EFFICIENCY_ACCOUNTING_KEYS if key in scope
    }


def _save_fault_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    controller: RandomWalkController,
    cycle_state: CycleState,
    velocity: Velocity,
    delta_tracker: DeltaTracker,
    run_dir: Path,
    train_batch_position: int,
    accepted_valid_history: list[float],
    dynfreeze: DynamicFreezeController | None,
    best_full_eval_loss: float,
    best_full_eval_perplexity: float | None,
    warmup_released: bool,
    warmup_cos_consecutive: int,
    lawa_averager: LAWAAverager | None,
    best_lawa_loss: float,
    triggered_target_steps: set[int] | list[int] | None,
    act_regime_tracker: ActivationFingerprintTracker | None,
    efficiency_accounting: dict | None,
    psa_prior: PSAPrior | None,
) -> None:
    """Save model + full training state on fault (OOM / CUDA error).

    ``dynfreeze`` is the reversible-freeze controller (``None`` when the Guard
    experiment is disabled). It must be threaded in explicitly — the controller
    lives in the caller's (``train_tg_lora``) scope and is unreachable from
    module globals — so the fault checkpoint records its state for resume.

    ``best_full_eval_loss`` / ``best_full_eval_perplexity`` use the same
    threading rationale: they are caller-scoped trackers that gate the
    ``best_model/`` save, so the fault checkpoint must record them or resume
    sees them as ``inf``/``None`` and clobbers the genuine best on the first
    post-fault full eval.

    ``warmup_released`` / ``warmup_cos_consecutive`` likewise: they are the
    caller-scoped two-phase gate state. A fault checkpoint taken mid-production
    must record ``warmup_released=True`` or resume drops back into the pilot-only
    warmup phase and re-disables convergence/acceleration adaptation and
    extrapolation until the gate re-fires.

    ``lawa_averager`` likewise: it is the caller-scoped mandatory-baseline
    (GOAL §3.3) weight-averaging window. A fault checkpoint taken after the
    window has started recording must serialize it or resume rebuilds it empty,
    ``is_ready`` is False, and the LAWA comparison plus LAWA-averaged JSON eval
    are silently skipped until ``start_cycle`` worth of new snapshots
    re-accumulate — the resumed headline baseline measured over a post-fault-only
    window.

    ``best_lawa_loss`` likewise: it is the caller-scoped minimum of the LAWA
    comparison loss (reported in the run summary). A fault checkpoint must record
    it or resume resets it to ``inf`` and the run-end ``best_lawa_loss`` headline
    reflects only post-resume cycles — the LAWA fault fix persisted the snapshot
    ``lawa_state`` window but left this tracker un-persisted. Mirrors the
    ``best_full_eval_loss`` threading.

    ``triggered_target_steps`` likewise: it is the caller-scoped set of
    linearity-budget target steps (250/500/.../1500) already fired by
    ``_check_and_save_linearity_budget_checkpoint``. A fault checkpoint must
    record it or resume resets it to empty and the first post-resume cycle
    re-fires every already-crossed target — redundant full evals, re-saved
    ``checkpoint-{target}`` dirs, and duplicate ``is_step_aligned_full_eval``
    records corrupting the linearity-budget vs-baseline comparison. Mirrors the
    accepted_valid_history threading (a caller-scoped mutable collection).

    ``act_regime_tracker`` likewise: it is the caller-scoped activation-fingerprint
    regime inventory (GOAL §4 step 1), reported in the run-end summary as
    ``activation_regime_inventory`` / ``stable_fraction`` and feeding the §7 null
    baseline. A fault checkpoint taken after the tracker has accumulated steps
    must serialize it (``state_dict()``) or resume rebuilds it empty and the
    summary's regime fractions reflect only post-fault steps — a silent
    resume-state-loss sibling to the fixed LAWA window. ``None`` when the feature
    is disabled (``activation_regime_enabled: false``).

    ``efficiency_accounting`` likewise: it is the caller-scoped snapshot of the
    run-wide efficiency-accounting counters (GOAL §5 / P3). A fault checkpoint
    must record it or resume rebuilds every counter at zero/empty and the
    run-end cost report (cache hit-rate, validation_forwards_total, subspace-ZO
    / alpha-line tallies, future-work projection mean) reflects only post-fault
    cycles — a silent resume-state-loss sibling to the fixed LAWA / act-regime
    gaps. Mirrors the accepted_valid_history threading (a caller-scoped mutable
    collection). ``None`` when no counters accumulated.

    ``psa_prior`` likewise: it is the GOAL §1.5 / §3.3 PSA subspace-prior object
    (``None`` when ``enable_psa: false``). A fault checkpoint must record its
    run-wide accumulation (per-step ``_delta_history`` + extracted PC1
    ``priors`` that drive amplification + ``_prev_priors`` + the
    ``_prior_cosines`` series + the ``should_update`` timing) or resume rebuilds
    it empty, amplification is silently off until the next extract, and the
    run-end ``layer_delta_analysis`` (GOAL §4) is omitted on a short residual
    run — a silent resume-state-loss sibling to the fixed act-regime / LAWA /
    efficiency-accounting gaps. Mirrors the act_regime_tracker threading.
    """
    try:
        oom_dir = run_dir / "oom_checkpoint"
        save_checkpoint(model, tokenizer, oom_dir)
        logger.info("OOM checkpoint saved to %s", oom_dir)
    except Exception as exc:
        logger.error("Failed to save OOM model checkpoint: %s", exc)

    try:
        oom_dir = run_dir / "oom_checkpoint"
        ts = TrainingState(
            cycle_state=cycle_state,
            controller_state=controller.state,
            velocity=velocity,
            delta_tracker=delta_tracker,
            cycle_offset=cycle_state.cycle,
            adapter_checkpoint_dir=str(oom_dir),
            train_batch_position=train_batch_position,
            accepted_valid_history=list(accepted_valid_history),
            dynfreeze_state=dynfreeze.state_dict() if dynfreeze is not None else None,
            best_full_eval_loss=best_full_eval_loss,
            best_full_eval_perplexity=best_full_eval_perplexity,
            warmup_released=warmup_released,
            warmup_cos_consecutive=warmup_cos_consecutive,
            lawa_state=lawa_averager.state_dict() if lawa_averager is not None else None,
            best_lawa_loss=best_lawa_loss,
            triggered_target_steps=sorted(triggered_target_steps)
            if triggered_target_steps is not None
            else None,
            act_regime_state=(
                act_regime_tracker.state_dict()
                if act_regime_tracker is not None
                else None
            ),
            efficiency_accounting=efficiency_accounting,
            psa_state=psa_prior.state_dict() if psa_prior is not None else None,
        )
        save_training_state(ts, run_dir / "training_state.pt")
    except Exception as exc:
        logger.error("Failed to save training state: %s", exc)


def _is_cuda_error(exc: RuntimeError) -> bool:
    """Check if a RuntimeError originated from a GPU backend."""
    msg = str(exc).lower()
    return "cuda" in msg or "device-side" in msg or "mps" in msg


def train_tg_lora(cfg: DictConfig, resume_path: str | None = None) -> None:
    set_seed(cfg.experiment.seed)

    from src.training.preflight import validate_max_seq_len

    logger.info("Loading tokenizer and model...")
    tokenizer = load_tokenizer(cfg)
    validate_max_seq_len(cfg, tokenizer)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    input_device = get_input_device(model)

    trainable_lora_scope = cfg.training.get("trainable_lora_scope", "all")
    fixed_active_names, fixed_active_indices = configure_trainable_lora_scope(
        model,
        trainable_lora_scope,
    )

    run_dir = ensure_dir(cfg.logging.run_dir)
    metrics = RunMetrics(run_dir, mode="tg_lora")

    raw_train_dataset = load_dataset(
        cfg.data.train_path,
        tokenizer,
        cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )
    train_dataset: Dataset = raw_train_dataset
    valid_quick_dataset: Dataset = load_dataset(
        cfg.data.valid_quick_path,
        tokenizer,
        cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )
    valid_full_dataset: Dataset = load_dataset(
        cfg.data.valid_full_path,
        tokenizer,
        cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )

    # JSON-extraction gold-eval records (Guard experiment §5.2). Held as raw
    # dicts for generation-based scoring; the §5.2 stop is resolved post-hoc
    # by the analysis script from the recorded gold_* trajectory.
    gold_eval_records: list[dict] = []
    if cfg.eval.get("gold_eval_enabled", False) and cfg.data.get("gold_test_path"):
        with open(cfg.data.gold_test_path) as _gold_file:
            gold_eval_records = [json.loads(line) for line in _gold_file if line.strip()]
        logger.info(
            "Gold eval enabled: %d records from %s (every %d cycles)",
            len(gold_eval_records),
            cfg.data.gold_test_path,
            cfg.eval.get("gold_eval_every_cycles", 5),
        )

    use_prefix_feature_cache = cfg.training.get(
        "prefix_feature_cache_experimental", False
    )
    prefix_feature_cache_summary = {
        "prefix_feature_cache_experimental": use_prefix_feature_cache,
        "trainable_lora_scope": trainable_lora_scope,
    }
    deterministic_data_order = bool(cfg.training.get("deterministic_data_order", True))
    save_batch_plan_manifest = bool(cfg.training.get("save_batch_plan_manifest", True))
    batch_plan_manifest = build_deterministic_batch_plan_for_dataset(
        raw_train_dataset,
        batch_size=cfg.training.batch_size,
    )
    prefix_cache_split_layer: int | None = None
    cache_loader_kwargs = _cached_loader_kwargs(cfg)
    prefix_feature_cache_dir = Path(
        str(cfg.training.get("prefix_feature_cache_dir", ".cache/prefix_feature_cache"))
    )
    prefix_feature_cache_force_rebuild = bool(
        cfg.training.get("prefix_feature_cache_force_rebuild", False)
    )
    prefix_feature_cache_mode = str(
        cfg.training.get("prefix_feature_cache_mode", "reuse")
    )
    prefix_feature_cache_lazy_disk = prefix_feature_cache_mode == "one_shot"
    prefix_feature_cache_share_across_seeds = bool(
        cfg.training.get("prefix_feature_cache_share_across_seeds", False)
    )
    prefix_feature_cache_offload_prefix_to_cpu = bool(
        cfg.training.get("prefix_feature_cache_offload_prefix_to_cpu", False)
    )
    cached_prefix_datasets: dict[Path, PrefixFeatureDatasetBase] = {}

    use_async_cache = bool(cfg.training.get("prefix_feature_cache_async", False))
    background_device = cfg.training.get("prefix_feature_cache_async_device", None)
    async_builder: AsyncCacheBuilder | None = None
    async_ready = False
    swap_cycle_vq: int | None = None
    swap_cycle_vf: int | None = None

    if use_prefix_feature_cache:
        prefix_feature_cache_dir.mkdir(parents=True, exist_ok=True)
        prefix_feature_cache_summary["prefix_feature_cache_dir"] = str(
            prefix_feature_cache_dir
        )
        prefix_feature_cache_summary["prefix_feature_cache_force_rebuild"] = (
            prefix_feature_cache_force_rebuild
        )
        prefix_feature_cache_summary["prefix_feature_cache_mode"] = (
            prefix_feature_cache_mode
        )
        prefix_feature_cache_summary["prefix_feature_cache_share_across_seeds"] = (
            prefix_feature_cache_share_across_seeds
        )
        prefix_feature_cache_summary["prefix_feature_cache_offload_prefix_to_cpu"] = (
            prefix_feature_cache_offload_prefix_to_cpu
        )
        prefix_feature_cache_summary["prefix_feature_cache_runtime_offload_applied"] = (
            False
        )

    if use_prefix_feature_cache and trainable_lora_scope != "last_25_percent":
        raise ValueError(
            "prefix_feature_cache_experimental requires training.trainable_lora_scope=last_25_percent"
        )

    def _maybe_cache_dataset(
        label: str,
        dataset: Dataset,
        dataset_path: str,
        enabled: bool,
    ) -> Dataset:
        prefix_feature_cache_summary[f"prefix_feature_cache_{label}_enabled"] = enabled
        if not enabled:
            return dataset
        if prefix_cache_split_layer is None:
            raise RuntimeError(
                "prefix cache split layer must be set before dataset caching"
            )

        metadata = build_prefix_feature_cache_metadata(
            dataset_path=dataset_path,
            model_name=cfg.model.name_or_path,
            seed=resolve_prefix_feature_cache_seed(
                cfg.experiment.seed,
                share_across_seeds=prefix_feature_cache_share_across_seeds,
            ),
            max_seq_len=cfg.data.max_seq_len,
            split_layer_idx=prefix_cache_split_layer,
            lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            lora_target_modules=cfg.lora.target_modules,
            trainable_lora_scope=trainable_lora_scope,
        )
        cache_path = get_prefix_feature_cache_path(prefix_feature_cache_dir, metadata)
        prefix_feature_cache_summary[f"prefix_feature_cache_{label}_path"] = str(
            cache_path
        )

        build_seconds = 0.0
        load_seconds = 0.0
        save_seconds = 0.0
        if not prefix_feature_cache_lazy_disk and cache_path in cached_prefix_datasets:
            cached_dataset = cached_prefix_datasets[cache_path]
            source = "memory"
        elif cache_path.exists() and not prefix_feature_cache_force_rebuild:
            started = time.perf_counter()
            cached_dataset = load_prefix_feature_dataset(
                cache_path,
                lazy=prefix_feature_cache_lazy_disk,
            )
            load_seconds = time.perf_counter() - started
            if not prefix_feature_cache_lazy_disk:
                cached_prefix_datasets[cache_path] = cached_dataset
            source = "disk"
        else:
            started = time.perf_counter()
            built_dataset = build_prefix_feature_dataset(
                model,
                dataset,
                batch_size=cfg.training.batch_size,
                device=input_device,
                split_layer_idx=prefix_cache_split_layer,
                num_workers=int(cast(int, cache_loader_kwargs.get("num_workers", 0))),
                pin_memory=bool(cache_loader_kwargs.get("pin_memory", False)),
                persistent_workers=bool(
                    cache_loader_kwargs.get("persistent_workers", False)
                ),
                prefetch_factor=cast(
                    int | None, cache_loader_kwargs.get("prefetch_factor")
                ),
            )
            build_seconds = time.perf_counter() - started
            save_started = time.perf_counter()
            save_prefix_feature_dataset(
                built_dataset,
                cache_path,
                metadata=metadata,
            )
            save_seconds = time.perf_counter() - save_started
            if prefix_feature_cache_lazy_disk:
                load_started = time.perf_counter()
                cached_dataset = load_prefix_feature_dataset(cache_path, lazy=True)
                load_seconds = time.perf_counter() - load_started
            else:
                cached_dataset = built_dataset
                cached_prefix_datasets[cache_path] = cached_dataset
            source = "built"

        prefix_feature_cache_summary.update(
            {
                f"prefix_feature_cache_{label}_source": source,
                f"prefix_feature_cache_{label}_cache_hit": source != "built",
                f"prefix_feature_cache_{label}_examples": len(cached_dataset),
                f"prefix_feature_cache_{label}_gib": round(
                    cached_dataset.total_bytes / 1024**3, 3
                ),
                f"prefix_feature_cache_{label}_build_seconds": round(build_seconds, 3),
                f"prefix_feature_cache_{label}_load_seconds": round(load_seconds, 3),
                f"prefix_feature_cache_{label}_save_seconds": round(save_seconds, 3),
            }
        )
        return cached_dataset

    if use_prefix_feature_cache:
        if cfg.lora.dropout != 0.0:
            raise ValueError(
                "prefix_feature_cache_experimental requires lora.dropout=0.0 so cached prefix activations remain deterministic"
            )

        prefix_cache_split_layer = min(fixed_active_indices)
        logger.info(
            "Prefix feature cache experimental mode enabled: split_layer=%d trainable_suffix_layers=%s",
            prefix_cache_split_layer,
            sorted(fixed_active_indices),
        )
        prefix_feature_cache_summary["prefix_feature_cache_split_layer"] = (
            prefix_cache_split_layer
        )

        if use_async_cache and background_device:
            prefix_feature_cache_summary["prefix_feature_cache_async"] = True
            prefix_feature_cache_summary["prefix_feature_cache_async_device"] = (
                background_device
            )
            train_dataset = _maybe_cache_dataset(
                "train",
                train_dataset,
                cfg.data.train_path,
                bool(cfg.training.get("prefix_feature_cache_train", True)),
            )
            async_builder = AsyncCacheBuilder(
                cfg=cfg,
                raw_datasets={
                    "valid_quick": valid_quick_dataset,
                    "valid_full": valid_full_dataset,
                },
                cache_loader_kwargs=cache_loader_kwargs,
                split_layer=prefix_cache_split_layer,
                cache_dir=prefix_feature_cache_dir,
                force_rebuild=prefix_feature_cache_force_rebuild,
                trainable_lora_scope=trainable_lora_scope,
                background_device=background_device,
            )
            async_builder.start()
            logger.info("Async cache builder started on %s", background_device)
        else:
            train_dataset = _maybe_cache_dataset(
                "train",
                train_dataset,
                cfg.data.train_path,
                bool(cfg.training.get("prefix_feature_cache_train", True)),
            )
            valid_quick_dataset = _maybe_cache_dataset(
                "valid_quick",
                valid_quick_dataset,
                cfg.data.valid_quick_path,
                bool(cfg.training.get("prefix_feature_cache_valid_quick", True)),
            )
            valid_full_dataset = _maybe_cache_dataset(
                "valid_full",
                valid_full_dataset,
                cfg.data.valid_full_path,
                bool(cfg.training.get("prefix_feature_cache_valid_full", True)),
            )
        build_seconds = 0.0
        load_seconds = 0.0
        for key, value in prefix_feature_cache_summary.items():
            if key.endswith("_build_seconds"):
                build_seconds += float(value)
            if key.endswith("_load_seconds"):
                load_seconds += float(value)
        prefix_feature_cache_summary["prefix_feature_cache_total_build_seconds"] = (
            round(build_seconds, 3)
        )
        prefix_feature_cache_summary["prefix_feature_cache_total_load_seconds"] = round(
            load_seconds, 3
        )
        prefix_feature_cache_summary["prefix_feature_cache_loader_num_workers"] = int(
            cast(int, cache_loader_kwargs.get("num_workers", 0))
        )
        prefix_feature_cache_summary["prefix_feature_cache_loader_pin_memory"] = bool(
            cache_loader_kwargs.get("pin_memory", False)
        )

    train_loader = _build_loader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=not deterministic_data_order,
        cached_loader_kwargs=cache_loader_kwargs,
        epoch_batches=(
            batch_plan_manifest.epoch_batches if deterministic_data_order else None
        ),
    )
    eval_batch_size = cfg.eval.get("eval_batch_size", 16)
    valid_quick_loader = _build_loader(
        valid_quick_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        cached_loader_kwargs=cache_loader_kwargs,
    )
    valid_full_loader = _build_loader(
        valid_full_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        cached_loader_kwargs=cache_loader_kwargs,
    )

    def _maybe_apply_prefix_runtime_offload() -> bool:
        if not prefix_feature_cache_offload_prefix_to_cpu:
            return False
        if prefix_cache_split_layer is None:
            prefix_feature_cache_summary[
                "prefix_feature_cache_runtime_offload_reason"
            ] = "split_layer_unset"
            return False
        if prefix_feature_cache_summary.get(
            "prefix_feature_cache_runtime_offload_applied"
        ):
            return False
        if not all(
            isinstance(loader.dataset, PrefixFeatureDatasetBase)
            for loader in (train_loader, valid_quick_loader, valid_full_loader)
        ):
            prefix_feature_cache_summary[
                "prefix_feature_cache_runtime_offload_reason"
            ] = "raw_dataset_remaining"
            return False

        allocated_before = _gpu_allocated_mb(input_device)
        offload_summary = offload_prefix_runtime_to_cpu(
            model,
            split_layer_idx=prefix_cache_split_layer,
        )
        allocated_after = _gpu_allocated_mb(input_device)
        prefix_feature_cache_summary.update(
            {
                "prefix_feature_cache_runtime_offload_applied": True,
                "prefix_feature_cache_runtime_offload_reason": "applied",
                "prefix_feature_cache_offloaded_prefix_modules": offload_summary[
                    "offloaded_prefix_modules"
                ],
                "prefix_feature_cache_offloaded_prefix_parameters": offload_summary[
                    "offloaded_prefix_parameters"
                ],
                "prefix_feature_cache_offloaded_prefix_input_embeddings": offload_summary[
                    "offloaded_prefix_input_embeddings"
                ],
            }
        )
        if allocated_before is not None:
            prefix_feature_cache_summary[
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before"
            ] = allocated_before
        if allocated_after is not None:
            prefix_feature_cache_summary[
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after"
            ] = allocated_after
        if allocated_before is not None and allocated_after is not None:
            prefix_feature_cache_summary[
                "prefix_feature_cache_runtime_offload_gpu_freed_mb"
            ] = round(allocated_before - allocated_after, 1)
        logger.info(
            "Prefix runtime offload applied: split_layer=%d modules=%d params=%d",
            prefix_cache_split_layer,
            offload_summary["offloaded_prefix_modules"],
            offload_summary["offloaded_prefix_parameters"],
        )
        return True

    _maybe_apply_prefix_runtime_offload()

    tg_cfg = cfg.tg_lora

    # JSON generation-quality eval set (structured-output domain task).
    # Raw records (dicts with text/completion/prompt) for batched generation +
    # scoring at full-eval cycles. Headline quality metric for the
    # plain/LAWA/PSA efficiency comparison (GOAL §3.3/§4.3).
    json_eval_records: list[dict] = []
    if tg_cfg.get("json_eval_enabled", False):
        json_eval_path = tg_cfg.get("json_eval_path", "")
        if json_eval_path:
            with open(json_eval_path) as _jf:
                json_eval_records = [json.loads(line) for line in _jf if line.strip()]
            logger.info(
                "JSON eval enabled: %d records from %s (every %d cycles)",
                len(json_eval_records),
                json_eval_path,
                tg_cfg.get("json_eval_every_cycles", 10),
            )
        else:
            logger.warning(
                "json_eval_enabled is true but json_eval_path is empty; skipping JSON eval."
            )

    controller = RandomWalkController(
        K_initial=tg_cfg.K_initial,
        K_candidates=list(tg_cfg.K_candidates),
        N_initial=tg_cfg.N_initial,
        N_candidates=list(tg_cfg.N_candidates),
        alpha_initial=tg_cfg.alpha_initial,
        alpha_min=tg_cfg.alpha_min,
        alpha_max=tg_cfg.alpha_max,
        alpha_log_sigma=tg_cfg.alpha_log_sigma,
        beta_initial=tg_cfg.beta_initial,
        beta_candidates=list(tg_cfg.beta_candidates),
        lr_initial=tg_cfg.get("lr_initial", cfg.training.learning_rate),
        lr_min=tg_cfg.get("lr_min", 1e-5),
        lr_max=tg_cfg.get("lr_max", 1e-3),
        lr_accept_boost=tg_cfg.get("lr_accept_boost", 1.2),
        lr_reject_decay=tg_cfg.get("lr_reject_decay", 0.5),
        active_layer_strategy=tg_cfg.active_layer_strategy,
        relative_update_cap=tg_cfg.relative_update_cap,
        rollback_tolerance=cfg.eval.rollback_tolerance,
        enable_random_walk=tg_cfg.get("enable_random_walk", True),
        enable_convergence_adaptation=tg_cfg.get("enable_convergence_adaptation", True),
        k_explore_prob=tg_cfg.get("k_explore_prob", None),
        n_explore_prob=tg_cfg.get("n_explore_prob", None),
        beta_explore_prob=tg_cfg.get("beta_explore_prob", None),
        strategy_explore_prob=tg_cfg.get("strategy_explore_prob", None),
        lr_explore_prob=tg_cfg.get("lr_explore_prob", None),
        lr_log_sigma=tg_cfg.get("lr_log_sigma", None),
        accel_instability_lr_decay=tg_cfg.get("accel_instability_lr_decay", None),
        accel_convergence_lr_boost=tg_cfg.get("accel_convergence_lr_boost", None),
    )

    velocity = Velocity(
        beta_short=beta_from_window(tg_cfg.get("cosine_n_selection_short_window", 3)),
        beta_long=beta_from_window(tg_cfg.get("cosine_n_selection_long_window", 10)),
    )
    rollback_mgr = RollbackManager()
    delta_tracker = DeltaTracker()
    optimizer_lifecycle = OptimizerLifecycleManager(
        model,
        lr=controller.state.lr,
        weight_decay=cfg.training.weight_decay,
        policy=cfg.training.get("optimizer_lifecycle", "recreate_per_cycle"),
    )
    batch_iter = InfiniteBatchIterator(train_loader, input_device)
    activation_cache = ActivationCache()
    num_decoder_layers = get_num_layers(model)

    # Progressive freeze controller (Phase 1 gate)
    progressive_freeze: ProgressiveFreezeController | None = None
    if tg_cfg.get("progressive_freeze_enabled", False):
        progressive_freeze = ProgressiveFreezeController(
            start_cycle=int(tg_cfg.get("progressive_freeze_start_cycle", 3)),
            freeze_layer=tg_cfg.get("progressive_freeze_layer", "last_active"),
            active_layer_indices=fixed_active_indices,
        )
        logger.info(
            "Progressive freeze enabled: start_cycle=%d layer=%s active=%s",
            progressive_freeze._start_cycle,
            tg_cfg.get("progressive_freeze_layer", "last_active"),
            sorted(fixed_active_indices),
        )

    # Dynamic reversible freeze controller (Guard experiment)
    dynfreeze: DynamicFreezeController | None = None
    if bool(tg_cfg.get("dynfreeze_enabled", False)):
        dynfreeze = DynamicFreezeController(
            tau=float(tg_cfg.get("dynfreeze_tau", 0.015)),
            window=int(tg_cfg.get("dynfreeze_window", 5)),
            stir_interval=int(tg_cfg.get("dynfreeze_stir_interval", 10)),
            upstream_activity_factor=float(tg_cfg.get("dynfreeze_upstream_activity_factor", 1.5)),
            epsilon_ratio=float(tg_cfg.get("dynfreeze_epsilon_ratio", 0.01)),
            a_mask_ratio=float(tg_cfg.get("dynfreeze_a_mask_ratio", 0.1)),
            all_layer_indices=sorted(fixed_active_indices),
        )
        logger.info(
            "Guard enabled: tau=%.4f window=%d stir=%d upstream_factor=%.1f layers=%s",
            dynfreeze._tau, dynfreeze._window, dynfreeze._stir_interval,
            dynfreeze._upstream_activity_factor, dynfreeze._all_layers,
        )
    train_batch_position = 0

    cycle_state = CycleState()
    triggered_target_steps = set()
    best_full_eval_perplexity = None
    best_full_eval_loss = float("inf")
    best_lawa_loss = float("inf")
    warmup_released = False
    warmup_cos_consecutive = 0
    production_start_full_backward_passes = 0
    restored_training_state: TrainingState | None = None

    # Resume from a previously saved training state if requested.
    cycle_offset = 0
    if resume_path is not None:
        ts = load_training_state(Path(resume_path))
        restored_training_state = ts
        controller.restore_state(ts.controller_state)
        velocity = ts.velocity
        delta_tracker = ts.delta_tracker
        cycle_state = ts.cycle_state
        cycle_offset = ts.cycle_offset
        train_batch_position = ts.train_batch_position
        if train_batch_position > 0:
            batch_iter.advance(train_batch_position)
        if ts.adapter_checkpoint_dir is not None:
            adapter_dir = Path(ts.adapter_checkpoint_dir)
            if adapter_dir.exists():
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file

                adapter_state = load_file(adapter_dir / "adapter_model.safetensors")
                set_peft_model_state_dict(model, adapter_state)
                logger.info("Restored LoRA adapter weights from %s", adapter_dir)
            else:
                logger.warning(
                    "Adapter checkpoint dir not found: %s — LoRA weights are fresh",
                    adapter_dir,
                )
        logger.info(
            "Resumed training from %s (cycle %d, batch_position=%d, acceptance_rate=%.1f%%)",
            resume_path,
            cycle_offset,
            train_batch_position,
            cycle_state.acceptance_rate * 100,
        )
        # Restore dynfreeze state if present
        if dynfreeze is not None and ts.dynfreeze_state is not None:
            dynfreeze.load_state_dict(ts.dynfreeze_state)
            logger.info(
                "DynFreeze state restored: frozen_layers=%s frozen_since=%d",
                sorted(dynfreeze.frozen_layer_indices),
                dynfreeze._frozen_since_cycle,
            )
        # Restore the best-full-eval trackers so the post-resume save-best gate
        # compares against the genuine pre-fault best, not inf — otherwise the
        # first full eval after resume unconditionally overwrites "best_model/".
        best_full_eval_loss = ts.best_full_eval_loss
        best_full_eval_perplexity = ts.best_full_eval_perplexity
        # Restore the warmup phase so a checkpoint taken mid-production does not
        # silently drop back into the pilot-only warmup phase on resume — which
        # would re-disable convergence/acceleration adaptation and extrapolation
        # until the gate re-fires. Mirrors the best_full_eval_* restore above.
        warmup_released = ts.warmup_released
        warmup_cos_consecutive = ts.warmup_cos_consecutive
        # Restore the best-LAWA-loss headline tracker so the resumed run-end
        # summary reflects the genuine run-wide minimum, not a post-resume-only
        # inf-restarted value. A plain float (no dependency on the averager being
        # built yet), so it restores here alongside the other best_* trackers —
        # unlike the lawa_state window which restores after the averager is built.
        best_lawa_loss = ts.best_lawa_loss
        # Restore the linearity-budget target-step set so a resumed run does not
        # re-fire every already-crossed target (250/500/.../1500) on its first
        # cycle — which would re-run redundant full evals, re-save
        # checkpoint-{target} dirs, and emit duplicate is_step_aligned_full_eval
        # records corrupting the vs-baseline comparison dataset. The serialized
        # form is a list (or None on a pre-fix checkpoint); restore to a set.
        triggered_target_steps = set(ts.triggered_target_steps or [])

    mlflow_cfg = cfg.logging.get("mlflow", {})
    batch_plan_manifest_path = run_dir / "batch_plan_manifest.json"
    if save_batch_plan_manifest:
        batch_plan_manifest.save(batch_plan_manifest_path)
    trajectory_key = build_trajectory_key(
        mode="tg_lora",
        epoch_batch_plan_key=batch_plan_manifest.epoch_batch_plan_key,
        trainable_lora_scope=trainable_lora_scope,
        optimizer_lifecycle=cfg.training.get("optimizer_lifecycle", None),
        model_name=cfg.model.name_or_path,
        max_seq_len=cfg.data.max_seq_len,
        deterministic_data_order=deterministic_data_order,
    )
    save_trajectory_delta_artifacts = bool(
        cfg.training.get("save_trajectory_delta_artifacts", False)
    )
    trajectory_delta_artifact_interval = int(
        cfg.training.get("trajectory_delta_artifact_interval", 1)
    )
    trajectory_delta_artifact_dir = run_dir / "trajectory_delta_artifacts"
    mlf = MLflowLogger(
        enabled=mlflow_cfg.get("enabled", True),
        tracking_uri=mlflow_cfg.get("tracking_uri", "") or None,
        experiment_name=mlflow_cfg.get("experiment_name", "") or None,
        run_name=cfg.experiment.name,
    )
    mlf.__enter__()
    if mlf.enabled:
        try:
            mlf.log_params(
                {
                    "model": cfg.model.name_or_path,
                    "lora_r": cfg.lora.r,
                    "lora_alpha": cfg.lora.alpha,
                    "batch_size": cfg.training.batch_size,
                    "grad_accumulation": cfg.training.grad_accumulation,
                    "learning_rate": cfg.training.learning_rate,
                    "optimizer_lifecycle": cfg.training.get(
                        "optimizer_lifecycle", "recreate_per_cycle"
                    ),
                    "trainable_lora_scope": trainable_lora_scope,
                    "prefix_feature_cache_experimental": use_prefix_feature_cache,
                    "prefix_feature_cache_train": cfg.training.get(
                        "prefix_feature_cache_train", True
                    ),
                    "prefix_feature_cache_valid_quick": cfg.training.get(
                        "prefix_feature_cache_valid_quick", True
                    ),
                    "prefix_feature_cache_valid_full": cfg.training.get(
                        "prefix_feature_cache_valid_full", True
                    ),
                    "max_cycles": cfg.training.max_cycles,
                    "seed": cfg.experiment.seed,
                    "K_initial": tg_cfg.K_initial,
                    "N_initial": tg_cfg.N_initial,
                    "alpha_initial": tg_cfg.alpha_initial,
                    "beta_initial": tg_cfg.beta_initial,
                }
            )
        except Exception:
            logger.warning("MLflow log_params failed, continuing without MLflow params")

    early_stop_patience = cfg.training.get("early_stopping_patience", None)
    min_cycles_before_stop = cfg.training.get("min_cycles_before_stop", 10)
    # §5.3 improvement-margin: a full-eval decrease must beat the best by more
    # than min_delta to count as a new best (Keras-style). Default 0.0 keeps
    # the historical contract. Threaded into CycleState + the stop decision so
    # the recorded stale count and the stop reason always agree.
    early_stopping_min_delta = cfg.training.get("early_stopping_min_delta", 0.0)
    cycle_state.min_delta = early_stopping_min_delta
    accept_eval_examples = _resolve_accept_eval_examples(cfg)

    # 学習開始前の基準 loss は fresh run だけで計算し、resume 時は保存済み状態を使う。
    if restored_training_state is None:
        logger.info("Computing initial eval loss for pilot rollback baseline...")
        original_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
        if original_use_cache is not None:
            model.config.use_cache = False
        try:
            initial_loss = eval_loss(
                model,
                valid_quick_loader,
                input_device,
                max_examples=accept_eval_examples,
            )
        finally:
            if original_use_cache is not None:
                model.config.use_cache = original_use_cache
        cycle_state.last_valid_loss = initial_loss
        accepted_valid_history: list[float] = [initial_loss]
        logger.info(f"Initial eval loss: {initial_loss:.4f}")
    else:
        initial_loss = cycle_state.last_valid_loss
        restored_history = restored_training_state.accepted_valid_history
        accepted_valid_history = list(restored_history or [initial_loss])
        logger.info(
            "Skipping initial eval on resume; restored last_valid_loss=%.4f with %d history points",
            initial_loss,
            len(accepted_valid_history),
        )
    moving_avg_window = cfg.eval.get("moving_avg_window", 3)
    soft_accept_temperature = cfg.eval.get("soft_accept_temperature", 0.0)
    validation_skip_enabled = bool(tg_cfg.get("validation_skip_enabled", False))
    validation_skip_high_cos = float(tg_cfg.get("validation_skip_high_cos", 0.85))
    validation_skip_mid_cos = float(tg_cfg.get("validation_skip_mid_cos", 0.70))
    validation_skip_mid_eval_every = int(
        tg_cfg.get("validation_skip_mid_eval_every", 3)
    )
    validation_skip_min_cycles = int(
        tg_cfg.get(
            "validation_skip_min_cycles",
            tg_cfg.get("confident_skip_min_cycles", 0),
        )
    )
    validation_skip_min_acceptance_rate = float(
        tg_cfg.get("validation_skip_min_acceptance_rate", 0.8)
    )
    validation_skip_force_eval_N = int(tg_cfg.get("validation_skip_force_eval_N", 20))
    subspace_zo_enabled = bool(tg_cfg.get("subspace_zo_enabled", False))
    subspace_zo_tau_dim = float(tg_cfg.get("subspace_zo_tau_dim", 0.15))
    subspace_zo_tau_cos = float(tg_cfg.get("subspace_zo_tau_cos", 0.70))
    subspace_zo_mu_ratio = float(tg_cfg.get("subspace_zo_mu_ratio", 0.001))
    subspace_zo_eps_curv = float(tg_cfg.get("subspace_zo_eps_curv", 1e-8))
    subspace_zo_eta_fallback_ratio = float(
        tg_cfg.get("subspace_zo_eta_fallback_ratio", 1e-2)
    )
    subspace_zo_max_step_ratio = float(tg_cfg.get("subspace_zo_max_step_ratio", 0.02))
    subspace_zo_max_steps_per_cycle = int(
        tg_cfg.get("subspace_zo_max_steps_per_cycle", 10)
    )
    subspace_zo_force_dim = int(tg_cfg.get("subspace_zo_force_dim", 0))
    subspace_zo_disable_curvature = bool(
        tg_cfg.get("subspace_zo_disable_curvature", False)
    )
    subspace_zo_stop_on_positive_g1 = bool(
        tg_cfg.get("subspace_zo_stop_on_positive_g1", True)
    )
    subspace_zo_g1_stop_epsilon = float(tg_cfg.get("subspace_zo_g1_stop_epsilon", 0.0))
    subspace_m9_enabled = bool(tg_cfg.get("subspace_m9_enabled", False))
    subspace_m9_fd_eps = float(tg_cfg.get("subspace_m9_fd_eps", 1e-3))
    subspace_m9_lr = float(tg_cfg.get("subspace_m9_lr", 0.5))
    subspace_m9_steps = int(tg_cfg.get("subspace_m9_steps", 1))

    psa_prior = None
    psa_gain_map: dict[str, float] = {}
    regime_detector = None
    psa_regime_reset_enabled = True
    if bool(tg_cfg.get("enable_psa", False)):
        from src.tg_lora.psa import amplify_gradients_psa, summarize_by_layer_type
        from src.tg_lora.regime import RegimeDetector
        psa_regime_reset_enabled = bool(tg_cfg.get("psa_regime_reset_enabled", True))
        psa_prior = PSAPrior(
            history_length=int(tg_cfg.get("psa_history_length", 6)),
            gain=float(tg_cfg.get("psa_gain", 0.5)),
            update_interval=int(tg_cfg.get("psa_update_interval", 3)),
            warmup_steps=int(tg_cfg.get("psa_warmup_steps", 4)),
            l2_reg=float(tg_cfg.get("psa_l2_reg", 0.01)),
            regime_plateau_gain=float(tg_cfg.get("psa_regime_plateau_gain", 0.5)),
        )
        regime_detector = RegimeDetector(
            window=int(tg_cfg.get("psa_regime_window", 8)),
            plateau_eps=float(tg_cfg.get("psa_regime_plateau_eps", 1e-4)),
            transition_z=float(tg_cfg.get("psa_regime_transition_z", 2.0)),
        )
        logger.info(
            "PSA enabled: gain=%.2f, history=%d, interval=%d, warmup=%d, l2_reg=%.4f, regime_reset=%s",
            psa_prior.gain, psa_prior.history_length,
            psa_prior.update_interval, psa_prior.warmup_steps, psa_prior.l2_reg,
            psa_regime_reset_enabled,
        )

    # Restore the PSA subspace-prior accumulation on resume. Placed here (after
    # the prior is constructed) rather than in the resume block above because
    # ``psa_prior`` is built just above — restoring there would hit
    # UnboundLocalError. ``restored_training_state`` is the None-when-not-
    # resuming handle. Without this restore a checkpoint taken after the prior
    # accumulated deltas / extracted priors loads an empty prior: amplification
    # is silently off until 2 deltas re-accumulate and the next extract fires,
    # and the run-end ``layer_delta_analysis`` (gated on ``history_count >= 2``)
    # is omitted if the residual run is short. Mirrors the LAWA / act-regime /
    # dynfreeze restores; guarded so PSA-disabled runs and pre-fix checkpoints
    # (None) are untouched.
    if (
        psa_prior is not None
        and restored_training_state is not None
        and restored_training_state.psa_state is not None
    ):
        psa_prior.load_state_dict(restored_training_state.psa_state)
        logger.info(
            "PSA prior restored: deltas=%d priors=%d last_update_step=%d",
            psa_prior.history_count,
            len(psa_prior.priors),
            psa_prior._last_update_step,
        )

    # Activation-fingerprint regime inventory (GOAL §4 step 1)
    act_regime_tracker = None
    if bool(tg_cfg.get("activation_regime_enabled", False)):
        from src.tg_lora.activation_regime import compute_regime_null_baseline
        act_regime_tracker = ActivationFingerprintTracker(
            window=int(tg_cfg.get("activation_regime_window", 10)),
            stable_threshold=float(tg_cfg.get("activation_regime_stable_threshold", 0.95)),
            chaotic_threshold=float(tg_cfg.get("activation_regime_chaotic_threshold", 0.5)),
            transition_drop_z=float(tg_cfg.get("activation_regime_transition_drop_z", 2.0)),
            min_history=int(tg_cfg.get("activation_regime_min_history", 3)),
        )
        # Register hook on the last decoder layer for fingerprint capture
        _target_layer = None
        for _name, _mod in model.named_modules():
            if hasattr(_mod, "layers") and hasattr(_mod, "__len__"):
                _target_layer = _mod[-1]
        if _target_layer is not None:
            act_regime_tracker.register_hook(_target_layer)
            logger.info("Activation regime tracker: hooked to last decoder layer")
        else:
            logger.warning("Activation regime tracker: could not find decoder layers, disabling")
            act_regime_tracker = None

    # Restore the activation-fingerprint regime inventory on resume. Placed here
    # (after the tracker is constructed and hooked) rather than in the resume
    # block above because ``act_regime_tracker`` is built just above — restoring
    # there would hit UnboundLocalError. ``restored_training_state`` is the
    # None-when-not-resuming handle. Without this restore a checkpoint taken
    # after the tracker has accumulated steps loads an empty tracker and the
    # run-end summary's ``activation_regime_inventory`` / ``stable_fraction``
    # (GOAL §4) reflect only post-resume steps. Mirrors the LAWA / dynfreeze /
    # best_full_eval restores; guarded so activation-regime-disabled runs and
    # pre-fix checkpoints (None) are untouched.
    if (
        act_regime_tracker is not None
        and restored_training_state is not None
        and restored_training_state.act_regime_state is not None
    ):
        act_regime_tracker.load_state_dict(restored_training_state.act_regime_state)
        logger.info(
            "Activation regime restored: cosines=%d stable_fraction=%.3f",
            len(act_regime_tracker._all_cosines),
            act_regime_tracker.stable_fraction,
        )

    # LAWA weight averaging baseline (GOAL §3.3)
    lawa_averager = None
    if bool(tg_cfg.get("enable_lawa", False)):
        lawa_averager = LAWAAverager(
            window_size=int(tg_cfg.get("lawa_window_size", 5)),
            start_cycle=int(tg_cfg.get("lawa_start_cycle", 10)),
        )
        logger.info(
            "LAWA enabled: window=%d, start_cycle=%d",
            lawa_averager.window_size, lawa_averager.start_cycle,
        )

    # Restore the LAWA snapshot window on resume. Placed here (after the
    # averager is constructed) rather than in the resume block above because
    # ``lawa_averager`` is built below that block — restoring there would hit
    # UnboundLocalError. ``restored_training_state`` is the None-when-not-
    # resuming handle (the resume block sets it to the loaded TrainingState).
    # Without this restore a checkpoint taken after the window started recording
    # loads an empty averager: ``is_ready`` is False and the LAWA comparison +
    # LAWA-averaged JSON eval are silently skipped until ``start_cycle`` worth of
    # new snapshots re-accumulate. Mirrors the dynfreeze/best_full_eval/warmup
    # restores; guarded so LAWA-disabled runs and pre-fix checkpoints (None) are
    # untouched.
    if (
        lawa_averager is not None
        and restored_training_state is not None
        and restored_training_state.lawa_state is not None
    ):
        lawa_averager.load_state_dict(restored_training_state.lawa_state)
        logger.info(
            "LAWA window restored: snapshots=%d recorded_count=%d",
            lawa_averager.count,
            lawa_averager._recorded_count,
        )

    warmup_release_cos = float(tg_cfg.get("warmup_release_cos", 0.75))
    warmup_release_count = int(tg_cfg.get("warmup_release_count", 0))

    alpha_line_cfg = cfg.get("alpha_line", {})
    alpha_line_enabled = bool(alpha_line_cfg.get("alpha_line_enabled", False))
    alpha_line_order = int(alpha_line_cfg.get("alpha_line_order", 0))
    alpha_line_m_steps = int(alpha_line_cfg.get("m_alpha_steps", 19))
    alpha_line_alpha_init = float(alpha_line_cfg.get("alpha_init", 0.0))
    alpha_line_alpha_lr = float(alpha_line_cfg.get("alpha_lr", 1e-2))
    alpha_line_v_update_every = int(alpha_line_cfg.get("v_update_every", 1))
    alpha_line_max_consecutive_reject = int(
        alpha_line_cfg.get("alpha_line_max_consecutive_reject", 3)
    )
    alpha_line_finite_diff_eps = float(
        alpha_line_cfg.get("alpha_line_finite_diff_eps", 1e-3)
    )
    alpha_line_future_work_metrics_enabled = bool(
        alpha_line_cfg.get("future_work_metrics_enabled", False)
    )
    alpha_line_future_work_internal_metrics_enabled = bool(
        alpha_line_cfg.get("future_work_internal_metrics_enabled", False)
    )
    activation_cache_build_count = 0
    activation_cache_eligible_count = 0
    activation_cache_hit_count = 0
    activation_cache_miss_count = 0
    pilot_validation_forward_count = 0
    post_validation_forward_count = 0
    post_extrapolation_eval_count = 0
    post_extrapolation_eval_skipped_count = 0
    post_extrapolation_eval_skip_reasons: dict[str, int] = {}
    subspace_zo_attempted_steps_total = 0
    subspace_zo_accepted_steps_total = 0
    subspace_zo_rejected_steps_total = 0
    subspace_zo_forward_count_total = 0
    subspace_zo_dim1_steps_total = 0
    subspace_zo_dim2_steps_total = 0
    alpha_line_steps_total = 0
    alpha_line_base_recompute_total = 0
    alpha_line_v_update_wall_seconds_total = 0.0
    alpha_line_alpha_wall_seconds_total = 0.0
    future_work_projection_ratios: list[float] = []
    future_work_internal_pair_count = 0

    # Restore the run-wide efficiency-accounting counters (GOAL §5 / P3) on
    # resume. These locals accumulate across the whole run and feed the run-end
    # summary; without this restore a fault/periodic resume rebuilds them at
    # zero/empty and the cost report (validation_forwards_total, cache hit-rate,
    # subspace-ZO / alpha-line tallies, future-work projection mean) reflects
    # only post-resume cycles — a silent resume-state-loss sibling to the fixed
    # LAWA (``lawa_state``) / act-regime (``act_regime_state``) / dynfreeze gaps.
    # Each counter falls back to its zero/empty init when the restored blob
    # omits it (pre-fix checkpoint / a counter that didn't exist yet), so a
    # partial restore never fabricates data. Mirrors the act_regime / lawa
    # restores; guarded so fresh runs and pre-fix checkpoints (None) are
    # untouched.
    if (
        restored_training_state is not None
        and restored_training_state.efficiency_accounting
    ):
        _eff = restored_training_state.efficiency_accounting
        activation_cache_build_count = _eff.get(
            "activation_cache_build_count", activation_cache_build_count
        )
        activation_cache_eligible_count = _eff.get(
            "activation_cache_eligible_count", activation_cache_eligible_count
        )
        activation_cache_hit_count = _eff.get(
            "activation_cache_hit_count", activation_cache_hit_count
        )
        activation_cache_miss_count = _eff.get(
            "activation_cache_miss_count", activation_cache_miss_count
        )
        pilot_validation_forward_count = _eff.get(
            "pilot_validation_forward_count", pilot_validation_forward_count
        )
        post_validation_forward_count = _eff.get(
            "post_validation_forward_count", post_validation_forward_count
        )
        post_extrapolation_eval_count = _eff.get(
            "post_extrapolation_eval_count", post_extrapolation_eval_count
        )
        post_extrapolation_eval_skipped_count = _eff.get(
            "post_extrapolation_eval_skipped_count",
            post_extrapolation_eval_skipped_count,
        )
        post_extrapolation_eval_skip_reasons = _eff.get(
            "post_extrapolation_eval_skip_reasons",
            post_extrapolation_eval_skip_reasons,
        )
        subspace_zo_attempted_steps_total = _eff.get(
            "subspace_zo_attempted_steps_total", subspace_zo_attempted_steps_total
        )
        subspace_zo_accepted_steps_total = _eff.get(
            "subspace_zo_accepted_steps_total", subspace_zo_accepted_steps_total
        )
        subspace_zo_rejected_steps_total = _eff.get(
            "subspace_zo_rejected_steps_total", subspace_zo_rejected_steps_total
        )
        subspace_zo_forward_count_total = _eff.get(
            "subspace_zo_forward_count_total", subspace_zo_forward_count_total
        )
        subspace_zo_dim1_steps_total = _eff.get(
            "subspace_zo_dim1_steps_total", subspace_zo_dim1_steps_total
        )
        subspace_zo_dim2_steps_total = _eff.get(
            "subspace_zo_dim2_steps_total", subspace_zo_dim2_steps_total
        )
        alpha_line_steps_total = _eff.get(
            "alpha_line_steps_total", alpha_line_steps_total
        )
        alpha_line_base_recompute_total = _eff.get(
            "alpha_line_base_recompute_total", alpha_line_base_recompute_total
        )
        alpha_line_v_update_wall_seconds_total = _eff.get(
            "alpha_line_v_update_wall_seconds_total",
            alpha_line_v_update_wall_seconds_total,
        )
        alpha_line_alpha_wall_seconds_total = _eff.get(
            "alpha_line_alpha_wall_seconds_total",
            alpha_line_alpha_wall_seconds_total,
        )
        future_work_projection_ratios = _eff.get(
            "future_work_projection_ratios", future_work_projection_ratios
        )
        future_work_internal_pair_count = _eff.get(
            "future_work_internal_pair_count", future_work_internal_pair_count
        )
        logger.info(
            "Efficiency-accounting counters restored: %d keys "
            "(validation_forwards_total=%d, post_extrap_eval=%d)",
            len(_eff),
            pilot_validation_forward_count + post_validation_forward_count,
            post_extrapolation_eval_count,
        )

    metrics.write_header(
        cfg,
        budget_type="cycles",
        budget_value=cfg.training.max_cycles,
        param_counts=count_parameters(model),
        comparison_keys={
            "deterministic_data_order": deterministic_data_order,
            "dataset_key": batch_plan_manifest.dataset_key,
            "epoch_batch_plan_key": batch_plan_manifest.epoch_batch_plan_key,
            "trajectory_key": trajectory_key,
            "batch_plan_strategy": batch_plan_manifest.strategy,
            "batch_plan_manifest": (
                str(batch_plan_manifest_path) if save_batch_plan_manifest else None
            ),
        },
        comparison_reference={
            "kind": "valid_quick_pretrain",
            "loss": initial_loss,
            "max_examples": cfg.eval.quick_eval_examples,
        },
    )

    # Free GPU cache after initial eval to maximize memory for training
    from src.utils.device import detect_device as _dd
    from src.utils.device import gpu_empty_cache

    gpu_empty_cache(_dd())

    if mlf.enabled:
        try:
            mlf.log_metrics({"initial_quick_valid_loss": initial_loss}, step=0)
        except Exception:
            logger.warning("MLflow initial metric logging failed, continuing")

    logger.info(f"Starting TG-LoRA training: {cfg.training.max_cycles} cycles")
    pbar = tqdm(range(cfg.training.max_cycles), desc="TG-LoRA")
    snapshot_taken = False
    fault_reason: str | None = None
    last_quick_valid_loss = float("inf")
    train_dataset_for_skip = train_loader.dataset
    if not isinstance(train_dataset_for_skip, Sized):
        raise TypeError("train_loader.dataset must be sized")
    max_empty_supervision_skips = len(train_dataset_for_skip)
    try:
        for cycle in pbar:
            if cycle < cycle_offset:
                pbar.update(0)
                continue
            # --- async cache swap ---
            if async_builder is not None and not async_ready:
                if async_builder.poll():
                    for _label in ("valid_quick", "valid_full"):
                        _result = async_builder.get_result(_label)
                        if _result and _result.dataset is not None:
                            _new_loader = _build_loader(
                                _result.dataset,
                                batch_size=cfg.training.batch_size,
                                shuffle=False,
                                cached_loader_kwargs=cache_loader_kwargs,
                            )
                            if _label == "valid_quick":
                                valid_quick_loader = _new_loader
                                swap_cycle_vq = cycle
                            else:
                                valid_full_loader = _new_loader
                                swap_cycle_vf = cycle
                            logger.info(
                                "Swapped %s to cached dataset at cycle %d",
                                _label,
                                cycle,
                            )
                    async_builder.join(timeout=30)
                    async_builder = None
                    async_ready = True
                    _maybe_apply_prefix_runtime_offload()
                elif async_builder.failed:
                    logger.warning(
                        "Async cache build failed: %s. Continuing with raw datasets.",
                        async_builder.error,
                    )
                    async_builder.join(timeout=10)
                    async_builder = None
                    async_ready = True

            # 1. Proposal for this cycle
            proposal = controller.propose()
            pilot_K = int(proposal.K)
            pilot_lr = float(proposal.lr)
            pilot_beta = float(proposal.beta)
            grad_accum = cfg.training.grad_accumulation
            pilot_backward_pass_count = pilot_K * grad_accum
            validation_forwards_this_cycle = 0
            pilot_validation_forwards_this_cycle = 0
            post_validation_forwards_this_cycle = 0
            post_extrapolation_eval = False
            _shadow_delta = None
            _shadow_cos = None
            post_extrapolation_eval_skipped = False
            post_extrapolation_eval_skip_reason = "not_reached"

            # 2. Pilot前snapshot
            W0 = snapshot_lora(model)

            # --- Progressive Freeze Gate (Phase 1) ---
            # After W0 snapshot (includes target layer) and before pilot steps.
            # On trigger: capture xin, freeze layer. Subsequent cycles skip this block.
            if progressive_freeze is not None and progressive_freeze.should_freeze(cycle):
                model.train()
                _, pf_xin_shape = progressive_freeze.cache_xin(
                    model, valid_quick_loader, input_device,
                )
                pf_result = progressive_freeze.apply_freeze(model)
                logger.info(
                    "Progressive freeze: cycle=%d layer=%d params=%d xin_shape=%s",
                    cycle,
                    pf_result.frozen_layer_idx,
                    pf_result.num_frozen_params,
                    pf_xin_shape,
                )

            # --- Dynamic Reversible Freeze (M10) ---
            dynfreeze_all_frozen = False
            if dynfreeze is not None:
                # One composed per-cycle step (compute → decide/apply unfreeze →
                # decide/apply freeze). The order inside run_cycle is load-bearing:
                # apply_unfreeze must precede decide_freeze so a §4-released layer's
                # cooldown (not its frozen-period 0.0 r_A history) holds it out of
                # the freeze decision — see DynamicFreezeController.run_cycle.
                dynfreeze_all_frozen = dynfreeze.run_cycle(model, cycle)

            # 2b. Measurement: noise SNR (multiple independent batch gradients at W0)
            _measurement_noise = cfg.training.get("measurement_noise_samples", 0)
            _measurement_results = {}
            _measurement_results_record = {}
            if progressive_freeze is not None:
                _measurement_results_record["pf_active"] = progressive_freeze.is_frozen
                if progressive_freeze.frozen_layer_idx is not None:
                    _measurement_results_record["pf_layer"] = (
                        progressive_freeze.frozen_layer_idx
                    )
            if _measurement_noise > 0:
                _noise_grads = []
                _noise_losses = []
                for _ni in range(_measurement_noise):
                    model.zero_grad()
                    _noise_batch = batch_iter.next()
                    train_batch_position += 1
                    _noise_loss = forward_backward(model, _noise_batch, 1.0)
                    _grad = {n: p.grad.detach().cpu().clone().float()
                             for n, p in iter_lora_params(model) if p.grad is not None}
                    _noise_grads.append(_grad)
                    _noise_losses.append(_noise_loss)
                # Compute pairwise cosines and SNR
                _ng_norms = []
                for _ng in _noise_grads:
                    _sq = sum(torch.dot(v.flatten(), v.flatten()).item() for v in _ng.values())
                    _ng_norms.append(math.sqrt(max(0, _sq)))
                _pair_cos = []
                for _i in range(len(_noise_grads)):
                    for _j in range(_i + 1, len(_noise_grads)):
                        _d = sum(torch.dot(_noise_grads[_i][k].flatten(),
                                           _noise_grads[_j][k].flatten()).item()
                                 for k in _noise_grads[_i].keys() & _noise_grads[_j].keys())
                        _n1, _n2 = _ng_norms[_i], _ng_norms[_j]
                        _pc = _d / (_n1 * _n2) if _n1 > 1e-12 and _n2 > 1e-12 else 0.0
                        _pair_cos.append(_pc)
                # Mean gradient and variance
                _mean_grad = {}
                for _k in _noise_grads[0]:
                    _mean_grad[_k] = torch.stack([ng[_k] for ng in _noise_grads]).mean(dim=0)
                _mean_sq = sum(torch.dot(v.flatten(), v.flatten()).item() for v in _mean_grad.values())
                _var_sq = sum(
                    sum(torch.dot((ng[k] - _mean_grad[k]).flatten(),
                                  (ng[k] - _mean_grad[k]).flatten()).item()
                        for k in _mean_grad)
                    for ng in _noise_grads
                ) / len(_noise_grads)
                _snr = _mean_sq / _var_sq if _var_sq > 1e-12 else 0.0
                _measurement_results = {
                    "noise_snr": _snr,
                    "noise_mean_grad_norm": math.sqrt(max(0, _mean_sq)),
                    "noise_grad_var_norm": math.sqrt(max(0, _var_sq)),
                    "noise_individual_norms": _ng_norms,
                    "noise_pair_cos_mean": float(np.mean(_pair_cos)) if _pair_cos else 0.0,
                    "noise_losses": _noise_losses,
                }
                _measurement_results_record = {
                    "measurement_noise_snr": _snr,
                    "measurement_noise_mean_grad_norm": math.sqrt(max(0, _mean_sq)),
                    "measurement_noise_grad_var_norm": math.sqrt(max(0, _var_sq)),
                    "measurement_noise_pair_cos": float(np.mean(_pair_cos)) if _pair_cos else 0.0,
                }
                logger.info("MEASUREMENT cycle=%d SNR=%.4f pair_cos=%.4f mean_norm=%.1f",
                            cycle, _snr,
                            float(np.mean(_pair_cos)) if _pair_cos else 0.0,
                            math.sqrt(max(0, _mean_sq)))
                model.zero_grad()
                del _noise_grads, _mean_grad
                torch.cuda.empty_cache()

            # 3. K stepの通常QLoRA（evalなし、中間snapshot保存）
            # 既定ではcycleごとにoptimizerを再作成する。
            # experimental policyではstate tensorをin-placeでzero resetし、
            # fresh optimizerに近い挙動を保ったまま再確保を避ける。
            if dynfreeze_all_frozen:
                # All layers frozen by dynfreeze — skip expensive training steps
                logger.info("DynFreeze: cycle=%d all layers frozen, skipping pilot steps", cycle)
                optimizer = None
            else:
                optimizer = optimizer_lifecycle.prepare_for_cycle(
                    lr=pilot_lr,
                )

            step_losses: list[float] = []
            intermediate_deltas: list[dict[str, torch.Tensor]] = []
            _psa_stats: dict[str, float] = {}
            cycle_batch_keys: list[str] = []
            cycle_sample_keys: list[str] = []
            pilot_grad_sum_cpu: dict[str, torch.Tensor] | None = None
            if optimizer is not None:
                for _ in range(pilot_K):
                    step_loss = 0.0
                    valid_micro_batches = 0
                    empty_supervision_skips = 0
                    while valid_micro_batches < grad_accum:
                        batch = batch_iter.next()
                        batch_position = train_batch_position
                        train_batch_position += 1
                        if not has_supervised_tokens(batch):
                            empty_supervision_skips += 1
                            if empty_supervision_skips > max_empty_supervision_skips:
                                raise RuntimeError(
                                    "Exceeded empty-supervision skip budget while filling a training step"
                                )
                            continue
                        if deterministic_data_order:
                            batch_locator = batch_plan_manifest.batch_locator_at_position(
                                batch_position
                            )
                            cycle_batch_keys.append(batch_locator.batch_key)
                            cycle_sample_keys.extend(batch_locator.sample_keys)
                        micro_loss = forward_backward(model, batch, grad_accum)
                        step_loss += micro_loss
                        valid_micro_batches += 1
                    if alpha_line_future_work_metrics_enabled:
                        pilot_grad_sum_cpu = _accumulate_lora_grads_cpu(
                            model,
                            pilot_grad_sum_cpu,
                        )
                    if psa_prior is not None:
                        _psa_stats = amplify_gradients_psa(
                            model, psa_prior, psa_gain_map, enabled=True
                        )
                    optimizer_step(
                        optimizer,
                        scheduler=None,
                        model=model,
                        max_grad_norm=cfg.training.max_grad_norm,
                    )
                    step_losses.append(step_loss / grad_accum)
                    new_snapshot = snapshot_lora_delta(model, W0)
                    intermediate_deltas.append(new_snapshot)
                    # Record incremental delta (step-to-step change) into PSA buffer
                    if psa_prior is not None and len(intermediate_deltas) >= 2:
                        prev = intermediate_deltas[-2]
                        incremental = {k: new_snapshot[k] - prev[k] for k in new_snapshot}
                        psa_prior.record_delta(incremental)

                    # Activation-fingerprint regime step (after optimizer step + forward hook captured)
                    if act_regime_tracker is not None:
                        act_regime_tracker.step()

            # PSA: regime detection + prior management (once per cycle)
            if regime_detector is not None and step_losses:
                cycle_loss = sum(step_losses) / len(step_losses)
                regime_detector.update(cycle_loss)
                if (
                    psa_regime_reset_enabled
                    and regime_detector.consume_reset_signal()
                    and psa_prior is not None
                ):
                    logger.info(
                        "PSA priors reset at cycle %d due to regime transition",
                        cycle,
                    )
                    psa_prior.reset_priors()
                    psa_gain_map = {}

            # PSA: extract priors from cross-cycle history (gated by update_interval)
            if psa_prior is not None and psa_prior.should_update(cycle):
                psa_prior.extract_priors()
                _regime_value = (
                    regime_detector.regime.value if regime_detector is not None else "stable"
                )
                psa_gain_map = psa_prior.compute_gain_map(model, regime=_regime_value)
                psa_prior.mark_updated(cycle)
                if cycle % 5 == 0:
                    logger.info(
                        "PSA priors updated at cycle %d, %d tensors, gain_mean=%.3f",
                        cycle, len(psa_prior.priors),
                        sum(psa_gain_map.values()) / max(1, len(psa_gain_map)),
                    )

            # Measurement: per-step delta norms
            _save_per_step = cfg.training.get("measurement_save_per_step_deltas", False)
            if _save_per_step:
                _per_step_norms = []
                for _si, _sd in enumerate(intermediate_deltas):
                    _sn = math.sqrt(max(0, sum(
                        torch.dot(v.flatten(), v.flatten()).item() for v in _sd.values()
                    )))
                    _per_step_norms.append(_sn)
                _measurement_results["per_step_delta_norms"] = _per_step_norms
                _measurement_results["per_step_losses"] = step_losses
                _measurement_results_record["measurement_per_step_delta_norms"] = _per_step_norms
                _measurement_results_record["measurement_per_step_losses"] = step_losses

            # PSA metrics for logging
            if psa_prior is not None:
                _measurement_results_record["psa_enabled"] = True
                _measurement_results_record["psa_prior_count"] = len(psa_prior.priors)
                _measurement_results_record["psa_gain_mean"] = (
                    sum(psa_gain_map.values()) / max(1, len(psa_gain_map))
                    if psa_gain_map else 0.0
                )
                if regime_detector is not None:
                    _measurement_results_record["psa_regime"] = regime_detector.regime.value
                    _measurement_results_record["psa_regime_transitions"] = regime_detector.transition_count
                # Per-layer-type diagnostics (GOAL §4 step 2)
                if _psa_stats:
                    _lt_diag = summarize_by_layer_type(
                        _psa_stats, psa_prior._prior_cosines
                    )
                    for _lt_name, _lt_vals in _lt_diag.items():
                        prefix = f"psa_lt_{_lt_name}"
                        _measurement_results_record[f"{prefix}_count"] = _lt_vals["count"]
                        _measurement_results_record[f"{prefix}_amp_mean"] = _lt_vals["amp_mean"]
                        _measurement_results_record[f"{prefix}_amp_std"] = _lt_vals["amp_std"]
                        if "prior_stability_mean" in _lt_vals:
                            _measurement_results_record[f"{prefix}_prior_stability"] = _lt_vals["prior_stability_mean"]
            else:
                _measurement_results_record["psa_enabled"] = False
                _measurement_results_record["psa_prior_count"] = 0
                _measurement_results_record["psa_gain_mean"] = 0.0

            # Activation regime inventory metrics
            if act_regime_tracker is not None:
                _act_summary = act_regime_tracker.summary()
                _measurement_results_record["act_regime"] = _act_summary["regime"]
                _measurement_results_record["act_stable_fraction"] = _act_summary["stable_fraction"]
                _measurement_results_record["act_cosine_latest"] = _act_summary["cosine_latest"]
                _measurement_results_record["act_cosine_mean"] = _act_summary["cosine_mean"]

            # --- Dynfreeze skip path: all layers frozen, skip expensive training ---
            if dynfreeze_all_frozen:
                # No training happened — use previous cycle's loss
                pilot_loss_avg = cycle_state.last_valid_loss if cycle_state.last_valid_loss < float("inf") else 0.0
                final_valid_loss = pilot_loss_avg
                accepted = True
                cos_sim = 0.0
                raw_delta_cos_sim = 0.0
                pilot_backward_pass_count = 0
                loss_after = None
                loss_pilot = pilot_loss_avg
                post_extrapolation_eval = False
                post_extrapolation_eval_skipped = True
                post_extrapolation_eval_skip_reason = "dynfreeze_frozen"
                WK = dict(W0)  # No change
                extrap_stats = ExtrapolationStats()
                proposal_N_before_cosine = 0
                predicted_consistency = 0.0
                velocity_norm_ratio = 0.0
                zo_attempted_steps = 0
                zo_accepted_steps = 0
                zo_rejected_steps = 0
                zo_forward_count = 0
                zo_dim1_steps = 0
                zo_dim2_steps = 0
                zo_last_stats = ZerothOrderStepStats()
                alpha_line_v_updated = False
                alpha_line_alpha_before = None
                alpha_line_alpha_after = None
                alpha_line_step_count = 0
                alpha_line_base_recompute = False
                alpha_line_base_out_jvp_cached = False
                alpha_line_base_term_cached = False
                alpha_line_stop_reason = None
                alpha_line_v_update_wall_seconds = 0.0
                alpha_line_alpha_wall_seconds = 0.0
                alpha_line_grad_values = None
                alpha_line_losses = None
                alpha_line_exact_losses = None
                alpha_line_approx_errors = None
                alpha_line_jvp_methods = None
                alpha_line_order = 0
                alpha_line_finite_diff_eps = None
                alpha_line_consecutive_rejects = 0
                alpha_line_interval_net_loss_delta = None
                _shadow_delta = None
                _shadow_cos = None
                _measurement_results_record = {}
                future_work_record = None
                pilot_validation_forwards_this_cycle = 0
                post_validation_forwards_this_cycle = 0
                # Skip directly to metrics recording
            else:
                pilot_loss_avg, _pilot_metrics = _compute_pilot_average(
                    step_losses, pilot_K
                )

            # 3. Pilot後snapshot — intermediate_deltas[-1]からWK復元（冗長コピー回避）
            if dynfreeze_all_frozen:
                WK = dict(W0)
            else:
                # 凍結層はpilotで更新されずintermediate_deltasに含まれない → W0のまま
                last_delta = intermediate_deltas[-1]
                WK = {
                    k: (W0[k] + last_delta[k]) if k in last_delta else W0[k]
                    for k in W0
                }

            if not dynfreeze_all_frozen:
                # 3b. Quick eval (pilot) — キャッシュ付きeval
                # 次の外挿で変更されるレイヤーを予測し、変更されないレイヤーの
                # hidden statesをキャッシュする。外挿後evalではキャッシュから再開。
                if use_prefix_feature_cache:
                    predicted_strategy = "prefix_feature_cache_suffix_only"
                    predicted_active_names = fixed_active_names
                    predicted_active_indices = fixed_active_indices
                    split_layer = (
                        prefix_cache_split_layer
                        if prefix_cache_split_layer is not None
                        else 0
                    )
                    use_cache = False
                    torch.cuda.empty_cache()  # Free memory before eval to prevent OOM
                    loss_pilot = eval_loss(
                        model,
                        valid_quick_loader,
                        input_device,
                        max_examples=accept_eval_examples,
                    )
                    validation_forwards_this_cycle += 1
                    pilot_validation_forwards_this_cycle += 1
                else:
                    predicted_strategy = (
                        "last_25_percent"
                        if tg_cfg.get("force_top_layers_only", False)
                        else proposal.active_layer_strategy
                    )
                    predicted_active_names, predicted_active_indices = select_active_layers(
                        model,
                        strategy=predicted_strategy,
                        random_middle=cfg.tg_lora.random_middle_layers,
                        layer_scores=controller.state.layer_scores,
                        temperature=cfg.tg_lora.layer_sample_temperature,
                    )
                    split_layer = determine_split_layer(
                        predicted_active_indices, num_decoder_layers
                    )

                    # split_layerが十分大きい場合のみキャッシュ（2層未満のスキップは意味なし）
                    use_cache = split_layer >= 2
                    if use_cache:
                        activation_cache_build_count += 1
                        loss_pilot = activation_cache.eval_and_cache(
                            model,
                            valid_quick_loader,
                            input_device,
                            split_layer_idx=split_layer,
                            max_examples=accept_eval_examples,
                        )
                        validation_forwards_this_cycle += 1
                        pilot_validation_forwards_this_cycle += 1
                    else:
                        activation_cache.clear()
                        loss_pilot = eval_loss(
                            model,
                            valid_quick_loader,
                            input_device,
                            max_examples=accept_eval_examples,
                        )
                        validation_forwards_this_cycle += 1
                        pilot_validation_forwards_this_cycle += 1

                # 3c. 中間点ロールバック判定
                # loss_pilotがW0より悪化していたら、最良の中間点を探して戻す
                # W0のlossは前cycleのloss_pilotを使う（初回はskip）
                pilot_rollback_n = None
                dW = None
                dW_steps = pilot_K
                pilot_full_rollback = False
                if loss_pilot > cycle_state.last_valid_loss + controller.rollback_tolerance:
                    # 中間点の中から最良を探す
                    best_intermediate_loss = loss_pilot
                    for i in range(len(intermediate_deltas) - 2, -1, -1):
                        apply_delta_snapshot(model, W0, intermediate_deltas[i])
                        inter_loss = eval_loss(
                            model,
                            valid_quick_loader,
                            input_device,
                            max_examples=accept_eval_examples,
                        )
                        validation_forwards_this_cycle += 1
                        pilot_validation_forwards_this_cycle += 1
                        if inter_loss < best_intermediate_loss:
                            best_intermediate_loss = inter_loss
                            pilot_rollback_n = i
                        if inter_loss <= cycle_state.last_valid_loss:
                            break

                    if pilot_rollback_n is not None:
                        apply_delta_snapshot(
                            model, W0, intermediate_deltas[pilot_rollback_n]
                        )
                        logger.info(
                            f"Pilot rollback: K={pilot_K} → N={pilot_rollback_n + 1}, "
                            f"loss {loss_pilot:.4f} → {best_intermediate_loss:.4f}"
                        )
                        loss_pilot = best_intermediate_loss
                        # dWとvelocityはロールバック先の状態で再計算
                        rb_delta = intermediate_deltas[pilot_rollback_n]
                        W_rollback = {
                            k: (W0[k] + rb_delta[k]) if k in rb_delta else W0[k]
                            for k in W0
                        }
                        dW = delta_tracker.compute_and_record(
                            W_rollback, W0, K=pilot_rollback_n + 1
                        )
                        dW_steps = pilot_rollback_n + 1
                        pilot_loss_avg = best_intermediate_loss
                    else:
                        # 全中間点よりW0の方が良い → W0に戻す
                        load_lora_snapshot(model, W0)
                        logger.info(f"Pilot full rollback to W0, loss {loss_pilot:.4f}")
                        pilot_full_rollback = True
                        controller.penalize(cycle_state.last_valid_loss, loss_pilot)
                        # loss_pilotをW0の実際のlossに更新（last_valid_lossと同じ）
                        loss_pilot = cycle_state.last_valid_loss
                else:
                    # 全Kステップ有効
                    dW = delta_tracker.compute_and_record(WK, W0, K=pilot_K)
                    dW_steps = pilot_K

                pilot_state_snapshot = (
                    snapshot_lora(model) if save_trajectory_delta_artifacts else None
                )
                pilot_backward_passes = (
                    cycle_state.full_backward_passes + pilot_backward_pass_count
                )
                if (
                    save_trajectory_delta_artifacts
                    and dW is not None
                    and cycle % trajectory_delta_artifact_interval == 0
                ):
                    pilot_metadata = build_trajectory_delta_artifact_metadata(
                        mode="tg_lora",
                        anchor_kind="after_pilot",
                        trajectory_key=trajectory_key,
                        epoch_batch_plan_key=batch_plan_manifest.epoch_batch_plan_key,
                        batch_plan_manifest=(
                            str(batch_plan_manifest_path)
                            if save_batch_plan_manifest
                            else None
                        ),
                        dataset_key=batch_plan_manifest.dataset_key,
                        delta_tensors=dW,
                        cycle=cycle,
                        total_backward_passes=pilot_backward_passes,
                        batch_keys=cycle_batch_keys,
                        sample_keys=cycle_sample_keys,
                        extra_metadata={
                            "K": pilot_K,
                            "pilot_lr": pilot_lr,
                            "pilot_loss_avg": pilot_loss_avg,
                            "pilot_valid_loss": loss_pilot,
                        },
                    )
                    save_trajectory_delta_artifact(
                        path=trajectory_delta_artifact_dir
                        / artifact_file_name(
                            mode="tg_lora",
                            anchor_kind="after_pilot",
                            cycle=cycle,
                        ),
                        metadata=pilot_metadata,
                        delta_tensors=dW,
                    )

                # Pilot完全失敗: velocity/extrapolationをスキップして次cycleへ
                if pilot_full_rollback:
                    accepted = False
                    cos_sim = 0.0
                    is_full_eval_cycle = should_run_full_eval(
                        cycle, cfg.eval.get("full_eval_every_cycles", 10)
                    )
                    cycle_state.record_cycle(
                        train_loss=pilot_loss_avg,
                        valid_loss=None if is_full_eval_cycle else loss_pilot,
                        accepted=None,
                        actual_backward_passes=pilot_backward_pass_count,
                        speculative_optimizer_steps=0,
                        optimizer_steps=pilot_K,
                        speculative_equivalent_backward_passes=0,
                    )
                    pilot_validation_forward_count += pilot_validation_forwards_this_cycle
                    post_validation_forward_count += post_validation_forwards_this_cycle
                    metrics.record_step(
                        step=cycle_state.full_backward_passes,
                        cycle=cycle,
                        loss_train=pilot_loss_avg,
                        loss_valid=loss_pilot,
                        backward_passes=pilot_backward_pass_count,
                        total_backward_passes=cycle_state.full_backward_passes,
                        tg_lora_accepted=False,
                        tg_lora_cosine_sim=0.0,
                        tg_lora_raw_delta_cosine_sim=0.0,
                        tg_lora_predicted_consistency=0.0,
                        tg_lora_short_long_norm_ratio=1.0,
                        tg_lora_reduction_rate=cycle_state.reduction_rate,
                        tg_lora_K=pilot_K,
                        tg_lora_N=0,
                        tg_lora_proposed_N=proposal.N,
                        tg_lora_alpha=None,
                        tg_lora_beta=pilot_beta,
                        tg_lora_lr=pilot_lr,
                        tg_lora_cache_built=use_cache,
                        tg_lora_cache_eligible=None,
                        tg_lora_cache_hit=None,
                        tg_lora_validation_forwards=validation_forwards_this_cycle,
                        tg_lora_pilot_validation_forwards=pilot_validation_forwards_this_cycle,
                        tg_lora_post_validation_forwards=post_validation_forwards_this_cycle,
                        tg_lora_post_extrapolation_eval=False,
                        tg_lora_post_extrapolation_eval_skipped=False,
                        tg_lora_post_extrapolation_eval_skip_reason="pilot_full_rollback",
                        tg_lora_rollback_triggered=False,
                        **_measurement_results_record,
                    )
                    if mlf.enabled:
                        mlf.log_metrics(
                            {
                                "loss_train": pilot_loss_avg,
                                "loss_valid": loss_pilot,
                                "accepted": 0,
                            },
                            step=cycle,
                        )
                    pbar.set_postfix(
                        loss=f"{loss_pilot:.4f}",
                        acc="N",
                        cos="0.000",
                        red=f"{cycle_state.reduction_rate:.1%}",
                    )
                    logger.info(
                        f"c={cycle} loss={loss_pilot:.4f} N (pilot full rollback) K={pilot_K}"
                    )
                    if is_full_eval_cycle:
                        full_result = eval_loss_detailed(
                            model, valid_full_loader, input_device
                        )
                        full_loss = full_result.avg_loss
                        logger.info(
                            f"Cycle {cycle} full eval: loss={full_loss:.4f} "
                            f"ppl={full_result.perplexity:.2f}"
                        )
                        cycle_state.record_full_eval(full_loss)
                        metrics.record_full_eval_loss(
                            cycle=cycle,
                            full_loss=full_loss,
                            total_backward_passes=cycle_state.full_backward_passes,
                        )
                        if full_loss < best_full_eval_loss:
                            best_full_eval_loss = full_loss
                            best_full_eval_perplexity = full_result.perplexity
                            save_checkpoint(model, tokenizer, run_dir / "best_model")
                    accepted_valid_history.append(loss_pilot)
                    if len(accepted_valid_history) > max(3, moving_avg_window * 2):
                        accepted_valid_history.pop(0)
                    last_quick_valid_loss = loss_pilot
                    continue

                # 4b. 収束トレンドに基づく能動的lr-K調整
                # [作業B] ウォームアップ期は lr 適応経路を全バイパスし lr を初期値で固定保持する。
                # 本番期（warmup_released=True）からのみ adapt_to_convergence を有効化する。
                if warmup_released:
                    controller.adapt_to_convergence(delta_tracker.convergence_trend())

                # 5. Velocity更新
                assert dW is not None
                raw_delta_cos_sim = velocity.cosine_similarity(dW)
                velocity.update(dW, beta=pilot_beta, lr=pilot_lr, K=dW_steps)
                current_velocity = velocity.state
                assert current_velocity is not None
                predicted_consistency = velocity.predicted_consistency()
                velocity_norm_ratio = velocity.short_long_norm_ratio()
                cos_sim = predicted_consistency

                # 5b. 加速度ベースの能動的lr-K調整
                # [作業B] ウォームアップ期は adapt_to_acceleration もバイパスする。
                acceleration = velocity.magnitude_acceleration()
                if warmup_released:
                    controller.adapt_to_acceleration(acceleration)

                proposal_N_before_cosine = int(proposal.N)
                selected_N = proposal.N
                if tg_cfg.get("cosine_n_selection_enabled", False):
                    selected_N = velocity.choose_N(
                        list(controller.N_candidates),
                        _cosine_n_threshold_map(
                            tg_cfg.get("cosine_n_selection_thresholds", None)
                        ),
                    )
                    if selected_N != proposal.N:
                        logger.debug(
                            "cosine N selection: cycle=%d consistency=%.4f "
                            "norm_ratio=%.4f N=%d->%d raw_delta_cos=%.4f",
                            cycle,
                            predicted_consistency,
                            velocity_norm_ratio,
                            proposal.N,
                            selected_N,
                            raw_delta_cos_sim,
                        )
                    proposal.N = selected_N

                # [N=1 diagnostic] Force N=1 to test if EMA direction
                # improves loss at minimal extrapolation distance.
                if warmup_released and selected_N > 0:
                    selected_N = 1
                    proposal.N = 1

                # Shadow mode: bypass warmup gate, force N=1
                shadow_enabled = tg_cfg.get("shadow_extrapolation_enabled", False)
                if shadow_enabled:
                    if not warmup_released:
                        warmup_released = True
                        production_start_full_backward_passes = cycle_state.full_backward_passes
                        logger.info("Shadow mode: warmup bypassed at cycle %d", cycle)
                    selected_N = 1
                    proposal.N = 1

                # --- Warmup gate (two-phase design) ---
                if not warmup_released:
                    if predicted_consistency >= warmup_release_cos:
                        warmup_cos_consecutive += 1
                    else:
                        warmup_cos_consecutive = 0
                    if warmup_cos_consecutive >= warmup_release_count:
                        warmup_released = True
                        production_start_full_backward_passes = (
                            cycle_state.full_backward_passes
                        )
                        logger.info(
                            "Warmup released at cycle %d: cos=%.4f consecutive=%d",
                            cycle,
                            predicted_consistency,
                            warmup_cos_consecutive,
                        )

                if not warmup_released:
                    # Warmup phase: pilot only, skip all extrapolation.
                    is_full_eval_cycle_warmup = should_run_full_eval(
                        cycle, cfg.eval.get("full_eval_every_cycles", 10)
                    )
                    cycle_state.record_cycle(
                        train_loss=pilot_loss_avg,
                        valid_loss=(
                            None if is_full_eval_cycle_warmup else loss_pilot
                        ),
                        accepted=None,
                        actual_backward_passes=pilot_backward_pass_count,
                        speculative_optimizer_steps=0,
                        optimizer_steps=pilot_K,
                        speculative_equivalent_backward_passes=0,
                    )
                    accepted_valid_history.append(loss_pilot)
                    if len(accepted_valid_history) > max(
                        3, moving_avg_window * 2
                    ):
                        accepted_valid_history.pop(0)
                    last_quick_valid_loss = loss_pilot
                    pbar.set_postfix(
                        loss=f"{loss_pilot:.4f}",
                        acc="W",
                        cos=f"{cos_sim:.3f}",
                        red=f"{cycle_state.reduction_rate:.1%}",
                    )
                    if cycle % cfg.logging.get("log_every_cycles", 1) == 0:
                        logger.info(
                            "c=%d loss=%.4f W cos=%.3f warmup_consec=%d",
                            cycle,
                            loss_pilot,
                            cos_sim,
                            warmup_cos_consecutive,
                        )
                    if is_full_eval_cycle_warmup:
                        full_result = eval_loss_detailed(
                            model, valid_full_loader, input_device
                        )
                        full_loss = full_result.avg_loss
                        logger.info(
                            "Cycle %d full eval (warmup): loss=%.4f ppl=%.2f",
                            cycle,
                            full_loss,
                            full_result.perplexity,
                        )
                        cycle_state.record_full_eval(full_loss)
                        metrics.record_full_eval_loss(
                            cycle=cycle,
                            full_loss=full_loss,
                            total_backward_passes=cycle_state.full_backward_passes,
                        )
                        if full_loss < best_full_eval_loss:
                            best_full_eval_loss = full_loss
                            best_full_eval_perplexity = full_result.perplexity
                            save_checkpoint(
                                model, tokenizer, run_dir / "best_model"
                            )
                    if mlf.enabled:
                        mlf.log_metrics(
                            {
                                "loss_train": pilot_loss_avg,
                                "loss_valid": loss_pilot,
                                "warmup_released": 0,
                                "warmup_cos_consecutive": warmup_cos_consecutive,
                                "predicted_consistency": predicted_consistency,
                            },
                            step=cycle,
                        )
                    # [作業A] ウォームアップ期にも run_metrics.jsonl へ step record を出力する。
                    # w_traj と lr は全サイクルで記録必須。M9 係数は外挿していないため None。
                    metrics.record_step(
                        step=cycle_state.full_backward_passes,
                        cycle=cycle,
                        loss_train=pilot_loss_avg,
                        loss_valid=loss_pilot,
                        backward_passes=pilot_backward_pass_count,
                        total_backward_passes=cycle_state.full_backward_passes,
                        tg_lora_accepted=None,
                        tg_lora_cosine_sim=cos_sim,
                        tg_lora_raw_delta_cosine_sim=raw_delta_cos_sim,
                        tg_lora_predicted_consistency=predicted_consistency,
                        tg_lora_short_long_norm_ratio=velocity_norm_ratio,
                        tg_lora_reduction_rate=cycle_state.reduction_rate,
                        tg_lora_K=pilot_K,
                        tg_lora_N=0,
                        tg_lora_proposed_N=None,
                        tg_lora_alpha=None,
                        tg_lora_beta=pilot_beta,
                        tg_lora_lr=pilot_lr,
                        tg_lora_validation_forwards=validation_forwards_this_cycle,
                        tg_lora_pilot_validation_forwards=pilot_validation_forwards_this_cycle,
                        tg_lora_post_validation_forwards=post_validation_forwards_this_cycle,
                        tg_lora_post_extrapolation_eval=False,
                        tg_lora_post_extrapolation_eval_skipped=False,
                        tg_lora_post_extrapolation_eval_skip_reason="warmup_phase",
                        tg_lora_rollback_triggered=False,
                        # M9 係数: ウォームアップ期は外挿なし → None
                        tg_lora_m9_alpha_fit=None,
                        tg_lora_m9_beta1_fit=None,
                        tg_lora_m9_beta2_fit=None,
                        tg_lora_w_traj=None,
                    **_measurement_results_record,
                    )
                    continue

                baseline_like_fallback, baseline_like_reason = (
                    _should_fallback_to_baseline_like(
                        proposal_N=proposal.N,
                        total_cycles=cycle_state.total_cycles,
                        acceptance_rate=cycle_state.acceptance_rate,
                        pilot_loss=loss_pilot,
                        previous_valid_loss=last_quick_valid_loss,
                        acceleration=acceleration,
                        velocity_anomalous=velocity.is_magnitude_anomalous(),
                        enabled=tg_cfg.get("linearity_guard_enabled", False),
                        warmup_cycles=tg_cfg.get("linearity_guard_warmup_cycles", 5),
                        min_acceptance_rate=tg_cfg.get(
                            "linearity_guard_min_acceptance_rate", 0.0
                        ),
                        pilot_margin=tg_cfg.get("linearity_guard_pilot_margin", 0.01),
                        max_positive_acceleration=tg_cfg.get(
                            "linearity_guard_max_positive_acceleration", 0.02
                        ),
                    )
                )

                if baseline_like_fallback:
                    controller.penalize(last_quick_valid_loss, loss_pilot)
                    is_full_eval_cycle = should_run_full_eval(
                        cycle, cfg.eval.get("full_eval_every_cycles", 10)
                    )
                    cycle_state.record_cycle(
                        train_loss=pilot_loss_avg,
                        valid_loss=None if is_full_eval_cycle else loss_pilot,
                        accepted=None,
                        actual_backward_passes=pilot_backward_pass_count,
                        speculative_optimizer_steps=0,
                        optimizer_steps=pilot_K,
                        speculative_equivalent_backward_passes=0,
                    )
                    last_quick_valid_loss = loss_pilot
                    pilot_validation_forward_count += pilot_validation_forwards_this_cycle
                    post_validation_forward_count += post_validation_forwards_this_cycle
                    metrics.record_step(
                        step=cycle_state.full_backward_passes,
                        cycle=cycle,
                        loss_train=pilot_loss_avg,
                        loss_valid=loss_pilot,
                        backward_passes=pilot_backward_pass_count,
                        total_backward_passes=cycle_state.full_backward_passes,
                        tg_lora_accepted=False,
                        tg_lora_cosine_sim=cos_sim,
                        tg_lora_raw_delta_cosine_sim=raw_delta_cos_sim,
                        tg_lora_predicted_consistency=predicted_consistency,
                        tg_lora_short_long_norm_ratio=velocity_norm_ratio,
                        tg_lora_reduction_rate=cycle_state.reduction_rate,
                        tg_lora_K=pilot_K,
                        tg_lora_N=0,
                        tg_lora_proposed_N=proposal_N_before_cosine,
                        tg_lora_alpha=None,
                        tg_lora_beta=pilot_beta,
                        tg_lora_lr=pilot_lr,
                        tg_lora_cache_built=use_cache,
                        tg_lora_cache_eligible=None,
                        tg_lora_cache_hit=None,
                        tg_lora_validation_forwards=validation_forwards_this_cycle,
                        tg_lora_pilot_validation_forwards=pilot_validation_forwards_this_cycle,
                        tg_lora_post_validation_forwards=post_validation_forwards_this_cycle,
                        tg_lora_post_extrapolation_eval=False,
                        tg_lora_post_extrapolation_eval_skipped=False,
                        tg_lora_post_extrapolation_eval_skip_reason="linearity_guard",
                        tg_lora_rollback_triggered=False,
                    **_measurement_results_record,
                    )
                    if mlf.enabled:
                        mlf.log_metrics(
                            {
                                "loss_train": pilot_loss_avg,
                                "loss_valid": loss_pilot,
                                "accepted": 0,
                                "raw_delta_cosine_sim": raw_delta_cos_sim,
                                "velocity_predicted_consistency": predicted_consistency,
                                "velocity_short_long_norm_ratio": velocity_norm_ratio,
                                "linearity_guard_triggered": 1,
                                "magnitude_acceleration": acceleration,
                                "acceptance_rate": cycle_state.acceptance_rate,
                            },
                            step=cycle,
                        )
                    pbar.set_postfix(
                        loss=f"{loss_pilot:.4f}",
                        acc="B",
                        cos=f"{cos_sim:.3f}",
                        red=f"{cycle_state.reduction_rate:.1%}",
                    )
                    logger.info(
                        "c=%d loss=%.4f B cos=%.3f red=%.1f%% K=%d N=0 (%s)",
                        cycle,
                        loss_pilot,
                        cos_sim,
                        cycle_state.reduction_rate * 100,
                        pilot_K,
                        baseline_like_reason,
                    )
                    if is_full_eval_cycle:
                        full_result = eval_loss_detailed(
                            model, valid_full_loader, input_device
                        )
                        full_loss = full_result.avg_loss
                        logger.info(
                            f"Cycle {cycle} full eval: loss={full_loss:.4f} "
                            f"ppl={full_result.perplexity:.2f}"
                        )
                        cycle_state.record_full_eval(full_loss)
                        metrics.record_full_eval_loss(
                            cycle=cycle,
                            full_loss=full_loss,
                            total_backward_passes=cycle_state.full_backward_passes,
                        )
                        if full_loss < best_full_eval_loss:
                            best_full_eval_loss = full_loss
                            best_full_eval_perplexity = full_result.perplexity
                            save_checkpoint(model, tokenizer, run_dir / "best_model")
                    _check_and_save_linearity_budget_checkpoint(
                        model,
                        tokenizer,
                        valid_full_loader,
                        input_device,
                        cycle_state,
                        grad_accum,
                        triggered_target_steps,
                        run_dir,
                        logger,
                        metrics,
                    )
                    accepted_valid_history.append(loss_pilot)
                    if len(accepted_valid_history) > max(3, moving_avg_window * 2):
                        accepted_valid_history.pop(0)
                    continue

                # 6. Pilot後状態をrollback用に保存
                rollback_mgr.save(model)
                snapshot_taken = True

                try:
                    # 7. Extrapolation — proposal already created at cycle top

                    # 8. Active layers選択 — strategyが予測時と同じなら再利用
                    # （二重ランダム選択を避け、キャッシュ有効性を保証）
                    if use_prefix_feature_cache:
                        active_names = fixed_active_names
                        active_indices = fixed_active_indices
                    else:
                        actual_strategy = (
                            "last_25_percent"
                            if tg_cfg.get("force_top_layers_only", False)
                            else proposal.active_layer_strategy
                        )
                        if actual_strategy == predicted_strategy:
                            active_names = predicted_active_names
                            active_indices = predicted_active_indices
                        else:
                            active_names, active_indices = select_active_layers(
                                model,
                                strategy=actual_strategy,
                                random_middle=cfg.tg_lora.random_middle_layers,
                                layer_scores=controller.state.layer_scores,
                                temperature=cfg.tg_lora.layer_sample_temperature,
                            )

                    # Exclude frozen params from M9 speculative update targets
                    if dynfreeze is not None:
                        frozen_names = dynfreeze.get_frozen_param_names(model)
                        if frozen_names:
                            active_names = active_names - frozen_names

                    zo_attempted_steps = 0
                    zo_accepted_steps = 0
                    zo_rejected_steps = 0
                    zo_forward_count = 0
                    zo_dim1_steps = 0
                    zo_dim2_steps = 0
                    zo_last_stats = ZerothOrderStepStats()
                    alpha_line_v_updated = False
                    alpha_line_alpha_before = None
                    alpha_line_alpha_after = None
                    alpha_line_step_count = 0
                    alpha_line_grad_values: list[float] = []
                    alpha_line_losses: list[float] = []
                    alpha_line_exact_losses: list[float] = []
                    alpha_line_approx_errors: list[float] = []
                    alpha_line_jvp_methods: list[str] = []
                    alpha_line_base_recompute = 0
                    alpha_line_base_term_cached = False
                    alpha_line_base_out_jvp_cached = False
                    alpha_line_stop_reason = "not_reached"
                    alpha_line_consecutive_rejects = 0
                    alpha_line_interval_loss_start = None
                    alpha_line_interval_probe_batch = None
                    alpha_line_interval_net_loss_delta = None
                    alpha_line_v_update_wall_seconds = 0.0
                    alpha_line_alpha_wall_seconds = 0.0
                    future_work_projection_ratio = None
                    future_work_internal_pairs: list[dict[str, float]] = []
                    cache_eligible = False
                    cache_hit = False
                    can_confident_skip = False
                    # [作業A] M9 フィット係数: subspace_m9_fit_step が実行された場合のみ更新される。
                    # 未実行サイクルでは None のまま記録される。
                    m9_cycle_stats: dict = {}


                    # 10. 外挿適用
                    accepted = True
                    loss_after = loss_pilot
                    if alpha_line_enabled:
                        accepted = False
                        loss_after = loss_pilot
                        proposal.N = 0
                        extrap_stats = ExtrapolationStats()

                        should_refresh_direction = (
                            velocity.current_direction() is None
                            or cycle % max(1, alpha_line_v_update_every) == 0
                        )
                        v_update_started = time.perf_counter()
                        if should_refresh_direction:
                            direction = velocity.build_direction(
                                dW,
                                lr=pilot_lr,
                                K=dW_steps,
                                cycle=cycle,
                                active_names=active_names,
                            )
                            alpha_line_v_updated = True
                        else:
                            direction = velocity.current_direction() or {}
                        alpha_line_v_update_wall_seconds = (
                            time.perf_counter() - v_update_started
                        )
                        cycle_state.v_fixed_since_cycle = velocity.fixed_since_cycle

                        if not direction:
                            alpha_line_stop_reason = "no_direction"
                        else:
                            if alpha_line_future_work_metrics_enabled:
                                future_work_projection_ratio = (
                                    _projection_ratio_to_direction(
                                        pilot_grad_sum_cpu,
                                        direction,
                                        active_names=active_names,
                                    )
                                )
                            base_alpha_snapshot = snapshot_lora(model)

                            alpha_current = alpha_line_alpha_init
                            alpha_line_alpha_before = alpha_current
                            _apply_alpha_direction_from_base(
                                model,
                                base_alpha_snapshot,
                                direction,
                                alpha_current,
                                active_names=active_names,
                            )

                            alpha_loop_started = time.perf_counter()
                            for _alpha_index in range(alpha_line_m_steps):
                                alpha_batch, train_batch_position = _next_supervised_batch(
                                    batch_iter=batch_iter,
                                    batch_plan_manifest=batch_plan_manifest,
                                    deterministic_data_order=deterministic_data_order,
                                    train_batch_position=train_batch_position,
                                    max_empty_supervision_skips=max_empty_supervision_skips,
                                    cycle_batch_keys=cycle_batch_keys,
                                    cycle_sample_keys=cycle_sample_keys,
                                )
                                alpha_t = torch.tensor(
                                    alpha_current,
                                    device=input_device,
                                    dtype=torch.float32,
                                    requires_grad=True,
                                )
                                use_first_order = (
                                    alpha_line_order == 1
                                    and "hidden_states" in alpha_batch
                                )
                                first_order_cache = None
                                if use_first_order:
                                    first_order_cache = compute_alpha_line_base_out_jvp(
                                        model,
                                        alpha_batch,
                                        base_alpha_snapshot,
                                        direction,
                                        active_names=active_names,
                                        finite_diff_eps=alpha_line_finite_diff_eps,
                                    )
                                    alpha_line_base_recompute += 1
                                    alpha_line_base_term_cached = True
                                    alpha_line_base_out_jvp_cached = True
                                    alpha_line_jvp_methods.append(
                                        first_order_cache.jvp_method
                                    )
                                    cycle_state.base_term_cached = True
                                    cycle_state.base_out_jvp_cached = True
                                    cycle_state.n_base_recompute += 1
                                    loss_t = alpha_line_loss_cached_first_order(
                                        model,
                                        first_order_cache,
                                        alpha_batch,
                                        alpha_t,
                                    )
                                    with torch.no_grad():
                                        exact_loss_t = alpha_line_loss_exact(
                                            model,
                                            alpha_batch,
                                            base_alpha_snapshot,
                                            direction,
                                            torch.tensor(
                                                alpha_current,
                                                device=input_device,
                                                dtype=torch.float32,
                                            ),
                                            active_names=active_names,
                                        )
                                else:
                                    loss_t = _alpha_line_functional_loss(
                                        model,
                                        alpha_batch,
                                        base_alpha_snapshot,
                                        direction,
                                        alpha_t,
                                        active_names=active_names,
                                    )
                                    exact_loss_t = loss_t.detach()
                                grad_t = torch.autograd.grad(
                                    loss_t,
                                    alpha_t,
                                    retain_graph=False,
                                    create_graph=False,
                                )[0]
                                loss_before = float(loss_t.detach().item())
                                exact_loss_before = float(exact_loss_t.detach().item())
                                grad_alpha = float(grad_t.detach().item())
                                alpha_line_losses.append(loss_before)
                                alpha_line_exact_losses.append(exact_loss_before)
                                alpha_line_approx_errors.append(
                                    abs(loss_before - exact_loss_before)
                                )
                                alpha_line_grad_values.append(grad_alpha)
                                if alpha_line_interval_loss_start is None:
                                    alpha_line_interval_loss_start = exact_loss_before
                                    alpha_line_interval_probe_batch = alpha_batch

                                if not math.isfinite(loss_before) or not math.isfinite(
                                    grad_alpha
                                ):
                                    alpha_line_stop_reason = "non_finite"
                                    break
                                if grad_alpha >= 0.0:
                                    alpha_line_stop_reason = "g_non_descent"
                                    alpha_line_consecutive_rejects += 1
                                    cycle_state.consecutive_rejects = (
                                        alpha_line_consecutive_rejects
                                    )
                                    if (
                                        alpha_line_consecutive_rejects
                                        >= alpha_line_max_consecutive_reject
                                    ):
                                        alpha_line_stop_reason = (
                                            "max_consecutive_reject"
                                        )
                                        break
                                    continue

                                alpha_candidate = (
                                    alpha_current - alpha_line_alpha_lr * grad_alpha
                                )
                                alpha_candidate_t = torch.tensor(
                                    alpha_candidate,
                                    device=input_device,
                                    dtype=torch.float32,
                                )
                                with torch.no_grad():
                                    if use_first_order and first_order_cache is not None:
                                        loss_new_t = alpha_line_loss_cached_first_order(
                                            model,
                                            first_order_cache,
                                            alpha_batch,
                                            alpha_candidate_t,
                                        )
                                        exact_loss_new_t = alpha_line_loss_exact(
                                            model,
                                            alpha_batch,
                                            base_alpha_snapshot,
                                            direction,
                                            alpha_candidate_t,
                                            active_names=active_names,
                                        )
                                    else:
                                        loss_new_t = _alpha_line_functional_loss(
                                            model,
                                            alpha_batch,
                                            base_alpha_snapshot,
                                            direction,
                                            alpha_candidate_t,
                                            active_names=active_names,
                                        )
                                        exact_loss_new_t = loss_new_t
                                loss_new = float(loss_new_t.detach().item())
                                exact_loss_new = float(exact_loss_new_t.detach().item())
                                if alpha_line_future_work_internal_metrics_enabled:
                                    future_work_internal_pairs.append(
                                        {
                                            "g_dot_v": grad_alpha,
                                            "exact_loss_delta": (
                                                exact_loss_new - exact_loss_before
                                            ),
                                        }
                                    )
                                alpha_line_losses.append(loss_new)
                                alpha_line_exact_losses.append(exact_loss_new)
                                alpha_line_approx_errors.append(
                                    abs(loss_new - exact_loss_new)
                                )
                                if (
                                    math.isfinite(loss_new)
                                    and loss_new
                                    <= loss_before + controller.rollback_tolerance
                                ):
                                    _apply_alpha_direction_from_base(
                                        model,
                                        base_alpha_snapshot,
                                        direction,
                                        alpha_candidate,
                                        active_names=active_names,
                                    )
                                    alpha_current = alpha_candidate
                                    alpha_line_step_count += 1
                                    alpha_line_consecutive_rejects = 0
                                    cycle_state.consecutive_rejects = 0
                                    cycle_state.alpha_steps_in_cycle = (
                                        alpha_line_step_count
                                    )
                                    cycle_state.current_alpha = alpha_current
                                    loss_after = exact_loss_new
                                    accepted = True
                                    alpha_line_stop_reason = "max"
                                    continue

                                alpha_line_stop_reason = "reject"
                                alpha_line_consecutive_rejects += 1
                                cycle_state.consecutive_rejects = (
                                    alpha_line_consecutive_rejects
                                )
                                if (
                                    alpha_line_consecutive_rejects
                                    >= alpha_line_max_consecutive_reject
                                ):
                                    alpha_line_stop_reason = "max_consecutive_reject"
                                    break
                                continue
                            alpha_line_alpha_wall_seconds = (
                                time.perf_counter() - alpha_loop_started
                            )
                            alpha_line_alpha_after = alpha_current
                            if (
                                alpha_line_interval_loss_start is not None
                                and alpha_line_interval_probe_batch is not None
                                and alpha_line_step_count > 0
                            ):
                                with torch.no_grad():
                                    alpha_line_interval_loss_end_t = alpha_line_loss_exact(
                                        model,
                                        alpha_line_interval_probe_batch,
                                        base_alpha_snapshot,
                                        direction,
                                        torch.tensor(
                                            alpha_current,
                                            device=input_device,
                                            dtype=torch.float32,
                                        ),
                                        active_names=active_names,
                                    )
                                alpha_line_interval_loss_end = float(
                                    alpha_line_interval_loss_end_t.detach().item()
                                )
                                alpha_line_interval_net_loss_delta = (
                                    alpha_line_interval_loss_end
                                    - alpha_line_interval_loss_start
                                )
                                cycle_state.interval_net_loss_delta = (
                                    alpha_line_interval_net_loss_delta
                                )
                            proposal.N = alpha_line_step_count
                            if alpha_line_step_count <= 0:
                                _apply_alpha_direction_from_base(
                                    model,
                                    base_alpha_snapshot,
                                    direction,
                                    alpha_line_alpha_init,
                                    active_names=active_names,
                                )
                                loss_after = loss_pilot
                                accepted = False
                                if alpha_line_stop_reason == "max":
                                    alpha_line_stop_reason = "no_accepted_steps"

                        post_extrapolation_eval = False
                        post_extrapolation_eval_skipped = True
                        post_extrapolation_eval_skip_reason = (
                            f"alpha_line_{alpha_line_stop_reason}"
                        )
                        if accepted:
                            controller.commit_proposal(proposal)
                            controller.reward(loss_pilot, loss_after)
                            controller.update_layer_scores(list(active_indices), 1.0)
                        else:
                            controller.penalize(loss_pilot, loss_after)
                            controller.update_layer_scores(list(active_indices), -1.0)
                        logger.debug(
                            "alpha-line stats: cycle=%d accepted_steps=%d alpha=%s->%s "
                            "stop=%s grads=%s losses=%s v_wall=%.4f alpha_wall=%.4f",
                            cycle,
                            alpha_line_step_count,
                            alpha_line_alpha_before,
                            alpha_line_alpha_after,
                            alpha_line_stop_reason,
                            alpha_line_grad_values,
                            alpha_line_losses,
                            alpha_line_v_update_wall_seconds,
                            alpha_line_alpha_wall_seconds,
                        )
                    elif subspace_m9_enabled:
                        accepted = False
                        loss_after = loss_pilot
                        proposal.N = 0
                        history_deltas = delta_tracker._history
                    
                        if len(history_deltas) < 2:
                            logger.warning(
                                "subspace M9: history length (%d) < 2, skipping extrapolation",
                                len(history_deltas)
                            )
                            post_extrapolation_eval = False
                            post_extrapolation_eval_skipped = True
                            post_extrapolation_eval_skip_reason = "subspace_m9_insufficient_history"
                            extrap_stats = ExtrapolationStats()
                        else:
                            empty_supervision_skips = 0
                            while True:
                                m9_batch = batch_iter.next()
                                batch_position = train_batch_position
                                train_batch_position += 1
                                if has_supervised_tokens(m9_batch):
                                    break
                                empty_supervision_skips += 1
                                if empty_supervision_skips > max_empty_supervision_skips:
                                    raise RuntimeError(
                                        "Exceeded empty-supervision skip budget while "
                                        "filling an M9 step"
                                    )
                        
                            if deterministic_data_order:
                                batch_locator = (
                                    batch_plan_manifest.batch_locator_at_position(
                                        batch_position
                                    )
                                )
                                cycle_batch_keys.append(batch_locator.batch_key)
                                cycle_sample_keys.extend(batch_locator.sample_keys)

                            def _m9_loss_fn(batch_data: dict[str, torch.Tensor]) -> float:
                                return _forward_loss_no_grad(model, batch_data)

                            m9_delta, m9_stats = subspace_m9_fit_step(
                                model=model,
                                history=history_deltas[-tg_cfg.get("N_initial", 10):],
                                active_names=active_names,
                                batch=m9_batch,
                                loss_fn=_m9_loss_fn,
                                selected_N=selected_N,
                                fd_epsilon=subspace_m9_fd_eps,
                                fit_lr=subspace_m9_lr,
                                fit_steps=subspace_m9_steps,
                                velocity_direction=velocity.long_state,
                            )

                            params_active = {name: p for name, p in model.named_parameters() if name in active_names}
                        
                            for name, p in params_active.items():
                                p.data.add_(m9_delta[name].to(p.device))

                            proposal.N = selected_N
                            post_extrapolation_eval = True
                            post_extrapolation_eval_skipped = False
                            post_extrapolation_eval_skip_reason = "subspace_m9_fitted"

                            delta_norm = float(sum(d.norm().item()**2 for d in m9_delta.values())**0.5)
                            extrap_stats = ExtrapolationStats(
                                num_tensors=1,
                                capped_tensors=0,
                                raw_update_norm=delta_norm,
                                applied_update_norm=delta_norm,
                                min_cap_ratio=1.0,
                                mean_cap_ratio=1.0,
                            )
                        
                            logger.info(
                                "subspace M9 stats: cycle=%d alpha_fit=%.4f beta1_fit=%.4f beta2_fit=%.4f loss_initial=%.4f loss_final=%.4f w_traj=%.6f v0=%s",
                                cycle,
                                m9_stats.get("alpha_fit", 1.0),
                                m9_stats.get("beta1_fit", 0.0),
                                m9_stats.get("beta2_fit", 0.0),
                                m9_stats.get("loss_initial", 0.0),
                                m9_stats.get("loss_final", 0.0),
                                m9_stats.get("w_traj", 0.0),
                                m9_stats.get("v0_source", "unknown"),
                            )
                            # [作業A] m9_stats を cycle 単位で保存 → main metrics.record_step へ渡す
                            m9_cycle_stats = m9_stats
                    elif subspace_zo_enabled:
                        accepted = False
                        loss_after = loss_pilot
                        proposal.N = 0
                        raw_step_norm_sq = 0.0
                        applied_step_norm_sq = 0.0
                        cap_ratio_sum = 0.0
                        min_cap_ratio = 1.0
                        capped_steps = 0
                        max_zo_steps = max(0, subspace_zo_max_steps_per_cycle)
                        if predicted_consistency < subspace_zo_tau_cos:
                            accepted = False
                            loss_after = loss_pilot
                            proposal.N = 0
                            post_extrapolation_eval = False
                            post_extrapolation_eval_skipped = True
                            post_extrapolation_eval_skip_reason = "subspace_zo_low_cos"
                        else:
                            for _zo_index in range(max_zo_steps):
                                basis = velocity.build_orthonormal_basis(
                                    active_names=active_names,
                                    tau_dim=subspace_zo_tau_dim,
                                    force_dim=subspace_zo_force_dim,
                                )
                                if basis.dim <= 0:
                                    post_extrapolation_eval_skip_reason = (
                                        "subspace_zo_no_basis"
                                    )
                                    break

                                empty_supervision_skips = 0
                                while True:
                                    zo_batch = batch_iter.next()
                                    batch_position = train_batch_position
                                    train_batch_position += 1
                                    if has_supervised_tokens(zo_batch):
                                        break
                                    empty_supervision_skips += 1
                                    if empty_supervision_skips > max_empty_supervision_skips:
                                        raise RuntimeError(
                                            "Exceeded empty-supervision skip budget while "
                                            "filling a zeroth-order step"
                                        )
                                if deterministic_data_order:
                                    batch_locator = (
                                        batch_plan_manifest.batch_locator_at_position(
                                            batch_position
                                        )
                                    )
                                    cycle_batch_keys.append(batch_locator.batch_key)
                                    cycle_sample_keys.extend(batch_locator.sample_keys)

                                def _zo_loss_closure(
                                    batch: dict[str, torch.Tensor] = zo_batch,
                                ) -> float:
                                    return _forward_loss_no_grad(model, batch)

                                zo_last_stats = subspace_zeroth_order_step(
                                    model=model,
                                    basis=basis,
                                    active_names=active_names,
                                    loss_closure=_zo_loss_closure,
                                    mu_ratio=subspace_zo_mu_ratio,
                                    eps_curv=subspace_zo_eps_curv,
                                    eta_fallback_ratio=subspace_zo_eta_fallback_ratio,
                                    max_step_ratio=subspace_zo_max_step_ratio,
                                    tolerance=controller.rollback_tolerance,
                                    disable_curvature=subspace_zo_disable_curvature,
                                    stop_on_positive_primary_g=(
                                        subspace_zo_stop_on_positive_g1
                                    ),
                                    primary_g_stop_epsilon=subspace_zo_g1_stop_epsilon,
                                )
                                zo_attempted_steps += 1
                                zo_forward_count += zo_last_stats.forward_count
                                if zo_last_stats.dim == 1:
                                    zo_dim1_steps += 1
                                elif zo_last_stats.dim == 2:
                                    zo_dim2_steps += 1
                                if zo_last_stats.capped:
                                    capped_steps += 1
                                min_cap_ratio = min(min_cap_ratio, zo_last_stats.cap_ratio)
                                if math.isfinite(zo_last_stats.raw_step_norm):
                                    raw_step_norm_sq += zo_last_stats.raw_step_norm**2
                                if math.isfinite(zo_last_stats.applied_step_norm):
                                    applied_step_norm_sq += (
                                        zo_last_stats.applied_step_norm**2
                                    )
                                cap_ratio_sum += zo_last_stats.cap_ratio
                                if zo_last_stats.accepted:
                                    zo_accepted_steps += 1
                                    loss_after = zo_last_stats.loss_new
                                else:
                                    zo_rejected_steps += 1
                                    if zo_accepted_steps == 0:
                                        loss_after = loss_pilot
                                    post_extrapolation_eval_skip_reason = (
                                        "subspace_zo_"
                                        f"{zo_last_stats.termination_reason or 'rejected'}"
                                    )
                                    break

                            accepted = zo_accepted_steps > 0
                            proposal.N = zo_accepted_steps
                            post_extrapolation_eval = False
                            post_extrapolation_eval_skipped = True
                            if post_extrapolation_eval_skip_reason == "not_reached":
                                post_extrapolation_eval_skip_reason = (
                                    "subspace_zo_train_probe"
                                )
                            if accepted:
                                controller.commit_proposal(proposal)
                                controller.reward(loss_pilot, loss_after)
                                controller.update_layer_scores(list(active_indices), 1.0)
                            else:
                                controller.penalize(loss_pilot, loss_after)
                                controller.update_layer_scores(list(active_indices), -1.0)

                        extrap_stats = ExtrapolationStats(
                            num_tensors=zo_attempted_steps,
                            capped_tensors=capped_steps,
                            raw_update_norm=raw_step_norm_sq**0.5,
                            applied_update_norm=applied_step_norm_sq**0.5,
                            min_cap_ratio=min_cap_ratio if zo_attempted_steps else 1.0,
                            mean_cap_ratio=(
                                cap_ratio_sum / zo_attempted_steps
                                if zo_attempted_steps
                                else 1.0
                            ),
                        )
                        logger.debug(
                            "subspace ZO stats: cycle=%d accepted=%d/%d rejected=%d "
                            "forwards=%d dim1=%d dim2=%d last_loss=%.4f->%.4f "
                            "last_residual=%.3f cap_ratio=%.3f",
                            cycle,
                            zo_accepted_steps,
                            zo_attempted_steps,
                            zo_rejected_steps,
                            zo_forward_count,
                            zo_dim1_steps,
                            zo_dim2_steps,
                            zo_last_stats.loss_initial,
                            zo_last_stats.loss_new,
                            zo_last_stats.residual_norm,
                            zo_last_stats.cap_ratio,
                        )
                    else:
                        extrap_stats = apply_extrapolation(
                            model=model,
                            velocity=current_velocity,
                            active_names=active_names,
                            n_steps=proposal.N,
                            lr=pilot_lr,
                            relative_update_cap=proposal.relative_update_cap,
                        )
                        if extrap_stats is None:
                            extrap_stats = ExtrapolationStats()
                        logger.debug(
                            "extrapolation cap stats: cycle=%d raw_norm=%.4e applied_norm=%.4e "
                            "global_ratio=%.3f mean_ratio=%.3f min_ratio=%.3f capped=%d/%d",
                            cycle,
                            extrap_stats.raw_update_norm,
                            extrap_stats.applied_update_norm,
                            extrap_stats.global_cap_ratio,
                            extrap_stats.mean_cap_ratio,
                            extrap_stats.min_cap_ratio,
                            extrap_stats.capped_tensors,
                            extrap_stats.num_tensors,
                        )

                    if (
                        save_trajectory_delta_artifacts
                        and pilot_state_snapshot is not None
                        and cycle % trajectory_delta_artifact_interval == 0
                    ):
                        spec_delta = snapshot_lora_delta(model, pilot_state_snapshot)
                        spec_metadata = build_trajectory_delta_artifact_metadata(
                            mode="tg_lora",
                            anchor_kind="after_speculative_update",
                            trajectory_key=trajectory_key,
                            epoch_batch_plan_key=batch_plan_manifest.epoch_batch_plan_key,
                            batch_plan_manifest=(
                                str(batch_plan_manifest_path)
                                if save_batch_plan_manifest
                                else None
                            ),
                            dataset_key=batch_plan_manifest.dataset_key,
                            delta_tensors=spec_delta,
                            cycle=cycle,
                            total_backward_passes=pilot_backward_passes,
                            batch_keys=cycle_batch_keys,
                            sample_keys=cycle_sample_keys,
                            extra_metadata={
                                "K": pilot_K,
                                "N": proposal.N,
                                "proposed_N_before_cosine": proposal_N_before_cosine,
                                "pilot_lr": pilot_lr,
                                "relative_update_cap": proposal.relative_update_cap,
                                "raw_delta_cosine_sim": raw_delta_cos_sim,
                                "velocity_predicted_consistency": predicted_consistency,
                                "velocity_short_long_norm_ratio": velocity_norm_ratio,
                                "raw_update_norm": extrap_stats.raw_update_norm,
                                "applied_update_norm": extrap_stats.applied_update_norm,
                                "global_cap_ratio": extrap_stats.global_cap_ratio,
                                "mean_cap_ratio": extrap_stats.mean_cap_ratio,
                                "min_cap_ratio": extrap_stats.min_cap_ratio,
                                "capped_fraction": extrap_stats.capped_fraction,
                                "subspace_zo_enabled": subspace_zo_enabled,
                                "subspace_zo_attempted_steps": zo_attempted_steps,
                                "subspace_zo_accepted_steps": zo_accepted_steps,
                                "subspace_zo_rejected_steps": zo_rejected_steps,
                                "subspace_zo_forward_count": zo_forward_count,
                                "subspace_zo_last_residual_norm": (
                                    zo_last_stats.residual_norm
                                ),
                            },
                        )
                        save_trajectory_delta_artifact(
                            path=trajectory_delta_artifact_dir
                            / artifact_file_name(
                                mode="tg_lora",
                                anchor_kind="after_speculative_update",
                                cycle=cycle,
                            ),
                            metadata=spec_metadata,
                            delta_tensors=spec_delta,
                        )

                    # 10b. 外挿後パラメータ有限性チェック
                    params_finite, non_finite_detail = check_lora_params_finite(model)
                    if not params_finite:
                        logger.warning(
                            f"Non-finite LoRA params after extrapolation: {non_finite_detail}. "
                            f"Rolling back and penalizing."
                        )
                        try:
                            rollback_mgr.rollback(model)
                        except (RuntimeError, IndexError) as exc:
                            logger.error(
                                f"Rollback failed, model state may be corrupted: {exc}"
                            )
                        controller.penalize(loss_pilot, float("inf"))
                        controller.update_layer_scores(list(active_indices), -1.0)
                        accepted = False
                        loss_after = float("inf")
                        # Skip the normal accept/rollback flow below
                        if snapshot_taken:
                            rollback_mgr.pop()
                            snapshot_taken = False
                        cycle_state.record_cycle(
                            train_loss=pilot_loss_avg,
                            valid_loss=loss_pilot,
                            accepted=accepted,
                            actual_backward_passes=pilot_backward_pass_count,
                            speculative_optimizer_steps=proposal.N if accepted else 0,
                            optimizer_steps=pilot_K,
                            speculative_equivalent_backward_passes=(proposal.N * grad_accum) if accepted else 0,
                        )
                        accepted_valid_history.append(loss_pilot)
                        if len(accepted_valid_history) > max(3, moving_avg_window * 2):
                            accepted_valid_history.pop(0)
                        last_quick_valid_loss = loss_pilot
                        continue

                    # 10c. [accept-after-sgd] M9着地点から1回SGDしてから評価する。
                    # 外挿はジグザグをショートカットするので着地点のlossは必ず上振れする。
                    # しかし着地点が「良い場所」なら1回SGDでpilot到達点以下に回復するはず。
                    # これをaccept判定に使うことで、外挿の真の価値を測る。
                    accept_after_sgd_steps = tg_cfg.get("accept_after_sgd_steps", 0)
                    if accept_after_sgd_steps > 0 and subspace_m9_enabled:
                        sgd_step_lr = tg_cfg.get("accept_after_sgd_lr", None)
                        if sgd_step_lr is None:
                            sgd_step_lr = cfg.training.learning_rate
                        for _sgd_i in range(accept_after_sgd_steps):
                            sgd_batch = batch_iter.next()
                            while not has_supervised_tokens(sgd_batch):
                                sgd_batch = batch_iter.next()
                            model.train()
                            with torch.set_grad_enabled(True):
                                sgd_loss = compute_loss(model, sgd_batch)
                                sgd_loss.backward()
                            with torch.no_grad():
                                for _n, _p in model.named_parameters():
                                    if _n in active_names and _p.grad is not None:
                                        _p.data -= sgd_step_lr * _p.grad.detach()
                                        _p.grad = None
                            model.eval()
                            logger.info(
                                "accept-after-sgd: cycle=%d step=%d loss_at_landing=%.4f",
                                cycle, _sgd_i, float(sgd_loss.detach().item()),
                            )

                    # 11. Accept-probe eval (外挿後) — cosine-gated skip or cache use.
                    cache_eligible = False
                    cache_hit = False
                    velocity_anomalous = velocity.is_magnitude_anomalous()
                    if validation_skip_enabled:
                        should_eval_after, skip_policy_reason = (
                            _decide_post_extrapolation_eval_policy(
                                consistency=cos_sim,
                                selected_N=int(proposal.N),
                                total_cycles=cycle_state.total_cycles,
                                acceptance_rate=cycle_state.acceptance_rate,
                                velocity_anomalous=velocity_anomalous,
                                enabled=True,
                                high_cos=validation_skip_high_cos,
                                mid_cos=validation_skip_mid_cos,
                                mid_eval_every=validation_skip_mid_eval_every,
                                min_cycles=validation_skip_min_cycles,
                                min_acceptance_rate=validation_skip_min_acceptance_rate,
                                force_eval_N=validation_skip_force_eval_N,
                            )
                        )
                    else:
                        confident_skip_threshold = cfg.tg_lora.get(
                            "confident_skip_cos", 0.0
                        )
                        confident_skip_min_cycles = cfg.tg_lora.get(
                            "confident_skip_min_cycles", 10
                        )
                        can_legacy_skip = (
                            confident_skip_threshold > 0
                            and cos_sim >= confident_skip_threshold
                            and cycle_state.acceptance_rate >= 0.8
                            and cycle_state.total_cycles >= confident_skip_min_cycles
                            and not velocity_anomalous
                        )
                        should_eval_after = not can_legacy_skip
                        skip_policy_reason = (
                            "legacy_confident_skip" if can_legacy_skip else "legacy_eval"
                        )

                    can_confident_skip = not should_eval_after
                    if subspace_zo_enabled:
                        can_confident_skip = True
                        if post_extrapolation_eval_skip_reason == "not_reached":
                            post_extrapolation_eval_skip_reason = "subspace_zo_train_probe"
                        skip_policy_reason = post_extrapolation_eval_skip_reason
                    if alpha_line_enabled:
                        # Only skip validation evaluation if the alpha-line search did not accept any steps (i.e. model rolled back to pilot state).
                        # If steps were accepted, we MUST evaluate on the validation set to guard against training-batch overfitting.
                        can_confident_skip = not accepted
                        skip_policy_reason = post_extrapolation_eval_skip_reason

                    if can_confident_skip:
                        # Eval skip: velocity direction is trusted for this cycle.
                        if not subspace_zo_enabled and not alpha_line_enabled:
                            loss_after = loss_pilot  # assume no degradation
                            accepted = True
                        post_extrapolation_eval_skipped = True
                        post_extrapolation_eval_skip_reason = skip_policy_reason
                        if not subspace_zo_enabled and not alpha_line_enabled:
                            controller.commit_proposal(proposal)
                            controller.reward(loss_pilot, loss_after)
                            controller.update_layer_scores(list(active_indices), 1.0)
                        logger.debug(
                            "Post-extrapolation eval skipped: cycle=%d reason=%s "
                            "cos=%.3f N=%d acc_rate=%.2f",
                            cycle,
                            skip_policy_reason,
                            cos_sim,
                            proposal.N,
                            cycle_state.acceptance_rate,
                        )
                    else:
                        post_extrapolation_eval_skipped = False
                        post_extrapolation_eval_skip_reason = skip_policy_reason
                        post_extrapolation_eval = True
                        # active_indicesが予測と一致し、キャッシュが有効なら高速eval
                        if use_prefix_feature_cache:
                            torch.cuda.empty_cache()  # Free memory before eval to prevent OOM
                            loss_after = eval_loss(
                                model,
                                valid_quick_loader,
                                input_device,
                                max_examples=accept_eval_examples,
                            )
                            validation_forwards_this_cycle += 1
                            post_validation_forwards_this_cycle += 1
                        else:
                            actual_split = determine_split_layer(
                                active_indices, num_decoder_layers
                            )
                            cache_eligible = use_cache
                            cache_usable = (
                                use_cache
                                and activation_cache.is_valid
                                and actual_split
                                >= split_layer  # 実際の変更がキャッシュ境界以降
                            )
                            if cache_eligible:
                                activation_cache_eligible_count += 1
                            if cache_usable:
                                cache_hit = True
                                activation_cache_hit_count += 1
                                loss_after = activation_cache.eval_from_cache(
                                    model, input_device
                                )
                                validation_forwards_this_cycle += 1
                                post_validation_forwards_this_cycle += 1
                            else:
                                if cache_eligible:
                                    activation_cache_miss_count += 1
                                torch.cuda.empty_cache()  # Free memory before eval to prevent OOM
                                loss_after = eval_loss(
                                    model,
                                    valid_quick_loader,
                                    input_device,
                                    max_examples=accept_eval_examples,
                                )
                                validation_forwards_this_cycle += 1
                                post_validation_forwards_this_cycle += 1

                    # 11b. Shadow extrapolation: measure without committing.
                    # If enabled, always rollback and record what WOULD have happened.
                    shadow_enabled = tg_cfg.get("shadow_extrapolation_enabled", False)
                    if shadow_enabled:
                        # Force eval (never skip in shadow mode)
                        if not post_extrapolation_eval:
                            torch.cuda.empty_cache()
                            loss_after = eval_loss(
                                model,
                                valid_quick_loader,
                                input_device,
                                max_examples=accept_eval_examples,
                            )
                            validation_forwards_this_cycle += 1
                            post_validation_forwards_this_cycle += 1
                            post_extrapolation_eval = True

                        shadow_delta = loss_after - loss_pilot
                        logger.info(
                            "SHADOW: cycle=%d cos=%.4f loss_pilot=%.4f loss_after=%.4f delta=%+.4f N=1",
                            cycle, cos_sim, loss_pilot, loss_after, shadow_delta,
                        )
                        # Always rollback — learning is unaffected
                        try:
                            rollback_mgr.rollback(model)
                        except (RuntimeError, IndexError):
                            pass
                        accepted = False
                        _shadow_delta = shadow_delta
                        _shadow_cos = cos_sim
                        loss_after = loss_pilot  # for cycle recording

                    # 12. Accept / Rollback (normal mode)
                    # Guard: reject non-finite loss_after to prevent bad accept decisions
                    if not math.isfinite(loss_after):
                        logger.warning(
                            "Non-finite loss_after=%s detected, treating as rejection",
                            loss_after,
                        )
                        loss_after = float("inf")
                    accepted, _reason = _decide_accept_rollback(
                        loss_pilot,
                        loss_after,
                        controller.rollback_tolerance,
                        loss_history=accepted_valid_history[-moving_avg_window:],
                        temperature=soft_accept_temperature,
                    )
                    logger.info(
                        "M9 accept eval: cycle=%d loss_pilot=%.4f loss_after=%.4f delta=%+.4f accepted=%s reason=%s",
                        cycle, loss_pilot, loss_after, loss_after - loss_pilot, accepted, _reason,
                    )
                    if accepted:
                        controller.commit_proposal(proposal)
                        controller.reward(loss_pilot, loss_after)
                        controller.update_layer_scores(list(active_indices), 1.0)
                        # [作業B] Accept後の履歴全捨て + ウォームアップ再突入
                        # 外挿で大きく飛んだ後は過去の方向が陳腐化するため、
                        # 履歴を全クリアして再ウォームアップで方向を溜め直す。
                        if subspace_m9_enabled:
                            delta_tracker._history.clear()
                            delta_tracker._norm_history.clear()
                            warmup_released = False
                            warmup_cos_consecutive = 0
                            logger.info(
                                "M9 accept: history cleared, warmup reset at cycle %d",
                                cycle,
                            )
                    else:
                        try:
                            rollback_mgr.rollback(model)
                        except (RuntimeError, IndexError) as exc:
                            logger.error(
                                f"Rollback failed, model state may be corrupted: {exc}"
                            )
                        controller.penalize(loss_pilot, loss_after)
                        controller.update_layer_scores(list(active_indices), -1.0)
                finally:
                    if snapshot_taken:
                        rollback_mgr.pop()

                # Record cycle state
                # On full-eval cycles, skip quick-eval stale tracking to avoid
                # double-counting; record_full_eval will handle it instead.
                is_full_eval_cycle = should_run_full_eval(
                    cycle, cfg.eval.get("full_eval_every_cycles", 10)
                )
                final_valid_loss = loss_after if accepted else loss_pilot
                cycle_state.record_cycle(
                    train_loss=pilot_loss_avg,
                    valid_loss=None if is_full_eval_cycle else final_valid_loss,
                    accepted=accepted,
                    actual_backward_passes=pilot_backward_pass_count,
                    speculative_optimizer_steps=proposal.N if accepted else 0,
                    optimizer_steps=pilot_K,
                    speculative_equivalent_backward_passes=(proposal.N * grad_accum) if accepted else 0,
                )
                accepted_valid_history.append(final_valid_loss)
                # LAWA: record current LoRA weights after accept/reject settled
                if lawa_averager is not None:
                    lawa_averager.record(model, cycle=cycle)
                if len(accepted_valid_history) > max(3, moving_avg_window * 2):
                    accepted_valid_history.pop(0)
                last_quick_valid_loss = final_valid_loss

                pilot_validation_forward_count += pilot_validation_forwards_this_cycle
                post_validation_forward_count += post_validation_forwards_this_cycle
                if post_extrapolation_eval:
                    post_extrapolation_eval_count += 1
                if post_extrapolation_eval_skipped:
                    post_extrapolation_eval_skipped_count += 1
                    post_extrapolation_eval_skip_reasons[
                        post_extrapolation_eval_skip_reason
                    ] = (
                        post_extrapolation_eval_skip_reasons.get(
                            post_extrapolation_eval_skip_reason, 0
                        )
                        + 1
                    )
                subspace_zo_attempted_steps_total += zo_attempted_steps
                subspace_zo_accepted_steps_total += zo_accepted_steps
                subspace_zo_rejected_steps_total += zo_rejected_steps
                subspace_zo_forward_count_total += zo_forward_count
                subspace_zo_dim1_steps_total += zo_dim1_steps
                subspace_zo_dim2_steps_total += zo_dim2_steps
                alpha_line_steps_total += alpha_line_step_count
                alpha_line_base_recompute_total += alpha_line_base_recompute
                alpha_line_v_update_wall_seconds_total += alpha_line_v_update_wall_seconds
                alpha_line_alpha_wall_seconds_total += alpha_line_alpha_wall_seconds
                future_work_record = None
                if (
                    alpha_line_future_work_metrics_enabled
                    or alpha_line_future_work_internal_metrics_enabled
                ):
                    future_work_record = {
                        "namespace": "future_work",
                        "paper_scope": "motivation_only",
                    }
                    if future_work_projection_ratio is not None:
                        future_work_record["projection_ratio"] = {
                            "value": future_work_projection_ratio,
                            "kind": "abs_dot_pilot_grad_sum_fixed_v_over_norms",
                        }
                        future_work_projection_ratios.append(future_work_projection_ratio)
                    if (
                        alpha_line_future_work_internal_metrics_enabled
                        and future_work_internal_pairs
                    ):
                        future_work_record["internal"] = {
                            "paper_exclude": True,
                            "g_dot_v_loss_delta_pairs": future_work_internal_pairs,
                        }
                        future_work_internal_pair_count += len(future_work_internal_pairs)
                    if len(future_work_record) == 2:
                        future_work_record = None

            # JSON-extraction gold eval (Guard §5.2): generate + score on the
            # held-out gold set when due. Recorded per-cycle via extra_fields;
            # the §5.2 stop (gold >= G*) is resolved post-hoc by the analyzer.
            gold_scores: dict[str, float] = {}
            if (
                gold_eval_records
                and cycle % cfg.eval.get("gold_eval_every_cycles", 5) == 0
            ):
                torch.cuda.empty_cache()  # Free memory before generation to prevent OOM
                gold_scores = evaluate_json_extraction_run(
                    model,
                    tokenizer,
                    gold_eval_records,
                    max_examples=cfg.eval.get("gold_eval_max_examples", 50),
                    max_new_tokens=cfg.eval.get("gold_eval_max_new_tokens", 128),
                    device=input_device,
                )

            metrics.record_step(
                step=cycle_state.full_backward_passes,
                cycle=cycle,
                loss_train=pilot_loss_avg,
                loss_valid=final_valid_loss,
                backward_passes=pilot_backward_pass_count,
                total_backward_passes=cycle_state.full_backward_passes,
                tg_lora_accepted=accepted,
                tg_lora_cosine_sim=cos_sim,
                tg_lora_raw_delta_cosine_sim=raw_delta_cos_sim,
                tg_lora_predicted_consistency=predicted_consistency,
                tg_lora_short_long_norm_ratio=velocity_norm_ratio,
                tg_lora_reduction_rate=cycle_state.reduction_rate,
                tg_lora_K=pilot_K,
                tg_lora_N=proposal.N,
                tg_lora_proposed_N=proposal_N_before_cosine,
                tg_lora_alpha=proposal.alpha,
                tg_lora_beta=controller.state.beta,
                tg_lora_lr=pilot_lr,
                tg_lora_cache_built=use_cache,
                tg_lora_cache_eligible=(None if can_confident_skip else cache_eligible),
                tg_lora_cache_hit=(None if can_confident_skip else cache_hit),
                tg_lora_validation_forwards=validation_forwards_this_cycle,
                tg_lora_pilot_validation_forwards=pilot_validation_forwards_this_cycle,
                tg_lora_post_validation_forwards=post_validation_forwards_this_cycle,
                tg_lora_post_extrapolation_eval=post_extrapolation_eval,
                tg_lora_post_extrapolation_eval_skipped=post_extrapolation_eval_skipped,
                tg_lora_post_extrapolation_eval_skip_reason=post_extrapolation_eval_skip_reason,
                tg_lora_rollback_triggered=bool((not accepted) and proposal.N > 0),
                tg_lora_loss_after=loss_after if post_extrapolation_eval else None,
                tg_lora_loss_pilot_eval=loss_pilot,
                tg_lora_shadow_delta=_shadow_delta,
                tg_lora_shadow_cos=_shadow_cos,
                # [作業A] M9 フィット係数と w_traj を毎サイクル記録する。
                # subspace_m9_fit_step が実行されなかったサイクルは None。
                tg_lora_m9_alpha_fit=m9_cycle_stats.get("alpha_fit") if m9_cycle_stats else None,
                tg_lora_m9_beta1_fit=m9_cycle_stats.get("beta1_fit") if m9_cycle_stats else None,
                tg_lora_m9_beta2_fit=m9_cycle_stats.get("beta2_fit") if m9_cycle_stats else None,
                tg_lora_w_traj=m9_cycle_stats.get("w_traj") if m9_cycle_stats else None,
                tg_lora_cap_global_ratio=extrap_stats.global_cap_ratio,
                tg_lora_cap_mean_ratio=extrap_stats.mean_cap_ratio,
                tg_lora_cap_min_ratio=extrap_stats.min_cap_ratio,
                tg_lora_cap_capped_fraction=extrap_stats.capped_fraction,
                tg_lora_cap_capped_tensors=extrap_stats.capped_tensors,
                tg_lora_cap_tensors=extrap_stats.num_tensors,
                tg_lora_raw_update_norm=extrap_stats.raw_update_norm,
                tg_lora_applied_update_norm=extrap_stats.applied_update_norm,
                tg_lora_zo_enabled=subspace_zo_enabled,
                tg_lora_zo_attempted_steps=zo_attempted_steps,
                tg_lora_zo_accepted_steps=zo_accepted_steps,
                tg_lora_zo_rejected_steps=zo_rejected_steps,
                tg_lora_zo_forward_count=zo_forward_count,
                tg_lora_zo_dim1_steps=zo_dim1_steps,
                tg_lora_zo_dim2_steps=zo_dim2_steps,
                tg_lora_zo_last_dim=zo_last_stats.dim,
                tg_lora_zo_last_residual_norm=zo_last_stats.residual_norm,
                tg_lora_zo_last_loss_initial=zo_last_stats.loss_initial,
                tg_lora_zo_last_loss_new=zo_last_stats.loss_new,
                tg_lora_zo_last_g1=(
                    zo_last_stats.directions[0].g
                    if len(zo_last_stats.directions) >= 1
                    else None
                ),
                tg_lora_zo_last_h1=(
                    zo_last_stats.directions[0].h
                    if len(zo_last_stats.directions) >= 1
                    else None
                ),
                tg_lora_zo_last_g2=(
                    zo_last_stats.directions[1].g
                    if len(zo_last_stats.directions) >= 2
                    else None
                ),
                tg_lora_zo_last_h2=(
                    zo_last_stats.directions[1].h
                    if len(zo_last_stats.directions) >= 2
                    else None
                ),
                tg_lora_zo_last_stop_reason=zo_last_stats.termination_reason,
                alpha_line_enabled=alpha_line_enabled,
                alpha_line_v_updated=alpha_line_v_updated,
                alpha_line_alpha_before=alpha_line_alpha_before,
                alpha_line_alpha_after=alpha_line_alpha_after,
                alpha_line_alpha_steps=alpha_line_step_count,
                alpha_line_grad_values=alpha_line_grad_values or None,
                alpha_line_losses=alpha_line_losses or None,
                alpha_line_exact_losses=alpha_line_exact_losses or None,
                alpha_line_approx_errors=alpha_line_approx_errors or None,
                alpha_line_jvp_methods=alpha_line_jvp_methods or None,
                alpha_line_order=alpha_line_order,
                alpha_line_finite_diff_eps=alpha_line_finite_diff_eps,
                alpha_line_consecutive_rejects=alpha_line_consecutive_rejects,
                alpha_line_interval_net_loss_delta=alpha_line_interval_net_loss_delta,
                alpha_line_base_recompute=alpha_line_base_recompute,
                alpha_line_base_out_jvp_cached=alpha_line_base_out_jvp_cached,
                alpha_line_base_term_cached=alpha_line_base_term_cached,
                alpha_line_stop_reason=alpha_line_stop_reason,
                alpha_line_v_update_wall_seconds=alpha_line_v_update_wall_seconds,
                alpha_line_alpha_wall_seconds=alpha_line_alpha_wall_seconds,
                future_work=future_work_record,
                **_measurement_results_record,
                **({f"guard_{k}": v for k, v in {
                    "block_size": dynfreeze.block_size,
                    "block_layers": ",".join(str(l) for l in dynfreeze.frozen_block),
                    **{f"r_A_L{li}": h[-1] for li, h in dynfreeze._r_A_history.items() if h},
                }.items()} if dynfreeze is not None else {}),
                **({f"gold_{k}": v for k, v in gold_scores.items()} if gold_scores else {}),
            )

            if mlf.enabled:
                tg_metrics = {
                    "loss_train": pilot_loss_avg,
                    "loss_valid": final_valid_loss,
                    "cosine_sim": cos_sim,
                    "raw_delta_cosine_sim": raw_delta_cos_sim,
                    "velocity_predicted_consistency": predicted_consistency,
                    "velocity_short_long_norm_ratio": velocity_norm_ratio,
                    "proposal_N_before_cosine": proposal_N_before_cosine,
                    "proposal_N_selected": proposal.N,
                    "reduction_rate": cycle_state.reduction_rate,
                    "acceptance_rate": cycle_state.acceptance_rate,
                    "warmup_released": int(warmup_released),
                    "activation_cache_hit_rate": (
                        activation_cache_hit_count / activation_cache_eligible_count
                        if activation_cache_eligible_count > 0
                        else 0.0
                    ),
                    "activation_cache_hit_count": activation_cache_hit_count,
                    "activation_cache_eligible_count": activation_cache_eligible_count,
                    "velocity_magnitude": velocity.magnitudes[-1]
                    if velocity.magnitudes
                    else 0.0,
                    "magnitude_acceleration": velocity.magnitude_acceleration(),
                    "accel_action": controller.last_accel_action,
                    "delta_total_norm": delta_tracker.last_stats.total_norm
                    if delta_tracker.last_stats
                    else 0.0,
                    "convergence_trend": delta_tracker.convergence_trend(),
                    "extrap_cap_global_ratio": extrap_stats.global_cap_ratio,
                    "extrap_cap_mean_ratio": extrap_stats.mean_cap_ratio,
                    "extrap_cap_min_ratio": extrap_stats.min_cap_ratio,
                    "extrap_cap_capped_fraction": extrap_stats.capped_fraction,
                    "extrap_raw_update_norm": extrap_stats.raw_update_norm,
                    "extrap_applied_update_norm": extrap_stats.applied_update_norm,
                    "validation_forwards": validation_forwards_this_cycle,
                    "pilot_validation_forwards": pilot_validation_forwards_this_cycle,
                    "post_validation_forwards": post_validation_forwards_this_cycle,
                    "post_extrapolation_eval": int(post_extrapolation_eval),
                    "post_extrapolation_eval_skipped": int(
                        post_extrapolation_eval_skipped
                    ),
                    "subspace_zo_enabled": int(subspace_zo_enabled),
                    "subspace_zo_attempted_steps": zo_attempted_steps,
                    "subspace_zo_accepted_steps": zo_accepted_steps,
                    "subspace_zo_rejected_steps": zo_rejected_steps,
                    "subspace_zo_forward_count": zo_forward_count,
                    "subspace_zo_dim1_steps": zo_dim1_steps,
                    "subspace_zo_dim2_steps": zo_dim2_steps,
                    "subspace_zo_last_residual_norm": zo_last_stats.residual_norm,
                    "alpha_line_enabled": int(alpha_line_enabled),
                    "alpha_line_steps": alpha_line_step_count,
                    "alpha_line_base_recompute": alpha_line_base_recompute,
                    "alpha_line_v_update_wall_seconds": alpha_line_v_update_wall_seconds,
                    "alpha_line_alpha_wall_seconds": alpha_line_alpha_wall_seconds,
                }
                # Per-layer scores as individual metrics
                for layer_idx, score in controller.state.layer_scores.items():
                    tg_metrics[f"layer_score_{layer_idx}"] = score
                # Guard experiment metrics
                if dynfreeze is not None:
                    tg_metrics["guard_block_size"] = dynfreeze.block_size
                    for layer_idx, hist in dynfreeze._r_A_history.items():
                        if hist:
                            tg_metrics[f"guard_r_A_L{layer_idx}"] = hist[-1]
                mlf.log_metrics(tg_metrics, step=cycle)

            pbar.set_postfix(
                loss=f"{final_valid_loss:.4f}",
                acc="Y" if accepted else "N",
                cos=f"{cos_sim:.3f}",
                red=f"{cycle_state.reduction_rate:.1%}",
            )

            if cycle % cfg.logging.get("log_every_cycles", 1) == 0:
                logger.info(
                    _format_cycle_progress(
                        cycle,
                        final_valid_loss,
                        accepted,
                        cos_sim,
                        cycle_state.reduction_rate,
                        controller.state.K,
                        proposal.N,
                    )
                )

            _check_and_save_linearity_budget_checkpoint(
                model,
                tokenizer,
                valid_full_loader,
                input_device,
                cycle_state,
                grad_accum,
                triggered_target_steps,
                run_dir,
                logger,
                metrics,
            )

            # Full eval
            if is_full_eval_cycle:
                full_result = eval_loss_detailed(model, valid_full_loader, input_device)
                full_loss = full_result.avg_loss
                logger.info(
                    f"Cycle {cycle} full eval: loss={full_loss:.4f} "
                    f"ppl={full_result.perplexity:.2f} "
                    f"batches={full_result.num_batches} "
                    f"min={full_result.min_loss:.4f} max={full_result.max_loss:.4f}"
                )

                prev_best = cycle_state.best_loss
                prev_stale = cycle_state.stale_cycles
                cycle_state.record_full_eval(full_loss)
                metrics.record_full_eval_loss(
                    cycle=cycle,
                    full_loss=full_loss,
                    total_backward_passes=cycle_state.full_backward_passes,
                )

                is_new_best, should_stop, eval_reason = _evaluate_full_eval_outcome(
                    full_loss,
                    prev_best,
                    prev_stale,
                    early_stop_patience,
                    min_cycles_before_stop,
                    cycle_state.cycle,
                    early_stopping_min_delta,
                )
                if full_loss < best_full_eval_loss:
                    best_full_eval_loss = full_loss
                    best_full_eval_perplexity = full_result.perplexity
                    save_checkpoint(model, tokenizer, run_dir / "best_model")
                    mlf.log_artifact(run_dir / "best_model", "checkpoints")
                    logger.info(f"New best model: {full_loss:.4f}")

                if should_stop:
                    logger.info(f"Early stopping: {eval_reason} at cycle {cycle}")
                    break

                # LAWA eval: compare averaged vs current weights
                if lawa_averager is not None and lawa_averager.is_ready:
                    from src.tg_lora.weight_averaging import evaluate_with_lawa
                    lawa_loss, _ = evaluate_with_lawa(
                        model, lawa_averager,
                        lambda m: eval_loss_detailed(m, valid_full_loader, input_device).avg_loss,
                        current_loss=full_loss,
                    )
                    if lawa_loss is not None:
                        if lawa_loss < best_lawa_loss:
                            best_lawa_loss = lawa_loss
                        logger.info(
                            "LAWA eval cycle %d: lawa_loss=%.4f current_loss=%.4f delta=%+.4f",
                            cycle, lawa_loss, full_loss, lawa_loss - full_loss,
                        )
                        mlf.log_metrics({
                            "lawa_loss": lawa_loss,
                            "lawa_vs_current_delta": lawa_loss - full_loss,
                            "lawa_window_count": lawa_averager.count,
                        })

                # JSON generation-quality eval (structured-output domain task).
                # Batched greedy generation on a held-out set, scored by
                # field-F1/validity. Headline metric for plain/LAWA/PSA.
                if (
                    json_eval_records
                    and cycle % int(tg_cfg.get("json_eval_every_cycles", 10)) == 0
                ):
                    from src.eval.json_generation import generate_and_score_json
                    from src.tg_lora.weight_averaging import averaged_weights_context

                    # LAWA's headline quality is measured on the window-averaged
                    # (shadow) weights once the averager is ready; plain/PSA
                    # (averager None) eval on the current weights. The context
                    # manager restores the live weights on exit.
                    with averaged_weights_context(model, lawa_averager) as _lawa_active:
                        json_scores = generate_and_score_json(
                            model,
                            tokenizer,
                            json_eval_records,
                            batch_size=int(tg_cfg.get("json_eval_batch_size", 8)),
                            max_new_tokens=int(tg_cfg.get("json_eval_max_new_tokens", 96)),
                            device=input_device,
                        )
                    json_scores.pop("_preview", None)
                    json_row = {
                        "cycle": cycle,
                        "full_backward_passes": cycle_state.full_backward_passes,
                        "speculative_backward_passes": cycle_state.speculative_equivalent_backward_passes,
                        "reduction_rate": cycle_state.reduction_rate,
                        "lawa_averaged": bool(_lawa_active),
                        **{k: v for k, v in json_scores.items() if not k.startswith("_")},
                    }
                    with open(run_dir / "json_eval_log.jsonl", "a") as _jf:
                        _jf.write(json.dumps(json_row) + "\n")
                    if mlf.enabled:
                        mlf.log_metrics(
                            {f"json_{k}": v for k, v in json_scores.items()},
                            step=cycle,
                        )
                    logger.info(
                        "Cycle %d JSON eval: combined=%.3f field_f1=%.3f "
                        "exact=%.3f computed_acc=%.3f strict=%.3f valid=%.3f",
                        cycle,
                        json_scores["combined"],
                        json_scores["field_f1"],
                        json_scores["exact_match"],
                        json_scores["computed_accuracy"],
                        json_scores["strict_valid"],
                        json_scores["valid"],
                    )

            # Periodic save
            if cycle > 0 and cycle % cfg.logging.get("save_every_cycles", 25) == 0:
                checkpoint_dir = run_dir / f"checkpoint-cycle-{cycle}"
                ts = TrainingState(
                    cycle_state=cycle_state,
                    controller_state=controller.state,
                    velocity=velocity,
                    delta_tracker=delta_tracker,
                    cycle_offset=cycle_state.cycle,
                    adapter_checkpoint_dir=str(checkpoint_dir),
                    train_batch_position=train_batch_position,
                    accepted_valid_history=list(accepted_valid_history),
                    dynfreeze_state=dynfreeze.state_dict() if dynfreeze is not None else None,
                    best_full_eval_loss=best_full_eval_loss,
                    best_full_eval_perplexity=best_full_eval_perplexity,
                    warmup_released=warmup_released,
                    warmup_cos_consecutive=warmup_cos_consecutive,
                    lawa_state=lawa_averager.state_dict() if lawa_averager is not None else None,
                    best_lawa_loss=best_lawa_loss,
                    triggered_target_steps=sorted(triggered_target_steps),
                    act_regime_state=(
                        act_regime_tracker.state_dict()
                        if act_regime_tracker is not None
                        else None
                    ),
                    efficiency_accounting=_snapshot_efficiency_accounting(locals()),
                    psa_state=psa_prior.state_dict() if psa_prior is not None else None,
                )
                # Bound on-disk checkpoint growth (M10.3 disk-death guard): the
                # save -> training-state -> artifact -> prune sequence lives in
                # save_periodic_cycle_checkpoint, the unit-tested seam extracted
                # from this block. keep_last retains only the newest N
                # checkpoint-cycle-* dirs; min_free_disk_gb reclaims oldest-first
                # when the filesystem is low. Both default to 0 (off) so unrelated
                # runs are untouched; the M10 configs enable them so bounded
                # accumulation is the default for the autonomous run paths.
                removed = save_periodic_cycle_checkpoint(
                    model, tokenizer, checkpoint_dir, run_dir, cfg, ts,
                    log_artifact=mlf.log_artifact,
                )
                for d in removed:
                    logger.info("Pruned old checkpoint to bound disk: %s", d)
                # Bound trajectory-delta-artifact growth — the second vector of the
                # M10.3 disk-death class. The cycle guard above only matches
                # checkpoint-cycle-* dirs, but save_trajectory_delta_artifacts also
                # writes 1-2 .pt files per cycle into run_dir/trajectory_delta_artifacts/
                # and never removed old ones, so this run accumulated them linearly
                # despite keep_last_checkpoints being "on". Same knobs, same default-off.
                for f in prune_trajectory_delta_artifacts_from_cfg(cfg, run_dir):
                    logger.info("Pruned old trajectory artifact to bound disk: %s", f)

    except torch.cuda.OutOfMemoryError as _oom:
        from src.utils.device import is_gpu_oom_error

        if not is_gpu_oom_error(_oom):
            raise
        fault_reason = "oom"
        logger.warning(
            "GPU OOM at cycle %d — saving fault checkpoint", cycle_state.cycle
        )
    except NumericalInstabilityError as exc:
        fault_reason = "numerical_instability"
        logger.warning(
            "Numerical instability at cycle %d: %s — saving fault checkpoint",
            cycle_state.cycle,
            exc,
        )
    except RuntimeError as exc:
        if _is_cuda_error(exc):
            fault_reason = "cuda_error"
            logger.warning(
                "GPU error at cycle %d: %s — saving fault checkpoint",
                cycle_state.cycle,
                exc,
            )
        else:
            pbar.close()
            metrics.close()
            mlf.__exit__(type(exc), exc, exc.__traceback__)
            raise
    finally:
        if async_builder is not None:
            logger.info("Cleaning up async cache builder...")
            async_builder.join(timeout=60)
            async_builder = None
        if fault_reason is not None:
            _save_fault_checkpoint(
                model,
                tokenizer,
                controller,
                cycle_state,
                velocity,
                delta_tracker,
                run_dir,
                train_batch_position,
                accepted_valid_history,
                dynfreeze,
                best_full_eval_loss,
                best_full_eval_perplexity,
                warmup_released,
                warmup_cos_consecutive,
                lawa_averager,
                best_lawa_loss,
                triggered_target_steps,
                act_regime_tracker,
                efficiency_accounting=_snapshot_efficiency_accounting(locals()),
                psa_prior=psa_prior,
            )

    pbar.close()

    # Final full validation evaluation at the end of training
    logger.info("Computing final full validation loss...")
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()  # Free memory before final eval to prevent OOM
    try:
        full_result = eval_loss_detailed(model, valid_full_loader, input_device)
        full_loss = full_result.avg_loss
        logger.info(f"Final full validation loss: {full_loss:.4f} (ppl: {full_result.perplexity:.2f})")
        cycle_state.record_full_eval(full_loss)
        metrics.record_full_eval_loss(
            cycle=cycle_state.cycle,
            full_loss=full_loss,
            total_backward_passes=cycle_state.full_backward_passes,
        )
        if full_loss < best_full_eval_loss:
            best_full_eval_loss = full_loss
            best_full_eval_perplexity = full_result.perplexity
            save_checkpoint(model, tokenizer, run_dir / "best_model")
    except torch.cuda.OutOfMemoryError:
        logger.warning("Final full eval skipped: OOM on 12GB GPU. Training results are still valid.")
        cycle_state.record_full_eval(float("inf"))

    summary = build_training_summary(controller, cycle_state, delta_tracker)
    ideal_speedup_upper_bound = (
        1.0 / (1.0 - cycle_state.reduction_rate)
        if cycle_state.reduction_rate < 1.0
        else float("inf")
    )
    summary.update(
        {
            "activation_cache_build_count": activation_cache_build_count,
            "activation_cache_eligible_count": activation_cache_eligible_count,
            "activation_cache_hit_count": activation_cache_hit_count,
            "activation_cache_miss_count": activation_cache_miss_count,
            "activation_cache_hit_rate": (
                activation_cache_hit_count / activation_cache_eligible_count
                if activation_cache_eligible_count > 0
                else 0.0
            ),
            "accept_eval_examples": accept_eval_examples,
            "validation_skip_enabled": validation_skip_enabled,
            "validation_skip_high_cos": validation_skip_high_cos,
            "validation_skip_mid_cos": validation_skip_mid_cos,
            "validation_skip_mid_eval_every": validation_skip_mid_eval_every,
            "validation_skip_min_cycles": validation_skip_min_cycles,
            "validation_skip_min_acceptance_rate": validation_skip_min_acceptance_rate,
            "validation_skip_force_eval_N": validation_skip_force_eval_N,
            "validation_forwards_total": (
                pilot_validation_forward_count + post_validation_forward_count
            ),
            "pilot_validation_forwards": pilot_validation_forward_count,
            "post_validation_forwards": post_validation_forward_count,
            "post_extrapolation_eval_count": post_extrapolation_eval_count,
            "post_extrapolation_eval_skipped_count": post_extrapolation_eval_skipped_count,
            "post_extrapolation_eval_skip_reasons": dict(
                sorted(post_extrapolation_eval_skip_reasons.items())
            ),
            "subspace_zo_enabled": subspace_zo_enabled,
            "subspace_zo_tau_dim": subspace_zo_tau_dim,
            "subspace_zo_tau_cos": subspace_zo_tau_cos,
            "subspace_zo_stop_on_positive_g1": subspace_zo_stop_on_positive_g1,
            "subspace_zo_g1_stop_epsilon": subspace_zo_g1_stop_epsilon,
            "subspace_zo_attempted_steps_total": subspace_zo_attempted_steps_total,
            "subspace_zo_accepted_steps_total": subspace_zo_accepted_steps_total,
            "subspace_zo_rejected_steps_total": subspace_zo_rejected_steps_total,
            "subspace_zo_forward_count_total": subspace_zo_forward_count_total,
            "subspace_zo_dim1_steps_total": subspace_zo_dim1_steps_total,
            "subspace_zo_dim2_steps_total": subspace_zo_dim2_steps_total,
            "alpha_line_enabled": alpha_line_enabled,
            "alpha_line_order": alpha_line_order,
            "alpha_line_m_alpha_steps": alpha_line_m_steps,
            "alpha_line_alpha_lr": alpha_line_alpha_lr,
            "alpha_line_v_update_every": alpha_line_v_update_every,
            "alpha_line_max_consecutive_reject": (
                alpha_line_max_consecutive_reject
            ),
            "alpha_line_finite_diff_eps": alpha_line_finite_diff_eps,
            "alpha_line_future_work_metrics_enabled": (
                alpha_line_future_work_metrics_enabled
            ),
            "alpha_line_future_work_internal_metrics_enabled": (
                alpha_line_future_work_internal_metrics_enabled
            ),
            "alpha_line_steps_total": alpha_line_steps_total,
            "alpha_line_base_recompute_total": alpha_line_base_recompute_total,
            "alpha_line_base_out_jvp_cached": cycle_state.base_out_jvp_cached,
            "alpha_line_consecutive_rejects": cycle_state.consecutive_rejects,
            "alpha_line_interval_net_loss_delta": (
                cycle_state.interval_net_loss_delta
            ),
            "alpha_line_v_update_wall_seconds_total": (
                alpha_line_v_update_wall_seconds_total
            ),
            "alpha_line_alpha_wall_seconds_total": alpha_line_alpha_wall_seconds_total,
            "future_work": {
                "paper_scope": "motivation_only",
                "projection_ratio_count": len(future_work_projection_ratios),
                "projection_ratio_mean": (
                    sum(future_work_projection_ratios)
                    / len(future_work_projection_ratios)
                    if future_work_projection_ratios
                    else None
                ),
                "projection_ratio_values": future_work_projection_ratios or None,
                "internal": {
                    "paper_exclude": True,
                    "g_dot_v_loss_delta_pair_count": future_work_internal_pair_count,
                },
            },
            "ideal_speculative_speedup_upper_bound": ideal_speedup_upper_bound,
        }
    )
    if act_regime_tracker is not None:
        summary["activation_regime_inventory"] = act_regime_tracker.regime_inventory
        summary["activation_regime_stable_fraction"] = act_regime_tracker.stable_fraction
        act_regime_tracker.remove_hooks()
        # GOAL §7: null baseline comparison (random temporal shuffle)
        all_cosines = act_regime_tracker.summary().get("all_cosines", [])
        if all_cosines:
            null_baseline = compute_regime_null_baseline(all_cosines)
            summary["activation_regime_null_baseline"] = {
                "stable_fraction_z": null_baseline["stable_fraction_z"],
                "stable_fraction_null_mean": null_baseline["stable_fraction_null_mean"],
                "stable_fraction_null_std": null_baseline["stable_fraction_null_std"],
            }
    summary.update(prefix_feature_cache_summary)
    # GOAL §4 step 2: full rank-1 dominance analysis with Marchenko-Pastur null
    if psa_prior is not None and psa_prior.history_count >= 2:
        from src.tg_lora.layer_delta_analysis import (
            analyze_tensor_deltas,
            group_by_layer_type,
        )
        _deltas = psa_prior.delta_history
        _per_tensor = analyze_tensor_deltas(_deltas)
        if _per_tensor:
            _lt_groups = group_by_layer_type(_per_tensor)
            summary["layer_delta_analysis"] = {
                lt: {k: v for k, v in info.items() if k != "tensor_names"}
                for lt, info in _lt_groups.items()
            }
            summary["layer_delta_analysis_n_tensors"] = len(_per_tensor)
    if lawa_averager is not None:
        summary["lawa_snapshots_recorded"] = lawa_averager._recorded_count
        summary["lawa_window_size"] = lawa_averager.window_size
        if best_lawa_loss < float("inf"):
            summary["best_lawa_loss"] = best_lawa_loss
    if swap_cycle_vq is not None:
        summary["async_cache_swap_cycle_valid_quick"] = swap_cycle_vq
    if swap_cycle_vf is not None:
        summary["async_cache_swap_cycle_valid_full"] = swap_cycle_vf

    if progressive_freeze is not None:
        summary["progressive_freeze"] = {
            "enabled": True,
            "frozen_layer": progressive_freeze.frozen_layer_idx,
            "start_cycle": progressive_freeze._start_cycle,
        }

    if fault_reason is not None:
        summary["status"] = "failed"
        summary["reason"] = fault_reason

    metrics.write_footer(
        best_valid_loss=cycle_state.best_loss,
        best_valid_step=cycle_state.best_step,
        final_train_loss=cycle_state.last_train_loss,
        tg_lora_summary=summary,
        perplexity=best_full_eval_perplexity,
    )
    metrics.close()

    if mlf.enabled:
        mlf.log_metrics(
            {
                "best_valid_loss": cycle_state.best_loss,
                "final_train_loss": cycle_state.last_train_loss,
                "best_valid_perplexity": math.exp(cycle_state.best_loss)
                if math.isfinite(cycle_state.best_loss) and cycle_state.best_loss < 100
                else float("inf"),
            }
        )
    mlf.__exit__(None, None, None)

    if fault_reason is not None:
        logger.error("Training failed: %s at cycle %d", fault_reason, cycle_state.cycle)
        raise SystemExit(2)

    logger.info(f"Training complete. Summary: {summary}")


def main() -> None:
    import argparse

    from src.training.config_schema import load_validate_and_build_config
    from src.training.preflight import PreflightError, validate_training_prerequisites
    from src.utils.logging import setup_logging

    parser = argparse.ArgumentParser(description="Run TG-LoRA training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config value (e.g. --override tg_lora.K_initial=5)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to training_state.pt to resume training from a fault checkpoint",
    )
    args = parser.parse_args()

    validated, cfg = load_validate_and_build_config(args.config, args.override)
    try:
        validate_training_prerequisites(validated, args.config)
    except PreflightError as exc:
        raise SystemExit(f"Preflight check failed:\n{exc}") from exc

    setup_logging()
    train_tg_lora(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
