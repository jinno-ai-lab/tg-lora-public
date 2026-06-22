import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import orjson

from src.utils.device import (
    detect_device,
    gpu_info_dict,
    gpu_peak_memory_mb,
    gpu_reset_peak_stats,
)
from src.utils.memory import vram_usage_mb

_logger = logging.getLogger("tg-lora")


_PPL_MAX = 1e100


def _cfg_value(section, key: str, default=None):
    if section is None:
        return default
    if hasattr(section, "get"):
        return section.get(key, default)
    return getattr(section, key, default)


def _cfg_plain_mapping(section, key: str) -> dict | None:
    value = _cfg_value(section, key)
    if value is None:
        return None
    if hasattr(value, "items"):
        return {str(k): v for k, v in value.items()}
    return value


def _sanitize_perplexity(perplexity: float | None) -> float | None:
    if perplexity is None:
        return None
    if not math.isfinite(perplexity):
        _logger.warning("Non-finite perplexity %r sanitized to None", perplexity)
        return None
    if perplexity <= 0 or perplexity > _PPL_MAX:
        _logger.warning("Out-of-range perplexity %r sanitized to None", perplexity)
        return None
    return perplexity


class RunMetrics:
    def __init__(
        self,
        run_dir: str | Path,
        mode: Literal["baseline", "tg_lora"],
        run_id: str | None = None,
    ) -> None:
        if mode not in ("baseline", "tg_lora"):
            raise ValueError(f"mode must be 'baseline' or 'tg_lora', got {mode!r}")
        if isinstance(run_id, str) and not run_id:
            raise ValueError("run_id must be a non-empty string when provided")

        self._dir = Path(run_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode
        self._run_id = run_id or f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._start_time = time.perf_counter()
        self._gpu_peak_mb = 0.0
        self._path = self._dir / "run_metrics.jsonl"
        self._file = open(self._path, "wb")

        try:
            gpu_reset_peak_stats(detect_device())
        except RuntimeError as exc:
            _logger.warning("Skipping GPU peak reset: %s", exc)

    @property
    def run_id(self) -> str:
        return self._run_id

    def write_header(
        self,
        cfg,
        budget_type: str,
        budget_value: int,
        param_counts: dict[str, int] | None = None,
        comparison_keys: dict | None = None,
        comparison_reference: dict | None = None,
    ) -> None:
        gpu_name = ""
        gpu_total_mb = 0
        _info = gpu_info_dict()
        gpu_name = _info["name"]
        gpu_total_mb = _info["total_mb"] or 0
        training = getattr(cfg, "training", None)
        tg_lora = getattr(cfg, "tg_lora", None)
        alpha_line = getattr(cfg, "alpha_line", None)

        record = {
            "type": "run_header",
            "run_id": self._run_id,
            "mode": self._mode,
            "config_path": getattr(cfg, "_path", None),
            "compute_budget": {
                "budget_type": budget_type,
                "budget_value": budget_value,
            },
            "model_name": cfg.model.name_or_path,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.alpha,
            "batch_size": cfg.training.batch_size,
            "grad_accumulation": cfg.training.grad_accumulation,
            "learning_rate": cfg.training.learning_rate,
            "optimizer_lifecycle": getattr(cfg.training, "optimizer_lifecycle", None),
            "trainable_lora_scope": _cfg_value(training, "trainable_lora_scope"),
            "train_on_prompt": _cfg_value(training, "train_on_prompt", False),
            "prefix_feature_cache_experimental": _cfg_value(
                training, "prefix_feature_cache_experimental", False
            ),
            "prefix_feature_cache_train": _cfg_value(
                training, "prefix_feature_cache_train", False
            ),
            "prefix_feature_cache_valid_quick": _cfg_value(
                training, "prefix_feature_cache_valid_quick", False
            ),
            "prefix_feature_cache_valid_full": _cfg_value(
                training, "prefix_feature_cache_valid_full", False
            ),
            "prefix_feature_cache_mode": _cfg_value(
                training, "prefix_feature_cache_mode"
            ),
            "prefix_feature_cache_share_across_seeds": _cfg_value(
                training, "prefix_feature_cache_share_across_seeds", False
            ),
            "prefix_feature_cache_offload_prefix_to_cpu": _cfg_value(
                training, "prefix_feature_cache_offload_prefix_to_cpu", False
            ),
            "seed": cfg.experiment.seed,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "gpu_name": gpu_name,
            "gpu_total_memory_mb": round(gpu_total_mb, 1),
            "param_trainable": param_counts.get("trainable", 0) if param_counts else 0,
            "param_total": param_counts.get("total", 0) if param_counts else 0,
            "comparison_keys": comparison_keys,
            "comparison_reference": comparison_reference,
        }
        if tg_lora is not None:
            record.update(
                {
                    "tg_lora_K_initial": _cfg_value(tg_lora, "K_initial"),
                    "tg_lora_N_initial": _cfg_value(tg_lora, "N_initial"),
                    "tg_lora_alpha_initial": _cfg_value(tg_lora, "alpha_initial"),
                    "tg_lora_beta_initial": _cfg_value(tg_lora, "beta_initial"),
                    "tg_lora_lr_initial": _cfg_value(tg_lora, "lr_initial"),
                    "accel_instability_lr_decay": _cfg_value(
                        tg_lora, "accel_instability_lr_decay"
                    ),
                    "accel_convergence_lr_boost": _cfg_value(
                        tg_lora, "accel_convergence_lr_boost"
                    ),
                    "enable_random_walk": _cfg_value(
                        tg_lora, "enable_random_walk", False
                    ),
                    "enable_convergence_adaptation": _cfg_value(
                        tg_lora, "enable_convergence_adaptation", False
                    ),
                    "cosine_n_selection_enabled": _cfg_value(
                        tg_lora, "cosine_n_selection_enabled", False
                    ),
                    "cosine_n_selection_short_window": _cfg_value(
                        tg_lora, "cosine_n_selection_short_window"
                    ),
                    "cosine_n_selection_long_window": _cfg_value(
                        tg_lora, "cosine_n_selection_long_window"
                    ),
                    "cosine_n_selection_thresholds": _cfg_plain_mapping(
                        tg_lora, "cosine_n_selection_thresholds"
                    ),
                    "tg_lora_validation_skip_enabled": _cfg_value(
                        tg_lora, "validation_skip_enabled", False
                    ),
                    "tg_lora_validation_skip_high_cos": _cfg_value(
                        tg_lora, "validation_skip_high_cos"
                    ),
                    "tg_lora_validation_skip_mid_cos": _cfg_value(
                        tg_lora, "validation_skip_mid_cos"
                    ),
                    "tg_lora_validation_skip_mid_eval_every": _cfg_value(
                        tg_lora, "validation_skip_mid_eval_every"
                    ),
                    "tg_lora_validation_skip_force_eval_N": _cfg_value(
                        tg_lora, "validation_skip_force_eval_N"
                    ),
                    "tg_lora_subspace_zo_enabled": _cfg_value(
                        tg_lora, "subspace_zo_enabled", False
                    ),
                    "tg_lora_subspace_zo_tau_dim": _cfg_value(
                        tg_lora, "subspace_zo_tau_dim"
                    ),
                    "tg_lora_subspace_zo_tau_cos": _cfg_value(
                        tg_lora, "subspace_zo_tau_cos"
                    ),
                    "tg_lora_subspace_zo_max_steps_per_cycle": _cfg_value(
                        tg_lora, "subspace_zo_max_steps_per_cycle"
                    ),
                    "tg_lora_subspace_zo_stop_on_positive_g1": _cfg_value(
                        tg_lora, "subspace_zo_stop_on_positive_g1"
                    ),
                    "tg_lora_subspace_m9_enabled": _cfg_value(
                        tg_lora, "subspace_m9_enabled", False
                    ),
                    "tg_lora_warmup_release_cos": _cfg_value(
                        tg_lora, "warmup_release_cos"
                    ),
                    "tg_lora_warmup_release_count": _cfg_value(
                        tg_lora, "warmup_release_count"
                    ),
                    "progressive_freeze_enabled": _cfg_value(
                        tg_lora, "progressive_freeze_enabled", False
                    ),
                    "progressive_freeze_start_cycle": _cfg_value(
                        tg_lora, "progressive_freeze_start_cycle"
                    ),
                    "progressive_freeze_layer": _cfg_value(
                        tg_lora, "progressive_freeze_layer", "last_active"
                    ),
                }
            )
        if alpha_line is not None:
            record.update(
                {
                    "alpha_line_enabled": _cfg_value(
                        alpha_line, "alpha_line_enabled", False
                    ),
                    "alpha_line_order": _cfg_value(alpha_line, "alpha_line_order", 0),
                    "alpha_line_b_logical": _cfg_value(alpha_line, "b_logical"),
                    "alpha_line_b_heavy": _cfg_value(alpha_line, "b_heavy"),
                    "alpha_line_b_light": _cfg_value(alpha_line, "b_light"),
                    "alpha_line_m_alpha_steps": _cfg_value(
                        alpha_line, "m_alpha_steps"
                    ),
                    "alpha_line_alpha_init": _cfg_value(alpha_line, "alpha_init"),
                    "alpha_line_alpha_lr": _cfg_value(alpha_line, "alpha_lr"),
                    "alpha_line_v_update_every": _cfg_value(
                        alpha_line, "v_update_every"
                    ),
                    "alpha_line_max_consecutive_reject": _cfg_value(
                        alpha_line, "alpha_line_max_consecutive_reject"
                    ),
                    "alpha_line_finite_diff_eps": _cfg_value(
                        alpha_line, "alpha_line_finite_diff_eps"
                    ),
                    "alpha_line_future_work_metrics_enabled": _cfg_value(
                        alpha_line, "future_work_metrics_enabled", False
                    ),
                    "alpha_line_future_work_internal_metrics_enabled": _cfg_value(
                        alpha_line, "future_work_internal_metrics_enabled", False
                    ),
                }
            )
        self._write(record)

    def record_step(
        self,
        *,
        step: int,
        cycle: int | None = None,
        loss_train: float,
        loss_valid: float | None = None,
        loss_valid_full: float | None = None,
        backward_passes: int = 1,
        total_backward_passes: int,
        grad_norm: float | None = None,
        tg_lora_accepted: bool | None = None,
        tg_lora_cosine_sim: float | None = None,
        tg_lora_raw_delta_cosine_sim: float | None = None,
        tg_lora_predicted_consistency: float | None = None,
        tg_lora_short_long_norm_ratio: float | None = None,
        tg_lora_reduction_rate: float | None = None,
        tg_lora_K: int | None = None,
        tg_lora_N: int | None = None,
        tg_lora_proposed_N: int | None = None,
        tg_lora_alpha: float | None = None,
        tg_lora_beta: float | None = None,
        tg_lora_lr: float | None = None,
        tg_lora_cache_built: bool | None = None,
        tg_lora_cache_eligible: bool | None = None,
        tg_lora_cache_hit: bool | None = None,
        tg_lora_validation_forwards: int | None = None,
        tg_lora_pilot_validation_forwards: int | None = None,
        tg_lora_post_validation_forwards: int | None = None,
        tg_lora_post_extrapolation_eval: bool | None = None,
        tg_lora_post_extrapolation_eval_skipped: bool | None = None,
        tg_lora_post_extrapolation_eval_skip_reason: str | None = None,
        tg_lora_rollback_triggered: bool | None = None,
        tg_lora_loss_after: float | None = None,
        tg_lora_loss_pilot_eval: float | None = None,
        tg_lora_shadow_delta: float | None = None,
        tg_lora_shadow_cos: float | None = None,
        tg_lora_m9_alpha_fit: float | None = None,
        tg_lora_m9_beta1_fit: float | None = None,
        tg_lora_m9_beta2_fit: float | None = None,
        tg_lora_w_traj: float | None = None,
        tg_lora_cap_global_ratio: float | None = None,
        tg_lora_cap_mean_ratio: float | None = None,
        tg_lora_cap_min_ratio: float | None = None,
        tg_lora_cap_capped_fraction: float | None = None,
        tg_lora_cap_capped_tensors: int | None = None,
        tg_lora_cap_tensors: int | None = None,
        tg_lora_raw_update_norm: float | None = None,
        tg_lora_applied_update_norm: float | None = None,
        tg_lora_zo_enabled: bool | None = None,
        tg_lora_zo_attempted_steps: int | None = None,
        tg_lora_zo_accepted_steps: int | None = None,
        tg_lora_zo_rejected_steps: int | None = None,
        tg_lora_zo_forward_count: int | None = None,
        tg_lora_zo_dim1_steps: int | None = None,
        tg_lora_zo_dim2_steps: int | None = None,
        tg_lora_zo_last_dim: int | None = None,
        tg_lora_zo_last_residual_norm: float | None = None,
        tg_lora_zo_last_loss_initial: float | None = None,
        tg_lora_zo_last_loss_new: float | None = None,
        tg_lora_zo_last_g1: float | None = None,
        tg_lora_zo_last_h1: float | None = None,
        tg_lora_zo_last_g2: float | None = None,
        tg_lora_zo_last_h2: float | None = None,
        tg_lora_zo_last_stop_reason: str | None = None,
        alpha_line_enabled: bool | None = None,
        alpha_line_v_updated: bool | None = None,
        alpha_line_alpha_before: float | None = None,
        alpha_line_alpha_after: float | None = None,
        alpha_line_alpha_steps: int | None = None,
        alpha_line_grad_values: list[float] | None = None,
        alpha_line_losses: list[float] | None = None,
        alpha_line_exact_losses: list[float] | None = None,
        alpha_line_approx_errors: list[float] | None = None,
        alpha_line_jvp_methods: list[str] | None = None,
        alpha_line_order: int | None = None,
        alpha_line_finite_diff_eps: float | None = None,
        alpha_line_consecutive_rejects: int | None = None,
        alpha_line_interval_net_loss_delta: float | None = None,
        alpha_line_base_recompute: int | None = None,
        alpha_line_base_out_jvp_cached: bool | None = None,
        alpha_line_base_term_cached: bool | None = None,
        alpha_line_stop_reason: str | None = None,
        alpha_line_v_update_wall_seconds: float | None = None,
        alpha_line_alpha_wall_seconds: float | None = None,
        future_work: dict | None = None,
        is_step_aligned_full_eval: bool | None = None,
        aligned_target: int | None = None,
        measurement_noise_snr: float | None = None,
        measurement_noise_mean_grad_norm: float | None = None,
        measurement_noise_grad_var_norm: float | None = None,
        measurement_noise_pair_cos: float | None = None,
        measurement_per_step_delta_norms: list[float] | None = None,
        measurement_per_step_losses: list[float] | None = None,
        psa_enabled: bool | None = None,
        psa_prior_count: int | None = None,
        psa_gain_mean: float | None = None,
        psa_amplification_ratio: float | None = None,
        psa_regime: str | None = None,
        psa_regime_transitions: int | None = None,
        act_regime: str | None = None,
        act_stable_fraction: float | None = None,
        act_cosine_latest: float | None = None,
        act_cosine_mean: float | None = None,
        **extra_fields: float | str | bool | int | list | None,
    ) -> dict:
        elapsed = time.perf_counter() - self._start_time

        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        vram = vram_usage_mb()
        if vram:
            gpu_allocated_mb = vram.get("gpu0_allocated_mb", 0.0)
            gpu_reserved_mb = vram.get("gpu0_reserved_mb", 0.0)
        peak = gpu_peak_memory_mb(detect_device())
        if peak is not None and peak > self._gpu_peak_mb:
            self._gpu_peak_mb = peak

        record = {
            "type": "step",
            "run_id": self._run_id,
            "mode": self._mode,
            "step": step,
            "cycle": cycle,
            "elapsed_seconds": round(elapsed, 3),
            "loss_train": loss_train,
            "loss_valid": loss_valid,
            "backward_passes": backward_passes,
            "total_backward_passes": total_backward_passes,
            "gpu_allocated_mb": round(gpu_allocated_mb, 1),
            "gpu_reserved_mb": round(gpu_reserved_mb, 1),
            "gpu_peak_mb": round(self._gpu_peak_mb, 1),
            "grad_norm": grad_norm,
            "tg_lora_accepted": tg_lora_accepted,
            "tg_lora_cosine_sim": tg_lora_cosine_sim,
            "tg_lora_raw_delta_cosine_sim": tg_lora_raw_delta_cosine_sim,
            "tg_lora_predicted_consistency": tg_lora_predicted_consistency,
            "tg_lora_short_long_norm_ratio": tg_lora_short_long_norm_ratio,
            "tg_lora_reduction_rate": tg_lora_reduction_rate,
            "tg_lora_K": tg_lora_K,
            "tg_lora_N": tg_lora_N,
            "tg_lora_proposed_N": tg_lora_proposed_N,
            "tg_lora_alpha": tg_lora_alpha,
            "tg_lora_beta": tg_lora_beta,
            "tg_lora_lr": tg_lora_lr,
            "tg_lora_cache_built": tg_lora_cache_built,
            "tg_lora_cache_eligible": tg_lora_cache_eligible,
            "tg_lora_cache_hit": tg_lora_cache_hit,
            "tg_lora_validation_forwards": tg_lora_validation_forwards,
            "tg_lora_pilot_validation_forwards": tg_lora_pilot_validation_forwards,
            "tg_lora_post_validation_forwards": tg_lora_post_validation_forwards,
            "tg_lora_post_extrapolation_eval": tg_lora_post_extrapolation_eval,
            "tg_lora_post_extrapolation_eval_skipped": tg_lora_post_extrapolation_eval_skipped,
            "tg_lora_post_extrapolation_eval_skip_reason": tg_lora_post_extrapolation_eval_skip_reason,
            "tg_lora_rollback_triggered": tg_lora_rollback_triggered,
            "tg_lora_loss_after": tg_lora_loss_after,
            "tg_lora_loss_pilot_eval": tg_lora_loss_pilot_eval,
            "tg_lora_shadow_delta": tg_lora_shadow_delta,
            "tg_lora_shadow_cos": tg_lora_shadow_cos,
            "tg_lora_m9_alpha_fit": tg_lora_m9_alpha_fit,
            "tg_lora_m9_beta1_fit": tg_lora_m9_beta1_fit,
            "tg_lora_m9_beta2_fit": tg_lora_m9_beta2_fit,
            "tg_lora_w_traj": tg_lora_w_traj,
            "tg_lora_cap_global_ratio": tg_lora_cap_global_ratio,
            "tg_lora_cap_mean_ratio": tg_lora_cap_mean_ratio,
            "tg_lora_cap_min_ratio": tg_lora_cap_min_ratio,
            "tg_lora_cap_capped_fraction": tg_lora_cap_capped_fraction,
            "tg_lora_cap_capped_tensors": tg_lora_cap_capped_tensors,
            "tg_lora_cap_tensors": tg_lora_cap_tensors,
            "tg_lora_raw_update_norm": tg_lora_raw_update_norm,
            "tg_lora_applied_update_norm": tg_lora_applied_update_norm,
            "tg_lora_zo_enabled": tg_lora_zo_enabled,
            "tg_lora_zo_attempted_steps": tg_lora_zo_attempted_steps,
            "tg_lora_zo_accepted_steps": tg_lora_zo_accepted_steps,
            "tg_lora_zo_rejected_steps": tg_lora_zo_rejected_steps,
            "tg_lora_zo_forward_count": tg_lora_zo_forward_count,
            "tg_lora_zo_dim1_steps": tg_lora_zo_dim1_steps,
            "tg_lora_zo_dim2_steps": tg_lora_zo_dim2_steps,
            "tg_lora_zo_last_dim": tg_lora_zo_last_dim,
            "tg_lora_zo_last_residual_norm": tg_lora_zo_last_residual_norm,
            "tg_lora_zo_last_loss_initial": tg_lora_zo_last_loss_initial,
            "tg_lora_zo_last_loss_new": tg_lora_zo_last_loss_new,
            "tg_lora_zo_last_g1": tg_lora_zo_last_g1,
            "tg_lora_zo_last_h1": tg_lora_zo_last_h1,
            "tg_lora_zo_last_g2": tg_lora_zo_last_g2,
            "tg_lora_zo_last_h2": tg_lora_zo_last_h2,
            "tg_lora_zo_last_stop_reason": tg_lora_zo_last_stop_reason,
            "alpha_line_enabled": alpha_line_enabled,
            "alpha_line_v_updated": alpha_line_v_updated,
            "alpha_line_alpha_before": alpha_line_alpha_before,
            "alpha_line_alpha_after": alpha_line_alpha_after,
            "alpha_line_alpha_steps": alpha_line_alpha_steps,
            "alpha_line_grad_values": alpha_line_grad_values,
            "alpha_line_losses": alpha_line_losses,
            "alpha_line_exact_losses": alpha_line_exact_losses,
            "alpha_line_approx_errors": alpha_line_approx_errors,
            "alpha_line_jvp_methods": alpha_line_jvp_methods,
            "alpha_line_order": alpha_line_order,
            "alpha_line_finite_diff_eps": alpha_line_finite_diff_eps,
            "alpha_line_consecutive_rejects": alpha_line_consecutive_rejects,
            "alpha_line_interval_net_loss_delta": alpha_line_interval_net_loss_delta,
            "alpha_line_base_recompute": alpha_line_base_recompute,
            "alpha_line_base_out_jvp_cached": alpha_line_base_out_jvp_cached,
            "alpha_line_base_term_cached": alpha_line_base_term_cached,
            "alpha_line_stop_reason": alpha_line_stop_reason,
            "alpha_line_v_update_wall_seconds": alpha_line_v_update_wall_seconds,
            "alpha_line_alpha_wall_seconds": alpha_line_alpha_wall_seconds,
            "future_work": future_work,
            "is_step_aligned_full_eval": is_step_aligned_full_eval,
            "aligned_target": aligned_target,
            "measurement_noise_snr": measurement_noise_snr,
            "measurement_noise_mean_grad_norm": measurement_noise_mean_grad_norm,
            "measurement_noise_grad_var_norm": measurement_noise_grad_var_norm,
            "measurement_noise_pair_cos": measurement_noise_pair_cos,
            "measurement_per_step_delta_norms": measurement_per_step_delta_norms,
            "measurement_per_step_losses": measurement_per_step_losses,
            "psa_enabled": psa_enabled,
            "psa_prior_count": psa_prior_count,
            "psa_gain_mean": psa_gain_mean,
            "psa_amplification_ratio": psa_amplification_ratio,
            "psa_regime": psa_regime,
            "psa_regime_transitions": psa_regime_transitions,
            "act_regime": act_regime,
            "act_stable_fraction": act_stable_fraction,
            "act_cosine_latest": act_cosine_latest,
            "act_cosine_mean": act_cosine_mean,
            **extra_fields,
        }
        # §5.1/§5.2 honesty (pluggable-receiver pattern §6.2/§6.3):
        # ``loss_valid_full`` is the full-eval loss, recorded ONLY on full-eval
        # cycles. Its *presence* (not its value) is what the analyzer keys on —
        # ``extract_loss_and_time(full_eval_only=True)`` selects cycles by
        # ``LOSS_VALID_FULL_KEY in r`` — so it MUST be written conditionally:
        # absent on pilot-only cycles (byte-identical legacy records, so the
        # honesty contract stays dormant until honest data arrives) and present
        # only on full-eval cycles (which activates the receiver). Writing it
        # unconditionally as ``None`` would make every cycle look like a
        # full-eval cycle and silently break L*.
        if loss_valid_full is not None:
            record["loss_valid_full"] = loss_valid_full
        self._write(record)
        return record

    def write_footer(
        self,
        best_valid_loss: float,
        best_valid_step: int,
        final_train_loss: float,
        tg_lora_summary: dict | None = None,
        perplexity: float | None = None,
    ) -> None:
        elapsed = time.perf_counter() - self._start_time
        safe_ppl = _sanitize_perplexity(perplexity)
        record = {
            "type": "run_footer",
            "run_id": self._run_id,
            "mode": self._mode,
            "total_wall_seconds": round(elapsed, 1),
            "best_valid_loss": best_valid_loss,
            "best_valid_step": best_valid_step,
            "final_train_loss": final_train_loss,
            "gpu_peak_mb": round(self._gpu_peak_mb, 1),
            "tg_lora_summary": tg_lora_summary,
            "perplexity": safe_ppl,
        }
        self._write(record)

    def __enter__(self) -> "RunMetrics":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def _write(self, record: dict) -> None:
        self._file.write(orjson.dumps(record) + b"\n")
        self._file.flush()
