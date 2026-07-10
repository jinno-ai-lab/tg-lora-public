import gc
import logging
import math
from collections.abc import Sized
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.build_seed_dataset import load_dataset
from src.eval.eval_loss import eval_loss, eval_loss_detailed
from src.model.load_model import (apply_lora, get_input_device,
                                  load_base_model, load_tokenizer)
from src.model.lora_utils import configure_trainable_lora_scope
from src.tg_lora.lora_state import snapshot_lora, snapshot_lora_delta
from src.training.batch_iter import InfiniteBatchIterator
from src.training.deterministic_batch_plan import (
    DeterministicBatchSampler, build_deterministic_batch_plan_for_dataset,
    build_trajectory_key)
from src.training.loss import has_supervised_tokens
from src.training.trainer_loop import (create_optimizer, create_scheduler,
                                       forward_backward, optimizer_step)
from src.training.trajectory_delta_artifact import (
    artifact_file_name, build_trajectory_delta_artifact_metadata,
    save_trajectory_delta_artifact)
from src.utils.checkpoint import (
    BaselineTrainingState,
    load_adapter_weights,
    load_baseline_training_state,
    prune_step_checkpoints_from_cfg,
    prune_trajectory_delta_artifacts_from_cfg,
    save_baseline_training_state,
    save_checkpoint,
)
from src.utils.logging import ensure_dir
from src.utils.memory import count_parameters
from src.utils.mlflow_logger import MLflowLogger
from src.utils.run_metrics import RunMetrics
from src.utils.seed import set_seed

logger = logging.getLogger("tg-lora")


def train_baseline(cfg: DictConfig, resume_path: str | None = None) -> None:
    set_seed(cfg.experiment.seed)

    from src.training.preflight import validate_max_seq_len

    logger.info("Loading tokenizer and model...")
    tokenizer = load_tokenizer(cfg)
    validate_max_seq_len(cfg, tokenizer)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    trainable_lora_scope = cfg.training.get("trainable_lora_scope", "all")
    active_names, active_indices = configure_trainable_lora_scope(
        model,
        trainable_lora_scope,
    )
    input_device = get_input_device(model)

    logger.info(
        "Configured baseline LoRA trainable scope: %s (layers=%s names=%d)",
        trainable_lora_scope,
        sorted(active_indices),
        len(active_names),
    )

    train_dataset = load_dataset(
        cfg.data.train_path,
        tokenizer,
        cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )
    valid_dataset = load_dataset(
        cfg.data.valid_quick_path,
        tokenizer,
        cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )
    deterministic_data_order = bool(
        getattr(cfg.training, "deterministic_data_order", True)
    )
    save_batch_plan_manifest = bool(
        getattr(cfg.training, "save_batch_plan_manifest", True)
    )
    batch_plan_manifest = build_deterministic_batch_plan_for_dataset(
        train_dataset,
        batch_size=cfg.training.batch_size,
    )

    train_collate_fn = (
        train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None
    )
    if deterministic_data_order:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DeterministicBatchSampler(batch_plan_manifest.epoch_batches),
            collate_fn=train_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            collate_fn=train_collate_fn,
        )
    eval_batch_size = cfg.eval.get("eval_batch_size", 16)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    optimizer = create_optimizer(
        model,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        num_training_steps=cfg.training.max_steps,
        warmup_steps=cfg.training.get("warmup_steps", 0),
        schedule_type=cfg.training.get("schedule_type", "linear"),
    )

    run_dir = ensure_dir(cfg.logging.run_dir)
    best_loss = float("inf")
    best_step = 0
    best_perplexity = None
    stale_steps = 0
    global_step = 0

    # --- Resume restoration ---
    if resume_path is not None:
        loaded = load_baseline_training_state(Path(resume_path))
        bs = loaded["state"]
        global_step = bs.global_step
        best_loss = bs.best_loss
        best_step = bs.best_step
        stale_steps = bs.stale_steps
        if loaded["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if loaded["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(loaded["scheduler_state_dict"])
        logger.info(
            "Resumed baseline training from step %d (best_loss=%.4f)",
            global_step,
            best_loss,
        )

    early_stop_patience = cfg.training.get("early_stopping_patience", None)
    min_steps_before_stop = cfg.training.get("min_steps_before_stop", 100)

    metrics = RunMetrics(run_dir, mode="baseline", append=resume_path is not None)
    batch_plan_manifest_path = run_dir / "batch_plan_manifest.json"
    if save_batch_plan_manifest:
        batch_plan_manifest.save(batch_plan_manifest_path)
    trajectory_key = build_trajectory_key(
        mode="baseline",
        epoch_batch_plan_key=batch_plan_manifest.epoch_batch_plan_key,
        trainable_lora_scope=trainable_lora_scope,
        optimizer_lifecycle=getattr(cfg.training, "optimizer_lifecycle", None),
        model_name=cfg.model.name_or_path,
        max_seq_len=cfg.data.max_seq_len,
        deterministic_data_order=deterministic_data_order,
    )

    logger.info("Computing initial eval loss for comparison reference...")
    original_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    free_cuda_mb = None
    if isinstance(input_device, str):
        input_device = torch.device(input_device)
    if input_device.type == "cuda":
        free_cuda_mb = torch.cuda.mem_get_info(input_device.index or 0)[0] / 1024**2

    if not cfg.eval.get("baseline_pretrain_reference_eval_enabled", True):
        logger.info("Skipping baseline comparison reference eval because it is disabled by config")
        initial_quick_valid_loss = None
    elif free_cuda_mb is not None and free_cuda_mb < 2300:
        logger.warning(
            "Skipping initial comparison reference eval because free CUDA memory is low: %.1f MiB",
            free_cuda_mb,
        )
        initial_quick_valid_loss = None
    else:
        if original_use_cache is not None:
            model.config.use_cache = False
        try:
            try:
                initial_quick_valid_loss = eval_loss(
                    model,
                    valid_loader,
                    input_device,
                    max_examples=cfg.eval.quick_eval_examples,
                )
            except RuntimeError as exc:
                logger.warning(
                    "Skipping initial comparison reference eval due to runtime error: %s",
                    exc,
                )
                initial_quick_valid_loss = None
                model.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
        finally:
            if original_use_cache is not None:
                model.config.use_cache = original_use_cache
    if isinstance(initial_quick_valid_loss, float) and not math.isfinite(initial_quick_valid_loss):
        logger.warning("Initial comparison reference eval returned non-finite loss; ignoring it")
        initial_quick_valid_loss = None
    if initial_quick_valid_loss is None:
        logger.info("Initial eval loss unavailable; comparison will fall back to post-hoc metrics")
    else:
        logger.info("Initial eval loss: %.4f", initial_quick_valid_loss)

    from src.utils.device import detect_device as _detect_device
    from src.utils.device import gpu_empty_cache, gpu_synchronize

    _device = _detect_device()
    gpu_empty_cache(_device)
    gpu_synchronize(_device)
    gpu_empty_cache(_device)

    metrics.write_header(
        cfg,
        budget_type="backward_passes",
        budget_value=cfg.training.max_steps,
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
            "loss": initial_quick_valid_loss,
            "max_examples": cfg.eval.quick_eval_examples,
        },
    )
    save_trajectory_delta_artifacts = bool(
        getattr(cfg.training, "save_trajectory_delta_artifacts", False)
    )
    trajectory_delta_artifact_interval = int(
        getattr(cfg.training, "trajectory_delta_artifact_interval", 1)
    )
    trajectory_delta_artifact_dir = run_dir / "trajectory_delta_artifacts"
    baseline_snapshot = snapshot_lora(model) if save_trajectory_delta_artifacts else None

    mlflow_cfg = cfg.logging.get("mlflow", {})
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
                    "trainable_lora_scope": trainable_lora_scope,
                    "max_steps": cfg.training.max_steps,
                    "seed": cfg.experiment.seed,
                    "schedule_type": cfg.training.get("schedule_type", "linear"),
                    "warmup_steps": cfg.training.get("warmup_steps", 0),
                }
            )
        except Exception:
            logger.warning("MLflow log_params failed, continuing without MLflow params")

        try:
            mlf.log_metrics({"initial_quick_valid_loss": initial_quick_valid_loss}, step=0)
        except Exception:
            logger.warning("MLflow initial metric logging failed, continuing")

    # Activation-fingerprint regime inventory (GOAL §4 step 1)
    act_regime_tracker = None
    if bool(cfg.training.get("activation_regime_enabled", False)):
        from src.tg_lora.activation_regime import (
            ActivationFingerprintTracker,
            compute_regime_null_baseline,
        )
        act_regime_tracker = ActivationFingerprintTracker(
            window=int(cfg.training.get("activation_regime_window", 10)),
            stable_threshold=float(cfg.training.get("activation_regime_stable_threshold", 0.95)),
            chaotic_threshold=float(cfg.training.get("activation_regime_chaotic_threshold", 0.5)),
            transition_drop_z=float(cfg.training.get("activation_regime_transition_drop_z", 2.0)),
            min_history=int(cfg.training.get("activation_regime_min_history", 3)),
        )
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

    logger.info("Starting baseline QLoRA training...")
    batch_iter = InfiniteBatchIterator(train_loader, input_device)
    train_batch_position = 0

    # --- Resume: restore adapter weights + advance batch iterator ---
    if resume_path is not None:
        loaded = load_baseline_training_state(Path(resume_path))
        bs = loaded["state"]
        train_batch_position = bs.train_batch_position
        if bs.adapter_checkpoint_dir is not None:
            from peft import set_peft_model_state_dict
            adapter_dir = Path(bs.adapter_checkpoint_dir)
            if adapter_dir.exists():
                # Route the baseline resume load through the integrity-checked
                # ``load_adapter_weights`` (load-before-apply, torn →
                # ``CheckpointIntegrityError``) — the symmetric counterpart to the
                # TG-LoRA resume seam's ``_restore_adapter_weights``. A bare
                # ``safetensors.load_file`` here would crash resume with an opaque
                # ``SafetensorError`` on a torn ``adapter_model.safetensors`` (the
                # costlier artifact to lose to a torn write), the very gap the
                # load-side integrity axis closed for the TG-LoRA path.
                adapter_state = load_adapter_weights(adapter_dir)
                set_peft_model_state_dict(model, adapter_state)
                logger.info("Restored adapter weights from %s", adapter_dir)
            else:
                logger.warning("Adapter checkpoint not found: %s — LoRA weights are fresh", adapter_dir)
        if train_batch_position > 0:
            batch_iter.advance(train_batch_position)
            logger.info("Advanced batch iterator to position %d", train_batch_position)
        pbar = tqdm(total=cfg.training.max_steps, desc="Training", initial=global_step)
    else:
        pbar = tqdm(total=cfg.training.max_steps, desc="Training")

    train_dataset_for_skip = train_loader.dataset
    if not isinstance(train_dataset_for_skip, Sized):
        raise TypeError("train_loader.dataset must be sized")
    max_empty_supervision_skips = len(train_dataset_for_skip)

    grad_accum = cfg.training.grad_accumulation
    try:
        while global_step < cfg.training.max_steps:
            step_loss = 0.0
            step_batch_keys: list[str] = []
            step_sample_keys: list[str] = []
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
                            "Exceeded empty-supervision skip budget while filling a baseline training step"
                        )
                    continue
                if deterministic_data_order:
                    batch_locator = batch_plan_manifest.batch_locator_at_position(
                        batch_position
                    )
                    step_batch_keys.append(batch_locator.batch_key)
                    step_sample_keys.extend(batch_locator.sample_keys)
                micro_loss = forward_backward(model, batch, grad_accum)
                step_loss += micro_loss
                valid_micro_batches += 1
            optimizer_step(optimizer, scheduler, model, cfg.training.max_grad_norm)

            loss = step_loss / grad_accum
            global_step += 1

            # Activation regime step (after optimizer step + forward hook captured)
            if act_regime_tracker is not None:
                act_regime_tracker.step()

            metrics.record_step(
                step=global_step,
                loss_train=loss,
                backward_passes=grad_accum,
                total_backward_passes=global_step * grad_accum,
            )

            if mlf.enabled:
                mlf.log_metrics({"loss_train": loss}, step=global_step)

            if (
                save_trajectory_delta_artifacts
                and baseline_snapshot is not None
                and global_step % trajectory_delta_artifact_interval == 0
            ):
                delta_tensors = snapshot_lora_delta(model, baseline_snapshot)
                metadata = build_trajectory_delta_artifact_metadata(
                    mode="baseline",
                    anchor_kind="after_optimizer_step",
                    trajectory_key=trajectory_key,
                    epoch_batch_plan_key=batch_plan_manifest.epoch_batch_plan_key,
                    batch_plan_manifest=(
                        str(batch_plan_manifest_path) if save_batch_plan_manifest else None
                    ),
                    dataset_key=batch_plan_manifest.dataset_key,
                    delta_tensors=delta_tensors,
                    step=global_step,
                    total_backward_passes=global_step * grad_accum,
                    batch_keys=step_batch_keys,
                    sample_keys=step_sample_keys,
                    extra_metadata={
                        "grad_accumulation": grad_accum,
                        "train_loss": loss,
                    },
                )
                save_trajectory_delta_artifact(
                    path=trajectory_delta_artifact_dir
                    / artifact_file_name(
                        mode="baseline",
                        anchor_kind="after_optimizer_step",
                        step=global_step,
                    ),
                    metadata=metadata,
                    delta_tensors=delta_tensors,
                )

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}", step=global_step)

            if global_step % cfg.logging.get("log_every_steps", 10) == 0:
                logger.info(f"Step {global_step}: loss={loss:.4f}")

            if global_step % cfg.eval.get("full_eval_every_steps", 250) == 0:
                # Free optimizer state + GPU cache before eval to prevent OOM on 12GB GPUs
                import gc as _gc
                if global_step >= cfg.training.max_steps:
                    del optimizer
                _gc.collect()
                torch.cuda.empty_cache()
                eval_result = eval_loss_detailed(
                    model,
                    valid_loader,
                    input_device,
                    max_examples=cfg.eval.quick_eval_examples,
                )
                valid_loss = eval_result.avg_loss
                logger.info(
                    f"Step {global_step}: valid_loss={valid_loss:.4f} "
                    f"ppl={eval_result.perplexity:.2f} "
                    f"min={eval_result.min_loss:.4f} max={eval_result.max_loss:.4f}"
                )

                metrics.record_step(
                    step=global_step,
                    loss_train=loss,
                    loss_valid=valid_loss,
                    backward_passes=grad_accum,
                    total_backward_passes=global_step * grad_accum,
                )

                if mlf.enabled:
                    mlf.log_metrics(
                        {
                            "loss_train": loss,
                            "loss_valid": valid_loss,
                            "perplexity": eval_result.perplexity,
                            "min_loss": eval_result.min_loss,
                            "max_loss": eval_result.max_loss,
                        },
                        step=global_step,
                    )

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_step = global_step
                    best_perplexity = eval_result.perplexity
                    stale_steps = 0
                    save_checkpoint(model, tokenizer, run_dir / "best_model")
                    mlf.log_artifact(run_dir / "best_model", "checkpoints")
                    logger.info(f"New best model saved to {run_dir / 'best_model'}")
                else:
                    stale_steps += 1

                if (
                    early_stop_patience is not None
                    and stale_steps >= early_stop_patience
                    and global_step >= min_steps_before_stop
                ):
                    logger.info(
                        f"Early stopping: stale={stale_steps}>=patience={early_stop_patience}"
                        f" at step {global_step}"
                    )
                    break

            if global_step % cfg.training.get("save_every_steps", 250) == 0:
                ckpt_dir = run_dir / f"checkpoint-{global_step}"
                save_checkpoint(model, tokenizer, ckpt_dir)
                mlf.log_artifact(ckpt_dir, "checkpoints")
                bs = BaselineTrainingState(
                    global_step=global_step,
                    best_loss=best_loss,
                    best_step=best_step,
                    stale_steps=stale_steps,
                    train_batch_position=train_batch_position,
                    adapter_checkpoint_dir=str(ckpt_dir),
                )
                try:
                    save_baseline_training_state(bs, optimizer, scheduler, ckpt_dir / "training_state.pt")
                except NameError:
                    pass  # optimizer deleted on final eval step
                # Bound on-disk growth (M10.3 disk-death guard) — mirrors the
                # TG-LoRA periodic-save path. The baseline writes
                # checkpoint-<step> dirs (the cycle regex never matches these)
                # and, when save_trajectory_delta_artifacts is on, one .pt per
                # step into trajectory_delta_artifacts/. Same knobs, same
                # default-off contract: a no-op until a baseline config opts in
                # via keep_last_checkpoints / min_free_disk_gb, preserving
                # today's unbounded behavior for the shipped baseline configs.
                for d in prune_step_checkpoints_from_cfg(cfg, run_dir):
                    logger.info("Pruned old checkpoint to bound disk: %s", d)
                for f in prune_trajectory_delta_artifacts_from_cfg(cfg, run_dir):
                    logger.info("Pruned old trajectory artifact to bound disk: %s", f)

        pbar.close()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_cuda_runtime = isinstance(exc, RuntimeError) and "CUDA" in str(exc)
        if isinstance(exc, torch.cuda.OutOfMemoryError) or is_cuda_runtime:
            logger.error("Fault during training: %s — saving checkpoint", exc)
            oom_dir = run_dir / "oom_checkpoint"
            try:
                save_checkpoint(model, tokenizer, oom_dir)
                bs = BaselineTrainingState(
                    global_step=global_step,
                    best_loss=best_loss,
                    best_step=best_step,
                    stale_steps=stale_steps,
                    train_batch_position=train_batch_position,
                    adapter_checkpoint_dir=str(oom_dir),
                )
                save_baseline_training_state(bs, optimizer, scheduler, oom_dir / "training_state.pt")
                logger.info("OOM checkpoint saved to %s", oom_dir)
            except Exception:
                logger.error("Failed to save OOM checkpoint", exc_info=True)
            # Producer half of the exit-code contract (AGENTS.md "Process exit
            # codes") — symmetric with train_tg_lora.py's fault path. A deferrable
            # GPU OOM (checkpoint saved, safe to resume at reduced batch) must emit
            # OOM_EXIT_CODE (3); a real CUDA fault that a smaller batch would still
            # reproduce must emit 2. Before this the handler bare-`raise`d the
            # original exception, so a *handled* baseline OOM exited 1 and was
            # keyable only off the log line — violating the documented contract.
            from src.utils.device import fault_exit_code, is_gpu_oom_error

            reason = "oom" if is_gpu_oom_error(exc) else "cuda_error"
            raise SystemExit(fault_exit_code(reason))
        raise

    # Activation regime inventory summary (GOAL §4 step 1)
    _baseline_summary: dict = {}
    if act_regime_tracker is not None:
        _baseline_summary["activation_regime_inventory"] = act_regime_tracker.regime_inventory
        _baseline_summary["activation_regime_stable_fraction"] = act_regime_tracker.stable_fraction
        act_regime_tracker.remove_hooks()
        all_cosines = act_regime_tracker.summary().get("all_cosines", [])
        if all_cosines:
            from src.tg_lora.activation_regime import compute_regime_null_baseline
            null_baseline = compute_regime_null_baseline(all_cosines)
            _baseline_summary["activation_regime_null_baseline"] = {
                "stable_fraction_z": null_baseline["stable_fraction_z"],
                "stable_fraction_null_mean": null_baseline["stable_fraction_null_mean"],
                "stable_fraction_null_std": null_baseline["stable_fraction_null_std"],
            }

    metrics.write_footer(
        best_valid_loss=best_loss,
        best_valid_step=best_step,
        final_train_loss=loss,
        perplexity=best_perplexity,
        tg_lora_summary=_baseline_summary if _baseline_summary else None,
    )
    metrics.close()

    if mlf.enabled:
        mlf.log_metrics(
            {
                "best_valid_loss": best_loss,
                "final_train_loss": loss,
                "best_valid_perplexity": math.exp(best_loss)
                if math.isfinite(best_loss) and best_loss < 100
                else float("inf"),
            }
        )
    mlf.__exit__(None, None, None)

    logger.info(f"Training complete. Best valid loss: {best_loss:.4f}")


def main() -> None:
    import argparse

    from src.training.config_schema import load_validate_and_build_config
    from src.training.preflight import (PreflightError,
                                        validate_training_prerequisites)
    from src.utils.logging import setup_logging

    parser = argparse.ArgumentParser(description="Run baseline QLoRA training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config value (e.g. --override training.learning_rate=5e-4)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to training_state.pt to resume training from a checkpoint",
    )
    args = parser.parse_args()

    validated, cfg = load_validate_and_build_config(args.config, args.override)
    try:
        validate_training_prerequisites(validated, args.config)
    except PreflightError as exc:
        raise SystemExit(f"Preflight check failed:\n{exc}") from exc

    setup_logging()
    train_baseline(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
