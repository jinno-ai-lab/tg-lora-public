#!/usr/bin/env python
"""MLX-LM QLoRA baseline training — RunMetrics-compatible output.

Produces run_metrics.jsonl in the same format as train_baseline_qlora.py
so that compare_runs.py works for MPS vs MLX comparison.

Usage:
    python scripts/train_mlx_lora.py --config configs/9b_baseline.yaml
    python scripts/train_mlx_lora.py --config configs/9b_baseline.yaml --model-override .cache/mlx_models/Qwen--Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as opt
from omegaconf import OmegaConf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX-LM QLoRA baseline training")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--model-override", default=None, help="Override model path")
    p.add_argument("--iters-override", type=int, default=None)
    return p.parse_args()


class MLXRunMetrics:
    """Lightweight RunMetrics-compatible JSONL writer for MLX runs."""

    def __init__(self, run_dir: str | Path, run_id: str):
        self._dir = Path(run_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._start = time.perf_counter()
        self._path = self._dir / "run_metrics.jsonl"
        self._file = open(self._path, "wb")
        self._gpu_peak_gb = 0.0

    def write_header(self, **kw) -> None:
        record = {"type": "run_header", "run_id": self._run_id, "mode": "baseline"}
        record.update(kw)
        self._write(record)

    def record_step(
        self,
        *,
        step: int,
        loss_train: float,
        loss_valid: float | None = None,
        backward_passes: int = 1,
        total_backward_passes: int = 0,
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
        tg_lora_post_extrapolation_eval: bool | None = None,
        tg_lora_rollback_triggered: bool | None = None,
        psa_regime: str | None = None,
        psa_regime_transitions: int | None = None,
        act_regime: str | None = None,
        act_stable_fraction: float | None = None,
        act_cosine_latest: float | None = None,
        act_cosine_mean: float | None = None,
        **kw,
    ) -> dict:
        elapsed = time.perf_counter() - self._start
        peak_gb = mx.get_peak_memory() / 1024**3
        self._gpu_peak_gb = max(self._gpu_peak_gb, peak_gb)
        record = {
            "type": "step",
            "run_id": self._run_id,
            "mode": "baseline",
            "step": step,
            "cycle": None,
            "elapsed_seconds": round(elapsed, 3),
            "loss_train": loss_train,
            "loss_valid": loss_valid,
            "backward_passes": backward_passes,
            "total_backward_passes": total_backward_passes,
            "gpu_allocated_mb": round(mx.get_active_memory() / 1024**2, 1),
            "gpu_reserved_mb": 0.0,
            "gpu_peak_mb": round(self._gpu_peak_gb * 1024, 1),
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
            "tg_lora_post_extrapolation_eval": tg_lora_post_extrapolation_eval,
            "tg_lora_rollback_triggered": tg_lora_rollback_triggered,
            "psa_regime": psa_regime,
            "psa_regime_transitions": psa_regime_transitions,
            "act_regime": act_regime,
            "act_stable_fraction": act_stable_fraction,
            "act_cosine_latest": act_cosine_latest,
            "act_cosine_mean": act_cosine_mean,
        }
        record.update(kw)
        self._write(record)
        return record

    def write_footer(
        self,
        *,
        best_valid_loss: float,
        best_valid_step: int,
        final_train_loss: float,
        perplexity: float | None = None,
        **kw,
    ) -> None:
        elapsed = time.perf_counter() - self._start
        record = {
            "type": "run_footer",
            "run_id": self._run_id,
            "mode": "baseline",
            "total_wall_seconds": round(elapsed, 1),
            "best_valid_loss": best_valid_loss,
            "best_valid_step": best_valid_step,
            "final_train_loss": final_train_loss,
            "gpu_peak_mb": round(self._gpu_peak_gb * 1024, 1),
            "perplexity": perplexity,
        }
        record.update(kw)
        self._write(record)

    def close(self):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def _write(self, record: dict):
        self._file.write(json.dumps(record, ensure_ascii=False).encode() + b"\n")
        self._file.flush()


class MetricsCallback:
    """Bridge between MLX-LM TrainingCallback and our RunMetrics."""

    def __init__(self, metrics: MLXRunMetrics, grad_accum: int):
        self.metrics = metrics
        self.grad_accum = grad_accum

    def on_train_loss_report(self, train_info: dict):
        step = train_info["iteration"]
        self.metrics.record_step(
            step=step,
            loss_train=train_info["train_loss"],
            backward_passes=self.grad_accum,
            total_backward_passes=step * self.grad_accum,
            tokens_per_second=train_info.get("tokens_per_second"),
        )

    def on_val_loss_report(self, val_info: dict):
        step = val_info["iteration"]
        self.metrics.record_step(
            step=step,
            loss_train=0.0,
            loss_valid=val_info["val_loss"],
            backward_passes=self.grad_accum,
            total_backward_passes=step * self.grad_accum,
        )


def _load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)

    # Config extraction
    model_name = args.model_override or cfg.model.name_or_path
    lora_r = cfg.lora.r
    lora_alpha = cfg.lora.alpha
    lora_dropout = cfg.lora.get("dropout", 0.0)
    grad_accum = cfg.training.grad_accumulation
    lr = float(cfg.training.learning_rate)
    seed = cfg.experiment.seed
    max_steps = args.iters_override or cfg.training.max_steps
    max_seq_len = cfg.data.max_seq_len
    eval_every = cfg.eval.get("full_eval_every_steps", 250)
    save_every = cfg.training.get("save_every_steps", 250)
    warmup_steps = cfg.training.get("warmup_steps", 0)
    schedule_type = cfg.training.get("schedule_type", "linear")

    train_path = cfg.data.train_path
    valid_path = cfg.data.valid_quick_path

    if not Path(train_path).exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Run: make download-data && make prepare-data")
        sys.exit(1)

    run_dir = (
        Path("runs")
        / f"mlx_{cfg.experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    mx.random.seed(seed)

    # Load model
    print(f"Loading model: {model_name}")
    from mlx_lm.utils import load

    model, tokenizer = load(model_name)

    # Apply LoRA
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    num_layers = len(model.layers) if hasattr(model, "layers") else 16
    lora_config = {
        "rank": lora_r,
        "dropout": lora_dropout,
        "scale": lora_alpha / lora_r,
    }
    linear_to_lora_layers(model, num_layers=num_layers, config=lora_config)

    model.freeze()
    model.unfreeze()

    trainable_params = sum(
        v.size for _, v in model.trainable_parameters().items() if hasattr(v, "size")
    )
    total_params = sum(
        v.size for _, v in model.parameters().items() if hasattr(v, "size")
    )
    print_trainable_parameters(model)

    # Optimizer + schedule
    if schedule_type == "cosine":
        if warmup_steps > 0:
            warmup = opt.schedulers.linear_schedule(0.0, lr, warmup_steps)
            main_sched = opt.schedulers.cosine_decay(lr)
            schedule = opt.schedulers.join_schedules(
                [warmup, main_sched], [warmup_steps]
            )
        else:
            schedule = opt.schedulers.cosine_decay(lr)
    else:
        if warmup_steps > 0:
            warmup = opt.schedulers.linear_schedule(0.0, lr, warmup_steps)
            main_sched = opt.schedulers.linear_schedule(lr, lr * 0.1, max_steps)
            schedule = opt.schedulers.join_schedules(
                [warmup, main_sched], [warmup_steps]
            )
        else:
            schedule = opt.schedulers.linear_schedule(lr, lr * 0.1, max_steps)

    optimizer = opt.AdamW(learning_rate=schedule)

    # Data
    train_data = _load_jsonl(train_path)
    valid_data = _load_jsonl(valid_path) if Path(valid_path).exists() else []

    from mlx_lm.tuner.datasets import CompletionsDataset, ChatDataset

    has_prompt = any("prompt" in r for r in train_data[:5])
    if has_prompt:
        train_ds = CompletionsDataset(
            train_data,
            tokenizer,
            prompt_key="prompt",
            completion_key="completion",
            mask_prompt=True,
        )
        valid_ds = (
            CompletionsDataset(
                valid_data,
                tokenizer,
                prompt_key="prompt",
                completion_key="completion",
                mask_prompt=True,
            )
            if valid_data
            else None
        )
    else:
        train_ds = ChatDataset(train_data, tokenizer)
        valid_ds = ChatDataset(valid_data, tokenizer) if valid_data else None

    # Metrics
    run_id = f"mlx_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = MLXRunMetrics(run_dir, run_id)
    metrics.write_header(
        config_path=None,
        compute_budget={"budget_type": "backward_passes", "budget_value": max_steps},
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=1,
        grad_accumulation=grad_accum,
        learning_rate=lr,
        optimizer_lifecycle=None,
        seed=seed,
        started_at=datetime.now(timezone.utc).isoformat(),
        gpu_name="Apple MLX",
        gpu_total_memory_mb=0,
        param_trainable=trainable_params,
        param_total=total_params,
        comparison_keys={"backend": "mlx"},
        tg_lora_params={
            "cosine_n_selection_enabled": False,
            "cosine_n_selection_short_window": None,
            "cosine_n_selection_long_window": None,
            "cosine_n_selection_thresholds": None,
        },
    )

    # Train using MLX-LM trainer
    from mlx_lm.tuner.trainer import train, TrainingArgs

    adapter_path = run_dir / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArgs(
        batch_size=1,
        iters=max_steps,
        val_batches=min(len(valid_data), cfg.eval.get("quick_eval_examples", 64)),
        steps_per_report=1,
        steps_per_eval=eval_every,
        steps_per_save=save_every,
        max_seq_length=max_seq_len,
        adapter_file=str(adapter_path / "adapters.safetensors"),
        grad_checkpoint=False,
        grad_accumulation_steps=grad_accum,
    )

    callback = MetricsCallback(metrics, grad_accum)

    print(f"\nStarting MLX QLoRA training: {max_steps} steps")
    print(f"LoRA: r={lora_r}, alpha={lora_alpha}, layers={num_layers}")
    print(f"Data: {len(train_data)} train, {len(valid_data)} valid")
    print(f"Run dir: {run_dir}\n")

    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=valid_ds,
        args=training_args,
        training_callback=callback,
    )

    # Find best validation loss from metrics
    best_loss = float("inf")
    best_step = 0
    final_loss = 0.0
    metrics.close()

    # Parse back metrics to find best
    total_tokens_per_sec = 0.0
    tokens_sec_count = 0
    with open(run_dir / "run_metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "step" and rec.get("loss_valid") is not None:
                if rec["loss_valid"] < best_loss:
                    best_loss = rec["loss_valid"]
                    best_step = rec["step"]
            if rec["type"] == "step":
                final_loss = rec.get("loss_train", final_loss)
                tps = rec.get("tokens_per_second")
                if tps is not None:
                    total_tokens_per_sec += tps
                    tokens_sec_count += 1

    avg_tokens_per_sec = (
        total_tokens_per_sec / tokens_sec_count if tokens_sec_count > 0 else 0.0
    )

    # Write footer
    metrics2 = MLXRunMetrics.__new__(MLXRunMetrics)
    metrics2._dir = run_dir
    metrics2._run_id = run_id
    metrics2._start = callback.metrics._start
    metrics2._gpu_peak_gb = callback.metrics._gpu_peak_gb
    metrics2._path = run_dir / "run_metrics.jsonl"
    metrics2._file = open(metrics2._path, "ab")

    ppl = math.exp(best_loss) if math.isfinite(best_loss) and best_loss < 100 else None
    metrics2.write_footer(
        best_valid_loss=best_loss,
        best_valid_step=best_step,
        final_train_loss=final_loss,
        perplexity=ppl,
    )
    metrics2.close()

    # Save best model copy
    if (adapter_path / "adapters.safetensors").exists():
        import shutil

        best_dir = run_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            adapter_path / "adapters.safetensors", best_dir / "adapters.safetensors"
        )
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(best_dir))

    # Generate summary.json
    total_wall_seconds = time.perf_counter() - callback.metrics._start
    gpu_peak_mb = callback.metrics._gpu_peak_gb * 1024
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f_sum:
        json.dump(
            {
                "seed": seed,
                "wall_seconds": round(total_wall_seconds, 1),
                "best_valid_loss": best_loss,
                "tokens_per_sec": round(avg_tokens_per_sec, 2),
                "gpu_peak_mb": round(gpu_peak_mb, 1),
                "max_seq_length": max_seq_len,
                "total_steps": max_steps,
            },
            f_sum,
            indent=2,
        )
    print(f"Summary written to {summary_path}")

    print(f"\nTraining complete. Best valid loss: {best_loss:.4f} at step {best_step}")
    print(f"Results: {run_dir}")
    print(f"Metrics: {run_dir / 'run_metrics.jsonl'}")


if __name__ == "__main__":
    main()
