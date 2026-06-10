# MLX & CUDA Experimental Results Coordination Rules

This document defines the rules for output paths, data formats, and merging workflows to ensure experimental results from Apple Silicon (MLX) and Linux (CUDA) environments can be compared apples-to-apples.

## 1. Current Assessment (現状把握)

Currently, results from the Mac (MLX) environment are **not merged** into the primary results stored in this repository (`docs/paper_results_snapshot.md` or `runs/`). 

### Discrepancy in Formats
- **CUDA/Linux Environment**: 
  - Produces highly structured outputs under `runs/<experiment>/`.
  - Saves step-by-step training curves in `run_metrics.jsonl` (e.g., `step`, `loss`, `learning_rate`, `tokens_per_sec`, `elapsed_time`, `gpu_peak_mb`).
  - Saves final execution metrics in `summary.json` and `summary_break_even.json` (e.g., `wall_seconds`, `best_valid_loss`, `gpu_peak_mb`).
  - Evaluated using automated comparison script `scripts/compare_runs.py` and gate evaluator `scripts/evaluate_paper_gates.py`.
- **MLX/Mac Environment**:
  - `scripts/train_mlx_lora_fixed.py` only saves checkpoint weights (`adapters.safetensors`) and the input config file (`adapter_config.json`).
  - Training metrics (losses, learning rate, peak memory, throughput) are printed to the console standard output, but **no structured JSON/JSONL metrics logs are generated on disk**.

---

## 2. Structured Output Rules (保存場所とフォーマットの統一)

To resolve the discrepancy, the MLX training pipeline must output metrics using the exact same format as the CUDA pipeline.

### Directory Naming Convention
All experimental run directories must adhere to the following prefix structures:
- **CUDA Baseline/TG-LoRA**: `runs/paper_memory_one_shot_ms3_s1024_20260525/` or similar.
- **MLX Baseline/TG-LoRA**: `runs/mlx_qlora_<timestamp>_seed_<seed>/` (for full training) or `runs/mlx_smoke_<timestamp>_seed_<seed>/` (for smoke tests).

### JSON/JSONL Specifications
Every MLX run directory must output the following files to enable automated script compatibility.

#### `run_metrics.jsonl` (Step-level log)
Created dynamically during training. Every line must contain:
```json
{
  "step": 10,
  "loss": 2.103,
  "lr": 0.0002,
  "tokens_per_sec": 420.5,
  "elapsed_time": 12.4,
  "gpu_peak_mb": 24102.5
}
```
*Note: Peak memory for MLX should represent the peak Metal active memory in MB.*

#### `summary.json` (Run-level summary)
Created at the end of the run:
```json
{
  "seed": 42,
  "wall_seconds": 320.5,
  "best_valid_loss": 1.821,
  "tokens_per_sec": 412.3,
  "gpu_peak_mb": 26310.2,
  "max_seq_length": 2048,
  "total_steps": 1500
}
```

---

## 3. Implementation Plan for MLX Script

We will update [scripts/train_mlx_lora_fixed.py](file:///home/jinno/tg-lora/scripts/train_mlx_lora_fixed.py) to:
1. Append metrics to `run_metrics.jsonl` at each `--steps-per-report` interval.
2. Generate `summary.json` upon completion of `train_fixed`.

---

## 4. Merging Workflow (マージ手順)

1. **Execution on Mac**:
   Run training on Mac (e.g., `make train-mlx MLX_ITERS=1500`). This generates `runs/mlx_qlora_<timestamp>_seed_<seed>/` containing `adapters.safetensors`, `adapter_config.json`, `run_metrics.jsonl`, and `summary.json`.
2. **Transfer to Linux**:
   Transfer the generated run directory from Mac to the Linux host workspace under the `runs/` directory using rsync or scp:
   ```bash
   rsync -avz user@mac-host:/path/to/tg-lora/runs/mlx_qlora_* /home/jinno/tg-lora/runs/
   ```
3. **Automated Comparison**:
   Once stored under `runs/`, run comparisons natively on the Linux machine:
   ```bash
   make compare-mlx BASELINE_RUN=runs/qlora_9b_baseline_* MLX_RUN=runs/mlx_qlora_*
   ```
   This consolidates the findings into [reports/](file:///home/jinno/tg-lora/reports/) to be cited in the final paper.
