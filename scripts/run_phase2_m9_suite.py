#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import time
import statistics
from pathlib import Path
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from scripts.run_paper_external_eval import _run_lm_eval

def run_cmd(cmd, env=None):
    print(f"\n[RUNNING] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, env=env, shell=isinstance(cmd, str))
    if res.returncode != 0:
        print(f"[ERROR] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    return res

def is_run_completed(run_dir):
    metrics_file = Path(run_dir) / "run_metrics.jsonl"
    if not metrics_file.exists():
        return False
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                if '"type":"run_footer"' in line or '"type": "run_footer"' in line:
                    return True
    except Exception:
        pass
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run TG-LoRA Phase 2 M9 Paper Suite with Step-aligned Checkpoints")
    parser.add_argument("--seeds", type=str, default="42 43 44", help="Space-separated list of seeds")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just collect and evaluate")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Batch size for down-stream evaluation")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    output_dir = repo_root / "runs" / "phase2_m9_results_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_config_path = repo_root / "configs/9b_baseline_suffix_only_last25.yaml"
    tg_config_path = repo_root / "configs/9b_tg_lora_m9.yaml"

    if not baseline_config_path.exists() or not tg_config_path.exists():
        print("ERROR: Config files missing.")
        sys.exit(1)

    print(f"=== TG-LoRA Phase 2 M9 Paper Suite (Step-Aligned) ===")
    print(f"Seeds: {seeds}")
    print(f"Output Directory: {output_dir}")
    print(f"=====================================================")

    env = os.environ.copy()
    target_steps = [250, 500, 750, 1000, 1250, 1500]
    
    # 1. Training loop
    if not args.skip_training:
        for seed in seeds:
            print(f"\n=======================================")
            print(f"  Starting Training for Seed {seed}")
            print(f"=======================================")
            
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            
            bl_run_dir = seed_dir / "baseline"
            tg_run_dir = seed_dir / "tg_lora"
            
            # --- Baseline QLoRA ---
            print(f"\n--- [1/2] Baseline QLoRA (Seed {seed}) ---")
            bl_cfg_path = seed_dir / "baseline_config.yaml"
            
            # Idempotency check: skip if baseline run completed successfully
            if is_run_completed(bl_run_dir):
                print(f"[SKIP] Baseline for seed {seed} already completed. Skipping training.")
            else:
                if bl_run_dir.exists():
                    shutil.rmtree(bl_run_dir)
                bl_cfg = OmegaConf.load(baseline_config_path)
                bl_cfg.experiment.seed = seed
                bl_cfg.experiment.name = f"baseline_seed_{seed}"
                bl_cfg.data.max_seq_len = 1024
                bl_cfg.eval.eval_batch_size = 1
                bl_cfg.logging.run_dir = str(bl_run_dir)
                bl_cfg.training.save_trajectory_delta_artifacts = True
                
                OmegaConf.save(bl_cfg, bl_cfg_path)
                
                run_cmd([
                    sys.executable, "-m", "src.training.train_baseline_qlora",
                    "--config", str(bl_cfg_path)
                ], env=env)
            
            # --- TG-LoRA M9 ---
            print(f"\n--- [2/2] TG-LoRA M9 (Seed {seed}) ---")
            tg_cfg_path = seed_dir / "tg_lora_config.yaml"
            
            # Idempotency check: skip if tg run completed successfully
            if is_run_completed(tg_run_dir):
                print(f"[SKIP] TG-LoRA for seed {seed} already completed. Skipping training.")
            else:
                resume_args = []
                last_checkpoint_dir = None
                last_cycle = -1
                
                if tg_run_dir.exists():
                    for p in tg_run_dir.glob("checkpoint-cycle-*"):
                        if p.is_dir():
                            try:
                                cycle_num = int(p.name.split("-")[-1])
                                if cycle_num > last_cycle:
                                    state_file = p / "training_state.pt"
                                    if state_file.exists():
                                        last_cycle = cycle_num
                                        last_checkpoint_dir = p
                            except ValueError:
                                pass
                
                if last_checkpoint_dir is not None:
                    state_file = last_checkpoint_dir / "training_state.pt"
                    print(f"[RESUME] Found existing checkpoint from cycle {last_cycle}. Resuming training...")
                    resume_args = ["--resume", str(state_file)]
                else:
                    if tg_run_dir.exists():
                        shutil.rmtree(tg_run_dir)
                
                tg_cfg = OmegaConf.load(tg_config_path)
                tg_cfg.experiment.seed = seed
                tg_cfg.experiment.name = f"tg_lora_m9_seed_{seed}"
                tg_cfg.data.max_seq_len = 1024
                tg_cfg.eval.eval_batch_size = 1
                tg_cfg.logging.run_dir = str(tg_run_dir)
                tg_cfg.training.save_trajectory_delta_artifacts = True
                tg_cfg.training.max_cycles = 120
                
                OmegaConf.save(tg_cfg, tg_cfg_path)
                
                cmd = [
                    sys.executable, "-m", "src.training.train_tg_lora",
                    "--config", str(tg_cfg_path)
                ]
                if resume_args:
                    cmd.extend(resume_args)
                
                run_cmd(cmd, env=env)

    # 2. Downstream Evaluation (lm-eval)
    print("\n=======================================")
    print(f"  Starting Downstream Evaluation")
    print(f"=======================================")
    
    tasks = ["arc_easy", "hellaswag", "truthfulqa_mc2"]
    base_model = "Qwen/Qwen3.5-9B"
    
    eval_results = {}
    
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        bl_run_dir = seed_dir / "baseline"
        tg_run_dir = seed_dir / "tg_lora"
        
        eval_results[str(seed)] = {
            "baseline": {},
            "tg_lora": {}
        }
        
        # Evaluate checkpoints at specific targets
        for target in target_steps:
            bl_ckpt = bl_run_dir / f"checkpoint-{target}"
            tg_ckpt = tg_run_dir / f"checkpoint-{target}"
            
            # Baseline checkpoint evaluation
            if bl_ckpt.exists():
                print(f"\nEvaluating Baseline Seed {seed} @ Step {target}...")
                bl_scores = _run_lm_eval(
                    base_model=base_model,
                    adapter_path=str(bl_ckpt),
                    tasks=tasks,
                    batch_size=args.eval_batch_size,
                )
                eval_results[str(seed)]["baseline"][str(target)] = bl_scores
            else:
                print(f"WARNING: Baseline checkpoint not found at {bl_ckpt}")
                
            # TG-LoRA checkpoint evaluation
            if tg_ckpt.exists():
                print(f"\nEvaluating TG-LoRA Seed {seed} @ Step {target}...")
                tg_scores = _run_lm_eval(
                    base_model=base_model,
                    adapter_path=str(tg_ckpt),
                    tasks=tasks,
                    batch_size=args.eval_batch_size,
                )
                eval_results[str(seed)]["tg_lora"][str(target)] = tg_scores
            else:
                print(f"WARNING: TG-LoRA checkpoint not found at {tg_ckpt}")

        # Also evaluate best_models
        bl_best = bl_run_dir / "best_model"
        tg_best = tg_run_dir / "best_model"
        
        if bl_best.exists():
            print(f"\nEvaluating Baseline Seed {seed} @ Best Model...")
            bl_scores = _run_lm_eval(
                base_model=base_model,
                adapter_path=str(bl_best),
                tasks=tasks,
                batch_size=args.eval_batch_size,
            )
            eval_results[str(seed)]["baseline"]["best_model"] = bl_scores
            
        if tg_best.exists():
            print(f"\nEvaluating TG-LoRA Seed {seed} @ Best Model...")
            tg_scores = _run_lm_eval(
                base_model=base_model,
                adapter_path=str(tg_best),
                tasks=tasks,
                batch_size=args.eval_batch_size,
            )
            eval_results[str(seed)]["tg_lora"]["best_model"] = tg_scores
            
    # Save raw evaluations
    eval_out_path = output_dir / "downstream_eval_results_aligned.json"
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\nAligned downstream evaluations saved to {eval_out_path}")

    # 3. Collect Metrics and Build Aligned Report
    print("\n=======================================")
    print(f"  Collecting Metrics and Summarizing")
    print(f"=======================================")
    
    summary_data = []
    
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        bl_run_dir = seed_dir / "baseline"
        tg_run_dir = seed_dir / "tg_lora"
        
        bl_metrics_path = bl_run_dir / "run_metrics.jsonl"
        tg_metrics_path = tg_run_dir / "run_metrics.jsonl"
        
        bl_steps_data = {}
        tg_steps_data = {}
        
        # Parse Baseline metrics step-by-step
        if bl_metrics_path.exists():
            with open(bl_metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    # Find exact target steps
                    step = data.get("step") if data.get("step") is not None else data.get("global_step")
                    if step in target_steps:
                        val_loss = data.get("loss_valid")
                        if val_loss is None:
                            val_loss = data.get("valid_loss")
                        if val_loss is None:
                            val_loss = data.get("best_valid_loss")
                        
                        if val_loss is not None or str(step) not in bl_steps_data:
                            bl_steps_data[str(step)] = {
                                "valid_loss": val_loss if val_loss is not None else float("nan"),
                                "actual_backward_passes": step * 8
                            }
                            
        # Parse TG-LoRA metrics and extract step-aligned full evaluations
        if tg_metrics_path.exists():
            with open(tg_metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("is_step_aligned_full_eval", False):
                        target = data.get("aligned_target")
                        val_loss = data.get("loss_valid")
                        actual_bp = data.get("total_backward_passes", 0)
                        red_rate = data.get("tg_lora_reduction_rate", 0.0)
                        if val_loss is not None:
                            tg_steps_data[str(target)] = {
                                "valid_loss": val_loss,
                                "actual_backward_passes": actual_bp,
                                "reduction_rate": red_rate
                            }
                    elif "total_backward_passes" in data or "tg_lora_reduction_rate" in data:
                        # Fallback for runs without step-aligned metrics (e.g. legacy/original runs)
                        actual_bp = data.get("total_backward_passes", 0)
                        red_rate = data.get("tg_lora_reduction_rate", 0.0)
                        total_bp_equiv = int(actual_bp / (1.0 - red_rate)) if red_rate < 1.0 else actual_bp
                        equiv_steps = total_bp_equiv // 8
                        for target in target_steps:
                            if equiv_steps >= target and str(target) not in tg_steps_data:
                                tg_steps_data[str(target)] = {
                                    "valid_loss": data.get("loss_valid", float("nan")),
                                    "actual_backward_passes": actual_bp,
                                    "reduction_rate": red_rate
                                }

        # Handle final state for best_model
        # (Using the best model record from the metrics end)
        bl_best_val = float("nan")
        if bl_metrics_path.exists():
            with open(bl_metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "best_valid_loss" in data:
                        bl_best_val = data["best_valid_loss"]
                        
        tg_best_val = float("nan")
        tg_final_metrics = {}
        if tg_metrics_path.exists():
            with open(tg_metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "best_valid_loss" in data or "loss_valid" in data:
                        tg_final_metrics = data
        tg_best_val = tg_final_metrics.get("best_valid_loss", float("nan"))
        
        summary_data.append({
            "seed": seed,
            "baseline_best_valid_loss": bl_best_val,
            "tg_best_valid_loss": tg_best_val,
            "baseline_final_backward_passes": 12000,
            "tg_final_backward_passes": tg_final_metrics.get("total_backward_passes", 0),
            "baseline_steps": bl_steps_data,
            "tg_steps": tg_steps_data,
            "downstream": eval_results.get(str(seed), {})
        })

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to {summary_path}")

    # Build Aligned Report Markdown
    lines = []
    lines.append("# TG-LoRA Phase 2 M9 Paper Aligned Report")
    lines.append("")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. Quality Comparison: Valid Loss Aligned by Data Digestion (Epochs)")
    lines.append("")
    lines.append("Comparing Baseline and TG-LoRA at the exact same equivalent step checkpoints (valid dataset size: 493).")
    lines.append("")
    
    # Headers: Target steps & conditions
    header = "| Step | Epoch | Seed 42 BL | Seed 42 TG | Seed 43 BL | Seed 43 TG | Seed 44 BL | Seed 44 TG | Mean BL | Mean TG |"
    lines.append(header)
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    
    for target in target_steps:
        epoch = target * 8 / 4000
        row_cells = [f"{target}", f"{epoch:.2f}"]
        
        bl_losses = []
        tg_losses = []
        
        for seed in seeds:
            s_data = next(r for r in summary_data if r["seed"] == seed)
            bl_l = s_data["baseline_steps"].get(str(target), {}).get("valid_loss", float("nan"))
            tg_l = s_data["tg_steps"].get(str(target), {}).get("valid_loss", float("nan"))
            
            bl_losses.append(bl_l)
            tg_losses.append(tg_l)
            
            row_cells.append(f"{bl_l:.4f}" if bl_l == bl_l else "N/A")
            row_cells.append(f"{tg_l:.4f}" if tg_l == tg_l else "N/A")
            
        # Means
        bl_mean = statistics.mean([l for l in bl_losses if l == l]) if any(l == l for l in bl_losses) else float("nan")
        tg_mean = statistics.mean([l for l in tg_losses if l == l]) if any(l == l for l in tg_losses) else float("nan")
        
        row_cells.append(f"{bl_mean:.4f}" if bl_mean == bl_mean else "N/A")
        row_cells.append(f"{tg_mean:.4f}" if tg_mean == tg_mean else "N/A")
        
        lines.append("| " + " | ".join(row_cells) + " |")
        
    lines.append("")
    
    lines.append("## 2. Efficiency Comparison: Cumulative Actual Backward Passes")
    lines.append("")
    lines.append("Cumulative actual backward passes (including all pilot, reject, and rollback costs) required to reach each data digestion target.")
    lines.append("")
    
    header_eff = "| Step | Epoch | Seed 42 BL | Seed 42 TG | Seed 43 BL | Seed 43 TG | Seed 44 BL | Seed 44 TG | Mean Reduction |"
    lines.append(header_eff)
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    
    for target in target_steps:
        epoch = target * 8 / 4000
        row_cells = [f"{target}", f"{epoch:.2f}"]
        
        reds = []
        for seed in seeds:
            s_data = next(r for r in summary_data if r["seed"] == seed)
            bl_bp = s_data["baseline_steps"].get(str(target), {}).get("actual_backward_passes", target * 8)
            tg_bp = s_data["tg_steps"].get(str(target), {}).get("actual_backward_passes", float("nan"))
            
            row_cells.append(f"{bl_bp}")
            if tg_bp == tg_bp:
                row_cells.append(f"{tg_bp}")
                red = 1.0 - (float(tg_bp) / bl_bp)
                reds.append(red)
            else:
                row_cells.append("N/A")
                
        mean_red = statistics.mean(reds) if reds else float("nan")
        row_cells.append(f"{mean_red*100:.2f}%" if mean_red == mean_red else "N/A")
        
        lines.append("| " + " | ".join(row_cells) + " |")
        
    lines.append("")

    lines.append("## 3. Aligned Downstream Quality Evaluation (G3 Gate Tasks)")
    lines.append("")
    for task in tasks:
        lines.append(f"### Task: {task.upper()}")
        lines.append("")
        lines.append("| Step | Epoch | Seed 42 BL | Seed 42 TG | Seed 43 BL | Seed 43 TG | Seed 44 BL | Seed 44 TG | Mean BL | Mean TG |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        
        for target in target_steps:
            epoch = target * 8 / 4000
            row_cells = [f"{target}", f"{epoch:.2f}"]
            
            bl_scores = []
            tg_scores = []
            
            for seed in seeds:
                seed_ds = eval_results.get(str(seed), {})
                bl_s = seed_ds.get("baseline", {}).get(str(target), {}).get(task, float("nan"))
                tg_s = seed_ds.get("tg_lora", {}).get(str(target), {}).get(task, float("nan"))
                
                bl_scores.append(bl_s)
                tg_scores.append(tg_s)
                
                row_cells.append(f"{bl_s:.4f}" if bl_s == bl_s else "N/A")
                row_cells.append(f"{tg_s:.4f}" if tg_s == tg_s else "N/A")
                
            bl_mean = statistics.mean([s for s in bl_scores if s == s]) if any(s == s for s in bl_scores) else float("nan")
            tg_mean = statistics.mean([s for s in tg_scores if s == s]) if any(s == s for s in tg_scores) else float("nan")
            
            row_cells.append(f"{bl_mean:.4f}" if bl_mean == bl_mean else "N/A")
            row_cells.append(f"{tg_mean:.4f}" if tg_mean == tg_mean else "N/A")
            
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")
        
    report_md_path = output_dir / "report_aligned.md"
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown aligned report generated at {report_md_path}")
    print("\n=== Aligned Phase 2 Tasks completed successfully ===")

if __name__ == "__main__":
    main()
