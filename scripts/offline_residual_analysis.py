import argparse
import logging
import os
import gc
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.model.load_model import load_base_model, load_tokenizer, apply_lora
from src.model.lora_utils import configure_trainable_lora_scope
from src.data.build_seed_dataset import load_dataset
from src.training.config_schema import load_validate_and_build_config
from src.training.trajectory_delta_artifact import load_trajectory_delta_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("offline-residual-analysis")


def flatten_tensor_dict(tensor_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    sorted_keys = sorted(tensor_dict.keys())
    return torch.cat([tensor_dict[k].flatten() for k in sorted_keys])


def unflatten_tensor_dict(flat_vector: torch.Tensor, template_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sorted_keys = sorted(template_dict.keys())
    restored = {}
    offset = 0
    for k in sorted_keys:
        shape = template_dict[k].shape
        numel = template_dict[k].numel()
        restored[k] = flat_vector[offset : offset + numel].view(shape).clone()
        offset += numel
    return restored


@torch.no_grad()
def apply_dict_delta_to_model(model, delta_dict):
    for name, p in model.named_parameters():
        if name in delta_dict:
            p.data.add_(delta_dict[name].to(p.device))


def compute_batch_gradient(model, dataloader, device, start_idx, end_idx, template_dict):
    model.zero_grad()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = None
            
    was_training = model.training
    model.train()
    
    # We use torch.enable_grad() to ensure gradient tracking is active
    with torch.enable_grad():
        for idx, batch in enumerate(dataloader):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
                
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss_scaled = loss / (end_idx - start_idx)
            loss_scaled.backward()
            
            # Explicit memory cleanup within loop
            del outputs, loss, loss_scaled, batch
            torch.cuda.empty_cache()
            gc.collect()
            
    grad_dict = {}
    for name, p in model.named_parameters():
        if name in template_dict:
            if p.grad is not None:
                # Move gradients to CPU to free GPU memory immediately
                grad_dict[name] = p.grad.clone().cpu()
            else:
                grad_dict[name] = torch.zeros_like(p).cpu()
                
    flat_grad = flatten_tensor_dict(grad_dict)
    
    if not was_training:
        model.eval()
        
    model.zero_grad()
    for p in model.parameters():
        p.grad = None
    torch.cuda.empty_cache()
    gc.collect()
    
    return flat_grad.double()


def compute_residual_stats(actual, prior, g_ascent, v_proj, name):
    v_proj = v_proj / v_proj.norm()
    
    proj_a = torch.dot(actual, v_proj) * v_proj
    proj_b = torch.dot(prior, v_proj) * v_proj
    
    delta_a = actual - proj_a
    delta_b = prior - proj_b
    
    norm_actual = actual.norm()
    norm_prior = prior.norm()
    norm_delta_a = delta_a.norm()
    norm_delta_b = delta_b.norm()
    
    ratio_a = (norm_delta_a / norm_actual).item()
    ratio_b = (norm_delta_b / norm_prior).item()
    
    norm_g = g_ascent.norm()
    
    cos_g_delta_a = 0.0
    if norm_delta_a > 1e-9:
        cos_g_delta_a = torch.dot(g_ascent, delta_a).item() / (norm_g.item() * norm_delta_a.item())
        
    cos_g_delta_b = 0.0
    if norm_delta_b > 1e-9:
        cos_g_delta_b = torch.dot(g_ascent, delta_b).item() / (norm_g.item() * norm_delta_b.item())
        
    cos_delta_a_b = 0.0
    if norm_delta_a > 1e-9 and norm_delta_b > 1e-9:
        cos_delta_a_b = torch.dot(delta_a, delta_b).item() / (norm_delta_a.item() * norm_delta_b.item())
    else:
        cos_delta_a_b = float('nan')
        
    return {
        "name": name,
        "norm_delta_a": norm_delta_a.item(),
        "norm_delta_b": norm_delta_b.item(),
        "ratio_a": ratio_a,
        "ratio_b": ratio_b,
        "cos_g_delta_a": cos_g_delta_a,
        "cos_g_delta_b": cos_g_delta_b,
        "cos_delta_a_b": cos_delta_a_b
    }


def main():
    parser = argparse.ArgumentParser(description="Offline residual analysis for actual and prior (w_traj * v0)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to past run trajectory_delta_artifacts folder")
    parser.add_argument("--gradients-dir", type=str, required=True, help="Path to collected true gradients folder")
    parser.add_argument("--target-cycle", type=int, default=7, help="Mid-training cycle to perform validation at")
    args = parser.parse_args()

    _, cfg = load_validate_and_build_config(args.config)

    logger.info("Loading model and dataset...")
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    trainable_lora_scope = cfg.training.get("trainable_lora_scope", "all")
    configure_trainable_lora_scope(model, trainable_lora_scope)

    valid_dataset = load_dataset(
        cfg.data.valid_quick_path,
        tokenizer,
        cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )
    collate_fn = getattr(valid_dataset, "collate_fn", None)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    input_device = next(model.parameters()).device

    # 1. Load delta artifacts
    delta_paths = sorted(Path(args.artifacts_dir).glob("*after_pilot*.pt"))
    deltas = []
    template_dict = None
    for p in delta_paths:
        art = load_trajectory_delta_artifact(p)
        if template_dict is None:
            template_dict = {k: v.clone() for k, v in art.delta_tensors.items()}
        flat_d = flatten_tensor_dict(art.delta_tensors).cpu() # Move to CPU immediately
        deltas.append(flat_d)
    deltas_stack = torch.stack(deltas)

    # Reconstruct parameters to target_cycle
    accept_history = [True, True, True, True, True, True, True, True, True, False, True, True, False, True, True]
    logger.info(f"Reconstructing parameters at cycle {args.target_cycle}...")
    for c in range(args.target_cycle):
        pilot_path = Path(args.artifacts_dir) / f"tg_lora_after_pilot_cycle_{c:06d}.pt"
        if pilot_path.exists():
            pilot_art = load_trajectory_delta_artifact(pilot_path)
            apply_dict_delta_to_model(model, pilot_art.delta_tensors)
        
        if accept_history[c]:
            spec_path = Path(args.artifacts_dir) / f"tg_lora_after_speculative_update_cycle_{c:06d}.pt"
            if spec_path.exists():
                spec_art = load_trajectory_delta_artifact(spec_path)
                apply_dict_delta_to_model(model, spec_art.delta_tensors)
        torch.cuda.empty_cache()
        gc.collect()

    original_state = {name: param.data.clone().cpu() for name, param in model.named_parameters() if name in template_dict}

    # Load 32-step cumulative update g_true
    grad_path = Path(args.gradients_dir) / f"gradient_step_{args.target_cycle + 1:06d}.pt"
    if not grad_path.exists():
        raise FileNotFoundError(f"Gradient artifact not found: {grad_path}")
    
    blob = torch.load(grad_path, map_location="cpu")
    g_true = flatten_tensor_dict(blob["gradients"]).cpu()

    # Double precision
    actual_double = deltas[args.target_cycle].double()
    g_true_double = g_true.double()
    
    norms = deltas_stack.norm(dim=1)
    w_traj = float(norms.median().item())
    mean_delta = deltas_stack.mean(dim=0)
    v0_double = (mean_delta / mean_delta.norm()).double()
    prior_double = w_traj * v0_double

    # 1. Static evaluation using g_ascent = -g_true
    g_ascent = -g_true_double

    # Evaluate the 3 patterns
    # Pattern A: Projection along TG direction v0
    stats_a = compute_residual_stats(actual_double, prior_double, g_ascent, v0_double, "Pattern A: Projection along TG direction v0")
    
    # Pattern B: Projection along True Update direction g_true
    v_gtrue = g_true_double / g_true_double.norm()
    stats_b = compute_residual_stats(actual_double, prior_double, g_ascent, v_gtrue, "Pattern B: Projection along True Update direction g_true")
    
    # Pattern C: Projection along mean direction of actual and prior
    v_mean = actual_double / actual_double.norm() + prior_double / prior_double.norm()
    v_mean = v_mean / v_mean.norm()
    stats_c = compute_residual_stats(actual_double, prior_double, g_ascent, v_mean, "Pattern C: Projection along mean direction of actual and prior")

    print("\n=== STATIC OFFLINE RESIDUAL ANALYSIS ===")
    for stats in [stats_a, stats_b, stats_c]:
        print(f"\n--- {stats['name']} ---")
        print(f"Residual norm: delta_a = {stats['norm_delta_a']:.4e}, delta_b = {stats['norm_delta_b']:.4e}")
        print(f"Norm ratio: ratio_a = {stats['ratio_a']:.4%}, ratio_b = {stats['ratio_b']:.4%}")
        print(f"cos(g, delta_a): {stats['cos_g_delta_a']:.4f}")
        print(f"cos(g, delta_b): {stats['cos_g_delta_b']:.4f}")
        print(f"cos(delta_a, delta_b): {stats['cos_delta_a_b']:.4f}")
    print("=========================================\n")

    # 2. Dynamic evaluation on 5 independent batches
    batch_size_slice = 64
    num_batches = 5
    slices = [(i * batch_size_slice, (i + 1) * batch_size_slice) for i in range(num_batches)]
    
    batch_stats = []
    logger.info("Computing gradients and checking residual alignment on 5 independent batches...")
    
    for i, (start, end) in enumerate(slices):
        logger.info(f"Batch {i+1}/5 (slice: {start} to {end})...")
        # Ensure parameters are at original state
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name].to(param.device))
                
        g_batch = compute_batch_gradient(model, valid_loader, input_device, start, end, template_dict)
        
        # We check alignment against the batch gradient g_batch (which is the loss ascent direction)
        # Pattern A
        stats_a_batch = compute_residual_stats(actual_double, prior_double, g_batch, v0_double, "Pattern A")
        # Pattern B
        stats_b_batch = compute_residual_stats(actual_double, prior_double, g_batch, v_gtrue, "Pattern B")
        # Pattern C
        stats_c_batch = compute_residual_stats(actual_double, prior_double, g_batch, v_mean, "Pattern C")
        
        batch_stats.append({
            "batch_idx": i + 1,
            "stats_a": stats_a_batch,
            "stats_b": stats_b_batch,
            "stats_c": stats_c_batch
        })

    # Restore final model state
    for name, param in model.named_parameters():
        if name in original_state:
            param.data.copy_(original_state[name].to(param.device))

    # Save Markdown Report
    report_path = Path("runs/offline_tg_w_validation_multi_batch/residual_analysis_report.md")
    with open(report_path, "w") as f:
        f.write("# Offline Residual Analysis Report\n\n")
        f.write(f"**Target Evaluation Cycle**: {args.target_cycle} (Mid-training)\n\n")
        
        f.write("## Overview of Comparison\n")
        f.write("- **Actual Delta**: Hand-tuned alpha extrapolation (`actual_delta`)\n")
        f.write("- **Prior Delta**: Extrapolated displacement using historical prior (`w_traj * v0`)\n\n")
        
        for name, key in [("Pattern A (Projection along TG direction $v_0$)", "stats_a"),
                           ("Pattern B (Projection along True Update direction $g_{true}$)", "stats_b"),
                           ("Pattern C (Projection along Mean of Actual and Prior)", "stats_c")]:
            f.write(f"### {name}\n\n")
            
            # Write static stats
            s_static = stats_a if key == "stats_a" else (stats_b if key == "stats_b" else stats_c)
            f.write("#### Static Analysis (against $g = -g_{true}$)\n")
            f.write(f"- **Residual Norms**: $\\|\\delta_a\\|$ = **{s_static['norm_delta_a']:.4e}**, $\\|\\delta_b\\|$ = **{s_static['norm_delta_b']:.4e}**\n")
            f.write(f"- **Residual Norm Ratio**: $\\|\\delta_a\\|/\\|actual\\|$ = **{s_static['ratio_a']:.2%}**, $\\|\\delta_b\\|/\\|prior\\|$ = **{s_static['ratio_b']:.2%}**\n")
            f.write(f"- **cos(g, $\\delta_a$)**: **{s_static['cos_g_delta_a']:.4f}**\n")
            f.write(f"- **cos(g, $\\delta_b$)**: **{s_static['cos_g_delta_b']:.4f}**\n")
            f.write(f"- **Residual cos($\\delta_a, \\delta_b$)**: **{s_static['cos_delta_a_b']:.4f}**\n\n")
            
            # Write batch stats
            f.write("#### Dynamic Multi-Batch Analysis (against $g_{batch}$)\n\n")
            f.write("| Batch | cos(g_batch, $\\delta_a$) | cos(g_batch, $\\delta_b$) | cos($\\delta_a, \\delta_b$) |\n")
            f.write("| --- | --- | --- | --- |\n")
            for bs in batch_stats:
                sb = bs[key]
                f.write(f"| {bs['batch_idx']} | {sb['cos_g_delta_a']:.4f} | {sb['cos_g_delta_b']:.4f} | {sb['cos_delta_a_b']:.4f} |\n")
            f.write("\n")
            
    logger.info(f"Residual analysis report written to {report_path}")


if __name__ == "__main__":
    main()
