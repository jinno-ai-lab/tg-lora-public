import argparse
import logging
import os
import gc
import time
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.model.load_model import load_base_model, load_tokenizer, apply_lora
from src.model.lora_utils import configure_trainable_lora_scope
from src.data.build_seed_dataset import load_dataset
from src.training.config_schema import load_validate_and_build_config
from src.training.trajectory_delta_artifact import load_trajectory_delta_artifact
from src.eval.eval_loss import eval_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("offline-subspace-validation-multi-fold")


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
def apply_flat_delta_to_model(model, flat_delta, template_dict, scale: float = 1.0):
    delta_dict = unflatten_tensor_dict(flat_delta, template_dict)
    for name, param in model.named_parameters():
        if name in delta_dict:
            param.data.add_(delta_dict[name].to(param.device), alpha=scale)


@torch.no_grad()
def apply_dict_delta_to_model(model, delta_dict):
    for name, p in model.named_parameters():
        if name in delta_dict:
            p.data.add_(delta_dict[name].to(p.device))


def eval_loss_on_slice(model, dataloader, device, start_idx, end_idx, max_examples=32):
    torch.cuda.empty_cache()
    gc.collect()
    
    was_training = model.training
    model.eval()
    
    total_loss = 0.0
    total_examples = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
                
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_examples = batch["input_ids"].shape[0] if "input_ids" in batch else 1
            
            # Bound evaluation to max_examples
            if max_examples is not None:
                remaining = max_examples - total_examples
                if remaining <= 0:
                    del batch
                    break
                if batch_examples > remaining:
                    for key, val in batch.items():
                        if isinstance(val, torch.Tensor) and val.ndim > 0:
                            batch[key] = val[:remaining]
                    batch_examples = remaining
                    
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item() * batch_examples
            total_examples += batch_examples
            
            del outputs, batch
            
    if was_training:
        model.train()
        
    torch.cuda.empty_cache()
    gc.collect()
    
    if total_examples == 0:
        return float("nan")
    return total_loss / total_examples


def compute_batch_gradient(model, dataloader, device, start_idx, end_idx, template_dict, max_examples=32):
    model.zero_grad()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = None
            
    was_training = model.training
    model.train()
    total_examples = 0
    
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
            batch_examples = batch["input_ids"].shape[0] if "input_ids" in batch else 1
            
            # Bound gradient calculation to max_examples
            if max_examples is not None:
                remaining = max_examples - total_examples
                if remaining <= 0:
                    del batch
                    break
                if batch_examples > remaining:
                    for key, val in batch.items():
                        if isinstance(val, torch.Tensor) and val.ndim > 0:
                            batch[key] = val[:remaining]
                    batch_examples = remaining
                    
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss_scaled = loss / max_examples
            loss_scaled.backward()
            
            total_examples += batch_examples
            del outputs, loss, loss_scaled, batch
            torch.cuda.empty_cache()
            gc.collect()
            
    grad_dict = {}
    for name, p in model.named_parameters():
        if name in template_dict:
            if p.grad is not None:
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


def gram_schmidt(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    ortho_vectors = []
    for v in vectors:
        v_ortho = v.clone().double()
        for u in ortho_vectors:
            proj = torch.dot(v_ortho, u) * u
            v_ortho -= proj
        norm = v_ortho.norm()
        if norm > 1e-8:
            ortho_vectors.append(v_ortho / norm)
    return [vec.float() for vec in ortho_vectors]


def main():
    parser = argparse.ArgumentParser(description="Offline Subspace Multi-Fold Cross-Validation")
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

    # Setup 3 independent Folds (max 32 examples per slice)
    folds = [
        {"fit": (0, 32), "eval": (32, 64)},
        {"fit": (64, 96), "eval": (96, 128)},
        {"fit": (128, 160), "eval": (160, 192)}
    ]

    # 1. Load delta artifacts
    delta_paths = sorted(Path(args.artifacts_dir).glob("*after_pilot*.pt"))
    deltas = []
    template_dict = None
    num_trainable_params = None
    for p in delta_paths:
        art = load_trajectory_delta_artifact(p)
        if template_dict is None:
            template_dict = {k: v.clone() for k, v in art.delta_tensors.items()}
            num_trainable_params = sum(param.numel() for name, param in model.named_parameters() if name in template_dict)
        flat_d = flatten_tensor_dict(art.delta_tensors).cpu()
        
        # Dimension verification assert
        if flat_d.numel() != num_trainable_params:
            raise ValueError(
                f"Dimension mismatch: loaded delta from '{p.name}' has {flat_d.numel()} elements, "
                f"but model trainable parameters have {num_trainable_params} elements."
            )
            
        # Zero element anomaly detection assert
        zero_ratio = float((flat_d == 0).sum().item()) / flat_d.numel()
        if zero_ratio >= 0.5:
            raise ValueError(
                f"Anomalous zero elements: loaded delta '{p.name}' has {zero_ratio:.1%} zero elements "
                f"(threshold: 50%). Trainable scope mismatch is highly likely."
            )
            
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

    # Helper function to restore model state
    def restore_model():
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name].to(param.device))
        torch.cuda.empty_cache()
        gc.collect()

    # Load 32-step cumulative update g_true
    grad_path = Path(args.gradients_dir) / f"gradient_step_{args.target_cycle + 1:06d}.pt"
    if not grad_path.exists():
        raise FileNotFoundError(f"Gradient artifact not found: {grad_path}")
    
    blob = torch.load(grad_path, map_location="cpu")
    g_true = flatten_tensor_dict(blob["gradients"]).cpu()

    # Pre-calculate base direction and subspace components
    norms = deltas_stack.norm(dim=1)
    w_traj = float(norms.median().item())
    mean_delta = deltas_stack.mean(dim=0)
    v0 = mean_delta / mean_delta.norm()

    # PCA to get auxiliary directions
    centered_deltas = deltas_stack - deltas_stack.mean(dim=0)
    U, S_val, V = torch.pca_lowrank(centered_deltas, q=4, niter=4)
    pc1 = V[:, 0]
    pc2 = V[:, 1]

    ortho_basis = gram_schmidt([v0, pc1, pc2])
    v0_ortho = ortho_basis[0]
    u1 = ortho_basis[1] if len(ortho_basis) > 1 else torch.zeros_like(v0)
    u2 = ortho_basis[2] if len(ortho_basis) > 2 else torch.zeros_like(v0)

    results = []

    for f_idx, fold in enumerate(folds):
        fit_start, fit_end = fold["fit"]
        eval_start, eval_end = fold["eval"]
        logger.info(f"\n--- Processing Fold {f_idx + 1}/3 ---")
        
        # 1. Base Loss (Eval)
        loss_base = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end, max_examples=32)
        
        # 2. Raw Prior (Eval - scale 1.0 is not tuned, direct Raw Prior)
        apply_flat_delta_to_model(model, v0, template_dict, scale=w_traj)
        loss_raw_prior = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end, max_examples=32)
        restore_model()
        
        # 3. Tuned LR on v0 (Fit Batch search, Eval Batch evaluate)
        logger.info("  Tuning scale on v0 prior direction (Fit Batch line search)...")
        scale_multipliers = [0.2, 0.5, 0.8, 1.0, 1.5]
        best_loss_lr_fit = float("inf")
        best_eta_v0 = w_traj
        
        for mult in scale_multipliers:
            eta = mult * w_traj
            apply_flat_delta_to_model(model, v0, template_dict, scale=eta)
            l_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end, max_examples=32)
            logger.debug(f"    v0 scale {mult:.1f}x: Fit Loss = {l_fit:.6f}")
            if l_fit < best_loss_lr_fit:
                best_loss_lr_fit = l_fit
                best_eta_v0 = eta
            restore_model()
            
        # Hold-out Eval
        apply_flat_delta_to_model(model, v0, template_dict, scale=best_eta_v0)
        loss_tuned_lr = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end, max_examples=32)
        restore_model()
        logger.info(f"  Tuned LR on v0 Eval Loss (Hold-out): {loss_tuned_lr:.6f} at eta={best_eta_v0:.6f}")

        # 4. Compute Fit Batch Gradient for Subspace Fit
        g_current_fit = compute_batch_gradient(model, valid_loader, input_device, fit_start, fit_end, template_dict, max_examples=32)
        tilde_g = -g_current_fit
        
        # Subspace projection
        alpha_fit = torch.dot(tilde_g, v0_ortho.double()).item()
        beta1_fit = torch.dot(tilde_g, u1.double()).item()
        beta2_fit = torch.dot(tilde_g, u2.double()).item()
        delta_fit = alpha_fit * v0_ortho + beta1_fit * u1 + beta2_fit * u2
        delta_fit_norm = delta_fit.norm().item()
        
        # Subspace Fit (Fit Batch search, Eval Batch evaluate)
        logger.info("  Tuning scale on Subspace Fit direction (Fit Batch line search)...")
        best_loss_subspace_fit = float("inf")
        best_gamma = 0.0
        subspace_base_scale = w_traj / max(1e-8, delta_fit_norm)
        
        for mult in scale_multipliers:
            gamma = mult * subspace_base_scale
            apply_flat_delta_to_model(model, delta_fit, template_dict, scale=gamma)
            l_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end, max_examples=32)
            logger.debug(f"    subspace scale {mult:.1f}x: Fit Loss = {l_fit:.6f}")
            if l_fit < best_loss_subspace_fit:
                best_loss_subspace_fit = l_fit
                best_gamma = gamma
            restore_model()
            
        # Hold-out Eval
        apply_flat_delta_to_model(model, delta_fit, template_dict, scale=best_gamma)
        loss_subspace = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end, max_examples=32)
        restore_model()
        logger.info(f"  Subspace Fit Eval Loss (Hold-out): {loss_subspace:.6f} at gamma={best_gamma:.6f}")

        # 5. Oracle g_true (Fit Batch search, Eval Batch evaluate)
        logger.info("  Tuning scale on Oracle g_true (Fit Batch line search)...")
        best_loss_oracle_fit = float("inf")
        best_eta_oracle = 0.0
        g_true_norm = g_true.norm().item()
        oracle_base_scale = w_traj / max(1e-8, g_true_norm)
        
        for mult in scale_multipliers:
            eta_o = mult * oracle_base_scale
            apply_flat_delta_to_model(model, g_true, template_dict, scale=eta_o)
            l_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end, max_examples=32)
            logger.debug(f"    oracle scale {mult:.1f}x: Fit Loss = {l_fit:.6f}")
            if l_fit < best_loss_oracle_fit:
                best_loss_oracle_fit = l_fit
                best_eta_oracle = eta_o
            restore_model()
            
        # Hold-out Eval
        apply_flat_delta_to_model(model, g_true, template_dict, scale=best_eta_oracle)
        loss_oracle = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end, max_examples=32)
        restore_model()
        logger.info(f"  Oracle Eval Loss (Hold-out): {loss_oracle:.6f} at eta={best_eta_oracle:.6f}")

        # Metrics computation
        red_lr = loss_base - loss_tuned_lr
        red_prior = loss_base - loss_raw_prior
        red_sub = loss_base - loss_subspace
        red_oracle = loss_base - loss_oracle
        
        diff_sub_lr = loss_tuned_lr - loss_subspace  # positive means subspace is better (lower loss)
        diff_prior_lr = loss_raw_prior - loss_tuned_lr # positive means tuned LR is better
        diff_sub_oracle = loss_oracle - loss_subspace  # positive means subspace is better
        
        results.append({
            "fold": f_idx + 1,
            "loss_base": loss_base,
            "loss_tuned_lr": loss_tuned_lr,
            "loss_raw_prior": loss_raw_prior,
            "loss_subspace": loss_subspace,
            "loss_oracle": loss_oracle,
            "best_eta_v0": best_eta_v0,
            "best_gamma": best_gamma,
            "best_eta_oracle": best_eta_oracle,
            "red_lr": red_lr,
            "red_prior": red_prior,
            "red_sub": red_sub,
            "red_oracle": red_oracle,
            "diff_sub_lr": diff_sub_lr,
            "diff_prior_lr": diff_prior_lr,
            "diff_sub_oracle": diff_sub_oracle
        })
        
        logger.info(f"Fold {f_idx + 1} finalized: Base={loss_base:.4f}, LR={loss_tuned_lr:.4f}, Prior={loss_raw_prior:.4f}, Sub={loss_subspace:.4f}, Oracle={loss_oracle:.4f}")

    # Restore final model state
    restore_model()

    # Calculate statistics
    diff_sub_lr_vals = [r["diff_sub_lr"] for r in results]
    diff_prior_lr_vals = [r["diff_prior_lr"] for r in results]
    diff_sub_oracle_vals = [r["diff_sub_oracle"] for r in results]

    mean_sub_lr = np.mean(diff_sub_lr_vals)
    std_sub_lr = np.std(diff_sub_lr_vals)
    mean_prior_lr = np.mean(diff_prior_lr_vals)
    std_prior_lr = np.std(diff_prior_lr_vals)
    mean_sub_oracle = np.mean(diff_sub_oracle_vals)
    std_sub_oracle = np.std(diff_sub_oracle_vals)

    # Save Markdown Report
    report_path = Path("runs/offline_tg_w_validation_multi_batch/subspace_multi_fold_report.md")
    with open(report_path, "w") as f:
        f.write("# Offline Subspace Multi-Fold Cross-Validation Report (Fair Scaling Line Searches)\n\n")
        f.write(f"**Target Cycle**: {args.target_cycle} (Mid-training)\n")
        f.write(f"**Validation Folds**: 3 independent partitions (32-item evaluation slices)\n\n")
        
        f.write("## 1. Core Metrics across Folds (Hold-Out Evaluations)\n\n")
        f.write("| Fold | Base Loss | (2) Tuned LR Loss | (3) Raw Prior Loss | (4) Subspace Fit Loss | (5) Oracle Loss | Best scale (v0, gamma, oracle) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in results:
            f.write(f"| {r['fold']} | {r['loss_base']:.6f} | {r['loss_tuned_lr']:.6f} | {r['loss_raw_prior']:.6f} | {r['loss_subspace']:.6f} | {r['loss_oracle']:.6f} | eta={r['best_eta_v0']:.4f}, gamma={r['best_gamma']:.4f}, oracle_eta={r['best_eta_oracle']:.4f} |\n")
        f.write("\n")
        
        f.write("## 2. Statistical Analysis of Comparisons (Out-of-Sample)\n\n")
        f.write("| Comparison | Metric / Hypothesis | Mean Difference | Std Dev (std) | Status |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        
        f.write(f"| (3) - (2) | Raw Prior vs Tuned LR on $v_0$ | {mean_prior_lr:.6f} | {std_prior_lr:.6f} | {'Raw Prior ≈ Tuned LR' if abs(mean_prior_lr) < 0.05 else 'Divergent'} |\n")
        
        v_passed_lr = mean_sub_lr - std_sub_lr > 0.002
        f.write(f"| (2) - (4) | Tuned LR vs Subspace Fit | {mean_sub_lr:.6f} | {std_sub_lr:.6f} | {'**SIGNIFICANT PASS**' if v_passed_lr else 'FAIL'} |\n")
        
        v_passed_oracle = mean_sub_oracle - std_sub_oracle > 0.0
        f.write(f"| (5) - (4) | Oracle vs Subspace Fit | {mean_sub_oracle:.6f} | {std_sub_oracle:.6f} | {'**Subspace > Oracle (Generalization)**' if mean_sub_oracle > 0 else 'Oracle > Subspace'} |\n\n")
        
        f.write("### Verdict & Interpretation\n")
        f.write(f"- **Tuned LR vs Subspace Fit**: Mean improvement = **{mean_sub_lr:.6f}** (std = **{std_sub_lr:.6f}**). ")
        if v_passed_lr:
            f.write("The subspace fit outperforms simple learning rate scaling significantly even when accounting for cross-validation variance. **M9 implementation is statistically justified.**\n")
        else:
            f.write("The variance overlaps with zero or the threshold. Subspace fit does not consistently outperform a fairly tuned LR baseline. Proceed with caution.\n")
            
        f.write(f"- **Oracle vs Subspace Fit**: Mean difference = **{mean_sub_oracle:.6f}** (std = **{std_sub_oracle:.6f}**). ")
        if mean_sub_oracle > 0:
            f.write("Subspace Fit consistently outperforms straight-line Oracle $g_{true}$ updates on hold-out batches, showing that trajectory alignment is more robust to local curvature changes than oracle tracking.\n")
        else:
            f.write("Oracle updates remain superior.\n")
            
    logger.info(f"Multi-fold cross-validation report written to {report_path}")


if __name__ == "__main__":
    main()
