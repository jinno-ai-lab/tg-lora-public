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
logger = logging.getLogger("offline-subspace-validation")


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


def eval_loss_on_slice(model, dataloader, device, start_idx, end_idx):
    """Evaluate loss on a specific slice of the dataset."""
    torch.cuda.empty_cache()
    gc.collect()
    
    was_training = model.training
    model.eval()
    
    total_loss = 0.0
    count = 0
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
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item() * batch_examples
            count += 1
            total_examples += batch_examples
            
    if was_training:
        model.train()
        
    torch.cuda.empty_cache()
    gc.collect()
    
    if total_examples == 0:
        return float("nan")
    return total_loss / total_examples


def compute_batch_gradient(model, dataloader, device, start_idx, end_idx, template_dict):
    """Compute local gradient vector on a specific data slice."""
    model.zero_grad()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = None
            
    was_training = model.training
    model.train()
    
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
    parser = argparse.ArgumentParser(description="Offline Subspace Validation (Hold-out evaluation)")
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

    # Setup Fit (0-64) and Eval (64-128) slices
    fit_start, fit_end = 0, 64
    eval_start, eval_end = 64, 128

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

    # Load 32-step cumulative update g_true (Oracle reference)
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

    # 1. Base loss evaluation (at W0)
    loss_base_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end)
    loss_base_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
    logger.info(f"Base Loss (W0): Fit Batch = {loss_base_fit:.6f}, Eval Batch = {loss_base_eval:.6f}")

    # ----------------------------------------------------
    # (2) Tuned LR on v0 (Hold-out evaluation)
    # ----------------------------------------------------
    logger.info("Tuning scale on trajectory prior direction v0 (Fit Batch)...")
    scale_multipliers = [0.2, 0.5, 0.8, 1.0, 1.5]
    
    best_loss_lr_fit = loss_base_fit
    best_eta_v0 = 0.0

    for mult in scale_multipliers:
        eta = mult * w_traj
        apply_flat_delta_to_model(model, v0, template_dict, scale=eta)
        l_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end)
        logger.info(f"  v0 Scale {mult:.1f}x (eta={eta:.6f}): Fit Loss = {l_fit:.6f}")
        if l_fit < best_loss_lr_fit:
            best_loss_lr_fit = l_fit
            best_eta_v0 = eta
        restore_model()

    # Hold-out evaluation with tuned eta*
    apply_flat_delta_to_model(model, v0, template_dict, scale=best_eta_v0)
    loss_tuned_lr_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
    restore_model()
    logger.info(f"Tuned LR on v0 Eval Loss (Hold-out): {loss_tuned_lr_eval:.6f} at eta={best_eta_v0:.6f}")

    # ----------------------------------------------------
    # (3) Raw Prior extrapolation loss (w_traj * v0)
    # ----------------------------------------------------
    apply_flat_delta_to_model(model, v0, template_dict, scale=w_traj)
    loss_raw_prior_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
    restore_model()
    logger.info(f"Raw Prior Eval Loss (Hold-out): {loss_raw_prior_eval:.6f}")

    # ----------------------------------------------------
    # (4) Subspace Fit extrapolation loss (Hold-out)
    # ----------------------------------------------------
    logger.info("Computing local gradient on Fit Batch...")
    g_current_fit = compute_batch_gradient(model, valid_loader, input_device, fit_start, fit_end, template_dict)
    
    # Target direction: steepest descent on Fit Batch
    tilde_g = -g_current_fit
    
    # Project tilde_g onto orthogonalized subspace span(v0, u1, u2)
    alpha_fit = torch.dot(tilde_g, v0_ortho.double()).item()
    beta1_fit = torch.dot(tilde_g, u1.double()).item()
    beta2_fit = torch.dot(tilde_g, u2.double()).item()
    
    delta_fit = alpha_fit * v0_ortho + beta1_fit * u1 + beta2_fit * u2
    delta_fit_norm = delta_fit.norm().item()
    
    logger.info(f"Subspace projection coefficients: alpha={alpha_fit:.4e}, beta1={beta1_fit:.4e}, beta2={beta2_fit:.4e}")

    # Line search for gamma scale on Fit Batch
    best_loss_subspace_fit = loss_base_fit
    best_gamma = 0.0
    subspace_base_scale = w_traj / max(1e-8, delta_fit_norm)

    for mult in scale_multipliers:
        gamma = mult * subspace_base_scale
        apply_flat_delta_to_model(model, delta_fit, template_dict, scale=gamma)
        l_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end)
        logger.info(f"  Subspace Scale {mult:.1f}x (gamma={gamma:.6f}): Fit Loss = {l_fit:.6f}")
        if l_fit < best_loss_subspace_fit:
            best_loss_subspace_fit = l_fit
            best_gamma = gamma
        restore_model()

    # Hold-out evaluation with tuned gamma* and fitted delta
    apply_flat_delta_to_model(model, delta_fit, template_dict, scale=best_gamma)
    loss_subspace_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
    restore_model()
    logger.info(f"Subspace Fit Eval Loss (Hold-out): {loss_subspace_eval:.6f} at gamma={best_gamma:.6f}")

    # ----------------------------------------------------
    # (5) Oracle Upper Bound (Line Search on g_true)
    # ----------------------------------------------------
    logger.info("Tuning scale on Oracle gradient g_true (Fit Batch)...")
    g_true_norm = g_true.norm().item()
    oracle_base_eta = w_traj / max(1e-8, g_true_norm)
    
    best_loss_oracle_fit = loss_base_fit
    best_eta_oracle = 0.0

    for mult in scale_multipliers:
        eta_o = mult * oracle_base_eta
        apply_flat_delta_to_model(model, g_true, template_dict, scale=eta_o)
        l_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end)
        logger.info(f"  Oracle Scale {mult:.1f}x (eta={eta_o:.6f}): Fit Loss = {l_fit:.6f}")
        if l_fit < best_loss_oracle_fit:
            best_loss_oracle_fit = l_fit
            best_eta_oracle = eta_o
        restore_model()

    # Hold-out evaluation with oracle eta*
    apply_flat_delta_to_model(model, g_true, template_dict, scale=best_eta_oracle)
    loss_oracle_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
    restore_model()
    logger.info(f"Oracle Eval Loss (Hold-out): {loss_oracle_eval:.6f} at eta={best_eta_oracle:.6f}")

    # ----------------------------------------------------
    # (6) Multi-Step Extrapolation Stability (Hold-out Eval)
    # ----------------------------------------------------
    logger.info("Evaluating multi-step hold-out stability...")
    steps = [0.0, 1.0, 2.0, 3.0]
    
    # Path 1: Tuned LR path (using best_eta_v0 * v0)
    loss_path_lr = [loss_base_eval]
    for step in steps[1:]:
        apply_flat_delta_to_model(model, v0, template_dict, scale=step * best_eta_v0)
        l_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
        loss_path_lr.append(l_eval)
        restore_model()

    # Path 2: Raw prior path (using w_traj * v0)
    loss_path_prior = [loss_base_eval]
    for step in steps[1:]:
        apply_flat_delta_to_model(model, v0, template_dict, scale=step * w_traj)
        l_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
        loss_path_prior.append(l_eval)
        restore_model()

    # Path 3: Subspace Fitted path (using best_gamma * delta_fit)
    loss_path_subspace = [loss_base_eval]
    for step in steps[1:]:
        apply_flat_delta_to_model(model, delta_fit, template_dict, scale=step * best_gamma)
        l_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
        loss_path_subspace.append(l_eval)
        restore_model()

    # Path 4: Steer on current gradient g_current_fit (LR scale path for comparison)
    # This represents standard gradient updates with enlarged step bounds (lr increase)
    g_current_norm = g_current_fit.norm().item()
    eta_g = w_traj / max(1e-8, g_current_norm)
    # Tuned on Fit batch first
    best_eta_g = 0.0
    best_l_g_fit = loss_base_fit
    for mult in scale_multipliers:
        apply_flat_delta_to_model(model, tilde_g, template_dict, scale=mult * eta_g)
        l_fit = eval_loss_on_slice(model, valid_loader, input_device, fit_start, fit_end)
        if l_fit < best_l_g_fit:
            best_l_g_fit = l_fit
            best_eta_g = mult * eta_g
        restore_model()

    loss_path_g_current = [loss_base_eval]
    for step in steps[1:]:
        apply_flat_delta_to_model(model, tilde_g, template_dict, scale=step * best_eta_g)
        l_eval = eval_loss_on_slice(model, valid_loader, input_device, eval_start, eval_end)
        loss_path_g_current.append(l_eval)
        restore_model()

    logger.info("\n=== MULTI-STEP EXTRAPOLATION TRAJECTORY (HOLD-OUT) ===")
    for idx, step in enumerate(steps):
        logger.info(f"  Step {step:.1f}: CurrentGrad={loss_path_g_current[idx]:.6f}, TunedLR_v0={loss_path_lr[idx]:.6f}, RawPrior={loss_path_prior[idx]:.6f}, Subspace={loss_path_subspace[idx]:.6f}")
    logger.info("======================================================\n")

    # Generate report
    report_path = Path("runs/offline_tg_w_validation_multi_batch/subspace_validation_report.md")
    with open(report_path, "w") as f:
        f.write("# Offline Subspace Validation Report (Hold-Out Evaluation)\n\n")
        f.write(f"**Target Cycle**: {args.target_cycle} (Mid-training)\n\n")
        
        f.write("## 1. Subspace Validation Core Metrics (Hold-Out)\n\n")
        f.write("| Condition | Description | Hold-Out Loss | Reduction vs Base |\n")
        f.write("| --- | --- | --- | --- |\n")
        f.write(f"| (1) Base | Model status at target cycle start | **{loss_base_eval:.6f}** | - |\n")
        f.write(f"| (2) Tuned LR on $v_0$ | Optimized scale on prior direction (eta={best_eta_v0:.6f}) | **{loss_tuned_lr_eval:.6f}** | **{loss_base_eval - loss_tuned_lr_eval:.6f}** |\n")
        f.write(f"| (3) Raw Prior | Direct prior update $w_{{traj}} \\cdot v_0$ | **{loss_raw_prior_eval:.6f}** | **{loss_base_eval - loss_raw_prior_eval:.6f}** |\n")
        f.write(f"| (4) Subspace Fit | Projected $g_{{fit\\_batch}}$ with scale tuning (gamma={best_gamma:.6f}) | **{loss_subspace_eval:.6f}** | **{loss_base_eval - loss_subspace_eval:.6f}** |\n")
        f.write(f"| (5) Oracle Upper Bound | Optimized scale on $g_{{true}}$ (eta={best_eta_oracle:.6f}) | **{loss_oracle_eval:.6f}** | **{loss_base_eval - loss_oracle_eval:.6f}** |\n\n")
        
        f.write("### Claims Verification Check (Against Hold-Out Loss)\n")
        f.write(f"- **Claim 1 (Raw Prior ≈ Tuned LR on $v_0$)**: (2)={loss_tuned_lr_eval:.4f} vs (3)={loss_raw_prior_eval:.4f} (Difference: **{abs(loss_tuned_lr_eval - loss_raw_prior_eval):.4f}**)\n")
        f.write(f"- **Claim 2 (Subspace Fit > Tuned LR on $v_0$)**: (4)={loss_subspace_eval:.4f} vs (2)={loss_tuned_lr_eval:.4f} (Hold-Out Improvement: **{loss_tuned_lr_eval - loss_subspace_eval:.4f}**)\n")
        
        # We define a significant pass if subspace fit lowers the evaluation loss more than simple LR by at least 0.002
        v_passed = loss_subspace_eval < (loss_tuned_lr_eval - 0.002)
        f.write(f"- **Claim 2 Verdict**: **{'PASS' if v_passed else 'FAIL'}** (Requires subspace fitting to significantly outperform simple LR optimization on hold-out data)\n\n")

        f.write("## 2. Multi-Step Extrapolation Stability Trajectory (Hold-Out Eval)\n\n")
        f.write("We evaluate the stability of extrapolation along the four trajectories without intervening normal optimizer updates to detect oscillation/divergence (LR optimization vs directional extrapolation).\n\n")
        f.write("| Extrapolation Step (Multiplier) | (1) CurrentGrad Path Loss | (2) Tuned LR v0 Path Loss | (3) Raw Prior Path Loss | (4) Subspace Fit Path Loss |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for idx, step in enumerate(steps):
            f.write(f"| {step:.1f}x | {loss_path_g_current[idx]:.6f} | {loss_path_lr[idx]:.6f} | {loss_path_prior[idx]:.6f} | {loss_path_subspace[idx]:.6f} |\n")
        
        f.write("\n## 3. Discussion\n")
        f.write("- **LR Divergence vs Subspace Flatness**: Observe whether the standard CurrentGrad path diverges or oscillates at 2.0x and 3.0x step limits, whereas the Raw Prior and Subspace Fit paths remain stable or continue reducing loss. This demonstrates that subspace tracking operates along flat valleys in the parameter space, allowing longer safe update step bounds than simple gradient scaling.")

    logger.info(f"Subspace validation report written to {report_path}")


if __name__ == "__main__":
    main()
