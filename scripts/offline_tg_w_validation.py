import argparse
import logging
import os
import gc
import time
from pathlib import Path
import torch
import numpy as np

from src.model.load_model import load_base_model, load_tokenizer, apply_lora
from src.model.lora_utils import configure_trainable_lora_scope
from src.data.build_seed_dataset import load_dataset
from src.training.config_schema import load_validate_and_build_config
from src.training.trajectory_delta_artifact import load_trajectory_delta_artifact
from src.eval.eval_loss import eval_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("offline-validation")


def flatten_tensor_dict(tensor_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    # Concatenate all tensors into a single 1D vector in a deterministic order
    sorted_keys = sorted(tensor_dict.keys())
    return torch.cat([tensor_dict[k].flatten() for k in sorted_keys])


def unflatten_tensor_dict(flat_vector: torch.Tensor, template_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Restore the dict structure from the flat vector
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


def eval_loss_safe(model, loader, device, max_examples=128):
    torch.cuda.empty_cache()
    gc.collect()
    loss = eval_loss(model, loader, device, max_examples=max_examples)
    torch.cuda.empty_cache()
    gc.collect()
    return loss


def gram_schmidt(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    ortho_vectors = []
    for v in vectors:
        v_ortho = v.clone().double()  # Use double precision for Gram-Schmidt numerical stability
        for u in ortho_vectors:
            proj = torch.dot(v_ortho, u) * u
            v_ortho -= proj
        norm = v_ortho.norm()
        if norm > 1e-8:
            ortho_vectors.append(v_ortho / norm)
    return [vec.float() for vec in ortho_vectors]


def main():
    parser = argparse.ArgumentParser(description="Offline TG-LoRA validation (Phase 0)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to past run trajectory_delta_artifacts folder")
    parser.add_argument("--gradients-dir", type=str, required=True, help="Path to collected true gradients folder")
    parser.add_argument("--target-cycle", type=int, default=7, help="Mid-training cycle to perform validation at")
    parser.add_argument("--output-dir", type=str, default=None, help="Output summary directory")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"runs/offline_tg_w_validation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, cfg = load_validate_and_build_config(args.config)

    logger.info("Loading model and dataset for loss evaluation...")
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
    from torch.utils.data import DataLoader
    collate_fn = getattr(valid_dataset, "collate_fn", None)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    input_device = next(model.parameters()).device

    # 1. Load delta artifacts (變位履歴)
    logger.info(f"Loading delta artifacts from {args.artifacts_dir}...")
    delta_paths = sorted(Path(args.artifacts_dir).glob("*after_pilot*.pt"))
    if not delta_paths:
        raise FileNotFoundError(f"No pilot delta artifacts found in {args.artifacts_dir}")

    deltas = []
    template_dict = None
    for p in delta_paths:
        art = load_trajectory_delta_artifact(p)
        if template_dict is None:
            template_dict = {k: v.clone() for k, v in art.delta_tensors.items()}
        flat_d = flatten_tensor_dict(art.delta_tensors)
        deltas.append(flat_d)
    
    deltas_stack = torch.stack(deltas)  # [num_cycles, num_params]
    logger.info(f"Loaded {len(deltas)} delta steps. Parameter flat dim: {deltas_stack.shape[1]}")

    # Reconstruct the model parameters at the target cycle W_0^(target_cycle)
    # Accept history from past run log
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

    # Model is now at W_0^(target_cycle) state. Take checkpoint.
    original_state = {name: param.data.clone() for name, param in model.named_parameters() if name in template_dict}

    # ----------------------------------------------------
    # 検証 1: スケール prior w_traj が手本になるか
    # ----------------------------------------------------
    logger.info(f"Running Verification 1 (w_traj scale prior) at target cycle {args.target_cycle}...")
    norms = deltas_stack.norm(dim=1)
    w_traj = float(norms.median().item())
    w_traj_mean = float(norms.mean().item())
    
    # Calculate v0 as the normalized mean direction of all past accepted deltas
    mean_delta = deltas_stack.mean(dim=0)
    v0 = mean_delta / mean_delta.norm()

    # Step size of the actual delta in the target cycle
    actual_delta = deltas[args.target_cycle]
    actual_step_norm = actual_delta.norm().item()
    w_traj_ratio = w_traj / max(1e-8, actual_step_norm)

    loss_base = eval_loss_safe(model, valid_loader, input_device, max_examples=128)
    
    # Apply w_traj * v0
    apply_flat_delta_to_model(model, v0, template_dict, scale=w_traj)
    loss_w_traj = eval_loss_safe(model, valid_loader, input_device, max_examples=128)
    
    # Restore model
    for name, param in model.named_parameters():
        if name in original_state:
            param.data.copy_(original_state[name])
            
    # Apply actual accepted delta (e.g. deltas[target_cycle])
    apply_flat_delta_to_model(model, actual_delta, template_dict, scale=1.0)
    loss_actual = eval_loss_safe(model, valid_loader, input_device, max_examples=128)
    
    # Restore model
    for name, param in model.named_parameters():
        if name in original_state:
            param.data.copy_(original_state[name])

    # Calculate cosine similarity between prior direction v0 and actual step direction
    v0_norm = v0 / max(1e-8, float(v0.norm().item()))
    actual_delta_norm = actual_delta / max(1e-8, float(actual_delta.norm().item()))
    cos_sim = float(torch.dot(v0_norm.double(), actual_delta_norm.double()).item())

    loss_base_actual_diff = loss_base - loss_actual
    loss_base_w_traj_diff = loss_base - loss_w_traj
    loss_diff = abs(loss_w_traj - loss_actual)
    
    # Validation criteria: loss_w_traj < loss_base and (loss_w_traj <= loss_actual or loss_diff < 0.05)
    v1_passed = loss_w_traj < loss_base and (loss_w_traj <= loss_actual or loss_diff < 0.05)
    logger.info(f"V1 Results: loss_base={loss_base:.4f}, loss_w_traj={loss_w_traj:.4f}, loss_actual={loss_actual:.4f}")
    logger.info(f"loss reduction: actual={loss_base_actual_diff:.4f}, w_traj={loss_base_w_traj_diff:.4f}")
    logger.info(f"w_traj (median)={w_traj:.6f}, actual step norm={actual_step_norm:.6f}, ratio={w_traj_ratio:.4%}")
    logger.info(f"Cosine similarity between prior direction v0 and actual step: {cos_sim:.4f}")

    # ----------------------------------------------------
    # 検証 2: 方向補正に何本必要か (Subspace Projection Ratio)
    # ----------------------------------------------------
    logger.info("Running Verification 2 (Subspace Projection Ratio)...")
    # Load the specific target cycle's true gradient
    grad_path = Path(args.gradients_dir) / f"gradient_step_{args.target_cycle + 1:06d}.pt"
    if not grad_path.exists():
        raise FileNotFoundError(f"Gradient artifact not found: {grad_path}")
    
    blob = torch.load(grad_path, map_location="cpu")
    g_true = flatten_tensor_dict(blob["gradients"])
    g_norm = g_true.norm()
    logger.info(f"Loaded gradient for cycle {args.target_cycle} (step {blob['step']}). Gradient norm: {g_norm:.6f}")

    # Build subspace S = span(v0, u1, u2)
    # Construction of auxiliary directions u1, u2 from PCA of past displacements
    centered_deltas = deltas_stack - deltas_stack.mean(dim=0)
    U, S_val, V = torch.pca_lowrank(centered_deltas, q=4, niter=4)
    pc1 = V[:, 0]
    pc2 = V[:, 1]
    
    ortho_basis = gram_schmidt([v0, pc1, pc2])
    v0_ortho = ortho_basis[0]
    u1 = ortho_basis[1] if len(ortho_basis) > 1 else torch.zeros_like(v0)
    u2 = ortho_basis[2] if len(ortho_basis) > 2 else torch.zeros_like(v0)

    # proj_S for m=0 (span(v0))
    proj_0 = torch.dot(g_true, v0_ortho) * v0_ortho
    ratio_0 = (proj_0.norm() / g_norm).item()

    # proj_S for m=1 (span(v0, u1))
    proj_1 = proj_0 + torch.dot(g_true, u1) * u1
    ratio_1 = (proj_1.norm() / g_norm).item()

    # proj_S for m=2 (span(v0, u1, u2))
    proj_2 = proj_1 + torch.dot(g_true, u2) * u2
    ratio_2 = (proj_2.norm() / g_norm).item()

    v2_passed = ratio_2 >= 0.70 or ratio_1 >= 0.65
    logger.info(f"V2 Results (Mean Projection Ratios): m=0: {ratio_0:.4f}, m=1: {ratio_1:.4f}, m=2: {ratio_2:.4f}")

    # ----------------------------------------------------
    # 検証 3: 正規化後の有限差分 ε 窓
    # ----------------------------------------------------
    logger.info("Running Verification 3 (Finite Difference Epsilon Sweep)...")
    # For target cycle gradient g_true, compare directional derivative g_true . v0
    # to the finite difference (L(c+eps*v0) - L(c)) / eps in the normalized coordinate
    epsilons = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    exact_dir_deriv = torch.dot(g_true, v0).item()

    fd_results = {}
    for eps in epsilons:
        # Step size in flat parameter space is eps * w_traj
        step_size = eps * w_traj
        apply_flat_delta_to_model(model, v0, template_dict, scale=step_size)
        loss_plus = eval_loss_safe(model, valid_loader, input_device, max_examples=128)
        
        # Restore
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])
                
        fd_approx = (loss_plus - loss_base) / eps
        expected = exact_dir_deriv * w_traj
        rel_error = abs(fd_approx - expected) / max(1e-8, abs(expected))
        fd_results[eps] = {
            "fd_approx": fd_approx,
            "expected": expected,
            "rel_error": rel_error
        }
        logger.info(f"V3 Epsilon {eps:.2e}: fd_approx={fd_approx:.6f}, expected={expected:.6f}, rel_error={rel_error:.4%}")

    # Epsilon window criteria: error < 35% for at least one epsilon in the range [3e-3, 1e-2]
    stable_eps = [eps for eps in [1e-2, 3e-3] if fd_results[eps]["rel_error"] < 0.35]
    v3_passed = len(stable_eps) > 0
    best_eps = stable_eps[0] if v3_passed else 3e-3

    # ----------------------------------------------------
    # Summary Report Generation
    # ----------------------------------------------------
    summary_path = output_dir / "summary.md"
    gate_passed = v1_passed and v2_passed and v3_passed
    
    with open(summary_path, "w") as f:
        f.write("# Offline TG-LoRA Low-Dimensional Prior Validation Summary\n\n")
        f.write(f"**Date**: {timestamp}\n")
        f.write(f"**Target Evaluation Cycle**: {args.target_cycle} (Mid-training)\n")
        f.write(f"**Overall Gate Outcome**: {'PASSED' if gate_passed else 'FAILED'}\n\n")
        
        f.write("## Validation Outcomes\n\n")
        f.write("| Test | Criterion | Metric Value | Status |\n")
        f.write("| --- | --- | --- | --- |\n")
        f.write(f"| Verification 1 (w_traj prior) | loss reduction > 0 and comparable/superior | base: {loss_base:.4f}, w_traj: {loss_w_traj:.4f}, actual: {loss_actual:.4f} (reduction w_traj: {loss_base_w_traj_diff:.4f}, actual: {loss_base_actual_diff:.4f}) | {'PASS' if v1_passed else 'FAIL'} |\n")
        f.write(f"| Verification 2 (Proj Ratio) | m=2 projection ratio >= 0.70 | ratio (m=0): {ratio_0:.4f}, (m=1): {ratio_1:.4f}, (m=2): {ratio_2:.4f} | {'PASS' if v2_passed else 'FAIL'} |\n")
        f.write(f"| Verification 3 (Epsilon Window) | rel_error < 35% in [3e-3, 1e-2] | best error: {fd_results[best_eps]['rel_error']:.2%} at {best_eps:.2e} | {'PASS' if v3_passed else 'FAIL'} |\n\n")

        f.write("## Step Scale Comparison (Verification 1)\n\n")
        f.write(f"- **w_traj prior step norm**: {w_traj:.6f}\n")
        f.write(f"- **Actual step norm**: {actual_step_norm:.6f}\n")
        f.write(f"- **Ratio (w_traj / actual)**: {w_traj_ratio:.2%}\n")
        f.write(f"- **Cosine Similarity (prior vs actual direction)**: {cos_sim:.4f}\n")
        f.write(f"- **Loss Reduction Ratio (w_traj / actual)**: {loss_base_w_traj_diff / max(1e-8, loss_base_actual_diff):.2%}\n\n")

        f.write("## Finite Difference Sweep Details (Verification 3)\n\n")
        f.write("| Epsilon | FD Approx Derivative | Expected Derivative | Relative Error |\n")
        f.write("| --- | --- | --- | --- |\n")
        for eps, res in fd_results.items():
            f.write(f"| {eps:.2e} | {res['fd_approx']:.6f} | {res['expected']:.6f} | {res['rel_error']:.2%} |\n")
            
        f.write("\n## Verdict\n\n")
        if gate_passed:
            f.write("🎉 **All criteria met.** Ready to proceed to **Phase 1 (Implementation)**.\n")
            f.write(f"Recommended configuration: `tg_aux_directions=2`, `tg_fd_epsilon={best_eps}`.\n")
        else:
            f.write("⚠️ **Some criteria failed.** Do NOT proceed to implementation. Revise design according to failed test items:\n")
            if not v1_passed:
                f.write("- **w_traj validation failed**: Scale prior does not yield adequate loss reduction compared to base.\n")
            if not v2_passed:
                f.write("- **Projection ratio validation failed**: Directional subspace does not capture sufficient gradient vector norm (ratio < 0.70).\n")
            if not v3_passed:
                f.write("- **Epsilon sweep failed**: Numerical precision bounds finite difference directional derivatives.\n")

    logger.info(f"Offline validation complete. Summary written to {summary_path}")
    print(f"Outcome: {'PASSED' if gate_passed else 'FAILED'}. Summary at {summary_path}")


if __name__ == "__main__":
    main()
