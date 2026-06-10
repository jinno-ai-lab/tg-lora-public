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
logger = logging.getLogger("offline-multi-batch-validation")


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
    """Evaluate loss on a specific slice of the dataloader (independent batch)."""
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


def main():
    parser = argparse.ArgumentParser(description="Offline validation on 5 independent batches")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to past run trajectory_delta_artifacts folder")
    parser.add_argument("--target-cycle", type=int, default=7, help="Mid-training cycle to perform validation at")
    args = parser.parse_args()

    _, cfg = load_validate_and_build_config(args.config)

    logger.info("Loading model and dataset for multi-batch loss evaluation...")
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
    # Use batch_size=1 for OOM safety
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    input_device = next(model.parameters()).device

    # 1. Load delta artifacts
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
    
    deltas_stack = torch.stack(deltas)

    # Reconstruct parameter state at W_0^(target_cycle)
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

    original_state = {name: param.data.clone() for name, param in model.named_parameters() if name in template_dict}

    # Setup 5 independent batch slices
    # Let's make 5 slices of 64 items (total 320 items) for optimized execution time
    batch_size_slice = 64
    num_batches = 5
    slices = [(i * batch_size_slice, (i + 1) * batch_size_slice) for i in range(num_batches)]
    
    norms = deltas_stack.norm(dim=1)
    w_traj = float(norms.median().item())
    mean_delta = deltas_stack.mean(dim=0)
    v0 = mean_delta / mean_delta.norm()
    actual_delta = deltas[args.target_cycle]
    
    # Calculate cosine similarity
    v0_norm = v0 / max(1e-8, float(v0.norm().item()))
    actual_delta_norm = actual_delta / max(1e-8, float(actual_delta.norm().item()))
    cos_sim = float(torch.dot(v0_norm.double(), actual_delta_norm.double()).item())
    
    results = []
    
    logger.info("Evaluating on 5 independent batches...")
    for i, (start, end) in enumerate(slices):
        logger.info(f"Batch {i+1}/5 (slice: {start} to {end})...")
        
        # 1. Base loss
        loss_base = eval_loss_on_slice(model, valid_loader, input_device, start, end)
        
        # 2. w_traj loss
        apply_flat_delta_to_model(model, v0, template_dict, scale=w_traj)
        loss_w_traj = eval_loss_on_slice(model, valid_loader, input_device, start, end)
        
        # Restore
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])
                
        # 3. Actual loss
        apply_flat_delta_to_model(model, actual_delta, template_dict, scale=1.0)
        loss_actual = eval_loss_on_slice(model, valid_loader, input_device, start, end)
        
        # Restore
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])
                
        red_w_traj = loss_base - loss_w_traj
        red_actual = loss_base - loss_actual
        
        results.append({
            "batch_idx": i + 1,
            "loss_base": loss_base,
            "loss_w_traj": loss_w_traj,
            "loss_actual": loss_actual,
            "red_w_traj": red_w_traj,
            "red_actual": red_actual
        })
        logger.info(f"Batch {i+1} results: Base={loss_base:.4f}, w_traj={loss_w_traj:.4f} (red={red_w_traj:.4f}), actual={loss_actual:.4f} (red={red_actual:.4f})")

    # Calculate mean and std
    red_w_traj_vals = [r["red_w_traj"] for r in results]
    red_actual_vals = [r["red_actual"] for r in results]
    
    mean_red_w_traj = np.mean(red_w_traj_vals)
    std_red_w_traj = np.std(red_w_traj_vals)
    mean_red_actual = np.mean(red_actual_vals)
    std_red_actual = np.std(red_actual_vals)
    
    # Save report
    report_dir = Path("runs/offline_tg_w_validation_multi_batch")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "multi_batch_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Offline Validation on 5 Independent Batches\n\n")
        f.write(f"**Target Evaluation Cycle**: {args.target_cycle} (Mid-training)\n")
        f.write(f"**Cosine Similarity (prior v0 vs actual)**: {cos_sim:.4f}\n\n")
        
        f.write("## Detailed Batch Results\n\n")
        f.write("| Batch | loss_base | loss_w_traj | loss_actual | w_traj Reduction | actual Reduction |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for r in results:
            f.write(f"| {r['batch_idx']} | {r['loss_base']:.4f} | {r['loss_w_traj']:.4f} | {r['loss_actual']:.4f} | {r['red_w_traj']:.4f} | {r['red_actual']:.4f} |\n")
        f.write("\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **w_traj Loss Reduction**: Mean = **{mean_red_w_traj:.4f}**, Std = **{std_red_w_traj:.4f}**\n")
        f.write(f"- **Actual Loss Reduction**: Mean = **{mean_red_actual:.4f}**, Std = **{std_red_actual:.4f}**\n")
        
    logger.info(f"Multi-batch evaluation complete. Report written to {report_path}")
    print(f"\n=== MULTI-BATCH VALIDATION SUMMARY ===")
    print(f"Cosine Similarity (prior v0 vs actual): {cos_sim:.4f}")
    print(f"w_traj Loss Reduction: Mean = {mean_red_w_traj:.4f}, Std = {std_red_w_traj:.4f}")
    print(f"Actual Loss Reduction: Mean = {mean_red_actual:.4f}, Std = {std_red_actual:.4f}")
    print(f"======================================\n")


if __name__ == "__main__":
    main()
