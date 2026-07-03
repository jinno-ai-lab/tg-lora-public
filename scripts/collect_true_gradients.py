import argparse
import logging
import os
import gc
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.model.load_model import load_base_model, load_tokenizer, apply_lora
from src.model.lora_utils import configure_trainable_lora_scope
from src.data.build_seed_dataset import load_dataset, LoraDataset
from src.training.trainer_loop import create_optimizer, forward_backward, optimizer_step
from src.training.config_schema import load_validate_and_build_config
from src.training.trajectory_delta_artifact import load_trajectory_delta_artifact
from src.utils.atomic_save import _atomic_torch_save

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect-gradients")


def apply_dict_delta_to_model(model, delta_dict):
    for name, p in model.named_parameters():
        if name in delta_dict:
            p.data.add_(delta_dict[name].to(p.device))


def main():
    parser = argparse.ArgumentParser(description="Collect true effective Adam updates matching past delta trajectory")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to past run trajectory_delta_artifacts folder")
    parser.add_argument("--output-dir", type=str, default="runs/collected_gradients", help="Directory to save gradients")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of cycles to collect gradients for")
    parser.add_argument("--target-cycle", type=int, default=None, help="Specific cycle (0-indexed) to collect updates for (skips others)")
    args = parser.parse_args()

    _, cfg = load_validate_and_build_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model and tokenizer...")
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    trainable_lora_scope = cfg.training.get("trainable_lora_scope", "all")
    configure_trainable_lora_scope(model, trainable_lora_scope)

    logger.info("Loading dataset...")
    train_dataset = load_dataset(
        cfg.data.train_path,
        tokenizer,
        cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )
    
    collate_fn = getattr(train_dataset, "collate_fn", None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # Keep deterministic order for reproducibility
        collate_fn=collate_fn,
    )

    input_device = next(model.parameters()).device
    grad_accum = cfg.training.grad_accumulation
    loader_iter = iter(train_loader)

    # Accept history from past run log
    # Cycle 0-8: True, Cycle 9: False (rollback), Cycle 10-11: True, Cycle 12: False (rollback), Cycle 13-14: True
    accept_history = [True, True, True, True, True, True, True, True, True, False, True, True, False, True, True]

    # Initialize optimizer
    optimizer = create_optimizer(
        model,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    logger.info(f"Starting Adam update collection matching past trajectory in {args.artifacts_dir}...")
    for cycle in range(args.num_steps):
        should_train = (args.target_cycle is None) or (cycle == args.target_cycle)
        
        if should_train:
            # 1. Backup model weights at W_0 state of the cycle
            W_backup = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
            
            # Reset optimizer state to act fresh for this step
            optimizer = create_optimizer(
                model,
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
            
            # Get b_logical from config (default to 32)
            b_logical = 32
            if "alpha_line" in cfg:
                b_logical = cfg.alpha_line.get("b_logical", 32)
            elif "tg_lora" in cfg:
                b_logical = cfg.tg_lora.get("b_logical", 32)
            logger.info(f"Running {b_logical} optimization steps for Cycle {cycle} to collect cumulative update...")

            total_loss = 0.0
            for step_idx in range(b_logical):
                model.zero_grad(set_to_none=True)
                step_loss = 0.0
                
                # Accumulate gradients over micro-batches
                for _ in range(grad_accum):
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(train_loader)
                        batch = next(loader_iter)
                    
                    batch = {
                        k: v.to(input_device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    micro_loss = forward_backward(model, batch, grad_accum)
                    step_loss += micro_loss
                
                # Perform optimizer step to apply Adam scaling and update weights
                optimizer_step(optimizer, None, model, cfg.training.max_grad_norm)
                total_loss += (step_loss / grad_accum)
            
            loss_val = total_loss / b_logical
            
            # 2. Capture effective updates (W_after - W_before)
            update_dict = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    update_dict[name] = (param.data - W_backup[name].to(param.device)).detach().cpu().clone()
            
            target_path = output_dir / f"gradient_step_{cycle+1:06d}.pt"
            # Atomic so an OOM kill / SIGINT mid-dump never leaves a torn
            # ``gradient_step_*.pt`` — these effective-update artifacts are costly
            # to recompute (a full forward/backward cycle each) and feed the
            # trajectory-delta evaluation downstream.
            _atomic_torch_save({
                "step": cycle + 1,
                "loss": loss_val,
                "gradients": update_dict, # Keep key "gradients" for compatibility with evaluation script
            }, target_path)
            logger.info(f"Saved effective updates for Cycle {cycle} (loss={loss_val:.4f}) to {target_path}")

            # 3. Restore model parameters back to W_0 before trajectory updates
            for name, param in model.named_parameters():
                if name in W_backup:
                    param.data.copy_(W_backup[name])

        # 4. Update model parameters to next cycle's W_0 (past delta artifacts)
        pilot_path = Path(args.artifacts_dir) / f"tg_lora_after_pilot_cycle_{cycle:06d}.pt"
        if not pilot_path.exists():
            logger.warning(f"Pilot delta artifact not found: {pilot_path}. Stopping reconstruction.")
            break
            
        pilot_art = load_trajectory_delta_artifact(pilot_path)
        apply_dict_delta_to_model(model, pilot_art.delta_tensors)
        
        if accept_history[cycle]:
            spec_path = Path(args.artifacts_dir) / f"tg_lora_after_speculative_update_cycle_{cycle:06d}.pt"
            if spec_path.exists():
                spec_art = load_trajectory_delta_artifact(spec_path)
                apply_dict_delta_to_model(model, spec_art.delta_tensors)
            else:
                logger.warning(f"Speculative update delta not found: {spec_path}")
        
        # Clean gradients and memory
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Collection complete!")


if __name__ == "__main__":
    main()
