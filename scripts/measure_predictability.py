"""Measure per-step deltas, batch noise SNR, and predictability at various batch sizes.

Runs ~20 cycles of plain LoRA training (no TG-LoRA/M9) and records:
  1. Per-step LoRA weight deltas (W_k - W_{k-1} for each SGD step)
  2. Multiple independent micro-batch gradients at W_0 (for noise SNR)
  3. Loss before/after each cycle

Usage:
  python scripts/measure_predictability.py --accum 1
  python scripts/measure_predictability.py --accum 8
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import numpy as np

# ─── Reuse existing utilities ───
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.load_model import load_base_model, load_tokenizer, apply_lora
from src.data.build_seed_dataset import load_dataset
from omegaconf import OmegaConf


def get_lora_state(model):
    """Return dict of LoRA parameter tensors (detach + clone)."""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.detach().clone().float()
    return state


def lora_delta(before, after):
    """Compute per-key delta: after - before."""
    delta = {}
    for k in after:
        if k in before:
            delta[k] = after[k] - before[k]
    return delta


def global_norm(tensors):
    total = 0.0
    for t in tensors.values():
        n = t.float().norm().item()
        if math.isfinite(n):
            total += n ** 2
    return math.sqrt(total)


def global_cosine(left, right):
    dot = 0.0; lsq = 0.0; rsq = 0.0
    for key in left.keys() & right.keys():
        a = left[key].float().flatten()
        b = right[key].float().flatten()
        d = torch.dot(a, b).item()
        l = torch.dot(a, a).item()
        r = torch.dot(b, b).item()
        if math.isfinite(d) and math.isfinite(l) and math.isfinite(r):
            dot += d; lsq += l; rsq += r
    denom = math.sqrt(lsq) * math.sqrt(rsq)
    return dot / denom if denom > 1e-12 else 0.0


def compute_grad_dict(model):
    """Extract gradient tensors for trainable params."""
    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads[name] = param.grad.detach().clone().float()
    return grads


def clear_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def forward_backward(model, batch, device):
    """Run forward + backward on a single batch, return loss scalar."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Standard cross-entropy loss (same as training)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    loss.backward()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--cycles", type=int, default=20, help="Number of training cycles")
    parser.add_argument("--K", type=int, default=3, help="Optimizer steps per cycle")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--noise-samples", type=int, default=4,
                        help="Number of independent micro-batch gradients for SNR")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"runs/measure_accum{args.accum}.jsonl"

    # ─── Config (minimal, reuses model/lora/data settings) ───
    cfg = OmegaConf.load("configs/9b_tg_lora_m9.yaml")

    # Override training params
    accum = args.accum
    K = args.K
    lr = args.lr
    n_cycles = args.cycles
    n_noise = args.noise_samples

    print(f"=== Predictability Measurement ===")
    print(f"accum={accum}, K={K}, lr={lr}, cycles={n_cycles}, noise_samples={n_noise}")

    # ─── Load model ───
    print("Loading model...")
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    device = next(p for p in model.parameters() if p.requires_grad).device
    print(f"Device: {device}")

    # ─── Load data ───
    print("Loading data...")
    train_dataset = load_dataset(
        cfg.data.train_path, tokenizer, 256,  # shortened for 12GB VRAM measurement
        # cfg.data.max_seq_len,
        train_on_prompt=cfg.training.get("train_on_prompt", False),
    )
    print(f"Training samples: {len(train_dataset)}")

    # ─── Results storage ───
    results = []
    all_step_deltas = []  # For consecutive predictability

    # ─── Training loop ───
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    # Pre-generate all batch indices
    torch.manual_seed(cfg.experiment.seed)
    indices = torch.randperm(len(train_dataset)).tolist()

    idx_ptr = 0  # Global index pointer into shuffled data
    sample_idx = 0

    for cycle in range(n_cycles):
        t0 = time.time()
        cycle_result = {"cycle": cycle, "accum": accum, "K": K, "lr": lr}

        # ─── Save W_0 ───
        w_before = get_lora_state(model)

        # ─── NOISE MEASUREMENT: n_noise independent micro-batch gradients at W_0 ───
        noise_grads = []
        for ni in range(n_noise):
            clear_grads(model)
            # Sample a single micro-batch
            if idx_ptr >= len(train_dataset):
                idx_ptr = 0  # wrap around
            batch_idx = [indices[idx_ptr % len(train_dataset)]]
            idx_ptr += 1
            batch = train_dataset[batch_idx[0]]
            # Wrap in batch format
            batch_dict = {
                "input_ids": batch["input_ids"].unsqueeze(0),
                "attention_mask": batch["attention_mask"].unsqueeze(0),
                "labels": batch["labels"].unsqueeze(0),
            }
            loss_val = forward_backward(model, batch_dict, device)
            grad = compute_grad_dict(model)
            noise_grads.append({
                "grad": grad,
                "loss": loss_val,
                "grad_norm": global_norm(grad),
            })

        # Compute SNR from noise_grads
        if len(noise_grads) >= 2:
            # Mean gradient
            mean_grad = {}
            for k in noise_grads[0]["grad"]:
                mean_grad[k] = torch.stack([ng["grad"][k] for ng in noise_grads]).mean(dim=0)

            # Variance of gradient
            var_norm_sq = 0.0
            mean_norm_sq = global_norm(mean_grad) ** 2
            for ng in noise_grads:
                diff_sq = 0.0
                for k in mean_grad:
                    d = ng["grad"][k] - mean_grad[k]
                    diff_sq += torch.dot(d.flatten(), d.flatten()).item()
                var_norm_sq += diff_sq
            var_norm_sq /= len(noise_grads)

            snr = mean_norm_sq / var_norm_sq if var_norm_sq > 1e-12 else 0.0

            # Also compute pairwise cosines of noise gradients
            pair_cos = []
            for i in range(len(noise_grads)):
                for j in range(i + 1, len(noise_grads)):
                    pair_cos.append(global_cosine(noise_grads[i]["grad"], noise_grads[j]["grad"]))

            cycle_result["noise_snr"] = snr
            cycle_result["noise_mean_grad_norm"] = math.sqrt(mean_norm_sq)
            cycle_result["noise_grad_var_norm"] = math.sqrt(var_norm_sq)
            cycle_result["noise_pair_cos_mean"] = np.mean(pair_cos) if pair_cos else 0
            cycle_result["noise_pair_cos_std"] = np.std(pair_cos) if pair_cos else 0
            cycle_result["noise_grad_norms"] = [ng["grad_norm"] for ng in noise_grads]

            print(f"  SNR={snr:.4f}  mean||g||={math.sqrt(mean_norm_sq):.1f}  "
                  f"noise||g||={math.sqrt(var_norm_sq):.1f}  pair_cos={np.mean(pair_cos):.4f}")

        # Clear noise gradients
        clear_grads(model)

        # ─── K SGD STEPS with per-step delta saving ───
        step_deltas_this_cycle = []
        for step_k in range(K):
            w_step_before = get_lora_state(model)

            # Accumulate gradients over 'accum' micro-batches
            step_loss = 0.0
            for micro in range(accum):
                if idx_ptr >= len(train_dataset):
                    idx_ptr = 0
                batch_idx = [indices[idx_ptr % len(train_dataset)]]
                idx_ptr += 1
                batch = train_dataset[batch_idx[0]]
                batch_dict = {
                    "input_ids": batch["input_ids"].unsqueeze(0),
                    "attention_mask": batch["attention_mask"].unsqueeze(0),
                    "labels": batch["labels"].unsqueeze(0),
                }
                loss_val = forward_backward(model, batch_dict, device)
                step_loss += loss_val

            step_loss /= accum

            # Gradient averaging
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(accum)

            # Optimizer step
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            # Save per-step delta
            w_step_after = get_lora_state(model)
            delta = lora_delta(w_step_before, w_step_after)
            delta_norm = global_norm(delta)

            step_deltas_this_cycle.append({
                "step": step_k,
                "delta_norm": delta_norm,
                "step_loss": step_loss,
            })
            all_step_deltas.append({
                "cycle": cycle,
                "step_k": step_k,
                "delta": delta,
                "delta_norm": delta_norm,
            })

            cycle_result[f"step_{step_k}_delta_norm"] = delta_norm
            cycle_result[f"step_{step_k}_loss"] = step_loss

        # ─── Cycle-level delta ───
        w_after = get_lora_state(model)
        cycle_delta = lora_delta(w_before, w_after)
        cycle_result["cycle_delta_norm"] = global_norm(cycle_delta)

        # ─── Consecutive cycle predictability ───
        if len(all_step_deltas) >= 2:
            # Full cycle delta predictability
            prev = all_step_deltas[-1]["delta"] if cycle > 0 else None

        elapsed = time.time() - t0
        cycle_result["elapsed"] = elapsed

        # Print summary
        step_norms = [sd["delta_norm"] for sd in step_deltas_this_cycle]
        print(f"Cycle {cycle:3d}: delta_norms={[f'{n:.2f}' for n in step_norms]}  "
              f"cycle_delta={cycle_result['cycle_delta_norm']:.2f}  "
              f"time={elapsed:.1f}s")

        results.append(cycle_result)

    # ─── Post-hoc analysis: consecutive step predictability ───
    print(f"\n{'=' * 70}")
    print("ANALYSIS: Per-step marginal improvement")
    print(f"{'=' * 70}")

    # Marginal: step_k norm / step_0 norm
    for k in range(K):
        norms_k = [r[f"step_{k}_delta_norm"] for r in results if f"step_{k}_delta_norm" in r]
        if norms_k:
            print(f"  Step {k}: mean_delta_norm={np.mean(norms_k):.2f}  "
                  f"ratio_to_step0={np.mean(norms_k)/np.mean([r['step_0_delta_norm'] for r in results]):.3f}")

    # Loss improvement per step
    for k in range(K):
        losses_k = [r[f"step_{k}_loss"] for r in results if f"step_{k}_loss" in r]
        if losses_k:
            print(f"  Step {k}: mean_loss={np.mean(losses_k):.4f}")

    # Step-to-step predictability within this run
    if len(all_step_deltas) > 1:
        print(f"\n--- Step-to-step predictability ---")
        consec_cos = []
        for i in range(1, len(all_step_deltas)):
            c = global_cosine(all_step_deltas[i]["delta"], all_step_deltas[i-1]["delta"])
            consec_cos.append(c)
        cc = np.array(consec_cos)
        print(f"  Consecutive steps: cos mean={cc.mean():+.4f}  |cos| mean={np.abs(cc).mean():.4f}")

        # Also cycle-level
        cycle_deltas_seq = []
        for cyc in range(n_cycles):
            r = results[cyc]
            w_before_cycle = None  # Don't have individual cycle deltas stored as tensors
            # Use sum of step deltas as proxy
            full_delta = None
            for sd in all_step_deltas:
                if sd["cycle"] == cyc:
                    if full_delta is None:
                        full_delta = {k: v.clone() for k, v in sd["delta"].items()}
                    else:
                        for k, v in sd["delta"].items():
                            if k in full_delta:
                                full_delta[k] += v
            if full_delta is not None:
                cycle_deltas_seq.append(full_delta)

        if len(cycle_deltas_seq) > 1:
            cycle_cos = []
            for i in range(1, len(cycle_deltas_seq)):
                c = global_cosine(cycle_deltas_seq[i], cycle_deltas_seq[i-1])
                cycle_cos.append(c)
            cca = np.array(cycle_cos)
            print(f"  Cycle-level: cos mean={cca.mean():+.4f}  |cos| mean={np.abs(cca).mean():.4f}")

    # SNR summary
    snr_vals = [r["noise_snr"] for r in results if "noise_snr" in r]
    if snr_vals:
        print(f"\n--- Batch noise SNR ---")
        print(f"  Mean SNR: {np.mean(snr_vals):.4f}")
        print(f"  Range: [{np.min(snr_vals):.4f}, {np.max(snr_vals):.4f}]")
        # Expected SNR improvement with accum
        for acc in [1, 2, 4, 8, 16]:
            expected_snr = np.mean(snr_vals) * acc
            print(f"  Expected SNR at accum={acc:2d}: {expected_snr:.4f}")

    # ─── Save results ───
    # Remove non-serializable tensors
    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if not isinstance(v, torch.Tensor) and not isinstance(v, dict)}
        save_results.append(sr)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in save_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {args.output}")

    # Clean up large tensors
    del all_step_deltas
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
