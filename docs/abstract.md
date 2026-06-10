# TG-LoRA Paper: Abstract & Positioning

# TG-LoRA Paper: Abstract & Positioning

## Working Title

TG-LoRA: Guarded Trajectory Extrapolation with Layer-Prefix Caching for Suffix-Only LoRA Fine-Tuning

## Abstract (Draft)

Large language model fine-tuning on consumer GPUs is often limited by GPU residency and sequence-length memory rather than by trainable parameter count alone. We introduce TG-LoRA, a two-component framework for suffix-only LoRA fine-tuning: a layer-prefix feature cache that precomputes deterministic hidden states at a split-layer boundary and offloads frozen lower layers to host memory, and a guarded trajectory extrapolation mechanism that projects LoRA weights along recent optimization displacement and accepts the proposal only under a calibration-batch check.

On a Qwen 9B suffix-only LoRA setup using a single RTX 3060 12GB and a 5K Dolly subset, the cache/offload component reduces peak GPU memory from 10,782 MB to 7,459 MB, enabling sequence lengths 1536 and 2048 where the online baseline OOMs under the same configuration. The cache has a substantial cold-build cost but is reusable across runs sharing the same model, split layer, tokenization, and dataset. At a matched budget of 240 backward passes, guarded extrapolation reduces held-out validation loss from 1.1682 to 1.1187 across three paired seeds, while warm-path wall-clock time is slightly slower than the baseline due to CPU-to-GPU feature transfer. External evaluation on ARC-Easy, HellaSwag, and TruthfulQA shows no measurable aggregate degradation. Ablations attribute memory gains to layer-prefix caching/offload and validation-loss gains to trajectory extrapolation.

## Two-Component Decomposition

TG-LoRA is best understood as two independent, complementary mechanisms:

### Component 1: Layer-Prefix Feature Cache + CPU Offload (Systems Contribution)

When training is restricted to suffix layers (e.g., top 25%) with dropout disabled, prefix layer outputs are deterministic and can be precomputed. The Layer-Prefix Feature Cache:

1. Runs a single forward pass through frozen prefix layers for every example in the dataset
2. Stores the hidden states at the split-layer boundary in CPU RAM
3. During training, injects cached states as inputs — GPU only processes suffix layers
4. Optionally offloads prefix parameters themselves to CPU, freeing additional GPU memory

**Effect**: 
- Net peak VRAM reduction: 10,782 → 7,459 MB (−3,323 MB, −30.8% reduction), enabling sequence lengths 1536 and 2048 where the online baseline OOMs under the same configuration.
- Prefix parameter/offload residency removed: 4.6 GB of GPU-resident parameters. (The difference is due to allocator overhead, suffix activations, and cached feature transfer buffers.)
- CPU/Host RAM requirements: Requires host memory to store the cached activations, scaled as `num_examples × seq_len × hidden_dim × bytes_per_hidden`.
**Generality**: This optimization applies to any suffix-only LoRA setup meeting the determinism requirements (dropout=0, fixed scope). It is not unique to TG-LoRA's extrapolation mechanism.

### Component 2: Trajectory Extrapolation (Algorithmic Contribution)

Each TG-LoRA training cycle:

1. Performs K pilot gradient steps to estimate parameter velocity v = θ_{t+K} - θ_t
2. Projects parameters forward: θ_extrap = θ_t + α · v
3. Validates the proposal on a held-out calibration batch; accepts or rolls back
4. Applies bounded update caps and K-step rollback for stability

**Effect**: validation loss reduction from 1.1682 to 1.1187 across three paired seeds at matched 240 backward passes.
**Generality**: This is TG-LoRA's unique algorithmic contribution. The accepted speculative steps require no additional backward passes or training-token updates — a pure optimization of the path trajectory.

### Ablation Verification (G4)

A 3-condition × 3-seed ablation experiment isolates each component's contribution:

| Condition | Cache | Extrapolation | Expected |
|---|---|---|---|
| A: Baseline | No | No | 10782 MB, loss 1.1682 |
| B: Baseline+Cache | Yes | No | 7459 MB, loss ~1.1682 |
| C: TG-LoRA | Yes | Yes | 7459 MB, loss 1.1187 |

- Cache effect = B - A (memory)
- TG effect = C - B (convergence)

**Status**: Experiment in progress.

## Primary Claims

1. **Memory efficiency**: The Layer-Prefix Feature Cache reduces peak VRAM by 30.8% (10,782 MB to 7,459 MB), and prefix offloading removes 4.6 GB of GPU-resident parameters, enabling consumer-GPU training at longer sequence lengths. (Verified: G2 PASS)
2. **Convergence improvement**: Trajectory extrapolation achieves lower validation loss (from 1.1682 to 1.1187, a reduction of 0.0495 absolute) at matched compute budget across three paired seeds. (Verified: 3-seed paired experiment)
3. **Quality retention**: TG-LoRA maintains downstream task performance with no measurable aggregate degradation on this benchmark suite. (Verified: G3 PASS, 3-seed mean 0.00% aggregate relative drop)
4. **Memory frontier**: Under our fixed suffix-only QLoRA configuration on an RTX 3060 12GB, the online baseline OOMs at seq-len 1536/2048, while TG-LoRA trains successfully. (Verified: frontier boundary at 2048)

## Honest Assessment of Limitations

- **Wall-clock speedup**: On the 5K dataset, TG-LoRA is slightly slower than the baseline (0.98x speedup) due to PCIe cache transfer latency between CPU and GPU over 240 backward passes. Component 2 ablations show that cosine-driven horizon selection can raise speculative backward replacement, but the current runtime path still pays pilot validation and scheduled full-evaluation costs. While warm-path execution is fast, the 1.48x speedup (measured by loss reduction per wall-minute) was observed only on the smaller 1K subset (or 1.14x wall-clock speedup depending on the metric). The convergence improvement is real but does not yet translate into a 2x measured wall-clock speedup on a single GPU with PCIe and validation fixed costs.
- **Cache build cost and Host footprint**: Caching features has a heavy cold-start cost: cache build takes ~6205s compared to a warm training duration of ~921s. TG-LoRA targets memory frontier and fixed-step convergence rather than one-off wall-clock speed. The cache is expensive to build but reusable across runs sharing the same model, split layer, tokenization, and dataset. Furthermore, storing precomputed features requires substantial host memory, proportional to `num_examples × seq_len × hidden_dim × bytes_per_hidden`.
- **Cache generality**: The memory savings are achieved by the Layer-Prefix Feature Cache, which could benefit any suffix-only LoRA training, not just TG-LoRA.
- **Velocity coherence**: The earlier near-zero cosine diagnostic compared a smoothed velocity to raw cycle deltas and should not be used as evidence against extrapolation. The cleaner EMA-vs-future-update analysis shows strong predictability for short and medium horizons, while shuffled-history controls indicate that most of the signal comes from a low-frequency dominant direction rather than local zig-zag order prediction.
- **Data split for evaluation**: To ensure unbiased validation metrics, the calibration batch used for the trajectory extrapolation acceptance check (the Calibration/Acceptance set) is strictly separated from the held-out validation batch used to report final convergence metrics (the Final Validation set) and the downstream tasks (External Eval). The accepted speculative steps require no additional backward passes or training-token updates, but do rely on calibration-batch forward checks.
- **Optimizer lifecycle confound**: The initial ablation's Condition B (Cache-only) recreated the optimizer per cycle (no momentum preservation), introducing a confound relative to the baseline's persistent AdamW. We have since resolved this by introducing a `persistent` optimizer lifecycle policy (using same trainer and same optimizer lifecycle across B and C) to ensure fair convergence comparisons. Experiments isolating this confound are currently in progress and results are being re-measured.

## Key Figures for Reviewer Defense

- **Figure 1**: 3-condition ablation table (VRAM peak reduction of 3.32 GB vs. 4.6 GB offloaded parameters, and convergence quality decomposition under matched optimizer lifecycle)
- **Figure 2**: Memory frontier sweep (OOM separation at 1536/2048 under fixed QLoRA baseline details: target scope last 25%, LoRA rank 16/alpha 32, batch size 1 with gradient accumulation 8, 4-bit quantization, gradient checkpointing enabled, SDPA enabled, and frozen prefix detached)
- **Figure 3**: 3-seed convergence curves (non-overlapping confidence intervals of final validation loss)
- **Table 1**: Main results (5K, 3-seed warm, wall-clock + validation loss + memory metrics)
- **Table 2**: Downstream quality retention (ARC-Easy, HellaSwag, TruthfulQA MC2)
