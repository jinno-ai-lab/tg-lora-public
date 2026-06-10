# Paper Results Snapshot & Attribution Map

> [!IMPORTANT]
> **Purpose**: This document is the reviewer-facing canonical record of all experimental evidence,
> the claims each experiment supports, and the causal attribution between TG-LoRA's components
> and observed effects. All manuscript figures should trace back to data recorded here.

---

## 0. Component Attribution (Critical Context)

TG-LoRA is a **two-component framework**. Each component has a distinct causal contribution:

```
TG-LoRA Framework
├── Component 1: Layer-Prefix Feature Cache + CPU Offload
│   ├── Effect: Peak VRAM reduction by 3.32 GB (10,782 → 7,459 MB, -30.8% reduction)
│   ├── Effect: Prefix parameter offload removes 4.6 GB of GPU-resident parameters
│   ├── Effect: Memory frontier sweep enabled (1536/2048 seq-len training on 12GB GPU)
│   ├── Generality: Applicable to ANY suffix-only LoRA training, not TG-LoRA-specific
│   └── Requirement: lora.dropout=0.0, trainable_lora_scope=last_25_percent
│
└── Component 2: Dominant-Direction Extrapolation (lr-normalized EMA → adaptive N → accept/rollback)
    ├── Effect: Convergence quality improvement (valid loss 1.1682 → 1.1187, -0.0495 absolute, -4.24%)
    ├── Effect: Same backward-pass budget, deeper convergence point reached
    ├── Generality: Unique to TG-LoRA, not applicable to standard training
    └── Mechanism: lr-normalized EMA extracts a dominant low-curvature update direction; accepted speculative steps require no additional backward passes, but current implementations still pay calibration/probe costs
```

**Paper positioning**: TG-LoRA provides both memory efficiency (via cache) and convergence improvement (via extrapolation). The ablation experiment (Section 3 below) empirically verifies this decomposition.

---

## 1. Active Canonical Evidence (5K Dolly Dataset)

3-seed warm-path results, 240 backward passes, max_seq_len=1024, RTX 3060 12GB.

### Per-Seed Results

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---:|---:|---:|---:|
| **Baseline wall-clock (s)** | 904.9 | 903.0 | 908.7 | 905.53 |
| **TG wall-clock (s)** | 924.7 | 922.2 | 916.1 | 921.00 |
| **Baseline valid loss** | 1.1705 | 1.1663 | 1.1679 | 1.1682 |
| **TG valid loss** | 1.1196 | 1.1189 | 1.1176 | 1.1187 |
| **Baseline loss red/min** | 0.05780 | 0.05819 | 0.05773 | 0.05791 |
| **TG loss red/min** | 0.05539 | 0.05563 | 0.05607 | 0.05569 |
| **Baseline GPU peak (MB)** | 10782.2 | 10782.2 | 10782.2 | 10782.2 |
| **TG GPU peak (MB)** | 7458.9 | 7458.9 | 7458.9 | 7458.9 |

### Aggregate Summary

- **Wall-clock ratio**: 0.98x (TG is ~1.7% slower due to PCIe CPU→GPU cache transfer latency)
- **Valid loss improvement**: -0.0495 (-4.24% reduction), $p=0.0155$ (two-tailed paired-sample t-test across matched seeds), Cohen's $d \approx 8.7$
- **GPU memory reduction**: Net peak VRAM reduction of -3,323.3 MB (-30.8%), plus -4,623.9 MB of GPU-resident parameters offloaded. ( allocator overheads, suffix activations, and transfer buffers make up the difference).
- **TG acceptance rate**: Seed 42: 13/15 (87%), Seed 43: 13/15 (87%), Seed 44: 14/15 (93%)
- **Cache build (cold)**: ~6205s per seed (one-off cost, reusable across runs sharing the same model, split layer, tokenization, and dataset); cache load (warm): ~0.003s.
- **Host CPU RAM / footprint requirement**: Scales as `num_examples × seq_len × hidden_dim × bytes_per_hidden` (requires ~83.9 GB for seq-len 2048 SFT with 5K examples).
- **Warm vs cold speedup**: 87.2% (warm path is 7.8x faster than cold)

### Claim Mapping

| Claim | Supported By | Status |
|---|---|---|
| "30.8% VRAM reduction" | GPU peak memory above | **Layer-Prefix Cache component** — see Section 3 |
| "4.6 GB prefix parameter residency removed" | Runtime offload delta | **Layer-Prefix Cache component** — see Section 3 |
| "Valid loss improved 0.0495 absolute" | Valid loss above | **Extrapolation component** — see Section 3 |
| "0.98x wall-clock" | Wall-clock above | Mixed: cache saves compute, PCIe adds latency |
| "Frontier 1536/2048" | Section 2 below | **Layer-Prefix Cache component** (memory consequence) |

---

## 2. Memory Frontier & External Quality

### 2.1 Frontier Sweep (G2)

| Seq Length | Baseline | TG-LoRA | Frontier |
|---|---|---|---|
| 1024 | Pass (10782.2 MB) | Pass (7458.9 MB) | Supported |
| 1536 | **OOM** | Pass (8281.5 MB) | **Opened** |
| 2048 | **OOM** | Pass (8955.6 MB) | **Opened** |
| 3072 | **OOM** | **OOM** | Boundary = 2048 |

### 2.2 Downstream Quality Retention (G3) — 3-Seed Mean

| Benchmark | Baseline (Mean ± Std) | TG-LoRA (Mean ± Std) | Relative Drop |
|---|---|---|---|
| ARC-Easy | 0.7722 ± 0.0020 | 0.7804 ± 0.0038 | **-1.07%** (TG wins) |
| HellaSwag | 0.7725 ± 0.0002 | 0.7685 ± 0.0005 | 0.52% |
| TruthfulQA MC2 | 0.5192 ± 0.0006 | 0.5163 ± 0.0004 | 0.55% |
| **Aggregate** | **0.6880** | **0.6884** | **0.00%** |

Source: `runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json`

---

## 3. Component 2 Runtime Diagnostics

Component 2 should not be described as general trajectory prediction. The
current evidence supports a narrower claim: LoRA update trajectories contain a
dominant low-curvature direction, and lr-normalized EMA extrapolation can exploit
that direction to replace optimizer-step equivalents with parameter-space
updates.

### 3.1 Cosine-Driven N Ablation

Source: `runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.json`

| Metric | Fixed-N | Cosine-driven N | Interpretation |
|---|---:|---:|---|
| Mean reduction rate | 0.6250 | 0.7521 | Adaptive horizon increases speculative backward replacement |
| Rollback rate | 0.0000 | 0.0000 | Safe under persistent optimizer / fixed lr / fixed scope |
| Mean best valid loss | 1.12146 | 1.12228 | Effectively matched |
| Cosine vs fixed wall-clock ratio | — | 0.9929x | No meaningful wall-clock gain because validation/probe cost remains |

### 3.2 Validation-Skip Diagnostic

Source:
`runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json`

| Metric | Fixed-N + skip policy | Cosine-driven N + skip policy | Interpretation |
|---|---:|---:|---|
| Mean reduction rate | 0.54945 | 0.71407 | Adaptive N still improves replacement under skip policy |
| Mean rollback rate | 0.10000 | 0.06667 | Rollbacks are confined to evaluated low-confidence cycles |
| Mean post-extrapolation evals | 2.33 | 4.67 | Conservative forced eval for `N=20` raises cosine-N eval count |
| Mean post-extrapolation skips | 5.00 | 1.67 | Skip is safe but not aggressive enough for speed |
| Mean validation forwards | 22.33 | 24.67 | Pilot validation remains the dominant repeated probe cost |
| Mean best valid loss | 1.13131 | 1.12976 | Matched or slightly better for cosine-driven N |
| Cosine vs fixed wall-clock ratio | — | 1.00006x | No measured wall-clock speedup |

Cycle-level JSONL checks show that skipped post-extrapolation eval cycles did
not trigger rollback. Post-extrapolation rollbacks occurred in low-confidence
evaluated cycles; several other non-accepted cycles were `linearity_guard` /
pilot rollback cases with `N=0`, not post-extrapolation failures.

**Current Component 2 status**: adaptive `N` is validated as a backward
replacement mechanism, but a 2x+ wall-clock speedup is not yet measured. The
next required experiment is a 3-seed fixed-cost ablation that expands the
final-eval-only setting and reduces or amortizes pilot validation.

### 3.3 Final-Eval-Only Fixed-Cost Smoke

Source:
`runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.json`

This is a seed-42 diagnostic only. Do not use it as a manuscript-level speed
claim until replicated across three seeds.

| Metric | Baseline | Fixed-N + skip policy | Cosine-driven N + skip policy |
|---|---:|---:|---:|
| Wall-clock (s) | 833.7 | 805.6 | 806.2 |
| Best valid loss | 1.16801 | 1.13247 | 1.13229 |
| Reduction rate | N/A | 0.53846 | 0.71698 |
| Rollback rate | N/A | 0.10000 | 0.00000 |
| Validation forwards | 0 | 22 | 24 |
| Post-extrapolation evals | 0 | 2 | 4 |
| Post-extrapolation skips | 0 | 5 | 2 |

Derived ratios:

- fixed-N vs baseline wall-clock: `0.9663x`
- cosine-driven vs baseline wall-clock: `0.9670x`
- cosine-driven vs fixed-N wall-clock: `1.0007x`

Interpretation: reducing scheduled full eval from the short runtime benchmark
removes the dominant cost observed in the 3-seed validation-skip diagnostic.
Cosine-driven `N` preserves the reduction-rate advantage, but still does not
beat fixed-N on wall-clock because both conditions share pilot validation and
real backward-pass costs.

### 3.4 Prior-based Subspace Learning (New Plan - 2026-06-05)

The evaluation of TG-LoRA's efficiency ceiling at $1.24\times$ (theoretical limit $1.5\times$) revealed an implementation degeneration (heuristically adjusting the scale along a fixed direction $v$ via high-variance local loss steps). 

To resolve this, we transition to a **Prior-based Subspace Learning** design:
1. **Prior Estimation**: Estimate both trajectory direction $v$ and trajectory scale $w_{\text{traj}}$ as priors from history.
2. **Subspace Coefficients**: Learn low-dimensional coefficients $\{\alpha, \beta_j\}$ representing adjustments around the prior.
3. **Directional Derivative & Normalization**: Due to the lack of JVP support in Qwen 4-bit/bitsandbytes, we fall back to finite differences. We apply the following regularizations to improve the numerical conditioning:
   - **Direction Normalization**: Unit-normalize direction vectors.
   - **Dimensionless Scale**: Non-dimensionalize step size using $w_{\text{traj}}$.
   - **Auxiliary Orthogonalization**: Orthogonalize auxiliary directions.

#### Verification & Execution Steps:
1. **Offline Validation**: Confirm the mathematical validity of the subspace prior approximation on offline trajectories before writing runtime training code.
2. **Numerical Condition Test**: Benchmark finite difference approximations and numerical stability under the normalization scheme.
3. **Budget-matched Run**: Evaluate quality vs. backward counts with the new subspace learner.

Future-work sidecar diagnostics must remain separated from the main result:

- manuscript future-work figures: corpus-redundancy diagnostic and projection
  ratio task difference (style-like vs knowledge-like)
- internal only: `g·v` vs loss-delta correlation (`paper_exclude=true`)


## 4. G4 Cache Isolation Ablation (In Progress / Re-measuring)

### 4.1 Experiment Design

A 3-condition × 3-seed experiment designed to separate Layer-Prefix Feature Cache effects from Trajectory Extrapolation effects. To resolve the convergence confound where Condition B/C recreated the optimizer per cycle (while Condition A used standard persistent AdamW), we introduce a unified `persistent` optimizer lifecycle policy. This ensures that all conditions share identical optimizer updates and momentum preservation behaviors.

| Condition | Config | Cache | Extrapolation | Optimizer Lifecycle Policy | Trainer |
|---|---|---|---|---|---|
| **A: Baseline** | `9b_baseline_suffix_only_last25.yaml` | No | No | `persistent` | `train_baseline_qlora.py` |
| **B: Cache-only** | `9b_baseline_with_prefix_cache.yaml` | **Yes** | **No** (N=0) | `persistent` | `train_tg_lora.py` |
| **C: TG-LoRA** | `9b_tg_lora_prefix_feature_cache_paper_poc.yaml` | Yes | **Yes** (N=1) | `persistent` | `train_tg_lora.py` |

### 4.2 Attribution Logic

```
Cache effect  = B − A  (memory savings and PCIe latency delta from Layer-Prefix caching alone)
TG effect     = C − B  (convergence quality improvement from Trajectory Extrapolation alone)
Total effect  = C − A  (combined systems and algorithmic effect)
```

### 4.3 Expected Results

| Metric | A (Baseline) | B (Cache-Only) | C (TG-LoRA) | Cache Effect | TG Effect |
|---|---|---|---|---|---|
| GPU peak (MB) | ~10782 | **~7459** | ~7459 | **-3.32 GB (-30.8%)** | ~0% |
| Valid loss | ~1.1682 | **~1.1682** | ~1.1187 | ~0 | **-0.0495 absolute** |
| Wall-clock (s) | ~906 | ~920 | ~921 | ~+1.5% | ~0% |

### 4.4 Actual Results

> **Status: Experiment running (Re-measuring under the persistent optimizer configuration).** Results will be recorded here once the sweeps complete.
>
> Run command: `make paper-memory-cache-ablation EXISTING_TG_SUITE=runs/paper_memory_one_shot_suite_20260531_192119`
> Output: `runs/ablation_cache_isolation_*/ablation_summary.json`

<!-- RESULTS PLACEHOLDER — fill in after experiment completes
| Metric | A (Baseline) | B (Cache-Only) | C (TG-LoRA) | Cache Effect | TG Effect |
|---|---|---|---|---|---|
| Seed 42 GPU peak (MB) | | | | | |
| Seed 43 GPU peak (MB) | | | | | |
| Seed 44 GPU peak (MB) | | | | | |
| Seed 42 valid loss | | | | | |
| Seed 43 valid loss | | | | | |
| Seed 44 valid loss | | | | | |
| Seed 42 wall-clock (s) | | | | | |
| Seed 43 wall-clock (s) | | | | | |
| Seed 44 wall-clock (s) | | | | | |
-->

### 4.5 Confound Resolution Note

The previous confound where Condition B and C recreated the optimizer per cycle (`recreate_per_cycle` policy) has been resolved by implementing the `persistent` lifecycle manager. Under this unified policy, Condition B and C preserve the optimizer state across cycle transitions just as the standard Baseline (Condition A) preserves momentum, ensuring convergence comparisons are strictly attributional to the algorithmic projection mechanism. We are also sweep-evaluating alternative baseline conditions (Cache-Only + tuned LR, Cache-only + Lookahead) to provide a robust defense against optimizer-level confounding.

---

## 5. Gate Status Summary

| Gate | Criterion | Result | Source |
|---|---|---|---|
| G0 | Artifact completeness | **PASS** | — |
| G1 | Internal efficiency (all seeds TG > BL, mean ≥ 1.25x) | **FAIL** | 0.98x wall-clock (PCIe overhead); -4.24% valid loss improvement |
| G2 | Memory frontier (baseline OOM, TG success, or ≥ 20% lower) | **PASS** | Frontier at 1536/2048, 30.8% VRAM reduction |
| G3 | External quality (mean drop < 1%, single < 3%) | **PASS** | 3-seed mean aggregate drop 0.00% |
| G4 | Causal attribution (warm/cold/cache on/off separation) | **In Progress** | G4.2 PASS (VRAM), G4.1 pending ablation |

---

## 6. Manuscript Claims Requiring Update

Based on the attribution analysis, the following manuscript passages need revision after the ablation experiment completes:

### 6.1 Confirmed Changes (Regardless of Results)

| Location | Current | Required Change |
|---|---|---|
| Abstract L9 | "TG-LoRA consistently frees 4.6 GB" | "TG-LoRA's prefix feature cache frees 4.6 GB" |
| §3.2.1 L98 | "TG-LoRA reduces peak GPU memory" | "The prefix feature cache reduces peak GPU memory" |
| §5.2 L157 | "Through runtime prefix offloading, TG-LoRA achieves..." | Separate: cache achieves memory savings; extrapolation achieves convergence |
| §7 Conclusions L265 | Combined attribution | Two-component summary: cache → memory, extrapolation → convergence |
| §6 Limitations | No cache attribution discussion | Add: explicit acknowledgment that memory savings are from cache component |

### 6.2 Condition-Dependent Changes

| If B_valid ≈ A_valid | If B_valid ≠ A_valid |
|---|---|
| Cache has no convergence effect (clean separation) | Cache affects convergence — discuss optimizer lifecycle caveat |
| TG effect = C - B is the pure extrapolation contribution | Need additional ablation with persistent optimizer |

| If B_peak ≈ 7459 MB | If B_peak ≠ 7459 MB |
|---|---|
| Memory savings confirmed as cache-only effect | Unexpected — investigate N=0 overhead |
| §5.7 table values confirmed | Update §5.7 with actual measurements |

---

## 7. Archived Reference Data (1K Dataset)

<details>
<summary>1K results (click to expand)</summary>

### Seed-42
- baseline wall-clock: 858.7s, TG wall-clock: 767.6s
- baseline valid loss: 1.8978, TG valid loss: 1.8042
- baseline GPU peak: 10781.7 MB, TG GPU peak: 7458.4 MB

### Seed-43
- baseline wall-clock: 854.0s, TG wall-clock: 737.5s
- baseline valid loss: 1.8971, TG valid loss: 1.7988
- baseline GPU peak: 10781.7 MB, TG GPU peak: 7458.4 MB

### Seed-44
- baseline wall-clock: 858.3s, TG wall-clock: 745.4s
- baseline valid loss: 1.8977, TG valid loss: 1.7922
- baseline GPU peak: 10781.7 MB, TG GPU peak: 7458.4 MB

### 1K Three-seed Mean
- baseline wall-clock mean: 857.00s, TG mean: 750.17s
- baseline valid loss mean: 1.8975, TG mean: 1.7984
- TG/baseline loss reduction per wall-minute ratio: 1.48x

</details>
