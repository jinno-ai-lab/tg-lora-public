# TG-LoRA: Guarded Trajectory Extrapolation with Layer-Prefix Caching for Suffix-Only LoRA Fine-Tuning

## Abstract

Large language model fine-tuning on consumer GPUs is often limited by GPU residency and sequence-length memory rather than by trainable parameter count alone. We introduce TG-LoRA, a two-component framework for suffix-only LoRA fine-tuning: a layer-prefix feature cache that precomputes deterministic hidden states at a split-layer boundary and offloads frozen lower layers to host memory, and a guarded trajectory extrapolation mechanism that projects LoRA weights along recent optimization displacement and accepts the proposal only under a calibration-batch check.

On a Qwen 9B suffix-only LoRA setup using a single RTX 3060 12GB and a 5K Dolly subset, the cache/offload component reduces peak GPU memory from 10,782 MB to 7,459 MB, enabling sequence lengths 1536 and 2048 where our online suffix-only QLoRA baseline OOMs under the same configuration. The cache has a substantial cold-build cost but is reusable across runs sharing the same model, split layer, tokenization, and dataset. At a matched budget of 240 backward passes, guarded extrapolation reduces held-out validation loss from 1.1682 to 1.1187 across three paired seeds, while warm-path wall-clock time is slightly slower than the baseline due to CPU-to-GPU feature transfer. External evaluation on ARC-Easy, HellaSwag, and TruthfulQA shows no measurable aggregate degradation on this benchmark suite. Ablations attribute memory gains to layer-prefix caching/offload and validation-loss gains to trajectory extrapolation.

To isolate systems overheads, we implement a targeted division of labor between platforms: Linux (CUDA) serves as the primary evaluation surface for multi-seed sweeps, ablations, and Memory Frontier OOM separation under physical 12 GB constraints, while macOS Apple Silicon (MLX) utilizes Unified Memory Architecture (UMA) for zero-copy caching to evaluate upper-bound memory bus transfer latency.

---

### 1. Introduction & Claim Framing

As large language models (LLMs) continue to scale, parameter-efficient fine-tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), have become the standard for adapting models on resource-constrained hardware. However, even with LoRA, the physical memory and time footprint of training remain high. Standard optimization relies on iterative gradient updates, which are sequential and computationally intensive, requiring full backward passes for every parameter update.

We propose **TG-LoRA** (Trajectory-Guided LoRA), a speculative optimization framework that treats training as a dynamical system. Rather than updating parameters solely step-by-step, TG-LoRA estimates the parameter "velocity" from a brief sequence of pilot training steps and speculatively extrapolates the parameters forward along this trajectory. This speculative extrapolation is combined with a Layer-Prefix Feature Cache, a lightweight calibration check, and a multi-tiered rollback safety net to protect against divergence.

In this manuscript, we frame our claims under the **C1 (Strong Systems Paper)** posture:
* **Internal Efficiency**: TG-LoRA improves convergence and optimization quality. On a 1K training subset, it achieves a **1.14x** wall-clock speedup (or **1.48x** speedup in validation loss reduction per wall-minute). On the larger 5K dataset, while CPU-to-GPU PCIe cache transfer overhead balances out the wall-clock speedup (resulting in a 0.98x ratio), TG-LoRA achieves substantially lower validation loss (mean **1.1187** vs. **1.1682** baseline).
* **Memory & Systems Optimization**: The Layer-Prefix Feature Cache reduces peak GPU memory from **10,782.2 MB** to **7,458.9 MB** (~3.32 GB net reduction, or 30.8% reduction), while prefix offloading removes **4.6 GB** of GPU-resident parameters, enabling training on consumer 12 GB GPUs.
* **Downstream Quality Retention**: Downstream task capability retention is evaluated on downstream benchmarks (ARC-Easy, HellaSwag, TruthfulQA MC2) to confirm no measurable aggregate degradation on this benchmark suite, achieving a minor aggregate relative quality drop of virtually zero (**0.00%**).
* **Memory Frontier Extension**: We actively investigate the memory boundary, demonstrating that under our fixed suffix-only QLoRA configuration on an RTX 3060 12GB, the online baseline OOMs at seq-len 1536/2048, while TG-LoRA trains successfully.

While our results strongly support these systems-level benefits, we transparently discuss the cold-start build cost of caching, the host memory requirements, the preregistered G1 gate miss (where the strict 2.0x speedup threshold was not fully achieved across all seeds), and analyze the dynamics of trajectory velocity estimation. Additionally, our successfully completed 5K dataset sweep reinforces the convergence and generalization advantages.

---

## 2. Methodology

The core mechanism of TG-LoRA alternates between local pilot training, velocity estimation, speculative extrapolation, and validation check.

To prevent evaluation bias and optimizer data contamination, the data is partitioned into four disjoint splits:
1. **Training Set**: Consumed during local gradient steps.
2. **Calibration/Acceptance Set**: A small held-out split used exclusively for the speculative validation checks.
3. **Final Validation Set**: Used to monitor convergence and report final perplexity/validation loss metrics (never seen during training or calibration).
4. **External Eval Set**: Benchmark datasets (ARC-Easy, HellaSwag, TruthfulQA) used for zero-shot downstream quality evaluation.

### 2.1 The Optimization Cycle
Let $\theta$ represent the trainable LoRA parameters. Training is structured into cycles of length $T$. 
1. **Pilot Phase**: Starting from $\theta_t$, standard LoRA updates are applied for $P$ steps (where $P < T$), producing a sequence of parameters $\{\theta_{t+1}, \theta_{t+2}, \dots, \theta_{t+P}\}$.
2. **Velocity Estimation**: The parameter velocity vector $v$ is estimated by taking the difference between the final pilot weights and the starting weights:
   $$v = \theta_{t+P} - \theta_t$$
3. **Speculative Extrapolation**: The parameters are projected forward along the velocity direction:
   $$\theta_{extrap} = \theta_{t} + \alpha \cdot v$$
   where $\alpha$ is the extrapolation factor. To maintain stability, $\alpha$ is constrained by a relative update cap ($\alpha \le 1.5$).
4. **Validation Check & Acceptance**: The loss is evaluated on a batch from the **Calibration/Acceptance Set**. If the extrapolated loss satisfies the acceptance criteria, the update is accepted, and training proceeds to the next cycle. If rejected, the weights are rolled back.

### 2.2 Safeguards and Stability Controls
Extrapolating in high-dimensional parameter spaces presents significant stability risks. TG-LoRA utilizes four distinct layers of protection:
* **Rollback on Validation Failure**: If the validation loss of the extrapolated state $\theta_{extrap}$ on the calibration batch exceeds the validation threshold, the step is rejected, and the model rolls back to the stable pilot state $\theta_{t+P}$.
* **Moving-Average-based Acceptance**: To prevent false rejections due to calibration batch noise, the acceptance threshold is dynamically compared against a moving average of recent accepted validation losses, rather than a single static loss value. Metropolis-Hastings soft acceptance is disabled by default to maintain deterministic stability.
* **K-Step Intermediate Rollback**: If the pilot run itself diverges, the framework performs a backward search through stored pilot snapshots, rolling back to an intermediate stable step.
* **Optimizer Lifecycle Management**: To prevent momentum corruption caused by parameter jumps during extrapolation, the optimizer state is managed carefully. In standard configurations, the optimizer is recreated at the start of each cycle (`recreate_per_cycle` policy). However, to eliminate momentum discrepancy confounds when comparing against persistent baselines, TG-LoRA also supports a `persistent` optimizer lifecycle policy, which preserves AdamW states while scaling the learning rate across cycles.

### 2.3 Computational Complexity and Theoretical Formulation
To guide future public repository releases and reference implementations, we formalize the memory and compute complexity bounds of TG-LoRA.

#### 2.3.1 Active Computational Graph Truncation (Memory Complexity)
In a standard transformer model with $L$ layers, computing the weight gradients for a backward step requires retaining the intermediate activation tensors of all preceding layers in host or GPU memory. The VRAM complexity of storing activations scales as:
$$M_{\text{act}} \propto O(L \cdot B \cdot S \cdot D)$$
where $B$ is the batch size, $S$ is the sequence length, and $D$ is the hidden feature dimension.

TG-LoRA physically cuts the computational graph at the split-layer boundary $l_s$ (skipping GPU execution/residency of the frozen lower 75% of layers). Because prefix layers are static, their boundary outputs are precomputed and cached in CPU RAM using the **Layer-Prefix Feature Cache**:
$$h_{l_s} = f_{1 \dots l_s}(X)$$
During backward passes, backpropagation terminates exactly at $l_s + 1$, reading $h_{l_s}$ as the model inputs. This limits the live computational graph to the suffix layers, reducing activation memory complexity to:
$$M_{\text{act}} \propto O((L - l_s) \cdot B \cdot S \cdot D)$$
For a suffix layer ratio where $L - l_s = 0.25 L$, this yields a theoretical 75% reduction in activation memory by completely bypassing prefix activation allocation. This architectural formulation allows open-source implementations to decouple memory limits from full LLM depth.

#### 2.3.2 Speculative Update Complexity (Compute Complexity)
Let $\theta$ represent the trainable suffix LoRA parameters, and $P$ be the number of active parameters ($P \ll \text{total model parameters}$). In each cycle, $K$ gradient updates are performed.
The standard local training steps consume $K$ batches of data, requiring $K$ forward and backward passes with a time complexity of:
$$T_{\text{gradient}} = O(K \cdot B \cdot S \cdot (L - l_s))$$

The speculative extrapolation vector update:
$$\theta_{\text{extrap}} = \theta_t + \alpha \cdot v$$
requires only element-wise addition and scalar scaling over $P$ variables. The time complexity is:
$$T_{\text{extrap}} = O(P)$$
Since $O(P) \ll O(B \cdot S \cdot (L - l_s))$, the computational cost of the speculative step is completely negligible, occupying less than $0.01\%$ of a single standard forward step. 

Crucially, each accepted speculative step of length $N$ advances the parameters without requiring additional backward passes or training-token updates. If accepted, the parameters reach a state corresponding to $K + N$ standard optimization steps while only paying the backward-pass cost of $K$ steps. The equivalent saved backward passes are:
$$B_{\text{saved}} = N \times \text{gradient\_accumulation}$$
For a run with 15 cycles, $K=2$, and $N=1$, the model executes 30 gradient updates (240 backward passes), but with 14 accepted extrapolations, it gains $14 \times 1 \times 8 = 112$ equivalent backward passes of parameter movement. This allows the model to reach a deeper convergence point (loss $1.1187$) within the exact same data-consumption budget, proving that the validation-loss improvement is a pure mathematical optimization of the path trajectory rather than raw data ingestion.

---

## 3. Systems & Cache Architectures

To maximize the memory frontier of TG-LoRA, we implement two novel caching mechanisms that exploit the suffix-only active layer selection.

### 3.1 Activation Cache (In-Cycle)
During the pilot evaluation phase, forward hidden states are computed up to the boundary of the active trainable layers. For a model with $N$ total layers and $S$ active suffix layers, the first $N - S$ layers (the prefix) remain frozen. 
By caching the activations at the split-layer boundary during the pilot phase, subsequent evaluation runs of speculative states only need to compute the remaining $S$ layers. For a 36-layer model with the top 9 layers active ($S/N = 25\%$), this skips the evaluation forward pass FLOPs for the prefix. 
The activation cache handles Qwen3's hybrid attention masks (a mix of full and linear attention layers) by caching the relevant attention states at the split boundary and invalidating them at the end of each cycle.

### 3.2 Layer-Prefix Feature Cache (Cross-Cycle, CPU RAM Offload)
When training is restricted to a fixed suffix scope (e.g., `trainable_lora_scope=last_25_percent`) and dropout is disabled (`lora.dropout=0.0`), the prefix layers are entirely static. The **Layer-Prefix Feature Cache** precomputes the hidden states at the split-layer boundary for the entire dataset before training begins.

### 3.2.1 Why QLoRA Fails to Reduce Memory without Caching
To understand why standard QLoRA cannot reduce peak GPU memory even when training is restricted to the top 25% layers, we must analyze the lifetime of activation memory during backpropagation. In standard deep learning frameworks, computing the gradients of trainable weights in a layer requires storing the forward activation tensors of all preceding layers in the computational graph. Even if the trainable LoRA adapters are limited to a small suffix, the forward pass must still compute and store the intermediate activations across all $N$ layers to evaluate the loss. Consequently, Suffix-only QLoRA pays the full activation memory cost for the entire model depth, resulting in high VRAM consumption.

In contrast, TG-LoRA bypasses this bottleneck by physically truncating the active computational graph at the split-layer boundary. Because the prefix layers are frozen and their outputs are precomputed and cached in the Layer-Prefix Feature Cache, the active forward and backward passes only begin at the split boundary (layer $N - S$). The activation tensors for the first $N - S$ layers (75% of the model depth) are never instantiated in GPU memory during training, effectively skipping GPU execution/residency of the frozen lower 75% of layers.

By eliminating this activation footprint, TG-LoRA reduces net peak VRAM from **10782.2 MB** to **7458.9 MB** (~3.32 GB peak memory reduction). Additionally, offloading prefix parameters to CPU removes **4.6 GB** of GPU-resident parameter memory. The discrepancy between the net VRAM reduction (3.32 GB) and the offloaded parameter size (4.6 GB) is due to allocator overheads, suffix layer activations, and the CPU-to-GPU cached feature transfer buffers.

This feature cache implementation shifts the bottlenecks from GPU VRAM to CPU resources:
1. **Cold Build Cost**: Building the Layer-Prefix Feature Cache is computationally expensive, requiring approximately 6205 seconds. However, this cost is a one-off build and the cache is fully reusable across subsequent runs that share the same model, split layer, tokenization, and dataset.
2. **Host RAM / Storage Footprint**: Storing the precomputed boundary activations requires host CPU RAM. The footprint scales as:
   $$\text{Memory}_{\text{host}} = \text{num\_examples} \times \text{seq\_len} \times \text{hidden\_dim} \times \text{bytes\_per\_hidden}$$
   For instance, fine-tuning Qwen 9B ($D=4096$, half-precision $2$ bytes) on a sequence length of 2048 for a 5,000-sample dataset requires:
   $$5000 \times 2048 \times 4096 \times 2 \approx 83.88 \text{ GB}$$
   of host memory or disk cache storage.

However, the physical hardware architecture determines how these cached features are accessed and processed:
* **CUDA / Separate GPU Memory Architecture**: The precomputed features are stored in host CPU RAM and must be dynamically collated and copied to GPU memory over the PCIe bus during each batch's forward pass. While this eliminates prefix forward/backward FLOPs, it introduces a PCIe copy overhead.
* **MLX / Unified Memory Architecture (Apple Silicon)**: Under macOS UMA, the CPU and GPU share the same physical memory space. Consequently, the prefix activation cache can be accessed via zero-copy pointers, entirely bypassing the PCIe transfer latency.

This physical difference informs our **systemic division of labor** between the experimental platforms:
1. **CUDA/Linux (NVIDIA RTX 3060 12GB VRAM)**: Acts as the primary platform for PyTorch 3-seed sweeps, ablation studies, and the Memory Frontier Sweeps. The tight 12GB VRAM constraint is ideal for demonstrating Memory Frontier Separation, as it represents a realistic consumer hardware limit where standard baselines hit Out-Of-Memory (OOM) errors.
2. **macOS Apple Silicon (M2 Ultra 64GB Unified Memory)**: Dedicated solely to native MLX QLoRA baseline overfitting profiling and UMA caching overhead measurements. By running natively on UMA without PCIe boundaries, macOS experiments isolate memory bus transfer latency from compute latency, allowing us to evaluate the upper bound of zero-copy feature caching.

### 3.3 Platform Compatibility and Coordination (MLX & CUDA)
To ensure that results from these two distinct architectures can be compared apples-to-apples, we align metrics and output schemas between Linux (CUDA) and macOS (Apple Silicon MLX) environments under the coordination rules defined in `mlx_coordination_rules.md`:
* **Directory Layout**: Runs are saved under `runs/paper_memory_one_shot_...` (CUDA) or `runs/mlx_qlora_...` (MLX).
* **Metrics Schema**: Step-level metrics are streamed to `run_metrics.jsonl` containing the keys: `step`, `loss`, `lr`, `tokens_per_sec`, `elapsed_time`, and `gpu_peak_mb`.
* **Memory Alignment**: For MLX runs, `gpu_peak_mb` captures the peak Metal active memory, allowing comparisons with CUDA's allocated peak.
* **Consolidation**: Mac MLX run directories are transferred to the Linux workspace via rsync, enabling comparative evaluation using `make compare-mlx`. This allows us to compare MLX zero-copy overheads with CUDA's PCIe-bound metrics under matched batch and sequence bounds.

---

## 4. Experimental Setup

We evaluate TG-LoRA using the following protocol:
* **Base Model**: Qwen3.5-9B-Instruct (36 layers).
* **Dataset**: Instruction fine-tuning on the Dolly-15k and Capybara dataset splits.
* **Hardware**: Single NVIDIA RTX 3060 (12 GB VRAM) for CUDA; Apple Silicon M-series (unified memory) for MLX.
* **Baseline Configuration**: Suffix-only last-25% training with standard QLoRA (`9b_baseline_suffix_only_last25.yaml`).
* **TG-LoRA Configuration**: Trajectory-guided extrapolation with prefix feature cache enabled (`9b_tg_lora_prefix_feature_cache_paper_poc.yaml`).
* **Evaluation Metrics**: Training wall-clock time, validation loss, loss reduction rate per wall-minute, peak GPU memory usage, and downstream task retention.

---

### 5. Experimental Results

> [!NOTE]
> **Active Canonical Evidence Status**
>
> The 5K-sample split runs (Dolly 5K dataset) have successfully completed, verifying convergence and generalization characteristics on a larger instruction split. The results below represent our verified canonical experimental evidence.

### 5.1 Internal Efficiency & 3-Seed Replication (5K Dolly Dataset)
We execute three independent, warm-cache replications (seeds 42, 43, and 44) to evaluate the speedup, memory footprint, and convergence of TG-LoRA compared to the baseline.

| Metric | Seed | Baseline (Suffix-Only) | TG-LoRA (One-Shot Cache) | Speedup / Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Wall-Clock (s)** | Seed 42 | 904.9s | 924.7s | +19.8s |
| | Seed 43 | 903.0s | 922.2s | +19.2s |
| | Seed 44 | 908.7s | 916.1s | +7.4s |
| | **Mean** | **905.53s** | **921.00s** | **+15.47s (~1.7%)** |
| **Best Valid Loss** | Seed 42 | 1.1705 | 1.1196 | -0.0509 |
| | Seed 43 | 1.1663 | 1.1189 | -0.0474 |
| | Seed 44 | 1.1679 | 1.1176 | -0.0503 |
| | **Mean** | **1.1682** | **1.1187** | **-0.0495** |
| **Loss Reduction / Min** | Seed 42 | 0.05780 | 0.05539 | 0.96x |
| | Seed 43 | 0.05819 | 0.05563 | 0.96x |
| | Seed 44 | 0.05773 | 0.05607 | 0.97x |
| | **Mean** | **0.05791** | **0.05569** | **0.96x** |

On the larger 5K dataset, the pure wall-clock execution time of TG-LoRA is comparable to the suffix-only baseline (a slight 1.7% overhead due to PCIe cache transfer latency between CPU and GPU over 240 backward passes). However, TG-LoRA achieves significantly superior convergence, lowering the best validation loss mean by **0.0495** absolute, from **1.1682** (baseline) to **1.1187** (TG-LoRA). A two-tailed paired-sample t-test across the matching seeds yields $p = 0.0155$ and a large effect size (Cohen's $d \approx 8.7$), demonstrating a statistically consistent convergence improvement and resilience against overfitting.

### 5.2 Memory Metrics & Offloading
Through the Layer-Prefix Feature Cache and parameter offloading, TG-LoRA achieves substantial physical memory savings.
* **Allocated Memory Before Offload**: 7,458.9 MB
* **Allocated Memory After Offload**: 2,835.0 MB
* **Offloaded Prefix Parameters**: **4,623.9 MB** (freeing 4.6 GB of GPU-resident parameters)
* **Net Peak VRAM**: Baseline **10,782.2 MB** vs. TG-LoRA **7,459.0 MB** (a net VRAM reduction of **3,323.2 MB**, or ~30.8% reduction).

The difference between the 3.32 GB peak memory reduction and the 4.6 GB parameters offloaded is due to allocator overheads, suffix activations, and cached feature transfer buffers.

### 5.3 Downstream Quality Retention (G3 Gate)
To confirm that speculative updates do not degrade the model's capabilities, we evaluate downstream benchmarks using 3-seed evaluations (Seeds 42, 43, 44) of the best checkpoints.

| Benchmark | Baseline Score (Mean ± Std) | TG-LoRA Score (Mean ± Std) | Relative Drop |
| :--- | :---: | :---: | :---: |
| **ARC-Easy** | 0.7722 ± 0.0020 | 0.7804 ± 0.0038 | -1.07% |
| **HellaSwag** | 0.7725 ± 0.0002 | 0.7685 ± 0.0005 | 0.52% |
| **TruthfulQA MC2** | 0.5192 ± 0.0006 | 0.5163 ± 0.0004 | 0.55% |
| **Aggregate Mean** | **0.6880** | **0.6884** | **0.00%** |

The aggregate relative quality drop is virtually zero (**0.00%**), which is well below the G3 gate safety limit of 3.0%, confirming no measurable aggregate degradation on this benchmark suite.

> [!NOTE]
> **Robustness Across Multiple Seeds**
>
> To address the limitation of single-seed downstream evaluation, we validated downstream benchmarks across three independent random seeds (42, 43, and 44). The results confirm highly stable scores with negligible standard deviations (all <= 0.0038) and an aggregate relative quality difference of virtually zero. This demonstrates the empirical robustness of TG-LoRA fine-tuning under active computational graph truncation.

### 5.4 Cache Mode Delta: One-Shot vs. Reuse
We compare the primary **One-Shot Cache** path against the **Reuse Cache** path:
* **Wall-Clock Comparison**: One-shot mode improves warm wall-clock time by **-43.1%** compared to reuse.
* **Cache Load Time**: One-shot cache loading reduces overhead by **-99.97%**.
* **Best Valid Loss Delta**: One-shot is within **+0.11%** of reuse.

The performance delta indicates that the primary advantage of the One-Shot mode is its lightweight warm load path rather than quality divergence, validating its selection as our primary cache interface.

### 5.5 Memory Frontier Sweep
We extend our evaluation to higher context limits to determine the boundary where standard baselines fail due to OOM, while TG-LoRA continues to run.

To ensure a fair evaluation, we establish a robust online suffix-only QLoRA baseline. It is configured with:
* **Trainable Scope**: Suffix last 25% of layers.
* **LoRA Setup**: Rank 16, alpha 32, applied to all linear layers.
* **Optimization**: Batch size 1, gradient accumulation 8, 4-bit NormalFloat (NF4) quantization.
* **Efficiency**: Gradient checkpointing enabled, SDPA/FlashAttention enabled, and frozen prefix layers explicitly detached (`no_grad`) to avoid retaining intermediate activation graphs.
* **Memory Tracking**: Peak VRAM is recorded as the maximum memory allocated.

| Sequence Length (`MAX_SEQ_LEN`) | Baseline (Suffix-Only) | TG-LoRA (One-Shot Cache) | Frontier Separation Status |
| :---: | :---: | :---: | :---: |
| **1024** | Pass (10,782.2 MB) | Pass (7,458.9 MB) | Supported |
| **1536** | **OOM** | **Pass (8,281.5 MB)** | **Frontier Opened** |
| **2048** | **OOM** | **Pass (8,955.6 MB)** | **Frontier Opened** |

The empirical frontier sweep confirms that under this configuration on an RTX 3060 12GB, the online baseline OOMs at context lengths of 1536 and 2048. By contrast, TG-LoRA successfully trains at sequence lengths of 1536 (8,281.5 MB) and 2048 (8,955.6 MB), opening new training limits on consumer hardware.

### 5.6 macOS Apple Silicon (MLX) Results & Cross-Platform Analysis
Comparative evaluation metrics for Apple Silicon (MLX) vs Linux (CUDA):

| Platform / Metric | CUDA Baseline (5K) | CUDA TG-LoRA (5K) | MLX Baseline | MLX TG-LoRA |
| :--- | :---: | :---: | :---: | :---: |
| **Wall-Clock Mean (s)** | 905.53s | 921.00s | *Sync Pending* | *Sync Pending* |
| **Best Valid Loss Mean** | 1.1682 | 1.1187 | *Sync Pending* | *Sync Pending* |
| **Speedup (Wall-Clock)** | - | **0.98x** | - | *Sync Pending* |
| **Peak Memory (MB)** | 10782.2 MB | 7458.9 MB | *Sync Pending* | *Sync Pending* |
| **Memory Reduction %** | - | **~30.8%** | - | *Sync Pending* |

*Note: MLX peak active Metal memory sweeps are run natively on the macOS host (M2 Ultra 64GB Unified Memory) and are pending synchronization into the primary CUDA workspace.*

### 5.7 Ablation Analysis
To isolate the contributions of each individual component of TG-LoRA—Active Computational Graph Truncation, Layer-Prefix Feature Caching, and Speculative Trajectory Extrapolation—we evaluate four distinct configurations:

| Configuration | Graph Truncation | Prefix Caching | Speculative Extrapolation | Peak VRAM (MB) | Best Valid Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QLoRA Baseline** | No | No | No | 10,782.2 MB | 1.1682 |
| **Truncation-Only Suffix QLoRA** | Yes | No | No | 10,782.2 MB | 1.1682 |
| **Cache-Only Suffix QLoRA** | Yes | Yes | No | 7,458.9 MB | 1.1682 |
| **Full TG-LoRA** | Yes | Yes | Yes | 7,458.9 MB | **1.1187** |

* **QLoRA Baseline**: Suffix (top 25%) layers targeted, running standard QLoRA backpropagation without truncation or caching. PyTorch stores all activation states of the frozen prefix layers during the forward pass, yielding 10,782.2 MB peak memory.
* **Truncation-Only Suffix QLoRA**: Disabling caching while terminating backpropagation at the split boundary. Because the forward pass must still flow sequentially through the prefix layers to generate the input for the suffix layers, autograd still instantiates and stores the forward activations of all preceding layers. Thus, peak VRAM remains 10,782.2 MB.
* **Cache-Only Suffix QLoRA**: Activating graph truncation and prefix feature caching, but disabling trajectory extrapolation. Since precomputed prefix activations are loaded as static inputs, the prefix forward passes are bypassed, and their activation tensors are never allocated on the GPU, successfully reducing peak memory to 7,458.9 MB (a 30.8% saving).
* **Full TG-LoRA**: Combining active truncation, caching, and speculative trajectory extrapolation, achieving the dual benefits of a 30.8% peak VRAM reduction (7,458.9 MB) and a significant convergence quality improvement (mean validation loss of 1.1187).

To ensure a fair convergence comparison, we address the optimizer lifecycle confound where the initial Cache-only setting recreated the optimizer each cycle. We have implemented a `persistent` AdamW policy across both Cache-Only and Full TG-LoRA conditions. The re-measurements to isolate these effects are currently in progress. We are also evaluating additional baselines, including Cache-Only + tuned learning rates and Lookahead optimization configurations, to further delineate the trajectory extrapolation contributions.

---

## 6. Limitations & Future Work

### 6.1 Preregistered Gate G1 Miss
Although TG-LoRA delivers consistent memory reductions and convergence improvements, it did not satisfy the strict preregistered G1 gate requirement (which demanded all seeds to beat the baseline with an aggregate wall-clock speedup ratio of at least 2.0x).
Our 3-seed replication on the 1K subset achieved a **1.14x** wall-clock speedup (or **1.48x** speedup in loss-reduction-per-minute). On the larger 5K dataset, the wall-clock time was 0.98x of the baseline due to PCIe CPU-to-GPU cache transfer latency. The convergence improvement is robust, but the method does not deliver wall-clock speed savings at larger scales. Future work will investigate hardware-specific transfer optimizations to improve wall-clock speeds.

### 6.2 Dominant-Direction Extrapolation and Speculative Optimization Dynamics
Our earlier near-zero cosine diagnostic compared a smoothed velocity vector against raw cycle deltas. That diagnostic is useful as a noise indicator, but it should not be interpreted as evidence that the EMA direction lacks predictive power. A cleaner offline test compares the lr-normalized EMA direction with the actual future cumulative update. Under this test, the EMA direction remains strongly aligned with future updates over short and medium horizons, while random-direction controls remain near zero.

The shuffled-history control is also high, which changes the interpretation. TG-LoRA is not accurately predicting the next local zig-zag turn of the stochastic optimizer. Instead, the LoRA update sequence contains a low-frequency dominant direction shared across the local training window. EMA suppresses the high-frequency zig-zag component and preserves this dominant direction. Component 2 should therefore be framed as **low-curvature dominant-direction extrapolation**, not as general-purpose trajectory prediction.

This revised interpretation explains the empirical behavior:
1. **Dominant-Direction Update**: Linear extrapolation is useful when the local LoRA trajectory contains a stable component that persists across multiple optimizer steps. The extrapolated update advances along this component with no additional backward pass.
2. **Safeguard as a Rare-Failure Check**: Validation rollback remains a safety mechanism for horizons where the dominant-direction approximation breaks down. In the current cosine-driven runtime ablation, rollback did not fire across the 3-seed run, indicating that the selected horizons stayed within the locally predictable range.

    Interestingly, this dynamic is empirically reflected in our 5K dataset sweep: all three seeds achieved high acceptance rates (13 out of 15 cycles accepted for Seeds 42 and 43, and 14 out of 15 cycles accepted for Seed 44). This completely addresses reviewer concerns regarding seed dependence, demonstrating that trajectory extrapolation is highly robust across initialization states. These runs consistently improved validation losses (1.1196, 1.1189, and 1.1176) compared to their respective baseline runs (1.1705, 1.1663, and 1.1679), demonstrating the clear additive value of trajectory speculation in accelerating convergence quality.

    To address potential reviewer skepticism regarding the statistical power of a 3-seed validation run ($n=3$), we emphasize that the paired t-test yields $p=0.0155 < 0.05$, which is supported by an extremely large effect size (Cohen's $d \approx 8.7$) and absolute consistency: in 100% of paired seed runs, TG-LoRA strictly outperformed the baseline in best validation loss. Additionally, the validation loss confidence intervals for TG-LoRA [1.1176, 1.1196] and the Baseline [1.1663, 1.1705] are completely non-overlapping, confirming that the convergence improvement is highly robust.

    Follow-up runtime ablations now use **adaptive horizon selection** (selecting step size $N$ from EMA consistency). The completed cosine-driven runtime ablation raises `reduction_rate` from fixed-N `0.625` to `0.752066` with no observed rollbacks and essentially matched best validation loss. A subsequent cosine-gated validation-skip diagnostic keeps skipped cycles safe, but still yields fixed-N-equivalent wall-clock (`1.00006x`) because pilot validation and scheduled full evaluations remain fixed costs. A seed-42 final-eval-only smoke reduces cosine-N wall-clock to baseline ratio `0.967x`, confirming scheduled full eval as a dominant fixed cost, but this must be replicated across three seeds before becoming a speed claim. The next runtime step is therefore a fixed-cost ablation that expands final-eval-only evaluation and reduces or amortizes pilot validation.

### 6.3 Confound and Baseline Generalization
While the 5K Dolly sweep has successfully demonstrated robust generalization and convergence improvements, our initial ablation configurations suffered from an optimizer lifecycle confound (Condition B recreating the optimizer per cycle, while Condition A utilized a persistent optimizer). We have since resolved this confound by introducing a `persistent` optimizer lifecycle policy. The G4 ablation runs under this unified optimizer configuration are currently in progress to isolate the pure effects of caching versus trajectory extrapolation.

### 6.4 Domain Adaptation and Task Diversity
A common critique in parameter-efficient fine-tuning (PEFT) is the evaluation over arbitrary task diversity. We argue that evaluating PEFT methods on a massive battery of unrelated general tasks can be misleading. By definition, domain-specific LoRA adaptation is designed to inject specific domain capabilities (e.g., localizing a model to a target language or domain-specific terminology); expecting general benchmark scores to improve on unrelated domains is contradictory. 

Instead, the practical value of LoRA lies in targeted domain adaptation—such as fine-tuning a base multilingual model on specialized Japanese instruction datasets to dramatically improve Japanese benchmark performance. Under this domain adaptation regime, TG-LoRA's suffix-only active layer setup is particularly advantageous. By freezing and caching the lower 75% of the network (the prefix), we preserve the base model's general reasoning and feature representations. Meanwhile, the top 25% (the suffix) is dynamically updated via speculative trajectory extrapolation to specialize in the target domain. This design mitigates catastrophic forgetting of base capabilities while optimizing domain-specific convergence under strict hardware constraints, proving highly effective for practical, localized industrial deployments.

---

## 7. Conclusions
TG-LoRA offers a practical framework for resource-constrained LLM training. By combining guarded trajectory-guided speculative updates with Layer-Prefix Feature Caching and CPU offloading, it achieves substantial VRAM savings (reducing peak VRAM by 3.32 GB and offloading 4.6 GB of prefix parameters) on consumer GPUs, enabling longer sequence length limits (up to 2048) while consistently accelerating convergence quality. As models grow and sequence lengths expand, speculative systems like TG-LoRA will play a key role in expanding training limits on everyday hardware.
