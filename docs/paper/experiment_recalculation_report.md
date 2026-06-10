# Experiment Recalculation & Audit Report (検算報告書)

This report logs the rigorous verification (検算) of all empirical claims, values, and metrics stated in the TG-LoRA manuscript. Every number mentioned in the draft has been cross-referenced and verified against the raw, low-level execution logs (`run_metrics.jsonl` and `summary.json` files).

> [!NOTE]
> All source paths listed below are located relative to the repository's `paper_evidence/` directory (either under `paper_evidence/paper_memory_one_shot_suite/` or `paper_evidence/frontier_sweep/`).

---

## 1. Metric Audit Checklist

The table below cross-references all claims in the paper with their raw values in the logs.

| Metric / Claim | Value in Paper | Value in Raw Logs | Verification Status | Source Log File / Path |
| :--- | :---: | :---: | :---: | :--- |
| **Seed 42 TG Best Valid Loss** | `1.1196` | `1.119611` | **Verified** | `seed_42/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Seed 43 TG Best Valid Loss** | `1.1189` | `1.118880` | **Verified** | `seed_43/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Seed 44 TG Best Valid Loss** | `1.1176` | `1.117645` | **Verified** | `seed_44/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Seed 42 Baseline Valid Loss** | `1.1705` | `1.170488` | **Verified** | `seed_42/coldwarm/warm/baseline/summary.json` |
| **Seed 43 Baseline Valid Loss** | `1.1663` | `1.166342` | **Verified** | `seed_43/coldwarm/warm/baseline/summary.json` |
| **Seed 44 Baseline Valid Loss** | `1.1679` | `1.167856` | **Verified** | `seed_44/coldwarm/warm/baseline/summary.json` |
| **Mean TG Valid Loss** | `1.1187` | `1.118712` | **Verified** | Calculated from above values |
| **Mean Baseline Valid Loss** | `1.1682` | `1.168228` | **Verified** | Calculated from above values |
| **Seed 42 Wall-Clock (TG / BL)** | `924.7s / 904.9s` | `924.7s / 904.9s` | **Verified** | `seed_42/coldwarm/summary.json` |
| **Seed 43 Wall-Clock (TG / BL)** | `922.2s / 903.0s` | `922.2s / 903.0s` | **Verified** | `seed_43/coldwarm/summary.json` |
| **Seed 44 Wall-Clock (TG / BL)** | `916.1s / 908.7s` | `916.1s / 908.7s` | **Verified** | `seed_44/coldwarm/summary.json` |
| **Mean Wall-Clock (TG / BL)** | `921.00s / 905.53s` | `921.00s / 905.533s`| **Verified** | Calculated from above values |
| **Peak GPU VRAM (TG / BL)** | `7458.9MB / 10782.2MB` | `7458.9MB / 10782.2MB`| **Verified** | `seed_42/coldwarm/summary.json` |
| **VRAM Memory Reduction** | `30.8%` | `30.82%` | **Verified** | Calculated from peak VRAM values |
| **Offload GPU Memory Freed** | `4623.9 MB` | `4623.9 MB` | **Verified** | `seed_42/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Offload Before / After** | `7458.9MB / 2835.0MB` | `7458.9MB / 2835.0MB` | **Verified** | `seed_42/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Extrapolation Count (Seed 42)**| `13 accepted` | `13` (13/14 accepted) | **Verified** | `seed_42/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Extrapolation Count (Seed 43)**| `13 accepted` | `13` (13/14 accepted) | **Verified** | `seed_43/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Extrapolation Count (Seed 44)**| `14 accepted` | `14` (14/14 accepted) | **Verified** | `seed_44/coldwarm/warm/tg_lora/run_metrics.jsonl` |
| **Frontier VRAM (`slen=1536`)** | `8281.5 MB` | `8281.5 MB` | **Verified** | `frontier_sweep/slen_1536/.../run_metrics.jsonl` |
| **Frontier VRAM (`slen=2048`)** | `8955.6 MB` | `8955.6 MB` | **Verified** | `frontier_sweep/slen_2048/.../run_metrics.jsonl` |
| **Frontier Baseline status** | `OOM` | `oom` | **Verified** | `frontier_sweep/frontier_report.json` |
| **Frontier TG-LoRA status** | `Pass` | `completed` | **Verified** | `frontier_sweep/frontier_report.json` |
| **Downstream relative drop** | `0.00%` | `0.00028%` | **Verified** | `paper_memory_one_shot_suite/external_eval_3seeds_summary.json` |

---

## 2. Findings and Discrepancy Resolution

* **Extrapolation Typo (Resolved)**: The previous draft mistakenly asserted "Seed 42/43 had 0 accepted extrapolations" because QLoRA Baseline statistics (which have 0 extrapolations) were incorrectly copied into the TG-LoRA columns during summary aggregation. The raw step-level logs `run_metrics.jsonl` for both Seed 42 and Seed 43 confirm that **13 out of 14 extrapolation steps were successfully accepted** (a 92.8% acceptance rate).
* **Quality Drop Figure (Resolved)**: The value **0.00%** is mathematically verified as the aggregate 3-seed mean relative quality drop (0.00028%) between the best TG-LoRA and baseline checkpoints on ARC-Easy, HellaSwag, and TruthfulQA.

## 3. Verifiability and Audit Trail Status

All metrics are now **100% aligned, verified, and trace directly to raw files**. The database logs (`run_metrics.jsonl`) are preserved in the repository `paper_evidence/` folder and can be audit-checked at any time.
