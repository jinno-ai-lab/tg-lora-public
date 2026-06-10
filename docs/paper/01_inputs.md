# Paper Inputs Registry

## Purpose

この文書は、論文執筆時に参照する入力資料の正本一覧です。
本文・数値・設定はここに複製せず、元ファイルへ直接リンクします。

## Core Paper Docs

- results snapshot: [../paper_results_snapshot.md](../paper_results_snapshot.md)
- experiment plan: [../paper_experiment_plan.md](../paper_experiment_plan.md)
- status sheet: [../eval_plan_and_status.md](../eval_plan_and_status.md)
- abstract draft: [../abstract.md](../abstract.md)
- dataset specification: [../datasets.md](../datasets.md)
- evaluation protocol: [../evaluation.md](../evaluation.md)
- runbook: [../runbook.md](../runbook.md)
- hyperparameter guide: [../hyperparameters.md](../hyperparameters.md)
- literature research hub: [../research/README.md](../research/README.md)

## Canonical Configs

### Main comparison surface

- baseline suffix-only last-25%: [../../configs/9b_baseline_suffix_only_last25.yaml](../../configs/9b_baseline_suffix_only_last25.yaml)
- TG prefix-feature-cache paper PoC: [../../configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml](../../configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml)

### Secondary / supporting configs

- deterministic paper PoC without prefix feature cache: [../../configs/9b_tg_lora_paper_poc.yaml](../../configs/9b_tg_lora_paper_poc.yaml)
- adaptive K5 branch: [../../configs/9b_tg_lora_adaptive_k5.yaml](../../configs/9b_tg_lora_adaptive_k5.yaml)
- adaptive K5 no-conv branch: [../../configs/9b_tg_lora_adaptive_k5_no_conv.yaml](../../configs/9b_tg_lora_adaptive_k5_no_conv.yaml)
- one-shot cache PoC: [../../configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml](../../configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml)

## Canonical Dataset Inputs

### Prepared splits used by training / eval

- training split: [../../data/train.jsonl](../../data/train.jsonl)
- quick validation split: [../../data/valid_quick.jsonl](../../data/valid_quick.jsonl)
- full validation split: [../../data/valid_full.jsonl](../../data/valid_full.jsonl)
- gold test split: [../../data/gold_test.jsonl](../../data/gold_test.jsonl)

### Raw source datasets

- raw Dolly 15k: [../../data/raw/dolly_15k.jsonl](../../data/raw/dolly_15k.jsonl)
- raw Capybara: [../../data/raw/capybara.jsonl](../../data/raw/capybara.jsonl)

## Canonical Scripts

- paper memory suite runner: [../../scripts/run_paper_memory_suite.sh](../../scripts/run_paper_memory_suite.sh)
- external quality eval runner: [../../scripts/run_paper_external_eval.py](../../scripts/run_paper_external_eval.py)
- gate evaluator: [../../scripts/evaluate_paper_gates.py](../../scripts/evaluate_paper_gates.py)
- frontier sweep runner: [../../scripts/run_frontier_sweep.sh](../../scripts/run_frontier_sweep.sh)
- paper result exporter: [../../scripts/export_paper_results.py](../../scripts/export_paper_results.py)
- paper result consolidator: [../../scripts/consolidate_paper_results.py](../../scripts/consolidate_paper_results.py)
- cache mode comparison helper: [../../scripts/compare_paper_memory_modes.py](../../scripts/compare_paper_memory_modes.py)

## Minimal Writing Bundle

論文を書き始めるときに最低限開いておくべきもの:

- [../paper_results_snapshot.md](../paper_results_snapshot.md)
- [../paper_experiment_plan.md](../paper_experiment_plan.md)
- [../eval_plan_and_status.md](../eval_plan_and_status.md)
- [../abstract.md](../abstract.md)
- [../datasets.md](../datasets.md)
- [../evaluation.md](../evaluation.md)
- [../../configs/9b_baseline_suffix_only_last25.yaml](../../configs/9b_baseline_suffix_only_last25.yaml)
- [../../configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml](../../configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml)
