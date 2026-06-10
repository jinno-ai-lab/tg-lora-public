# Paper Writing Map

## Purpose

この文書は、論文の各 section ごとに参照すべき doc / config / raw artifact を対応付ける map です。

## Section Map

| Paper section                | Primary docs                                                                                                             | Primary raw artifacts                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Title / Abstract             | [../abstract.md](../abstract.md), [../paper_results_snapshot.md](../paper_results_snapshot.md)                           | [../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json), [../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json)                                                                           |
| Introduction / Claim framing | [../paper_experiment_plan.md](../paper_experiment_plan.md), [../eval_plan_and_status.md](../eval_plan_and_status.md)     | [../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json)                                                                                                                                                                                                    |
| Method                       | [../abstract.md](../abstract.md), [../hyperparameters.md](../hyperparameters.md), [04_component2_positioning.md](04_component2_positioning.md)                                         | [../../configs/9b_baseline_suffix_only_last25.yaml](../../configs/9b_baseline_suffix_only_last25.yaml), [../../configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml](../../configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml), [../../configs/9b_tg_lora_cosine_n_persistent.yaml](../../configs/9b_tg_lora_cosine_n_persistent.yaml), [../../configs/9b_tg_lora_cosine_n_skip_persistent.yaml](../../configs/9b_tg_lora_cosine_n_skip_persistent.yaml) |
| Dataset / Setup              | [../datasets.md](../datasets.md), [../runbook.md](../runbook.md), [../evaluation.md](../evaluation.md)                   | [../../data/train.jsonl](../../data/train.jsonl), [../../data/valid_quick.jsonl](../../data/valid_quick.jsonl), [../../data/valid_full.jsonl](../../data/valid_full.jsonl), [../../data/gold_test.jsonl](../../data/gold_test.jsonl)                                                                                                                                                                                           |
| Main results                 | [../paper_results_snapshot.md](../paper_results_snapshot.md)                                                             | [../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json), [../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json)                                                                           |
| Ablation / Control           | [../paper_results_snapshot.md](../paper_results_snapshot.md), [../paper_experiment_plan.md](../paper_experiment_plan.md), [04_component2_positioning.md](04_component2_positioning.md) | [../../runs/paper_memory_one_shot_offload_smoke_v2/aggregate_summary.json](../../runs/paper_memory_one_shot_offload_smoke_v2/aggregate_summary.json), [../../runs/paper_memory_reuse_offload_smoke_v1/aggregate_summary.json](../../runs/paper_memory_reuse_offload_smoke_v1/aggregate_summary.json), [../../runs/paper_memory_offload_mode_compare_smoke_v2.json](../../runs/paper_memory_offload_mode_compare_smoke_v2.json), [../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.json](../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.json), [../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json](../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json) |
| Limitations / Future work    | [../eval_plan_and_status.md](../eval_plan_and_status.md), [../paper_experiment_plan.md](../paper_experiment_plan.md)     | next source data is not available yet; future frontier artifacts will be created by [../../scripts/run_frontier_sweep.sh](../../scripts/run_frontier_sweep.sh)                                                                                                                                                                                                                                                                 |

## Safe vs Unsafe Writing Boundary

### Safe now

- C1-supporting efficiency + memory narrative on 12GB hardware
- quality retention confirmed on `arc_easy`, `hellaswag`, `truthfulqa_mc2`
- one-shot as main cache surface, reuse as control surface
- G2 memory frontier claim under the matched 12GB CUDA setup
- Component 2 as low-curvature dominant-direction extrapolation, not as a
  general trajectory predictor
- cosine-driven `N` as a mechanism that increases `reduction_rate` safely in
  the completed runtime ablation
- validation-skip diagnostic as a fixed-cost diagnosis: skipped cycles did not
  roll back, but wall-clock stayed fixed-N equivalent because pilot/full eval
  costs remained

### Unsafe now

- strict preregistered G1 pass claim
- revolutionary memory claim
- 2x or larger measured wall-clock speedup claim from Component 2
- validation-skip as a measured speedup claim

## Next Missing Input

The next missing input for the Component 2 runtime claim is a fixed-cost
ablation that expands the final-eval-only smoke to 3 seeds and then reduces or
amortizes pilot validation. The completed validation-skip summary is:

- [../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json](../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json)

The seed 42 final-eval-only smoke is available here:

- [../../runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.json](../../runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.json)

After the fixed-cost ablation completes, update [02_source_data.md](02_source_data.md),
[04_component2_positioning.md](04_component2_positioning.md), and the main
results snapshot with the new wall-clock, validation-forward, rollback, and
best-valid-loss numbers.
