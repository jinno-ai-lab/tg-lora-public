# Paper Source Data Registry

## Purpose

この文書は、論文の主張を支える raw experimental artifacts へのリンク集です。
表や図に転記する値は、ここから raw artifact を開いて確認してください。

## Primary Canonical Artifacts

- one-shot 3-seed aggregate JSON: [../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.json)
- one-shot 3-seed aggregate markdown: [../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.md](../../runs/paper_memory_one_shot_suite_20260531_192119/aggregate_summary.md)
- external quality retention JSON (3-seed summary): [../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json)

## Per-seed Canonical Summaries

### Seed 42

- summary: [../../runs/paper_memory_one_shot_suite_20260531_192119/seed_42/coldwarm/summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/seed_42/coldwarm/summary.json)

### Seed 43

- summary: [../../runs/paper_memory_one_shot_suite_20260531_192119/seed_43/coldwarm/summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/seed_43/coldwarm/summary.json)

### Seed 44

- summary: [../../runs/paper_memory_one_shot_suite_20260531_192119/seed_44/coldwarm/summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/seed_44/coldwarm/summary.json)

## Supporting / Control Artifacts

- one-shot smoke aggregate: [../../runs/paper_memory_one_shot_offload_smoke_v2/aggregate_summary.json](../../runs/paper_memory_one_shot_offload_smoke_v2/aggregate_summary.json)
- one-shot smoke break-even: [../../runs/paper_memory_one_shot_offload_smoke_v2/aggregate_summary_break_even.json](../../runs/paper_memory_one_shot_offload_smoke_v2/aggregate_summary_break_even.json)
- reuse smoke aggregate: [../../runs/paper_memory_reuse_offload_smoke_v1/aggregate_summary.json](../../runs/paper_memory_reuse_offload_smoke_v1/aggregate_summary.json)
- reuse smoke break-even: [../../runs/paper_memory_reuse_offload_smoke_v1/aggregate_summary_break_even.json](../../runs/paper_memory_reuse_offload_smoke_v1/aggregate_summary_break_even.json)
- cache mode comparison JSON: [../../runs/paper_memory_offload_mode_compare_smoke_v2.json](../../runs/paper_memory_offload_mode_compare_smoke_v2.json)

## Component 2 Mechanism Artifacts

- offline predictability CLI: [../../scripts/analyze_extrapolation_predictability.py](../../scripts/analyze_extrapolation_predictability.py)
- runtime cosine-N ablation launcher: [../../scripts/run_cosine_n_ablation.sh](../../scripts/run_cosine_n_ablation.sh)
- runtime cosine-N ablation summarizer: [../../scripts/summarize_cosine_n_ablation.py](../../scripts/summarize_cosine_n_ablation.py)
- cosine-N skip config: [../../configs/9b_tg_lora_cosine_n_skip_persistent.yaml](../../configs/9b_tg_lora_cosine_n_skip_persistent.yaml)
- completed cosine-N runtime ablation summary: [../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.json](../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.json)
- completed cosine-N runtime ablation report: [../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.md](../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.md)
- completed validation-skip runtime ablation summary: [../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json](../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json)
- completed validation-skip runtime ablation report: [../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.md](../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.md)
- validation-skip runtime log: [../../runs/cosine_n_skip_ablation_20260603_083151/run.log](../../runs/cosine_n_skip_ablation_20260603_083151/run.log)
- final-eval-only runtime smoke summary (seed 42): [../../runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.json](../../runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.json)
- final-eval-only runtime smoke report (seed 42): [../../runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.md](../../runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.md)
- runtime ablation output pattern: `runs/cosine_n_ablation_*/cosine_n_ablation_summary.json`
- validation-skip ablation output pattern: `runs/cosine_n_skip_ablation_*/cosine_n_ablation_summary.json`

As of 2026-06-03, the validation-skip run above is complete. It should be cited
as a fixed-cost diagnostic: cosine-driven `N` still raises `reduction_rate`, but
wall-clock remains fixed-N equivalent because pilot validation and scheduled full
evaluation still dominate runtime.

The final-eval-only smoke is a single-seed diagnostic, not a paper-level
3-seed result. It shows that reducing scheduled full eval points sharply lowers
runtime and should be expanded to 3 seeds before manuscript claims are made.

## Source Data Usage Rules

1. main text / tables の primary evidence には one-shot canonical suite を使う。
2. smoke / reuse artifacts は mechanism sanity check か control surface としてだけ使う。
3. seed-level interpretation が必要なときだけ per-seed summary を開く。
4. external quality retention の multi-seed 数値は [../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json](../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_3seeds_summary.json) を正本とする。単一 best-checkpoint 比較の [../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_results.json](../../runs/paper_memory_one_shot_suite_20260531_192119/external_eval_results.json) は補助 artifact としてのみ扱う。
