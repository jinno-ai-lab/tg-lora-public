---
title: Repo Wiki Log
genre: repository-analysis
type: concept
sources:
  - extract-skill-meta planning artifacts
  - wiki regeneration events
related:
  - Repository Overview
  - Module Index
  - Repository Risk Register
  - Repo Wiki Schema
status: generated
---

# Repo Wiki Log

## [2026-05-30] comprehensive wiki generation | full 19-module coverage with architecture and risk analysis

- **scope**: Complete regeneration of `docs/llm-wiki/` from 4 files (3 trivial modules) to 25 files covering all 19 logical modules plus 6 concept pages.
- **pages created**:
  - [[Repository Overview]] — project identity, scale metrics, key results, architecture layers
  - [[Module Index]] — all 19 modules with dependency graph
  - [[Repository Risk Register]] — 28 risks (3 critical, 5 high, 12 medium, 8 low) with evidence and mitigation status
  - [[Processing Progress]] — architecture diagram, development timeline, experiment coverage, open work items
  - [[File Inventory]] — complete file listing with line counts across all 7 source packages
  - [[Repo Wiki Schema]] — wiki structure definition and content standards
  - 16 new module pages covering: Velocity Tracking, Extrapolation, Delta Tracking, Cycle State, Layer Sampling, Rollback Manager, Random Walk Controller, LoRA State, Trajectory Analysis, Cycle Monitor, Training Advisor, Activation and Prefix Cache, Training Loop, Configuration Schema, Optimizer Lifecycle, Batch Planning, Data Pipeline, Evaluation, Utilities
- **quality indicators**:
  - every module page includes: Role, Key Exports, Algorithm/Mechanism, Integration Points, Risk Signals
  - cross-references use `[[wiki-link]]` syntax throughout
  - risk register entries cite specific evidence from experiment runs and source code
  - architecture diagrams use ASCII art for tool-agnostic rendering
- **decision_reason**: Prior wiki covered only 3 peripheral modules (baselines, github, hypothesis). A 669-file research repo with active paper experiments requires comprehensive documentation to enable onboarding, audit, and future-self context recovery.

---

## [2026-05-25] mode-compare canonicalization | v2 compare artifact is now the current reference

- removed obsolete compare outputs `runs/paper_memory_offload_mode_compare_smoke_v1.json` and `runs/paper_memory_offload_mode_compare_smoke_v1.md` so generated inventories stop carrying stale mode-comparison references.
- regenerated `docs/llm-wiki/` via the canonical AI Hub bootstrap path with `AI_HUB_REPO_LLM_WIKI_REFRESH=1` and confirmed the refresh used `03_extract_skill_meta`.
- current comparison reference is `runs/paper_memory_offload_mode_compare_smoke_v2.json` with companion markdown `runs/paper_memory_offload_mode_compare_smoke_v2.md`; generated wiki pages now enumerate only the v2 compare artifact.

## [2026-05-25] paper-memory aggregate freeze | canonical 3-seed one-shot replication closed

- completed `runs/paper_memory_one_shot_ms3_s1024_20260525` through seed 44 and froze `aggregate_summary.json` / `aggregate_summary.md` as the canonical 3-seed one-shot reference for the paper-facing 1024-token surface.
- updated `docs/paper_results_snapshot.md` and `docs/paper_experiment_plan.md` from the earlier 2-seed interim posture to the completed 3-seed aggregate posture.
- the resulting warm aggregate keeps TG faster on all three seeds, lowers mean best validation loss (`1.8975 -> 1.7984`), improves mean `loss_red_per_wall_minute` to about `1.48x` baseline, and preserves the warm runtime-offload delta of about `4623.6 MB` freed.
- the strict pre-registered G1 rule remains open because seed 42 is a slight miss on warm `loss_red_per_wall_minute` and the 3-seed mean does not reach the planned `2x` threshold.

## [2026-05-27] static-review remediation | efficiency accounting and eval surfaces hardened

- fixed TG-LoRA cycle accounting to snapshot `pilot_K` / `pilot_lr` before controller adaptation, record actual micro backward passes separately from speculative optimizer steps, and keep pilot-full-rollback cycles at zero speculative steps.
- centralized CLI override handling so dotlist overrides are merged before preflight and training receives a typed OmegaConf rebuilt from Pydantic validation.
- switched `scripts/run_paper_external_eval.py` to warm adapter discovery plus PEFT-aware lm-eval invocation (`pretrained=<base>,peft=<adapter>,load_in_4bit=True`) instead of treating adapter checkpoints as standalone models.
- added standard QLoRA k-bit preparation, strict dtype alias validation, prompt masking controls (`training.train_on_prompt`), SciPy-backed statistical tests, richer run-header metadata for sweep analysis, and fixes for the paper/ablation/high-lr shell scripts' budgeting and override bugs.
- **modules affected**: [[Module Training Loop]], [[Module Configuration Schema]], [[Module Cycle State]], [[Module Evaluation]]

## [2026-05-27] tensor-only artifact load + quick-eval semantics | safer torch load and exact example caps

- added `src/utils/tensor_artifact.py` and moved training-state, prefix-feature-cache, trajectory-delta-artifact, and diagnose/recover state loads to `weights_only=True` tensor-only deserialization.
- kept artifact payloads in builtin containers plus tensors so cache/checkpoint reads no longer depend on arbitrary object reconstruction.
- changed quick validation call sites to honor `eval.quick_eval_examples` as an exact example cap via `max_examples`, including activation-cache pilot eval, instead of silently treating the field as a batch count.
- fixed follow-on regressions in CUDA runtime-error recovery (`train_tg_lora`) and test harness run-dir mocks so the safer load path and new eval semantics are covered by integration tests.
- **modules affected**: [[Module Utilities]], [[Module Evaluation]], [[Module Activation and Prefix Cache]]

## [2026-05-27] blocker remediation | OOM exit, resume adapter, prompt masking, device priority, proposal lifecycle, cache equivalence, frontier sweep

- OOM / CUDA errors now write `status: failed` to footer and raise `SystemExit(2)` instead of exiting 0.
- `--resume` restores LoRA adapter weights from the sibling `oom_checkpoint/` via `set_peft_model_state_dict`; `recover.py` suggests the correct `--resume` command.
- Prompt masking prioritises ChatML `<|im_start|>assistant\n` boundary over `rec["prompt"]`; warns when all labels are masked after truncation.
- `get_device_map` now treats explicit `model.device` as highest priority and returns `{"": device}` dict for single-device placement.
- RandomWalk proposal is created at cycle top and used for pilot K/lr/beta; `commit_proposal()` on accept, `penalize()` only on reject.
- Activation cache runs startup equivalence check on first batch, disabling itself if full-forward vs cached loss diverges by more than 0.01.
- Frontier sweep runs baseline and TG as separate sub-processes with independent exit codes so baseline OOM does not prevent TG execution.
- Shell script indentation, `diagnose.py --config` invocation, `eval_from_cache` batch-size weighting, and MPS+4bit preflight guard also fixed.
- **modules affected**: [[Module Training Loop]], [[Module Rollback Manager]], [[Module Random Walk Controller]], [[Module Activation and Prefix Cache]], [[Module Data Pipeline]], [[Module Utilities]]

## [2026-05-28] high-lr comparison summary fix | heredoc replaces shell-unsafe inline python

- `scripts/run_high_lr_comparison.sh` completed all runs but exited non-zero in the summary phase because the inline `python -c` payload embedded brace-heavy f-strings that broke shell quoting at runtime.
- replaced the summary block with a heredoc that reads `OUTPUT_BASE` from the environment, and updated `tests/test_operational_scripts.py` to assert the safer pattern.
- validated with `bash -n scripts/run_high_lr_comparison.sh`, `pytest tests/test_operational_scripts.py`, and the fixed summary logic against real output in `runs/high_lr_compare_smoke_20260528/`.

## [2026-05-28] post-hoc comparison reference guard | compare uses shared reference only when both sides actually share one

- added optional `comparison_reference` metadata to `run_metrics.jsonl` headers plus `comparison_reference_loss/kind` propagation through `src/utils/run_query.py` and `scripts/compare_runs.py`.
- pairwise reports now use header-based reference loss only when both runs expose the same finite reference kind; otherwise they fall back together to the older post-hoc path instead of mixing scales.
- dashboard aggregation similarly uses header references only when the whole run set shares one common reference kind.
- baseline pre-training reference eval is best-effort only: on single-GPU Qwen3.5-9B it can OOM before training starts, so the code now logs a warning and leaves the reference unset instead of crashing the run.
- validated the compare logic with `pytest tests/test_compare_runs.py` and by regenerating the existing smoke pairwise report for `runs/high_lr_compare_smoke_20260528`, which now reports `N/A` for unsupported efficiency deltas instead of misleading negative TG efficiency values.
- **modules affected**: [[Module Utilities]], [[Module Evaluation]]

## [2026-05-28] high-lr comparison parity mode | launcher can now enforce a shared backward-pass scale

- `scripts/run_high_lr_comparison.sh` now accepts `COMPARISON_MODE=stability|parity`.
- parity mode computes a shared comparable backward-pass unit from the baseline grad accumulation and TG cycle cost (`lcm(baseline_grad_accum, tg_grad_accum * K_initial)`), normalizes the requested budget to that unit, and rejects budgets that are too small to be comparable.
- the launcher summary now prints actual `total_backward_passes` per run so post-hoc inspection can immediately see whether runs were compute-matched.
- stability mode remains available for the original "high-LR robustness smoke" use case and explicitly warns that equal backward-pass budgets are not enforced.

## [2026-05-28] baseline 1024 OOM avoidance | optional pretrain reference eval is now baseline-configurable

- added `eval.baseline_pretrain_reference_eval_enabled` to the validated training schema and honored it in `src/training/train_baseline_qlora.py`.
- this disables only the optional baseline comparison-reference eval; it does not change the TG-LoRA cycle-0 rollback baseline path.
- verified that a fresh 1024-token baseline run which previously OOMed now completes when `eval.baseline_pretrain_reference_eval_enabled=false` is set.
- wired the same toggle into `scripts/run_high_lr_comparison.sh` and covered it in `tests/test_operational_scripts.py`.
- **modules affected**: [[Module Configuration Schema]], [[Module Training Loop]]

## [2026-05-28] mixed-reference dashboard fix | fallback now stays on one scale when baseline references are disabled

- `scripts/compare_runs.py` now uses the earliest train-loss fallback consistently whenever a shared finite header `comparison_reference` is unavailable, instead of mixing baseline train-loss starts with TG pretrain-valid references.
- `gather_runs()` now stores `fallback_initial_loss`, and multi-run dashboard rows use that fallback for all runs when the group does not share one finite reference scale.
- added regression coverage in `tests/test_compare_runs.py` for pairwise mixed-reference reports and multi-run mixed-reference suites.

## [2026-05-28] 1024 large-budget parity suite | baseline survives, TG high-lr variants diverge under 240-pass budget

- completed `runs/high_lr_compare_parity_20260528_1024_b240` with `COMPARISON_MODE=parity`, `BUDGET=240`, `MAX_SEQ_LEN=1024`, `QUICK_EVAL_EXAMPLES=4`, and `BASELINE_PRETRAIN_REFERENCE_EVAL_ENABLED=false` on the single RTX 3060 path.
- baseline runs all completed at the matched `240` backward-pass budget: `b_safe_lr` best valid `0.7261`, `b_2e-3` `0.8992`, `b_5e-3` `1.1772`.
- TG high-lr runs both diverged before completion on this 1024 / 240-pass surface: `tg_2e-3` last recorded at `232` backward passes, `tg_5e-3` at `224` backward passes.
- corrected dashboard analysis over the completed suite now reports the baseline runs from a consistent first-train-loss scale (`2.5050` initial loss) rather than mixing in TG-only pretrain references.
- **key result**: Speculative extrapolation at aggressive LRs (2e-3, 5e-3) is unstable at 1024 tokens without the linearity guard.

## [2026-05-28] linearity guard | skip speculative extrapolation when velocity signals degrade

- added `_should_fallback_to_baseline_like()` to `train_tg_lora.py` and five `linearity_guard_*` config fields to `TGLoRAParams`.
- the guard monitors velocity anomaly, positive acceleration, and acceptance-rate + pilot-stability. When triggered, the cycle keeps only the pilot update (N=0), preventing divergent speculative steps on the 1024 / high-budget surface.
- covered by 6 pure-function unit tests in `tests/test_training_pure_functions.py`.
- **modules affected**: [[Module Training Loop]], [[Module Configuration Schema]], [[Module Velocity Tracking]]

## [2026-05-28] 1024 guard probe rerun | N=0 fallback suppresses speculative divergence but exposes masked-label NaN path

- reran `tg_2e-3` and `tg_5e-3` at `MAX_SEQ_LEN=1024`, `training.max_cycles=10`, `eval.quick_eval_examples=4`, and the same deterministic batch plan into `runs/high_lr_guard_probe_20260528_1024_b240/`.
- `tg_2e-3` now crossed the previous failure region (`240` total backward passes vs prior divergence at `232`) with repeated `N=0` fallback cycles (`cycle 3+`) and reduced pilot lr / increased K (`0.002 -> 0.0014 -> 0.00098`, `K 3 -> 5 -> 8`).
- `tg_5e-3` likewise crossed its previous failure region (`240` total backward passes vs prior divergence at `224`) with `N=0` fallback from `cycle 2` onward and progressively more conservative pilot settings (`lr 0.005 -> 0.0035 -> 0.00245`, `K 3 -> 5 -> 8`).
- in both reruns the original speculative high-lr divergence was suppressed, but training still terminated later with `NumericalInstabilityError` on a batch where all labels were masked after prompt masking (`Sample 248 has all labels masked (-100) after prompt masking`).
- **practical read**: the local-linearity guard does what it was designed to do, but a separate masked-label / NaN path remains and now dominates once speculative updates are disabled.
- **modules affected**: [[Module Training Loop]], [[Module Data Pipeline]]

## [2026-05-28] exact resume | periodic checkpoints now carry batch cursor and validation history

- extended `TrainingState` with `train_batch_position` and `accepted_valid_history`, and taught `InfiniteBatchIterator` to fast-forward by batch count.
- `train_tg_lora.py` now restores the iterator cursor on `--resume`, skips redundant initial quick-eval on resume, and writes `training_state.pt` alongside periodic `checkpoint-cycle-*` saves.
- numerical instability faults now save a resumable checkpoint too, not just OOM / CUDA faults.
- practical result: future verification runs can save frequently and resume near the exact failure boundary instead of restarting from cycle 0, while still reusing the on-disk prefix feature cache for forward/eval acceleration.
- **modules affected**: [[Module Utilities]], [[Module Training Loop]], [[Module Batch Planning]]

## [2026-05-28] all-masked sample diagnosis | existing data showed 25 train and 2 valid prompt-truncation hazards at 1024

- replayed the already-collected dataset with the current tokenizer and masking logic at `max_seq_len=1024`.
- found `25` train samples and `2` validation samples where `prompt_tokens == total_tokens == 1024`, so the assistant side is fully truncated and `labels` become all `-100`.
- confirmed the previous crash boundary directly: around training position `246`, the real loader now encounters `248` as the first skipped item and continues with valid positions `249..254` to fill the grad-accum window.
- implemented `has_supervised_tokens()` and used it in both train loops and eval so fully masked batches are skipped instead of producing NaN loss.
- **modules affected**: [[Module Data Pipeline]], [[Module Training Loop]], [[Module Evaluation]]

## [2026-05-30] masked-supervision hardening | cache eval now skips empty targets and dataset load prunes prompt-truncated rows

- hardened `src/tg_lora/activation_cache.py` so both cached eval and uncached fallback eval skip batches whose labels are entirely `-100`, matching the existing train/eval policy instead of letting cache-specific paths see NaN-producing supervision.
- added activation-cache regressions proving mixed masked/valid loaders behave the same as valid-only loaders in both the normal cache path and the fallback path.
- upgraded `src/data/build_seed_dataset.py` so `load_dataset()` drops records that become `all_masked` after prompt masking and truncation, while leaving direct `LoraDataset(...)` use unchanged for low-level tests.
- at `max_seq_len=1024`, effective dataset sizes now become `train 3000 -> 2975`, `valid_quick 300 -> 298`, and `valid_full 300 -> 298`, removing the exact prompt-truncation hazards previously identified in the repo log.
- focused validation passed across `tests/test_activation_cache.py`, `tests/test_build_seed_dataset.py`, plus masked-supervision integration checks in `tests/test_training_integration.py` and `tests/test_baseline_training.py`.
- **modules affected**: [[Module Activation and Prefix Cache]], [[Module Data Pipeline]]

## [2026-05-30] frontier separation confirmed | TG-LoRA trains where baseline OOMs at 1536 and 2048 tokens

- ran `make paper-memory-frontier-sweep SEQS="1536 2048" SEEDS="42" TARGET_BP=240` on the RTX 3060 12 GB.
- **baseline @1536: OOM** — backward pass at step 0 fails with `torch.OutOfMemoryError` requesting 1.42 GiB on top of 10.8 GB resident.
- **baseline @2048: OOM** — same immediate failure pattern.
- **TG-LoRA @1536: 80 cycles completed** — acceptance 70/80 (87.5%), best valid loss 1.0324, final loss 1.0399, GPU peak ~9.2 GB.
- **TG-LoRA @2048: 80 cycles completed** — acceptance 69/80 (86.3%), best valid loss 1.0202, final loss 1.0254.
- prefix feature cache sizes at 1536 tokens: train 37.7 GB, valid_quick 3.8 GB, valid_full 3.8 GB (stored on disk, forward computed in ~75 min).
- `frontier_report.json` now correctly reports `frontier_separation_detected: true` and `boundary: 2048` after fixing a bug in `scripts/frontier_report.py` where the frontier sweep metadata format (`seeds[]` array) was unrecognized, causing both arms to default to exit 0.
- multi-seed replication (seed 43, 44) launched to complete the 3-seed validation.
- **paper condition**: C2 confirmed — "consumer GPU frontier — baseline fails, TG-LoRA succeeds" at both 1536 and 2048 context lengths.
- **modules affected**: [[Module Activation and Prefix Cache]], [[Module Training Loop]]

## [2026-05-31] frontier multi-seed closure | seed 44 replicated the 1536/2048 boundary and closed the 3-seed C2 sweep

- completed `runs/frontier_sweep_20260530_seed44/` with the same boundary already seen on seeds 42 and 43: at both `1536` and `2048`, baseline exits with OOM while TG-LoRA completes all `80` cycles.
- `frontier_report.json` now agrees across all three seed-specific runs: `runs/frontier_sweep_20260530/`, `runs/frontier_sweep_20260530_seed43/`, and `runs/frontier_sweep_20260530_seed44/` all report `frontier_boundary: 2048` and `frontier_separation_detected: true`.
- this closes the previously open multi-seed frontier replication item in [[Processing Progress]] and upgrades the frontier evidence from single-seed confirmation to 3-seed replication on the RTX 3060 12 GB path.
- immediately advanced to Gate G3 by launching external benchmark evaluation from the canonical one-shot aggregate `runs/paper_memory_one_shot_ms3_s1024_20260525/aggregate_summary.json` against the best TG seed (`44`) and best baseline seed (`43`).
- **paper condition**: C2 remains supported, now with replicated boundary evidence rather than a single-seed frontier witness.

---

## Initial ingest | tg-lora

- target: `/home/jinno/tg-lora`
- files: 669
- logical modules: 19
- risk findings: 932
- created_or_updated:
  - [[Repository Overview]]
  - [[Module Index]]
  - [[Repository Risk Register]]
  - [[Processing Progress]]
  - [[File Inventory]]
  - [[Repo Wiki Schema]]
- decision_reason: Generated during extract-skill-meta planning so repository understanding and risk context compound alongside skill extraction.
## bootstrap | repo LLM wiki

- Generated or refreshed repository wiki for `tg-lora`.
- Source: deterministic ai-hub fallback writer.

