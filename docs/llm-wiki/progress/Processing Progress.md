---
title: Processing Progress
genre: repository-analysis
type: concept
sources:
  - extract-skill-meta planning artifacts
related:
  - Repository Wiki Index
  - Repository Overview
  - File Inventory
status: generated
---
# Processing Progress

## Summary

- Run index: stable
- Processed files: 37
- New files: 37
- Changed files: 0
- Unchanged files: 0
- Removed files: 0
- Needs processing: 37
- Skipped unchanged: 0
- Digest generated: 37
- Digest reused: 0
- Files with risk signals: 22

## State File

- Machine-readable progress is stored at `_state/progress.json`.
- Large runs shard source entries under `_state/progress_shards/` and rehydrate them through the loader.
- Append-only source change events are stored at `_state/progress_events.jsonl`.
- The state file is the source of truth for large-repo resumability and skipped/changed source accounting.
- For very large repositories, automation should read `_state/progress.json` instead of parsing Markdown tables.

## New Sources

- `.gitignore`
- `LICENSE`
- `Makefile`
- `README.md`
- `digest.txt`
- `pyproject.toml`
- `reports/downstream_eval_mlx/report_base.json`
- `reports/downstream_eval_mlx/report_base.md`
- `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.json`
- `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.md`
- `scripts/eval_downstream.py`
- `scripts/eval_downstream_mlx.py`
- `scripts/eval_llm_jp_eval.py`
- `scripts/eval_llm_jp_eval_mlx.py`
- `tests/conftest.py`
- `tests/test_cycle_state.py`
- `tests/test_delta_tracker.py`
- `tests/test_extrapolator.py`
- `tests/test_layer_sampler.py`
- `tests/test_lora_state.py`
- `tests/test_lora_utils.py`
- `tests/test_random_walk_controller.py`
- `tests/test_rollback_manager.py`
- `tests/test_tg_lora_workflow.py`
- `tests/test_trajectory.py`
- `tests/test_velocity.py`
- `tg_lora/__init__.py`
- `tg_lora/cycle_state.py`
- `tg_lora/delta_tracker.py`
- `tg_lora/extrapolator.py`
- `tg_lora/layer_sampler.py`
- `tg_lora/lora_state.py`
- `tg_lora/lora_utils.py`
- `tg_lora/random_walk_controller.py`
- `tg_lora/rollback_manager.py`
- `tg_lora/trajectory.py`
- `tg_lora/velocity.py`

## Changed Sources

- None detected in this run.

## Removed Sources

- None detected in this run.
