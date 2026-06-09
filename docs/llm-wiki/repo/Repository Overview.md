---
title: Repository Overview
genre: repository-analysis
type: synthesis
sources:
  - extract-skill-meta planning artifacts
related:
  - Repository Wiki Index
  - Module Index
  - Repository Risk Register
  - Processing Progress
status: generated
---
# Repository Overview

## Scope

- Repository: `tg-lora-public`
- Repository root: `/home/jinno/tg-lora-public`
- Requested focus path: `/home/jinno/tg-lora-public`
- Matched source files: 37
- Matched source bytes: 384156
- Wiki context logical chunks: 5
- Wiki context agent bundles: 1

## Selection Rules

- Include globs:
  - `**/*.py`
  - `**/*.ts`
  - `**/*.tsx`
  - `**/*.js`
  - `**/*.jsx`
  - `**/*.mjs`
  - `**/*.cjs`
  - `**/*.go`
  - `**/*.rs`
  - `**/*.java`
  - `**/*.kt`
  - `**/*.swift`
  - `**/*.c`
  - `**/*.cc`
  - `**/*.cpp`
  - `**/*.h`
  - `**/*.hpp`
  - `**/*.cs`
  - `**/*.rb`
  - `**/*.php`
  - `**/*.scala`
  - `**/*.lua`
  - `**/*.sh`
  - `**/*.bash`
  - `**/*.zsh`
  - `**/*.ps1`
  - `**/*.sql`
  - `**/*.toml`
  - `**/*.yaml`
  - `**/*.yml`
  - `**/*.json`
- Ignore globs:
  - `**/.git/**`
  - `**/.extract_skill_meta/**`
  - `**/.extract_skill_meta_*/**`
  - `**/node_modules/**`
  - `**/.venv/**`
  - `**/venv/**`
  - `**/__pycache__/**`
  - `**/dist/**`
  - `**/build/**`
  - `**/coverage/**`
  - `**/.next/**`
  - `**/.turbo/**`
  - `**/target/**`
  - `**/*.lock`
  - `**/html/**`
  - `**/playwright-report/**`
  - `**/test-results/**`
  - `**/docs/llm-wiki/**`

## Language Mix

- json: 2
- markdown: 3
- python: 27
- text: 3
- toml: 1
- txt: 1

## Logical Module Map

| Module | Files | Bytes | Languages | Risk Links |
| --- | ---: | ---: | --- | --- |
| [[Module root-config]] | 6 | 92560 | markdown, text, toml | [[Repository Risk Register]] |
| [[Module support-scripts]] | 4 | 36760 | python | [[Repository Risk Register]] |
| [[Module reports]] | 4 | 12966 | json, markdown | [[Repository Risk Register]] |
| [[Module tests]] | 12 | 172498 | python | [[Repository Risk Register]] |
| [[Module tg-lora]] | 11 | 69372 | python | [[Repository Risk Register]] |

## Directory Structure Snapshot

```text
data/
  downstream/
    format_json.jsonl
    jp_capability.jsonl
scripts/
  eval_downstream_mlx.py
  eval_downstream.py
  eval_llm_jp_eval_mlx.py
  eval_llm_jp_eval.py
tests/
  conftest.py
  test_cycle_state.py
  test_delta_tracker.py
  test_extrapolator.py
  test_layer_sampler.py
  test_lora_state.py
  test_lora_utils.py
  test_random_walk_controller.py
  test_rollback_manager.py
  test_tg_lora_workflow.py
  test_trajectory.py
  test_velocity.py
tg_lora/
  __init__.py
  cycle_state.py
  delta_tracker.py
  extrapolator.py
  layer_sampler.py
  lora_state.py
  lora_utils.py
  random_walk_controller.py
  rollback_manager.py
  trajectory.py
  velocity.py
.gitignore
digest.txt
LICENSE
Makefile
pyproject.toml
README.md
```
