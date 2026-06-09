---
title: Repository Risk Register
genre: repository-analysis
type: synthesis
sources:
  - extract-skill-meta planning artifacts
related:
  - Repository Overview
  - Module Index
  - File Inventory
status: generated
---
# Repository Risk Register

## Summary

- Total findings: 36
- High: 5
- Medium: 26
- Low: 5

## Findings

| ID | Severity | Category | File | Signal | Evidence |
| --- | --- | --- | --- | --- | --- |
| RISK-0001 | medium | Parser Or Heuristic | `pyproject.toml` | `toml` | path contains `toml` |
| RISK-0002 | high | Security Boundary | `reports/downstream_eval_mlx/report_base.json` | `auth` | L81: "prompt": "次の文章から本の商品情報を抽出し、JSONフォーマットで出力してください。JSONのみを出力してください。\n文章:「『吾輩は猫である』は夏目漱石によって書かれ、価格は800円です。」\nフォーマット: {\"title\": \"書名\", \"author\": \"著者\", \"price\": 価格}", |
| RISK-0003 | medium | Parser Or Heuristic | `reports/downstream_eval_mlx/report_base.json` | `json` | path contains `json` |
| RISK-0004 | medium | Parser Or Heuristic | `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.json` | `json` | path contains `json` |
| RISK-0005 | high | Security Boundary | `scripts/eval_downstream.py` | `token` | L11: from transformers import AutoModelForCausalLM, AutoTokenizer |
| RISK-0006 | medium | Parser Or Heuristic | `scripts/eval_downstream.py` | `parse` | L1: import argparse |
| RISK-0007 | low | High Attention File | `scripts/eval_downstream.py` | `attention_score=100` | L1: import argparse |
| RISK-0008 | high | Security Boundary | `scripts/eval_downstream_mlx.py` | `token` | L76: tokenizer, |
| RISK-0009 | medium | Parser Or Heuristic | `scripts/eval_downstream_mlx.py` | `parse` | L1: import argparse |
| RISK-0010 | low | High Attention File | `scripts/eval_downstream_mlx.py` | `attention_score=100` | L1: import argparse |
| RISK-0011 | high | Security Boundary | `scripts/eval_llm_jp_eval.py` | `token` | L10: from transformers import AutoModelForCausalLM, AutoTokenizer |
| RISK-0012 | medium | Parser Or Heuristic | `scripts/eval_llm_jp_eval.py` | `parse` | L1: import argparse |
| RISK-0013 | low | High Attention File | `scripts/eval_llm_jp_eval.py` | `attention_score=100` | L1: import argparse |
| RISK-0014 | high | Security Boundary | `scripts/eval_llm_jp_eval_mlx.py` | `token` | L101: model, tokenizer = mlx_load(args.model_path) |
| RISK-0015 | medium | Parser Or Heuristic | `scripts/eval_llm_jp_eval_mlx.py` | `parse` | L1: import argparse |
| RISK-0016 | medium | Persistence Or State | `scripts/eval_llm_jp_eval_mlx.py` | `cache` | L90: parser.add_argument("--model-path", type=str, default=".cache/mlx_models/Qwen--Qwen3.5-9B", help="Path to MLX model folder") |
| RISK-0017 | low | High Attention File | `scripts/eval_llm_jp_eval_mlx.py` | `attention_score=100` | L1: import argparse |
| RISK-0018 | medium | Concurrency Or Timing | `tests/conftest.py` | `async` | L57: # Shared fixtures for AsyncCacheBuilder tests |
| RISK-0019 | medium | Persistence Or State | `tests/conftest.py` | `cache` | L57: # Shared fixtures for AsyncCacheBuilder tests |
| RISK-0020 | medium | Persistence Or State | `tests/test_cycle_state.py` | `state` | path contains `state` |
| RISK-0021 | medium | Persistence Or State | `tests/test_delta_tracker.py` | `state` | L75: def test_tracker_initial_state(): |
| RISK-0022 | medium | Parser Or Heuristic | `tests/test_layer_sampler.py` | `fallback` | L93: def test_lisa_weighted_empty_scores_fallback(): |
| RISK-0023 | medium | Persistence Or State | `tests/test_lora_state.py` | `state` | path contains `state` |
| RISK-0024 | medium | Persistence Or State | `tests/test_random_walk_controller.py` | `state` | L10: def test_initial_state(): |
| RISK-0025 | low | High Attention File | `tests/test_random_walk_controller.py` | `attention_score=100` | L3: from unittest.mock import patch |
| RISK-0026 | medium | Persistence Or State | `tests/test_tg_lora_workflow.py` | `state` | L73: cycle_state = CycleState() |
| RISK-0027 | medium | Parser Or Heuristic | `tests/test_velocity.py` | `fallback` | L142: # All magnitudes are nearly identical → std ≈ 0, uses mean*2 fallback |
| RISK-0028 | medium | Persistence Or State | `tests/test_velocity.py` | `state` | L46: def test_cosine_similarity_no_state(): |
| RISK-0029 | medium | Persistence Or State | `tg_lora/__init__.py` | `state` | L1: from tg_lora.cycle_state import CycleState |
| RISK-0030 | medium | Persistence Or State | `tg_lora/cycle_state.py` | `state` | path contains `state` |
| RISK-0031 | medium | Persistence Or State | `tg_lora/delta_tracker.py` | `state` | L10: from tg_lora.lora_state import diff_lora |
| RISK-0032 | medium | Parser Or Heuristic | `tg_lora/lora_state.py` | `parse` | L16: to storing full snapshots, since deltas are typically sparse/small. |
| RISK-0033 | medium | Persistence Or State | `tg_lora/lora_state.py` | `state` | path contains `state` |
| RISK-0034 | medium | Persistence Or State | `tg_lora/random_walk_controller.py` | `state` | L25: class ControllerState: |
| RISK-0035 | medium | Persistence Or State | `tg_lora/rollback_manager.py` | `state` | L47: def _sanitize_snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: |
| RISK-0036 | medium | Persistence Or State | `tg_lora/velocity.py` | `state` | L19: def state(self) -> dict[str, torch.Tensor] \| None: |

## Review Guidance

- High severity findings should be reviewed before converting extracted knowhow into reusable skills.
- Check whether the source has tests or guardrails for the cited trust boundary.
- Treat this register as a static-analysis triage list, not a proof of vulnerability.
