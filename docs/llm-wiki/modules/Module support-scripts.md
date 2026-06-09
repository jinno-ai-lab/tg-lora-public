---
title: Module support-scripts
genre: repository-analysis
type: entity
sources:
  - extract-skill-meta planning artifacts
related:
  - Module Index
  - Repository Risk Register
  - File Inventory
status: generated
---
# Module support-scripts

## Role

- Rationale: Support scripts provide validation or maintenance helpers around the main code.
- Roots: scripts
- Languages: python
- Files: 4
- Bytes: 36760

## Key Files

- `scripts/eval_downstream.py`
- `scripts/eval_downstream_mlx.py`
- `scripts/eval_llm_jp_eval.py`
- `scripts/eval_llm_jp_eval_mlx.py`

## Risk Signals

- RISK-0005 (high, Security Boundary) in `scripts/eval_downstream.py`: Authentication, authorization, or credential handling can create trust-boundary failures. Evidence: L11: from transformers import AutoModelForCausalLM, AutoTokenizer
- RISK-0006 (medium, Parser Or Heuristic) in `scripts/eval_downstream.py`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: L1: import argparse
- RISK-0007 (low, High Attention File) in `scripts/eval_downstream.py`: The digest found several implementation signals worth manual review. Evidence: L1: import argparse
- RISK-0008 (high, Security Boundary) in `scripts/eval_downstream_mlx.py`: Authentication, authorization, or credential handling can create trust-boundary failures. Evidence: L76: tokenizer,
- RISK-0009 (medium, Parser Or Heuristic) in `scripts/eval_downstream_mlx.py`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: L1: import argparse
- RISK-0010 (low, High Attention File) in `scripts/eval_downstream_mlx.py`: The digest found several implementation signals worth manual review. Evidence: L1: import argparse
- RISK-0011 (high, Security Boundary) in `scripts/eval_llm_jp_eval.py`: Authentication, authorization, or credential handling can create trust-boundary failures. Evidence: L10: from transformers import AutoModelForCausalLM, AutoTokenizer
- RISK-0012 (medium, Parser Or Heuristic) in `scripts/eval_llm_jp_eval.py`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: L1: import argparse
- RISK-0013 (low, High Attention File) in `scripts/eval_llm_jp_eval.py`: The digest found several implementation signals worth manual review. Evidence: L1: import argparse
- RISK-0014 (high, Security Boundary) in `scripts/eval_llm_jp_eval_mlx.py`: Authentication, authorization, or credential handling can create trust-boundary failures. Evidence: L101: model, tokenizer = mlx_load(args.model_path)
- RISK-0015 (medium, Parser Or Heuristic) in `scripts/eval_llm_jp_eval_mlx.py`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: L1: import argparse
- RISK-0016 (medium, Persistence Or State) in `scripts/eval_llm_jp_eval_mlx.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L90: parser.add_argument("--model-path", type=str, default=".cache/mlx_models/Qwen--Qwen3.5-9B", help="Path to MLX model folder")
- RISK-0017 (low, High Attention File) in `scripts/eval_llm_jp_eval_mlx.py`: The digest found several implementation signals worth manual review. Evidence: L1: import argparse

## Files

- `scripts/eval_downstream.py` — python, 277 lines, attention 100
- `scripts/eval_downstream_mlx.py` — python, 239 lines, attention 100
- `scripts/eval_llm_jp_eval.py` — python, 258 lines, attention 100
- `scripts/eval_llm_jp_eval_mlx.py` — python, 233 lines, attention 100
