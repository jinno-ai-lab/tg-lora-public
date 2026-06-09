---
title: Module tests
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
# Module tests

## Role

- Rationale: Files under tests form a shared path-level boundary.
- Roots: tests
- Languages: python
- Files: 12
- Bytes: 172498

## Key Files

- `tests/test_extrapolator.py`
- `tests/conftest.py`
- `tests/test_cycle_state.py`
- `tests/test_delta_tracker.py`
- `tests/test_layer_sampler.py`
- `tests/test_lora_state.py`
- `tests/test_lora_utils.py`
- `tests/test_random_walk_controller.py`

## Risk Signals

- RISK-0018 (medium, Concurrency Or Timing) in `tests/conftest.py`: Timing-sensitive code needs retry, cancellation, and race-condition review. Evidence: L57: # Shared fixtures for AsyncCacheBuilder tests
- RISK-0019 (medium, Persistence Or State) in `tests/conftest.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L57: # Shared fixtures for AsyncCacheBuilder tests
- RISK-0020 (medium, Persistence Or State) in `tests/test_cycle_state.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: path contains `state`
- RISK-0021 (medium, Persistence Or State) in `tests/test_delta_tracker.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L75: def test_tracker_initial_state():
- RISK-0022 (medium, Parser Or Heuristic) in `tests/test_layer_sampler.py`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: L93: def test_lisa_weighted_empty_scores_fallback():
- RISK-0023 (medium, Persistence Or State) in `tests/test_lora_state.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: path contains `state`
- RISK-0024 (medium, Persistence Or State) in `tests/test_random_walk_controller.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L10: def test_initial_state():
- RISK-0025 (low, High Attention File) in `tests/test_random_walk_controller.py`: The digest found several implementation signals worth manual review. Evidence: L3: from unittest.mock import patch
- RISK-0026 (medium, Persistence Or State) in `tests/test_tg_lora_workflow.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L73: cycle_state = CycleState()
- RISK-0027 (medium, Parser Or Heuristic) in `tests/test_velocity.py`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: L142: # All magnitudes are nearly identical → std ≈ 0, uses mean*2 fallback
- RISK-0028 (medium, Persistence Or State) in `tests/test_velocity.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L46: def test_cosine_similarity_no_state():

## Files

- `tests/conftest.py` — python, 88 lines, attention 28
- `tests/test_cycle_state.py` — python, 356 lines, attention 36
- `tests/test_delta_tracker.py` — python, 416 lines, attention 0
- `tests/test_extrapolator.py` — python, 371 lines, attention 14
- `tests/test_layer_sampler.py` — python, 321 lines, attention 42
- `tests/test_lora_state.py` — python, 221 lines, attention 8
- `tests/test_lora_utils.py` — python, 109 lines, attention 0
- `tests/test_random_walk_controller.py` — python, 1999 lines, attention 100
- `tests/test_rollback_manager.py` — python, 149 lines, attention 0
- `tests/test_tg_lora_workflow.py` — python, 253 lines, attention 14
- `tests/test_trajectory.py` — python, 522 lines, attention 0
- `tests/test_velocity.py` — python, 448 lines, attention 14
