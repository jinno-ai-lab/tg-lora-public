---
title: Module tg-lora
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
# Module tg-lora

## Role

- Rationale: Files under tg_lora form a shared path-level boundary.
- Roots: tg_lora
- Languages: python
- Files: 11
- Bytes: 69372

## Key Files

- `tg_lora/extrapolator.py`
- `tg_lora/__init__.py`
- `tg_lora/cycle_state.py`
- `tg_lora/delta_tracker.py`
- `tg_lora/layer_sampler.py`
- `tg_lora/lora_state.py`
- `tg_lora/lora_utils.py`
- `tg_lora/random_walk_controller.py`

## Risk Signals

- RISK-0029 (medium, Persistence Or State) in `tg_lora/__init__.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L1: from tg_lora.cycle_state import CycleState
- RISK-0030 (medium, Persistence Or State) in `tg_lora/cycle_state.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: path contains `state`
- RISK-0031 (medium, Persistence Or State) in `tg_lora/delta_tracker.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L10: from tg_lora.lora_state import diff_lora
- RISK-0032 (medium, Parser Or Heuristic) in `tg_lora/lora_state.py`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: L16: to storing full snapshots, since deltas are typically sparse/small.
- RISK-0033 (medium, Persistence Or State) in `tg_lora/lora_state.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: path contains `state`
- RISK-0034 (medium, Persistence Or State) in `tg_lora/random_walk_controller.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L25: class ControllerState:
- RISK-0035 (medium, Persistence Or State) in `tg_lora/rollback_manager.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L47: def _sanitize_snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
- RISK-0036 (medium, Persistence Or State) in `tg_lora/velocity.py`: Persistent state needs consistency, schema, and partial-write handling. Evidence: L19: def state(self) -> dict[str, torch.Tensor] | None:

## Files

- `tg_lora/__init__.py` — python, 27 lines, attention 0
- `tg_lora/cycle_state.py` — python, 270 lines, attention 64
- `tg_lora/delta_tracker.py` — python, 160 lines, attention 0
- `tg_lora/extrapolator.py` — python, 64 lines, attention 28
- `tg_lora/layer_sampler.py` — python, 128 lines, attention 0
- `tg_lora/lora_state.py` — python, 60 lines, attention 50
- `tg_lora/lora_utils.py` — python, 128 lines, attention 0
- `tg_lora/random_walk_controller.py` — python, 570 lines, attention 0
- `tg_lora/rollback_manager.py` — python, 59 lines, attention 0
- `tg_lora/trajectory.py` — python, 392 lines, attention 0
- `tg_lora/velocity.py` — python, 143 lines, attention 0
