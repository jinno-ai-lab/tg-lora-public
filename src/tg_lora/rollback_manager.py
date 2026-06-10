import logging

import torch

from src.tg_lora.lora_state import load_lora_snapshot, snapshot_lora

logger = logging.getLogger(__name__)

_MAX_HISTORY = 100


class RollbackManager:
    def __init__(self, max_history: int = _MAX_HISTORY) -> None:
        if max_history <= 0:
            raise ValueError(f"max_history must be positive, got {max_history}")
        self._history: list[dict[str, torch.Tensor]] = []
        self._max_history = max_history

    def save(self, model: torch.nn.Module) -> int:
        state = snapshot_lora(model)
        if not state:
            raise RuntimeError("snapshot_lora returned empty state — cannot save rollback point")
        state = _sanitize_snapshot(state)
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        return len(self._history) - 1

    def rollback(self, model: torch.nn.Module, index: int = -1) -> None:
        if not self._history:
            raise RuntimeError("No saved states to rollback to")
        if index >= len(self._history) or (
            index < 0 and len(self._history) + index < 0
        ):
            raise IndexError(
                f"Rollback index {index} out of range for history of length {len(self._history)}"
            )
        state = self._history[index]
        load_lora_snapshot(model, state)

    def pop(self) -> None:
        if self._history:
            self._history.pop()

    def clear(self) -> None:
        self._history.clear()


def _sanitize_snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace NaN/Inf values in a snapshot so rollbacks never restore corruption."""
    sanitized = False
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            if not sanitized:
                logger.warning(
                    "Non-finite values detected in rollback snapshot — sanitizing"
                )
                sanitized = True
            state[name] = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
    return state
