"""LAWA (LAtest-Window Weight Averaging) baseline.

Averages the last K LoRA weight checkpoints for a smoother model.
Mandatory baseline per GOAL §3.3: PSA must beat this to have value.

Reference: "LAWA: Late-Window Weight Averaging" — averaging recent
checkpoints along the optimization trajectory reliably improves
generalization over the final checkpoint alone.
"""

import logging
from collections import deque
from contextlib import contextmanager
from typing import Iterator

import torch

from src.model.lora_utils import iter_lora_params

logger = logging.getLogger("tg-lora")


class LAWAAverager:
    """Maintains a sliding window of LoRA weight snapshots and computes
    their arithmetic mean for evaluation.

    Usage in training loop:
        averager = LAWAAverager(window_size=5, start_cycle=10)
        # after each cycle's accept/reject:
        averager.record(model)
        # at eval time:
        snapshot = averager.average_snapshot()
        if snapshot:
            load_lora_snapshot(model, snapshot)
            loss_avg = eval_loss(model, ...)
            load_lora_snapshot(model, averager.latest())  # restore
    """

    def __init__(
        self,
        window_size: int = 5,
        start_cycle: int = 0,
    ):
        # A window_size of 0 backs a deque(maxlen=0) that discards every
        # recorded snapshot — a misconfiguration that should surface here
        # rather than silently never becoming ready.
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.window_size = window_size
        self.start_cycle = start_cycle
        self._buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=window_size)
        self._cycle: int = 0
        self._recorded_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self._cycle >= self.start_cycle and len(self._buffer) >= 2

    @property
    def count(self) -> int:
        return len(self._buffer)

    def record(self, model: torch.nn.Module, cycle: int | None = None) -> None:
        """Snapshot current LoRA weights into the ring buffer."""
        if cycle is not None:
            self._cycle = cycle
        snapshot = {
            name: p.detach().cpu().clone()
            for name, p in iter_lora_params(model)
        }
        self._buffer.append(snapshot)
        self._recorded_count += 1

    def latest(self) -> dict[str, torch.Tensor] | None:
        """Return the most recent snapshot (for restoring after eval)."""
        if not self._buffer:
            return None
        return self._buffer[-1]

    def average_snapshot(self) -> dict[str, torch.Tensor] | None:
        """Compute arithmetic mean of all buffered snapshots.

        Returns None if fewer than 2 snapshots available.
        """
        if len(self._buffer) < 2:
            return None

        keys = sorted(self._buffer[0].keys())
        avg: dict[str, torch.Tensor] = {}
        for key in keys:
            stacked = torch.stack([buf[key] for buf in self._buffer])
            avg[key] = stacked.mean(dim=0)

        return avg

    def reset(self) -> None:
        self._buffer.clear()
        self._recorded_count = 0

    def state_dict(self) -> dict:
        """Serialize the snapshot window + counters for checkpoint resume.

        Mirrors the resume contract of velocity/delta_tracker: the weight
        snapshots are caller-local state (the averager is constructed in
        ``train_tg_lora``'s scope) and must survive resume. Without it a
        fault-resume starts the window empty, ``is_ready`` is False, and the
        LAWA comparison (plus LAWA-averaged JSON eval) are silently skipped
        until ``start_cycle`` worth of new snapshots re-accumulate. LAWA is a
        mandatory baseline (GOAL §3.3), so its window fidelity across resume is
        load-bearing. Snapshots are already CPU-detached at ``record`` time, so
        the checkpoint is device-agnostic.
        """
        return {
            "window_size": self.window_size,
            "start_cycle": self.start_cycle,
            "cycle": self._cycle,
            "recorded_count": self._recorded_count,
            "buffer": [dict(snapshot) for snapshot in self._buffer],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a serialized state produced by :meth:`state_dict`.

        The deque is rebuilt with ``maxlen=window_size`` from the saved state so
        a buffer recorded under a given window trims back to that window —
        matching the live rolling-window contract.
        """
        self.window_size = int(state["window_size"])
        self.start_cycle = int(state["start_cycle"])
        self._buffer = deque(
            (dict(snapshot) for snapshot in state.get("buffer", [])),
            maxlen=self.window_size,
        )
        self._cycle = int(state.get("cycle", 0))
        self._recorded_count = int(state.get("recorded_count", 0))


@torch.no_grad()
def evaluate_with_lawa(
    model: torch.nn.Module,
    averager: LAWAAverager,
    eval_fn,
    current_loss: float | None = None,
) -> tuple[float | None, float]:
    """Evaluate with LAWA-averaged weights, then restore originals.

    Args:
        model: The model (LoRA weights modified in-place then restored).
        averager: The LAWA averager with buffered snapshots.
        eval_fn: Callable that returns (loss, ...) — only first element used.
        current_loss: If provided, skips the current-weight evaluation (avoids
            redundant full eval when loss is already known from the caller).

    Returns:
        (lawa_loss, current_loss) — lawa_loss is None if averager not ready.
    """
    # Snapshot current weights so we can restore
    current = {
        name: p.detach().cpu().clone()
        for name, p in iter_lora_params(model)
    }

    if current_loss is None:
        current_loss = eval_fn(model)

    if not averager.is_ready:
        return None, current_loss

    # Load averaged weights
    avg_snapshot = averager.average_snapshot()
    assert avg_snapshot is not None

    for name, p in iter_lora_params(model):
        if name in avg_snapshot:
            p.copy_(avg_snapshot[name].to(device=p.device, dtype=p.dtype))

    lawa_loss = eval_fn(model)

    # Restore original weights
    for name, p in iter_lora_params(model):
        if name in current:
            p.copy_(current[name].to(device=p.device, dtype=p.dtype))

    return lawa_loss, current_loss


@contextmanager
def averaged_weights_context(
    model: torch.nn.Module,
    averager: LAWAAverager | None,
) -> Iterator[bool]:
    """Temporarily load the LAWA-averaged LoRA weights, restoring on exit.

    Yields True if the averaged weights were swapped in, False otherwise (e.g.
    ``averager`` is None or not yet ready). Used so the headline JSON-quality
    eval measures the *averaged* model — LAWA's proposition that the
    window-average generalizes better than the current checkpoint.

    No-op for the plain/PSA conditions where ``averager`` is None.
    """
    if averager is None or not averager.is_ready:
        yield False
        return
    avg = averager.average_snapshot()
    if avg is None:
        yield False
        return
    restore = {
        name: p.detach().cpu().clone() for name, p in iter_lora_params(model)
    }
    # no_grad: the LoRA params are leaf tensors with requires_grad=True, so the
    # in-place copy_ must run outside autograd (same reason evaluate_with_lawa
    # is decorated @torch.no_grad()).
    with torch.no_grad():
        for name, p in iter_lora_params(model):
            if name in avg:
                p.copy_(avg[name].to(device=p.device, dtype=p.dtype))
    try:
        yield True
    finally:
        with torch.no_grad():
            for name, p in iter_lora_params(model):
                if name in restore:
                    p.copy_(restore[name].to(device=p.device, dtype=p.dtype))
