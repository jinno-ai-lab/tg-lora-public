"""Dynamic reversible freeze controller for TG-LoRA (Guard experiment).

§3: Freeze from output side (L31) as a contiguous block.  Scan L31→L24,
include quiet layers (r_A_window < τ) until the first noisy layer stops
the block.  No orphans.

§4: Unfreeze from upstream end of block (L_k) one layer at a time.
Never release from the output side (L31).  Triggers: forced stir (R cycles)
or upstream activity (r_A > τ × 1.5).

r_A is computed per-series via the Frobenius norm of the LoRA delta weight:
    A_fro = ||B @ A||_F  (trace trick: O(r³))
    r_A   = |A_fro_now - A_fro_prev| / (A_fro_prev + eps)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.model.lora_utils import iter_all_lora_params_by_layer

logger = logging.getLogger("tg-lora")


@dataclass
class DynFreezeState:
    """Serializable controller state for checkpoint resume."""

    frozen_layer_indices: list[int] = field(default_factory=list)
    r_A_history: dict[int, list[float]] = field(default_factory=dict)
    frozen_since_cycle: int = 0
    prev_A_fro: dict[str, float] = field(default_factory=dict)
    median_A: float = 0.0
    epsilon: float = 1e-6
    # layer_idx → cycle at which it was last released by §4. Lets a resumed run
    # honor the release cooldown (a released layer must re-train for a window
    # before it can be re-frozen). Absent on pre-existing checkpoints.
    released_at: dict[int, int] = field(default_factory=dict)


class DynamicFreezeController:
    """Output-side contiguous freeze / upstream-sequential unfreeze."""

    def __init__(
        self,
        tau: float = 0.015,
        window: int = 5,
        stir_interval: int = 10,
        upstream_activity_factor: float = 1.5,
        epsilon_ratio: float = 0.01,
        a_mask_ratio: float = 0.1,
        all_layer_indices: list[int] | None = None,
    ) -> None:
        self._tau = tau
        self._window = window
        self._stir_interval = stir_interval
        self._upstream_activity_factor = upstream_activity_factor
        self._epsilon_ratio = epsilon_ratio
        self._a_mask_ratio = a_mask_ratio

        # Ordered list of trainable layer indices (sorted descending for output→input scan)
        self._all_layers = sorted(all_layer_indices or [], reverse=True)

        # Frozen block: contiguous from L31 downward, stored sorted descending
        self._frozen_block: list[int] = []  # e.g. [31, 30, 29]
        self._r_A_history: dict[int, deque[float]] = {}
        self._frozen_since_cycle: int = 0
        # layer_idx → cycle last released by §4. While a layer is in its release
        # cooldown (cycle - released_at < window) it may NOT be re-frozen: the
        # frozen-period 0.0 r_A history is not real quietness, so §4 would
        # otherwise silently re-freeze the layer in the same cycle it was
        # released, making the reversible release a no-op.
        self._released_at: dict[int, int] = {}
        self._prev_A_fro: dict[str, float] = {}
        self._median_A: float = 0.0
        self._epsilon: float = 1e-6

    # ------------------------------------------------------------------
    # r_A computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_r_A(self, model: nn.Module, cycle: int) -> dict[int, float]:
        """Compute per-layer r_A from current vs previous LoRA state.

        Returns {layer_idx: r_A_layer} averaged across all series in the layer.
        Frozen layers record 0.0 to keep history aligned.
        """
        layer_map = iter_all_lora_params_by_layer(model)
        current_A_fro: dict[str, float] = {}
        per_layer_r_A: dict[int, float] = {}
        frozen_set = set(self._frozen_block)

        for layer_idx in self._all_layers:
            # Frozen layers: record 0.0, don't compute
            if layer_idx in frozen_set:
                per_layer_r_A[layer_idx] = 0.0
                if layer_idx not in self._r_A_history:
                    self._r_A_history[layer_idx] = deque(maxlen=self._window)
                self._r_A_history[layer_idx].append(0.0)
                continue

            params = layer_map.get(layer_idx, [])
            a_series: dict[str, torch.Tensor] = {}
            b_series: dict[str, torch.Tensor] = {}

            for name, param in params:
                if "lora_A" in name:
                    key = name.replace(".lora_A", "")
                    a_series[key] = param.data
                elif "lora_B" in name:
                    key = name.replace(".lora_B", "")
                    b_series[key] = param.data

            layer_r_A: list[float] = []
            for key in a_series:
                if key not in b_series:
                    continue
                A = a_series[key].float()
                B = b_series[key].float()

                # ||B @ A||_F via trace trick: O(r³)
                AtA = A @ A.T  # (r, r)
                BtB = B.T @ B  # (r, r)
                A_fro = torch.trace(BtB @ AtA).sqrt().item()
                current_A_fro[key] = A_fro

                if key in self._prev_A_fro and self._prev_A_fro[key] > 0:
                    dA = abs(A_fro - self._prev_A_fro[key])
                    if self._median_A > 0 and A_fro < self._a_mask_ratio * self._median_A:
                        r_A_val = 0.0
                    else:
                        r_A_val = dA / (self._prev_A_fro[key] + self._epsilon)
                    layer_r_A.append(r_A_val)

            avg_r_A = sum(layer_r_A) / len(layer_r_A) if layer_r_A else 0.0
            per_layer_r_A[layer_idx] = avg_r_A

            if layer_idx not in self._r_A_history:
                self._r_A_history[layer_idx] = deque(maxlen=self._window)
            self._r_A_history[layer_idx].append(avg_r_A)

        self._prev_A_fro = current_A_fro

        # Update median A for early-cycle masking
        all_fro = [v for v in current_A_fro.values() if v > 0]
        if all_fro:
            sorted_fro = sorted(all_fro)
            mid = len(sorted_fro) // 2
            self._median_A = sorted_fro[mid]
            self._epsilon = self._epsilon_ratio * self._median_A

        return per_layer_r_A

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _r_A_window(self, layer_idx: int) -> float:
        """Windowed mean r_A for a single layer."""
        hist = self._r_A_history.get(layer_idx)
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def _is_quiet(self, layer_idx: int) -> bool:
        """Whether a layer's r_A_window < τ (quiet → can be frozen)."""
        return self._r_A_window(layer_idx) < self._tau

    def _in_release_cooldown(self, layer_idx: int, cycle: int) -> bool:
        """Whether ``layer_idx`` was released by §4 too recently to re-freeze.

        A released layer must accumulate a full fresh window of real r_A
        observations (one per cycle) before §3 may judge it quiet again, so the
        §4 reversible release actually lets the layer re-train instead of being
        silently re-frozen on its frozen-period 0.0 history.
        """
        released_at = self._released_at.get(layer_idx)
        return released_at is not None and cycle - released_at < self._window

    def decide_freeze(self, cycle: int) -> list[int]:
        """§3: Build contiguous block from L31. Return NEW layers to freeze."""
        if cycle < self._window:
            return []

        frozen_set = set(self._frozen_block)
        new_layers: list[int] = []

        # Scan from output (L31) toward input (L24)
        for layer_idx in self._all_layers:
            if layer_idx in frozen_set:
                continue  # Already frozen — skip, block continues
            if self._in_release_cooldown(layer_idx, cycle):
                # Just released by §4 and still re-training: its 0.0 history is
                # frozen-period artifact, not real quietness. The block cannot
                # extend past it (it is the adjacency point), so stop here.
                break
            if self._is_quiet(layer_idx):
                new_layers.append(layer_idx)
            else:
                break  # First noisy layer → block stops

        # Only accept if contiguous with existing block
        if new_layers and frozen_set:
            # New layers must be adjacent to the upstream end of current block
            upstream_end = min(self._frozen_block)
            expected_next = upstream_end - 1
            # All new layers must form a contiguous extension
            if not all(
                new_layers[i] == expected_next - i
                for i in range(len(new_layers))
            ):
                # Not contiguous — skip
                return []

        return new_layers

    def decide_unfreeze(self, cycle: int) -> list[int]:
        """§4: Unfreeze upstream-most frozen layer, one at a time.

        Triggers: (a) forced stir after R cycles, or (b) upstream activity.
        Never release from output side (L31).
        """
        if not self._frozen_block:
            return []

        # Upstream end = smallest layer index in the block
        upstream_end = min(self._frozen_block)

        # §4 invariant: the output-side layer (largest index) is never released
        # — it anchors the contiguous frozen block so backward truncation keeps a
        # frozen boundary (``10_guard_experiment.md`` §4: "出力側は最後まで固定を
        # 保ち連続塊を守る"). When sustained upstream activity means released
        # layers do not re-settle, the block drains toward the output side; once
        # ``upstream_end`` IS the output layer there is nothing upstream left to
        # release, and both the stir and the activity triggers must stop here
        # rather than handing back ``[output_layer]``.
        # ``_all_layers`` is sorted descending (``__init__``), so ``[0]`` is the
        # output side.
        if self._all_layers and upstream_end == self._all_layers[0]:
            return []

        # (a) Forced stir
        cycles_frozen = cycle - self._frozen_since_cycle
        if cycles_frozen >= self._stir_interval:
            logger.info(
                "Guard: cycle=%d frozen %d cycles ≥ R=%d → stir release L%d",
                cycle, cycles_frozen, self._stir_interval, upstream_end,
            )
            return [upstream_end]

        # (b) Upstream activity: layer just upstream of block is noisy
        if cycle >= self._window:
            upstream_neighbor = upstream_end - 1
            if upstream_neighbor in set(self._all_layers):
                upstream_r_A = self._r_A_window(upstream_neighbor)
                threshold = self._tau * self._upstream_activity_factor
                if upstream_r_A > threshold:
                    logger.info(
                        "Guard: cycle=%d upstream L%d r_A=%.6f > %.6f → release L%d",
                        cycle, upstream_neighbor, upstream_r_A, threshold, upstream_end,
                    )
                    return [upstream_end]

        return []

    # ------------------------------------------------------------------
    # Apply decisions
    # ------------------------------------------------------------------

    def apply_freeze(self, model: nn.Module, layers: list[int], cycle: int) -> int:
        """Freeze given layers. Returns number of parameters frozen."""
        if not layers:
            return 0

        layer_map = iter_all_lora_params_by_layer(model)
        frozen_count = 0

        for layer_idx in layers:
            if layer_idx in layer_map:
                for _name, param in layer_map[layer_idx]:
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_count += 1

        self._frozen_block.extend(layers)
        self._frozen_block.sort(reverse=True)  # Maintain L31→L24 order

        if layers and not any(layer == min(self._frozen_block) for layer in layers):
            pass  # Existing block, just extending
        else:
            self._frozen_since_cycle = cycle

        logger.info(
            "Guard: froze L%s (cycle %d, %d params, block=%s)",
            ",".join(str(layer) for layer in layers), cycle, frozen_count, self._frozen_block,
        )
        return frozen_count

    def apply_unfreeze(
        self, model: nn.Module, layers: list[int], *, cycle: int | None = None
    ) -> int:
        """Unfreeze given layers. Returns number of parameters unfrozen.

        ``cycle`` (passed by the trainer) re-arms the §4(a) stir timer and marks
        each released layer as in-cooldown so the reversible release takes
        effect instead of being undone by §3 in the same cycle. Omit it only for
        ad-hoc/test callers that do not run the full decide→apply loop.
        """
        if not layers or not self._frozen_block:
            return 0

        layer_map = iter_all_lora_params_by_layer(model)
        unfrozen_count = 0
        release_set = set(layers)

        for layer_idx in layers:
            if layer_idx in layer_map:
                for _name, param in layer_map[layer_idx]:
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += 1

        self._frozen_block = [layer for layer in self._frozen_block if layer not in release_set]

        if cycle is not None:
            # Block changed → the "held R cycles" counter for §4(a) stir resets,
            # so stir is periodic (one release, then wait again) rather than
            # draining the block one layer per cycle.
            self._frozen_since_cycle = cycle
            for layer_idx in release_set:
                self._released_at[layer_idx] = cycle
                # Drop the frozen-period 0.0 history so the next window is real.
                self._r_A_history.pop(layer_idx, None)

        if unfrozen_count:
            logger.info(
                "Guard: released L%s (block now=%s, %d params unfrozen)",
                ",".join(str(layer) for layer in layers), self._frozen_block, unfrozen_count,
            )
        return unfrozen_count

    # ------------------------------------------------------------------
    # Per-cycle trainer seam
    # ------------------------------------------------------------------

    def run_cycle(self, model: nn.Module, cycle: int) -> bool:
        """One trainer per-cycle step in the load-bearing order.

        Executes ``compute_r_A → decide_unfreeze → apply_unfreeze →
        decide_freeze → apply_freeze`` as a single unit — the exact sequence
        ``train_tg_lora.py`` runs each cycle — and returns whether every
        trainable layer is now frozen (the trainer then skips the expensive
        training steps; ``10_guard_experiment.md`` §9 "スキップ").

        The order is not arbitrary: ``apply_unfreeze`` MUST precede
        ``decide_freeze``. A §4-released layer's frozen-period r_A history is
        all ``0.0`` (an artifact of being frozen, not real quietness), so if
        ``decide_freeze`` ran first it would re-freeze the layer in the same
        cycle and the reversible release would be a silent no-op.
        ``apply_unfreeze(cycle=)`` arms the release cooldown that holds the
        layer out of the freeze decision, and re-arms the §4(a) stir timer so
        stir is periodic rather than draining the block one layer per cycle.
        Dropping the ``cycle`` kwarg breaks both.
        """
        self.compute_r_A(model, cycle)
        to_unfreeze = self.decide_unfreeze(cycle)
        self.apply_unfreeze(model, to_unfreeze, cycle=cycle)
        to_freeze = self.decide_freeze(cycle)
        self.apply_freeze(model, to_freeze, cycle)
        return self.block_size == len(self._all_layers)

    # ------------------------------------------------------------------
    # active_names filtering for M9
    # ------------------------------------------------------------------

    def get_frozen_param_names(self, model: nn.Module) -> set[str]:
        """Return names of currently frozen LoRA parameters."""
        if not self._frozen_block:
            return set()
        layer_map = iter_all_lora_params_by_layer(model)
        names: set[str] = set()
        for layer_idx in self._frozen_block:
            if layer_idx in layer_map:
                for name, param in layer_map[layer_idx]:
                    if not param.requires_grad:
                        names.add(name)
        return names

    @property
    def frozen_layer_indices(self) -> set[int]:
        return set(self._frozen_block)

    @property
    def frozen_block(self) -> list[int]:
        """Frozen block sorted L31→L24."""
        return list(self._frozen_block)

    @property
    def is_frozen(self) -> bool:
        return len(self._frozen_block) > 0

    @property
    def block_size(self) -> int:
        return len(self._frozen_block)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> DynFreezeState:
        r_A_hist: dict[int, list[float]] = {
            k: list(v) for k, v in self._r_A_history.items()
        }
        return DynFreezeState(
            frozen_layer_indices=list(self._frozen_block),
            r_A_history=r_A_hist,
            frozen_since_cycle=self._frozen_since_cycle,
            prev_A_fro=dict(self._prev_A_fro),
            median_A=self._median_A,
            epsilon=self._epsilon,
            released_at=dict(self._released_at),
        )

    def load_state_dict(self, state: DynFreezeState) -> None:
        self._frozen_block = sorted(state.frozen_layer_indices, reverse=True)
        self._r_A_history = {
            k: deque(v, maxlen=self._window)
            for k, v in state.r_A_history.items()
        }
        self._frozen_since_cycle = state.frozen_since_cycle
        self._prev_A_fro = dict(state.prev_A_fro)
        self._median_A = state.median_A
        self._epsilon = state.epsilon
        # ``released_at`` is absent on checkpoints predating the cooldown fix.
        self._released_at = dict(getattr(state, "released_at", {}))
