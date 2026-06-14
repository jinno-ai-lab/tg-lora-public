"""Dynamic reversible freeze controller for TG-LoRA Guard (MLX port).

MLX-native port of src/tg_lora/dynamic_freeze.py. The decision algorithm
(decide_freeze contiguous-from-output, decide_unfreeze upstream stir/activity)
is identical; only the tensor ops and the freeze mechanism are swapped to MLX.

§3: Freeze from output side as a contiguous block. Scan L_max→L_min, include
quiet layers (r_A_window < τ) until the first noisy layer stops the block.
§4: Unfreeze from the upstream end of the block one layer at a time.

r_A per series = Frobenius norm of the LoRA weight delta lora_a @ lora_b via
the trace trick (O(r³)), then averaged across the layer's LoRA modules.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear

logger = logging.getLogger("tg-lora")


@dataclass
class DynFreezeState:
    """Serializable controller state for checkpoint resume."""

    frozen_layer_indices: list[int] = field(default_factory=list)
    r_A_history: dict[int, list[float]] = field(default_factory=dict)
    frozen_since_cycle: int = 0
    prev_A_fro: dict[str, float] = field(default_factory=dict)
    peak_r_A: dict[int, float] = field(default_factory=dict)
    median_A: float = 0.0
    epsilon: float = 1e-6


class DynamicFreezeController:
    """Output-side contiguous freeze / upstream-sequential unfreeze (MLX)."""

    def __init__(
        self,
        settled_ratio: float = 0.15,
        window: int = 5,
        stir_interval: int = 10,
        unfreeze_ratio: float = 0.5,
        min_trainable: int = 2,
        epsilon_ratio: float = 0.01,
        a_mask_ratio: float = 0.1,
        all_layer_indices: list[int] | None = None,
    ) -> None:
        # Adaptive threshold: a layer is "settled" when its windowed r_A has
        # decayed to <= settled_ratio * its own running peak. Scale-invariant —
        # tracks whatever r_A scale the dataset/lr/LoRA produce, so no per-task
        # tau retuning. Replaces the former absolute `tau`.
        self._settled_ratio = settled_ratio
        self._unfreeze_ratio = unfreeze_ratio
        self._min_trainable = min_trainable
        self._window = window
        self._stir_interval = stir_interval
        self._epsilon_ratio = epsilon_ratio
        self._a_mask_ratio = a_mask_ratio

        # Trainable layer indices, sorted descending for output→input scan.
        self._all_layers = sorted(all_layer_indices or [], reverse=True)

        self._frozen_block: list[int] = []  # e.g. [31, 30, 29], sorted descending
        self._r_A_history: dict[int, deque[float]] = {}
        self._peak_r_A: dict[int, float] = {}  # running max of r_A_window per layer
        self._frozen_since_cycle: int = 0
        self._prev_A_fro: dict[str, float] = {}
        self._median_A: float = 0.0
        self._epsilon: float = 1e-6

    @property
    def _max_freezable(self) -> int:
        """Never freeze all layers — leave >= min_trainable trainable."""
        return max(0, len(self._all_layers) - self._min_trainable)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lora_modules(layer) -> list[tuple[str, LoRALinear]]:
        """(name, module) for every LoRALinear directly within a decoder layer."""
        out: list[tuple[str, LoRALinear]] = []
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                out.append((name, module))
        return out

    @staticmethod
    def _fro_delta_norm(module: LoRALinear) -> float:
        """||lora_a @ lora_b||_F via the trace trick: O(r³).

        ||AB||_F² = trace((AᵀA)(BBᵀ)); both factors are (r×r).
        """
        a = module.lora_a.astype(mx.float32)  # [in, r]
        b = module.lora_b.astype(mx.float32)  # [r, out]
        ata = a.T @ a  # [r, r]
        bbt = b @ b.T  # [r, r]
        sq = mx.trace(ata @ bbt)
        mx.eval(sq)
        val = float(sq)
        return val ** 0.5 if val > 0 else 0.0

    # ------------------------------------------------------------------
    # r_A computation
    # ------------------------------------------------------------------

    def compute_r_A(self, model, cycle: int) -> dict[int, float]:
        """Per-layer r_A averaged across the layer's LoRA modules.

        Frozen layers record 0.0 to keep history aligned.
        """
        per_layer_r_A: dict[int, float] = {}
        frozen_set = set(self._frozen_block)
        current_A_fro: dict[str, float] = {}

        for layer_idx in self._all_layers:
            if layer_idx in frozen_set:
                per_layer_r_A[layer_idx] = 0.0
                self._r_A_history.setdefault(
                    layer_idx, deque(maxlen=self._window)
                ).append(0.0)
                continue

            modules = self._lora_modules(model.layers[layer_idx])
            layer_r_A: list[float] = []
            for name, module in modules:
                a_fro = self._fro_delta_norm(module)
                current_A_fro[name] = a_fro
                prev = self._prev_A_fro.get(name)
                if prev is not None and prev > 0:
                    dA = abs(a_fro - prev)
                    if (
                        self._median_A > 0
                        and a_fro < self._a_mask_ratio * self._median_A
                    ):
                        r_A_val = 0.0
                    else:
                        r_A_val = dA / (prev + self._epsilon)
                    layer_r_A.append(r_A_val)

            avg = sum(layer_r_A) / len(layer_r_A) if layer_r_A else 0.0
            per_layer_r_A[layer_idx] = avg
            self._r_A_history.setdefault(
                layer_idx, deque(maxlen=self._window)
            ).append(avg)
            # Track this layer's running peak of windowed r_A (adaptive baseline).
            win = self._r_A_window(layer_idx)
            if win > self._peak_r_A.get(layer_idx, 0.0):
                self._peak_r_A[layer_idx] = win

        self._prev_A_fro = current_A_fro
        all_fro = [v for v in current_A_fro.values() if v > 0]
        if all_fro:
            all_fro.sort()
            self._median_A = all_fro[len(all_fro) // 2]
            self._epsilon = self._epsilon_ratio * self._median_A

        return per_layer_r_A

    # ------------------------------------------------------------------
    # Decision logic (verbatim from the PyTorch controller)
    # ------------------------------------------------------------------

    def _r_A_window(self, layer_idx: int) -> float:
        hist = self._r_A_history.get(layer_idx)
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def _is_quiet(self, layer_idx: int) -> bool:
        # Settled = windowed r_A decayed to <= settled_ratio * its own peak.
        # Scale-invariant: adapts to whatever r_A scale the run produces.
        peak = self._peak_r_A.get(layer_idx, 0.0)
        if peak <= 0.0:
            return False
        return self._r_A_window(layer_idx) <= self._settled_ratio * peak

    def decide_freeze(self, cycle: int) -> list[int]:
        if cycle < self._window:
            return []
        frozen_set = set(self._frozen_block)
        new_layers: list[int] = []
        for layer_idx in self._all_layers:
            if layer_idx in frozen_set:
                continue
            if self._is_quiet(layer_idx):
                new_layers.append(layer_idx)
            else:
                break
        # Cap: keep >= min_trainable layers unfrozen (prevents freezing all
        # layers when they settle simultaneously, e.g. on small/overfit data).
        budget = self._max_freezable - len(frozen_set)
        if budget <= 0:
            return []
        new_layers = new_layers[:budget]
        if new_layers and frozen_set:
            upstream_end = min(self._frozen_block)
            expected_next = upstream_end - 1
            if not all(
                new_layers[i] == expected_next - i for i in range(len(new_layers))
            ):
                return []
        return new_layers

    def decide_unfreeze(self, cycle: int) -> list[int]:
        if not self._frozen_block:
            return []
        upstream_end = min(self._frozen_block)
        cycles_frozen = cycle - self._frozen_since_cycle
        if cycles_frozen >= self._stir_interval:
            logger.info(
                "Guard: cycle=%d frozen %d cycles ≥ R=%d → stir release L%d",
                cycle, cycles_frozen, self._stir_interval, upstream_end,
            )
            return [upstream_end]
        if cycle >= self._window:
            upstream_neighbor = upstream_end - 1
            if upstream_neighbor in set(self._all_layers):
                peak = self._peak_r_A.get(upstream_neighbor, 0.0)
                if peak > 0 and self._r_A_window(upstream_neighbor) > self._unfreeze_ratio * peak:
                    logger.info(
                        "Guard: cycle=%d upstream L%d r_A=%.6f > %.6f×peak(%.6f) → release L%d",
                        cycle, upstream_neighbor,
                        self._r_A_window(upstream_neighbor),
                        self._unfreeze_ratio, peak, upstream_end,
                    )
                    return [upstream_end]
        return []

    # ------------------------------------------------------------------
    # Apply decisions
    # ------------------------------------------------------------------

    def apply_freeze(self, model, layers: list[int], cycle: int) -> int:
        if not layers:
            return 0
        frozen_count = 0
        for layer_idx in layers:
            layer = model.layers[layer_idx]
            for _name, module in self._lora_modules(layer):
                module.freeze(keys=["lora_a", "lora_b"])
                frozen_count += 2  # one A + one B per module
        self._frozen_block.extend(layers)
        self._frozen_block.sort(reverse=True)
        if not any(l == min(self._frozen_block) for l in layers):
            pass
        else:
            self._frozen_since_cycle = cycle
        logger.info(
            "Guard: froze L%s (cycle %d, block=%s)",
            ",".join(str(l) for l in layers), cycle, self._frozen_block,
        )
        return frozen_count

    def apply_unfreeze(self, model, layers: list[int]) -> int:
        if not layers or not self._frozen_block:
            return 0
        release_set = set(layers)
        unfrozen = 0
        for layer_idx in layers:
            layer = model.layers[layer_idx]
            for _name, module in self._lora_modules(layer):
                module.unfreeze(keys=["lora_a", "lora_b"])
                unfrozen += 2
        self._frozen_block = [l for l in self._frozen_block if l not in release_set]
        if unfrozen:
            logger.info(
                "Guard: released L%s (block now=%s)",
                ",".join(str(l) for l in layers), self._frozen_block,
            )
        return unfrozen

    # ------------------------------------------------------------------
    # Properties / serialization
    # ------------------------------------------------------------------

    @property
    def frozen_block(self) -> list[int]:
        return list(self._frozen_block)

    @property
    def block_size(self) -> int:
        return len(self._frozen_block)

    @property
    def is_frozen(self) -> bool:
        return len(self._frozen_block) > 0

    @property
    def r_A_history(self) -> dict[int, deque[float]]:
        return self._r_A_history

    def state_dict(self) -> DynFreezeState:
        return DynFreezeState(
            frozen_layer_indices=list(self._frozen_block),
            r_A_history={k: list(v) for k, v in self._r_A_history.items()},
            frozen_since_cycle=self._frozen_since_cycle,
            prev_A_fro=dict(self._prev_A_fro),
            peak_r_A=dict(self._peak_r_A),
            median_A=self._median_A,
            epsilon=self._epsilon,
        )

    def load_state_dict(self, state: DynFreezeState) -> None:
        self._frozen_block = sorted(state.frozen_layer_indices, reverse=True)
        self._r_A_history = {
            k: deque(v, maxlen=self._window) for k, v in state.r_A_history.items()
        }
        self._frozen_since_cycle = state.frozen_since_cycle
        self._prev_A_fro = dict(state.prev_A_fro)
        self._peak_r_A = dict(state.peak_r_A)
        self._median_A = state.median_A
        self._epsilon = state.epsilon
