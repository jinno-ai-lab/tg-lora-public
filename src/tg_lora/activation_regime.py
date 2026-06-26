"""Activation-fingerprint cosine for regime inventory measurement.

GOAL §4 step 1: "activation-fingerprint cosine 時系列を取得し 3相分類"
GOAL §2: "stable 相の割合＝効率の理論上限を確定"

Measures cosine similarity between model activations at consecutive training
steps to classify the training regime. Forward-only diagnostic — no extra
backward passes. Piggybacks on existing forward passes via a hook.

Regime classification (activation-based):
- STABLE: high, consistent cosine (activations change smoothly)
- TRANSITION: sudden drop in cosine (landscape shift)
- CHAOTIC: low or volatile cosine (unpredictable dynamics)

The fraction of steps in STABLE regime is the theoretical upper bound on
efficiency gains from any subspace-based method (PSA, extrapolation, etc.).
"""

import logging
import math
from collections import deque
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger("tg-lora")


class ActivationRegime(str, Enum):
    STABLE = "stable"
    TRANSITION = "transition"
    CHAOTIC = "chaotic"


class ActivationFingerprintTracker:
    """Tracks activation cosine between consecutive steps for regime inventory.

    Usage:
        tracker = ActivationFingerprintTracker(model, target_layer_name)
        # Forward hooks auto-capture activations during training
        tracker.step()  # called once per training step
        regime = tracker.regime  # current regime classification
        inventory = tracker.regime_inventory  # cumulative counts
    """

    def __init__(
        self,
        window: int = 10,
        stable_threshold: float = 0.95,
        chaotic_threshold: float = 0.5,
        transition_drop_z: float = 2.0,
        min_history: int = 3,
    ):
        self.window = window
        self.stable_threshold = stable_threshold
        self.chaotic_threshold = chaotic_threshold
        self.transition_drop_z = transition_drop_z
        self.min_history = min_history

        self._cosines: deque[float] = deque(maxlen=window)
        self._all_cosines: list[float] = []
        self._regime = ActivationRegime.STABLE
        self._counts: dict[ActivationRegime, int] = {
            r: 0 for r in ActivationRegime
        }
        self._prev_act: torch.Tensor | None = None
        self._current_act: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def register_hook(self, module: nn.Module) -> None:
        """Register a forward hook on the given module to capture activations."""
        hook = module.register_forward_hook(self._capture_hook)
        self._hooks.append(hook)

    def _capture_hook(
        self, module: nn.Module, input: tuple, output: tuple | torch.Tensor
    ) -> None:
        """Forward hook: capture the activation tensor."""
        act = output
        if isinstance(output, tuple):
            act = output[0]
        if isinstance(act, torch.Tensor):
            # Detach, move to CPU, flatten to 1-D fingerprint
            self._current_act = act.detach().float().cpu().flatten()
            # Cap fingerprint size to control memory
            if self._current_act.numel() > 4096:
                self._current_act = self._current_act[:4096]

    def step(self) -> ActivationRegime:
        """Call once per training step. Computes cosine and classifies regime."""
        if self._current_act is None:
            return self._regime

        if self._prev_act is not None and self._prev_act.numel() == self._current_act.numel():
            cos = _cosine_similarity(self._prev_act, self._current_act)
            if math.isfinite(cos):
                self._cosines.append(cos)
                self._all_cosines.append(cos)

        self._prev_act = self._current_act
        self._current_act = None

        if len(self._cosines) >= self.min_history:
            self._regime = self._classify()

        self._counts[self._regime] += 1
        return self._regime

    @property
    def regime(self) -> ActivationRegime:
        return self._regime

    @property
    def cosines(self) -> list[float]:
        return list(self._cosines)

    @property
    def regime_inventory(self) -> dict[str, float]:
        """Fraction of steps spent in each regime."""
        total = sum(self._counts.values())
        if total == 0:
            return {r.value: 0.0 for r in ActivationRegime}
        return {r.value: c / total for r, c in self._counts.items()}

    @property
    def stable_fraction(self) -> float:
        inv = self.regime_inventory
        return inv.get("stable", 0.0)

    def _classify(self) -> ActivationRegime:
        cosines = list(self._cosines)
        n = len(cosines)
        if n < self.min_history:
            return ActivationRegime.STABLE

        latest = cosines[-1]

        # Chaotic: latest cosine below threshold
        if latest < self.chaotic_threshold:
            return ActivationRegime.CHAOTIC

        # Transition detection: sudden drop relative to recent history
        if n >= self.min_history + 1:
            older = cosines[:-1]
            older_mean = sum(older) / len(older)
            older_var = sum((c - older_mean) ** 2 for c in older) / len(older)
            if older_var > 1e-10:
                older_std = math.sqrt(older_var)
                drop_z = (older_mean - latest) / older_std
                if drop_z > self.transition_drop_z:
                    return ActivationRegime.TRANSITION

        # Stable: high cosine and no transition detected
        if latest >= self.stable_threshold:
            return ActivationRegime.STABLE

        # Between stable and chaotic but no transition spike
        return ActivationRegime.CHAOTIC

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self) -> None:
        self._cosines.clear()
        self._all_cosines.clear()
        self._prev_act = None
        self._current_act = None
        self._regime = ActivationRegime.STABLE
        self._counts = {r: 0 for r in ActivationRegime}
        self.remove_hooks()

    def state_dict(self) -> dict:
        """Serialize the run-wide regime state for checkpoint resume.

        Mirrors ``LAWAAverager.state_dict()``. The resume-persistent surface is
        the run-wide accumulation — the full cosine series (``_all_cosines``,
        needed for the GOAL §7 null baseline at summary time), the per-regime
        ``_counts`` (which drive ``regime_inventory`` / ``stable_fraction``), the
        classification window (``_cosines``), and the current ``_regime`` — NOT
        the transient per-step activation tensors (``_prev_act`` /
        ``_current_act`` are repopulated by the next forward hook, as at run
        start) or the registered hooks (re-registered on construction). Without
        persisting this a fault/periodic resume rebuilds the tracker empty and
        the run-end summary's ``activation_regime_inventory`` /
        ``stable_fraction`` (GOAL §4) reflect only post-resume steps — a silent
        resume-state-loss sibling to the fixed LAWA / dynfreeze / best_full_eval
        gaps. Enum keys are serialized as their ``.value`` strings.
        """
        return {
            "all_cosines": list(self._all_cosines),
            "cosines": list(self._cosines),
            "counts": {r.value: c for r, c in self._counts.items()},
            "regime": self._regime.value,
        }

    def load_state_dict(self, state: dict | None) -> None:
        """Restore run-wide regime state from a checkpoint.

        Inverse of :meth:`state_dict`. Transient activation tensors and hooks
        are left untouched (``None`` / re-registered); the next forward hook +
        :meth:`step` repopulates them as at run start, so the first post-resume
        step simply has no predecessor (the same cold-start as the run's first
        step). Tolerant of a partial / legacy dict and of ``None``: a missing key
        leaves the constructed default, so a pre-fix checkpoint (or a checkpoint
        from an ``activation_regime_enabled: false`` run) loads cleanly as an
        empty tracker. The classification window is rebuilt with the constructed
        ``maxlen`` so a window larger than the checkpoint's is trimmed the same
        way a live run would have trimmed it.
        """
        if not state:
            return
        if "all_cosines" in state:
            self._all_cosines = list(state["all_cosines"])
        if "cosines" in state:
            self._cosines = deque(list(state["cosines"]), maxlen=self.window)
        if "counts" in state:
            self._counts = {
                r: int(state["counts"].get(r.value, 0)) for r in ActivationRegime
            }
        if "regime" in state:
            self._regime = ActivationRegime(state["regime"])

    def summary(self) -> dict:
        cosines = list(self._cosines)
        return {
            "regime": self._regime.value,
            "stable_fraction": self.stable_fraction,
            "regime_inventory": self.regime_inventory,
            "cosine_mean": sum(cosines) / len(cosines) if cosines else None,
            "cosine_latest": cosines[-1] if cosines else None,
            "total_steps": sum(self._counts.values()),
            "all_cosines": list(self._all_cosines),
        }


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = torch.dot(a, b).item()
    na = a.norm().item()
    nb = b.norm().item()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def compute_regime_null_baseline(
    cosines: list[float],
    *,
    stable_threshold: float = 0.95,
    chaotic_threshold: float = 0.5,
    transition_drop_z: float = 2.0,
    min_history: int = 3,
    window: int = 10,
    n_shuffles: int = 1000,
    seed: int = 42,
) -> dict:
    """Null baseline for regime inventory via temporal shuffling.

    GOAL §7: "すべての指標にランダム帰無基準を併記".

    Shuffles the cosine time series to destroy temporal structure,
    then classifies regimes on each shuffle using the same algorithm.
    Returns the null distribution of regime fractions so the observed
    fractions can be compared against random expectation.

    Args:
        cosines: Full cosine time series from tracker.summary()["all_cosines"].
        n_shuffles: Number of random shuffles.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with null distribution statistics:
        - "stable_fraction_null_mean": mean stable fraction across shuffles
        - "stable_fraction_null_std": std of stable fraction across shuffles
        - "stable_fraction_z": z-score of observed vs null
        - "transition_fraction_null_mean": mean transition fraction
        - "chaotic_fraction_null_mean": mean chaotic fraction
        - "per_shuffle_fractions": list of per-shuffle regime fractions
    """
    import random

    if len(cosines) < min_history + 1:
        return {
            "stable_fraction_null_mean": None,
            "stable_fraction_null_std": None,
            "stable_fraction_z": None,
            "transition_fraction_null_mean": None,
            "chaotic_fraction_null_mean": None,
            "per_shuffle_fractions": [],
        }

    rng = random.Random(seed)
    n = len(cosines)

    def _classify_series(series: list[float]) -> dict[str, float]:
        counts = {r: 0 for r in ActivationRegime}
        sliding = deque(maxlen=window)
        for val in series:
            sliding.append(val)
            if len(sliding) < min_history:
                counts[ActivationRegime.STABLE] += 1
                continue
            regime = _classify_cosine(
                list(sliding), stable_threshold, chaotic_threshold,
                transition_drop_z, min_history,
            )
            counts[regime] += 1
        total = sum(counts.values())
        return {r.value: c / total for r, c in counts.items()} if total > 0 else {}

    observed = _classify_series(cosines)
    observed_stable = observed.get("stable", 0.0)

    shuffle_stable: list[float] = []
    shuffle_transition: list[float] = []
    shuffle_chaotic: list[float] = []
    per_shuffle: list[dict[str, float]] = []

    for _ in range(n_shuffles):
        shuffled = list(cosines)
        rng.shuffle(shuffled)
        fracs = _classify_series(shuffled)
        shuffle_stable.append(fracs.get("stable", 0.0))
        shuffle_transition.append(fracs.get("transition", 0.0))
        shuffle_chaotic.append(fracs.get("chaotic", 0.0))
        per_shuffle.append(fracs)

    null_mean = sum(shuffle_stable) / len(shuffle_stable)
    null_var = sum((s - null_mean) ** 2 for s in shuffle_stable) / len(shuffle_stable)
    null_std = math.sqrt(null_var) if null_var > 0 else 0.0

    z_score = (observed_stable - null_mean) / null_std if null_std > 1e-12 else 0.0

    return {
        "stable_fraction_null_mean": null_mean,
        "stable_fraction_null_std": null_std,
        "stable_fraction_z": z_score,
        "transition_fraction_null_mean": sum(shuffle_transition) / len(shuffle_transition),
        "chaotic_fraction_null_mean": sum(shuffle_chaotic) / len(shuffle_chaotic),
        "per_shuffle_fractions": per_shuffle,
    }


def _classify_cosine(
    cosines: list[float],
    stable_threshold: float,
    chaotic_threshold: float,
    transition_drop_z: float,
    min_history: int,
) -> ActivationRegime:
    """Classify a single step from a cosine series (mirrors tracker._classify)."""
    n = len(cosines)
    if n < min_history:
        return ActivationRegime.STABLE

    latest = cosines[-1]

    if latest < chaotic_threshold:
        return ActivationRegime.CHAOTIC

    if n >= min_history + 1:
        older = cosines[:-1]
        older_mean = sum(older) / len(older)
        older_var = sum((c - older_mean) ** 2 for c in older) / len(older)
        if older_var > 1e-10:
            older_std = math.sqrt(older_var)
            drop_z = (older_mean - latest) / older_std
            if drop_z > transition_drop_z:
                return ActivationRegime.TRANSITION

    if latest >= stable_threshold:
        return ActivationRegime.STABLE

    return ActivationRegime.CHAOTIC
