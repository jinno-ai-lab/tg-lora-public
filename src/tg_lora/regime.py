"""Training regime detection for PSA prior management.

Detects transitions in training dynamics (stable → transition → plateau)
so PSA can reset priors when the loss landscape regime shifts.

Uses a sliding window of loss differences (velocities) to detect:
- STABLE: velocity consistently negative (loss decreasing)
- PLATEAU: velocity near zero for extended period
- TRANSITION: sudden change in velocity magnitude or direction

Design notes:
- Intentionally simple statistical approach (no ML).
- GOAL §3: "regime-aware 制御" — PSA amplification is effective in STABLE
  but priors become stale during TRANSITION.
- GOAL §4 step 2: "相転移リセット有無の効果測定" — ablation gate.
"""

import logging
import math
from collections import deque
from enum import Enum

logger = logging.getLogger("tg-lora")


class Regime(str, Enum):
    STABLE = "stable"
    PLATEAU = "plateau"
    TRANSITION = "transition"


class RegimeDetector:
    """Detects training regime transitions from loss history."""

    def __init__(
        self,
        window: int = 8,
        plateau_eps: float = 1e-4,
        transition_z: float = 2.0,
        min_history: int = 3,
    ):
        self.window = window
        self.plateau_eps = plateau_eps
        self.transition_z = transition_z
        self.min_history = min_history

        self._losses: deque[float] = deque(maxlen=window + 1)
        self._velocities: deque[float] = deque(maxlen=window)
        self._regime = Regime.STABLE
        self._transition_count: int = 0
        self._reset_signaled: bool = False

    def update(self, loss: float) -> Regime:
        """Record a new loss value and return the detected regime."""
        if not math.isfinite(loss):
            return self._regime

        self._losses.append(loss)
        if len(self._losses) < 2:
            return self._regime

        velocity = self._losses[-1] - self._losses[-2]
        self._velocities.append(velocity)

        if len(self._velocities) < self.min_history:
            return self._regime

        prev_regime = self._regime
        self._regime = self._classify()
        self._reset_signaled = (
            self._regime == Regime.TRANSITION
            and prev_regime != Regime.TRANSITION
        )

        if self._reset_signaled:
            self._transition_count += 1
            logger.info(
                "Regime transition detected (#%d): %s → %s, velocity=%.6f",
                self._transition_count,
                prev_regime.value,
                self._regime.value,
                velocity,
            )

        return self._regime

    @property
    def regime(self) -> Regime:
        return self._regime

    @property
    def should_reset_priors(self) -> bool:
        """True when a regime transition was just detected (peek, non-consuming)."""
        return self._reset_signaled

    def consume_reset_signal(self) -> bool:
        """Return and clear the reset signal (one-shot consumption).

        Preferred over ``should_reset_priors`` in the training loop so that
        the signal cannot be consumed twice between updates.
        """
        signaled = self._reset_signaled
        self._reset_signaled = False
        return signaled

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def _classify(self) -> Regime:
        vels = list(self._velocities)
        n = len(vels)
        if n < self.min_history:
            return Regime.STABLE

        recent = vels[-self.min_history:]

        # Plateau: recent velocities all near zero
        if all(abs(v) < self.plateau_eps for v in recent):
            return Regime.PLATEAU

        # Transition: latest velocity is an outlier relative to window stats
        if n >= self.min_history + 1:
            older = vels[: -self.min_history]
            if len(older) >= 2:
                older_mean = sum(older) / len(older)
                older_var = sum((v - older_mean) ** 2 for v in older) / len(older)
                if older_var < 1e-10:
                    # Near-zero variance — velocities are essentially constant,
                    # so z-score is numerically unstable. Only flag transition
                    # if latest velocity deviates significantly in absolute terms.
                    if abs(vels[-1] - older_mean) > self.plateau_eps * 10:
                        return Regime.TRANSITION
                else:
                    older_std = math.sqrt(older_var)
                    z = abs(vels[-1] - older_mean) / older_std
                    if z > self.transition_z:
                        return Regime.TRANSITION

        return Regime.STABLE

    def reset(self) -> None:
        """Clear all internal state."""
        self._losses.clear()
        self._velocities.clear()
        self._regime = Regime.STABLE
        self._transition_count = 0
        self._reset_signaled = False

    def state_dict(self) -> dict:
        """Serialize the run-wide regime state for checkpoint resume.

        Mirrors ``PSAPrior.state_dict()`` /
        ``ActivationFingerprintTracker.state_dict()``. The resume-persistent
        surface is the run-wide accumulation — the loss / velocity classification
        windows (``_losses`` / ``_velocities``, so the first post-resume
        classifications are not a cold-start warm-up), the current ``_regime``
        (which the next ``update`` compares against to detect a fresh
        transition), and the run-wide ``_transition_count`` (reported per cycle as
        ``psa_regime_transitions`` and persisted to ``run_metrics.jsonl``) — NOT
        the config (re-supplied in ``__init__``) or the transient one-shot
        ``_reset_signaled`` (consumed each cycle; persisting it could re-fire a
        prior-reset the loop already acted on). Without persisting this a
        fault/periodic resume rebuilds the detector fresh and the per-cycle
        ``psa_regime_transitions`` resets to 0 — a silent resume-state-loss
        sibling to the fixed PSA prior (``psa_state``) / activation-regime
        (``act_regime_state``) gaps. Enum ``_regime`` is serialized as its
        ``.value`` string.
        """
        return {
            "losses": list(self._losses),
            "velocities": list(self._velocities),
            "regime": self._regime.value,
            "transition_count": self._transition_count,
        }

    def load_state_dict(self, state: dict | None) -> None:
        """Restore run-wide regime state from a checkpoint.

        Inverse of :meth:`state_dict`. The transient ``_reset_signaled`` is left
        untouched (it is consumed each cycle and repopulated by the next
        :meth:`update`, so the first post-resume cycle simply has no pending
        signal — the same cold-start as the run's first cycle). Tolerant of a
        partial / legacy dict and of ``None``: a missing key leaves the
        constructed default, so a pre-fix checkpoint (or a checkpoint from an
        ``enable_psa: false`` run) loads cleanly as a fresh detector. The windows
        are rebuilt with the constructed ``maxlen`` so a window larger than the
        checkpoint's trims the same way a live run would have trimmed it, and a
        smaller checkpoint window fills a larger one without padding.
        """
        if not state:
            return
        if "losses" in state:
            self._losses = deque(list(state["losses"]), maxlen=self.window + 1)
        if "velocities" in state:
            self._velocities = deque(list(state["velocities"]), maxlen=self.window)
        if "regime" in state:
            self._regime = Regime(state["regime"])
        if "transition_count" in state:
            self._transition_count = int(state["transition_count"])
