from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import get_args

from src.tg_lora.layer_sampler import StrategyName

_ALL_STRATEGIES: list[StrategyName] = list(get_args(StrategyName))


def relative_degradation(loss_before: float, loss_after: float) -> float:
    """Relative loss degradation from ``loss_before`` to ``loss_after``.

    This is the **single source of truth** for the relative-tolerance metric
    that every accept/rollback gate in the loop must share. ``rollback_tolerance``
    (default ``0.005``) is a *relative* fraction — 0.5% — not an absolute loss
    margin; that contract is pinned magnitude-invariant by
    ``tests/test_accept_property.py::test_accept_magnitude_consistency`` and
    mirrored here so every site that consumes ``rollback_tolerance`` reads the
    same scale-invariant number.

    Returns the signed relative change ``(loss_after - loss_before) /
    max(abs(loss_before), 1e-8)``: negative ⇒ loss improved, positive ⇒ loss
    degraded. The ``1e-8`` floor matches :meth:`RandomWalkController.accept` and
    keeps the division finite when ``loss_before`` ≈ 0. Non-finite inputs return
    ``+inf`` so any ``<= tolerance`` check rejects (never accepts a NaN/Inf loss).
    """
    if not (math.isfinite(loss_before) and math.isfinite(loss_after)):
        return math.inf
    return (loss_after - loss_before) / max(abs(loss_before), 1e-8)


@dataclass
class Proposal:
    K: int
    N: int
    alpha: float
    beta: float
    lr: float
    active_layer_strategy: StrategyName
    relative_update_cap: float


@dataclass
class ControllerState:
    K: int
    N: int
    alpha: float
    beta: float
    lr: float
    active_layer_strategy: StrategyName
    relative_update_cap: float

    # reward tracking
    layer_scores: dict[int, float] = field(default_factory=dict)
    total_cycles: int = 0
    accepted_count: int = 0
    rolled_back_count: int = 0

    # adaptive boost/decay
    alpha_accept_boost: float = 1.1
    alpha_reject_decay: float = 0.5
    lr_accept_boost: float = 1.2
    lr_reject_decay: float = 0.5

    def summary(self) -> dict:
        return {
            "K": self.K,
            "N": self.N,
            "alpha": self.alpha,
            "beta": self.beta,
            "lr": self.lr,
            "active_layer_strategy": self.active_layer_strategy,
            "relative_update_cap": self.relative_update_cap,
            "layer_scores": dict(self.layer_scores),
            "total_cycles": self.total_cycles,
            "accepted_count": self.accepted_count,
            "rolled_back_count": self.rolled_back_count,
            "alpha_accept_boost": self.alpha_accept_boost,
            "alpha_reject_decay": self.alpha_reject_decay,
            "lr_accept_boost": self.lr_accept_boost,
            "lr_reject_decay": self.lr_reject_decay,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ControllerState:
        return cls(
            K=data["K"],
            N=data["N"],
            alpha=data["alpha"],
            beta=data["beta"],
            lr=data["lr"],
            active_layer_strategy=data["active_layer_strategy"],
            relative_update_cap=data["relative_update_cap"],
            layer_scores=data.get("layer_scores", {}),
            total_cycles=data.get("total_cycles", 0),
            accepted_count=data.get("accepted_count", 0),
            rolled_back_count=data.get("rolled_back_count", 0),
            alpha_accept_boost=data.get("alpha_accept_boost", 1.1),
            alpha_reject_decay=data.get("alpha_reject_decay", 0.5),
            lr_accept_boost=data.get("lr_accept_boost", 1.2),
            lr_reject_decay=data.get("lr_reject_decay", 0.5),
        )


class RandomWalkController:
    # Exploration probabilities (propose)
    _DEFAULT_K_EXPLORE_PROB: float = 0.4
    _DEFAULT_N_EXPLORE_PROB: float = 0.4
    _DEFAULT_BETA_EXPLORE_PROB: float = 0.15
    _DEFAULT_STRATEGY_EXPLORE_PROB: float = 0.08
    _DEFAULT_LR_EXPLORE_PROB: float = 0.3

    # Reward/penalize adjustment probabilities
    _K_DECREASE_ON_REWARD_PROB: float = 0.2
    _N_INCREASE_ON_REWARD_PROB: float = 0.3
    _N_DECREASE_ON_PENALIZE_PROB: float = 0.5
    _STRATEGY_CHANGE_ON_PENALIZE_PROB: float = 0.1

    # Log-normal walk sigmas
    _LR_LOG_SIGMA: float = 0.1

    # Convergence adaptation
    _CONVERGENCE_LR_DECAY: float = 0.8
    _DEFAULT_ACCEL_DEADZONE: float = 0.02

    def __init__(
        self,
        K_initial: int = 3,
        K_candidates: list[int] | None = None,
        N_initial: int = 5,
        N_candidates: list[int] | None = None,
        alpha_initial: float = 0.3,
        alpha_min: float = 0.03,
        alpha_max: float = 1.5,
        alpha_log_sigma: float = 0.15,
        beta_initial: float = 0.8,
        beta_candidates: list[float] | None = None,
        lr_initial: float = 5e-4,
        lr_min: float = 1e-5,
        lr_max: float = 1e-3,
        lr_accept_boost: float = 1.2,
        lr_reject_decay: float = 0.5,
        active_layer_strategy: StrategyName = "last_25_percent_plus_random_2",
        relative_update_cap: float = 0.005,
        rollback_tolerance: float = 0.005,
        enable_random_walk: bool = True,
        enable_convergence_adaptation: bool = True,
        k_explore_prob: float | None = None,
        n_explore_prob: float | None = None,
        beta_explore_prob: float | None = None,
        strategy_explore_prob: float | None = None,
        lr_explore_prob: float | None = None,
        lr_log_sigma: float | None = None,
        accel_instability_lr_decay: float | None = None,
        accel_convergence_lr_boost: float | None = None,
        accel_deadzone: float | None = None,
    ) -> None:
        K_candidates = K_candidates or [2, 3, 5, 8]
        N_candidates = N_candidates or [1, 3, 5, 10, 20]
        beta_candidates = beta_candidates or [0.5, 0.8, 0.9, 0.95]

        if K_initial <= 0:
            raise ValueError(f"K_initial must be positive, got {K_initial}")
        if N_initial <= 0:
            raise ValueError(f"N_initial must be positive, got {N_initial}")
        if any(k <= 0 for k in K_candidates):
            raise ValueError("All K_candidates must be positive")
        if any(n <= 0 for n in N_candidates):
            raise ValueError("All N_candidates must be positive")
        if lr_min >= lr_max:
            raise ValueError(f"lr_min ({lr_min}) must be < lr_max ({lr_max})")
        if alpha_min >= alpha_max:
            raise ValueError(
                f"alpha_min ({alpha_min}) must be < alpha_max ({alpha_max})"
            )
        if alpha_initial < alpha_min or alpha_initial > alpha_max:
            raise ValueError(
                f"alpha_initial ({alpha_initial}) must be in "
                f"[alpha_min={alpha_min}, alpha_max={alpha_max}]"
            )
        if not (0.0 <= beta_initial <= 1.0):
            raise ValueError(f"beta_initial must be in [0, 1], got {beta_initial}")
        if lr_initial < lr_min or lr_initial > lr_max:
            raise ValueError(
                f"lr_initial ({lr_initial}) must be in "
                f"[lr_min={lr_min}, lr_max={lr_max}]"
            )
        if relative_update_cap <= 0:
            raise ValueError(
                f"relative_update_cap must be positive, got {relative_update_cap}"
            )
        if rollback_tolerance < 0:
            raise ValueError(
                f"rollback_tolerance must be non-negative, got {rollback_tolerance}"
            )
        if alpha_log_sigma <= 0:
            raise ValueError(
                f"alpha_log_sigma must be positive, got {alpha_log_sigma}"
            )
        if lr_log_sigma is not None and (
            not math.isfinite(lr_log_sigma) or lr_log_sigma <= 0
        ):
            raise ValueError(
                f"lr_log_sigma must be positive and finite, got {lr_log_sigma}"
            )

        if accel_instability_lr_decay is not None and not (
            0.0 < accel_instability_lr_decay < 1.0
        ):
            raise ValueError(
                f"accel_instability_lr_decay must be in (0, 1), "
                f"got {accel_instability_lr_decay}"
            )
        if accel_convergence_lr_boost is not None and (
            not math.isfinite(accel_convergence_lr_boost)
            or accel_convergence_lr_boost <= 1.0
        ):
            raise ValueError(
                f"accel_convergence_lr_boost must be > 1.0, "
                f"got {accel_convergence_lr_boost}"
            )
        if accel_deadzone is not None and (
            not math.isfinite(accel_deadzone) or accel_deadzone < 0.0
        ):
            raise ValueError(
                f"accel_deadzone must be finite and non-negative, got {accel_deadzone}"
            )

        self.K_candidates = K_candidates
        self.N_candidates = N_candidates
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_log_sigma = alpha_log_sigma
        self.beta_candidates = beta_candidates
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.rollback_tolerance = rollback_tolerance
        self.enable_random_walk = enable_random_walk
        self.enable_convergence_adaptation = enable_convergence_adaptation
        self.k_explore_prob = (
            k_explore_prob
            if k_explore_prob is not None
            else self._DEFAULT_K_EXPLORE_PROB
        )
        self.n_explore_prob = (
            n_explore_prob
            if n_explore_prob is not None
            else self._DEFAULT_N_EXPLORE_PROB
        )
        self.beta_explore_prob = (
            beta_explore_prob
            if beta_explore_prob is not None
            else self._DEFAULT_BETA_EXPLORE_PROB
        )
        self.strategy_explore_prob = (
            strategy_explore_prob
            if strategy_explore_prob is not None
            else self._DEFAULT_STRATEGY_EXPLORE_PROB
        )
        self.lr_explore_prob = (
            lr_explore_prob
            if lr_explore_prob is not None
            else self._DEFAULT_LR_EXPLORE_PROB
        )
        self.lr_log_sigma = (
            lr_log_sigma if lr_log_sigma is not None else self._LR_LOG_SIGMA
        )
        self.accel_instability_lr_decay = (
            accel_instability_lr_decay
            if accel_instability_lr_decay is not None
            else self._DEFAULT_ACCEL_INSTABILITY_LR_DECAY
        )
        self.accel_convergence_lr_boost = (
            accel_convergence_lr_boost
            if accel_convergence_lr_boost is not None
            else self._DEFAULT_ACCEL_CONVERGENCE_LR_BOOST
        )
        self.accel_deadzone = (
            accel_deadzone
            if accel_deadzone is not None
            else self._DEFAULT_ACCEL_DEADZONE
        )

        self.last_accel_action: int = 0

        self.state = ControllerState(
            K=K_initial,
            N=N_initial,
            alpha=alpha_initial,
            beta=beta_initial,
            lr=lr_initial,
            active_layer_strategy=active_layer_strategy,
            relative_update_cap=relative_update_cap,
            lr_accept_boost=lr_accept_boost,
            lr_reject_decay=lr_reject_decay,
        )

    def restore_state(self, state: ControllerState) -> None:
        """Replace the current state with a previously saved state.

        Used for resuming training from a checkpoint. The controller keeps
        its config (candidates, bounds, tolerances, exploration probs) but
        adopts the saved state values (K, N, alpha, beta, lr, counts).
        """
        self.state = state
        self.last_accel_action = 0

    def propose(self) -> Proposal:
        if not self.enable_random_walk:
            return Proposal(
                K=self.state.K,
                N=self.state.N,
                alpha=self.state.alpha,
                beta=self.state.beta,
                lr=self.state.lr,
                active_layer_strategy=self.state.active_layer_strategy,
                relative_update_cap=self.state.relative_update_cap,
            )

        # Log-normal random walk for alpha
        if self.state.alpha > 0:
            log_alpha = math.log(self.state.alpha)
            noise = random.gauss(0, self.alpha_log_sigma)
            new_alpha = math.exp(min(log_alpha + noise, 700))
            new_alpha = max(self.alpha_min, min(self.alpha_max, new_alpha))
        else:
            new_alpha = self.alpha_min

        # Explore K: biased toward adjacent candidates
        idx_k = (
            self.K_candidates.index(self.state.K)
            if self.state.K in self.K_candidates
            else 0
        )
        if random.random() < self.k_explore_prob and len(self.K_candidates) > 1:
            step = 1 if random.random() < 0.5 else -1
            new_idx_k = max(0, min(len(self.K_candidates) - 1, idx_k + step))
            new_K = self.K_candidates[new_idx_k]
        else:
            new_K = self.state.K

        # Explore N: biased toward adjacent candidates
        idx_n = (
            self.N_candidates.index(self.state.N)
            if self.state.N in self.N_candidates
            else 0
        )
        if random.random() < self.n_explore_prob and len(self.N_candidates) > 1:
            step = 1 if random.random() < 0.5 else -1
            new_idx_n = max(0, min(len(self.N_candidates) - 1, idx_n + step))
            new_N = self.N_candidates[new_idx_n]
        else:
            new_N = self.state.N

        # Explore beta: occasionally switch
        if random.random() < self.beta_explore_prob:
            new_beta = random.choice(self.beta_candidates)
        else:
            new_beta = self.state.beta

        # Explore strategy: occasionally switch
        if random.random() < self.strategy_explore_prob:
            others = [
                s for s in _ALL_STRATEGIES if s != self.state.active_layer_strategy
            ]
            new_strategy = random.choice(others)
        else:
            new_strategy = self.state.active_layer_strategy

        # Log-normal random walk for lr
        if random.random() < self.lr_explore_prob and self.state.lr > 0:
            log_lr = math.log(self.state.lr)
            noise = random.gauss(0, self.lr_log_sigma)
            new_lr = math.exp(min(log_lr + noise, 700))
            new_lr = max(self.lr_min, min(self.lr_max, new_lr))
        else:
            new_lr = self.state.lr

        return Proposal(
            K=new_K,
            N=new_N,
            alpha=new_alpha,
            beta=new_beta,
            lr=new_lr,
            active_layer_strategy=new_strategy,
            relative_update_cap=self.state.relative_update_cap,
        )

    def commit_proposal(self, proposal: Proposal) -> None:
        """Adopt a successful proposal into the controller state."""
        self.state.K = proposal.K
        self.state.N = proposal.N
        self.state.alpha = proposal.alpha
        self.state.beta = proposal.beta
        self.state.lr = proposal.lr
        self.state.active_layer_strategy = proposal.active_layer_strategy

    def accept(self, loss_pilot: float, loss_after: float) -> bool:
        if not (math.isfinite(loss_pilot) and math.isfinite(loss_after)):
            return False
        if loss_after <= loss_pilot:
            return True
        return relative_degradation(loss_pilot, loss_after) <= self.rollback_tolerance

    def reward(self, loss_pilot: float, loss_after: float) -> None:
        self.state.accepted_count += 1
        self.state.total_cycles += 1

        if not self.enable_random_walk:
            return

        # Boost alpha slightly on acceptance
        self.state.alpha = min(
            self.state.alpha * self.state.alpha_accept_boost,
            self.alpha_max,
        )

        # Boost lr: accept means aggressive exploration is working
        self.state.lr = min(
            self.state.lr * self.state.lr_accept_boost,
            self.lr_max,
        )

        # Slightly decrease K: high lr is working, be more aggressive
        idx_k = (
            self.K_candidates.index(self.state.K)
            if self.state.K in self.K_candidates
            else 0
        )
        if idx_k > 0 and random.random() < self._K_DECREASE_ON_REWARD_PROB:
            self.state.K = self.K_candidates[idx_k - 1]

        # Slightly increase N
        idx = (
            self.N_candidates.index(self.state.N)
            if self.state.N in self.N_candidates
            else -1
        )
        if (
            idx < len(self.N_candidates) - 1
            and random.random() < self._N_INCREASE_ON_REWARD_PROB
        ):
            self.state.N = self.N_candidates[idx + 1]

    def penalize(self, loss_pilot: float, loss_after: float) -> None:
        self.state.rolled_back_count += 1
        self.state.total_cycles += 1

        if not self.enable_random_walk:
            return

        # Decay alpha on rejection
        self.state.alpha = max(
            self.state.alpha * self.state.alpha_reject_decay,
            self.alpha_min,
        )

        # Decay lr: reject means we're too aggressive, slow down
        self.state.lr = max(
            self.state.lr * self.state.lr_reject_decay,
            self.lr_min,
        )

        # Increase K deterministically: lower lr needs more steps to find direction
        idx_k = (
            self.K_candidates.index(self.state.K)
            if self.state.K in self.K_candidates
            else 0
        )
        if idx_k < len(self.K_candidates) - 1:
            self.state.K = self.K_candidates[idx_k + 1]

        # Decrease N
        idx = (
            self.N_candidates.index(self.state.N)
            if self.state.N in self.N_candidates
            else 0
        )
        if idx > 0 and random.random() < self._N_DECREASE_ON_PENALIZE_PROB:
            self.state.N = self.N_candidates[idx - 1]

        # Occasionally change strategy
        if random.random() < self._STRATEGY_CHANGE_ON_PENALIZE_PROB:
            others = [
                s for s in _ALL_STRATEGIES if s != self.state.active_layer_strategy
            ]
            self.state.active_layer_strategy = random.choice(others)

    def adapt_to_convergence(self, convergence_trend: float) -> None:
        """Proactively shift to precise mode when exploration stalls.

        Called each cycle with delta_tracker.convergence_trend().
        Negative trend = deltas shrinking (healthy convergence).
        Positive/near-zero trend = deltas flat or growing (stalling).
        """
        if not self.enable_convergence_adaptation:
            return

        if not math.isfinite(convergence_trend):
            return

        if convergence_trend >= 0 and self.state.total_cycles > 2:
            # Exploration is stalling — proactively reduce lr, increase K
            self.state.lr = max(
                self.state.lr * self._CONVERGENCE_LR_DECAY,
                self.lr_min,
            )
            idx_k = (
                self.K_candidates.index(self.state.K)
                if self.state.K in self.K_candidates
                else 0
            )
            if idx_k < len(self.K_candidates) - 1:
                self.state.K = self.K_candidates[idx_k + 1]

    # Acceleration-based adaptation
    _DEFAULT_ACCEL_INSTABILITY_LR_DECAY: float = 0.7
    _DEFAULT_ACCEL_CONVERGENCE_LR_BOOST: float = 1.1

    def adapt_to_acceleration(self, acceleration: float) -> None:
        """Adjust lr and K based on velocity magnitude acceleration.

        Positive acceleration = magnitudes growing faster (instability).
        Negative acceleration = growth slowing (convergence).
        Zero or near-zero = stable.
        Called each cycle after velocity.update() with
        velocity.magnitude_acceleration().
        """
        if not self.enable_random_walk and not self.enable_convergence_adaptation:
            self.last_accel_action = 0
            return

        if not math.isfinite(acceleration):
            self.last_accel_action = 0
            return

        if acceleration > self.accel_deadzone:
            self.last_accel_action = 1
            # Instability detected — reduce lr aggressively, increase K
            self.state.lr = max(
                self.state.lr * self.accel_instability_lr_decay,
                self.lr_min,
            )
            idx_k = (
                self.K_candidates.index(self.state.K)
                if self.state.K in self.K_candidates
                else 0
            )
            if idx_k < len(self.K_candidates) - 1:
                self.state.K = self.K_candidates[idx_k + 1]
        elif acceleration < -self.accel_deadzone:
            self.last_accel_action = -1
            # Healthy deceleration — allow slightly more aggressive lr
            self.state.lr = min(
                self.state.lr * self.accel_convergence_lr_boost,
                self.lr_max,
            )
        else:
            self.last_accel_action = 0

    def update_layer_scores(
        self, active_layer_indices: list[int], reward: float
    ) -> None:
        for idx in active_layer_indices:
            self.state.layer_scores[idx] = (
                self.state.layer_scores.get(idx, 0.0) + reward
            )

    def acceptance_rate(self) -> float:
        if self.state.total_cycles == 0:
            return 0.0
        return self.state.accepted_count / self.state.total_cycles

    def summary(self) -> dict:
        return {
            "total_cycles": self.state.total_cycles,
            "accepted": self.state.accepted_count,
            "rolled_back": self.state.rolled_back_count,
            "acceptance_rate": self.acceptance_rate(),
            "current_alpha": self.state.alpha,
            "current_N": self.state.N,
            "current_K": self.state.K,
            "current_beta": self.state.beta,
            "current_lr": self.state.lr,
            "strategy": self.state.active_layer_strategy,
            "last_accel_action": self.last_accel_action,
        }
