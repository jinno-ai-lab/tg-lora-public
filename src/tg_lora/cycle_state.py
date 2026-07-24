from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CycleState:
    """Tracks aggregate state across TG-LoRA training cycles.

    Extracted from train_tg_lora.py so the reduction-rate calculation,
    early-stopping guard, and best-model tracking can be tested without
    a GPU or model.
    """

    cycle: int = 0
    optimizer_steps: int = 0
    full_backward_passes: int = 0
    extrapolation_steps: int = 0
    speculative_equivalent_backward_passes: int = 0
    best_loss: float = float("inf")
    best_step: int = 0
    stale_cycles: int = 0
    last_train_loss: float = 0.0
    last_valid_loss: float = float("inf")
    accepted_count: int = 0
    rejected_count: int = 0
    current_alpha: float = 0.0
    v_fixed_since_cycle: int | None = None
    alpha_steps_in_cycle: int = 0
    base_term_cached: bool = False
    base_out_jvp_cached: bool = False
    n_base_recompute: int = 0
    consecutive_rejects: int = 0
    interval_net_loss_delta: float | None = None
    # §5.3 improvement-margin for the full-eval early-stopping signal
    # (docs/design/10_guard_experiment.md §5.3: "改善幅 < 0.01 なら打ち切り").
    # A full-eval loss only counts as a new best when the decrease strictly
    # exceeds min_delta (Keras-style). Default 0.0 keeps the historical "any
    # strict decrease wins" contract bit-identical.
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.speculative_equivalent_backward_passes == 0
            and self.extrapolation_steps > 0
        ):
            self.speculative_equivalent_backward_passes = self.extrapolation_steps
        if self.cycle < 0:
            raise ValueError(f"cycle must be non-negative, got {self.cycle}")
        if self.optimizer_steps < 0:
            raise ValueError(
                f"optimizer_steps must be non-negative, got {self.optimizer_steps}"
            )
        if self.full_backward_passes < 0:
            raise ValueError(
                f"full_backward_passes must be non-negative, got {self.full_backward_passes}"
            )
        if self.extrapolation_steps < 0:
            raise ValueError(
                f"extrapolation_steps must be non-negative, got {self.extrapolation_steps}"
            )
        if self.speculative_equivalent_backward_passes < 0:
            raise ValueError(
                "speculative_equivalent_backward_passes must be non-negative, "
                f"got {self.speculative_equivalent_backward_passes}"
            )
        if self.best_step < 0:
            raise ValueError(f"best_step must be non-negative, got {self.best_step}")
        if self.stale_cycles < 0:
            raise ValueError(f"stale_cycles must be non-negative, got {self.stale_cycles}")
        if self.accepted_count < 0:
            raise ValueError(
                f"accepted_count must be non-negative, got {self.accepted_count}"
            )
        if self.rejected_count < 0:
            raise ValueError(
                f"rejected_count must be non-negative, got {self.rejected_count}"
            )
        if self.alpha_steps_in_cycle < 0:
            raise ValueError(
                f"alpha_steps_in_cycle must be non-negative, got {self.alpha_steps_in_cycle}"
            )
        if self.n_base_recompute < 0:
            raise ValueError(
                f"n_base_recompute must be non-negative, got {self.n_base_recompute}"
            )
        if self.consecutive_rejects < 0:
            raise ValueError(
                f"consecutive_rejects must be non-negative, got {self.consecutive_rejects}"
            )
        if self.v_fixed_since_cycle is not None and self.v_fixed_since_cycle < 0:
            raise ValueError(
                "v_fixed_since_cycle must be non-negative when set, "
                f"got {self.v_fixed_since_cycle}"
            )
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {self.min_delta}")

    def record_cycle(
        self,
        *,
        train_loss: float,
        valid_loss: float | None = None,
        accepted: bool | None = True,
        actual_backward_passes: int | None = None,
        speculative_optimizer_steps: int | None = None,
        optimizer_steps: int | None = None,
        speculative_equivalent_backward_passes: int | None = None,
        K: int | None = None,
        N: int | None = None,
        grad_accum: int | None = None,
    ) -> None:
        (
            resolved_actual_backward_passes,
            resolved_speculative_optimizer_steps,
            resolved_optimizer_steps,
            resolved_speculative_equivalent_backward_passes,
        ) = self._resolve_cycle_counts(
            actual_backward_passes=actual_backward_passes,
            speculative_optimizer_steps=speculative_optimizer_steps,
            optimizer_steps=optimizer_steps,
            speculative_equivalent_backward_passes=speculative_equivalent_backward_passes,
            K=K,
            N=N,
            grad_accum=grad_accum,
        )

        self.cycle += 1
        self.optimizer_steps += resolved_optimizer_steps
        self.full_backward_passes += resolved_actual_backward_passes
        self.extrapolation_steps += resolved_speculative_optimizer_steps
        self.speculative_equivalent_backward_passes += (
            resolved_speculative_equivalent_backward_passes
        )
        self.last_train_loss = train_loss

        if valid_loss is not None:
            self.last_valid_loss = valid_loss
            # §5.3 improvement-margin: a quick-eval ``valid_loss`` only counts
            # as a new best (resetting ``stale_cycles`` and lowering
            # ``best_loss``) when it beats the current best by strictly more
            # than ``min_delta`` — the same gate ``record_full_eval`` applies
            # to the full-eval path. ``stale_cycles`` is the early-stopping
            # signal §5.3 governs (field doc + ``record_full_eval`` docstring),
            # and the producer calls ``record_cycle`` every non-full-eval cycle
            # with ``valid_loss=loss_pilot`` (the quick-eval subset loss), so a
            # raw ``valid_loss < best_loss`` lets a sub-min_delta quick-eval
            # wobble reset the counter and defeats early stopping exactly when
            # the insurance is meant to fire. Default ``min_delta=0.0`` keeps
            # the historical ``valid_loss < best_loss`` bit-identical
            # (TASK-0203; same family as the best_model save-site gate,
            # TASK-0202). Non-finite NaN/+Inf quick-eval stays excluded:
            # ``best_loss - nan = nan`` and ``best_loss - inf = -inf`` are both
            # ``> min_delta`` → False, so a diverging quick-eval never lowers
            # ``best_loss``.
            if self.best_loss - valid_loss > self.min_delta:
                self.best_loss = valid_loss
                self.best_step = self.full_backward_passes
                self.stale_cycles = 0
            else:
                self.stale_cycles += 1

        if accepted is True:
            self.accepted_count += 1
        elif accepted is False:
            self.rejected_count += 1

    @staticmethod
    def _resolve_cycle_counts(
        *,
        actual_backward_passes: int | None,
        speculative_optimizer_steps: int | None,
        optimizer_steps: int | None,
        speculative_equivalent_backward_passes: int | None,
        K: int | None,
        N: int | None,
        grad_accum: int | None,
    ) -> tuple[int, int, int, int]:
        if actual_backward_passes is None:
            if K is None or grad_accum is None:
                raise TypeError(
                    "record_cycle requires either explicit cycle counts or legacy "
                    "K/grad_accum inputs"
                )
            actual_backward_passes = K * grad_accum
            if speculative_optimizer_steps is None:
                if N is None:
                    raise TypeError(
                        "record_cycle requires speculative_optimizer_steps or legacy N"
                    )
                speculative_optimizer_steps = N
            if optimizer_steps is None:
                optimizer_steps = K
            if speculative_equivalent_backward_passes is None:
                # Preserve historical behavior for legacy callers that still
                # interpret extrapolation steps in optimizer-step units.
                speculative_equivalent_backward_passes = speculative_optimizer_steps
        else:
            if speculative_optimizer_steps is None:
                raise TypeError(
                    "record_cycle requires speculative_optimizer_steps when "
                    "actual_backward_passes is provided"
                )
            if optimizer_steps is None:
                optimizer_steps = 0
            if speculative_equivalent_backward_passes is None:
                speculative_equivalent_backward_passes = speculative_optimizer_steps

        values = {
            "actual_backward_passes": actual_backward_passes,
            "speculative_optimizer_steps": speculative_optimizer_steps,
            "optimizer_steps": optimizer_steps,
            "speculative_equivalent_backward_passes": speculative_equivalent_backward_passes,
        }
        for name, value in values.items():
            if value is None:
                raise TypeError(f"record_cycle could not resolve {name}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        return (
            actual_backward_passes,
            speculative_optimizer_steps,
            optimizer_steps,
            speculative_equivalent_backward_passes,
        )

    @property
    def reduction_rate(self) -> float:
        total = (
            self.full_backward_passes + self.speculative_equivalent_backward_passes
        )
        if total == 0:
            return 0.0
        return 1.0 - self.full_backward_passes / total

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_count + self.rejected_count
        if total == 0:
            return 0.0
        return self.accepted_count / total

    @property
    def total_cycles(self) -> int:
        return self.accepted_count + self.rejected_count

    def should_stop(
        self,
        patience: int | None = None,
        min_cycles: int = 10,
    ) -> bool:
        if patience is None:
            return False
        return self.stale_cycles >= patience and self.cycle >= min_cycles

    def record_full_eval(self, full_loss: float) -> bool:
        """Update best_loss / stale_cycles from a full-validation-set eval.

        Separate from ``record_cycle`` so the training loop can use quick-eval
        for per-cycle monitoring and full-eval for early-stopping decisions
        without double-counting stale cycles.

        A loss only counts as a new best when the decrease over ``best_loss``
        strictly exceeds ``min_delta`` (§5.3 improvement-margin insurance).
        With the default ``min_delta=0.0`` this reduces to ``full_loss <
        best_loss`` — bit-identical to the pre-§5.3 contract.

        Returns whether this eval recorded a new best (the same min_delta-gated
        predicate the producer must gate ``best_model`` checkpoint saves on).
        ``_evaluate_full_eval_outcome`` computes the identical ``is_new_best``
        from the same ``prev_best`` / ``min_delta``; surfacing it here lets the
        full-eval save sites that don't call that helper (the cached / final /
        linearity-budget branches) gate on the run's official best-loss policy
        instead of a raw ``full_loss < best_full_eval_loss`` comparison that
        would diverge from ``cycle_state.best_loss`` (TASK-0202).
        """
        if self.best_loss - full_loss > self.min_delta:
            self.best_loss = full_loss
            self.best_step = self.full_backward_passes
            self.stale_cycles = 0
            return True
        self.stale_cycles += 1
        return False

    def summary(self) -> dict:
        return {
            "cycles": self.cycle,
            "optimizer_steps": self.optimizer_steps,
            "full_backward_passes": self.full_backward_passes,
            "micro_backward_passes": self.full_backward_passes,
            "extrapolation_steps": self.extrapolation_steps,
            "speculative_optimizer_steps": self.extrapolation_steps,
            "speculative_equivalent_backward_passes": self.speculative_equivalent_backward_passes,
            "reduction_rate": self.reduction_rate,
            "best_valid_loss": self.best_loss,
            "best_valid_step": self.best_step,
            "stale_cycles": self.stale_cycles,
            "acceptance_rate": self.acceptance_rate,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "final_train_loss": self.last_train_loss,
            "last_valid_loss": self.last_valid_loss,
            "current_alpha": self.current_alpha,
            "v_fixed_since_cycle": self.v_fixed_since_cycle,
            "alpha_steps_in_cycle": self.alpha_steps_in_cycle,
            "base_term_cached": self.base_term_cached,
            "n_base_recompute": self.n_base_recompute,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CycleState:
        """Reconstruct a CycleState from a dict produced by ``summary()``.

        Accepts both ``summary()`` keys (``cycles``, ``best_valid_loss``, ...)
        and checkpoint-format keys (``cycle``, ``best_loss``, ...) for
        backward compatibility with saved training states.
        """
        return cls(
            cycle=data.get("cycles", data.get("cycle", 0)),
            optimizer_steps=data.get("optimizer_steps", 0),
            full_backward_passes=data.get("full_backward_passes", 0),
            extrapolation_steps=data.get("extrapolation_steps", 0),
            speculative_equivalent_backward_passes=data.get(
                "speculative_equivalent_backward_passes",
                data.get("extrapolation_steps", 0),
            ),
            best_loss=data.get("best_valid_loss", data.get("best_loss", float("inf"))),
            best_step=data.get("best_valid_step", data.get("best_step", 0)),
            stale_cycles=data.get("stale_cycles", 0),
            last_train_loss=data.get(
                "final_train_loss", data.get("last_train_loss", 0.0)
            ),
            last_valid_loss=data.get("last_valid_loss", float("inf")),
            accepted_count=data.get("accepted_count", 0),
            rejected_count=data.get("rejected_count", 0),
            current_alpha=data.get("current_alpha", 0.0),
            v_fixed_since_cycle=data.get("v_fixed_since_cycle", None),
            alpha_steps_in_cycle=data.get("alpha_steps_in_cycle", 0),
            base_term_cached=data.get("base_term_cached", False),
            n_base_recompute=data.get("n_base_recompute", 0),
        )
