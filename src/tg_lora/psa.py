"""Prior-based Subspace Amplification (PSA).

Amplifies gradients along stable per-tensor PC1 directions extracted from
DeltaTracker history. Zero extra forward passes compared to M9 extrapolation.

Formula: G_amplified = G + gamma * <G, v_PSA> * v_PSA

Design decisions:
- Power iteration for rank-1 extraction (cheaper than full SVD)
- Per-tensor priors (inter-layer cos ~ 0, layers are independent)
- L2 regularization on prior update (mandatory per RNA theory, arXiv 1805.09639)
- Warmup period before amplification activates
"""

import logging
from collections import deque

import torch

from src.model.lora_utils import iter_lora_params
from src.tg_lora.layer_type import LayerType, classify_layer_type

logger = logging.getLogger("tg-lora")


class PSAPrior:
    """Per-tensor PC1 prior extraction and gradient amplification."""

    def __init__(
        self,
        history_length: int = 6,
        gain: float = 0.5,
        update_interval: int = 3,
        warmup_steps: int = 4,
        l2_reg: float = 0.01,
        regime_plateau_gain: float = 0.5,
    ):
        # Reject nonsensical configurations up front so misconfigurations
        # surface before training rather than producing silent no-ops.
        # history_length backs the internal ring buffer (deque(maxlen=...));
        # a length of 0 would discard every recorded delta. gain is allowed to
        # be 0.0 (the no-amplification ablation baseline) but must not be
        # negative (would amplify against the prior direction).
        if history_length < 1:
            raise ValueError("history_length must be >= 1")
        if gain < 0:
            raise ValueError("gain must be non-negative")

        self.history_length = history_length
        self.gain = gain
        self.update_interval = update_interval
        self.warmup_steps = warmup_steps
        self.l2_reg = l2_reg
        self.regime_plateau_gain = regime_plateau_gain

        self.priors: dict[str, torch.Tensor] = {}
        self.gain_map: dict[str, float] = {}
        self._delta_history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=history_length
        )
        self._last_update_step: int = -1
        self._prev_priors: dict[str, torch.Tensor] = {}
        self._prior_cosines: dict[str, list[float]] = {}

    def should_update(self, step: int) -> bool:
        if len(self._delta_history) < 2:
            return False
        if step < self.warmup_steps:
            return False
        if step - self._last_update_step >= self.update_interval:
            return True
        return False

    def record_delta(self, delta: dict[str, torch.Tensor]) -> None:
        """Record a per-step incremental delta into the internal ring buffer."""
        self._delta_history.append(delta)

    @property
    def history_count(self) -> int:
        return len(self._delta_history)

    @property
    def delta_history(self) -> list[dict[str, torch.Tensor]]:
        """Read-only snapshot of the internal ring buffer for external analysis."""
        return list(self._delta_history)

    def extract_priors(
        self,
        delta_history: list[dict[str, torch.Tensor]] | None = None,
    ) -> None:
        history = list(self._delta_history) if self._delta_history else delta_history
        if not history:
            return

        n = len(history)
        if n < 2:
            return

        tensor_names = sorted(history[0].keys())
        new_priors: dict[str, torch.Tensor] = {}

        for name in tensor_names:
            rows = []
            for d in history:
                if name not in d:
                    break
                rows.append(d[name].flatten().to(torch.float32))
            if len(rows) < 2:
                continue
            mat = torch.stack(rows)  # [H, numel]

            # Power iteration for PC1 — warm-start from previous prior
            seed = self.priors.get(name)
            v = _power_iteration_pc1(mat, n_iters=20, initial_guess=seed)
            v = v / (v.norm() + 1e-12)

            # L2 regularize toward previous prior (RNA theory, arXiv 1805.09639)
            if name in self.priors and self.l2_reg > 0:
                prev = self.priors[name]
                # Blend: v_new = (1 - l2_reg) * v + l2_reg * v_prev
                # Re-normalize after blending to keep unit norm
                cos_sim = torch.dot(v, prev).clamp(-1, 1)
                # If directions agree (cos > 0), blend toward previous
                v = (1 - self.l2_reg) * v + self.l2_reg * (cos_sim.sign() * prev)
                v = v / (v.norm() + 1e-12)

            new_priors[name] = v

        # Track prior stability (cosine between consecutive priors)
        for name, v in new_priors.items():
            if name in self._prev_priors:
                prev = self._prev_priors[name]
                if prev.numel() == v.numel():
                    cos = torch.dot(v, prev).clamp(-1, 1).item()
                    self._prior_cosines.setdefault(name, []).append(cos)
        self._prev_priors = dict(new_priors)

        self.priors = new_priors
        self._last_update_step = -1  # caller sets step externally

    def mark_updated(self, step: int) -> None:
        self._last_update_step = step

    def reset_priors(self) -> None:
        """Clear priors and history — called on regime transition."""
        self.priors.clear()
        self.gain_map.clear()
        self._delta_history.clear()
        self._last_update_step = -1
        self._prev_priors.clear()
        self._prior_cosines.clear()

    def state_dict(self) -> dict:
        """Serialize the run-wide PSA accumulation for checkpoint resume.

        Mirrors ``LAWAAverager.state_dict()`` /
        ``ActivationFingerprintTracker.state_dict()``. The resume-persistent
        surface is the accumulated subspace-prior state — the per-step
        incremental ``_delta_history`` ring buffer (which feeds
        :meth:`extract_priors` AND the run-end ``layer_delta_analysis`` /
        GOAL §4 rank-1-dominance summary), the extracted PC1 ``priors`` (which
        drive :meth:`amplify_gradients` in the production path), ``_prev_priors``
        (the L2-reg blend anchor for the next extract), the unbounded
        ``_prior_cosines`` stability series, and ``_last_update_step`` (the
        :meth:`should_update` timing) — NOT the config (re-supplied in
        ``__init__``) or ``gain_map`` (recomputed every ``update_interval``
        cycles by :meth:`compute_gain_map`). Without persisting this a
        fault/periodic resume rebuilds ``psa_prior`` empty: amplification is
        silently off until 2 deltas re-accumulate and the next extract fires,
        and the run-end ``layer_delta_analysis`` (gated on ``history_count >=
        2``) is omitted entirely if the residual run is short — a silent
        resume-state-loss sibling to the fixed LAWA / act-regime / dynfreeze /
        efficiency-accounting gaps. Tensors move to CPU at snapshot time for
        checkpoint portability and move back to device at use-site
        (``extract_priors`` runs on CPU; ``amplify_gradients`` does
        ``v.to(g.device)``), so the unit-normalized PC1 directions round-trip
        unchanged.
        """
        return {
            "delta_history": [
                {k: v.cpu() for k, v in d.items()} for d in self._delta_history
            ],
            "priors": {k: v.cpu() for k, v in self.priors.items()},
            "prev_priors": {k: v.cpu() for k, v in self._prev_priors.items()},
            "prior_cosines": {k: list(v) for k, v in self._prior_cosines.items()},
            "last_update_step": self._last_update_step,
        }

    def load_state_dict(self, state: dict | None) -> None:
        """Restore run-wide PSA accumulation from a checkpoint.

        Inverse of :meth:`state_dict`. Tolerant of a partial / legacy dict and
        of ``None``: a missing key leaves the constructed default, so a pre-fix
        checkpoint (or a checkpoint from an ``enable_psa: false`` run) loads
        cleanly as a fresh prior. The ``_delta_history`` ring buffer is rebuilt
        with the constructed ``maxlen`` so a history larger than the checkpoint's
        ``history_length`` is trimmed the same way a live run would have trimmed
        it, and a smaller checkpoint history fills a larger window without
        padding.
        """
        if not state:
            return
        if "delta_history" in state:
            self._delta_history = deque(
                [{k: v for k, v in d.items()} for d in state["delta_history"]],
                maxlen=self.history_length,
            )
        if "priors" in state:
            self.priors = dict(state["priors"])
        if "prev_priors" in state:
            self._prev_priors = dict(state["prev_priors"])
        if "prior_cosines" in state:
            self._prior_cosines = {
                k: list(v) for k, v in state["prior_cosines"].items()
            }
        if "last_update_step" in state:
            self._last_update_step = int(state["last_update_step"])

    def amplify_gradients(
        self,
        model: torch.nn.Module,
        gain_override: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Amplify gradients in-place. Returns per-tensor amplification stats."""
        stats: dict[str, float] = {}
        if not self.priors:
            return stats

        for name, param in iter_lora_params(model):
            if param.grad is None or name not in self.priors:
                continue

            v = self.priors[name]
            g = param.grad.data.flatten().to(torch.float32)
            v = v.to(g.device)

            gamma = self.gain
            if gain_override and name in gain_override:
                gamma = gain_override[name]

            proj = torch.dot(g, v)
            amplified = g + gamma * proj * v

            # Clamp to prevent gradient explosion (safety)
            orig_norm = g.norm()
            new_norm = amplified.norm()
            if orig_norm > 1e-12 and new_norm > 2.0 * orig_norm:
                amplified = amplified * (2.0 * orig_norm / new_norm)

            param.grad.data = amplified.reshape(param.grad.shape).to(
                param.grad.dtype
            )
            stats[name] = (new_norm / (orig_norm + 1e-12)).item()

        return stats

    def compute_gain_map(
        self,
        model: torch.nn.Module,
        regime: str = "stable",
    ) -> dict[str, float]:
        """Compute per-tensor adaptive gain based on module type and regime.

        out_proj gets higher gain (stable signal, early_dir_cos 0.30-0.42).
        MLP deep layers get lower gain (noisier).
        Regime scaling: STABLE=1.0, PLATEAU=configurable, TRANSITION=0.0.
        """
        regime_scale = self.regime_gain_factor(regime)
        gain_map: dict[str, float] = {}
        for name, _param in iter_lora_params(model):
            if name not in self.priors:
                gain_map[name] = 0.0
                continue

            base = self.gain
            lt = classify_layer_type(name)
            if lt == LayerType.ATTENTION_OUT:
                base = self.gain * 1.2
            elif lt == LayerType.ATTENTION_V:
                base = self.gain * 1.1
            elif lt == LayerType.MLP:
                base = self.gain * 0.7

            gain_map[name] = base * regime_scale

        self.gain_map = gain_map
        return gain_map

    def regime_gain_factor(self, regime: str) -> float:
        """Return gain scaling factor for the given training regime.

        STABLE: full amplification (priors are reliable).
        PLATEAU: reduced amplification (priors may be stale).
        TRANSITION: zero amplification (priors are invalid).
        """
        if regime == "stable":
            return 1.0
        if regime == "transition":
            return 0.0
        if regime == "plateau":
            return self.regime_plateau_gain
        return 1.0


def _power_iteration_pc1(
    mat: torch.Tensor,
    n_iters: int = 20,
    initial_guess: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract PC1 direction via power iteration on mat^T @ mat.

    Args:
        mat: [H, numel] matrix where each row is a delta snapshot.
        n_iters: Number of power iteration steps.
        initial_guess: Optional [numel] vector to seed iteration. When the
            previous prior is available, passing it here gives faster
            convergence and smoother prior continuity.

    Returns:
        Unit vector [numel] — the dominant eigenvector (PC1).
    """
    # Covariance-like: C = mat^T @ mat, shape [numel, numel]
    # But we don't need to materialize C; just apply mat^T @ (mat @ v)
    numel = mat.shape[1]
    if initial_guess is not None and initial_guess.numel() == numel:
        v = initial_guess.to(device=mat.device, dtype=mat.dtype)
    else:
        v = torch.randn(numel, device=mat.device, dtype=mat.dtype)
    v = v / (v.norm() + 1e-12)

    for _ in range(n_iters):
        # w = C @ v = mat^T @ (mat @ v)
        w = mat.T @ (mat @ v)
        norm = w.norm()
        if norm < 1e-12:
            break
        v = w / norm

    return v


def amplify_gradients_psa(
    model: torch.nn.Module,
    psa_prior: PSAPrior,
    gain_map: dict[str, float] | None = None,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    """Training loop hook — amplify gradients along PSA priors.

    Called between loss.backward() and optimizer.step().
    """
    if not enabled or not psa_prior.priors:
        return {}
    return psa_prior.amplify_gradients(model, gain_override=gain_map)


def summarize_by_layer_type(
    per_tensor_stats: dict[str, float],
    prior_cosines: dict[str, list[float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-tensor PSA stats by layer type.

    GOAL §4 step 2: "DeltaNet / Attention / FFN で rank-1 支配度・方向安定性を分離測定"

    Args:
        per_tensor_stats: Per-tensor amplification ratios from amplify_gradients.
        prior_cosines: Per-tensor cosine history from PSAPrior._prior_cosines.

    Returns:
        Dict mapping layer_type → {count, amp_mean, amp_std, prior_stability_mean}.
        Only layer types with at least one tensor are included.
    """
    grouped: dict[str, list[float]] = {}
    for name, val in per_tensor_stats.items():
        lt = classify_layer_type(name).value
        grouped.setdefault(lt, []).append(val)

    result: dict[str, dict[str, float]] = {}
    for lt, vals in sorted(grouped.items()):
        n = len(vals)
        mean = sum(vals) / n if n > 0 else 0.0
        var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        entry: dict[str, float] = {
            "count": float(n),
            "amp_mean": mean,
            "amp_std": var**0.5,
        }
        # Prior stability: mean cosine of consecutive prior directions
        if prior_cosines:
            cosines_for_type = [
                coses[-1]
                for name, coses in prior_cosines.items()
                if classify_layer_type(name).value == lt and coses
            ]
            if cosines_for_type:
                entry["prior_stability_mean"] = sum(cosines_for_type) / len(
                    cosines_for_type
                )
        result[lt] = entry

    return result
