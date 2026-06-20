"""Stability-epoch estimation: regime signal → compromise-policy input (GOAL §3.1 Phase 2 candidate 3).

``freeze_schedule``'s ``compromise`` policy (design §4.1, GOAL §3.1 candidate 3)
freezes output-side-first but defers each layer until its *stability floor* is
reached: ``frozen_at_epoch = max(nominal, stability_epoch)``. That
``stability_epoch`` map was the one piece the planner flagged as a separate,
[UNVERIFIED] input (the ``freeze_schedule`` module docstring: "estimating
per-layer stability ... is a separate, [UNVERIFIED] step"). Nothing produced
it, so candidate 3 was only ever exercised with hand-authored maps
(``tests/test_freeze_schedule.TestCompromise``).

``estimate_stability_epochs`` closes that gap. It turns a per-layer stability
time-series — produced each epoch by
:meth:`src.tg_lora.dynamic_freeze.DynamicFreezeController.compute_r_A`
(``{layer: r_A}``, *lower = quieter* LoRA delta = stabler, design §5.3
"安定した層から固める") — into the ``{layer: earliest_confirmed_stable_epoch}``
map the compromise planner consumes. Pure arithmetic, model-free: GOAL §7's
"verify the mechanism before trusting a GPU run".

Convention (matches the producer): a lower metric is quieter/stabler. A layer
is *confirmed stable* at epoch ``e`` once the trailing ``patience`` observations
are all ``<= threshold`` — a single one-off dip does not confirm (the blip
test below). ``min_epoch`` forbids trusting stability before warmup.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.lora_utils import iter_all_lora_params_by_layer, set_trainable_lora_layers
from src.tg_lora.dynamic_freeze import DynamicFreezeController
from src.tg_lora.freeze_schedule import FreezeSchedule, FreezeScheduleConfig
from src.tg_lora.freeze_stability import estimate_stability_epochs

NUM_LAYERS = 6
HIDDEN = 8
VOCAB = 16


# ---------------------------------------------------------------------------
# 1. Pure-arithmetic estimation (no model)
# ---------------------------------------------------------------------------


class TestEstimateStabilityEpochs:
    def test_quiet_layer_confirms_at_patience_window_end(self):
        # Layer 5 goes quiet from epoch 1. patience=2 needs two consecutive
        # quiet observations (epochs 1 and 2), so confirmation lands at e=2.
        stab = estimate_stability_epochs(
            {5: [0.5, 0.01, 0.01, 0.01]}, threshold=0.02, patience=2
        )
        assert stab == {5: 2}

    def test_patience_one_confirms_at_first_quiet_epoch(self):
        # patience=1: any single quiet observation confirms immediately.
        stab = estimate_stability_epochs(
            {5: [0.5, 0.5, 0.01, 0.5]}, threshold=0.02, patience=1
        )
        assert stab == {5: 2}

    def test_never_quiet_layer_is_omitted(self):
        # Stays above threshold the whole run -> never confirmed -> absent.
        stab = estimate_stability_epochs(
            {5: [0.5, 0.5, 0.5]}, threshold=0.02, patience=1
        )
        assert stab == {}

    def test_single_quiet_blip_below_patience_is_not_confirmed(self):
        # One quiet epoch sandwiched by noise: never `patience` consecutive.
        stab = estimate_stability_epochs(
            {5: [0.5, 0.01, 0.5, 0.01, 0.5]}, threshold=0.02, patience=2
        )
        assert stab == {}

    def test_min_epoch_delays_confirmation_past_warmup(self):
        # Quiet from epoch 0, but min_epoch=3 forbids trusting it before
        # warmup -> first admissible confirmation is the quiet epoch at e=3.
        stab = estimate_stability_epochs(
            {5: [0.01, 0.01, 0.01, 0.01, 0.01]}, threshold=0.02, patience=1, min_epoch=3
        )
        assert stab == {5: 3}

    def test_min_epoch_combines_with_patience(self):
        # patience=2, min_epoch=3: the trailing-2 window must end at e>=3.
        # e=3 (window epochs 2,3) is the first admissible all-quiet window.
        stab = estimate_stability_epochs(
            {5: [0.01, 0.01, 0.01, 0.01, 0.01]}, threshold=0.02, patience=2, min_epoch=3
        )
        assert stab == {5: 3}

    def test_multiple_layers_estimated_independently(self):
        # Two layers, different convergence times, estimated independently.
        stab = estimate_stability_epochs(
            {31: [0.5, 0.5, 0.01, 0.01], 30: [0.5, 0.5, 0.5, 0.01, 0.01, 0.01]},
            threshold=0.02,
            patience=2,
        )
        # 31: trailing-2 window epochs 2,3 -> e=3.
        # 30: trailing-2 window epochs 3,4 -> e=4.
        assert stab == {31: 3, 30: 4}

    def test_empty_series_returns_empty(self):
        assert estimate_stability_epochs({}, threshold=0.02) == {}
        # A layer with no observations cannot be confirmed.
        assert estimate_stability_epochs({5: []}, threshold=0.02) == {}

    def test_invalid_patience_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="patience"):
            estimate_stability_epochs({5: [0.01]}, threshold=0.02, patience=0)

    def test_invalid_min_epoch_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="min_epoch"):
            estimate_stability_epochs({5: [0.01]}, threshold=0.02, min_epoch=-1)


# ---------------------------------------------------------------------------
# 2. Estimator → compromise planner composition (pure arithmetic)
# ---------------------------------------------------------------------------


class TestCompromiseComposition:
    def test_estimated_epochs_drive_compromise_freeze_timing(self):
        # The estimator output plugs straight into FreezeScheduleConfig under
        # 'compromise': frozen_at_epoch = max(nominal, stability_epoch).
        stab = estimate_stability_epochs(
            {31: [0.5, 0.5, 0.01, 0.01], 30: [0.5, 0.5, 0.5, 0.01, 0.01, 0.01],
             29: [0.01, 0.01, 0.01]},
            threshold=0.02,
            patience=2,
        )
        # 31 -> e=3 (window 2,3); 30 -> e=4 (window 3,4); 29 -> e=1 (window 0,1).
        assert stab == {31: 3, 30: 4, 29: 1}

        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=[29, 30, 31],
                num_epochs=10,
                max_depth=3,
                start_epoch=2,
                spacing=1,
                policy="compromise",
                stability_epoch=stab,
            )
        )
        # output-descending order [31, 30, 29]; nominals 2, 3, 4.
        # max(nominal, stability): 31->max(2,3)=3, 30->max(3,4)=4, 29->max(4,1)=4.
        assert sched.frozen_at_epoch == {31: 3, 30: 4, 29: 4}


# ---------------------------------------------------------------------------
# 3. Real-producer bridge: DynamicFreezeController.compute_r_A → estimator
#    (the actual r_A signal drives the estimate, not synthetic numbers)
# ---------------------------------------------------------------------------


class _LoRALinear(nn.Module):
    """Frozen base + trainable LoRA, both on the forward graph (lora_B init 0)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, hidden, bias=False)
        self.lora_A = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.lora_A.t() @ self.lora_B.t()


class _Layer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = _LoRALinear(hidden)

    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        return (hidden_states + self.proj(hidden_states),)  # residual


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_Layer(HIDDEN) for _ in range(NUM_LAYERS)])

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        del labels, attention_mask
        from types import SimpleNamespace

        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)[0]
        out = SimpleNamespace()
        out.loss = None
        out.logits = h
        return out


class TestRealRAProducerBridge:
    def test_compute_r_A_series_yields_stability_epochs(self):
        # Output-side layers {5,4,3} stay quiet (lora_B left at 0 init -> A_fro
        # ~0 -> r_A ~0); front layers {2,1,0} are made noisy (lora_B scaled up
        # each cycle -> A_fro grows -> r_A well above tau). The r_A the real
        # producer emits each cycle feeds the estimator directly.
        torch.manual_seed(0)
        model = _Model()
        set_trainable_lora_layers(model, set(range(NUM_LAYERS)))
        quiet = {5, 4, 3}
        noisy = {2, 1, 0}

        dfc = DynamicFreezeController(
            tau=0.02, window=4, all_layer_indices=list(range(NUM_LAYERS))
        )
        # Seed noisy layers with a non-zero LoRA delta first (lora_B starts at
        # 0, so mul_-only would leave them quiet); then scale up each cycle so
        # A_fro grows and r_A stays well above tau (mirrors the invivo suite's
        # _seed_r_A_history).
        with torch.no_grad():
            for layer in noisy:
                for _name, p in iter_all_lora_params_by_layer(model)[layer]:
                    if "lora_B" in _name:
                        p.add_(0.1)
        series: dict[int, list[float]] = {layer: [] for layer in range(NUM_LAYERS)}
        for cycle in range(6):
            if cycle >= 1:
                with torch.no_grad():
                    for layer in noisy:
                        for _name, p in iter_all_lora_params_by_layer(model)[layer]:
                            if "lora_B" in _name:
                                p.mul_(1.5)
            r_A = dfc.compute_r_A(model, cycle)
            for layer in range(NUM_LAYERS):
                series[layer].append(r_A[layer])

        stab = estimate_stability_epochs(series, threshold=0.02, patience=2)

        # Exactly the quiet output-side block confirms; none of the noisy front.
        assert set(stab) == quiet
        assert set(stab).isdisjoint(noisy)
        assert all(v <= 2 for v in stab.values())  # early confirmation

        # The data-driven map is accepted by the compromise planner and freezes
        # each quiet layer no earlier than its confirmed-stable epoch.
        sched = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=list(range(NUM_LAYERS)),
                num_epochs=20,
                max_depth=NUM_LAYERS,
                start_epoch=0,
                spacing=1,
                policy="compromise",
                stability_epoch=stab,
            )
        )
        for layer in quiet:
            assert layer in sched.frozen_at_epoch
            assert sched.frozen_at_epoch[layer] >= stab[layer]
