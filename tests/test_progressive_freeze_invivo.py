"""In-vivo MS-007 trio measurement: same quality, fewer backwards (design §7/§8).

The per-module and composition suites
(:mod:`tests.test_progressive_freeze_e2e` and its dependencies) prove the
*mechanism*: a schedule predicts freezes, a controller applies them, a local
loss trains the front while the suffix is cut, a frontier predicts the savings,
a cost accountant tallies the FLOPs. None of them runs the MS-007 trio
(``activation_cache`` + ``split_layer`` via :meth:`compute_local_loss` +
``dynamic_freeze``) against a **live backward pass** to show the headline claim
— *same quality, fewer backwards* — as a *positive, realized* result rather than
a model-free FLOPs gate that merely blocks false positives (design §7:
"verify the mechanism before trusting a GPU run"; GOAL §4: the run is a success
only if valid_loss stays within tolerance **and** backward work actually drops).

This file closes that gap with a tiny but genuinely learnable LoRA model driven
through full backprop vs. the Level-2 trio vs. the Level-1 freeze-only baseline
(MS-006). A ``register_full_backward_hook`` per decoder layer counts the
*realized* backward traversals, so the savings are measured on the actual
autograd graph, not predicted:

* **Level-1 (MS-006) in-vivo savings == 0.** Freezing a layer's weight grad
  does not stop the activation gradient from traversing it, so the backward
  hook fires for every layer every epoch — no realized reduction. This is the
  *in-vivo* corroboration that MS-006 is structurally unworkable for cutting
  backward, and it exposes that :class:`FreezeCostAccountant`'s Level-1 model
  *overstates* realizable savings (it credits weight-grad FLOPs that never
  translate into fewer traversals).

* **Level-2 trio in-vivo savings > 0, and == the accountant prediction.** The
  local loss is computed at the frozen boundary's input, so backward never
  enters the frozen suffix — the realized reduction exactly matches the
  model-free accountant's Level-2 prediction. This is the positive result:
  the trio really does cut backwards, by exactly the predicted amount.

* **Same quality.** Held at the warmup-end point, the trio's final valid_loss
  matches full backprop within GOAL §4 tolerance while spending fewer
  backwards — the efficiency frontier this feature exists to produce.

* **``dynamic_freeze`` drives the same output-first contiguous order.** On a
  controlled quiet/noisy ``r_A`` state, :meth:`DynamicFreezeController.decide_freeze`
  freezes exactly the output-side contiguous quiet block the trio's schedule
  freezes — the third leg of the trio is the same decision, made dynamically.

The fixture is deliberately **different** from the e2e suite's
``_LoRADecoderLayer``: a residual connection, a small frozen base (std=0.02),
and a parameter-free ``LayerNorm`` keep a 6-layer stack's activations stable
over 60 epochs, so the model genuinely learns (3.5 → 0.4 cross-entropy) instead
of saturating softmax to uniform the way 8 stacked frozen random linears do.
The LoRA path stays on the forward graph with the base frozen, so
``requires_grad=False`` from :meth:`ProgressiveFreezeController.apply_freeze`
genuinely removes a layer from the gradient flow — the same faithful property
the e2e suite relies on.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.lora_utils import iter_all_lora_params_by_layer, set_trainable_lora_layers
from src.tg_lora.activation_matching import ActivationMatchingLoss
from src.tg_lora.dynamic_freeze import DynamicFreezeController
from src.tg_lora.freeze_cost import FreezeCostAccountant, LayerBackwardCost
from src.tg_lora.freeze_schedule import FreezeSchedule, FreezeScheduleConfig
from src.tg_lora.progressive_freeze import ProgressiveFreezeController

NUM_LAYERS = 6
HIDDEN = 24
VOCAB = 32

# Few-data, many-epoch regime (GOAL §4.3): one batch, many epochs. Single batch
# also makes the per-epoch backward-traversal count a clean integer (one fire
# per layer per epoch), so the in-vivo reduction is exact arithmetic.
WARMUP = 45   # freeze only after the model has converged onto a flat tail
TOTAL = 60
DEPTH = 3     # freeze the 3 output-side layers (5, 4, 3)
LR = 1.0


# ---------------------------------------------------------------------------
# Faithful learnable fixture: residual + small frozen base + param-free norm
# ---------------------------------------------------------------------------


class _LoRALinear(nn.Module):
    """Frozen base + trainable LoRA, both on the forward graph.

    ``base`` is frozen with a small init (std=0.02) so a stack of them does not
    explode/vanish the hidden state; ``lora_B`` starts at zero (standard LoRA
    init). Toggling the LoRA params' ``requires_grad`` is exactly what gates
    whether gradient reaches the layer — the property the backward-hook counts
    observe.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.base.weight, std=0.02)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.lora_A.t() @ self.lora_B.t()


class _Layer(nn.Module):
    """Residual + parameter-free LayerNorm: keeps activations stable over depth."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = _LoRALinear(hidden)

    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        h = hidden_states + self.proj(hidden_states)  # residual
        return (F.layer_norm(h, (h.shape[-1],)),)      # param-free norm


class _Model(nn.Module):
    """Tiny causal-LM-shaped model that genuinely runs its decoder layers."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_Layer(HIDDEN) for _ in range(NUM_LAYERS)])
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        del labels
        h = self.embed_tokens(input_ids)
        h = F.layer_norm(h, (HIDDEN,))
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        out = SimpleNamespace()
        out.loss = None
        out.logits = self.lm_head(h)
        return out


def _build_model(seed: int = 0) -> _Model:
    torch.manual_seed(seed)
    model = _Model()
    set_trainable_lora_layers(model, set(range(NUM_LAYERS)))
    return model


def _make_batches(batch: int = 4, seq: int = 6, seed: int = 123) -> list[dict]:
    """Single fixed batch (few-data regime)."""
    g = torch.Generator().manual_seed(seed)
    b = {
        "input_ids": torch.randint(0, VOCAB, (batch, seq), generator=g),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.randint(0, VOCAB, (batch, seq), generator=g),
    }
    return [b]


def _lm_loss(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    labels = batch["labels"]
    mask = batch["attention_mask"][:, 1:].float()
    s_logits = logits[:, :-1, :].contiguous()
    s_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(
        s_logits.reshape(-1, VOCAB), s_labels.reshape(-1), reduction="none"
    ).reshape(s_labels.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def _eval_loss(model: nn.Module, batches: list[dict]) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            tot += _lm_loss(out.logits, b).item()
            n += 1
    model.train()
    return tot / n


class _BackwardCounter:
    """Counts per-decoder-layer backward traversals via ``full_backward_hook``.

    One fire per layer per ``loss.backward()`` call — the realized backward
    work, as opposed to the FLOPs the accountant predicts from a cost table.
    """

    def __init__(self, model: nn.Module):
        self.counts = [0] * NUM_LAYERS
        self._hooks = [
            layer.register_full_backward_hook(self._mk(i))
            for i, layer in enumerate(model.layers)
        ]

    def _mk(self, i: int):
        def _hook(module, grad_input, grad_output):
            del module, grad_input, grad_output
            self.counts[i] += 1
        return _hook

    def total(self) -> int:
        return sum(self.counts)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()


def _trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Training regimes
# ---------------------------------------------------------------------------


def _run_baseline(batches: list[dict]) -> tuple[list[float], int]:
    model = _build_model(0)
    opt = torch.optim.SGD(_trainable_params(model), lr=LR)
    bc = _BackwardCounter(model)
    evals = [_eval_loss(model, batches)]
    for _ in range(TOTAL):
        for b in batches:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            _lm_loss(out.logits, b).backward()
            opt.step()
        evals.append(_eval_loss(model, batches))
    bc.remove()
    return evals, bc.total()


def _schedule() -> FreezeSchedule:
    return FreezeSchedule.plan(
        FreezeScheduleConfig(
            active_layer_indices=list(range(NUM_LAYERS)),
            num_epochs=TOTAL,
            max_depth=DEPTH,
            start_epoch=WARMUP,
            spacing=1,
            policy="output_first",
        )
    )


def _run_trio(batches: list[dict], schedule: FreezeSchedule) -> tuple[list[float], int]:
    """MS-007 trio: progressive freeze + boundary local loss (Level-2 suffix cut)."""
    model = _build_model(0)
    opt = torch.optim.SGD(_trainable_params(model), lr=LR)
    bc = _BackwardCounter(model)
    ctrl = ProgressiveFreezeController(
        start_cycle=WARMUP, active_layer_indices=set(range(NUM_LAYERS)), schedule=schedule
    )
    loss_fn = ActivationMatchingLoss()
    evals = [_eval_loss(model, batches)]
    for epoch in range(TOTAL):
        ctrl.progress(model, epoch, batches, "cpu")
        for b_idx, b in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            if ctrl.frozen_layers:
                boundary = min(ctrl.frozen_layers)
                loss = ctrl.compute_local_loss(model, b, loss_fn, batch_idx=b_idx, layer_idx=boundary)
            else:
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                loss = _lm_loss(out.logits, b)
            loss.backward()
            opt.step()
        evals.append(_eval_loss(model, batches))
    bc.remove()
    return evals, bc.total()


def _run_level1(batches: list[dict], schedule: FreezeSchedule) -> tuple[list[float], int]:
    """MS-006: freeze weight grads only, keep training on the FINAL task loss.

    The activation gradient still propagates through frozen layers to reach the
    unfrozen front (Level 1), so the suffix is *not* cut — the diagnostic for
    why this is unworkable for real backward savings.
    """
    model = _build_model(0)
    opt = torch.optim.SGD(_trainable_params(model), lr=LR)
    bc = _BackwardCounter(model)
    layer_map = iter_all_lora_params_by_layer(model)
    frozen: set[int] = set()
    evals = [_eval_loss(model, batches)]
    for epoch in range(TOTAL):
        for layer, at in schedule.frozen_at_epoch.items():
            if at == epoch and layer not in frozen:
                for _, p in layer_map[layer]:
                    p.requires_grad = False
                frozen.add(layer)
        for b in batches:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            _lm_loss(out.logits, b).backward()
            opt.step()
        evals.append(_eval_loss(model, batches))
    bc.remove()
    return evals, bc.total()


@pytest.fixture(scope="module")
def _invivo():
    """Train all three regimes once (~1-2s); every in-vivo test reads this."""
    batches = _make_batches()
    schedule = _schedule()
    base_evals, base_bw = _run_baseline(batches)
    trio_evals, trio_bw = _run_trio(batches, schedule)
    l1_evals, l1_bw = _run_level1(batches, schedule)
    return SimpleNamespace(
        batches=batches,
        schedule=schedule,
        base_evals=base_evals,
        base_bw=base_bw,
        trio_evals=trio_evals,
        trio_bw=trio_bw,
        l1_evals=l1_evals,
        l1_bw=l1_bw,
        warmup_final=base_evals[WARMUP],   # full-backprop quality at the freeze epoch
        base_final=base_evals[-1],
        trio_final=trio_evals[-1],
        l1_final=l1_evals[-1],
    )


# ---------------------------------------------------------------------------
# 1. Realized backward cost — the headline positive (and negative) result
# ---------------------------------------------------------------------------


class TestInVivoBackwardCost:
    """Measured backward traversals on the live autograd graph, vs. the accountant."""

    def test_fixture_genuinely_learns(self, _invivo):
        # Guard against a fixture regression that makes every comparison trivial:
        # the model must actually descend from near-uniform to a learned state.
        uniform = float(torch.log(torch.tensor(float(VOCAB))))
        assert _invivo.base_evals[0] > 3.0, "init loss should sit near uniform"
        assert _invivo.base_final < uniform - 0.5, "model must learn to converge"
        assert _invivo.warmup_final < _invivo.base_evals[0], "warmup must improve on init"

    def test_level1_freeze_only_cuts_no_backward_in_vivo(self, _invivo):
        # MS-006 in-vivo: freezing weight grads does not stop the activation
        # gradient, so every layer is still traversed every epoch — zero
        # realized reduction, despite the same schedule as the trio.
        invivo_l1_reduction = 1.0 - _invivo.l1_bw / _invivo.base_bw
        assert invivo_l1_reduction == pytest.approx(0.0, abs=1e-9)
        assert _invivo.l1_bw == _invivo.base_bw

    def test_level2_trio_cuts_backward_in_vivo(self, _invivo):
        # The positive result: the trio's boundary local loss never backprops
        # into the frozen suffix, so realized backward traversals drop.
        invivo_l2_reduction = 1.0 - _invivo.trio_bw / _invivo.base_bw
        assert invivo_l2_reduction > 0.0
        # ... and strictly more than the Level-1 freeze (which cuts nothing).
        invivo_l1_reduction = 1.0 - _invivo.l1_bw / _invivo.base_bw
        assert invivo_l2_reduction > invivo_l1_reduction

    def test_in_vivo_level2_matches_accountant_prediction_exactly(self, _invivo):
        # The realized reduction on the live graph equals the model-free
        # accountant's Level-2 prediction (uniform per-layer cost → both reduce
        # to the same frozen-suffix fraction of full backward). This is the
        # accountant's Level-2 model validated, not merely asserted.
        accountant = FreezeCostAccountant(
            layer_costs={
                i: LayerBackwardCost(weight_grad_flops=1, act_grad_flops=1)
                for i in range(NUM_LAYERS)
            },
            steps_per_epoch=1,
            num_epochs=TOTAL,
            frozen_at_epoch=dict(_invivo.schedule.frozen_at_epoch),
        )
        invivo_l2_reduction = 1.0 - _invivo.trio_bw / _invivo.base_bw
        assert invivo_l2_reduction == pytest.approx(
            accountant.reduction_rate(level=2), rel=1e-9
        )

    def test_accountant_level1_overstates_realizable_savings_in_vivo(self, _invivo):
        # The MS-006 gap: the accountant's Level-1 model credits weight-grad
        # FLOPs as saved (reduction_rate(level=1) > 0), but in vivo the
        # activation gradient still traverses frozen layers so zero backward
        # work is actually elided. Level 1 overstates; Level 2 does not.
        accountant = FreezeCostAccountant(
            layer_costs={
                i: LayerBackwardCost(weight_grad_flops=1, act_grad_flops=1)
                for i in range(NUM_LAYERS)
            },
            steps_per_epoch=1,
            num_epochs=TOTAL,
            frozen_at_epoch=dict(_invivo.schedule.frozen_at_epoch),
        )
        assert accountant.reduction_rate(level=1) > 0.0  # the overstatement
        invivo_l1_reduction = 1.0 - _invivo.l1_bw / _invivo.base_bw
        assert accountant.reduction_rate(level=1) > invivo_l1_reduction + 0.01


# ---------------------------------------------------------------------------
# 2. Same quality — the trio holds the warmup-end point for fewer backwards
# ---------------------------------------------------------------------------


class TestInVivoQuality:
    def test_trio_preserves_warmup_end_quality(self, _invivo):
        # Held at the freeze point, the boundary local loss pins the front to
        # its cached xin (the output it emitted at warmup end), so the trio
        # holds that quality rather than degrading it. The "same quality" half
        # of the claim, checked against the point full backprop reached.
        assert _invivo.trio_final <= _invivo.warmup_final * 1.03

    def test_trio_matches_full_backprop_within_tolerance(self, _invivo):
        # GOAL §4 success: valid_loss within tolerance of full backprop while
        # spending fewer backwards. The trio lands within 5% of the converged
        # full-backprop model (observed ~1.3%).
        assert _invivo.trio_final <= _invivo.base_final * 1.05

    def test_baseline_is_on_a_flat_tail_at_freeze_epoch(self, _invivo):
        # Honest framing of the quality result (GOAL §7: never overclaim): the
        # gap to full backprop is small because the baseline is already on a
        # flat tail at the freeze epoch, so the trio is not being asked to give
        # up much future gain. This documents the regime, not a hidden caveat.
        tail_drift = _invivo.warmup_final / _invivo.base_final
        assert tail_drift <= 1.05


# ---------------------------------------------------------------------------
# 3. dynamic_freeze drives the same output-first contiguous freeze order
# ---------------------------------------------------------------------------


def _seed_r_A_history(dfc: DynamicFreezeController, model: nn.Module,
                      quiet_layers: set[int], window: int) -> None:
    """Build a controlled quiet/noisy ``r_A`` history so ``decide_freeze`` fires.

    "Quiet" layers keep their LoRA state fixed (``A_fro`` unchanged → ``r_A``
    ~0); every other ("noisy") layer has its ``lora_B`` scaled up each cycle,
    so ``A_fro`` grows and ``r_A`` lands well above τ. After ``window+1`` cycles
    the controller has enough history for :meth:`decide_freeze` to act.
    """
    layers = sorted(dfc._all_layers)
    # Give noisy layers a non-zero LoRA delta first (lora_B starts at 0).
    with torch.no_grad():
        for l in layers:
            if l in quiet_layers:
                continue
            for name, p in iter_all_lora_params_by_layer(model)[l]:
                if "lora_B" in name:
                    p.add_(0.1)
    for cycle in range(window + 1):
        if cycle >= 1:
            with torch.no_grad():
                for l in layers:
                    if l in quiet_layers:
                        continue
                    for name, p in iter_all_lora_params_by_layer(model)[l]:
                        if "lora_B" in name:
                            p.mul_(1.5)
        dfc.compute_r_A(model, cycle)


class TestDynamicFreezeDrivesTrio:
    """The trio's third leg: ``decide_freeze`` picks the output-side contiguous
    block the trio's schedule freezes, but dynamically from ``r_A`` quietness."""

    def test_freezes_output_side_contiguous_quiet_block(self):
        # Quiet output layers {5,4,3}, noisy front {2,1,0}: the block is
        # {5,4,3} — exactly the output_first schedule's depth-3 freeze set.
        model = _build_model(0)
        dfc = DynamicFreezeController(
            tau=0.02, window=4, all_layer_indices=list(range(NUM_LAYERS))
        )
        _seed_r_A_history(dfc, model, quiet_layers={5, 4, 3}, window=4)
        assert dfc.decide_freeze(4) == [5, 4, 3]

    def test_block_stops_at_first_noisy_layer(self):
        # Only layer 5 quiet: block is {5}, not {5,4,...} — the scan halts at
        # the first noisy layer, so no quiet layer upstream of noise is frozen.
        model = _build_model(0)
        dfc = DynamicFreezeController(
            tau=0.02, window=4, all_layer_indices=list(range(NUM_LAYERS))
        )
        _seed_r_A_history(dfc, model, quiet_layers={5}, window=4)
        assert dfc.decide_freeze(4) == [5]

    def test_no_quiet_layers_freezes_nothing(self):
        # All layers noisy: empty decision — the guard never freezes orphans.
        model = _build_model(0)
        dfc = DynamicFreezeController(
            tau=0.02, window=4, all_layer_indices=list(range(NUM_LAYERS))
        )
        _seed_r_A_history(dfc, model, quiet_layers=set(), window=4)
        assert dfc.decide_freeze(4) == []

    def test_block_extends_contiguously_from_existing_block(self):
        # Block already {5,4}; layers 3 and 2 go quiet: extend by exactly {3,2}.
        # The scan skips the frozen 5 and 4, then includes the contiguous quiet
        # run {3,2} and halts at the first noisy layer — the decision returns
        # only the *new* layers to freeze, preserving the output-first order.
        model = _build_model(0)
        dfc = DynamicFreezeController(
            tau=0.02, window=4, all_layer_indices=list(range(NUM_LAYERS))
        )
        dfc._frozen_block = [5, 4]
        _seed_r_A_history(dfc, model, quiet_layers={5, 4, 3, 2}, window=4)
        assert dfc.decide_freeze(4) == [3, 2]

    def test_existing_block_rejects_non_contiguous_extension(self):
        # The defensive contiguity guard: if the already-frozen block is not
        # output-contiguous (an invariant violation), a "quiet" layer upstream
        # of the gap is NOT frozen — the block never creates orphans. Here
        # {5,3} has a gap at 4; quiet {4} would extend 5 but not 3, so the
        # non-contiguous new-layer set is rejected and nothing is frozen.
        model = _build_model(0)
        dfc = DynamicFreezeController(
            tau=0.02, window=4, all_layer_indices=list(range(NUM_LAYERS))
        )
        dfc._frozen_block = [5, 3]  # gap at 4 (invariant violation, defensive case)
        _seed_r_A_history(dfc, model, quiet_layers={5, 4, 3, 2}, window=4)
        assert dfc.decide_freeze(4) == []
