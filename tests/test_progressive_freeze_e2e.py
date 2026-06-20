"""End-to-end integration: freeze schedule + frontier + activation-matching loss.

The per-module suites each lock down one piece in isolation:

* ``test_progressive_freeze``  — the controller, a single frozen layer
* ``test_freeze_schedule``     — the planner, pure arithmetic (no model)
* ``test_freeze_frontier``     — planner → accountant glue, pure arithmetic
* ``test_activation_matching`` — the loss, pure tensor math (no model)
* ``test_freeze_cost``         — the accountant, pure arithmetic

None of them drives a real model through the **composition**: a
:class:`FreezeSchedule` predicting which layers freeze when, a
:class:`ProgressiveFreezeController` actually freezing them on a live autograd
graph, an activation-matching local loss training the still-active front, and
the freeze-frontier predicting the realized depth/savings of the very schedule
the loop just executed. That composition — in particular gradient flow across
*partially-frozen multi-layer* states — is the highest-risk regression surface
for this feature set (a silent bug where the second freeze breaks the first, or
where the suffix cut leaks gradient into a frozen layer, passes every per-module
suite). It is pinned here with a tiny but real LoRA transformer.

The fixture below is deliberately **faithful**: the LoRA path is on the forward
graph and the base path is frozen, so ``requires_grad=False`` from
:meth:`ProgressiveFreezeController.apply_freeze` genuinely removes a layer's
params from the gradient flow — the exact relationship the proj-only unit fixture
cannot observe.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from src.model.lora_utils import iter_all_lora_params_by_layer, set_trainable_lora_layers
from src.tg_lora.activation_matching import ActivationMatchingLoss
from src.tg_lora.freeze_cost import LayerBackwardCost
from src.tg_lora.freeze_frontier import FrontierSpec, evaluate_schedule, frontier
from src.tg_lora.freeze_schedule import FreezeSchedule, FreezeScheduleConfig
from src.tg_lora.progressive_freeze import ProgressiveFreezeController

NUM_LAYERS = 6
HIDDEN = 8
VOCAB = 16


class _LoRALinear(nn.Module):
    """Base (frozen) + LoRA (trainable) linear, both on the forward graph.

    ``lora_B`` starts at zero so the adapter contributes nothing until trained
    (standard LoRA init). The base weight has ``requires_grad=False``, mirroring
    real PEFT where only adapters train — so toggling the LoRA params'
    ``requires_grad`` is exactly what gates whether gradient reaches the layer.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, hidden, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.lora_A.t() @ self.lora_B.t()


class _LoRADecoderLayer(nn.Module):
    """Decoder layer whose forward runs an actual LoRA linear transform.

    The ``self_attn`` q/v LoRA linears exist only for layer-index-mapping
    realism (they are not on the forward path, like the unit-test fixture); the
    layer's real transform is ``proj``, whose LoRA params are the ones the
    gradient-flow assertions inspect.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = _LoRALinear(hidden)
        self.self_attn.v_proj = _LoRALinear(hidden)
        self.proj = _LoRALinear(hidden)

    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        return (self.proj(hidden_states),)


class _LoRAModel(nn.Module):
    """Tiny causal-LM-shaped model that genuinely runs its decoder layers."""

    def __init__(self, num_layers: int = NUM_LAYERS, hidden: int = HIDDEN, vocab: int = VOCAB) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_LoRADecoderLayer(hidden) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        del labels
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        out = SimpleNamespace()
        out.loss = None
        out.logits = self.lm_head(h)
        return out


def _build_model(seed: int = 0) -> _LoRAModel:
    """All layers trainable; deterministic init."""
    torch.manual_seed(seed)
    model = _LoRAModel()
    set_trainable_lora_layers(model, set(range(NUM_LAYERS)))
    return model


def _loader(batch: int = 2, seq: int = 5) -> list[dict]:
    one = {
        "input_ids": torch.randint(0, VOCAB, (batch, seq)),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.randint(0, VOCAB, (batch, seq)),
    }
    return [one]


def _frozen_layer_indices(model: nn.Module) -> set[int]:
    """Layers whose LoRA params are all frozen (requires_grad=False).

    Valid as the "frozen set" here because every layer is active in the fixture,
    so a non-trainable LoRA param can only come from ``apply_freeze``.
    """
    frozen: set[int] = set()
    for idx, params in iter_all_lora_params_by_layer(model).items():
        if all(not p.requires_grad for _, p in params):
            frozen.add(idx)
    return frozen


def _drive_schedule(
    model: nn.Module, schedule: FreezeSchedule, loader: list[dict]
) -> dict[int, ProgressiveFreezeController]:
    """Freeze every layer the schedule names, in freeze-epoch order.

    Each layer gets its own controller (the controller is a single-freeze gate),
    with ``xin`` cached at the moment it freezes. Returns ``{layer: controller}``
    so callers can later run that layer's activation-matching local loss.
    """
    controllers: dict[int, ProgressiveFreezeController] = {}
    for layer_idx, _epoch in sorted(schedule.frozen_at_epoch.items(), key=lambda kv: kv[1]):
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            freeze_layer=layer_idx,
            active_layer_indices=set(range(NUM_LAYERS)),
        )
        ctrl.cache_xin(model, loader, "cpu")
        ctrl.apply_freeze(model)
        controllers[layer_idx] = ctrl
    return controllers


# ---------------------------------------------------------------------------
# 1. Schedule ↔ controller ↔ frontier, driven epoch-by-epoch on a live model
# ---------------------------------------------------------------------------


class TestScheduleControllerFrontierEndToEnd:
    """The structural composition: what the schedule predicts must be exactly
    what the controller freezes on the model, and exactly what the frontier
    reports as the realized depth — checked at every epoch, not just the end."""

    def test_loop_freezes_exactly_the_scheduled_layers_each_epoch(self):
        # output_first over all 6 layers: freeze 5,4,3,2,1 at epochs 1..5
        # (layer 0 stays active as the deepest trainable front). spacing=1.
        schedule = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=list(range(NUM_LAYERS)),
                num_epochs=NUM_LAYERS,
                max_depth=NUM_LAYERS - 1,
                start_epoch=1,
                spacing=1,
                policy="output_first",
            )
        )
        assert schedule.frozen_at_epoch == {5: 1, 4: 2, 3: 3, 2: 4, 1: 5}
        assert schedule.realized_depth == NUM_LAYERS - 1

        model = _build_model()
        loader = _loader()
        controllers: dict[int, ProgressiveFreezeController] = {}

        for epoch in range(NUM_LAYERS):
            due = [l for l, e in schedule.frozen_at_epoch.items() if e == epoch]
            for layer_idx in due:
                ctrl = ProgressiveFreezeController(
                    start_cycle=1,
                    freeze_layer=layer_idx,
                    active_layer_indices=set(range(NUM_LAYERS)),
                )
                ctrl.cache_xin(model, loader, "cpu")
                ctrl.apply_freeze(model)
                controllers[layer_idx] = ctrl

            expected = {l for l, e in schedule.frozen_at_epoch.items() if e <= epoch}
            # The controller-frozen set matches the schedule prediction exactly,
            # at every epoch — and freezing more never unfreezes a prior layer.
            assert _frozen_layer_indices(model) == expected

        assert set(controllers) == set(schedule.frozen_at_epoch)

    def test_frontier_realized_depth_matches_actually_frozen_count(self):
        schedule = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=list(range(NUM_LAYERS)),
                num_epochs=NUM_LAYERS,
                max_depth=NUM_LAYERS - 1,
                start_epoch=1,
                policy="output_first",
            )
        )
        model = _build_model()
        _drive_schedule(model, schedule, _loader())

        spec = FrontierSpec(
            layer_costs={i: LayerBackwardCost() for i in range(NUM_LAYERS)},
            steps_per_epoch=1,
            num_epochs=NUM_LAYERS,
            active_layer_indices=tuple(range(NUM_LAYERS)),
            start_epoch=1,
        )
        point = evaluate_schedule(
            spec, "output_first", depth=schedule.realized_depth, level=1
        )

        # The frontier point for this depth describes the same schedule the loop
        # ran, and its realized depth equals the number of layers actually frozen
        # on the model — planner, accountant, and controller agree.
        assert point.frozen_at_epoch == schedule.frozen_at_epoch
        assert point.depth == schedule.realized_depth
        assert point.depth == len(_frozen_layer_indices(model))

    def test_frontier_reduction_is_monotonic_and_origin_anchored(self):
        spec = FrontierSpec(
            layer_costs={i: LayerBackwardCost() for i in range(NUM_LAYERS)},
            steps_per_epoch=1,
            num_epochs=NUM_LAYERS,
            active_layer_indices=tuple(range(NUM_LAYERS)),
            start_epoch=1,
            policies=("output_first",),
            levels=(1, 2),
        )
        points = frontier(spec)
        # Depth-0 origin per (policy, level): no freeze, no savings.
        origins = [p for p in points if p.depth == 0]
        assert len(origins) == 2
        for origin in origins:
            assert origin.reduction_rate == 0.0
            assert origin.peak_vram_saved_bytes == 0
            assert origin.progressive_backward_flops == origin.full_backward_flops

        # Deeper freeze only ever removes backward work: non-decreasing per series.
        by_series: dict[tuple[str, int], list[float]] = {}
        for p in points:
            by_series.setdefault((p.policy, p.level), []).append(p.reduction_rate)
        for rates in by_series.values():
            assert rates == sorted(rates)


# ---------------------------------------------------------------------------
# 2. Gradient-flow partition across a partially-frozen multi-layer state
#    (the highest-risk regression: the suffix cut must hold layer-by-layer)
# ---------------------------------------------------------------------------


class TestGradientFlowAcrossPartialFreeze:
    def test_local_loss_drives_front_and_cuts_frozen_suffix(self):
        # Freeze only the top two layers (5 then 4): a genuine partially-frozen
        # state with {0,1,2,3} still active.
        schedule = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=list(range(NUM_LAYERS)),
                num_epochs=NUM_LAYERS,
                max_depth=2,
                start_epoch=1,
                policy="output_first",
            )
        )
        assert schedule.frozen_at_epoch == {5: 1, 4: 2}

        model = _build_model()
        loader = _loader()
        controllers = _drive_schedule(model, schedule, loader)
        assert _frozen_layer_indices(model) == {4, 5}

        # Freeze flags are exactly where the schedule put them.
        for i in (4, 5):
            assert not model.layers[i].proj.lora_B.requires_grad
        for i in range(4):
            assert model.layers[i].proj.lora_B.requires_grad

        # Create a real learning gap: turn on the immediate front (layer 3) LoRA
        # path so the current input to layer 4 no longer matches its cached xin.
        with torch.no_grad():
            model.layers[3].proj.lora_B.add_(0.5)

        model.zero_grad(set_to_none=True)
        loss = controllers[4].compute_local_loss(model, loader[0], ActivationMatchingLoss())

        # A live, finite, gradient-bearing scalar from a full forward+backward.
        assert loss.requires_grad
        assert torch.isfinite(loss)
        assert loss.item() > 0
        loss.backward()

        # Suffix cut (Level 2): gradient reaches every still-active front layer's
        # adapter, and none of the frozen suffix. This is the partition the
        # per-module suites cannot see across two frozen layers.
        for i in range(4):
            grad = model.layers[i].proj.lora_B.grad
            assert grad is not None
            assert grad.abs().sum().item() > 0, f"front layer {i} got no gradient"
        for i in (4, 5):
            assert model.layers[i].proj.lora_B.grad is None, (
                f"frozen layer {i} leaked gradient through the suffix cut"
            )


# ---------------------------------------------------------------------------
# 3. The local loss is a genuine training signal through the real forward
# ---------------------------------------------------------------------------


class TestLocalLossTrainsFront:
    def test_loss_decreases_across_optimizer_steps(self):
        # Freeze just the top layer (5); its front (4..0) is driven toward xin.
        schedule = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=list(range(NUM_LAYERS)),
                num_epochs=NUM_LAYERS,
                max_depth=1,
                start_epoch=1,
                policy="output_first",
            )
        )
        model = _build_model()
        loader = _loader()
        controllers = _drive_schedule(model, schedule, loader)
        ctrl = controllers[5]

        # Push the front off the cached xin so the loss starts non-zero.
        with torch.no_grad():
            model.layers[4].proj.lora_B.add_(0.5)

        trainable_lora = [
            p for n, p in model.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        ]
        opt = torch.optim.SGD(trainable_lora, lr=0.3)

        losses = []
        for _ in range(8):
            model.zero_grad(set_to_none=True)
            loss = ctrl.compute_local_loss(model, loader[0], ActivationMatchingLoss())
            losses.append(loss.item())
            loss.backward()
            opt.step()

        # design §3.3: the front-vs-xin gap is a real signal — descent shrinks it.
        assert losses[0] > 0
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# 4. One controller drives the whole progressive schedule (design §4.1)
#    — the N-controller _drive_schedule workaround above is no longer needed.
# ---------------------------------------------------------------------------


class TestSingleControllerProgressiveFreeze:
    """The controller's native progressive path (Phase 2, design §4.1).

    ``_drive_schedule`` above admits it works around "the controller is a
    single-freeze gate" by spinning up one controller per layer. With the
    schedule-driven progressive API a *single* controller now drives the entire
    multi-epoch, multi-layer plan: the frozen set grows exactly as the schedule
    predicts, and the local-loss backward still partitions gradient across the
    final cumulative suffix.
    """

    def test_one_controller_drives_full_schedule_cumulatively(self):
        # output_first, max_depth=5 over 6 layers: {5:1, 4:2, 3:3, 2:4, 1:5}.
        schedule = FreezeSchedule.plan(
            FreezeScheduleConfig(
                active_layer_indices=list(range(NUM_LAYERS)),
                num_epochs=NUM_LAYERS,
                max_depth=NUM_LAYERS - 1,
                start_epoch=1,
                spacing=1,
                policy="output_first",
            )
        )
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            active_layer_indices=set(range(NUM_LAYERS)),
            schedule=schedule,
        )

        for epoch in range(NUM_LAYERS):
            ctrl.progress(model, epoch, loader, "cpu")
            expected = {l for l, e in schedule.frozen_at_epoch.items() if e <= epoch}
            # One controller, one growing frozen set — exactly the schedule.
            assert ctrl.frozen_layers == expected
            assert _frozen_layer_indices(model) == expected

        assert ctrl.frozen_layers == {1, 2, 3, 4, 5}
        # Layer 0 is the deepest trainable front; local loss on layer 1's xin
        # backprops into it and into nothing of the 5-layer frozen suffix.
        with torch.no_grad():
            model.layers[0].proj.lora_B.add_(0.5)
        model.zero_grad(set_to_none=True)
        loss = ctrl.compute_local_loss(
            model, loader[0], ActivationMatchingLoss(), layer_idx=1
        )
        loss.backward()
        assert model.layers[0].proj.lora_B.grad is not None
        assert model.layers[0].proj.lora_B.grad.abs().sum().item() > 0
        for i in range(1, NUM_LAYERS):
            assert model.layers[i].proj.lora_B.grad is None
