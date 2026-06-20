"""Progressive multi-layer freezing (design §4.1) — single controller, many layers.

``test_progressive_freeze`` pins the single-shot gate (one frozen layer);
``test_progressive_freeze_e2e`` pins the schedule→controller→frontier→loss
*composition* but drives each layer with its **own** controller — its
``_drive_schedule`` helper openly works around the fact that "the controller is
a single-freeze gate". That workaround is exactly the gap design §4.1 names:
"Progressive" Freezing means one run freezes ``X`` at ``T1``, ``X-1`` at
``T2``, … with the frozen set only growing.

This suite pins the controller's native progressive capability:

* one controller + one :class:`FreezeSchedule` drives the whole multi-epoch,
  multi-layer plan (``layers_due_at`` / ``progress``);
* the frozen set grows cumulatively and never unfreezes;
* :meth:`apply_freeze_layer` is idempotent per layer but accepts distinct
  layers across calls;
* each frozen layer keeps its own ``xin`` cache, and ``compute_local_loss``'s
  ``layer_idx`` selects which frozen layer's ``xin`` is the target;
* across a cumulative frozen suffix, the local-loss backward still reaches the
  trainable front and leaks nothing into any frozen layer — and that signal
  genuinely descends under optimizer steps.

The fixture is the same faithful LoRA-in-forward transformer as the E2E suite
(base frozen, adapter on the graph) so ``requires_grad=False`` genuinely gates
gradient flow — the relationship the assertions inspect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model.lora_utils import iter_all_lora_params_by_layer, set_trainable_lora_layers
from src.tg_lora.activation_matching import ActivationMatchingLoss
from src.tg_lora.freeze_schedule import FreezeSchedule, FreezeScheduleConfig
from src.tg_lora.progressive_freeze import ProgressiveFreezeController

NUM_LAYERS = 6
HIDDEN = 8
VOCAB = 16


class _LoRALinear(nn.Module):
    """Base (frozen) + LoRA (trainable) linear, both on the forward graph."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, hidden, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.lora_A.t() @ self.lora_B.t()


class _LoRADecoderLayer(nn.Module):
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
    def __init__(
        self, num_layers: int = NUM_LAYERS, hidden: int = HIDDEN, vocab: int = VOCAB
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_LoRADecoderLayer(hidden) for _ in range(num_layers)])
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
    """Layers whose LoRA params are all frozen — only apply_freeze* sets that."""
    frozen: set[int] = set()
    for idx, params in iter_all_lora_params_by_layer(model).items():
        if all(not p.requires_grad for _, p in params):
            frozen.add(idx)
    return frozen


def _full_schedule(max_depth: int = NUM_LAYERS - 1) -> FreezeSchedule:
    """output_first over all 6 layers: {5:1, 4:2, 3:3, 2:4, 1:5} for max_depth=5."""
    return FreezeSchedule.plan(
        FreezeScheduleConfig(
            active_layer_indices=list(range(NUM_LAYERS)),
            num_epochs=NUM_LAYERS,
            max_depth=max_depth,
            start_epoch=1,
            spacing=1,
            policy="output_first",
        )
    )


def _layer_input(model: nn.Module, layer_idx: int, batch: dict) -> torch.Tensor:
    """No-grad capture of a layer's input via a one-shot pre-hook (reference)."""
    target = model.layers[layer_idx]
    captured: list[torch.Tensor] = []

    def _hook(module, args, kwargs):
        del module
        if args:
            captured.append(args[0])
        elif "hidden_states" in kwargs:
            captured.append(kwargs["hidden_states"])

    hook = target.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        hook.remove()
    return captured[0].detach()


# ---------------------------------------------------------------------------
# 1. Schedule-driven progressive freeze (one controller, the whole plan)
# ---------------------------------------------------------------------------


class TestProgressiveScheduleDrivenFreeze:
    def test_progress_freezes_layers_due_each_epoch_cumulatively(self):
        schedule = _full_schedule()  # {5:1, 4:2, 3:3, 2:4, 1:5}
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            active_layer_indices=set(range(NUM_LAYERS)),
            schedule=schedule,
        )

        seen: list[frozenset[int]] = []
        for epoch in range(NUM_LAYERS):
            ctrl.progress(model, epoch, loader, "cpu")
            expected = frozenset(
                l for l, e in schedule.frozen_at_epoch.items() if e <= epoch
            )
            # The controller's cumulative set and the model's actually-frozen
            # set both match the schedule prediction, at every epoch.
            assert ctrl.frozen_layers == expected
            assert _frozen_layer_indices(model) == expected
            seen.append(ctrl.frozen_layers)

        # Frozen set only ever grows (design §4.1) — never unfreezes a layer.
        assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))
        # Deepest front (layer 0) stays trainable; everything above froze.
        assert ctrl.frozen_layers == frozenset({1, 2, 3, 4, 5})
        assert 0 not in ctrl.frozen_layers

    def test_layers_due_at_returns_schedule_layers_for_epoch(self):
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            active_layer_indices=set(range(NUM_LAYERS)),
            schedule=_full_schedule(),
        )
        assert ctrl.layers_due_at(0) == []
        assert ctrl.layers_due_at(1) == [5]
        assert ctrl.layers_due_at(2) == [4]
        assert ctrl.layers_due_at(3) == [3]

    def test_layers_due_at_requires_schedule(self):
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices={0, 1}
        )
        with pytest.raises(RuntimeError, match="schedule"):
            ctrl.layers_due_at(0)

    def test_progress_returns_one_freeze_result_per_due_layer(self):
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            active_layer_indices=set(range(NUM_LAYERS)),
            schedule=_full_schedule(),
        )
        results = ctrl.progress(model, 1, loader, "cpu")
        assert [r.frozen_layer_idx for r in results] == [5]
        assert results[0].num_frozen_params > 0
        assert ctrl.frozen_layer_idx == 5

    def test_progress_with_no_layers_due_returns_empty(self):
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            active_layer_indices=set(range(NUM_LAYERS)),
            schedule=_full_schedule(),
        )
        assert ctrl.progress(model, 0, loader, "cpu") == []
        assert ctrl.frozen_layers == frozenset()


# ---------------------------------------------------------------------------
# 2. Per-layer idempotency (design §4.1: the frozen set only grows)
# ---------------------------------------------------------------------------


class TestApplyFreezeLayerIdempotency:
    def test_refreezing_same_layer_raises(self):
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices=set(range(NUM_LAYERS))
        )
        ctrl.apply_freeze_layer(model, 5, loader, "cpu")
        with pytest.raises(RuntimeError, match="already frozen"):
            ctrl.apply_freeze_layer(model, 5, loader, "cpu")

    def test_freezing_distinct_layers_succeeds(self):
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices=set(range(NUM_LAYERS))
        )
        ctrl.apply_freeze_layer(model, 5, loader, "cpu")
        ctrl.apply_freeze_layer(model, 4, loader, "cpu")  # distinct layer: no raise
        assert ctrl.frozen_layers == frozenset({4, 5})
        assert ctrl.frozen_layer_idx == 4  # most recent

    def test_single_shot_apply_freeze_guard_intact(self):
        # Cross-check: the Phase 1 single-shot path still rejects a double freeze.
        model = _build_model()
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices={4, 5}
        )
        ctrl.apply_freeze(model)  # freezes last_active = 5
        with pytest.raises(RuntimeError, match="already frozen"):
            ctrl.apply_freeze(model)


# ---------------------------------------------------------------------------
# 3. Per-layer xin caches and layer_idx target selection
# ---------------------------------------------------------------------------


class TestPerLayerXinCaches:
    def test_each_frozen_layer_has_own_correct_xin(self):
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices=set(range(NUM_LAYERS))
        )
        ctrl.apply_freeze_layer(model, 5, loader, "cpu")
        ctrl.apply_freeze_layer(model, 4, loader, "cpu")

        assert set(ctrl._xin_caches) == {4, 5}
        # Each cache holds its OWN layer's input (not the other's).
        with torch.no_grad():
            h = model.embed_tokens(loader[0]["input_ids"])
            ref: dict[int, torch.Tensor] = {}
            for i, layer in enumerate(model.layers):
                ref[i] = h.clone()
                h = layer(h)[0]
        assert torch.equal(ctrl._xin_caches[5][0][0], ref[5])
        assert torch.equal(ctrl._xin_caches[4][0][0], ref[4])
        # Different layers, different activations — the two caches are distinct.
        assert not torch.equal(ctrl._xin_caches[5][0][0], ctrl._xin_caches[4][0][0])

    def test_compute_local_loss_layer_idx_pairs_correct_xin(self):
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices=set(range(NUM_LAYERS))
        )
        ctrl.apply_freeze_layer(model, 5, loader, "cpu")
        ctrl.apply_freeze_layer(model, 4, loader, "cpu")
        # Push the front off both xins so the loss is non-zero and selective.
        with torch.no_grad():
            model.layers[3].proj.lora_B.add_(0.5)

        loss4 = ctrl.compute_local_loss(
            model, loader[0], ActivationMatchingLoss(), layer_idx=4
        )
        # layer_idx=4 must pair layer 4's current input against xin(4) — verified
        # by hand: it equals MSE(front4, xin4) and differs from MSE(front4, xin5).
        front4 = _layer_input(model, 4, loader[0])
        xin4 = ctrl._xin_caches[4][0][0]
        xin5 = ctrl._xin_caches[5][0][0]
        expected = ((front4 - xin4) ** 2).mean()
        wrong = ((front4 - xin5) ** 2).mean()
        assert torch.isclose(loss4.detach(), expected, atol=1e-6)
        assert not torch.isclose(loss4.detach(), wrong, atol=1e-5)

    def test_compute_local_loss_layer_idx_requires_frozen(self):
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices=set(range(NUM_LAYERS))
        )
        ctrl.apply_freeze_layer(model, 5, loader, "cpu")
        # Layer 4 is not frozen yet — selecting it must fail loudly.
        with pytest.raises(RuntimeError, match="not frozen"):
            ctrl.compute_local_loss(
                model, loader[0], ActivationMatchingLoss(), layer_idx=4
            )

    def test_compute_local_loss_layer_idx_and_single_shot_coexist(self):
        # A controller that froze via apply_freeze_layer can still be queried
        # for a single-shot-style local loss on the most-recent frozen layer.
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1, active_layer_indices=set(range(NUM_LAYERS))
        )
        ctrl.apply_freeze_layer(model, 5, loader, "cpu")
        # layer_idx given (progressive path) works ...
        loss_a = ctrl.compute_local_loss(
            model, loader[0], ActivationMatchingLoss(), layer_idx=5
        )
        assert torch.isfinite(loss_a) and loss_a.requires_grad


# ---------------------------------------------------------------------------
# 4. Gradient partition across a cumulative frozen suffix (highest-risk)
# ---------------------------------------------------------------------------


class TestGradientFlowAcrossCumulativeFreeze:
    def test_local_loss_cuts_cumulative_frozen_suffix(self):
        # Schedule freezes layers 5 then 4 — a genuine 2-layer frozen suffix
        # with {0,1,2,3} still active. Driven by ONE controller across epochs.
        schedule = _full_schedule(max_depth=2)  # {5:1, 4:2}
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            active_layer_indices=set(range(NUM_LAYERS)),
            schedule=schedule,
        )
        ctrl.progress(model, 1, loader, "cpu")  # freeze 5
        ctrl.progress(model, 2, loader, "cpu")  # freeze 4
        assert ctrl.frozen_layers == frozenset({4, 5})

        # Create a learning gap at the immediate front of layer 4 (layer 3).
        with torch.no_grad():
            model.layers[3].proj.lora_B.add_(0.5)

        model.zero_grad(set_to_none=True)
        loss = ctrl.compute_local_loss(
            model, loader[0], ActivationMatchingLoss(), layer_idx=4
        )
        assert loss.requires_grad and torch.isfinite(loss) and loss.item() > 0
        loss.backward()

        # Gradient reaches every still-active front layer, and NONE of the
        # cumulative frozen suffix {4,5}. This is the partition a single-shot
        # controller cannot exercise across two frozen layers.
        for i in range(4):
            grad = model.layers[i].proj.lora_B.grad
            assert grad is not None
            assert grad.abs().sum().item() > 0, f"front layer {i} got no gradient"
        for i in (4, 5):
            assert model.layers[i].proj.lora_B.grad is None, (
                f"frozen layer {i} leaked gradient"
            )

    @pytest.mark.parametrize("layer_idx", [4, 5])
    def test_each_front_descends_under_optimizer(self, layer_idx):
        # The local loss against each frozen layer's xin is a genuine signal:
        # optimizer steps on the trainable front shrink it. layer_idx=5 exercises
        # activation-gradient transit through the frozen middle layer 4.
        schedule = _full_schedule(max_depth=2)  # {5:1, 4:2}
        model = _build_model()
        loader = _loader()
        ctrl = ProgressiveFreezeController(
            start_cycle=1,
            active_layer_indices=set(range(NUM_LAYERS)),
            schedule=schedule,
        )
        ctrl.progress(model, 1, loader, "cpu")
        ctrl.progress(model, 2, loader, "cpu")

        with torch.no_grad():
            for i in range(4):
                model.layers[i].proj.lora_B.add_(0.3)

        trainable = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        ]
        opt = torch.optim.SGD(trainable, lr=0.3)

        first = last = None
        for _ in range(8):
            model.zero_grad(set_to_none=True)
            loss = ctrl.compute_local_loss(
                model, loader[0], ActivationMatchingLoss(), layer_idx=layer_idx
            )
            if first is None:
                first = loss.item()
            last = loss.item()
            loss.backward()
            opt.step()

        assert first > 0
        assert last < first
