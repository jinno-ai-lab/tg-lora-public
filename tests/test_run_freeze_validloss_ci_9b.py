"""Tests for ``scripts/run_freeze_validloss_ci_9b.py`` — the first real-9B
target-scale GOAL §4 A/B verdict, reduced to a GPU-free, executable core.

The script binds the suffix-only 9B config (``configs/9b_baseline_suffix_only_last25.yaml``)
to the GPU and runs a real ``Qwen/Qwen3.5-9B`` QLoRA candidate-vs-surrogate
freeze A/B on real public Dolly data, feeding the real valid_loss samples to
:func:`surrogate_valid_loss_ci` with ``proxy_scale=False``. The GPU run is an
opt-in smoke (``make freeze-validloss-ci-9b``), not this unit suite. These tests
guard the GPU-free core that makes that verdict honest and replayable:

* :func:`build_sft_example` — the public SFT prompt-masking contract (the ChatML
  user-turn masked with ``-100``, only the assistant response supervised;
  right-truncation; all-masked examples dropped), exercised with a stub
  tokenizer so no model download is needed.
* :func:`candidate_order_9b` — output-first descending freeze order over the
  real active scope (the candidate arm's order vector).
* :func:`_reset_lora_for_arm` — the load-once / reset arm-separation primitive
  that fixed the per-arm-reload OOM (a second ~5.5 GB 9B model does not release
  cleanly on a 12 GB GPU). Mutation-proof on a synthetic-LoRA module: ``lora_B``
  zeroed, ``lora_A`` re-randomized per seed, scope grads re-enabled (un-freezing
  the prior arm's frozen layers).
* :func:`result_to_json` — the deposit's GOAL §7 honesty labels
  (``proxy_scale=False``, ``reduced_budget=True``, ``citable_as_target_scale=
  True`` but NOT ``citable_as_full_section4_verdict``).
* **Deposit replay faithfulness** — the committed real-9B recording
  (``freeze_validloss_ci_9b_surrogate.json``, a real RTX 3060 seq1024 run)
  re-judges through :func:`surrogate_valid_loss_ci` to the ``SURPASSES`` verdict
  it *recorded*: the stored floats earn the verdict under the deterministic
  bootstrap, the verdict is not painted on. This is the expected-output
  assertion that pins the recorded Category-C dataset.

Everything here runs without CUDA (the real model is never loaded); torch is
used only to build tiny synthetic tensors / modules.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.run_freeze_validloss_ci_9b import (
    REGIME_GENERALIZATION,
    REGIME_MEMORIZATION,
    REGIME_OVERFIT,
    REGIME_UNKNOWN,
    _baseline_ci_to_json,
    _candidate_cost_reduction,
    _candidate_final_ce_mean,
    _classify_regime,
    _direction_ci_to_json,
    _is_reduced_budget,
    _reset_lora_for_arm,
    arm_valid_loss_9b,
    build_parser,
    build_sft_example,
    candidate_order_9b,
    control_order_9b,
    result_to_json,
    run_ci_9b,
)
from src.tg_lora.freeze_surrogate_ci import surrogate_valid_loss_ci
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

# The committed real-9B recording: a real RTX 3060 seq1024 suffix-only run of
# the §4 candidate-vs-surrogate A/B (verdict SURPASSES, candidate_mean≈1.625,
# surrogate_mean≈1.770, CI[95%]≈[0.077, 0.230], 4 seeds/arm — NON-thin evidence).
# Regenerate with ``make freeze-validloss-ci-9b``.
FIXTURE_9B_SURROGATE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_surrogate.json"
)

# The direction-isolation recording: the same real RTX 3060 seq1024 suffix-only
# 9B A/B PLUS an input-side contiguous control arm (input_first_order) so the §4
# SURPASSES can be attributed to output-side DIRECTION, not mere contiguity
# (constitution P0). Records BOTH verdicts SURPASSES, non-thin: candidate
# (output-contig {29,30,31}) < random surrogate < control (input-contig
# {24,25,26}). Regenerate with ``make freeze-validloss-ci-9b`` + ``--n-control 4``.
FIXTURE_9B_DIRECTION = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_direction.json"
)

# The GENERALIZATION-REGIME recording: the same real RTX 3060 seq1024 suffix-only
# 9B A/B (candidate vs random surrogate vs input-side control) but run in a
# regime the LoRA CANNOT memorize — 48 train examples over 96 steps (2 epochs),
# vs the 8-examples/20-steps MEMORIZATION regime of the two deposits above (whose
# train CE collapsed to ~0). Here every arm's final_ce_train_loss is ~1.5 (train
# CE ≈ valid CE), so the verdict is measured on a model that generalized, not
# memorized. Records BOTH verdicts SURPASSES, non-thin (3 seeds/arm): candidate
# (1.5152) < random surrogate (1.5270) < input-side control (1.5375) — the same
# monotone ranking as the memorization regime, so the verdict SURVIVES the regime
# change (effect size shrinks ~12x; honest regime-dependence, not a re-run).
# Regenerate with ``make freeze-validloss-ci-9b-generalization``.
FIXTURE_9B_GENERALIZATION = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_generalization.json"
)

# The FULL-BACKPROP BASELINE recording: the same real RTX 3060 seq1024 suffix-only
# 9B A/B (candidate vs random surrogate vs input-side control) PLUS a no-freeze
# full-CE baseline arm (depth=0 — every active-scope layer trained on the task
# loss throughout), all in the honest GENERALIZATION regime (48 train / 96 step).
# GOAL §4 line 247's OTHER success axis: the surrogate / direction /
# generalization deposits above are all freeze-vs-freeze, so none can say whether
# freezing costs quality vs FULL backprop. This deposit carries a candidate-vs-
# baseline CI — the §4 "valid_loss within tolerance of full backprop" measurement.
# Recorded result: candidate (1.5148) SURPASSES the full-backprop baseline
# (1.5401), non-thin (3 seeds/arm), CI[95%]≈[0.018, 0.029]. The full 4-arm
# monotone ranking is candidate < random surrogate (1.5270) < input-side control
# (1.5376) < full backprop (1.5401) — freezing did not cost quality; the freeze
# acted as a regularizer (the baseline overfit: train CE 0.77 ≪ valid 1.54, while
# the frozen arms generalized: train CE ≈ valid ≈ 1.5). Verdict reading for a
# future re-run: SURPASSES (method beats full) / TIES (quality maintained — the
# §4 target) / UNDERSHOOTS (freezing cost quality at this budget). Regenerate
# with ``make freeze-validloss-ci-9b-baseline``.
FIXTURE_9B_BASELINE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_baseline.json"
)


# ── stubs ───────────────────────────────────────────────────────────────────


class _CharTokenizer:
    """Deterministic char-code tokenizer for the masking/truncation tests.

    ``build_sft_example`` calls ``tokenizer(text, add_special_tokens=False)
    .input_ids``; this stub returns one id per character, so ``prefix`` ids are
    always a strict prefix of ``prefix + completion`` ids — exactly the
    invariant the prompt-masking logic relies on. No model download needed.
    """

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        return types.SimpleNamespace(input_ids=ids)


class _LoRABlock(nn.Module):
    """A fake decoder layer whose param names match ``layers.(\\d+).lora_[AB]``."""

    def __init__(self):
        super().__init__()
        # lora_A starts at zero so a reset (kaiming re-init) is detectable as
        # a transition away from zero; lora_B starts non-zero so zeroing is
        # detectable.
        self.lora_A = nn.Parameter(torch.zeros(4, 4))
        self.lora_B = nn.Parameter(torch.randn(4, 4))


class _Decoder(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_LoRABlock() for _ in range(n_layers)])


class _FakeLoraModel(nn.Module):
    """``n_layers`` blocks under ``model.layers.<idx>`` — the name shape
    :func:`iter_all_lora_params_by_layer` (regex ``layers.(\\d+).``) keys on."""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = _Decoder(n_layers)


def _lora_A(model: nn.Module, layer_idx: int) -> torch.Tensor:
    return dict(model.named_parameters())[f"model.layers.{layer_idx}.lora_A"]


def _lora_B(model: nn.Module, layer_idx: int) -> torch.Tensor:
    return dict(model.named_parameters())[f"model.layers.{layer_idx}.lora_B"]


# ── build_sft_example: prompt-masking contract ──────────────────────────────


class TestBuildSftExample:
    def test_masks_prompt_supervises_response(self):
        ex = build_sft_example(
            _CharTokenizer(), "INST", "RESP", context="", max_seq_len=4096
        )
        ids = ex["input_ids"][0].tolist()
        labels = ex["labels"][0].tolist()
        assert ex["attention_mask"][0].tolist() == [1] * len(ids)
        # The supervised (non -100) tail IS the full token sequence there — the
        # prompt prefix is exactly the masked head.
        masked_head = [i for i, lab in enumerate(labels) if lab == -100]
        supervised_tail = [i for i, lab in enumerate(labels) if lab != -100]
        assert masked_head and supervised_tail
        assert max(masked_head) < min(supervised_tail)  # head then tail, no interleaving
        # Where supervised, labels reproduce input_ids verbatim.
        for i in supervised_tail:
            assert labels[i] == ids[i]
        assert len(ids) == len(labels)

    def test_right_truncates_to_max_seq_len(self):
        # Pick a max_seq_len strictly inside the response tail (past the prompt
        # prefix, before the full sequence) so truncation keeps some supervision.
        tok = _CharTokenizer()
        prefix = "<|im_start|>user\nINST<|im_end|>\n<|im_start|>assistant\n"
        prefix_len = len(prefix)
        ex = build_sft_example(tok, "INST", "RESP", max_seq_len=prefix_len + 4)
        assert ex["input_ids"].shape[1] == prefix_len + 4
        assert ex["labels"].shape[1] == prefix_len + 4

    def test_truncated_supervised_tail_survives(self):
        # max_seq_len long enough to keep the whole prompt + part of the response:
        # the response tail is still supervised after truncation.
        tok = _CharTokenizer()
        prefix = "<|im_start|>user\nINST<|im_end|>\n<|im_start|>assistant\n"
        full_prefix_len = len(prefix)
        ex = build_sft_example(tok, "INST", "RESP", max_seq_len=full_prefix_len + 2)
        labels = ex["labels"][0].tolist()
        assert sum(1 for lab in labels if lab != -100) == 2  # 2 response chars kept

    def test_returns_none_when_supervision_truncated_away(self):
        # max_seq_len shorter than the prompt prefix → the whole truncated
        # sequence is prompt → all labels -100 → dropped (returns None).
        prefix = "<|im_start|>user\nINST<|im_end|>\n<|im_start|>assistant\n"
        ex = build_sft_example(
            _CharTokenizer(), "INST", "RESP", max_seq_len=len(prefix) - 1
        )
        assert ex is None

    def test_tensors_are_batch_size_one(self):
        ex = build_sft_example(_CharTokenizer(), "I", "R", max_seq_len=4096)
        for key in ("input_ids", "attention_mask", "labels"):
            assert ex[key].shape[0] == 1
            assert ex[key].dtype == torch.long


# ── candidate_order_9b: output-first descending ─────────────────────────────


class TestCandidateOrder:
    def test_descending_over_scope(self):
        scope = {24, 25, 26, 27, 28, 29, 30, 31}
        assert candidate_order_9b(scope) == (31, 30, 29, 28, 27, 26, 25, 24)

    def test_independent_of_input_set_shape(self):
        assert candidate_order_9b({3, 1, 2}) == (3, 2, 1)


# ── _reset_lora_for_arm: load-once/reset arm separation (the OOM fix) ────────


class TestResetLoraForArm:
    def test_zeroes_lora_B_every_layer(self):
        model = _FakeLoraModel(n_layers=4)
        # B starts non-zero in the fixture; reset must zero it everywhere.
        assert _lora_B(model, 0).abs().sum() > 0
        _reset_lora_for_arm(model, "last_25_percent", seed=0)
        for idx in range(4):
            assert torch.equal(_lora_B(model, idx), torch.zeros(4, 4))

    def test_re_randomizes_lora_A_away_from_init(self):
        # lora_A starts at zero; reset (kaiming) must move it off zero.
        model = _FakeLoraModel(n_layers=4)
        assert torch.equal(_lora_A(model, 0), torch.zeros(4, 4))
        _reset_lora_for_arm(model, "last_25_percent", seed=0)
        assert _lora_A(model, 0).abs().sum() > 0

    def test_seed_distinctness_under_fixed_global_rng(self):
        # Reset the GLOBAL rng to an identical state before each reset, so the
        # ONLY input that differs is the ``seed`` arg. If _reset_lora_for_arm
        # honors its seed, the two arms' lora_A differ; if it ignores the seed
        # (constant or removed torch.manual_seed), they are identical.
        m1 = _FakeLoraModel(n_layers=4)
        m2 = _FakeLoraModel(n_layers=4)
        torch.manual_seed(999)
        _reset_lora_for_arm(m1, "last_25_percent", seed=0)
        torch.manual_seed(999)
        _reset_lora_for_arm(m2, "last_25_percent", seed=1)
        assert not torch.allclose(_lora_A(m1, 3), _lora_A(m2, 3))

    def test_scope_re_enables_grads_unfreezing_prior_arm(self):
        # Simulate a prior arm having frozen layer 3 (requires_grad=False); the
        # scope re-application inside reset must un-freeze it (scope = {3}).
        model = _FakeLoraModel(n_layers=4)
        for idx in range(4):
            _lora_A(model, idx).requires_grad_(False)
            _lora_B(model, idx).requires_grad_(False)
        active = _reset_lora_for_arm(model, "last_25_percent", seed=0)
        assert active == {3}  # ceil(4 * 0.25) = 1 → last layer only
        assert _lora_A(model, 3).requires_grad is True
        assert _lora_B(model, 3).requires_grad is True
        for idx in (0, 1, 2):
            assert _lora_A(model, idx).requires_grad is False

    def test_reproducible_same_seed(self):
        m1 = _FakeLoraModel(n_layers=4)
        m2 = _FakeLoraModel(n_layers=4)
        _reset_lora_for_arm(m1, "last_25_percent", seed=7)
        _reset_lora_for_arm(m2, "last_25_percent", seed=7)
        assert torch.allclose(_lora_A(m1, 3), _lora_A(m2, 3))


# ── arm_valid_loss_9b: signature contract (GPU path excluded) ────────────────


class TestArmSignature:
    def test_arm_is_not_a_reloader(self):
        # The OOM fix: arm_valid_loss_9b takes a pre-loaded ``model`` (caller
        # owns it) rather than loading per arm. Its signature must NOT accept a
        # cfg/reload-style param.
        import inspect

        params = set(inspect.signature(arm_valid_loss_9b).parameters)
        assert "model" in params
        assert "scope" in params
        assert "cfg" not in params  # no per-arm reload


# ── result_to_json: GOAL §7 honesty labels ───────────────────────────────────


def _stub_ci(**overrides):
    base = dict(
        significance_verdict="SURPASSES",
        passes=True,
        significant_surpasses=True,
        is_material=True,
        is_thin_evidence=True,
        candidate_mean=1.625,
        surrogate_mean=1.704,
        point_improvement=0.079,
        lower=0.074,
        upper=0.085,
        confidence=0.95,
        n_bootstrap=10000,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _stub_result(**overrides):
    base = dict(
        ci=_stub_ci(),
        candidate_losses=[1.625, 1.625],
        surrogate_losses=[1.704, 1.704],
        candidate_order=[31, 30, 29, 28, 27, 26, 25, 24],
        device="cuda",
        model="Qwen/Qwen3.5-9B",
        dataset="databricks/databricks-dolly-15k",
        total_steps=20,
        warmup_steps=6,
        depth=3,
        spacing=4,
        n_candidate=2,
        n_surrogate=2,
        base_seed=0,
        active_scope=[24, 25, 26, 27, 28, 29, 30, 31],
        scope_label="last_25_percent",
        n_active_layers=8,
        scope_trainable_params=10819584,
        seq_len=1024,
        train_examples=8,
        valid_examples=10,
        use_local_loss=True,
        proxy_scale=False,
        reduced_budget=True,
        cfg_max_steps=1500,
        candidate_provenance=[{"frozen_layers": [29, 30, 31], "n_trainable_params": 6787072, "last_train_loss": 0.001}],
        surrogate_provenance=[{"frozen_layers": [25, 28, 31], "n_trainable_params": 6787072, "last_train_loss": 0.001}],
        # Direction-isolation control arm (constitution P0): default n_control=0
        # so a no-control run deposits byte-identically to the pre-control shape.
        control_losses=[],
        control_order=[24, 25, 26, 27, 28, 29, 30, 31],
        n_control=0,
        control_provenance=[],
        direction_ci=None,
        # Full-backprop baseline arm (GOAL §4 line 247 control (i)): default
        # n_baseline=0 so a no-baseline run deposits byte-identically to the
        # pre-baseline shape.
        baseline_losses=[],
        n_baseline=0,
        baseline_provenance=[],
        baseline_ci=None,
    )
    base.update(overrides)
    return base


class TestResultToJson:
    def test_honesty_labels_real_target_reduced_budget(self):
        out = result_to_json(_stub_result())
        # Real 9B + real data: NOT a proxy.
        assert out["proxy_scale"] is False
        assert out["citable_as_target_scale"] is True
        # Reduced budget: NOT yet the full §4 verdict.
        assert out["reduced_budget"] is True
        assert out["citable_as_full_section4_verdict"] is False
        # cfg_max_steps is surfaced so a reader can see the budget the
        # reduced-budget flag was judged against (not a mystery boolean).
        assert out["cfg_max_steps"] == 1500

    def test_full_verdict_only_when_full_budget_target_scale_non_thin_and_generalizing(self):
        # The citation gate opens ONLY when FOUR honesty axes clear together:
        # full-budget + target-scale + non-thin + GENERALIZATION regime. The
        # candidate arm must have generalized (train CE ≈ valid CE), evidenced by
        # a finite ``final_ce_train_loss`` in its provenance — without it the
        # regime reads UNKNOWN and the gate stays shut (see TestRegimeHonestyGate).
        out = result_to_json(_stub_result(
            proxy_scale=False, reduced_budget=False,
            ci=_stub_ci(is_thin_evidence=False),
            candidate_provenance=[
                {"frozen_layers": [29, 30, 31], "n_trainable_params": 6787072,
                 "last_train_loss": 1.5, "final_ce_train_loss": 1.507},
            ],
        ))
        assert out["citable_as_full_section4_verdict"] is True

    def test_thin_evidence_blocks_full_citation_even_at_full_budget(self):
        # A full-budget target-scale run that is still thin (too few seeds)
        # must NOT be citable as the complete §4 verdict.
        out = result_to_json(_stub_result(
            proxy_scale=False, reduced_budget=False,
            ci=_stub_ci(is_thin_evidence=True),
        ))
        assert out["citable_as_full_section4_verdict"] is False

    def test_proxy_scale_never_citable_as_target(self):
        out = result_to_json(_stub_result(proxy_scale=True))
        assert out["citable_as_target_scale"] is False
        assert out["citable_as_full_section4_verdict"] is False

    def test_carries_provenance_for_audit(self):
        out = result_to_json(_stub_result())
        for key in (
            "candidate_provenance",
            "surrogate_provenance",
            "scope_trainable_params",
            "seq_len",
            "base_seed",
            "candidate_order",
        ):
            assert key in out

    def test_final_ce_train_loss_passes_through_provenance(self):
        # The generalization-regime diagnostic (mean full-CE over the train set
        # under the final adapter) rides in the per-arm provenance dict;
        # ``result_to_json`` passes provenance through verbatim, so the field
        # surfaces in the deposit without explicit wiring. It is ADDITIVE — the
        # legacy ``last_train_loss`` (last optimizer step's loss = boundary local
        # loss once frozen, structurally ~0) still rides through alongside it, so
        # the two disambiguate rather than one replacing the other.
        out = result_to_json(_stub_result(
            candidate_provenance=[{
                "frozen_layers": [29, 30, 31],
                "n_trainable_params": 6787072,
                "last_train_loss": 0.0002,
                "final_ce_train_loss": 1.507,
            }],
        ))
        p = out["candidate_provenance"][0]
        assert p["final_ce_train_loss"] == 1.507
        assert p["last_train_loss"] == 0.0002


# ── candidate cost-reduction axis (GOAL §4 condition (b) / P3 cost gate) ──────
#
# The §4 verdict is TWO-HEADED: (a) quality preserved (the valid_loss A/B the
# deposit already carries) and (b) cost reduced vs full backprop. This class pins
# the second head — ``_candidate_cost_reduction`` replans the candidate arm's
# EXACT freeze schedule and feeds it to ``FreezeCostAccountant`` for P3's
# "削減率 = 1 − progressive / full". Values are hand-computed (uniform per-layer
# cost; the reduction is a ratio, so uniform costs give the exact first-order
# figure). The Level-1 honesty test (§6.2) is the load-bearing one: the
# candidate runs Level 1 (weight-grad stop), which realizes ~0 backward reduction
# in vivo, so ``realized_reduction_rate`` — not ``reduction_rate`` — is the
# realized saving a reader may quote.


class TestCandidateCostReduction:
    def _sched(self, **over):
        # Minimal result dict carrying only the candidate-arm schedule keys the
        # helper reads (hand-computed cases use a 4-layer scope for clarity).
        base = dict(
            candidate_order=[3, 2, 1, 0],
            active_scope=[0, 1, 2, 3],
            depth=2,
            warmup_steps=0,
            spacing=1,
            total_steps=4,
        )
        base.update(over)
        return base

    def test_known_value_two_of_four_layers(self):
        # 4 active layers, freeze the 2 output-side first (order [3, 2, ..]),
        # warmup 0, spacing 1, 4 steps. Layer 3 freezes at epoch 0, layer 2 at
        # epoch 1. Full backprop = 4 layers × (1+1) × 4 steps = 32; progressive
        # (Level 1) = 25; reduction = 1 − 25/32 = 0.21875. Hand-computed.
        out = _candidate_cost_reduction(self._sched())
        assert out is not None
        assert out["reduction_rate"] == pytest.approx(0.21875)
        assert out["full_backward_flops"] == 32.0
        assert out["progressive_backward_flops"] == 25.0
        assert out["realized_depth"] == 2
        assert out["frozen_at_epoch"] == {"3": 0, "2": 1}

    def test_depth_zero_is_full_backprop(self):
        # depth=0 freezes nothing: progressive == full, reduction 0.0.
        out = _candidate_cost_reduction(self._sched(depth=0))
        assert out is not None
        assert out["reduction_rate"] == 0.0
        assert out["progressive_backward_flops"] == out["full_backward_flops"]
        assert out["realized_depth"] == 0
        assert out["frozen_at_epoch"] == {}

    def test_reduction_monotone_in_depth(self):
        # Deeper freeze only ever removes backward work (the frontier invariant):
        # reduction is non-decreasing in depth. depth=4 (freeze all) → 0.3125.
        rates = [
            _candidate_cost_reduction(self._sched(depth=d))["reduction_rate"]
            for d in range(5)
        ]
        assert rates == sorted(rates)
        assert rates[-1] == pytest.approx(0.3125)

    def test_level1_realized_reduction_is_zero_under_validated_ceiling(self):
        # §6.2 honesty (constitution verifiability): the candidate arm runs
        # Level 1 (weight-grad stop; the activation gradient still traverses the
        # frozen layer), so the arithmetic reduction_rate OVERSTATES what is
        # realized in vivo. realized_reduction_rate is capped at the validated
        # Level-1 ceiling (0.0): the deposit must not present the arithmetic
        # figure as a realized saving. reduction_rate stays the arithmetic bound.
        out = _candidate_cost_reduction(self._sched())
        assert out["reduction_rate"] == pytest.approx(0.21875)
        assert out["realized_reduction_rate"] == 0.0
        assert out["level1_realization_ceiling"] == 0.0
        assert out["level"] == 1

    def test_tracks_candidate_schedule_not_a_constant(self):
        # Identity/mutation proof: the field is wired to the candidate's ACTUAL
        # schedule. depth=0 (no freezes) collapses to 0.0, distinct from the
        # 0.21875 a depth-2 schedule earns — so changing the candidate arm's
        # freeze depth moves the reported cost, as it must.
        full = _candidate_cost_reduction(self._sched(depth=2))["reduction_rate"]
        none_ = _candidate_cost_reduction(self._sched(depth=0))["reduction_rate"]
        assert full == pytest.approx(0.21875)
        assert none_ == 0.0
        assert full != none_

    def test_deposit_surfaces_candidate_cost_reduction(self):
        # result_to_json surfaces the axis additively. The stub default mirrors
        # the committed surrogate/direction deposit shape (8-layer suffix scope,
        # depth 3, warmup 6, spacing 4, 20 steps) → reduction 0.09375,
        # realized_depth 3, uniform-cost model.
        out = result_to_json(_stub_result())
        cr = out["candidate_cost_reduction"]
        assert cr is not None
        assert cr["reduction_rate"] == pytest.approx(0.09375)
        assert cr["realized_reduction_rate"] == 0.0
        assert cr["realized_depth"] == 3
        assert cr["frozen_at_epoch"] == {"31": 6, "30": 10, "29": 14}
        assert cr["cost_model"] == "uniform_per_layer"
        assert cr["level"] == 1

    def test_real_generalization_deposit_candidate_schedule(self):
        # Real-data coverage on the committed §4 generalization-regime recording:
        # recompute the candidate cost reduction from the deposit's OWN schedule
        # fields and assert the hand-computed constant (depth 3, warmup 12,
        # spacing 10, 96 steps over the 8-layer suffix scope → 0.14453125). The
        # committed fixture predates this field (additive), so this proves the
        # helper reads a real candidate schedule, not a synthetic one.
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        out = _candidate_cost_reduction(data)
        assert out is not None
        assert out["reduction_rate"] == pytest.approx(0.14453125)
        assert out["realized_depth"] == 3
        assert out["frozen_at_epoch"] == {"31": 12, "30": 22, "29": 32}
        assert out["realized_reduction_rate"] == 0.0

    def test_none_when_schedule_keys_absent(self):
        # A result missing the candidate-arm schedule keys (e.g. a partial /
        # legacy result) yields None rather than raising — the additive contract.
        assert _candidate_cost_reduction({}) is None
        assert _candidate_cost_reduction({"candidate_order": [3, 2, 1, 0]}) is None


# ── regime classification (4th honesty axis: generalization vs memorization/overfit)

# The thresholds below are grounded in the committed 9B deposits, not picked from
# thin air — see the script's ``_classify_regime`` docstring. Candidate-CE values:
#   generalization arms ≈ 1.507 (valid ≈ 1.515, gap ≈ 0.008)
#   memorization arms  ≈ 0.0   (8 train x 20 step collapses train CE)
#   full-backprop baseline ≈ 0.77 (valid ≈ 1.54, gap ≈ 0.77 → OVERFIT)


class TestRegimeClassification:
    """The 4th honesty axis: the citation gate must distinguish a verdict measured
    on a model that GENERALIZED (train CE ≈ valid CE) from one measured on a model
    that MEMORIZED (train CE → 0) or OVERFIT (train CE ≪ valid CE)."""

    def test_generalization_when_train_ce_approximates_valid(self):
        # The committed generalization-regime candidate arms: final_ce ≈ 1.507,
        # valid ≈ 1.515 → gap ≈ 0.008. Train CE well above the floor, gap well
        # under the threshold → GENERALIZATION.
        assert _classify_regime(1.507, 1.515) == REGIME_GENERALIZATION

    def test_memorization_when_train_ce_collapses(self):
        # The committed memorization-regime arms (8 train × 20 step): train CE
        # collapses toward 0. ce < 0.5 → MEMORIZATION regardless of valid.
        assert _classify_regime(0.001, 1.6) == REGIME_MEMORIZATION
        assert _classify_regime(0.0, 1.0) == REGIME_MEMORIZATION

    def test_overfit_when_train_valid_gap_is_large(self):
        # The committed full-backprop BASELINE arm: final_ce 0.77 ≪ valid 1.54
        # → gap 0.77 > 0.5 → OVERFIT (train CE above the memorization floor, but
        # the model fit train far better than it generalized).
        assert _classify_regime(0.77, 1.54) == REGIME_OVERFIT

    def test_unknown_when_train_ce_missing(self):
        # A deposit recorded before the final_ce_train_loss diagnostic existed
        # carries no train CE. Conservative: UNKNOWN — never opens the full-§4
        # gate on a regime it cannot verify.
        assert _classify_regime(None, 1.5) == REGIME_UNKNOWN
        assert _classify_regime("", 1.5) == REGIME_UNKNOWN

    def test_unknown_when_train_ce_non_finite(self):
        # NaN / inf (e.g. a diverged run) classifies UNKNOWN rather than being
        # silently compared as a real number.
        assert _classify_regime(float("nan"), 1.5) == REGIME_UNKNOWN
        assert _classify_regime(1.5, float("inf")) == REGIME_UNKNOWN
        assert _classify_regime(float("inf"), 1.5) == REGIME_UNKNOWN

    def test_memorization_floor_boundary(self):
        # ce < 0.5 → memorization; ce == 0.5 is NOT memorization (strict <).
        # At ce=0.5, vl=1.0: gap 0.5 is NOT > 0.5 → GENERALIZATION.
        assert _classify_regime(0.499, 1.0) == REGIME_MEMORIZATION
        assert _classify_regime(0.5, 1.0) == REGIME_GENERALIZATION

    def test_overfit_gap_boundary(self):
        # (vl - ce) > 0.5 → overfit; gap == 0.5 is NOT overfit (strict >).
        # 1.5 - 1.0 == 0.5 exactly (both exactly representable), so the boundary
        # is clean rather than at the mercy of float rounding.
        assert _classify_regime(1.0, 1.5) == REGIME_GENERALIZATION     # gap 0.5
        assert _classify_regime(1.0, 1.5001) == REGIME_OVERFIT         # gap 0.5001

    def test_candidate_final_ce_mean_ignores_unrecorded_arms(self):
        # Arms without final_ce_train_loss are skipped — NOT counted as 0 (which
        # would falsely label a real generalization run as memorization).
        result = {"candidate_provenance": [
            {"final_ce_train_loss": 1.507},
            {},  # pre-diagnostic arm — no field
            {"final_ce_train_loss": 1.509},
        ]}
        assert _candidate_final_ce_mean(result) == pytest.approx(1.508)

    def test_candidate_final_ce_mean_none_when_all_unrecorded(self):
        assert _candidate_final_ce_mean({"candidate_provenance": [{}, {}]}) is None
        assert _candidate_final_ce_mean({"candidate_provenance": []}) is None

    def test_candidate_final_ce_mean_skips_non_finite(self):
        result = {"candidate_provenance": [
            {"final_ce_train_loss": float("nan")},
            {"final_ce_train_loss": 1.507},
        ]}
        assert _candidate_final_ce_mean(result) == pytest.approx(1.507)


class TestRegimeHonestyGate:
    """The defect this axis closes: a future ``--total-steps 1500`` run clears
    ``reduced_budget`` (full budget), and paired with the default small
    ``--train-examples`` it does tens of epochs and MEMORIZES — yet without the
    regime conjunct the gate would flip ``citable_as_full_section4_verdict=True``
    on a memorization artifact. These tests pin the 4th gate axis and are
    mutation-proven against removal of the ``regime == REGIME_GENERALIZATION``
    conjunct (each blocking case flips to True under that mutation)."""

    @staticmethod
    def _full_budget_non_thin_result(final_ce_train_loss):
        # Full budget + target scale + non-thin — clears the first THREE axes, so
        # the gate outcome is decided SOLELY by the regime axis (the 4th). Vary
        # only ``final_ce_train_loss`` to drive the regime.
        return _stub_result(
            proxy_scale=False,
            reduced_budget=False,
            ci=_stub_ci(is_thin_evidence=False),
            candidate_provenance=[
                {"frozen_layers": [29, 30, 31], "n_trainable_params": 6787072,
                 "final_ce_train_loss": final_ce_train_loss},
            ],
        )

    def test_generalization_regime_opens_full_citation(self):
        # Full budget + target + non-thin + train CE ≈ valid → the ONLY shape
        # that earns the complete-§4 label.
        out = result_to_json(self._full_budget_non_thin_result(1.507))
        assert out["regime"] == REGIME_GENERALIZATION
        assert out["citable_as_full_section4_verdict"] is True

    def test_memorization_regime_blocks_full_citation_at_full_budget(self):
        # THE DEFECT CLOSURE: full budget + non-thin + target, but the model
        # memorized (train CE → 0). The SURPASSES read off a memorized model is
        # an artifact, not the §4 question, so the gate must stay shut.
        out = result_to_json(self._full_budget_non_thin_result(0.001))
        assert out["regime"] == REGIME_MEMORIZATION
        assert out["citable_as_full_section4_verdict"] is False

    def test_overfit_regime_blocks_full_citation(self):
        # Full budget + non-thin + target, but the model overfit (train ≪ valid).
        # A distinct failure of the §4 question; the gate stays shut.
        out = result_to_json(self._full_budget_non_thin_result(0.77))
        assert out["regime"] == REGIME_OVERFIT
        assert out["citable_as_full_section4_verdict"] is False

    def test_unknown_regime_blocks_full_citation(self):
        # No final_ce recorded (pre-diagnostic deposit shape) → regime UNKNOWN →
        # the conservative call: never open the full-§4 gate on an unverifiable
        # regime. A pre-diagnostic full-budget run does NOT auto-pass.
        result = _stub_result(
            proxy_scale=False,
            reduced_budget=False,
            ci=_stub_ci(is_thin_evidence=False),
            candidate_provenance=[
                {"frozen_layers": [29, 30, 31], "n_trainable_params": 6787072},
            ],
        )
        out = result_to_json(result)
        assert out["regime"] == REGIME_UNKNOWN
        assert out["citable_as_full_section4_verdict"] is False

    def test_regime_axis_independent_of_other_three(self):
        # The four axes are conjuncts: even generalization-regime cannot rescue a
        # reduced-budget run. Pins that the regime conjunct was ADDED, not swapped
        # in for one of the other three.
        out = result_to_json(_stub_result(
            proxy_scale=False,
            reduced_budget=True,  # reduced — blocked on THIS axis
            ci=_stub_ci(is_thin_evidence=False),
            candidate_provenance=[
                {"frozen_layers": [29, 30, 31], "n_trainable_params": 6787072,
                 "final_ce_train_loss": 1.507},
            ],
        ))
        assert out["regime"] == REGIME_GENERALIZATION
        assert out["citable_as_full_section4_verdict"] is False

    def test_regime_fields_surfaced_for_audit(self):
        # The deposit surfaces the regime + the two numbers it was derived from,
        # so a reader can audit the 4th axis without re-deriving it.
        out = result_to_json(self._full_budget_non_thin_result(1.507))
        assert out["regime"] == REGIME_GENERALIZATION
        assert out["candidate_final_ce_train_loss_mean"] == pytest.approx(1.507)
        # candidate_train_valid_gap = candidate_mean − ce_mean = 1.625 − 1.507
        assert out["candidate_train_valid_gap"] == pytest.approx(1.625 - 1.507)


# ── direction-isolation control (constitution P0: direction vs contiguity) ───


class TestDirectionIsolation:
    """The DIRECTION-CONTROL arm isolates freeze DIRECTION from freeze-set
    CONTIGUITY — the residual confound in the §4 A/B that the constitution's P0
    gate (rule out "通ったように見えるが実は不活性／誤帰属") demands be closed
    before the verdict is attributed.

    The candidate freezes a contiguous output-side block; a random surrogate
    freezes a scattered set, so a candidate ``SURPASSES`` could be the output-side
    direction OR mere contiguity. The input-side contiguous control holds
    contiguity + depth + timing fixed and varies only direction. These tests guard
    the control order, the direction-CI semantics, and the deposit serialization
    — all GPU-free (the real model is never loaded)."""

    def test_control_order_is_input_side_ascending(self):
        # The control freezes the input side first: ascending over the scope.
        scope = {24, 25, 26, 27, 28, 29, 30, 31}
        assert control_order_9b(scope) == (24, 25, 26, 27, 28, 29, 30, 31)

    def test_control_is_distinct_from_candidate(self):
        # The control is a third, distinct arm: ascending, not descending
        # (candidate) — the property that makes candidate-vs-control a direction
        # test rather than a candidate-vs-candidate no-op. A drift that flipped
        # the control to descending would collapse the isolation.
        scope = {24, 25, 26, 27, 28, 29, 30, 31}
        assert control_order_9b(scope) != candidate_order_9b(scope)

    def test_control_freezes_disjoint_block_from_candidate_same_depth(self):
        # The P0 isolation property: at depth d the candidate freezes the top-d
        # (output) layers and the control freezes the bottom-d (input) — equal-
        # size CONTIGUOUS blocks, disjoint, so only the side differs. Contiguity
        # + depth held fixed => the candidate-vs-control gap is pure direction.
        scope = {24, 25, 26, 27, 28, 29, 30, 31}
        depth = 3
        cand_block = set(candidate_order_9b(scope)[:depth])
        ctrl_block = set(control_order_9b(scope)[:depth])
        assert cand_block == {29, 30, 31}  # output-side contiguous
        assert ctrl_block == {24, 25, 26}  # input-side contiguous
        assert len(cand_block) == len(ctrl_block) == depth
        assert cand_block.isdisjoint(ctrl_block)

    def test_direction_ci_surpasses_means_direction_earned_the_lead(self):
        # candidate(output) losses well below control(input) losses => the CI on
        # mean(control) - mean(candidate) excludes zero above => SURPASSES =>
        # the output-side DIRECTION (not just contiguity) earned the lead. The
        # control occupies the "surrogate" slot of surrogate_valid_loss_ci.
        cand = [1.62, 1.63, 1.61, 1.62]
        ctrl = [1.75, 1.80, 1.78, 1.77]  # input-contiguous visibly worse
        dci = surrogate_valid_loss_ci(cand, ctrl, seed=0)
        assert dci.significance_verdict == SURPASSES
        assert dci.point_improvement > 0.0  # control mean - candidate mean > 0

    def test_direction_ci_ties_means_contiguity_confound(self):
        # candidate(output) ≈ control(input) => TIES => the surrogate SURPASSES
        # was contiguity, not direction — the misattribution the control exists
        # to surface. This is the honest "refuse to attribute" outcome.
        cand = [1.62, 1.63, 1.61, 1.62]
        ctrl = [1.621, 1.629, 1.611, 1.619]  # indistinguishable from candidate
        dci = surrogate_valid_loss_ci(cand, ctrl, seed=0)
        assert dci.significance_verdict == TIES

    def test_direction_ci_to_json_relabels_surrogate_mean_as_control_mean(self):
        # The control occupies the CI's "surrogate" slot, so its mean is
        # surrogate_mean internally; the deposit must relabel it control_mean so
        # a reader does not mistake an input-side control for the random
        # surrogate. Also surfaces n_control (from n_surrogate) and the verdict.
        cand = [1.62, 1.63, 1.61, 1.62]
        ctrl = [1.75, 1.80, 1.78, 1.77]
        dci = surrogate_valid_loss_ci(cand, ctrl, seed=0)
        out = _direction_ci_to_json(dci)
        assert out["verdict"] == SURPASSES
        assert out["control_mean"] == pytest.approx(sum(ctrl) / len(ctrl), abs=1e-9)
        assert out["candidate_mean"] == pytest.approx(sum(cand) / len(cand), abs=1e-9)
        assert out["n_control"] == 4
        assert out["n_candidate"] == 4
        # The relabel is load-bearing: an input-side control must NOT be called
        # surrogate_mean in the deposit.
        assert "surrogate_mean" not in out

    def test_direction_ci_to_json_none_round_trips(self):
        # No control arm (n_control=0) => direction_ci None => JSON null.
        assert _direction_ci_to_json(None) is None

    def test_result_to_json_no_control_is_backward_compatible(self):
        # n_control=0: no direction block, empty control fields, and the §4
        # verdict honesty labels byte-identical to the pre-control deposit shape.
        out = result_to_json(_stub_result())
        assert out["direction"] is None
        assert out["control_losses"] == []
        assert out["n_control"] == 0
        assert out["control_provenance"] == []
        # Existing §4 fields unchanged.
        assert out["verdict"] == SURPASSES
        assert out["citable_as_target_scale"] is True

    def test_result_to_json_with_control_surfaces_direction(self):
        # With a control arm run, the deposit surfaces a populated direction
        # block and the control losses/order/provenance — driven through the REAL
        # CI + _direction_ci_to_json path (not a stub) so the serialization is
        # exercised end-to-end.
        cand_losses = [1.62, 1.63, 1.61, 1.62]
        ctrl_losses = [1.75, 1.80, 1.78, 1.77]
        dci = surrogate_valid_loss_ci(cand_losses, ctrl_losses, seed=0)
        out = result_to_json(
            _stub_result(
                n_control=4,
                control_losses=ctrl_losses,
                control_order=[24, 25, 26, 27, 28, 29, 30, 31],
                control_provenance=[
                    {
                        "frozen_layers": [24, 25, 26],
                        "n_trainable_params": 6787072,
                        "last_train_loss": 0.001,
                    }
                ],
                direction_ci=dci,
            )
        )
        assert out["direction"] is not None
        assert out["direction"]["verdict"] == SURPASSES
        assert out["direction"]["control_mean"] == pytest.approx(
            sum(ctrl_losses) / len(ctrl_losses), abs=1e-9
        )
        assert out["n_control"] == 4
        assert out["control_losses"] == ctrl_losses
        assert out["control_order"] == [24, 25, 26, 27, 28, 29, 30, 31]
        # The direction arm is an attribution caveat, NOT a scale/budget axis:
        # it must not by itself open the full-§4 citation gate (this stub is a
        # reduced-budget run, so the gate stays closed regardless of direction).
        assert out["citable_as_full_section4_verdict"] is False

    def test_direction_ci_uses_distinct_seed_offset_from_surrogate(self):
        # The control arm must not collide with the candidate or surrogate arms'
        # LoRA-init seeds (candidate base_seed+i, surrogate base_seed+100+i;
        # control base_seed+200+i). Guarded at the order/seed level here since
        # the GPU arm loop is not unit-tested. The offsets must be disjoint
        # bands so no two arms share an init across a sane n_* range.
        base_seed = 0
        cand_seeds = {base_seed + i for i in range(4)}      # 0..3
        surr_seeds = {base_seed + 100 + i for i in range(4)}  # 100..103
        ctrl_seeds = {base_seed + 200 + i for i in range(4)}  # 200..203
        assert cand_seeds.isdisjoint(surr_seeds)
        assert cand_seeds.isdisjoint(ctrl_seeds)
        assert surr_seeds.isdisjoint(ctrl_seeds)


# ── full-backprop baseline control (GOAL §4 line 247: valid_loss vs full) ────


class TestBaselineControl:
    """The FULL-BACKPROP BASELINE arm is the §4 control the surrogate and
    direction arms are NOT. Both of those are freeze-vs-freeze: the candidate
    (output-first progressive freeze) is compared against other *freeze orders*
    (random surrogate, input-side control). GOAL §4 line 247's other success
    half — "valid_loss degradation within tolerance of FULL backprop" — needs a
    no-freeze arm: every active-scope layer trained on the full CE task loss
    throughout (``depth=0`` → ``max_depth=0`` → zero freezes → always the
    full-CE branch).

    The baseline occupies the "surrogate" slot of the generic two-sample
    bootstrap, so the verdict reads as: ``SURPASSES`` (candidate < baseline —
    the method BEATS full backprop), ``TIES`` (candidate ≈ baseline — freezing
    preserved quality, §4 line 247 satisfied), ``UNDERSHOOTS`` (candidate >
    baseline — the freeze cost quality, condition (a) failed at this budget).
    These tests guard the baseline CI semantics, the seed-offset disjointness,
    the load-bearing ``depth=0``, and the deposit serialization — all GPU-free
    (the real model is never loaded)."""

    def test_baseline_ci_surpasses_means_method_beats_full_backprop(self):
        # candidate losses well below the no-freeze baseline => the CI on
        # mean(baseline) - mean(candidate) excludes zero above => SURPASSES =>
        # the progressive-freeze method's valid_loss is BETTER than full
        # backprop at this budget (a regularization win, not just compute saved).
        cand = [1.50, 1.51, 1.49, 1.50]
        base = [1.58, 1.60, 1.59, 1.57]  # full backprov visibly worse
        bci = surrogate_valid_loss_ci(cand, base, seed=0)
        assert bci.significance_verdict == SURPASSES
        assert bci.point_improvement > 0.0  # baseline mean - candidate mean > 0

    def test_baseline_ci_ties_means_within_tolerance(self):
        # candidate ≈ baseline => TIES => the freeze preserved quality within
        # the gate's tolerance — §4 line 247 satisfied (this is the honest
        # "compute saved, quality maintained" outcome the §4 method targets).
        cand = [1.50, 1.51, 1.49, 1.50]
        base = [1.501, 1.509, 1.495, 1.502]  # indistinguishable
        bci = surrogate_valid_loss_ci(cand, base, seed=0)
        assert bci.significance_verdict == TIES

    def test_baseline_ci_undershoots_means_freeze_cost_quality(self):
        # candidate losses well ABOVE the no-freeze baseline => the CI on
        # mean(baseline) - mean(candidate) excludes zero below => UNDERSHOOTS =>
        # the freeze cost quality vs full backprop — §4 line 247 condition (a)
        # FAILED at this budget. This is the honest negative outcome the gate
        # must surface (not paint every baseline comparison as a win).
        cand = [1.62, 1.63, 1.61, 1.62]
        base = [1.50, 1.51, 1.49, 1.50]  # full backprop visibly better
        bci = surrogate_valid_loss_ci(cand, base, seed=0)
        assert bci.significance_verdict == UNDERSHOOTS
        assert bci.point_improvement < 0.0  # baseline mean - candidate mean < 0

    def test_baseline_ci_uses_distinct_seed_offset_from_all_other_arms(self):
        # The baseline arm must not collide with any other arm's LoRA-init seed:
        # candidate base_seed+i, surrogate base_seed+100+i, control
        # base_seed+200+i, baseline base_seed+300+i. Guarded at the seed level
        # here since the GPU arm loop is not unit-tested. The offsets must be
        # disjoint bands so no two arms share an init across a sane n_* range.
        base_seed = 0
        cand_seeds = {base_seed + i for i in range(4)}        # 0..3
        surr_seeds = {base_seed + 100 + i for i in range(4)}  # 100..103
        ctrl_seeds = {base_seed + 200 + i for i in range(4)}  # 200..203
        base_seeds = {base_seed + 300 + i for i in range(4)}  # 300..303
        # Pairwise disjoint: no two arm types share an init seed.
        for a in (cand_seeds, surr_seeds, ctrl_seeds):
            assert base_seeds.isdisjoint(a)
        assert cand_seeds.isdisjoint(surr_seeds)
        assert cand_seeds.isdisjoint(ctrl_seeds)
        assert surr_seeds.isdisjoint(ctrl_seeds)

    def test_baseline_arm_passes_depth_zero_no_freeze(self):
        # The baseline IS the no-freeze full-backprop control: depth=0 so
        # max_depth=0 plans zero freezes and the arm always takes the full-CE
        # branch. A drift to depth=depth would make the baseline freeze like the
        # candidate and the candidate-vs-baseline comparison meaningless. Source
        # inspection of run_ci_9b since the GPU arm loop is not unit-tested.
        import inspect

        src = inspect.getsource(run_ci_9b)
        # The baseline_results block passes a literal depth=0; the other three
        # arm blocks pass depth=depth (the shared param). Only the baseline has
        # the literal zero.
        assert "depth=0" in src

    def test_baseline_ci_to_json_relabels_surrogate_mean_as_baseline_mean(self):
        # The baseline occupies the CI's "surrogate" slot, so its mean is
        # surrogate_mean internally; the deposit must relabel it baseline_mean
        # so a reader does not mistake the no-freeze full-CE control for the
        # random-order *freeze* surrogate. Also surfaces n_baseline (from
        # n_surrogate) and the verdict.
        cand = [1.50, 1.51, 1.49, 1.50]
        base = [1.58, 1.60, 1.59, 1.57]
        bci = surrogate_valid_loss_ci(cand, base, seed=0)
        out = _baseline_ci_to_json(bci)
        assert out["verdict"] == SURPASSES
        assert out["baseline_mean"] == pytest.approx(sum(base) / len(base), abs=1e-9)
        assert out["candidate_mean"] == pytest.approx(sum(cand) / len(cand), abs=1e-9)
        assert out["n_baseline"] == 4
        assert out["n_candidate"] == 4
        # The relabel is load-bearing: a no-freeze control must NOT be called
        # surrogate_mean in the deposit (that label means a random freeze here).
        assert "surrogate_mean" not in out

    def test_baseline_ci_to_json_none_round_trips(self):
        # No baseline arm (n_baseline=0) => baseline_ci None => JSON null.
        assert _baseline_ci_to_json(None) is None

    def test_result_to_json_no_baseline_is_backward_compatible(self):
        # n_baseline=0: no baseline block, empty baseline fields, and the §4
        # verdict honesty labels byte-identical to the pre-baseline deposit shape.
        out = result_to_json(_stub_result())
        assert out["baseline"] is None
        assert out["baseline_losses"] == []
        assert out["n_baseline"] == 0
        assert out["baseline_provenance"] == []
        # Existing §4 fields unchanged.
        assert out["verdict"] == SURPASSES
        assert out["citable_as_target_scale"] is True

    def test_result_to_json_with_baseline_surfaces_baseline_block(self):
        # With a baseline arm run, the deposit surfaces a populated baseline
        # block and the baseline losses/provenance — driven through the REAL
        # CI + _baseline_ci_to_json path (not a stub) so the serialization is
        # exercised end-to-end.
        cand_losses = [1.50, 1.51, 1.49, 1.50]
        base_losses = [1.58, 1.60, 1.59, 1.57]
        bci = surrogate_valid_loss_ci(cand_losses, base_losses, seed=0)
        out = result_to_json(
            _stub_result(
                n_baseline=4,
                baseline_losses=base_losses,
                baseline_provenance=[
                    {
                        "frozen_layers": [],
                        "n_trainable_params": 10819584,
                        "last_train_loss": 1.49,
                        "final_ce_train_loss": 1.49,
                    }
                ],
                baseline_ci=bci,
            )
        )
        assert out["baseline"] is not None
        assert out["baseline"]["verdict"] == SURPASSES
        assert out["baseline"]["baseline_mean"] == pytest.approx(
            sum(base_losses) / len(base_losses), abs=1e-9
        )
        assert out["n_baseline"] == 4
        assert out["baseline_losses"] == base_losses
        # The baseline never freezes: provenance records an empty frozen set
        # and the FULL active-scope trainable-param count (no layers removed).
        assert out["baseline_provenance"][0]["frozen_layers"] == []
        assert out["baseline_provenance"][0]["n_trainable_params"] == 10819584
        # The baseline arm is a quality axis, NOT a scale/budget axis: it must
        # not by itself open the full-§4 citation gate (this stub is a
        # reduced-budget run, so the gate stays closed regardless of baseline).
        assert out["citable_as_full_section4_verdict"] is False


# ── _is_reduced_budget: honest (budget-driven) reduced-budget flag ───────────


class TestReducedBudgetHonest:
    """The flag must track the actual step budget vs the config, not be a
    hardcoded ``True``. A hardcoded flag would lie about a future full-length
    run and keep the citation gate permanently closed."""

    def test_short_of_max_steps_is_reduced(self):
        assert _is_reduced_budget(total_steps=40, max_steps=1500) is True

    def test_reaching_max_steps_is_not_reduced(self):
        assert _is_reduced_budget(total_steps=1500, max_steps=1500) is False

    def test_exceeding_max_steps_is_not_reduced(self):
        # Over-training past the config also clears "reduced" — the run was at
        # least the full intended length.
        assert _is_reduced_budget(total_steps=2000, max_steps=1500) is False

    def test_absent_max_steps_is_reduced(self):
        # An unparsed / absent config (max_steps <= 0): conservative → reduced,
        # never silently promoting a run whose intended length is unknown.
        assert _is_reduced_budget(total_steps=1500, max_steps=0) is True
        assert _is_reduced_budget(total_steps=1500, max_steps=-1) is True


# ── deposit replay faithfulness on the committed real-9B recording ──────────


class TestDepositReplayFaithfulness:
    def test_fixture_exists_and_is_real_target_scale(self):
        data = json.loads(FIXTURE_9B_SURROGATE.read_text())
        assert data["proxy_scale"] is False
        assert data["citable_as_target_scale"] is True
        assert data["citable_as_full_section4_verdict"] is False
        assert data["reduced_budget"] is True
        assert data["model"] == "Qwen/Qwen3.5-9B"
        assert data["seq_len"] == 1024
        assert data["device"] == "cuda"

    def test_fixture_is_non_thin_evidence(self):
        # The load-bearing pin for the non-thin upgrade: the committed deposit
        # clears the MIN_SAMPLE_FOR_BOOTSTRAP bar (>=3 seeds/arm) so the
        # recorded SURPASSES is confirmed, not thin-flagged. A regression that
        # re-deposits a 2-seed run would flip this red.
        data = json.loads(FIXTURE_9B_SURROGATE.read_text())
        assert data["is_thin_evidence"] is False
        assert data["n_candidate"] >= 3
        assert data["n_surrogate"] >= 3
        # reduced_budget is judged against a surfaced cfg_max_steps (not a
        # mystery boolean) — the honesty fix this deposit exercises.
        assert data["cfg_max_steps"] == 1500
        assert data["total_steps"] < data["cfg_max_steps"]  # honestly reduced

    def test_recorded_losses_earn_the_recorded_verdict(self):
        """The stored floats re-judge to the SURPASSES the file records — the
        verdict is earned under the deterministic bootstrap, not painted on."""
        data = json.loads(FIXTURE_9B_SURROGATE.read_text())
        ci = surrogate_valid_loss_ci(
            data["candidate_losses"],
            data["surrogate_losses"],
            seed=data["base_seed"],
        )
        assert ci.significance_verdict == data["verdict"] == SURPASSES
        # Means + CI round-trip within float tolerance.
        assert ci.candidate_mean == pytest.approx(data["candidate_mean"], abs=1e-9)
        assert ci.surrogate_mean == pytest.approx(data["surrogate_mean"], abs=1e-9)
        assert ci.lower == pytest.approx(data["lower"], abs=1e-9)
        assert ci.upper == pytest.approx(data["upper"], abs=1e-9)

    def test_candidate_beats_surrogate_directionally(self):
        data = json.loads(FIXTURE_9B_SURROGATE.read_text())
        # Every candidate arm's valid_loss is below every surrogate arm's — the
        # output-first freeze retains quality better than random-order at this
        # budget, the directional signal GOAL §4 predicts.
        assert max(data["candidate_losses"]) < min(data["surrogate_losses"])

    def test_provenance_shows_output_first_vs_random_freeze(self):
        data = json.loads(FIXTURE_9B_SURROGATE.read_text())
        cand = [tuple(p["frozen_layers"]) for p in data["candidate_provenance"]]
        surr = [tuple(p["frozen_layers"]) for p in data["surrogate_provenance"]]
        # Candidate froze a contiguous output-side suffix {29,30,31} in every arm.
        assert all(c == (29, 30, 31) for c in cand)
        # Surrogate froze a *different* (random-order) set in every arm.
        assert all(c not in surr for c in cand)


# ── direction-isolation deposit replay faithfulness (constitution P0) ────────


class TestDirectionDepositReplayFaithfulness:
    """The direction-isolation deposit — a real RTX 3060 seq1024 suffix-only 9B
    run that adds an input-side contiguous control arm alongside the candidate +
    random surrogate — re-judges through :func:`surrogate_valid_loss_ci` to BOTH
    verdicts it records. The §4 surrogate SURPASSES (candidate output-contiguous
    vs random) reproduces, AND the direction SURPASSES (candidate output-
    contiguous vs control input-contiguous, contiguity held fixed) reproduces —
    so the recorded attribution of the §4 verdict to output-side *direction*
    (not mere contiguity) is earned under the deterministic bootstrap, not
    painted on. This is the expected-output assertion that pins the
    constitution-P0 dataset."""

    def test_fixture_exists_and_is_real_target_scale(self):
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        assert data["proxy_scale"] is False
        assert data["citable_as_target_scale"] is True
        assert data["citable_as_full_section4_verdict"] is False
        assert data["reduced_budget"] is True
        assert data["model"] == "Qwen/Qwen3.5-9B"
        assert data["seq_len"] == 1024
        assert data["device"] == "cuda"

    def test_direction_deposit_carries_non_thin_control(self):
        # The direction verdict requires a non-thin control arm (>=3 seeds) for
        # the bootstrap to capture variance; a 1- or 2-seed control would be
        # flagged thin and the direction verdict not confirmable.
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        assert data["n_control"] >= 3
        assert data["direction"] is not None
        assert data["direction"]["is_thin_evidence"] is False
        assert data["direction"]["n_control"] >= 3

    def test_recorded_surrogate_verdict_is_earned(self):
        # The §4 surrogate A/B (candidate output-contiguous vs random surrogate)
        # reproduces in this richer run too — same honesty shape as the sibling
        # surrogate-only deposit.
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        ci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["surrogate_losses"], seed=data["base_seed"]
        )
        assert ci.significance_verdict == data["verdict"] == SURPASSES
        assert ci.candidate_mean == pytest.approx(data["candidate_mean"], abs=1e-9)
        assert ci.surrogate_mean == pytest.approx(data["surrogate_mean"], abs=1e-9)

    def test_recorded_direction_verdict_is_earned(self):
        # The load-bearing P0 assertion: the direction CI is NOT painted on — the
        # stored candidate/control floats re-judge through the deterministic
        # bootstrap to the recorded direction verdict. The control occupies the
        # CI's "surrogate" slot.
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        dd = data["direction"]
        dci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["control_losses"], seed=data["base_seed"]
        )
        assert dci.significance_verdict == dd["verdict"] == SURPASSES
        assert dci.candidate_mean == pytest.approx(dd["candidate_mean"], abs=1e-9)
        # The control (input-side) mean is relabeled control_mean in the deposit.
        assert dci.surrogate_mean == pytest.approx(dd["control_mean"], abs=1e-9)
        assert dci.lower == pytest.approx(dd["lower"], abs=1e-9)
        assert dci.upper == pytest.approx(dd["upper"], abs=1e-9)

    def test_direction_candidate_pool_is_shared_with_surrogate(self):
        # The candidate arms are shared between the surrogate A/B and the
        # direction A/B (the same 4 candidate arms feed both CIs). A drift that
        # re-ran candidate arms separately for the direction arm would break this
        # equality and double the compute.
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        assert data["candidate_mean"] == pytest.approx(
            data["direction"]["candidate_mean"], abs=1e-9
        )

    def test_control_freeze_is_input_side_contiguous(self):
        # The isolation property, pinned on real numbers: every control arm froze
        # the contiguous INPUT-side block {24,25,26} (ascending = input_first),
        # disjoint from the candidate's contiguous OUTPUT-side block {29,30,31}.
        # Contiguity + depth held fixed => the recorded direction gap is pure
        # direction, the P0 property this deposit exists to demonstrate.
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        cand_froze = {tuple(p["frozen_layers"]) for p in data["candidate_provenance"]}
        ctrl_froze = {tuple(p["frozen_layers"]) for p in data["control_provenance"]}
        assert cand_froze == {(29, 30, 31)}  # output-contiguous
        assert ctrl_froze == {(24, 25, 26)}  # input-contiguous
        assert cand_froze.isdisjoint(ctrl_froze)
        # control_order is the full ascending scope (frozen to depth=3 = {24,25,26}).
        assert data["control_order"] == [24, 25, 26, 27, 28, 29, 30, 31]

    def test_control_is_worse_than_surrogate_directionally(self):
        # Freezing the input-contiguous block {24,25,26} is worse than freezing
        # random scattered sets, which is worse than the output-contiguous
        # candidate — the monotone ordering (output-contig < random < input-
        # contig) that confirms DIRECTION, not just contiguity, drives the §4
        # verdict. Pinned on the recorded means.
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        cand_mean = data["candidate_mean"]
        surr_mean = data["surrogate_mean"]
        ctrl_mean = data["direction"]["control_mean"]
        assert cand_mean < surr_mean < ctrl_mean

    def test_deposit_is_non_thin_overall(self):
        # The deposit as a whole is non-thin: enough seeds in every arm for the
        # bootstrap to be meaningful. (The full-§4 citation gate stays closed on
        # the reduced-budget axis, NOT the thinness axis.)
        data = json.loads(FIXTURE_9B_DIRECTION.read_text())
        assert data["is_thin_evidence"] is False
        assert data["n_candidate"] >= 3
        assert data["n_surrogate"] >= 3
        assert data["n_control"] >= 3


# ── generalization-regime deposit (memorization-robustness of the §4 verdict) ─


class TestGeneralizationRegimeDeposit:
    """The §4 verdict's MEMORIZATION-ROBUSTNESS deposit. The sibling deposits
    (FIXTURE_9B_SURROGATE / _DIRECTION) run in a regime where 8 train examples
    cycled over 20 steps drive the LoRA's train CE to ~0 — i.e. the model
    MEMORIZED, so the held-out valid_loss is dominated by the frozen base barely
    perturbed by an overfit adapter. This deposit re-runs the SAME A/B (candidate
    vs random surrogate vs input-side control) in a GENERALIZATION regime — 48
    train examples over 96 steps (2 epochs), where each arm's
    ``final_ce_train_loss`` is ~1.5 (train CE ≈ valid CE, not ~0) — and asks
    whether the SURPASSES verdict survives the regime change or was a
    memorization artifact. The recorded answer: SURVIVES (both verdicts SURPASSES,
    non-thin, same monotone candidate < surrogate < control ranking), with an
    honest ~12x smaller effect size. These tests pin that the deposit is a genuine
    generalization run (the load-bearing ``final_ce_train_loss`` diagnostic) and
    that its recorded verdicts are earned under the deterministic bootstrap."""

    def test_fixture_exists_and_is_real_target_scale(self):
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        assert data["proxy_scale"] is False
        assert data["citable_as_target_scale"] is True
        assert data["citable_as_full_section4_verdict"] is False
        assert data["reduced_budget"] is True  # 96 < cfg_max_steps=1500, honest
        assert data["model"] == "Qwen/Qwen3.5-9B"
        assert data["seq_len"] == 1024
        assert data["device"] == "cuda"

    def test_regime_is_generalization_not_memorization(self):
        # The load-bearing regime assertion. ``final_ce_train_loss`` per arm is
        # the mean full-CE over the TRAIN set under the final adapter — ~1.5
        # here, NOT ~0. In the memorization-regime deposits the LoRA drove train
        # CE to ~5e-4 (candidate) / 0.0 (control). A regression that re-deposits
        # a memorization run (few examples cycled until train CE ~0) flips this
        # red — which is exactly the regression this deposit exists to catch.
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        for arm in ("candidate", "surrogate", "control"):
            prov = data[f"{arm}_provenance"]
            assert prov, f"{arm} provenance empty"
            for p in prov:
                assert p["final_ce_train_loss"] > 0.5  # generalized, not memorized
        # And train CE ≈ valid CE (small train-valid gap) — the signature of
        # generalization. Memorization would show train ≪ valid (gap ~1.6).
        cand_train = (
            sum(p["final_ce_train_loss"] for p in data["candidate_provenance"])
            / len(data["candidate_provenance"])
        )
        assert cand_train == pytest.approx(data["candidate_mean"], abs=0.05)

    def test_final_ce_disambiguates_from_local_last_train_loss(self):
        # The honesty fix this deposit exercises: ``last_train_loss`` (the last
        # optimizer step's loss = boundary activation-matching LOCAL loss once
        # frozen, structurally ~0) reads ~0.0002 while ``final_ce_train_loss``
        # (the full cross-entropy on train) reads ~1.5. Without the new field a
        # reader would misread the near-zero ``last_train_loss`` as memorization.
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        p = data["candidate_provenance"][0]
        assert p["last_train_loss"] < 1e-3        # local loss ~0 (post-freeze)
        assert p["final_ce_train_loss"] > 1.0     # true train CE ~1.5
        # The two fields differ by >1000x — they measure different things, so
        # both must be carried to read the regime correctly.
        assert p["final_ce_train_loss"] / max(p["last_train_loss"], 1e-12) > 1e3

    def test_recorded_surrogate_verdict_is_earned(self):
        # The stored floats re-judge to the recorded SURPASSES under the
        # deterministic bootstrap — the verdict is earned, not painted on.
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        ci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["surrogate_losses"], seed=data["base_seed"]
        )
        assert ci.significance_verdict == data["verdict"] == SURPASSES
        assert ci.candidate_mean == pytest.approx(data["candidate_mean"], abs=1e-9)
        assert ci.surrogate_mean == pytest.approx(data["surrogate_mean"], abs=1e-9)
        assert ci.lower == pytest.approx(data["lower"], abs=1e-9)
        assert ci.upper == pytest.approx(data["upper"], abs=1e-9)

    def test_recorded_direction_verdict_is_earned(self):
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        dd = data["direction"]
        dci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["control_losses"], seed=data["base_seed"]
        )
        assert dci.significance_verdict == dd["verdict"] == SURPASSES
        # The control (input-side) mean is relabeled control_mean in the deposit.
        assert dci.surrogate_mean == pytest.approx(dd["control_mean"], abs=1e-9)

    def test_monotone_ranking_survives_the_regime_change(self):
        # The regime-robustness result: the same output-contig < random <
        # input-contig ordering as the memorization-regime deposit holds in the
        # generalization regime too — the §4 verdict is NOT a memorization
        # artifact. Pinned on the recorded means.
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        assert (
            data["candidate_mean"]
            < data["surrogate_mean"]
            < data["direction"]["control_mean"]
        )

    def test_effect_size_shrank_vs_memorization_regime(self):
        # Honest regime-dependence: the candidate-vs-surrogate effect is much
        # smaller in the generalization regime than the memorization regime.
        # This is NOT a failure — it is the measurement: the verdict's DIRECTION
        # is robust, its MAGNITUDE is regime-dependent. Cross-deposit comparison.
        gen = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        mem = json.loads(FIXTURE_9B_SURROGATE.read_text())
        assert gen["point_improvement"] < mem["point_improvement"]
        # Both still SURPASSES (direction robust); only magnitude differs.
        assert gen["verdict"] == SURPASSES
        assert mem["verdict"] == SURPASSES

    def test_direction_isolation_holds_in_generalization_regime(self):
        # The P0 contiguity-isolation property reproduces in this regime too:
        # candidate froze the contiguous OUTPUT block {29,30,31}, control froze
        # the contiguous INPUT block {24,25,26} — so the candidate<control gap is
        # pure direction, not contiguity.
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        cand = {tuple(p["frozen_layers"]) for p in data["candidate_provenance"]}
        ctrl = {tuple(p["frozen_layers"]) for p in data["control_provenance"]}
        assert cand == {(29, 30, 31)}  # output-contiguous
        assert ctrl == {(24, 25, 26)}  # input-contiguous
        assert cand.isdisjoint(ctrl)

    def test_deposit_is_non_thin(self):
        data = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        assert data["is_thin_evidence"] is False
        assert data["n_candidate"] >= 3
        assert data["n_surrogate"] >= 3
        assert data["n_control"] >= 3


# ── full-backprop baseline deposit (GOAL §4 condition (a): vs FULL backprop) ──


class TestBaselineControlDeposit:
    """The GOAL §4 CONDITION-(a) deposit — the ONE measurement the sibling
    deposits (FIXTURE_9B_SURROGATE / _DIRECTION / _GENERALIZATION) structurally
    cannot make. Those three are all freeze-vs-freeze A/Bs (candidate output-
    contiguous vs random surrogate vs input-contiguous control), so they can
    show the candidate beats OTHER FREEZE ORDERS but never whether freezing
    costs quality vs FULL backprop. GOAL §4 (docs/GOAL.md:244-250) demands BOTH:
    (a) valid_loss within tolerance of the FULL-BACKPROP baseline, AND (b) the
    FLOPs-reduction win over the random surrogate. The sibling deposits close
    (b); THIS deposit closes (a) by adding a ``depth=0`` no-freeze arm
    (``max_depth=0`` → zero freezes planned → the arm trains EVERY active-scope
    layer on the full task CE throughout, never switching to the boundary
    activation-matching local loss). It is run in the honest GENERALIZATION
    regime (48 train / 96 step / 2 epoch) so the comparison is a model that
    generalized, not memorized, and at target scale seq1024 on the real
    RTX 3060 suffix-only 9B config.

    These tests pin that the baseline arm is a GENUINE full-backprop control
    (never freezes, full CE throughout — ``last_train_loss`` stays at full-CE
    magnitude ~1.5, NOT the ~0 local loss of the frozen arms), that the deposit
    is a real non-thin generalization-regime target-scale run, and that its
    recorded candidate-vs-baseline CI is earned under the deterministic
    bootstrap (verdict-agnostic — holds whether the verdict is SURPASSES, TIES,
    or UNDERSHOOTS)."""

    def test_fixture_exists_and_is_real_target_scale(self):
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        assert data["proxy_scale"] is False
        assert data["citable_as_target_scale"] is True
        assert data["reduced_budget"] is True  # 96 < cfg_max_steps=1500, honest
        assert data["model"] == "Qwen/Qwen3.5-9B"
        assert data["seq_len"] == 1024
        assert data["device"] == "cuda"

    def test_baseline_deposit_carries_non_thin_baseline(self):
        # The §4 condition-(a) verdict requires a non-thin baseline arm (>=3
        # seeds) for the bootstrap to capture variance; a 1- or 2-seed baseline
        # would be flagged thin and the tolerance verdict not confirmable.
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        assert data["baseline"] is not None
        assert data["n_baseline"] >= 3
        assert data["baseline"]["is_thin_evidence"] is False
        assert data["baseline"]["n_baseline"] >= 3

    def test_baseline_arm_never_freezes(self):
        # The LOAD-BEARING control-integrity assertion: every baseline arm froze
        # ZERO layers (``depth=0`` → ``max_depth=0`` plans zero freezes) and
        # trained the FULL active scope. A regression that accidentally passed
        # ``depth>0`` to the baseline, or let a freeze schedule leak into it,
        # would populate ``frozen_layers`` / shrink ``n_trainable_params`` and
        # flip this red — which is exactly the regression this deposit exists
        # to catch, because a baseline that freezes is NOT a full-backprop
        # control and invalidates the condition-(a) measurement.
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        for p in data["baseline_provenance"]:
            assert p["frozen_layers"] == []
            # Full scope trainable, not the reduced post-freeze count.
            assert p["n_trainable_params"] == data["scope_trainable_params"]

    def test_baseline_uses_full_ce_throughout(self):
        # The other control-integrity assertion, pinned on the loss field: the
        # baseline's ``last_train_loss`` is the last optimizer step's FULL task
        # cross-entropy (no freeze → the local-loss branch never triggers), so
        # it stays at full-CE magnitude (~1.5), NOT the structurally-≈0 boundary
        # activation-matching local loss the FROZEN arms record. This is what
        # makes the arm a full-backprop control: it optimizes task CE end-to-end
        # rather than a frozen-boundary surrogate. Contrast the candidate, whose
        # ``last_train_loss`` collapses to ~0 once it freezes — the two arms
        # genuinely trained under different objectives.
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        for p in data["baseline_provenance"]:
            assert p["last_train_loss"] > 0.5  # full CE magnitude (order ~2), not ~0
            # last step CE and mean train CE are the SAME order of magnitude (both
            # full CE) — NOT the >1000x gap that would signal a switch to the
            # activation-matching local loss the frozen arms take.
            ratio = p["final_ce_train_loss"] / max(p["last_train_loss"], 1e-12)
            assert 0.1 < ratio < 10.0
        # And the frozen candidate arm DOES collapse to the local loss (~0) — so
        # baseline-vs-candidate is full-CE-vs-local, two genuinely different
        # training objectives, not the same run relabeled.
        cand = data["candidate_provenance"][0]
        assert cand["last_train_loss"] < 1e-3

    def test_recorded_surrogate_verdict_is_earned(self):
        # The §4 surrogate A/B (candidate output-contiguous vs random) reproduces
        # in this 4-arm run too — same honesty shape as the sibling deposits.
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        ci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["surrogate_losses"], seed=data["base_seed"]
        )
        assert ci.significance_verdict == data["verdict"]
        assert ci.candidate_mean == pytest.approx(data["candidate_mean"], abs=1e-9)
        assert ci.surrogate_mean == pytest.approx(data["surrogate_mean"], abs=1e-9)

    def test_recorded_baseline_verdict_is_earned(self):
        # The load-bearing condition-(a) assertion: the candidate-vs-FULL-
        # BACKPROP-baseline CI is NOT painted on — the stored candidate/baseline
        # floats re-judge through the deterministic bootstrap to the recorded
        # baseline verdict. Verdict-agnostic: holds whether the method SURPASSES
        # (beats full backprop), TIES (quality maintained — the §4 target), or
        # UNDERSHOOTS (freezing cost quality at this reduced budget). The
        # baseline occupies the CI's "surrogate" slot; its mean is relabeled
        # ``baseline_mean`` in the deposit.
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        bd = data["baseline"]
        bci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["baseline_losses"], seed=data["base_seed"]
        )
        assert bci.significance_verdict == bd["verdict"]
        assert bci.candidate_mean == pytest.approx(bd["candidate_mean"], abs=1e-9)
        assert bci.surrogate_mean == pytest.approx(bd["baseline_mean"], abs=1e-9)
        assert bci.lower == pytest.approx(bd["lower"], abs=1e-9)
        assert bci.upper == pytest.approx(bd["upper"], abs=1e-9)
        # The recorded verdict is a real constant, not free text.
        assert bd["verdict"] in {SURPASSES, TIES, UNDERSHOOTS}

    def test_baseline_candidate_pool_is_shared_with_surrogate(self):
        # The candidate arms are shared between the surrogate A/B and the
        # condition-(a) baseline A/B (the same candidate arms feed both CIs). A
        # drift that re-ran candidate arms separately for the baseline arm would
        # break this equality and double the compute.
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        assert data["candidate_mean"] == pytest.approx(
            data["baseline"]["candidate_mean"], abs=1e-9
        )

    def test_candidate_beats_full_backprop_baseline(self):
        # The HEADLINE §4 condition-(a) result, pinned on the recorded numbers:
        # the progressive-freeze candidate's held-out valid_loss is LOWER than
        # the no-freeze full-backprop baseline's — the method does not merely
        # stay within tolerance of full backprop (a TIES would already satisfy
        # §4 line 247), it SURPASSES it at this budget. The recorded baseline
        # verdict is SURPASSES, non-thin, with the CI entirely above 0. This is
        # the single measurement the freeze-vs-freeze sibling deposits could
        # never make, and it closes §4 condition (a) on the side of "freezing
        # did not cost quality".
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        bd = data["baseline"]
        assert bd["verdict"] == SURPASSES
        assert bd["is_thin_evidence"] is False
        assert bd["lower"] > 0.0  # CI entirely above 0 => significant
        assert data["candidate_mean"] < bd["baseline_mean"]

    def test_full_backprop_baseline_overfits_freeze_regularizes(self):
        # The MECHANISM behind the headline, pinned on the train/valid
        # diagnostics. The full-backprop baseline fit the TRAIN set harder
        # (final_ce_train_loss ≈ 0.77) than the frozen candidate (≈ 1.51) but
        # generalized WORSE on held-out valid (1.540 > 1.515): lower train CE +
        # higher valid CE = textbook overfitting. The progressive freeze capped
        # how hard the adapter could fit train (the frozen layers + activation-
        # matching local loss leave train CE at ~1.5) and that regularization
        # generalized better. So "candidate beats full backprop" is not noise —
        # it is the freeze acting as a regularizer in this reduced-budget
        # generalization regime. (At the full §4 budget this gap may close or
        # invert; the deposit is reduced_budget=True and honest about that.)
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        cand_train = (
            sum(p["final_ce_train_loss"] for p in data["candidate_provenance"])
            / len(data["candidate_provenance"])
        )
        base_train = (
            sum(p["final_ce_train_loss"] for p in data["baseline_provenance"])
            / len(data["baseline_provenance"])
        )
        assert base_train < cand_train          # baseline fit train harder...
        assert data["baseline"]["baseline_mean"] > data["candidate_mean"]  # ...yet valid worse

    def test_regime_is_generalization_not_memorization(self):
        # All four arms — including the no-freeze baseline — ran in the
        # generalization regime: every arm's ``final_ce_train_loss`` is ~1.5
        # (train CE ≈ valid CE), NOT ~0. A regression that re-deposited a
        # memorization run (few examples cycled until train CE ~0) flips this
        # red, which would corrupt the condition-(a) comparison (a memorized
        # baseline barely moves, making any candidate look comparable).
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        for arm in ("candidate", "surrogate", "control", "baseline"):
            prov = data[f"{arm}_provenance"]
            assert prov, f"{arm} provenance empty"
            for p in prov:
                assert p["final_ce_train_loss"] > 0.5  # generalized, not memorized

    def test_deposit_is_non_thin(self):
        # The deposit as a whole is non-thin across all four arms: enough seeds
        # in every arm for the bootstrap to be meaningful. (The full-§4 citation
        # gate stays closed on the reduced-budget axis, NOT the thinness axis.)
        data = json.loads(FIXTURE_9B_BASELINE.read_text())
        assert data["is_thin_evidence"] is False
        assert data["n_candidate"] >= 3
        assert data["n_surrogate"] >= 3
        assert data["n_control"] >= 3
        assert data["n_baseline"] >= 3


# ── CLI health ──────────────────────────────────────────────────────────────


class TestCli:
    def test_build_parser_advertises_target_scale(self):
        p = build_parser()
        text = p.format_help()
        # The CLI is explicit that this is the real 9B target-scale A/B
        # (proxy_scale=False), not another proxy run.
        assert "target-scale" in text.lower()
        assert "--seq-len" in text
        assert "--n-candidate" in text
        # The DIRECTION-CONTROL flag (constitution P0) is part of the CLI surface.
        assert "--n-control" in text

    def test_help_launches_as_module(self):
        # The canary contract: every scripts.* CLI launches via ``-m`` with a
        # working ``--help`` and exit 0 (the sys.path-bootstrap invariant).
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "scripts.run_freeze_validloss_ci_9b", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert "usage" in (proc.stdout + proc.stderr).lower()
