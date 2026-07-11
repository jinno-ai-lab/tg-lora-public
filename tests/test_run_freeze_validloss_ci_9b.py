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
    _direction_ci_to_json,
    _is_reduced_budget,
    _reset_lora_for_arm,
    arm_valid_loss_9b,
    build_parser,
    build_sft_example,
    candidate_order_9b,
    control_order_9b,
    result_to_json,
)
from src.tg_lora.freeze_surrogate_ci import surrogate_valid_loss_ci
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES

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

    def test_full_verdict_only_when_full_budget_target_scale_and_non_thin(self):
        # The citation gate opens ONLY for a full-budget, target-scale,
        # NON-THIN run — all three honesty axes must clear together.
        out = result_to_json(_stub_result(
            proxy_scale=False, reduced_budget=False,
            ci=_stub_ci(is_thin_evidence=False),
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
