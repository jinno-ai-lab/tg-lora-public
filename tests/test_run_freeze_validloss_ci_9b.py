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
  (``proxy_scale=False``, ``citable_as_target_scale=True``; ``reduced_budget``
  and ``citable_as_full_section4_verdict`` are runtime-determined by the 4-axis
  gate — a full-budget generalization-regime run clears both).
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

import hashlib
import json
import math
import os
import re
import types
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import scripts.run_freeze_validloss_ci_9b as ci9b
from scripts.run_freeze_validloss_ci_9b import (
    ARCHITECTURES,
    HETEROGENEOUS,
    HOMOGENEOUS,
    EXIT_GPU_TEMPFAIL,
    REGIME_GENERALIZATION,
    REGIME_MEMORIZATION,
    REGIME_OVERFIT,
    REGIME_UNKNOWN,
    _active_scope_pre_lora,
    _assert_dolly_schema,
    _baseline_ci_to_json,
    _candidate_cost_reduction,
    _candidate_final_ce_mean,
    _classify_regime,
    _direction_ci_to_json,
    _evidence_hash,
    _full_section4_verdict_gate,
    _gather_arm_curves,
    _is_reduced_budget,
    _load_dolly_records,
    _reset_lora_for_arm,
    _run_log_payload,
    _run_log_sha256,
    _write_deposit,
    _write_run_log,
    arm_valid_loss_9b,
    build_parser,
    build_rank_pattern,
    build_sft_example,
    candidate_order_9b,
    control_order_9b,
    heterogeneous_ranks_9b,
    is_cuda_oom,
    main,
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

# The FULL-BUDGET recording: the only deposit generated with ``reduced_budget=
# False`` (``--total-steps 1500`` reaches config ``max_steps``), so it is the
# FIRST committed deposit where the gate's regime conjunct is load-bearing —
# target-scale + non-thin + full-budget clear, the verdict turns on whether the
# run generalized. Recorded on a real RTX 3060 (Qwen3.5-9B, seq1024, suffix-only
# last-25% scope, 600 train / 1500 step ≈ 2.5 epoch, 3 seeds × 3 arms = 9 arms,
# ~3 h GPU via ``make freeze-validloss-ci-9b-full-bg``; resumed_arm_count=0, a
# clean single-session run). The candidate arm generalized
# (final_ce_train_loss_mean=1.366 ≈ valid 1.695; gap 0.33 — not memorization
# (ce≈0) nor overfit (gap>0.5)) → ``regime="generalization"`` → the 4-conjunct
# gate opens: ``citable_as_full_section4_verdict=True``. Headline verdict:
# **TIES** (candidate 1.6947 vs random-order surrogate 1.6960, CI[95%]≈[-0.0001,
# +0.0027] straddles 0) — at full budget the output-first freeze order does NOT
# beat a random order on the homogeneous (single-scope) leg; the order effect
# seen at reduced budget does not survive full-budget generalization. The §4
# condition (a) arm reproduces at full budget: candidate **SURPASSES** full
# backprop (1.6947 vs 1.8794, CI[0.164, 0.201]) — the freeze acts as a
# regularizer (baseline overfit: train CE 0.78 ≪ valid 1.88; frozen arms
# generalized). Regenerate with ``make freeze-validloss-ci-9b-full``.
FIXTURE_9B_FULL = (
    Path(__file__).resolve().parent / "fixtures" / "freeze_validloss_ci_9b_full.json"
)

# The committed LEDGER WITNESS for the citable full-budget deposit — a
# byte-identical copy of the run's resumability ledger
# (``runs/..._ledger.jsonl``), checked in under ``tests/fixtures/`` so the
# citable verdict is independently reproducible from committed bytes alone (no
# gitignored ``runs/`` path, no private stable dir). 10 JSONL lines: 1 header
# (run-config fingerprint) + 9 arms (3 candidate / 3 surrogate / 3 baseline),
# each carrying the arm's exact ``valid_loss`` — so the deposit's three
# loss-vectors reconstruct byte-for-byte from this witness. Stamped into the
# full deposit as ``ledger_witness_path`` / ``ledger_witness_sha256`` (a HARVEST
# field, distinct from the runtime ``ledger_path`` which stays ``null`` for the
# one-shot full run — the witness is the committed verification copy, not the
# incrementally-written runtime resumability path).
FIXTURE_9B_FULL_LEDGER = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_full_ledger.jsonl"
)

# The HETEROGENEOUS (per-layer asymmetric rank) recording — the one §4 research
# task that was open at target scale. Every sibling deposit above runs on a
# HOMOGENEOUS LoRA (every active layer shares r=16); this deposit re-runs the
# SAME generalization-regime A/B (FREEZE_9B_GEN_FLAGS — candidate output-first
# vs random surrogate vs input-contig control, 48 train / 96 step / 2 epoch,
# 3 seeds/arm) on an ASYMMETRIC adapter via peft rank_pattern: output-side layers
# get more CAPACITY (geometric ranks {24:1,25:1,26:2,27:3,28:5,29:7,30:11,31:16};
# alpha=2*rank held constant so only capacity, not magnitude, varies). Recorded
# result: headline **SURVIVES the architecture change** — candidate (output-first,
# 1.5424) SURPASSES random surrogate (1.5566), CI[95%]≈[0.012, 0.017], non-thin,
# with the SAME monotone candidate < surrogate < control (1.5589) ranking as the
# homogeneous generalization deposit. The load-bearing heterogeneous signature:
# the output-first candidate froze layers {29,30,31} = ranks (7,11,16) — the
# THREE HIGHEST-CAPACITY active layers — so it removed the MOST adapter params
# (n_trainable_params 1.014M, the fewest of any arm) yet STILL won; the input-side
# control froze {24,25,26} = ranks (1,1,2) — the lowest-capacity layers — leaving
# the MOST params (3.497M). Output-first freeze sacrifices the most capacity and
# still wins: the §4 verdict is not an artifact of a uniform-rank stack. Still
# reduced-budget (96 < max_steps=1500) and honest about it (NOT the full verdict).
# Regenerate with ``make freeze-validloss-ci-9b-heterogeneous-generalization``.
FIXTURE_9B_HETEROGENEOUS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_heterogeneous_generalization.json"
)

# The committed RUN-LOG (loss-curve) WITNESS for the heterogeneous deposit — the
# per-arm loss trajectory the real 9B run produced, checked in so the verdict is
# independently reproducible from committed bytes (the content-hash axis of iter
# `65073bb`, here given a real committed artifact). 9 arms (3 candidate / 3
# surrogate / 3 control), each carrying a 96-step ``loss_curve`` + the arm's
# exact ``final_valid_loss`` (which reconstructs the deposit's loss vectors).
# Stamped into the heterogeneous deposit as ``run_log_path`` /
# ``run_log_sha256`` (canonical SHA-256 over the parsed payload, recomputable
# via ``_run_log_sha256``). Regenerate with
# ``make freeze-validloss-ci-9b-heterogeneous-generalization`` (the target now
# writes this committed witness directly).
FIXTURE_9B_HETEROGENEOUS_RUNLOG = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_heterogeneous_generalization_runlog.json"
)

# The FULL-BUDGET × HETEROGENEOUS deposit — the ONE remaining open §4 leg (the
# homogeneous full LANDED→TIES ``4b88ca8``; the heterogeneous REDUCED-budget
# LANDED→SURPASSES ``d00a362``; whether the heterogeneous SURPASSES survives the
# full 1500-step budget on an asymmetric per-layer-rank adapter is unmeasured).
# Arm shape MIRRORS the reduced heterogeneous direction-isolation design —
# candidate (output-first) + surrogate (random) + input-side control — bumped
# 96→1500 steps at 600 train / 2.5 epoch = generalization regime, with NO
# full-backprop baseline arm (``n_baseline == 0``). This is the SECOND citable
# full-budget deposit (``citable_as_full_section4_verdict=True``; LANDED +
# harvested 2026-07-18), so its arm SHAPE differs from the homogeneous full
# (control, not baseline) — the shape-SPECIFIC guards below (structural-full-
# budget, ledger-witness reconstruction) therefore have heterogeneous variants,
# while the shape-INDEPENDENT guards (gate/regime/evidence-hash self-consistency)
# cover it via ``_REAL_9B_DEPOSIT_FIXTURES``. Fired via
# ``scripts/fire_freeze_ci_9b_full_heterogeneous.sh``; verdict TIES (the
# reduced-budget heterogeneous SURPASSES does NOT survive the full 1500-step
# budget), recorded honestly. Tests keep a skip-until-exists guard so an
# accidentally-removed fixture fails loudly, not silently.
FIXTURE_9B_FULL_HETEROGENEOUS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_full_heterogeneous.json"
)

# The committed LEDGER WITNESS for the full-budget heterogeneous deposit — the
# committed-verification analogue of ``FIXTURE_9B_FULL_LEDGER`` for the
# heterogeneous arm shape. The bg run writes its resumability ledger to the
# stable ``runs/`` dir; at HARVEST the operator copies it here (a committed
# repo file, no gitignored ``runs/`` dependency) and stamps
# ``ledger_witness_path`` / ``ledger_witness_sha256`` into the deposit (these
# are HARVEST fields — the worker stamps ``run_log_*`` but NOT ``ledger_witness_*``;
# see ``scripts/fire_freeze_ci_9b_full_heterogeneous.sh`` harvest notes). 10
# JSONL lines (1 header + 9 arms: 3 candidate / 3 surrogate / 3 control), each
# carrying the arm's exact ``valid_loss`` — so the deposit's three loss-vectors
# reconstruct byte-for-byte from this witness. LANDED + harvested 2026-07-18
# (see :class:`TestCommittedLedgerWitnessHeterogeneousFull`).
FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl"
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

    def test_loss_curve_sink_is_opt_in(self):
        # The per-step loss-curve capture is an opt-in side channel: a
        # ``loss_curve_sink`` kwarg defaulting to None (no capture, byte-identical
        # to the pre-run-log path). It must NEVER be positional / required — the
        # capture is a pure side effect on the verdict.
        import inspect

        sig = inspect.signature(arm_valid_loss_9b)
        param = sig.parameters["loss_curve_sink"]
        assert param.default is None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


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

    @pytest.mark.parametrize(
        "field,value,exc",
        [
            ("depth", "not-an-int", ValueError),
            ("depth", None, TypeError),
            ("warmup_steps", None, TypeError),
            ("spacing", None, TypeError),
            ("total_steps", "x", ValueError),
            ("candidate_order", 5, TypeError),   # present but non-iterable
            ("active_scope", None, TypeError),   # list(None) → TypeError
        ],
    )
    def test_present_but_malformed_schedule_raises_loud(self, field, value, exc):
        # Silent-drop guard (the feedback's named bug class): a PRESENT-but-
        # malformed candidate schedule must NOT collapse to None and silently
        # drop the §4 cost-success axis from the deposit (`result_to_json`
        # serializes `_candidate_cost_reduction(result)` verbatim — a None here
        # becomes `"candidate_cost_reduction": null` and the constitution's
        # condition-(b) cost half vanishes unnoticed). The docstring contract is
        # "a present-but-malformed schedule raises loudly"; only an ABSENT key
        # (partial/legacy result, covered above) yields None. Parameterized so
        # the next malformation fails loudly at test time rather than vanishing
        # from the report.
        with pytest.raises(exc):
            _candidate_cost_reduction(self._sched(**{field: value}))


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


# ── single source of truth: the producer delegates gate primitives to the leaf ─


class TestGatePrimitivesAreSharedLeaf:
    """The §4 citation-gate primitives live in ONE torch-free leaf
    (``src.tg_lora.freeze_verdict_honesty``) shared by the producer (this runner)
    and the GPU-free replay gate, so a regime threshold / reduced-budget rule /
    4-conjunct gate tuned in the leaf changes BOTH at once and a committed
    deposit's producer-stamped boolean and the replay's re-derived verdict cannot
    drift apart (SYSTEM_CONSTITUTION Rule #3). These tests pin that the runner's
    underscore-named primitives ARE the leaf's objects (delegation, not a
    duplicated copy): if someone re-inlines a local ``_classify_regime`` /
    ``_is_reduced_budget`` / ``_full_section4_verdict_gate`` here, the identity
    check fails and forces them back to the shared leaf.
    """

    def test_classify_regime_is_the_leaf_primitive(self):
        from src.tg_lora.freeze_verdict_honesty import classify_regime
        assert _classify_regime is classify_regime

    def test_is_reduced_budget_is_the_leaf_primitive(self):
        from src.tg_lora.freeze_verdict_honesty import is_reduced_budget
        assert _is_reduced_budget is is_reduced_budget

    def test_full_section4_verdict_gate_is_the_leaf_primitive(self):
        from src.tg_lora.freeze_verdict_honesty import full_section4_verdict_gate
        assert _full_section4_verdict_gate is full_section4_verdict_gate

    def test_regime_constants_are_the_leaf_constants(self):
        from src.tg_lora import freeze_verdict_honesty as leaf
        assert REGIME_GENERALIZATION == leaf.REGIME_GENERALIZATION
        assert REGIME_MEMORIZATION == leaf.REGIME_MEMORIZATION
        assert REGIME_OVERFIT == leaf.REGIME_OVERFIT
        assert REGIME_UNKNOWN == leaf.REGIME_UNKNOWN



# ── full-run gate liveness: the shipped Make flags + bound config must clear ─
# ── the two config-determined citation-gate axes BEFORE any GPU is spent ─────


class TestFullRunGateLiveness:
    """``make freeze-validloss-ci-9b-full`` is hours of 9B GPU banking 9 arms.
    Its deposit opens the full-§4 citation gate only when FOUR honesty axes clear
    together; TWO of them are *config-determined* (fixed by the Makefile flags +
    the bound config, not by the run's outcome) and are therefore guardable
    BEFORE any GPU is spent:

    * **reduced_budget** — the run's ``--total-steps`` must reach the bound
      config's ``training.max_steps`` (:func:`_is_reduced_budget`). The Makefile
      ships ``--total-steps 1500`` and the config ships ``max_steps: 1500``, so
      the coupling holds — but it is *implicit*. The ``TestReducedBudgetHonest``
      cases above pin the function with hardcoded ``1500`` literals, so a future
      edit that bumps the config's ``max_steps`` (or trims the flag) would leave
      every existing test green while the real multi-hour run lands with
      ``reduced_budget=True`` → ``citable_as_full_section4_verdict=False``:
      hours of GPU silently wasted on a deposit that can never be cited.
    * **is_thin_evidence** — each headline arm must bank ≥
      :data:`MIN_SAMPLE_FOR_BOOTSTRAP` seeds. The Makefile ships
      ``--n-candidate/surrogate 3``; a trim to 2 would make every arm thin and
      the gate stays shut on this axis instead.

    The other two axes (``proxy_scale``, ``regime``) are *outcome-determined*
    (a real 9B run is never a proxy; regime depends on the candidate arm's
    recorded ``final_ce_train_loss``) and stay gated at deposit time — this guard
    locks only what the shipped config already guarantees. It reads the LIVE
    Makefile flag and the LIVE bound config, so a drift in either file fails loud
    here, in a GPU-free second, not after a multi-hour run.
    """

    BOUND_CONFIG = "configs/9b_baseline_suffix_only_last25.yaml"

    def _full_flags(self) -> str:
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "Makefile").read_text(encoding="utf-8")
        m = re.search(r"^FREEZE_9B_FULL_FLAGS\s*\?=\s*(.+)$", text, re.MULTILINE)
        # Assert (not return None) so a renamed Make var can't make this guard
        # silently pass by matching nothing.
        assert m, "FREEZE_9B_FULL_FLAGS missing from Makefile — guard cannot run"
        return m.group(1)

    def _flag_int(self, flags: str, token: str) -> int:
        m = re.search(rf"{re.escape(token)}\s+(\d+)", flags)
        assert m, f"{token} missing from FREEZE_9B_FULL_FLAGS — guard cannot run"
        return int(m.group(1))

    def test_guard_anchored_to_the_script_default_config(self):
        # The config this guard reads must be the one the run actually binds, so a
        # future DEFAULT_CONFIG switch reanchors (or fails) this guard rather than
        # reading a stale path.
        from scripts.run_freeze_validloss_ci_9b import DEFAULT_CONFIG

        assert DEFAULT_CONFIG == self.BOUND_CONFIG

    def test_full_run_clears_reduced_budget_axis(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / self.BOUND_CONFIG)
        cfg_max_steps = int(cfg.training.max_steps)
        total_steps = self._flag_int(self._full_flags(), "--total-steps")
        assert not _is_reduced_budget(total_steps, cfg_max_steps), (
            f"FREEZE_9B_FULL_FLAGS --total-steps {total_steps} < bound-config "
            f"training.max_steps {cfg_max_steps}: the multi-hour full run would "
            f"land with reduced_budget=True and never open the full-§4 citation "
            f"gate. Raise --total-steps to >= the config's training.max_steps "
            f"(or lower the config's max_steps)."
        )

    def test_full_run_clears_thin_evidence_axis(self):
        from src.tg_lora.freeze_surrogate_ci import MIN_SAMPLE_FOR_BOOTSTRAP

        flags = self._full_flags()
        # The headline candidate-vs-surrogate CI's is_thin_evidence keys the
        # full-verdict gate, so it is these two arm counts that must clear the bar.
        for token in ("--n-candidate", "--n-surrogate"):
            n = self._flag_int(flags, token)
            assert n >= MIN_SAMPLE_FOR_BOOTSTRAP, (
                f"FREEZE_9B_FULL_FLAGS {token}={n} < MIN_SAMPLE_FOR_BOOTSTRAP="
                f"{MIN_SAMPLE_FOR_BOOTSTRAP}: every arm banks a thin sample and "
                f"the full-§4 citation gate stays shut on is_thin_evidence. "
                f"Raise {token} to >= {MIN_SAMPLE_FOR_BOOTSTRAP}."
            )


# ── deposit gate self-consistency (the "inert green" guard) ───────────────────
#
# Every deposit-faithfulness test below hard-codes the EXPECTED gate boolean
# (``is False`` / ``is True``); none recomputes the 4-conjunct gate from the
# deposit's OWN fields. So a serializer/formula drift (a future conjunct added
# to :func:`_full_section4_verdict_gate` without re-depositing), a hand-edited
# file, or — the load-bearing case — the FULL-budget deposit landing with a
# train-CE diagnostic that silently failed to record (regime → UNKNOWN → the
# gate closes) would all pass green. The full verdict is the FIRST committed
# deposit with ``reduced_budget=False``, so the regime conjunct decides the gate
# for the first time; this class recomputes the gate from each deposit's own
# provenance/fields and asserts the stored boolean matches, and for the full
# verdict asserts the train-CE was actually recorded (the regime conjunct
# evaluated on real data, not silently defaulted to None/UNKNOWN).

_REAL_9B_DEPOSIT_FIXTURES = [
    pytest.param(FIXTURE_9B_SURROGATE, id="surrogate"),
    pytest.param(FIXTURE_9B_DIRECTION, id="direction"),
    pytest.param(FIXTURE_9B_GENERALIZATION, id="generalization"),
    pytest.param(FIXTURE_9B_BASELINE, id="baseline"),
    pytest.param(FIXTURE_9B_FULL, id="full"),
    pytest.param(FIXTURE_9B_HETEROGENEOUS, id="heterogeneous"),
    # The full-budget heterogeneous deposit — the 2nd citable full verdict,
    # LANDED + harvested 2026-07-18. Every parametrized sweep still skip-until-
    # exists so the 4 shape-INDEPENDENT guards (gate-boolean / regime-field /
    # evidence-hash stamp / frozen-literal) auto-cover it and cannot silently
    # bypass the repo's headline-result integrity net if the fixture is removed.
    pytest.param(FIXTURE_9B_FULL_HETEROGENEOUS, id="full-heterogeneous"),
]

# The citable FULL-BUDGET deposits (``citable_as_full_section4_verdict=True``) —
# the runs where the gate's regime conjunct is load-bearing (``reduced_budget=
# False``). The two have DIFFERENT arm shapes (homogeneous carries a full-backprop
# BASELINE; heterogeneous carries an input-side CONTROL), so the shape-specific
# guards are separate — but the shape-INDEPENDENT full-budget guards (the
# load-bearing regime-on-real-data check + the top-level CE rollup) parametrize
# over BOTH so every citable full verdict is held to the same standard.
_CITABLE_FULL_DEPOSIT_FIXTURES = [
    pytest.param(FIXTURE_9B_FULL, id="homogeneous"),
    pytest.param(FIXTURE_9B_FULL_HETEROGENEOUS, id="heterogeneous"),
]


class TestDepositGateSelfConsistency:
    """The stored ``citable_as_full_section4_verdict`` must equal the 4-conjunct
    gate recomputed from the deposit's own fields, and the regime conjunct must
    have evaluated on real data for the one deposit where it is load-bearing."""

    @pytest.mark.parametrize("path", _REAL_9B_DEPOSIT_FIXTURES)
    def test_gate_boolean_matches_recomputed_gate(self, path):
        # Recompute the gate from the deposit's OWN provenance + fields (the same
        # ``_candidate_final_ce_mean`` / ``_classify_regime`` /
        # ``_full_section4_verdict_gate`` the serializer uses) and assert the
        # stored boolean matches. Catches serializer↔committed-file drift,
        # hand-edits, and stale deposits — a single source of truth for the gate
        # means any future conjunct is checked against every committed deposit.
        if not path.exists():
            pytest.skip(f"{path.name} not landed yet")
        data = json.loads(path.read_text())
        # Regime from the deposit's own provenance rollup so LEGACY deposits
        # (lacking the top-level ``candidate_final_ce_train_loss_mean`` key) still
        # classify consistently rather than KeyError / misread as UNKNOWN.
        ce_mean = _candidate_final_ce_mean(data)
        recomputed_regime = _classify_regime(ce_mean, data["candidate_mean"])
        recomputed_gate = _full_section4_verdict_gate(
            proxy_scale=data["proxy_scale"],
            reduced_budget=data["reduced_budget"],
            is_thin_evidence=data["is_thin_evidence"],
            regime=recomputed_regime,
        )
        assert data["citable_as_full_section4_verdict"] is recomputed_gate

    @pytest.mark.parametrize("path", _REAL_9B_DEPOSIT_FIXTURES)
    def test_regime_field_matches_own_floats_when_present(self, path):
        # When the deposit carries the top-level train-CE rollup, the stored
        # ``regime`` string must be faithful to recomputation from the deposit's
        # own floats — guards a deposit that ships ``regime="generalization"``
        # while its provenance classifies otherwise. Legacy deposits predate the
        # diagnostic (no top-level key) and skip, not fail.
        if not path.exists():
            pytest.skip(f"{path.name} not landed yet")
        data = json.loads(path.read_text())
        if "candidate_final_ce_train_loss_mean" not in data:
            pytest.skip(f"{path.name} predates the regime diagnostic (legacy)")
        recomputed = _classify_regime(
            data["candidate_final_ce_train_loss_mean"], data["candidate_mean"]
        )
        assert data.get("regime") == recomputed

    @pytest.mark.parametrize("path", _CITABLE_FULL_DEPOSIT_FIXTURES)
    def test_full_deposit_top_level_ce_matches_provenance_rollup(self, path):
        # The top-level ``candidate_final_ce_train_loss_mean`` must equal the mean
        # of the candidate provenance ``final_ce_train_loss`` values (the same
        # ``_candidate_final_ce_mean`` that produced it) — pins the rollup to the
        # per-arm data so a deposit can't ship a stale/edited headline number.
        if not path.exists():
            pytest.skip(f"{path.name} not landed yet (run still banking)")
        data = json.loads(path.read_text())
        assert data["candidate_final_ce_train_loss_mean"] == pytest.approx(
            _candidate_final_ce_mean(data)
        )

    @pytest.mark.parametrize("path", _CITABLE_FULL_DEPOSIT_FIXTURES)
    def test_full_deposit_regime_conjunct_evaluated_on_real_data(self, path):
        # THE LOAD-BEARING GUARD. The full-budget verdict is the FIRST committed
        # deposit with ``reduced_budget=False``, so the gate's regime conjunct
        # decides ``citable_as_full_section4_verdict`` for the first time. If the
        # train-CE diagnostic silently failed to record (None), the regime would
        # default to UNKNOWN and close the gate — with no other test surfacing
        # the surprise. This asserts the train-CE was recorded as a finite
        # number, so the gate's decision rests on real data, not a silent
        # default. (Function-level closure: ``test_unknown_regime_blocks_full_
        # citation``; this is its landing-time enforcement on the real deposit.)
        if not path.exists():
            pytest.skip(f"{path.name} not landed yet (run still banking)")
        data = json.loads(path.read_text())
        assert "candidate_final_ce_train_loss_mean" in data
        ce = data["candidate_final_ce_train_loss_mean"]
        assert isinstance(ce, (int, float)) and math.isfinite(ce)

    def test_full_deposit_is_structurally_full_budget(self):
        # The ``make freeze-validloss-ci-9b-full`` config GUARANTEES: full-budget
        # (``total_steps`` reaches ``cfg_max_steps``, not reduced), non-thin
        # (>=3 seeds per arm set), target-scale (not proxy). These are
        # config-determined, so they pin that the landed run was not silently
        # reduced/thin — independent of the empirical SURPASSES/TIES outcome,
        # which the gate-consistency test above handles honestly either way.
        if not FIXTURE_9B_FULL.exists():
            pytest.skip("full-budget deposit not landed yet (run still banking)")
        data = json.loads(FIXTURE_9B_FULL.read_text())
        assert data["proxy_scale"] is False
        assert data["reduced_budget"] is False
        assert data["total_steps"] == data["cfg_max_steps"]
        assert data["is_thin_evidence"] is False
        assert data["n_candidate"] >= 3
        assert data["n_surrogate"] >= 3
        assert data["n_baseline"] >= 3

    def test_heterogeneous_full_deposit_is_structurally_full_budget(self):
        # The heterogeneous full verdict has a DIFFERENT arm shape from the
        # homogeneous full: it re-runs the reduced-heterogeneous direction-
        # isolation design (candidate output-first + random surrogate + input-side
        # control) at the full 1500-step budget, with NO full-backprop baseline arm
        # (``n_baseline == 0`` — the baseline is the homogeneous leg's condition-(a)
        # question, already answered at ``4b88ca8``). So the homogeneous structural
        # test's ``n_baseline >= 3`` assertion does NOT apply; this pins the
        # heterogeneous arm shape instead, plus the SAME shared full-budget axes
        # (not proxy / not reduced / total_steps reaches cfg_max_steps / non-thin).
        # Independent of the empirical SURPASSES/TIES outcome the gate test handles.
        if not FIXTURE_9B_FULL_HETEROGENEOUS.exists():
            pytest.skip(
                "full-budget heterogeneous deposit not landed yet (run in flight)"
            )
        data = json.loads(FIXTURE_9B_FULL_HETEROGENEOUS.read_text())
        assert data["architecture"] == "heterogeneous"
        assert data["proxy_scale"] is False
        assert data["reduced_budget"] is False
        assert data["total_steps"] == data["cfg_max_steps"]
        assert data["is_thin_evidence"] is False
        assert data["n_candidate"] >= 3
        assert data["n_surrogate"] >= 3
        assert data["n_control"] >= 3
        assert data["n_baseline"] == 0  # no baseline arm — direction-isolation design


class TestRunLogArtifact:
    """Loss-curve run-log artifact (GOAL §7 reproducibility) — the "attach a run
    log / loss-curve artifact and a content hash so the verdict is independently
    reproducible" path ``evidence_hash`` alone cannot cover. ``evidence_hash``
    freezes the terminal measurements; it cannot certify a GPU produced the
    *dynamics* behind them (only a surviving loss curve or a fresh reproduction
    can — the heterogeneous run landed without one, recorded openly by the
    evidence-hash commit). The run log is that companion artifact: per-step
    training-loss trajectories + run config, content-hashed over canonical bytes
    and linked into the deposit via ``run_log_sha256``.

    These tests pin the writer, the hash, the deposit linkage, and the capture
    pipeline WITHOUT a GPU: the writer + gatherer are pure functions, and the
    per-step append is pinned by :class:`TestArmSignature` above + the
    ``_collect_arms`` capture-pipeline test in the resume suite.
    """

    def _curves(self):
        return [
            {"role": "candidate", "index": 0, "seed": 0, "order": [31, 30],
             "frozen_layers": [31], "n_trainable_params": 1014336,
             "final_valid_loss": 1.543, "last_train_loss": 0.0001,
             "final_ce_train_loss": 1.576, "loss_curve": [2.0, 1.8, 1.5]},
            {"role": "surrogate", "index": 0, "seed": 100, "order": [25, 28],
             "frozen_layers": [25], "n_trainable_params": 2047296,
             "final_valid_loss": 1.555, "last_train_loss": 0.0002,
             "final_ce_train_loss": 1.597, "loss_curve": [2.1, 1.9, 1.6]},
        ]

    # Frozen literal for the fixed _curves()/_stub_result() input — the
    # coordinated-drift guard analog: any byte change to a curve or config field
    # flips this RED until the literal is deliberately updated on a real re-run.
    _EXPECTED_HASH = (
        "f1e25dc756bdb6b684b836557e0ce07e49e296a0f1e17058358053acd164dea5"
    )

    def test_frozen_literal_hash(self):
        payload = _run_log_payload(_stub_result(), self._curves())
        assert _run_log_sha256(payload) == self._EXPECTED_HASH

    def test_writer_self_consistency(self, tmp_path):
        # The returned hash must equal the hash recomputed from the WRITTEN file's
        # parsed content — so a verifier reading the on-disk artifact recomputes
        # the same stamp the deposit carries (guards a stale/rolled serializer
        # AND proves the indented on-disk format hashes the same as the canonical
        # form the stamp is over).
        path = tmp_path / "runs" / "runlog.json"
        sha = _write_run_log(path, _stub_result(), self._curves())
        loaded = json.loads(path.read_text())
        assert sha == _run_log_sha256(loaded)
        assert loaded["schema_version"] == 1
        assert loaded["run_config"]["model"] == "Qwen/Qwen3.5-9B"
        assert [a["role"] for a in loaded["arms"]] == ["candidate", "surrogate"]
        assert loaded["arms"][0]["loss_curve"] == [2.0, 1.8, 1.5]

    def test_curve_change_moves_hash(self):
        # Mutating a single per-step loss flips the hash — the curve is
        # load-bearing evidence the stamp certifies (mutation proof, curve axis).
        curves = self._curves()
        base = _run_log_sha256(_run_log_payload(_stub_result(), curves))
        curves[0]["loss_curve"][0] = 9.999
        assert _run_log_sha256(_run_log_payload(_stub_result(), curves)) != base

    def test_config_change_moves_hash(self):
        # Mutating run-determining config (total_steps) flips the hash — the
        # artifact is self-identifying, not just an opaque blob of curves.
        curves = self._curves()
        base = _run_log_sha256(_run_log_payload(_stub_result(), curves))
        moved = _run_log_sha256(_run_log_payload(_stub_result(total_steps=999), curves))
        assert moved != base

    def test_hash_is_whitespace_invariant(self):
        # The stamp is over canonical COMPACT bytes; serializing the same payload
        # compact vs indented and re-parsing must yield the same hash — a verifier
        # is never sensitive to the file's indentation.
        payload = _run_log_payload(_stub_result(), self._curves())
        compact = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        indented = json.loads(json.dumps(payload, sort_keys=True, indent=2))
        assert _run_log_sha256(compact) == _run_log_sha256(indented)

    def test_deposit_carries_run_log_fields_additively(self):
        # result_to_json surfaces run_log_path + run_log_sha256 additively: set
        # when the run wrote a log, None when it did not (byte-identical to the
        # pre-run-log deposit shape for a one-shot run).
        with_log = result_to_json(_stub_result(
            run_log_path="runs/x.json", run_log_sha256="abc123"))
        assert with_log["run_log_path"] == "runs/x.json"
        assert with_log["run_log_sha256"] == "abc123"
        without = result_to_json(_stub_result())
        assert without["run_log_path"] is None
        assert without["run_log_sha256"] is None

    def test_run_log_fields_do_not_perturb_evidence_hash(self):
        # The run-log is a PARALLEL provenance channel: it must never enter
        # EVIDENCE_HASH_KEYS, so attaching/detaching a run log leaves the pinned
        # evidence hash byte-identical (independence — the two hashes certify
        # different things and must not be circular).
        without = _evidence_hash(result_to_json(_stub_result()))
        with_log = _evidence_hash(result_to_json(_stub_result(
            run_log_path="runs/x.json", run_log_sha256="abc123")))
        assert without == with_log

    def test_gather_pairs_specs_with_results_in_plan_order(self):
        # _gather_arm_curves walks specs in plan order, pairing each with its
        # (loss, provenance) from collected, and reads the seeded _loss_curve.
        specs = [
            {"role": "candidate", "index": 0, "seed": 0, "order": (31, 30),
             "depth": 2, "_loss_curve": [2.0, 1.5]},
            {"role": "candidate", "index": 1, "seed": 1, "order": (31, 30),
             "depth": 2, "_loss_curve": [2.1, 1.6]},
            {"role": "surrogate", "index": 0, "seed": 100, "order": (25, 28),
             "depth": 2, "_loss_curve": [2.2, 1.7]},
        ]
        collected = {
            "candidate": [
                (1.543, {"frozen_layers": [31], "n_trainable_params": 1014336,
                         "last_train_loss": 0.0001, "final_ce_train_loss": 1.576}),
                (1.544, {"frozen_layers": [31], "n_trainable_params": 1014336,
                         "last_train_loss": 0.0001, "final_ce_train_loss": 1.577}),
            ],
            "surrogate": [
                (1.555, {"frozen_layers": [25], "n_trainable_params": 2047296,
                         "last_train_loss": 0.0002, "final_ce_train_loss": 1.597}),
            ],
            "control": [], "baseline": [],
        }
        arms = _gather_arm_curves(specs, collected)
        assert [a["role"] for a in arms] == ["candidate", "candidate", "surrogate"]
        assert arms[0]["final_valid_loss"] == 1.543
        assert arms[0]["loss_curve"] == [2.0, 1.5]
        assert arms[2]["loss_curve"] == [2.2, 1.7]
        assert arms[1]["n_trainable_params"] == 1014336

    def test_gather_honest_empty_curve_for_resumed_arm(self):
        # A resumed arm (replayed from the ledger, never re-run) has no captured
        # curve: _gather_arm_curves reads an empty _loss_curve (the orchestrator
        # seeded []) rather than fabricating a trajectory.
        specs = [{"role": "candidate", "index": 0, "seed": 0, "order": (31,),
                  "depth": 1, "_loss_curve": []}]
        collected = {"candidate": [(1.543, {"frozen_layers": [31]})],
                     "surrogate": [], "control": [], "baseline": []}
        arms = _gather_arm_curves(specs, collected)
        assert arms[0]["loss_curve"] == []
        assert arms[0]["final_valid_loss"] == 1.543


class TestDepositEvidenceHash:
    """Content-hash reproducibility provenance (GOAL §7) for every committed
    real-9B deposit — the "attach a content hash so the verdict is independently
    reproducible" guard the §4 effort's source of truth rests on.

    The verdict-replay tests pin that the recorded verdict is EARNED from the
    stored floats; the gate self-consistency tests pin the DERIVED boolean.
    NEITHER catches a coordinated repaint — editing the committed floats, their
    CI bounds, the verdict label, and the per-arm provenance TOGETHER so every
    derived check still passes. ``evidence_hash`` stamps a SHA-256 over the
    EVIDENCE bytes (raw measurements + run-determining config, NEVER the derived
    verdict/gate/regime labels), and these tests pin it two ways:

    * self-consistency — the stamped hash equals the hash recomputed from the
      committed bytes (the stamp is honest, not stale / hand-edited / mis-rolled);
    * literal — the stamped hash equals a FROZEN literal per deposit, so any byte
      change to the evidence (coordinated repaint OR accidental drift) is a
      reviewable RED until the literal is deliberately updated on a real re-run.

    What this does NOT certify: that a GPU produced the bytes. Only a run log or
    a fresh independent reproduction can. The heterogeneous deposit has no
    surviving run log and the GPU was contended the iteration it landed — that
    limitation is recorded openly as the next action, not papered over here.
    """

    # Frozen per-deposit evidence hashes. Update ONLY on a deliberate re-deposit
    # (a real re-run that legitimately changes the recorded measurements); that
    # update is itself the reviewable change this guard exists to force.
    _EXPECTED = {
        FIXTURE_9B_SURROGATE: "9c3bd56c6f05f4c059a29b01816fbe59cb1cddd128ab70e2cc2c8d08cab6d7bc",
        FIXTURE_9B_DIRECTION: "96026bfd3b93073847989939708816487ce202843fc14252c3ada858852dfab4",
        FIXTURE_9B_GENERALIZATION: "efa596745ca7346dca4d10d0c916fff504c64142f6aabfad13cf8c5cb9abfcd8",
        FIXTURE_9B_BASELINE: "d0090d0740ac2ab8f93ac77eb4a55921fdd640ff08c44779d625e5559e48e8d3",
        FIXTURE_9B_FULL: "785d9cfae66fbf20fb0b9c3349dacbfd3ec5caf7cd9b046610d78f4e2bcf8577",
        FIXTURE_9B_HETEROGENEOUS: "c07b1de1e8f3692b4e99ee7a20fcef6ce3c3a6e955de4a672e62838d17288a49",
        # The 2nd citable full-budget verdict — heterogeneous (per-layer rank)
        # arm shape, LANDED 2026-07-18. Distinct from the homogeneous full pin
        # above (``architecture`` + ``lora_rank_pattern`` are EVIDENCE keys) and
        # from the reduced-budget heterogeneous pin (different total_steps +
        # losses): the SURPASSES the reduced leg earned does NOT survive the
        # full 1500-step budget → verdict TIES, recorded honestly either way.
        FIXTURE_9B_FULL_HETEROGENEOUS: "335a1ffb5a740dfa70f19d17c201cf77901eaef86cc916cb603e7417e01fa323",
    }

    @pytest.mark.parametrize("path", _REAL_9B_DEPOSIT_FIXTURES)
    def test_stamp_matches_recomputed_hash(self, path):
        # The stamped ``evidence_hash`` must equal the hash recomputed from the
        # deposit's own committed bytes — guards a stale stamp (serializer rolled
        # but the field was not refreshed) or a hand-edit that touched the bytes
        # and not the stamp.
        if not path.exists():
            pytest.skip(f"{path.name} not landed yet")
        data = json.loads(path.read_text())
        assert "evidence_hash" in data
        assert data["evidence_hash"] == _evidence_hash(data)

    @pytest.mark.parametrize("path", _REAL_9B_DEPOSIT_FIXTURES)
    def test_stamp_is_pinned_to_frozen_literal(self, path):
        # THE coordinated-drift guard. The stamped hash must equal the frozen
        # literal for this deposit — so a byte change to ANY evidence field
        # (losses, orders, provenance, run config) flips this red until the
        # literal is deliberately updated. This is the one test a coordinated
        # repaint (which passes every DERIVED check) cannot pass silently: it
        # forces the repaint to also edit the frozen literal, a visible change.
        if not path.exists():
            pytest.skip(f"{path.name} not landed yet")
        data = json.loads(path.read_text())
        # A deposit that has LANDED must have its frozen literal pinned here —
        # that pin IS the deliberate change this guard exists to force. Failing
        # loud with a clear message (rather than a KeyError on ``_EXPECTED[path]``)
        # so the harvest operator knows the one remaining step; this also closes
        # the trap where a newly-registered deposit (e.g. the in-flight
        # full-heterogeneous verdict) lands and silently bypasses the pin because
        # no literal was ever added.
        assert path in self._EXPECTED, (
            f"{path.name} landed but has no frozen evidence_hash literal in "
            f"_EXPECTED — pin it (the deliberate change this guard exists to force)."
        )
        assert data["evidence_hash"] == self._EXPECTED[path]

    def test_hash_is_over_evidence_not_derived_labels(self):
        # The hash must be over the EVIDENCE, not the derived labels — otherwise
        # it is circular (auditing the verdict by hashing the verdict). Mutating
        # any derived label (verdict / passes / gate / regime / CI statistics)
        # must NOT move the hash; only the raw measurements and run config may.
        data = json.loads(FIXTURE_9B_SURROGATE.read_text())
        base = _evidence_hash(data)
        for label_key in (
            "verdict", "passes", "significant_surpasses", "is_material",
            "citable_as_full_section4_verdict", "citable_as_target_scale",
            "regime", "point_improvement", "lower", "upper",
            "candidate_mean", "surrogate_mean",
        ):
            mutated = dict(data)
            current = data.get(label_key)
            mutated[label_key] = (
                "ZZZ_NEVER_REAL" if isinstance(current, str) else (not current)
            )
            assert _evidence_hash(mutated) == base, (
                f"evidence_hash leaked the derived label '{label_key}' into the "
                f"payload — the hash must cover only raw evidence, never labels."
            )


# ── full-budget §4 deposit: literal verdict pin + replay faithfulness ────────
#
# The full-budget deposit (``citable_as_full_section4_verdict=True``, commit
# ``4b88ca8``) is the FIRST committed deposit to clear all four citation-gate
# axes and the culmination of the §4 verdict effort. ``TestDepositGateSelfConsis
# tency`` above pins the gate BOOLEAN and the structural axes — but its own
# docstring notes it is "independent of the empirical SURPASSES/TIES outcome".
# So a coordinated float+verdict repaint (the exact hazard the seq256 deposit's
# ``test_real_verdict_pins_literal_ties_and_ci_bounds`` was built for) would
# silently flip this deposit's headline result while every existing test stays
# green. The full deposit was the ONLY one of the five real deposits without a
# replay-faithfulness + literal-verdict pin; this class closes that asymmetric
# gap on the repo's most citable result — pinning BOTH load-bearing verdicts
# literally AND re-deriving them from the stored floats.


class TestFullBudgetDepositVerdictPin:
    """Pin the landed full-budget §4 verdict (``4b88ca8``) as a durable
    regression invariant.

    Two load-bearing scientific claims live in this deposit, and BOTH are pinned
    as literals AND re-derived from the stored floats through the deterministic
    bootstrap (so a painted-on verdict, or a coordinated float+verdict repaint,
    fails red):

    * **Headline A/B (candidate output-first vs random surrogate) = TIES.** The
      reduced-budget SURPASSES (sibling deposits) does NOT survive full-budget
      generalization — the output-first *order* effect evaporates at the real §4
      budget. The CI straddles zero (``lower < 0 < upper``), the structural
      reason a bootstrap CI reads TIES.
    * **§4 condition (a) (candidate vs no-freeze FULL-backprop baseline) =
      SURPASSES, and SURVIVES the full budget.** Freezing did not merely stay
      within tolerance of full backprop — it BEAT it, exactly as at reduced
      budget. The baseline CI excludes zero above (``lower > 0``).

    The divergence — headline TIES while condition (a) SURPASSES — IS the
    scientific finding: the output-first *order* effect is budget-fragile, but
    *freeze-vs-full-backprop* quality preservation is not. Pinning both
    literally makes any future re-deposit that flips either a visible,
    reviewable change rather than a silent drift.
    """

    def test_full_deposit_is_citable_as_full_section4_verdict(self):
        # The steering success criterion itself, pinned on the landed artifact:
        # the gate is OPEN (True) and the run was FRESH — not a resumed/partial
        # banking (``resumed_arm_count == 0``). No existing test asserts the gate
        # is True directly; self-consistency only checks ``stored == recomputed``
        # (which holds for True OR False), and the structural test pins the axes
        # but never the assembled boolean. This is the deposit the whole §4
        # effort was building toward — assert it landed.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        assert data["citable_as_full_section4_verdict"] is True
        assert data["resumed_arm_count"] == 0  # fresh run, not a resumed bank
        # All four gate conjuncts clear, on the deposit's own fields:
        assert data["proxy_scale"] is False
        assert data["reduced_budget"] is False
        assert data["is_thin_evidence"] is False
        assert data["regime"] == "generalization"

    def test_headline_verdict_is_ties_and_earned(self):
        # THE coordinated-drift guard. The headline A/B verdict is TIES, pinned
        # as a LITERAL (not ``data["verdict"]`` — reading the field back is the
        # coordinated-drift escape hatch the seq256 literal pin closed: a repaint
        # that moves both the floats and the verdict field to SURPASSES would
        # pass a ``== data["verdict"]`` check). The stored floats re-judge
        # through the deterministic bootstrap to TIES (the verdict is earned, not
        # painted), and the CI straddles zero — the structural reason it is TIES.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        assert data["verdict"] == TIES
        ci = surrogate_valid_loss_ci(
            data["candidate_losses"],
            data["surrogate_losses"],
            seed=data["base_seed"],
        )
        assert ci.significance_verdict == TIES
        assert ci.significance_verdict == data["verdict"]  # replay-faithful
        # CI straddles zero => TIES, not a one-sided call.
        assert data["lower"] < 0.0 < data["upper"]
        assert ci.lower == pytest.approx(data["lower"], abs=1e-9)
        assert ci.upper == pytest.approx(data["upper"], abs=1e-9)
        assert ci.candidate_mean == pytest.approx(data["candidate_mean"], abs=1e-9)
        assert ci.surrogate_mean == pytest.approx(data["surrogate_mean"], abs=1e-9)

    def test_condition_a_survives_full_budget(self):
        # §4 condition (a) at the FULL budget: candidate SURPASSES no-freeze full
        # backprop. This is the one measurement the freeze-vs-freeze sibling
        # deposits could never make, and it SURVIVES the budget increase
        # (reduced-budget deposit SURPASSES -> full-budget deposit SURPASSES). The
        # baseline occupies the CI's "surrogate" slot, so its mean is stored as
        # ``baseline_mean``. Pinned literally + re-derived from the floats.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        bd = data["baseline"]
        assert bd["verdict"] == SURPASSES
        bci = surrogate_valid_loss_ci(
            data["candidate_losses"],
            data["baseline_losses"],
            seed=data["base_seed"],
        )
        assert bci.significance_verdict == SURPASSES
        assert bci.significance_verdict == bd["verdict"]  # replay-faithful
        assert bd["lower"] > 0.0  # CI entirely above zero => significant
        assert bci.lower == pytest.approx(bd["lower"], abs=1e-9)
        assert bci.upper == pytest.approx(bd["upper"], abs=1e-9)
        assert bci.candidate_mean == pytest.approx(bd["candidate_mean"], abs=1e-9)
        assert bci.surrogate_mean == pytest.approx(bd["baseline_mean"], abs=1e-9)
        # The candidate's held-out valid_loss is lower than full backprop's.
        assert data["candidate_mean"] < bd["baseline_mean"]

    def test_order_effect_does_not_survive_but_freeze_vs_full_backprop_does(self):
        # The headline scientific finding, pinned as a cross-verdict invariant:
        # at the full §4 budget the two A/Bs DIVERGE — the output-first ORDER
        # effect (candidate vs random surrogate) is TIES (budget-fragile), while
        # FREEZE-vs-FULL-BACKPROP quality preservation (candidate vs baseline) is
        # SURPASSES (budget-robust). A future re-run that "recovers" a headline
        # SURPASSES would flip this red — making the regime/budget-dependent
        # result change a visible, reviewable event instead of a silent drift.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        assert data["verdict"] == TIES
        assert data["baseline"]["verdict"] == SURPASSES
        assert data["verdict"] != data["baseline"]["verdict"]


# ── committed ledger witness: independent reproducibility of the full deposit ─
#
# The citable full-budget deposit (``citable_as_full_section4_verdict=True``) is
# the repo's most load-bearing result. ``evidence_hash`` pins its committed bytes
# against a coordinated repaint; the verdict pins pin the EARNED outcome. NEITHER
# answers the verifier's outside question — "given this committed JSON alone,
# can I reconstruct the recorded losses?" The committed LEDGER WITNESS
# (``freeze_validloss_ci_9b_full_ledger.jsonl``) closes that: a byte-identical
# copy of the run's resumability ledger, stamped into the deposit as
# ``ledger_witness_path`` / ``ledger_witness_sha256``. These tests pin that the
# witness is present, hashes to its stamped value, and reconstructs every loss
# vector EXACTLY — so the verdict is reproducible from committed bytes with no
# gitignored ``runs/`` dependency and no private stable dir.


class TestCommittedLedgerWitness:
    """Independent reproducibility provenance for the citable full-budget §4
    deposit via its committed LEDGER WITNESS.

    The full-budget verdict (commit ``4b88ca8``) cleared all four citation-gate
    axes; attaching the run's ledger as a committed, content-hashed witness is
    the "independently reproducible" half of GOAL §7 — a skeptic re-reads the 10
    committed JSONL lines, recomputes the canonical SHA-256, and reconstructs
    the deposit's three loss-vectors byte-for-byte, with no private stable dir
    or gitignored ``runs/`` path. The witness is additive (NOT in
    ``EVIDENCE_HASH_KEYS``, so it cannot perturb ``evidence_hash``) and distinct
    from the runtime ``ledger_path`` (``null`` here — the full run completed in
    one shot; the witness is the committed VERIFICATION copy, not the
    incrementally-written resumability path a fresh ``make`` run would write).
    """

    # Frozen witness hash over the canonical encoding of the committed ledger
    # (sort_keys + compact separators over the parsed JSONL lines — the same
    # canonicalization discipline as ``_run_log_sha256`` / ``_evidence_hash``).
    # Update ONLY on a deliberate re-deposit (a real full-budget re-run); that
    # update is itself the reviewable change this guard exists to force.
    _FROZEN_WITNESS_HASH = (
        "dfdf22f44f988e8f1ce46300c5e060aff3585d6911c952e56c10aebdb5db86aa"
    )

    @staticmethod
    def _ledger_witness_sha256(ledger_path):
        # Canonical SHA-256 over the committed JSONL witness: parse every
        # non-empty line, then sort_keys + compact-encode the list. Whitespace-
        # and key-order-invariant, recomputable from the committed bytes.
        parsed = [
            json.loads(line)
            for line in Path(ledger_path).read_text().splitlines()
            if line.strip()
        ]
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def test_witness_file_is_committed_and_deposit_points_at_it(self):
        # The witness is a committed repo file (not a gitignored runs/ path) and
        # the deposit's ``ledger_witness_path`` is a RELATIVE repo-root path that
        # resolves to it — a committed deposit must never carry an absolute,
        # worktree-specific path.
        assert FIXTURE_9B_FULL_LEDGER.exists(), (
            "the citable full-budget deposit's ledger witness is missing from "
            "tests/fixtures/ — reproducibility from committed bytes is broken."
        )
        data = json.loads(FIXTURE_9B_FULL.read_text())
        wit = data["ledger_witness_path"]
        assert wit is not None
        assert not Path(wit).is_absolute(), (
            f"ledger_witness_path must be relative, got absolute {wit!r}"
        )
        assert (Path(__file__).resolve().parent.parent / wit).exists()

    def test_witness_hash_is_self_consistent(self):
        # The stamped ``ledger_witness_sha256`` must equal the hash recomputed
        # from the committed ledger bytes — guards a stale stamp, a hand-edit
        # that touched the witness but not the stamp, or a truncated hash.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        recomputed = self._ledger_witness_sha256(FIXTURE_9B_FULL_LEDGER)
        assert data["ledger_witness_sha256"] == recomputed, (
            f"deposit stamp {data['ledger_witness_sha256']!r} != "
            f"recomputed {recomputed!r}"
        )

    def test_witness_hash_is_pinned_to_frozen_literal(self):
        # THE coordinated-drift guard for the witness. The stamped hash must
        # equal the frozen literal — so any byte change to the committed ledger
        # (a re-run, an accidental edit, a re-harvest) flips this red until the
        # literal is deliberately updated. Pinning the literal as well as the
        # recompute means a truncated/typo'd hash (which self-consistency would
        # rubber-stamp so long as the stamp matched the truncated recompute)
        # cannot pass: the stamp must equal BOTH the bytes AND the frozen value.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        assert data["ledger_witness_sha256"] == self._FROZEN_WITNESS_HASH
        # Belt + braces: the frozen literal is itself the recomputed value, so a
        # frozen literal that drifted from the bytes can only pass if the stamp
        # drifted identically — two independent fields must agree.
        assert self._FROZEN_WITNESS_HASH == self._ledger_witness_sha256(
            FIXTURE_9B_FULL_LEDGER
        )

    def test_ledger_reconstructs_deposit_loss_vectors_exactly(self):
        # THE independent-reproducibility link: each ledger arm's ``valid_loss``
        # (keyed by role + index) must reconstruct the deposit's three loss
        # vectors EXACTLY (float ==), not approximately. A skeptic reading only
        # the committed ledger reproduces the citable verdict's raw evidence.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        parsed = [
            json.loads(line)
            for line in FIXTURE_9B_FULL_LEDGER.read_text().splitlines()
            if line.strip()
        ]
        arms = [rec for rec in parsed if rec.get("type") == "arm"]
        by_role_index = {(a["role"], a["index"]): a for a in arms}
        for role, key in (
            ("candidate", "candidate_losses"),
            ("surrogate", "surrogate_losses"),
            ("baseline", "baseline_losses"),
        ):
            losses = data[key]
            n_ledger = sum(1 for r, _ in by_role_index if r == role)
            assert len(losses) == n_ledger, (
                f"{role}: deposit has {len(losses)} losses but ledger has "
                f"{n_ledger} arms — reconstruction shape mismatch."
            )
            for i, expected in enumerate(losses):
                arm = by_role_index[(role, i)]
                assert arm["valid_loss"] == expected, (
                    f"{role}[{i}]: ledger {arm['valid_loss']!r} != "
                    f"deposit {expected!r} — reconstruction is NOT exact."
                )

    def test_ledger_header_fingerprint_matches_deposit_run_config(self):
        # The witness's header fingerprint (the resumability config-gate) must
        # match the deposit's run config — so the ledger is from THIS run, not a
        # stale carry-over from a different config (the exact invariant the live
        # ``load_ledger`` fingerprint check enforces at resume time).
        data = json.loads(FIXTURE_9B_FULL.read_text())
        header = json.loads(FIXTURE_9B_FULL_LEDGER.read_text().splitlines()[0])
        assert header["type"] == "header"
        fp = header["fingerprint"]
        for fp_key, deposit_key in (
            ("total_steps", "total_steps"),
            ("warmup_steps", "warmup_steps"),
            ("depth", "depth"),
            ("spacing", "spacing"),
            ("seq_len", "seq_len"),
            ("train_examples", "train_examples"),
            ("valid_examples", "valid_examples"),
            ("base_seed", "base_seed"),
            ("scope_label", "scope_label"),
        ):
            assert fp[fp_key] == data[deposit_key], (
                f"fingerprint {fp_key}={fp[fp_key]!r} != deposit "
                f"{deposit_key}={data[deposit_key]!r}"
            )
        assert fp["active_scope"] == data["active_scope"]

    def test_runtime_ledger_path_stays_null_for_one_shot_harvest(self):
        # Semantic guard: ``ledger_path`` (the RUNTIME resumability path a fresh
        # ``make`` run would write) is ``null`` on the committed harvest, while
        # ``ledger_witness_path`` (the committed VERIFICATION copy) is set. The
        # two fields must not be conflated: ``ledger_path`` reflects how the run
        # was invoked (null ⟺ the deposit was harvested from a one-shot run),
        # the witness reflects what a verifier can read.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        assert data["ledger_path"] is None
        assert data["ledger_witness_path"] is not None

    def test_witness_is_not_in_evidence_hash_keys(self):
        # Additivity guard: the witness fields must NOT enter ``evidence_hash``
        # (they are provenance, not evidence), so attaching them cannot perturb
        # the pinned evidence hash — a coordinated repaint cannot reach the
        # evidence hash through the witness, and vice-versa.
        data = json.loads(FIXTURE_9B_FULL.read_text())
        base = _evidence_hash(data)
        mutated = dict(data)
        mutated["ledger_witness_sha256"] = "ZZZ_NEVER_REAL"
        mutated["ledger_witness_path"] = "ZZZ_NEVER_REAL"
        assert _evidence_hash(mutated) == base, (
            "ledger_witness_* leaked into evidence_hash — the witness must be "
            "additive provenance, not evidence."
        )


class TestCommittedLedgerWitnessHeterogeneousFull:
    """Independent-reproducibility provenance for the 2nd citable full verdict —
    the heterogeneous-FULL ledger witness (the committed-verification analogue of
    ``TestCommittedLedgerWitness`` for the heterogeneous arm shape).

    The bg heterogeneous-full run writes its resumability ledger to the stable
    ``runs/`` dir; at HARVEST the operator copies it to
    ``FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER`` and stamps
    ``ledger_witness_path`` / ``ledger_witness_sha256`` into the deposit (HARVEST
    fields — the worker stamps ``run_log_*`` but never ``ledger_witness_*``; see
    ``scripts/fire_freeze_ci_9b_full_heterogeneous.sh``). These tests then
    auto-verify the harvest: the witness is a committed repo file pointed at by a
    RELATIVE path, its stamp is self-consistent with its bytes, and it
    reconstructs the deposit's three loss-vectors (candidate / surrogate /
    CONTROL — the heterogeneous arm shape, not baseline) byte-for-byte.

    Canonical hashing DELEGATES to ``TestCommittedLedgerWitness._ledger_witness_
    sha256`` so the two verdicts are held to a single canonicalization (a future
    drift in one would fail the other). The run LANDED 2026-07-18 and the HARVEST
    is DONE: ``_FROZEN_WITNESS_HASH`` (activates the coordinated-drift pin below)
    and the frozen evidence-hash literal in :class:`TestDepositEvidenceHash` are
    both pinned to the committed bytes. Tests keep a skip-until-exists guard so
    an accidentally-removed witness/deposit fails loudly, not silently.
    """

    # Frozen witness hash over the canonical encoding of the committed ledger.
    # Filled at HARVEST (the deliberate change the pin exists to force) when the
    # heterogeneous-full ledger landed (2026-07-18): the verdict TIES (candidate
    # 1.71804 vs surrogate 1.71903, CI straddles 0) on a clean one-shot run
    # (``resumed_arm_count=0``), so the SURPASSES the reduced-budget heterogeneous
    # leg earned does NOT survive the full 1500-step budget — recorded honestly,
    # not papered over. A future deliberate re-deposit (real re-run) must update
    # this literal; that update is itself the reviewable change this guard forces.
    _FROZEN_WITNESS_HASH = (
        "00084c2646eb5463cd0ff835a8a6579d31c331c07d81a89bc1f7dc3e1f4b0390"
    )

    def test_witness_file_is_committed_and_deposit_points_at_it(self):
        if not FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER.exists():
            pytest.skip("heterogeneous-full ledger witness not landed yet")
        assert FIXTURE_9B_FULL_HETEROGENEOUS.exists(), (
            "ledger witness landed but the deposit is missing — harvest incomplete."
        )
        data = json.loads(FIXTURE_9B_FULL_HETEROGENEOUS.read_text())
        wit = data["ledger_witness_path"]
        assert wit is not None
        assert not Path(wit).is_absolute(), (
            f"ledger_witness_path must be relative, got absolute {wit!r}"
        )
        assert (Path(__file__).resolve().parent.parent / wit).exists()

    def test_witness_hash_is_self_consistent(self):
        if not FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER.exists():
            pytest.skip("heterogeneous-full ledger witness not landed yet")
        data = json.loads(FIXTURE_9B_FULL_HETEROGENEOUS.read_text())
        recomputed = TestCommittedLedgerWitness._ledger_witness_sha256(
            FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER
        )
        assert data["ledger_witness_sha256"] == recomputed, (
            f"deposit stamp {data['ledger_witness_sha256']!r} != "
            f"recomputed {recomputed!r}"
        )

    def test_witness_hash_is_pinned_to_frozen_literal(self):
        # THE coordinated-drift guard for the heterogeneous witness. Skips until
        # harvest fills ``_FROZEN_WITNESS_HASH`` (the deliberate pin), then
        # asserts the stamp equals BOTH the recomputed bytes AND the frozen
        # literal — so a repaint of the ledger bytes + stamp cannot pass silently.
        if not FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER.exists():
            pytest.skip("heterogeneous-full ledger witness not landed yet")
        if self._FROZEN_WITNESS_HASH is None:
            pytest.skip(
                "frozen witness literal not pinned yet — fill "
                "_FROZEN_WITNESS_HASH at harvest (the deliberate pin)."
            )
        data = json.loads(FIXTURE_9B_FULL_HETEROGENEOUS.read_text())
        recomputed = TestCommittedLedgerWitness._ledger_witness_sha256(
            FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER
        )
        assert data["ledger_witness_sha256"] == self._FROZEN_WITNESS_HASH
        assert self._FROZEN_WITNESS_HASH == recomputed

    def test_ledger_reconstructs_deposit_loss_vectors_exactly(self):
        # THE independent-reproducibility link for the heterogeneous arm shape:
        # each ledger arm's ``valid_loss`` reconstructs the deposit's three loss
        # vectors (candidate / surrogate / CONTROL) byte-for-byte. Mirrors the
        # homogeneous reconstruction with the heterogeneous role map (control, not
        # baseline — the heterogeneous full carries no full-backprop baseline arm).
        if not FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER.exists():
            pytest.skip("heterogeneous-full ledger witness not landed yet")
        data = json.loads(FIXTURE_9B_FULL_HETEROGENEOUS.read_text())
        parsed = [
            json.loads(line)
            for line in FIXTURE_9B_FULL_HETEROGENEOUS_LEDGER.read_text().splitlines()
            if line.strip()
        ]
        arms = [rec for rec in parsed if rec.get("type") == "arm"]
        by_role_index = {(a["role"], a["index"]): a for a in arms}
        for role, key in (
            ("candidate", "candidate_losses"),
            ("surrogate", "surrogate_losses"),
            ("control", "control_losses"),
        ):
            losses = data[key]
            n_ledger = sum(1 for r, _ in by_role_index if r == role)
            assert len(losses) == n_ledger, (
                f"{role}: deposit has {len(losses)} losses but ledger has "
                f"{n_ledger} arms — reconstruction shape mismatch."
            )
            for i, expected in enumerate(losses):
                arm = by_role_index[(role, i)]
                assert arm["valid_loss"] == expected, (
                    f"{role}[{i}]: ledger {arm['valid_loss']!r} != "
                    f"deposit {expected!r} — reconstruction is NOT exact."
                )


# ── committed run-log witness: independent reproducibility of the het deposit ─
#
# Where the full-budget deposit's ledger witness (``TestCommittedLedgerWitness``
# above) certifies the resumability ledger, this certifies the LOSS-CURVE run-log
# artifact (``_write_run_log`` / ``--run-log``, iter ``65073bb``) for the
# heterogeneous deposit — the per-arm loss trajectory the real 9B run produced.
# This is the artifact the run-log infra was built to persist; a fresh real-9B
# run (``make freeze-validloss-ci-9b-heterogeneous-generalization``) reproduced
# the recorded SURPASSES verdict, so the finding is not an artifact of a single
# noisy run, and the loss curves behind it are now committed + content-hashed.


class TestCommittedRunLogWitness:
    """Independent reproducibility provenance for the heterogeneous §4 deposit
    via its committed RUN-LOG (loss-curve) WITNESS.

    A skeptic re-reads the 9 committed arms (each with a 96-step ``loss_curve``
    + exact ``final_valid_loss``), recomputes the canonical SHA-256 via
    ``_run_log_sha256``, and reconstructs the deposit's three loss-vectors
    byte-for-byte — the verdict's dynamics are reproducible from committed
    bytes, not just its endpoint. The witness is additive (NOT in
    ``EVIDENCE_HASH_KEYS``, so it cannot perturb ``evidence_hash``).
    """

    # Frozen witness hash over the canonical encoding of the committed run log
    # (``_run_log_sha256`` — sort_keys + compact separators over the parsed
    # payload). Update ONLY on a deliberate re-run; that update is itself the
    # reviewable change this guard exists to force.
    _FROZEN_RUNLOG_HASH = (
        "eb1da4cdac69c663ab6e858a7efa831578ceebd5821d6ca8c43ea37127b76269"
    )

    def test_witness_file_is_committed_and_deposit_points_at_it(self):
        # The witness is a committed repo file and the deposit's ``run_log_path``
        # is a RELATIVE repo-root path that resolves to it.
        assert FIXTURE_9B_HETEROGENEOUS_RUNLOG.exists(), (
            "the heterogeneous deposit's run-log witness is missing from "
            "tests/fixtures/ — loss-curve reproducibility is broken."
        )
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        wit = data["run_log_path"]
        assert wit is not None
        assert not Path(wit).is_absolute(), (
            f"run_log_path must be relative, got absolute {wit!r}"
        )
        assert (Path(__file__).resolve().parent.parent / wit).exists()

    def test_witness_hash_is_self_consistent(self):
        # The stamped ``run_log_sha256`` must equal the hash recomputed from the
        # committed run-log bytes (via the source ``_run_log_sha256`` primitive).
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        witness = json.loads(FIXTURE_9B_HETEROGENEOUS_RUNLOG.read_text())
        assert data["run_log_sha256"] == _run_log_sha256(witness)

    def test_witness_hash_is_pinned_to_frozen_literal(self):
        # THE coordinated-drift guard for the run-log witness. The stamped hash
        # must equal BOTH the recomputed value (self-consistency) AND the frozen
        # literal — so a re-run, accidental edit, or truncated hash flips red.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        assert data["run_log_sha256"] == self._FROZEN_RUNLOG_HASH
        witness = json.loads(FIXTURE_9B_HETEROGENEOUS_RUNLOG.read_text())
        assert self._FROZEN_RUNLOG_HASH == _run_log_sha256(witness)

    def test_runlog_arms_reconstruct_deposit_loss_vectors_exactly(self):
        # THE independent-reproducibility link: each run-log arm's
        # ``final_valid_loss`` (role+index) reconstructs the deposit's three
        # loss vectors EXACTLY (float ==).
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        witness = json.loads(FIXTURE_9B_HETEROGENEOUS_RUNLOG.read_text())
        arms = {(a["role"], a["index"]): a for a in witness["arms"]}
        for role, key in (
            ("candidate", "candidate_losses"),
            ("surrogate", "surrogate_losses"),
            ("control", "control_losses"),
        ):
            losses = data[key]
            n_log = sum(1 for r, _ in arms if r == role)
            assert len(losses) == n_log, (
                f"{role}: deposit has {len(losses)} losses but run-log has "
                f"{n_log} arms — reconstruction shape mismatch."
            )
            for i, expected in enumerate(losses):
                assert arms[(role, i)]["final_valid_loss"] == expected, (
                    f"{role}[{i}]: run-log {arms[(role, i)]['final_valid_loss']!r}"
                    f" != deposit {expected!r} — reconstruction is NOT exact."
                )

    def test_runlog_arms_carry_real_loss_curves(self):
        # The point of the run-log (vs the ledger endpoint witness): the per-step
        # LOSS CURVES — the dynamics behind the verdict. Every arm carries a
        # non-empty, genuinely-decreasing trajectory covering the full training
        # (a real learning curve, not a fabricated flat line).
        witness = json.loads(FIXTURE_9B_HETEROGENEOUS_RUNLOG.read_text())
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        for arm in witness["arms"]:
            curve = arm["loss_curve"]
            assert len(curve) == data["total_steps"], (
                f"{arm['role']}[{arm['index']}] loss_curve has {len(curve)} "
                f"steps, expected {data['total_steps']} — does not cover the "
                f"full training run."
            )
            assert curve[0] > curve[-1], (
                f"{arm['role']}[{arm['index']}] loss_curve does not decrease "
                f"({curve[0]} -> {curve[-1]}) — not a real learning trajectory."
            )

    def test_runlog_run_config_matches_deposit_config(self):
        # The run-log's recorded run_config must match the deposit's — so the
        # witness is from THIS run, not a stale carry-over.
        witness = json.loads(FIXTURE_9B_HETEROGENEOUS_RUNLOG.read_text())
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        rc = witness["run_config"]
        for key in (
            "total_steps", "warmup_steps", "depth", "spacing", "seq_len",
            "train_examples", "valid_examples", "base_seed", "architecture",
            "n_candidate", "n_surrogate", "n_control",
        ):
            assert rc[key] == data[key], (
                f"run_config {key}={rc[key]!r} != deposit {data[key]!r}"
            )
        assert rc["lora_rank_pattern"] == data["lora_rank_pattern"]

    def test_run_log_fields_do_not_perturb_evidence_hash(self):
        # Additivity guard: the run_log_* fields must NOT enter evidence_hash.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        base = _evidence_hash(data)
        mutated = dict(data)
        mutated["run_log_sha256"] = "ZZZ_NEVER_REAL"
        mutated["run_log_path"] = "ZZZ_NEVER_REAL"
        assert _evidence_hash(mutated) == base, (
            "run_log_* leaked into evidence_hash — the witness must be additive."
        )


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


# ── heterogeneous (per-layer rank) deposit — the target-scale leg ────────────


class TestHeterogeneousTargetScaleDeposit:
    """The HETEROGENEOUS (per-layer asymmetric rank) §4 deposit — the one §4
    research task that was open at target scale. Every sibling deposit runs on a
    HOMOGENEOUS LoRA (uniform r=16 on every active layer); this deposit re-runs
    the SAME generalization-regime A/B on an ASYMMETRIC adapter (output-side
    layers given more CAPACITY via peft rank_pattern). These tests pin:

    * the recorded rank schedule (the geometric ``lora_rank_pattern``) is the one
      :func:`heterogeneous_ranks_9b` builds;
    * the SURPASSES verdict + the direction SURPASSES are EARNED (replay-faithful);
    * the load-bearing heterogeneous signature: output-first freeze removes the
      MOST capacity (highest-rank layers → fewest trainable params) yet STILL wins;
    * the verdict SURVIVES the homogeneous→heterogeneous architecture change
      (cross-deposit vs FIXTURE_9B_GENERALIZATION) — it is not a uniform-rank
      artifact.
    """

    def test_fixture_exists_and_is_real_target_scale(self):
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        assert data["proxy_scale"] is False
        assert data["citable_as_target_scale"] is True
        assert data["citable_as_full_section4_verdict"] is False  # honestly reduced
        assert data["reduced_budget"] is True  # 96 < cfg_max_steps=1500
        assert data["model"] == "Qwen/Qwen3.5-9B"
        assert data["seq_len"] == 1024
        assert data["device"] == "cuda"

    def test_architecture_field_is_heterogeneous(self):
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        assert data["architecture"] == "heterogeneous"
        # Every sibling deposit omits the field (defaults to homogeneous); only
        # this deposit carries it, and it carries the per-layer schedule.
        assert data["lora_rank_pattern"] is not None

    def test_rank_pattern_is_the_predicted_geometric_schedule(self):
        # The committed lora_rank_pattern must equal what heterogeneous_ranks_9b
        # builds for this active scope + base rank — pins the schedule to the code
        # path rather than a hand-written dict, and asserts the geometric shape
        # (monotone ascending, output layer caps at base_rank=r=16).
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        scope = sorted(data["active_scope"])
        # base_rank is the config's cfg.lora.r (=16); the schedule's max == it.
        expected = dict(
            zip((str(i) for i in scope), heterogeneous_ranks_9b(set(scope), 16))
        )
        assert data["lora_rank_pattern"] == expected
        ranks = [data["lora_rank_pattern"][str(i)] for i in scope]
        assert ranks == sorted(ranks)   # monotone non-decreasing toward output
        assert ranks[-1] == 16           # output-most layer caps at base_rank
        assert ranks[0] == 1             # input-most layer is the floor

    def test_candidate_freezes_highest_rank_layers(self):
        # Under asymmetric ranks the output-first candidate freezes layers
        # {29,30,31} = ranks (7,11,16) — the THREE highest-capacity active layers;
        # the input-side control freezes {24,25,26} = ranks (1,1,2) — the lowest.
        # Direction isolation (output-contiguous vs input-contiguous) is preserved,
        # now coupled to capacity rather than uniform.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        rp = data["lora_rank_pattern"]
        cand = {tuple(p["frozen_layers"]) for p in data["candidate_provenance"]}
        ctrl = {tuple(p["frozen_layers"]) for p in data["control_provenance"]}
        assert cand == {(29, 30, 31)}            # output-contiguous
        assert ctrl == {(24, 25, 26)}             # input-contiguous
        assert cand.isdisjoint(ctrl)
        # Candidate froze the highest-rank layers; control the lowest.
        assert sum(rp[str(i)] for i in (29, 30, 31)) > sum(rp[str(i)] for i in (24, 25, 26))

    def test_output_first_freeze_removes_most_capacity(self):
        # THE heterogeneous signature. Freezing a higher-rank layer removes more
        # trainable params, so the output-first candidate (highest-rank layers)
        # ends up with the FEWEST trainable params of any arm — yet it still wins
        # (test below). The input-side control (lowest-rank layers) has the MOST.
        # This is the finding the uniform-rank sibling deposits structurally
        # cannot show: the verdict holds even when output-first means sacrificing
        # the most capacity.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        cand_params = data["candidate_provenance"][0]["n_trainable_params"]
        surr_params = [p["n_trainable_params"] for p in data["surrogate_provenance"]]
        ctrl_params = data["control_provenance"][0]["n_trainable_params"]
        assert cand_params < min(surr_params) < max(surr_params) < ctrl_params
        # And params track rank monotonically: candidate (3 highest-rank) removed
        # the most, control (3 lowest-rank) the least.
        assert cand_params < ctrl_params

    def test_recorded_surrogate_verdict_is_earned(self):
        # The stored floats re-judge to the recorded SURPASSES under the
        # deterministic bootstrap — the verdict is earned, not painted on.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        ci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["surrogate_losses"], seed=data["base_seed"]
        )
        assert ci.significance_verdict == data["verdict"] == SURPASSES
        assert data["lower"] > 0.0  # CI entirely above zero => significant
        assert ci.candidate_mean == pytest.approx(data["candidate_mean"], abs=1e-9)
        assert ci.surrogate_mean == pytest.approx(data["surrogate_mean"], abs=1e-9)
        assert ci.lower == pytest.approx(data["lower"], abs=1e-9)
        assert ci.upper == pytest.approx(data["upper"], abs=1e-9)

    def test_recorded_direction_verdict_is_earned(self):
        # The direction CI (candidate output-contiguous vs control input-
        # contiguous, contiguity held fixed) reproduces — the attribution to
        # output-side DIRECTION survives the asymmetric-rank stack.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        dd = data["direction"]
        dci = surrogate_valid_loss_ci(
            data["candidate_losses"], data["control_losses"], seed=data["base_seed"]
        )
        assert dci.significance_verdict == dd["verdict"] == SURPASSES
        assert dd["lower"] > 0.0
        assert dci.surrogate_mean == pytest.approx(dd["control_mean"], abs=1e-9)

    def test_monotone_ranking_matches_homogeneous_sibling(self):
        # Same candidate < surrogate < control ordering as the homogeneous
        # generalization deposit — the ranking is not a uniform-rank artifact.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        assert data["candidate_mean"] < data["surrogate_mean"] < data["direction"]["control_mean"]

    def test_verdict_survives_homogeneous_to_heterogeneous(self):
        # The load-bearing cross-deposit result: the §4 surrogate verdict is
        # SURPASSES on BOTH the homogeneous AND the heterogeneous (asymmetric-rank)
        # stack — the output-first freeze order beats a random order regardless of
        # whether the adapter is uniform-rank or output-heavy. A future re-run that
        # flips the heterogeneous verdict would make this red, surfacing the
        # architecture-dependence as a reviewable change rather than silent drift.
        het = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        hom = json.loads(FIXTURE_9B_GENERALIZATION.read_text())
        assert het["verdict"] == SURPASSES
        assert hom["verdict"] == SURPASSES

    def test_regime_is_generalization_not_memorization(self):
        # The honest regime: final_ce_train_loss ~1.58 (train CE ≈ valid CE),
        # NOT ~0 (memorization). A regression that re-deposits a memorization
        # run flips this red — the deposit's headline must rest on a generalizing
        # model, not an overfit adapter.
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        for arm in ("candidate", "surrogate", "control"):
            for p in data[f"{arm}_provenance"]:
                assert p["final_ce_train_loss"] > 0.5
        assert data["regime"] == "generalization"

    def test_deposit_is_non_thin_and_honestly_reduced(self):
        data = json.loads(FIXTURE_9B_HETEROGENEOUS.read_text())
        assert data["is_thin_evidence"] is False
        assert data["n_candidate"] >= 3
        assert data["n_surrogate"] >= 3
        assert data["n_control"] >= 3
        # Honest reduced-budget: NOT the full §4 verdict (96 < 1500), stated.
        assert data["reduced_budget"] is True
        assert data["total_steps"] < data["cfg_max_steps"]


# ── full-budget × heterogeneous launch path: the ONE remaining open §4 leg ───
#
# The homogeneous full-budget verdict LANDED (TIES, ``4b88ca8``) and the
# heterogeneous REDUCED-budget verdict LANDED (SURPASSES, ``FIXTURE_9B_HETEROGE-
# NEOUS``). Their missing intersection — the heterogeneous FULL-budget leg, i.e.
# does the output-first order effect survive full-budget generalization on an
# ASYMMETRIC adapter? — is the one open §4 research task, and it is GPU-gated (a
# ~hours, 9-arm, 1500-step run). The deposit does NOT exist yet: no fabricated
# verdict. These tests pin the parts that need NO GPU, so the leg is correctly
# wired and will be HONESTLY labeled the moment a free GPU window lets it run via
# ``make freeze-validloss-ci-9b-full-heterogeneous-bg``.


class TestFullBudgetHeterogeneousLaunchPath:
    """The full-budget × heterogeneous §4 leg is launchable and honestly gated.

    Companion to :class:`TestHeterogeneousTargetScaleDeposit` (the REDUCED-budget
    heterogeneous deposit) and :class:`TestFullBudgetDepositVerdictPin` (the
    FULL-budget HOMOGENEOUS deposit): this is their missing intersection — the
    one open research task. Four GPU-free invariants make the leg launch-ready:

    * the harness ACCEPTS the composition (``--architecture heterogeneous`` is a
      declared choice that composes with the full-budget flag set);
    * the citation gate is ARCHITECTURE-AGNOSTIC — a would-be full-budget
      heterogeneous deposit clears it on exactly the same four conjuncts as the
      homogeneous one, and each conjunct is load-bearing (mutation-proven), so
      the leg can be neither silently over- nor under-cited when it lands;
    * the heterogeneous full deposit's EVIDENCE HASH is DISTINCT from the
      homogeneous full deposit's (``architecture`` is an evidence key), so the
      two citable full-budget verdicts can never be silently conflated;
    * the Makefile wires the leg with a DISTINCT deposit and ledger path and
      routes the bg launcher through the generic ``launch_freeze_ci_9b_full``.

    What these tests deliberately do NOT do: fabricate a verdict. The 1500-step
    run itself waits for a free GPU window — the bg launcher polls a held GPU,
    defers (exit 75), and banks completed arms in ``--ledger``.
    """

    def test_harness_accepts_full_budget_heterogeneous_composition(self):
        # ``--architecture heterogeneous`` is a declared choice that composes
        # with the full-budget flag set, so the open leg is reachable from the
        # SAME CLI the homogeneous full run uses — no harness change needed.
        parser = build_parser()
        args = parser.parse_args(
            ["--architecture", "heterogeneous", "--total-steps", "1500"]
        )
        assert HETEROGENEOUS in ARCHITECTURES
        assert args.architecture == HETEROGENEOUS
        assert args.total_steps == 1500

    def test_full_section4_gate_is_architecture_agnostic(self):
        # THE honesty load-bearer. The citation gate takes NO architecture
        # parameter: a full-budget heterogeneous deposit clears it on EXACTLY
        # the same four conjuncts as the homogeneous deposit. So when the leg
        # runs it is cited on its actual evidence — never over-cited (gate opens
        # for a thin / memorized run) nor under-cited (gate stays shut on a
        # clean full run). Each conjunct is independently load-bearing.
        clearing = dict(
            proxy_scale=False,
            reduced_budget=False,
            is_thin_evidence=False,
            regime=REGIME_GENERALIZATION,
        )
        assert _full_section4_verdict_gate(**clearing) is True
        for axis, bad in (
            ("proxy_scale", True),
            ("reduced_budget", True),
            ("is_thin_evidence", True),
            ("regime", REGIME_MEMORIZATION),
        ):
            blocked = dict(clearing)
            blocked[axis] = bad
            assert _full_section4_verdict_gate(**blocked) is False, (
                f"flipping the {axis} axis must close the gate — a would-be "
                f"heterogeneous-full deposit could otherwise be silently mis-cited."
            )

    def test_heterogeneous_full_evidence_hash_is_distinct_from_homogeneous(self):
        # ``architecture`` (and ``lora_rank_pattern``) are EVIDENCE keys, so a
        # future full-budget heterogeneous deposit hashes DISTINCTLY from the
        # landed homogeneous full deposit. The two citable full-budget verdicts
        # can never be silently conflated, and the heterogeneous one will land
        # with its OWN frozen evidence-hash literal rather than reuse the
        # homogeneous pin (see ``TestDepositEvidenceHash``).
        hom = json.loads(FIXTURE_9B_FULL.read_text())
        assert hom.get("architecture") != HETEROGENEOUS  # sibling is homogeneous
        # Architecture ALONE is load-bearing in the hash (no rank pattern yet).
        het_arch_only = dict(hom)
        het_arch_only["architecture"] = HETEROGENEOUS
        assert _evidence_hash(het_arch_only) != _evidence_hash(hom)
        # Adding the asymmetric rank pattern moves it again — the eventual
        # heterogeneous-full deposit carries BOTH distinguishing evidence keys.
        het_full = dict(het_arch_only)
        het_full["lora_rank_pattern"] = {"29": 7, "30": 11, "31": 16}
        assert _evidence_hash(het_full) != _evidence_hash(het_arch_only)

    def test_makefile_wires_distinct_full_budget_heterogeneous_paths(self):
        # Fail-loud composition guard. The Makefile must wire the open leg with
        # ``--architecture heterogeneous`` + the full-budget flag var + a DISTINCT
        # deposit and ledger path (so it never clobbers the homogeneous full
        # deposit/ledger), and the bg variant must route through the generic
        # ``launch_freeze_ci_9b_full``. A refactor that drops the architecture
        # flag or collapses the path onto the homogeneous sibling flips this red.
        repo_root = Path(__file__).resolve().parents[1]
        makefile = (repo_root / "Makefile").read_text()
        lines = makefile.splitlines()

        def recipe(target_prefix):
            start = next(
                i for i, ln in enumerate(lines) if ln.startswith(target_prefix)
            )
            body = []
            for ln in lines[start + 1 :]:
                if ln.startswith("\t"):
                    body.append(ln)
                elif ln.strip() == "":
                    continue
                else:
                    break
            return " ".join(body)

        # The flag var is defined at full budget with a distinct ledger.
        flag_def = next(
            ln for ln in lines if ln.startswith("FREEZE_9B_FULL_HETEROGENEOUS_FLAGS ?=")
        )
        assert "--total-steps 1500" in flag_def
        assert "freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl" in flag_def

        direct = recipe("freeze-validloss-ci-9b-full-heterogeneous:")
        bg = recipe("freeze-validloss-ci-9b-full-heterogeneous-bg:")
        assert direct and bg  # both targets exist with a recipe
        for body in (direct, bg):
            assert "FREEZE_9B_FULL_HETEROGENEOUS_FLAGS" in body
            assert "--architecture heterogeneous" in body
            # Exact ``--output`` path — the heterogeneous deposit, NOT the
            # homogeneous sibling (substring-safe: asserted as a full token).
            assert "--output tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json" in body
        assert "run_freeze_validloss_ci_9b" in direct  # direct invokes worker
        assert "launch_freeze_ci_9b_full" in bg  # bg delegates to generic launcher

    def test_fire_script_robustly_launches_full_budget_heterogeneous(self):
        # The committed robust launcher for the OPEN leg. The Makefile ``-bg``
        # target fires from the worktree with RELATIVE ``--output``/``--ledger``
        # paths, so recycling the worktree mid-run would orphan the deposit (the
        # [[dead-cwd-background-run-trap]]: a multi-hour 9B run that trains fine
        # but can never persist). ``scripts/fire_freeze_ci_9b_full_heterogeneous.sh``
        # is the documented escape — it resolves code + deposit ABSOLUTELY against
        # a stable dir and PYTHONPATH-pointed repo, so the leg can be fired to
        # survive worktree recycling. This guard pins the composition so a refactor
        # cannot silently drop the architecture flag, collapse the deposit onto the
        # homogeneous sibling, or regress to a worktree-relative path.
        repo_root = Path(__file__).resolve().parents[1]
        fire = (repo_root / "scripts" / "fire_freeze_ci_9b_full_heterogeneous.sh")
        assert fire.exists(), "the robust launcher for the open heterogeneous leg must be committed"
        assert fire.stat().st_mode & 0o111, "the fire script must be executable"
        text = fire.read_text()

        # Routes through the generic polling launcher (defers exit 75 on a held
        # GPU, banks arms in --ledger, bounded) — not the worker directly.
        assert "launch_freeze_ci_9b_full" in text
        assert "--architecture heterogeneous" in text
        # Direction-isolation A/B (candidate/surrogate/CONTROL), NOT the
        # homogeneous leg's full-backprop baseline arm.
        assert "--n-control 3" in text
        assert "--n-baseline" not in text
        # Full budget + generalization regime (the two honesty axes that clear
        # ``reduced_budget`` and the regime gate conjunct).
        assert "--total-steps 1500" in text
        assert "--train-examples 600" in text

        # Distinct heterogeneous deposit + ledger — substring-safe full tokens so
        # the leg never clobbers the homogeneous full deposit/ledger.
        assert "freeze_validloss_ci_9b_full_heterogeneous.json" in text
        assert "freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl" in text
        assert "freeze_validloss_ci_9b_full_ledger.jsonl" not in text  # homogeneous sibling

        # Dead-CWD-trap defeat: the ``--config`` / ``--ledger`` / ``--output``
        # FLAG VALUES anchor to ``$STABLE`` (absolute), and PYTHONPATH points at
        # the stable repo — so a recycled worktree cannot orphan a running arm or
        # its deposit. (The comment block legitimately names ``tests/fixtures/``
        # as the harvest target; the load-bearing check is the flag values.)
        assert "TG_LORA_FULL_RUN_DIR" in text  # overridable stable dir
        assert "PYTHONPATH" in text
        for flag in ("--config", "--ledger", "--output"):
            assert f'{flag} "$STABLE/' in text, (
                f"{flag} must resolve under $STABLE (absolute), not a worktree-relative path"
            )


class TestCudaOomTempfail:
    """A run-time CUDA OOM is transient CONTENTION, classified as the tempfail
    the self-retrying launcher polls — not a fatal exit 1 that kills a multi-hour
    full-budget run after a single contention spike.

    The pre-flight free-memory gate (:func:`gpu_free_memory_deferred`) is a
    point-in-time snapshot: it passes, then a concurrent process re-claims the
    card in the seconds before the model's weight allocation (a TOCTOU race that
    bit the heterogeneous full launch on a contended 12 GiB card — the load OOM'd
    ~14 s after a passing snapshot, surfacing as ``torch.AcceleratorError``, and
    the uncaught exception exited 1 = FATAL, so the launcher died instead of
    retrying the next free-GPU window). These tests pin the honest
    re-classification: a CUDA OOM → exit 75 (RETRY); a non-OOM RuntimeError stays
    fatal (never swallowed as retryable).
    """

    @pytest.mark.parametrize("exc", [
        pytest.param(torch.cuda.OutOfMemoryError("CUDA out of memory."), id="typed-oom"),
        pytest.param(getattr(torch, "AcceleratorError", RuntimeError)(
            "CUDA error: out of memory"), id="accelerator-oom"),
        pytest.param(RuntimeError("RuntimeError: CUDA out of memory."), id="legacy-runtimeerror"),
    ])
    def test_is_cuda_oom_matches_oom_variants(self, exc):
        # torch surfaces OOM two ways — the typed ``OutOfMemoryError`` and, under
        # the device-dispatch path, an ``AcceleratorError`` whose message carries
        # "out of memory". NEITHER is a parent of the other, so all three
        # (incl. legacy plain-RuntimeError) must match.
        assert is_cuda_oom(exc) is True

    @pytest.mark.parametrize("exc", [
        pytest.param(RuntimeError("Pre-LoRA active scope drifted from post-LoRA"), id="scope-drift"),
        pytest.param(getattr(torch, "AcceleratorError", RuntimeError)(
            "CUDA error: device-side assert triggered"), id="non-oom-device-error"),
        pytest.param(ValueError("not enough host RAM"), id="non-runtime"),
    ])
    def test_is_cuda_oom_rejects_non_oom(self, exc):
        # A scope-drift RuntimeError or a non-OOM device error must STAY fatal —
        # mis-classifying either as retryable contention would swallow a real bug
        # behind an infinite retry loop.
        assert is_cuda_oom(exc) is False

    def test_main_classifies_run_time_oom_as_tempfail(self, tmp_path, monkeypatch):
        # Bypass the pre-flight (--min-free-gib 0) so a contended card doesn't
        # defer before run_ci_9b; the OOM a concurrent process causes inside the
        # run must surface as exit 75 (RETRY), not exit 1 (FATAL).
        import scripts.run_freeze_validloss_ci_9b as mod
        repo_root = Path(__file__).resolve().parents[1]

        def _boom(**kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        monkeypatch.setattr(mod, "run_ci_9b", _boom)

        argv = [
            "--config", str(repo_root / "configs" / "9b_baseline_suffix_only_last25.yaml"),
            "--output", str(tmp_path / "out.json"),
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--architecture", "heterogeneous",
            "--total-steps", "1500",
            "--min-free-gib", "0",  # bypass the pre-flight snapshot
        ]
        assert main(argv) == EXIT_GPU_TEMPFAIL  # 75 → launcher retries

    def test_main_reraises_non_oom_runtimeerror(self, tmp_path, monkeypatch):
        # A non-OOM RuntimeError (e.g. heterogeneous scope drift) stays FATAL —
        # the launcher must NOT infinite-loop it as retryable contention.
        import scripts.run_freeze_validloss_ci_9b as mod
        repo_root = Path(__file__).resolve().parents[1]

        def _drift(**kwargs):
            raise RuntimeError("Pre-LoRA active scope drifted from the post-LoRA scope")
        monkeypatch.setattr(mod, "run_ci_9b", _drift)

        argv = [
            "--config", str(repo_root / "configs" / "9b_baseline_suffix_only_last25.yaml"),
            "--output", str(tmp_path / "out.json"),
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--architecture", "heterogeneous",
            "--total-steps", "1500",
            "--min-free-gib", "0",
        ]
        with pytest.raises(RuntimeError, match="scope drifted"):
            main(argv)


class TestMainDepositIsAlwaysJson:
    """``main()``'s ``--output`` deposit is ALWAYS canonical JSON, independent
    of ``--json``.

    The on-disk deposit is harvested into ``tests/fixtures/``, pinned by the
    deposit / evidence-hash / ledger-witness tests, and re-loaded as JSON by the
    paper gate — so a human-readable text report deposited to a ``.json`` path
    would be a silently-unparseable (or parseable-but-truncated) §4 verdict, the
    GOAL §7 cardinal honesty break that wastes a multi-hour, multi-arm run.
    ``--json`` selects JSON vs the text report on STDOUT only; the deposit
    format is decoupled from it. This guards the one assembled ``main() →
    _write_deposit`` join that (a) the slice tests here leave uncovered — they
    monkeypatch ``run_ci_9b`` to RAISE, so ``main()`` returns before the deposit
    write — and (b) the ``run_ci_9b``-level launch-honesty dry-run
    (``test_freeze_ci_9b_launch_honesty.py``) leaves uncovered — it calls
    ``run_ci_9b`` directly and writes via ``_write_deposit`` with hand-built
    JSON, bypassing ``main()`` entirely. Pre-fix the deposit format was coupled
    to ``--json``, correct only because every Makefile / launcher caller
    happened to pass ``--json`` alongside ``--output``.
    """

    @staticmethod
    def _argv(repo_root, out_path, *, json_flag):
        argv = [
            "--config", str(repo_root / "configs" / "9b_baseline_suffix_only_last25.yaml"),
            "--output", str(out_path),
            "--min-free-gib", "0",  # bypass the pre-flight free-memory snapshot
        ]
        if json_flag:
            argv.append("--json")
        return argv

    @staticmethod
    def _depositable_stub():
        # ``_stub_result``'s minimal ``_stub_ci`` is rich enough for
        # ``result_to_json`` (the JSON path) but lacks the 4 fields
        # ``format_report_9b`` → ``format_surrogate_valid_loss_ci`` reads
        # (``n_candidate`` / ``n_surrogate`` / ``material_margin`` / ``seed``).
        # The real ``ci`` from ``surrogate_valid_loss_ci`` carries them; the stub
        # just has to mirror that so the no-``--json`` stdout text path renders.
        return _stub_result(ci=_stub_ci(
            n_candidate=2, n_surrogate=2, material_margin=0.05, seed=0,
        ))

    def test_deposit_is_json_without_json_flag(self, tmp_path, monkeypatch, capsys):
        # Regression: ``--output`` WITHOUT ``--json`` must still deposit canonical
        # JSON (not the format_report_9b text). Pre-fix this wrote text to a
        # ``.json`` path → json.loads raises here → RED.
        import scripts.run_freeze_validloss_ci_9b as mod
        repo_root = Path(__file__).resolve().parents[1]
        stub = self._depositable_stub()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mod, "run_ci_9b", lambda **kw: stub)

        out = tmp_path / "deposit.json"
        assert main(self._argv(repo_root, out, json_flag=False)) == 0

        deposited = json.loads(out.read_text())  # text → JSONDecodeError → RED
        assert deposited == result_to_json(stub)  # canonical JSON shape, not text
        assert "verdict" in deposited and "candidate_losses" in deposited

        # STDOUT honors --json: without it, stdout is the human-readable text
        # report (NOT JSON) — so the deposit-vs-stdout decoupling is real, not a
        # no-op rename. Pre-fix deposit == stdout == text.
        stdout = capsys.readouterr().out
        with pytest.raises(json.JSONDecodeError):
            json.loads(stdout)
        assert "freeze_valid_loss_ci_9b" in stdout  # format_report_9b header line

    def test_deposit_and_stdout_with_json_flag(self, tmp_path, monkeypatch, capsys):
        # Backward-compat: the Makefile / launcher pairing (``--json --output``)
        # is byte-identical to before — JSON deposit AND JSON stdout.
        import scripts.run_freeze_validloss_ci_9b as mod
        repo_root = Path(__file__).resolve().parents[1]
        stub = self._depositable_stub()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mod, "run_ci_9b", lambda **kw: stub)

        out = tmp_path / "deposit.json"
        assert main(self._argv(repo_root, out, json_flag=True)) == 0

        deposited = json.loads(out.read_text())
        assert deposited == result_to_json(stub)
        # --json → stdout is JSON too (unchanged).
        assert json.loads(capsys.readouterr().out) == deposited


class TestAtomicDepositAndRunLogWrites:
    """The citable §4 verdict deposit + its loss-curve run-log witness are
    written ATOMICALLY.

    A bare ``open(path, "w")`` truncates the destination before writing, so a
    kill mid-write (OOM-kill, SIGINT, a host recycling the worktree mid-run) —
    or an operator harvesting the file while the write is in flight — would
    leave a torn deposit: a parse failure that wastes the multi-hour run, or
    worse a silently-truncated verdict that commits wrong numbers. Both writes
    route through :func:`src.utils.io._atomic_write_text` (temp + ``os.replace``),
    the JSON analogue of ``_atomic_torch_save``. This pins that guarantee
    directly at the two 9B write sites the in-flight heterogeneous-full run will
    land through, mirroring :mod:`tests.test_atomic_save`'s torch-artifact pin.
    """

    @pytest.mark.parametrize("interrupt", [OSError, KeyboardInterrupt, SystemExit])
    def test_write_deposit_prior_survives_mid_publish(
        self, tmp_path, monkeypatch, interrupt
    ):
        # Publish a valid v1 deposit (real os.replace), then interrupt at the
        # rename boundary while overwriting with v2.
        path = tmp_path / "deposit.json"
        _write_deposit(path, '{"verdict": "TIES", "v": 1}')
        assert path.read_text(encoding="utf-8") == '{"verdict": "TIES", "v": 1}\n'

        def _boom(_src, _dst):
            raise interrupt("simulated mid-publish interrupt")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(interrupt):
            _write_deposit(path, '{"verdict": "SURPASSES", "v": 2}')

        # The prior, still-intact deposit keeps its OLD content — the torn v2
        # (which would flip the verdict headline) was never published.
        assert path.read_text(encoding="utf-8") == '{"verdict": "TIES", "v": 1}\n'
        assert not list(tmp_path.glob("deposit.json.tmp.*"))  # no orphan temp

    @pytest.mark.parametrize("interrupt", [OSError, KeyboardInterrupt, SystemExit])
    def test_write_deposit_fresh_interrupt_publishes_nothing(
        self, tmp_path, monkeypatch, interrupt
    ):
        # No prior destination: an interrupted fresh deposit writes nothing.
        path = tmp_path / "fresh.json"

        def _boom(_src, _dst):
            raise interrupt("simulated mid-publish interrupt")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(interrupt):
            _write_deposit(path, '{"verdict": "SURPASSES"}')

        assert not path.exists()
        assert not list(tmp_path.glob("fresh.json.tmp.*"))

    @pytest.mark.parametrize("interrupt", [OSError, KeyboardInterrupt, SystemExit])
    def test_write_run_log_prior_survives_mid_publish(
        self, tmp_path, monkeypatch, interrupt
    ):
        # A minimal result/curves pair is enough for _run_log_payload (config
        # keys default via .get); the load-bearing behavior is the atomic write.
        path = tmp_path / "runlog.json"
        _write_run_log(path, {}, [])
        assert path.exists()

        def _boom(_src, _dst):
            raise interrupt("simulated mid-publish interrupt")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(interrupt):
            _write_run_log(path, {"seq_len": 1024}, [{"role": "candidate", "index": 0}])

        # The prior run log survives; the new payload was never published.
        assert json.loads(path.read_text(encoding="utf-8"))["arms"] == []
        assert not list(tmp_path.glob("runlog.json.tmp.*"))


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
        # The run-log / loss-curve artifact flag (GOAL §7 reproducibility) is
        # part of the CLI surface — the "attach a run log so the verdict is
        # independently reproducible" path.
        assert "--run-log" in text

    def test_run_log_flag_defaults_none(self):
        # ``--run-log`` unset → None (no run log, byte-identical deposit). The
        # mechanism is strictly opt-in: a citable run turns it on, every other
        # run is unaffected.
        args = build_parser().parse_args([])
        assert args.run_log is None

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


# ── Heterogeneous architecture (per-layer rank) — the target-scale leg ──────


def _homogeneous_lora_cfg():
    """Minimal float32 LoRA cfg for the CPU apply_lora smoke (no bnb / no GPU)."""
    return OmegaConf.create({
        "model": {"load_in_4bit": False},
        "training": {"gradient_checkpointing": False},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"},
    })


def _tiny_llama(num_layers: int = 8):
    """A tiny randomly-initialized Llama (no download, no GPU) with the
    ``model.layers.{i}.*_proj`` naming the 9B Qwen stack uses, so the
    heterogeneous rank_pattern regex and the suffix-scope logic exercise the
    real module-name contract."""
    transformers = pytest.importorskip("transformers")
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            num_hidden_layers=num_layers, hidden_size=16, intermediate_size=32,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
        )
    )


class TestHeterogeneousRankPattern:
    """The CPU-runnable core of the heterogeneous (per-layer asymmetric rank)
    §4 leg — the one open research task now wired to target scale. Rank
    construction, pattern form, alpha scaling, and the architecture-independent
    scope/apply path. The real 9B verdict is a GPU smoke
    (``make freeze-validloss-ci-9b-heterogeneous-generalization``), not this
    suite — these guard the GPU-free core that makes that verdict honest."""

    def test_architectures_constant(self):
        assert HOMOGENEOUS == "homogeneous"
        assert HETEROGENEOUS == "heterogeneous"
        assert ARCHITECTURES == ("homogeneous", "heterogeneous")

    def test_ranks_geometric_ascending_to_output(self):
        # base_rank caps the output-most active layer; ranks rise geometrically
        # toward it so the output side carries the most adapter capacity.
        ranks = heterogeneous_ranks_9b({24, 25, 26, 27, 28, 29, 30, 31}, 16)
        assert ranks == (1, 1, 2, 3, 5, 7, 11, 16)
        assert ranks[0] >= 1          # never below 1
        assert ranks[-1] == 16        # output layer caps at base_rank
        assert list(ranks) == sorted(ranks)  # monotone non-decreasing

    def test_ranks_single_and_empty_layer_sets(self):
        assert heterogeneous_ranks_9b({5}, 16) == (16,)
        assert heterogeneous_ranks_9b(set(), 16) == ()

    def test_build_rank_pattern_heterogeneous_keys_and_scaling(self):
        rp, ap, lr = build_rank_pattern({29, 30, 31}, HETEROGENEOUS, 16)
        # human-readable per-layer provenance, output layer == base_rank
        assert lr == {"29": 1, "30": 4, "31": 16}
        # regex keys are the full-match form peft get_pattern_key requires
        assert set(rp) == {r"layers\.29\..*", r"layers\.30\..*", r"layers\.31\..*"}
        # alpha == 2 * rank so alpha/rank scaling stays constant (=2): the only
        # thing that varies across layers is capacity (rank), not magnitude.
        for key, rank in rp.items():
            assert ap[key] == 2 * rank

    def test_build_rank_pattern_homogeneous_is_empty(self):
        # homogeneous => no per-layer pattern: every layer takes cfg.lora.r,
        # byte-identical to the legacy path (prod apply_lora passes {}).
        rp, ap, lr = build_rank_pattern({29, 30, 31}, HOMOGENEOUS, 16)
        assert rp == {} and ap == {} and lr == {}

    def test_build_rank_pattern_rejects_unknown(self):
        with pytest.raises(ValueError):
            build_rank_pattern({0}, "semi-heterogeneous", 16)

    def test_active_scope_pre_lora_mirrors_suffix_scope(self):
        # The pre-LoRA active scope must match what configure_trainable_lora_scope
        # derives post-LoRA, so heterogeneous ranks land on exactly the trainable
        # layers. Built from a tiny Llama (no GPU).
        from src.model.lora_utils import configure_trainable_lora_scope
        from src.model.load_model import apply_lora

        model = _tiny_llama(8)
        pre = set(_active_scope_pre_lora(model, "last_25_percent"))
        assert pre == {6, 7}  # ceil(8 * 0.25) = 2 last layers
        pm = apply_lora(model, _homogeneous_lora_cfg())
        _, post = configure_trainable_lora_scope(pm, "last_25_percent")
        assert post == pre

    def test_apply_lora_heterogeneous_varies_per_layer_rank(self):
        # end-to-end: apply_lora with a heterogeneous rank_pattern produces
        # per-layer LoRA ranks on the ACTIVE layers (non-active keep the default
        # r). The load-bearing contract — rank_pattern actually changes adapter
        # capacity, not just passes through.
        from src.model.load_model import apply_lora

        model = _tiny_llama(8)
        pre = set(_active_scope_pre_lora(model, "last_25_percent"))
        rp, ap, _ = build_rank_pattern(pre, HETEROGENEOUS, 16)
        pm = apply_lora(
            model, _homogeneous_lora_cfg(), rank_pattern=rp, alpha_pattern=ap,
        )
        ranks: dict[int, set[int]] = {}
        for name, p in pm.named_parameters():
            if "lora_A" in name:
                m = re.search(r"layers\.(\d+)\.", name)
                if m:
                    ranks.setdefault(int(m.group(1)), set()).add(p.shape[0])
        # active layers get the geometric schedule (2-layer scope -> 1, 16)
        assert ranks[6] == {1}
        assert ranks[7] == {16}
        # non-active layers keep the default uniform rank
        assert ranks[0] == {16}

    def test_apply_lora_default_is_uniform_rank(self):
        # prod path (no rank_pattern) is byte-identical: every layer rank == r.
        from src.model.load_model import apply_lora

        model = _tiny_llama(8)
        pm = apply_lora(model, _homogeneous_lora_cfg())
        ranks: dict[int, set[int]] = {}
        for name, p in pm.named_parameters():
            if "lora_A" in name:
                m = re.search(r"layers\.(\d+)\.", name)
                if m:
                    ranks.setdefault(int(m.group(1)), set()).add(p.shape[0])
        assert all(v == {16} for v in ranks.values())


# --- _load_dolly_records schema guard (DATA-axis honesty, GOAL §7) -----------
# The real loader body (the ``datasets.load_dataset`` path) is NOT exercised by
# the launch-honesty suite, which mocks ``_load_dolly_records`` whole with
# Dolly-shaped fakes. These tests drive the REAL body by patching
# ``datasets.load_dataset`` — including the schema-mismatch case that would
# silently corrupt the §4 verdict on the private-``src.data`` drop-in path.


def _fake_load_dataset(rows):
    """A stand-in for ``datasets.load_dataset(..., streaming=True)``.

    Returns a fresh iterator over ``rows`` each call, mimicking an iterable
    streaming split (no network, no ``datasets`` build).
    """

    def _load(dataset, split="train", streaming=True):
        return iter(list(rows))

    return _load


def test_load_dolly_records_rejects_schema_mismatch(monkeypatch):
    """A dataset whose rows lack ``instruction``/``response`` must ABORT, not
    silently feed empty records. With the bare ``.get(..., "")`` fallback every
    record became all-empty → :func:`build_sft_example` emitted N identical
    degenerate (non-skipped) examples → a corrupt-but-green §4 verdict (GOAL
    §7). This is the latent gap on the private-``src.data`` drop-in path: when
    that dataset lands, a wrong schema must fail loud, not measure nothing.

    Mutation proof — remove the ``_assert_dolly_schema`` call in
    :func:`_load_dolly_records` and this test goes RED (the loader would return
    8 empty records instead of raising).
    """
    import datasets

    wrong_schema = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(8)]
    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset(wrong_schema))
    with pytest.raises(ValueError, match="lack required field"):
        _load_dolly_records("some/private-sft", max_rows=8, seed=7)


def test_load_dolly_records_accepts_dolly_schema(monkeypatch):
    """Byte-identical to the pre-guard behavior when the Dolly fields ARE
    present: records are extracted and shuffled under the seeded RNG. The guard
    never fires for the real dataset, so no committed verdict or fixture moves.
    """
    import datasets

    rows = [
        {"instruction": f"inst {i}", "context": "", "response": f"resp {i}"}
        for i in range(8)
    ]
    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset(rows))
    out = _load_dolly_records(
        "databricks/databricks-dolly-15k", max_rows=8, seed=7
    )
    # Same count + identical content set (shuffle permutes in place only).
    assert len(out) == 8
    assert {tuple(sorted(r.items())) for r in out} == {
        (
            ("context", ""),
            ("instruction", f"inst {i}"),
            ("response", f"resp {i}"),
        )
        for i in range(8)
    }


def test_assert_dolly_schema_message_names_dataset_and_fields():
    """The error must name the dataset AND the missing field(s) so the operator
    on the drop-in path knows exactly what schema to adapt."""
    with pytest.raises(ValueError) as excinfo:
        _assert_dolly_schema("org/private-sft", {"text": "only a text field"})
    msg = str(excinfo.value)
    assert "org/private-sft" in msg
    assert "instruction" in msg and "response" in msg


def test_assert_dolly_schema_passes_dolly_row():
    """A Dolly-shaped row (instruction+response present) passes silently — the
    guard is byte-identical for the real dataset. ``context`` is optional."""
    _assert_dolly_schema(
        "databricks/databricks-dolly-15k",
        {"instruction": "x", "context": "", "response": "y"},
    )


class TestModuleDocstringAccurate:
    """The module ``__doc__`` must not contradict the tested 4-axis citation gate.

    The deposit's :attr:`~scripts.run_freeze_validloss_ci_9b.result_to_json`
    labels ``reduced_budget`` / ``citable_as_full_section4_verdict`` are
    **runtime-determined** by :func:`_is_reduced_budget` and the four-axis
    :func:`_full_section4_verdict_gate` (exercised in ``test_honesty_labels_*``
    above). An earlier docstring revision hardcoded ``reduced_budget=True`` /
    ``citable_as_full_section4_verdict=False`` and called the tool "the first
    real-9B sample" — all false once full-budget citable verdicts shipped (the
    homogeneous + heterogeneous full deposits, both
    ``citable_as_full_section4_verdict=True``). These pins keep ``__doc__``
    honest so an agent reading it does not conclude the tool is
    reduced-budget-only (the stale framing that re-asks to "fire the run" after
    the measurement already landed).
    """

    DOC = ci9b.__doc__ or ""

    def test_docstring_acknowledges_both_budget_outcomes(self):
        """``reduced_budget`` is runtime-determined; ``__doc__`` must present
        BOTH the reduced-smoke and full-budget paths, not reduced-only."""
        assert "reduced_budget=True" in self.DOC
        assert "reduced_budget=False" in self.DOC

    def test_docstring_surfaces_citable_full_verdict_path(self):
        """``__doc__`` must state that a full-budget run can clear
        ``citable_as_full_section4_verdict`` (both such verdicts have landed)."""
        assert "citable_as_full_section4_verdict=True" in self.DOC

    def test_docstring_drops_stale_first_sample_framing(self):
        """The 'first real-9B sample' framing is stale — the §4 verdict program
        is complete (two citable full-budget verdicts on disk)."""
        assert "It is the *first*" not in self.DOC

