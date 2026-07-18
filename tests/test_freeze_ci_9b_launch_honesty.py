"""End-to-end launch-honesty dry-run of the heterogeneous §4 freeze-ci-9b path.

The recent per-commit fixes each closed ONE silent-corruption site in ISOLATION:

* ``e3c9155`` wired the full-budget heterogeneous launch path;
* ``1c2c833`` stopped ``_candidate_cost_reduction`` silently nulling the §4 cost axis;
* ``d9ca7f5`` stopped G3 silently dropping unscored eval tasks;
* ``54a4cd8`` made the citable §4 deposit + run-log + io-leaf JSON writes atomic;
* ``4afc5e9`` classified a run-time CUDA OOM as the launcher's tempfail;
* ``e823641`` rejected unknown ``freeze_layer`` specs instead of silently freezing
  the last layer.

Each was proven against a *slice*: a hand-built ``_stub_result`` dict, a
``run_ci_9b`` replaced wholesale with a stub, or a committed fixture replay.
NONE proved the **assembled** heterogeneous path is silent-free end-to-end —
the integration scale where a gap that hides behind the interaction of the
parts (but not behind any one slice) would surface.

This module is that proof. It drives the REAL
``run_ci_9b(architecture="heterogeneous")`` → ``result_to_json`` →
``_write_deposit`` / ``_write_run_log`` with ONLY the GPU/expensive boundaries
stubbed (a tiny CPU Llama for the 9B loader, in-memory Dolly records for the
dataset download, a deterministic toy arm for the GPU training loop), then
asserts the five honesty invariants the per-commit fixes enforce — each at
integration scale, against outputs the REAL assembly code produced.

GPU-independent: the stubs replace the model download, the dataset download,
and the per-step GPU arm training only. Everything else — ``apply_lora``, the
heterogeneous rank pattern, ``configure_trainable_lora_scope``, the scope-drift
guard, ``_collect_arms`` (+ ledger banking), ``surrogate_valid_loss_ci``, the
result-dict construction, ``result_to_json``, the cost accountant, the regime
classifier, ``is_cuda_oom``, and the atomic writes — runs for REAL on CPU. So a
silent-corruption site that lives in the assembly, not any one slice, surfaces
here, not in the slice tests.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from scripts.run_freeze_validloss_ci_9b import (
    EVIDENCE_HASH_KEYS,
    EXIT_DONE,
    EXIT_UNEXPECTED,
    HETEROGENEOUS,
    RUN_LOG_SCHEMA_VERSION,
    _evidence_hash,
    _run_log_sha256,
    _write_deposit,
    candidate_order_9b,
    is_cuda_oom,
    result_to_json,
    run_ci_9b,
)


# ── the stubbed GPU/expensive boundaries ──────────────────────────────────────
#
# Every stub below replaces something that NEEDS a network download or a GPU.
# Nothing below substitutes for the honesty/assembly logic under test — that
# (run_ci_9b, result_to_json, the writes, the gate, the controller) runs real.


class _CharTokenizer:
    """One id per character — ``build_sft_example``'s prefix-is-strict-prefix
    contract holds without a tokenizer download."""

    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": [ord(c) for c in text]})()


def _tiny_llama(num_layers: int = 8):
    """A tiny randomly-initialized Llama (no download, no GPU) whose
    ``model.layers.{i}.*_proj`` naming matches the 9B stack, so the real
    ``apply_lora`` + heterogeneous ``rank_pattern`` + ``configure_trainable_
    lora_scope`` + scope-drift guard all exercise their real module-name
    contract."""
    transformers = pytest.importorskip("transformers")
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            num_hidden_layers=num_layers, hidden_size=16, intermediate_size=32,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
        )
    )


def _cpu_cfg(*, max_steps: int = 6, base_rank: int = 16, lr: float = 1e-4):
    """Minimal float32 cfg for the REAL apply_lora + run_ci_9b on CPU
    (``load_in_4bit=False`` skips bnb/GPU). Carries every key run_ci_9b and
    apply_lora read, so the assembled path needs no real 9B config file. ``lr``
    is exposed so a resume-fingerprint test can vary the learning rate (the
    optimizer ``lr`` every arm's ``AdamW`` reads) against a shared ledger."""
    return OmegaConf.create({
        "model": {"name_or_path": "tiny-llama-stub", "load_in_4bit": False},
        "training": {
            "trainable_lora_scope": "last_25_percent",
            "learning_rate": lr,
            "max_steps": max_steps,
            "gradient_checkpointing": False,
        },
        "lora": {
            "r": base_rank, "alpha": 32, "dropout": 0.0,
            "target_modules": "all-linear",
        },
    })


def _fake_dolly_records(n: int = 12):
    """In-memory Dolly-shaped records (no ``datasets`` network download)."""
    return [
        {"instruction": f"Summarize task {i}.", "context": "", "response": f"Done reply number {i}."}
        for i in range(n)
    ]


def _make_toy_arm(base_seed: int):
    """A deterministic stand-in for the GPU ``arm_valid_loss_9b`` training loop.

    Returns a closure with the EXACT signature of ``arm_valid_loss_9b`` so it
    drops into the real ``_make_arm_runner``. The valid_loss is a deterministic
    function of the spec's seed bucket — candidate arms (seed ``base_seed+i``)
    score lower than surrogate arms (``base_seed+100+i``) — so the assembled A/B
    is a well-formed SURPASSES without spending GPU. The provenance records the
    frozen layers the candidate's schedule would freeze (``order[:depth]``) and a
    generalization-regime ``final_ce_train_loss`` (~1.5), so the assembled
    deposit's cost axis, freeze-layer record, and regime label are all real.
    """

    def _arm(
        model, order, seed, *, scope, active_indices, train_batches,
        valid_batches, device, total_steps, warmup_steps, depth, spacing, lr,
        use_local_loss, loss_curve_sink=None,
    ):
        del scope, active_indices, train_batches, valid_batches, device
        del warmup_steps, spacing, lr, use_local_loss
        rng = random.Random(seed * 1_000_003)
        # Candidate seeds are < base_seed+50; surrogate/control/baseline higher.
        valid_loss = 1.5 + 0.001 * max(0, seed - base_seed) + rng.uniform(-1e-6, 1e-6)
        frozen = sorted(int(x) for x in order[:depth]) if depth > 0 else []
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if loss_curve_sink is not None:
            for _ in range(total_steps):
                loss_curve_sink.append(float(1.45 + rng.uniform(-0.02, 0.02)))
        provenance = {
            "frozen_layers": frozen,
            "n_trainable_params": n_trainable,
            "last_train_loss": float(valid_loss),
            "final_ce_train_loss": 1.507,  # generalization regime (~1.5)
        }
        return float(valid_loss), provenance

    return _arm


def _assemble(monkeypatch, tmp_path, *, run_log=True):
    """Drive the REAL assembled heterogeneous path; return ``(result, deposit,
    deposit_path, run_log_path)``.

    Stubs ONLY the network/GPU boundaries (tokenizer, base model, dataset,
    arm). Returns the deposit the real ``run_ci_9b`` → ``result_to_json`` →
    ``_write_deposit`` / ``_write_run_log`` assembly produced, so a caller can
    assert any honesty invariant at integration scale.
    """
    import scripts.run_freeze_validloss_ci_9b as mod

    monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
    monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
    monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())
    monkeypatch.setattr(mod, "arm_valid_loss_9b", _make_toy_arm(base_seed=7))

    run_log_path = str(tmp_path / "runlog.json") if run_log else None
    # seq_len is generous because the char tokenizer emits one id per CHARACTER
    # (a real BPE tokenizer would fit the same record in ~20 ids): the ChatML
    # user-turn prefix alone is ~50 chars, so a small seq_len truncates away the
    # response and build_sft_example drops every record as prompt-dominant.
    result = run_ci_9b(
        cfg=_cpu_cfg(max_steps=6, base_rank=16),
        device=torch.device("cpu"),
        seq_len=256,
        train_examples=4,
        valid_examples=4,
        total_steps=6,
        warmup_steps=1,
        depth=1,
        spacing=2,
        n_candidate=2,
        n_surrogate=2,
        base_seed=7,
        dataset="dolly",
        max_dataset_rows=12,
        architecture=HETEROGENEOUS,
        run_log_path=run_log_path,
    )
    deposit = result_to_json(result)
    deposit_path = tmp_path / "deposit.json"
    _write_deposit(str(deposit_path), json.dumps(deposit, indent=2))
    return result, deposit, deposit_path, run_log_path


# ── (1) cost axis non-null + accounted, at integration scale (1c2c833) ────────


class TestAssembledCostAxis:
    def test_heterogeneous_deposit_carries_nonnull_cost_axis(self, monkeypatch, tmp_path):
        # The isolated TestCandidateCostReduction feeds a HAND-BUILT _stub_result;
        # this runs the real run_ci_9b assembly under heterogeneous ranks and
        # checks the cost axis survives the REAL result_to_json serialization
        # (the 1c2c833 invariant, at the assembled scale where it could be
        # silently nulled by a KeyError/type mismatch in the real result shape).
        _result, deposit, _dp, _rl = _assemble(monkeypatch, tmp_path)
        cost = deposit["candidate_cost_reduction"]
        assert cost is not None, (
            "candidate_cost_reduction was silently nulled in the assembled "
            "heterogeneous deposit — the §4 cost-success head (1c2c833) dropped."
        )
        assert cost["reduction_rate"] > 0.0
        # Level-1 honesty: the arithmetic figure carries its realized cap so a
        # reader cannot quote the arithmetic reduction as a realized saving.
        assert "realized_reduction_rate" in cost
        assert cost["cost_model"] == "uniform_per_layer"


# ── (2) no unscored eval task silently dropped from the gate (d9ca7f5) ────────


class TestAssembledGateNoSilentDrop:
    def test_g3_names_requested_but_unscored_task(self, tmp_path):
        # A task whose lm-eval metric wasn't recognized (truthfulqa_mc2 reports
        # ``mc2``, not acc/acc_norm) silently fails to enter task_relative_drops.
        # Checking the REQUEST list would let that drop fool G3.3 into PASSING;
        # the fix checks the compared-tasks reality and names the dropped task.
        from scripts.evaluate_paper_gates import _check_g3

        eval_path = tmp_path / "external_eval_results.json"
        eval_path.write_text(json.dumps({
            "tasks": ["truthfulqa_mc2", "arc_easy", "hellaswag"],
            "comparison": {
                "aggregate_relative_drop": 0.001,
                "task_relative_drops": {"arc_easy": 0.001, "hellaswag": 0.001},
                # truthfulqa_mc2 is REQUESTED but absent here — it was never
                # scored, the silent-drop case d9ca7f5 closes.
                "compared_tasks": ["arc_easy", "hellaswag"],
            },
        }))
        g3 = _check_g3({}, external_eval_path=eval_path)
        g33 = next(c for c in g3["checks"] if c["check"] == "G3.3_required_tasks_present")
        assert g33["pass"] is False, (
            "G3.3 passed despite truthfulqa_mc2 being requested but unscored — "
            "the silent-drop (d9ca7f5) regressed."
        )
        assert "truthfulqa_mc2" in g33["detail"], (
            "G3.3 did not NAME the requested-but-unscored task in its detail."
        )


# ── (3) atomic + complete deposit/run-log writes (54a4cd8) ───────────────────


class TestAssembledAtomicCompleteWrites:
    def test_assembled_deposit_survives_mid_publish_interrupt(self, monkeypatch, tmp_path):
        # The assembled deposit payload (from the real result_to_json) must be
        # published atomically: an interrupt at the os.replace boundary leaves
        # the prior deposit intact, never a torn verdict. Pinned in isolation
        # for a hand-built payload (54a4cd8); here the payload is the REAL
        # assembled heterogeneous deposit.
        _result, deposit, _dp, _rl = _assemble(monkeypatch, tmp_path)
        path = tmp_path / "live.json"
        _write_deposit(str(path), json.dumps({"verdict": "TIES", "v": 1}))
        prior = path.read_text(encoding="utf-8")

        def _boom(_src, _dst):
            raise OSError("simulated mid-publish interrupt")
        monkeypatch.setattr(__import__("os"), "replace", _boom)
        with pytest.raises(OSError):
            _write_deposit(str(path), json.dumps(deposit))
        # The torn overwrite was never published — the prior deposit stands.
        assert path.read_text(encoding="utf-8") == prior

    def test_no_silently_omitted_freeze_layer(self, monkeypatch, tmp_path):
        # The candidate's frozen layer must be present in the deposit's
        # provenance AND in the run-log arm — a silently-omitted freeze layer
        # would hide which layer carried the adapter capacity (54a4cd8's
        # "complete" half, which the atomic-write-only slice tests cannot check).
        result, deposit, _dp, run_log_path = _assemble(monkeypatch, tmp_path)
        # The candidate freezes the FIRST ``depth`` layers of its output-first
        # order (candidate_order_9b is descending — the output-most layer freezes
        # first), then the deposit stores that set sorted. Slice in order FIRST,
        # then sort, or layer 7 (output-most) is wrongly expected as layer 6.
        expected_frozen = sorted(
            candidate_order_9b(set(result["active_scope"]))[:result["depth"]]
        )
        deposited = deposit["candidate_provenance"][0]["frozen_layers"]
        assert deposited == expected_frozen, (
            f"candidate frozen layers silently changed in the assembled deposit: "
            f"{deposited} != intended {expected_frozen}"
        )
        # The run-log arm must agree with the deposit's provenance — the two
        # artifacts cannot disagree about what froze.
        run_log = json.loads(Path(run_log_path).read_text())
        cand_arm = next(a for a in run_log["arms"] if a["role"] == "candidate")
        assert cand_arm["frozen_layers"] == expected_frozen

    def test_assembled_runlog_hash_round_trips(self, monkeypatch, tmp_path):
        # The deposit's run_log_sha256 must equal a hash recomputed from the
        # written run-log file — the verifier's reproducibility round-trip, at
        # the assembled scale (the committed-fixture witness test only checks a
        # static fixture).
        _result, deposit, _dp, run_log_path = _assemble(monkeypatch, tmp_path)
        payload = json.loads(Path(run_log_path).read_text())
        assert payload["schema_version"] == RUN_LOG_SCHEMA_VERSION
        assert deposit["run_log_sha256"] == _run_log_sha256(payload), (
            "deposit run_log_sha256 disagrees with the written run-log file — "
            "the reproducibility witness (54a4cd8) does not round-trip assembled."
        )


# ── (4) run-time OOM from inside the arm loop → tempfail (4afc5e9) ───────────


class TestAssembledOomTempfail:
    def test_oom_from_real_arm_loop_is_tempfail_and_banks_ledger(
        self, monkeypatch, tmp_path,
    ):
        # The isolated TestCudaOomTempfail REPLACES run_ci_9b wholesale. This
        # injects the OOM from INSIDE the real _collect_arms arm loop (arm 2 of
        # the sweep), so the REAL run_ci_9b finally-cleanup + propagation runs,
        # and the banked arm survives in the ledger — the resume design the
        # tempfail classification exists to protect.
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())

        toy = _make_toy_arm(base_seed=7)
        calls = {"n": 0}

        def _oom_arm(*a, **kw):
            calls["n"] += 1
            if calls["n"] > 1:  # OOM on the 2nd arm (after 1 banked)
                raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
            return toy(*a, **kw)

        monkeypatch.setattr(mod, "arm_valid_loss_9b", _oom_arm)
        ledger_path = str(tmp_path / "ledger.jsonl")

        with pytest.raises(RuntimeError) as excinfo:
            run_ci_9b(
                cfg=_cpu_cfg(max_steps=6, base_rank=16),
                device=torch.device("cpu"),
                seq_len=256, train_examples=4, valid_examples=4,
                total_steps=6, warmup_steps=1, depth=1, spacing=2,
                n_candidate=2, n_surrogate=2, base_seed=7,
                dataset="dolly", max_dataset_rows=12,
                architecture=HETEROGENEOUS, ledger_path=ledger_path,
            )
        # The propagated exception is recognized as a retryable OOM, not a fatal
        # logic error — the re-classification 4afc5e9 added, reached through the
        # REAL assembly (not a replaced run_ci_9b).
        assert is_cuda_oom(excinfo.value) is True
        # The 1 banked arm survives in the ledger — the resume the tempfail
        # exists to enable. (No ledger line = the tempfail would waste all GPU.)
        lines = [
            ln for ln in Path(ledger_path).read_text().splitlines()
            if ln.strip() and json.loads(ln).get("type") == "arm"
        ]
        assert len(lines) == 1, (
            f"expected exactly the 1 pre-OOM arm banked, found {len(lines)} — "
            "the ledger did not bank incrementally before the mid-loop OOM."
        )


# ── (5) unknown freeze_layer spec rejected (e823641) ─────────────────────────


class TestAssembledFreezeSpecReject:
    def test_unknown_spec_rejected_not_silently_last_layer(self):
        # The §4 experimental variable is WHICH layer freezes; a typo'd spec
        # that silently froze the LAST layer would corrupt a run with no signal
        # (e823641). The guard fires at construction, before any training.
        from src.tg_lora.progressive_freeze import ProgressiveFreezeController

        with pytest.raises(ValueError, match="last_active"):
            ProgressiveFreezeController(
                start_cycle=1, active_layer_indices={6, 7},
                freeze_layer="first_active",  # the typo/speculative value
            )
        # bool is an int subclass — True must not silently pin layer 1.
        with pytest.raises(TypeError):
            ProgressiveFreezeController(
                start_cycle=1, active_layer_indices={6, 7}, freeze_layer=True,
            )

    def test_valid_specs_accepted(self):
        # The rejection is narrow: an int index and the literal "last_active"
        # are the only valid specs, and both construct cleanly (so the §4 path,
        # which uses the default, is unaffected).
        from src.tg_lora.progressive_freeze import ProgressiveFreezeController

        ProgressiveFreezeController(
            start_cycle=1, active_layer_indices={6, 7}, freeze_layer="last_active",
        )
        ProgressiveFreezeController(
            start_cycle=1, active_layer_indices={6, 7}, freeze_layer=7,
        )


# ── launch-readiness: one self-consistent assembled artifact ─────────────────


class TestAssembledLaunchReadiness:
    def test_assembled_heterogeneous_deposit_is_self_consistent(self, monkeypatch, tmp_path):
        # The single assertion that ties the assembled pass together: the real
        # run_ci_9b(heterogeneous) → result_to_json → _write_deposit produces a
        # complete, parseable, self-consistent artifact — architecture stamped,
        # regime classified, evidence hash present and reproducible, and a
        # verdict field that matches the stored floats. If any per-commit
        # invariant regressed at the assembled scale, this would fail too.
        result, deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        loaded = json.loads(deposit_path.read_text())

        assert loaded["architecture"] == HETEROGENEOUS
        assert loaded["lora_rank_pattern"] == result.get("lora_rank_pattern")
        assert loaded["regime"] == "generalization"
        # The evidence hash is reproducible: re-stamp the loaded deposit and it
        # matches the frozen one (no coordinated drift slipped through).
        assert loaded["evidence_hash"] == _evidence_hash(loaded)
        # The verdict field agrees with the stored candidate/surrogate means —
        # a verdict painted on that disagrees with the floats fails here.
        assert loaded["candidate_mean"] < loaded["surrogate_mean"]
        assert loaded["verdict"] in ("SURPASSES", "TIES", "UNDERSHOOTS")


# ── (6) a LoRA-config change does not replay stale ledger arms ────────────────
#
# The five invariants above each pin a per-commit fix in isolation. This one
# pins the gap the assembled dry-run surfaced on THIS iteration: the resume-
# ledger fingerprint is the SOLE config-gate (``load_ledger`` replays an arm
# iff its header fingerprint matches exactly), yet it carried no LoRA adapter
# config. Under HETEROGENEOUS, ``lora_r`` sets the WHOLE per-layer geometric
# schedule — the §4 experimental variable itself — so a re-run at a different
# base rank shared the old run's fingerprint and silently replayed its
# WRONG-RANK arms from the ledger: a corrupt-but-green §4 verdict (GOAL §7
# cardinal failure). The fix threads lora_r / lora_alpha / lora_dropout /
# target_modules into ``_config_fingerprint``; this proves at the assembled
# scale that the second run re-executes every arm rather than replaying them.


class TestAssembledLoraFingerprintGate:
    def test_heterogeneous_rerun_at_new_base_rank_re_executes_not_replays(
        self, monkeypatch, tmp_path,
    ):
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(
            mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records()
        )

        calls = {"n": 0}
        base_toy = _make_toy_arm(base_seed=7)

        def _counting_arm(*a, **kw):
            calls["n"] += 1
            return base_toy(*a, **kw)

        monkeypatch.setattr(mod, "arm_valid_loss_9b", _counting_arm)
        ledger_path = str(tmp_path / "ledger.jsonl")
        common = dict(
            device=torch.device("cpu"), seq_len=256, train_examples=4,
            valid_examples=4, total_steps=6, warmup_steps=1, depth=1,
            spacing=2, n_candidate=2, n_surrogate=2, base_seed=7,
            dataset="dolly", max_dataset_rows=12, architecture=HETEROGENEOUS,
            ledger_path=ledger_path,
        )

        # First run at base rank 16 banks all 4 arms (2 candidate + 2 surrogate).
        calls["n"] = 0
        run1 = run_ci_9b(cfg=_cpu_cfg(max_steps=6, base_rank=16), **common)
        assert run1["resumed_arm_count"] == 0
        assert calls["n"] == 4, (
            f"the rank-16 run invoked the arm runner {calls['n']} time(s), "
            "expected 4 (2 candidate + 2 surrogate) — harness setup drift."
        )

        # Second run at base rank 32 — a DIFFERENT geometric schedule, i.e. a
        # genuinely different §4 experiment — against the SAME ledger. Before the
        # fix the fingerprints matched and these 4 arms replayed silently from
        # the rank-16 ledger; after the fix lora_r distinguishes them so every
        # arm re-runs. The replay would be silent: the toy arm is seed-based, so
        # a replayed arm's LOSS is identical — only the replay COUNT (and the
        # differing lora_rank_pattern) exposes the corruption.
        calls["n"] = 0
        run2 = run_ci_9b(cfg=_cpu_cfg(max_steps=6, base_rank=32), **common)
        assert run2["resumed_arm_count"] == 0, (
            "the rank-32 heterogeneous re-run replayed rank-16 arms from the "
            "ledger — the LoRA-config fingerprint gate (GOAL §7) regressed: "
            f"resumed {run2['resumed_arm_count']} arm(s) instead of re-executing."
        )
        assert calls["n"] == 4, (
            f"the rank-32 re-run invoked the arm runner {calls['n']} time(s), "
            "expected 4 — stale rank-16 ledger arms were silently replayed "
            "under a different per-layer rank schedule."
        )
        # The two runs ARE different experiments (different per-layer ranks), so
        # a replay would have seeded the verdict with arms trained under the
        # wrong schedule — the corrupt-but-green case the gate exists to prevent.
        assert run1["lora_rank_pattern"] != run2["lora_rank_pattern"]


# ── (7) a learning_rate change does not replay stale ledger arms ──────────────
#
# Sibling of (6) for the OTHER config dimension the resume-ledger fingerprint
# missed: ``learning_rate``. ``arm_valid_loss_9b`` builds every arm's optimizer
# as ``AdamW(trainable, lr=learning_rate)``, so two runs at different learning
# rates are NOT interchangeable — yet ``_config_fingerprint`` carried every
# other training-regime field (total_steps / warmup_steps / use_local_loss) and
# the whole LoRA adapter config, but NOT ``lr``. An operator resuming an
# interrupted run after tuning ``lr`` (the single most-edited hyperparameter)
# would hit a matching fingerprint and silently replay the old-``lr`` arms:
# corrupt-but-green (GOAL §7), the same class as the ``lora_r`` gap (4ad9a73).
# The fix threads ``learning_rate`` into ``_config_fingerprint``; this proves
# at the assembled scale that the second run re-executes every arm. Unlike
# ``lora_r`` (which rewrites ``lora_rank_pattern``), ``lr`` leaves no separate
# deposit field, so only the replay COUNT exposes the corruption — the toy arm
# is seed-based, so a replayed arm's LOSS is identical at either ``lr``.


class TestAssembledLearningRateFingerprintGate:
    def test_heterogeneous_rerun_at_new_lr_re_executes_not_replays(
        self, monkeypatch, tmp_path,
    ):
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(
            mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records()
        )

        calls = {"n": 0}
        base_toy = _make_toy_arm(base_seed=7)

        def _counting_arm(*a, **kw):
            calls["n"] += 1
            return base_toy(*a, **kw)

        monkeypatch.setattr(mod, "arm_valid_loss_9b", _counting_arm)
        ledger_path = str(tmp_path / "ledger.jsonl")
        common = dict(
            device=torch.device("cpu"), seq_len=256, train_examples=4,
            valid_examples=4, total_steps=6, warmup_steps=1, depth=1,
            spacing=2, n_candidate=2, n_surrogate=2, base_seed=7,
            dataset="dolly", max_dataset_rows=12, architecture=HETEROGENEOUS,
            ledger_path=ledger_path,
        )

        # First run at lr=1e-4 banks all 4 arms (2 candidate + 2 surrogate).
        calls["n"] = 0
        run1 = run_ci_9b(cfg=_cpu_cfg(max_steps=6, base_rank=16, lr=1e-4), **common)
        assert run1["resumed_arm_count"] == 0
        assert calls["n"] == 4, (
            f"the lr=1e-4 run invoked the arm runner {calls['n']} time(s), "
            "expected 4 (2 candidate + 2 surrogate) — harness setup drift."
        )

        # Second run at lr=2e-4 — the same geometry, the same ranks, the ONLY
        # difference is the AdamW learning rate every arm trained under — against
        # the SAME ledger. Before the fix the fingerprints matched (lr was not in
        # the gate) and these 4 arms replayed silently from the lr=1e-4 ledger;
        # after the fix learning_rate distinguishes them so every arm re-runs.
        # The replay would be silent: the toy arm is seed-based, so a replayed
        # arm's LOSS is identical at either lr — only the replay COUNT exposes it.
        calls["n"] = 0
        run2 = run_ci_9b(cfg=_cpu_cfg(max_steps=6, base_rank=16, lr=2e-4), **common)
        assert run2["resumed_arm_count"] == 0, (
            "the lr=2e-4 heterogeneous re-run replayed lr=1e-4 arms from the "
            "ledger — the learning_rate fingerprint gate (GOAL §7) regressed: "
            f"resumed {run2['resumed_arm_count']} arm(s) instead of re-executing."
        )
        assert calls["n"] == 4, (
            f"the lr=2e-4 re-run invoked the arm runner {calls['n']} time(s), "
            "expected 4 — stale lr=1e-4 ledger arms were silently replayed under "
            "a different optimizer learning rate (corrupt-but-green, GOAL §7)."
        )
        # lr leaves no separate deposit field (unlike lora_r → lora_rank_pattern),
        # so the re-execution count above is the SOLE corruption signal: the two
        # runs share every deposit-visible field yet are different experiments.
        assert run1["lora_rank_pattern"] == run2["lora_rank_pattern"]


# ── (8) a max_dataset_rows change does not replay stale ledger arms ───────────
#
# Sibling of (6)/(7) for the DATA-SAMPLING dimension the resume-ledger
# fingerprint missed: ``max_dataset_rows``. ``_load_dolly_records`` draws the
# first ``max_dataset_rows`` records and shuffles that POOL with
# ``random.Random(base_seed)``; a Fisher–Yates shuffle of a length-N list
# permutes differently than one of length-M under the SAME seed, so the
# train/valid slice (``offset=0`` / ``offset=train_examples``) lands on DIFFERENT
# record content. Two runs differing only in ``max_dataset_rows`` are therefore
# different §4 experiments, yet before the fix they shared a fingerprint and the
# 2nd replayed the 1st's arms — corrupt-but-green (GOAL §7), the same class as
# the ``lora_r`` (4ad9a73) and ``learning_rate`` (d6af3cd) gaps. The fix threads
# ``max_dataset_rows`` into ``_config_fingerprint``; this proves at the assembled
# scale that the second run re-executes every arm rather than replaying them.


class TestAssembledMaxDatasetRowsFingerprintGate:
    def test_heterogeneous_rerun_at_new_pool_size_re_executes_not_replays(
        self, monkeypatch, tmp_path,
    ):
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(
            mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records()
        )

        calls = {"n": 0}
        base_toy = _make_toy_arm(base_seed=7)

        def _counting_arm(*a, **kw):
            calls["n"] += 1
            return base_toy(*a, **kw)

        monkeypatch.setattr(mod, "arm_valid_loss_9b", _counting_arm)
        ledger_path = str(tmp_path / "ledger.jsonl")
        common = dict(
            cfg=_cpu_cfg(max_steps=6, base_rank=16),
            device=torch.device("cpu"), seq_len=256, train_examples=4,
            valid_examples=4, total_steps=6, warmup_steps=1, depth=1,
            spacing=2, n_candidate=2, n_surrogate=2, base_seed=7,
            dataset="dolly", architecture=HETEROGENEOUS,
            ledger_path=ledger_path,
        )

        # First run with a 12-record pool banks all 4 arms (2 candidate + 2
        # surrogate).
        calls["n"] = 0
        run1 = run_ci_9b(max_dataset_rows=12, **common)
        assert run1["resumed_arm_count"] == 0
        assert calls["n"] == 4, (
            f"the pool-12 run invoked the arm runner {calls['n']} time(s), "
            "expected 4 (2 candidate + 2 surrogate) — harness setup drift."
        )

        # Second run with a 4000-record pool — a DIFFERENT seeded shuffle, i.e.
        # different train/valid record content (a genuinely different §4
        # experiment) — against the SAME ledger. Before the fix the fingerprints
        # matched (max_dataset_rows was not in the gate) and these 4 arms
        # replayed silently from the pool-12 ledger; after the fix
        # max_dataset_rows distinguishes them so every arm re-runs. The replay
        # would be silent: the toy arm is seed-based, so a replayed arm's LOSS is
        # identical at either pool size — only the replay COUNT exposes it.
        calls["n"] = 0
        run2 = run_ci_9b(max_dataset_rows=4000, **common)
        assert run2["resumed_arm_count"] == 0, (
            "the pool-4000 heterogeneous re-run replayed pool-12 arms from the "
            "ledger — the max_dataset_rows fingerprint gate (GOAL §7) regressed: "
            f"resumed {run2['resumed_arm_count']} arm(s) instead of re-executing."
        )
        assert calls["n"] == 4, (
            f"the pool-4000 re-run invoked the arm runner {calls['n']} time(s), "
            "expected 4 — stale pool-12 ledger arms were silently replayed under "
            "a different data-subsample shuffle (corrupt-but-green, GOAL §7)."
        )
        # max_dataset_rows leaves no separate deposit field (unlike lora_r →
        # lora_rank_pattern), so the re-execution count above is the SOLE
        # corruption signal: the two runs share every deposit-visible field yet
        # are different experiments (different train/valid record content).
        assert run1["lora_rank_pattern"] == run2["lora_rank_pattern"]


# ── (9) EVERY fingerprint dim re-executes at assembled scale, not just the 3 ──
#
# Siblings (6)/(7)/(8) proved three dims (lora_r / learning_rate /
# max_dataset_rows) re-execute rather than silently replay. But
# ``_config_fingerprint`` gates ~13 OTHER run-determining fields
# (total_steps, warmup_steps, depth, spacing, seq_len, train_examples,
# valid_examples, use_local_loss, base_seed, dataset, lora_alpha,
# lora_dropout, lora_target_modules) — each one changes an arm's result, and
# each was only ever proven as a pure-function unit assertion
# (``test_fingerprint_changes_with_*`` in test_freeze_validloss_ci_9b_resume.py),
# NEVER at the assembled scale. The assembled scale is where a WIRING gap would
# hide: the dim flows into ``arm_valid_loss_9b`` (so it changes the result) but
# not into the ``_config_fingerprint`` call in ``run_ci_9b`` (so the ledger
# fingerprint matches and the 2nd run replays the 1st's now-WRONG arms) — or the
# reverse. A unit test on the pure ``_config_fingerprint`` function cannot catch
# that, because the unit test never runs ``run_ci_9b``'s wiring.
#
# This parametrizes the same two-run replay check (6)/(7)/(8) use across every
# remaining dim — the directive's "EXPECT >=1 invariant to fail at integration
# scale that passed in isolation" applied to the FULL fingerprint, not just the
# three newest dims. A silent replay here is the corrupt-but-green §4 verdict
# (GOAL §7) the gate exists to prevent.


def _cfg_full(**deep_overrides):
    """``_cpu_cfg`` extended to override ANY lora/training/model field, so the
    cfg-sourced fingerprint dims (alpha / dropout / target_modules) can be
    varied in the same two-run replay check as the kwarg-sourced dims. Mirrors
    ``_cpu_cfg``'s shape exactly (the baseline path here is byte-identical to a
    ``_cpu_cfg(max_steps=6, base_rank=16)`` run)."""
    base = {
        "model": {"name_or_path": "tiny-llama-stub", "load_in_4bit": False},
        "training": {
            "trainable_lora_scope": "last_25_percent",
            "learning_rate": 1e-4, "max_steps": 6, "gradient_checkpointing": False,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"},
    }
    for section, overrides in deep_overrides.items():
        base[section] = {**base.get(section, {}), **overrides}
    return OmegaConf.create(base)


def _replay_check(monkeypatch, tmp_path, *, run2_kw=None, run2_cfg_overrides=None):
    """Run the assembled heterogeneous path twice against ONE ledger; return
    ``(run2_resumed_arm_count, run2_arm_calls)``.

    Run 1 banks 4 arms (2 candidate + 2 surrogate) under the baseline config.
    Run 2 re-runs against the SAME ledger with exactly one fingerprint dim
    varied (a kwarg dim via ``run2_kw``, or a cfg dim via ``run2_cfg_overrides``).
    If that dim is wired into ``_config_fingerprint`` the two fingerprints
    differ, run 2 ignores the stale ledger, and every arm re-executes
    (``resumed_arm_count == 0`` and the arm runner fires 4×). A silent replay
    (``resumed_arm_count > 0`` or fewer than 4 calls) is the corrupt-but-green
    §4 failure (GOAL §7) the gate exists to prevent — and the signal the per-dim
    unit tests structurally cannot produce.
    """
    import scripts.run_freeze_validloss_ci_9b as mod

    monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
    monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
    monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())

    calls = {"n": 0}
    base_toy = _make_toy_arm(base_seed=7)

    def _counting_arm(*a, **kw):
        calls["n"] += 1
        return base_toy(*a, **kw)

    monkeypatch.setattr(mod, "arm_valid_loss_9b", _counting_arm)
    ledger_path = str(tmp_path / "ledger.jsonl")
    common = dict(
        device=torch.device("cpu"), seq_len=256, train_examples=4,
        valid_examples=4, total_steps=6, warmup_steps=1, depth=1, spacing=2,
        n_candidate=2, n_surrogate=2, base_seed=7, dataset="dolly",
        max_dataset_rows=12, architecture=HETEROGENEOUS, ledger_path=ledger_path,
    )

    calls["n"] = 0
    run1 = run_ci_9b(cfg=_cfg_full(), **common)
    assert run1["resumed_arm_count"] == 0, "baseline run banked from a stale ledger"
    assert calls["n"] == 4, (
        f"baseline run invoked the arm runner {calls['n']}x, expected 4 "
        "(2 candidate + 2 surrogate) — harness setup drift."
    )

    run2_kw_final = {**common, **(run2_kw or {})}
    calls["n"] = 0
    run2 = run_ci_9b(
        cfg=_cfg_full(**(run2_cfg_overrides or {})), **run2_kw_final,
    )
    return run2["resumed_arm_count"], calls["n"]


# One fingerprint dim varied per case; the rest held at the (6)/(7)/(8)
# baseline. Each variation is a genuine §4-experiment change (different
# training length / freeze depth / context length / data slice / loss regime /
# seed / dataset / LoRA adapter), so a replay under it would seed the verdict
# with arms trained under the WRONG config — the corrupt-but-green case.
_REPLAY_DIMS = [
    pytest.param(dict(run2_kw={"total_steps": 8}), id="total_steps"),
    pytest.param(dict(run2_kw={"warmup_steps": 2}), id="warmup_steps"),
    pytest.param(dict(run2_kw={"depth": 2}), id="depth"),
    pytest.param(dict(run2_kw={"spacing": 3}), id="spacing"),
    pytest.param(dict(run2_kw={"seq_len": 192}), id="seq_len"),
    pytest.param(dict(run2_kw={"train_examples": 3}), id="train_examples"),
    pytest.param(dict(run2_kw={"valid_examples": 3}), id="valid_examples"),
    pytest.param(dict(run2_kw={"use_local_loss": False}), id="use_local_loss"),
    pytest.param(dict(run2_kw={"base_seed": 11}), id="base_seed"),
    pytest.param(dict(run2_kw={"dataset": "dolly-15k"}), id="dataset"),
    pytest.param(
        dict(run2_cfg_overrides={"lora": {"alpha": 64}}), id="lora_alpha",
    ),
    pytest.param(
        dict(run2_cfg_overrides={"lora": {"dropout": 0.1}}), id="lora_dropout",
    ),
    pytest.param(
        dict(run2_cfg_overrides={"lora": {"target_modules": ["q_proj", "v_proj"]}}),
        id="lora_target_modules",
    ),
]


class TestAssembledFingerprintGateAllDims:
    """The three sibling gates (6)/(7)/(8) cover lora_r / lr / max_dataset_rows.
    This class closes the remaining coverage: every OTHER ``_config_fingerprint``
    dim, proven at the assembled scale the unit tests cannot reach."""

    @pytest.mark.parametrize("variation", _REPLAY_DIMS)
    def test_dim_change_re_executes_not_replays(
        self, monkeypatch, tmp_path, variation, request,
    ):
        dim = request.node.callspec.id
        resumed, calls = _replay_check(monkeypatch, tmp_path, **variation)
        assert resumed == 0, (
            f"the {dim} re-run replayed stale ledger arms — that fingerprint "
            f"dim is wired into the arm but NOT into the ``_config_fingerprint`` "
            f"call in run_ci_9b (GOAL §7 corrupt-but-green §4 verdict): resumed "
            f"{resumed} arm(s) instead of re-executing."
        )
        assert calls == 4, (
            f"the {dim} re-run invoked the arm runner {calls}x, expected 4 — "
            f"stale ledger arms were silently replayed under a different "
            f"run-config (corrupt-but-green, GOAL §7)."
        )


# ── (10) a MATCHING fingerprint replays FAITHFULLY: resumed verdict == fresh ──
#
# Siblings (6)–(9) each prove the MISMATCH half of the resume-ledger gate — a
# config change re-executes rather than silently replaying wrong arms. They
# leave the gate's DEFINING contract unproven at assembled scale: when the
# fingerprint MATCHES (a genuine resume — an interrupted full-budget run
# re-fired on the next free-GPU window), the ledger-replayed arms must
# reproduce the FRESH one-shot run's verdict, every loss sample, both CIs, all
# provenance, and the ``evidence_hash`` byte-for-byte. The unit tests
# (``test_freeze_validloss_ci_9b_resume.py``) prove the replay MECHANICS on
# ``_collect_arms`` with STUB runners and round-numbered stub values — they
# never run ``run_ci_9b`` assembled, so a corruption that lives in the
# assembled join (a replayed arm's ``(valid_loss, provenance)`` mis-flowing
# through ``surrogate_valid_loss_ci`` → ``result_to_json`` → the cost
# accountant / regime classifier) would pass the unit tests and rot the
# verdict here. The multi-window launcher architecture (TASK-0161/0162) banks
# each arm to this ledger and resumes across GPU windows, so this faithfulness
# is the property that architecture's headline deliverable rests on — and it
# is the one direction of the gate no prior invariant pins.
#
# This exercises control arms too (``n_control``), the exact arm mix the
# harvested heterogeneous full-budget verdict (``1224057``) used, so the
# direction-isolation CI — candidate vs the input-contiguous control — is
# proven reproducible under resume, not just the headline surrogate A/B.


def _resume_pair(monkeypatch, tmp_path, *, n_candidate=2, n_surrogate=2, n_control=2):
    """Run the REAL assembled heterogeneous path fresh, then resume it from a
    ledger whose fingerprint MATCHES, returning ``(deposit_fresh,
    deposit_resumed, resumed_arm_count)``.

    Run 1 is a fresh one-shot (no ledger). Run 2 banks every arm to a ledger,
    then run 3 re-fires against that SAME ledger — the fingerprint is
    identical, so every arm replays and ``run_ci_9b`` assembles the verdict
    entirely from ledger-cached ``(valid_loss, provenance)`` pairs. Both
    deposits come from the REAL ``run_ci_9b`` → ``result_to_json`` assembly
    (only the network/GPU boundaries are stubbed), so a faithfulness gap that
    hides in the assembled replay join — not the unit-tested
    ``_collect_arms`` mechanics — surfaces as a diff between the two deposits.
    """
    import scripts.run_freeze_validloss_ci_9b as mod

    monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
    monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
    monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())
    monkeypatch.setattr(mod, "arm_valid_loss_9b", _make_toy_arm(base_seed=7))

    ledger_path = str(tmp_path / "ledger.jsonl")
    common = dict(
        cfg=_cpu_cfg(max_steps=6, base_rank=16), device=torch.device("cpu"),
        seq_len=256, train_examples=4, valid_examples=4, total_steps=6,
        warmup_steps=1, depth=1, spacing=2, n_candidate=n_candidate,
        n_surrogate=n_surrogate, n_control=n_control, base_seed=7,
        dataset="dolly", max_dataset_rows=12, architecture=HETEROGENEOUS,
    )
    deposit_fresh = result_to_json(run_ci_9b(**common))
    # Bank every arm, then resume against the same matching-fingerprint ledger.
    run_ci_9b(ledger_path=ledger_path, **common)
    resumed = run_ci_9b(ledger_path=ledger_path, **common)
    return deposit_fresh, result_to_json(resumed), resumed["resumed_arm_count"]


class TestAssembledResumeFidelity:
    """The MATCHING-fingerprint half of the resume-ledger gate: a resumed run
    reproduces the fresh one-shot verdict. Siblings (6)–(9) cover MISMATCH;
    this covers the gate's defining contract at assembled scale."""

    def test_resumed_run_reproduces_fresh_verdict_and_evidence_hash(
        self, monkeypatch, tmp_path,
    ):
        dep_f, dep_r, n_resumed = _resume_pair(monkeypatch, tmp_path)
        total_arms = 2 + 2 + 2  # candidate + surrogate + control
        assert n_resumed == total_arms, (
            f"a matching-fingerprint resume replayed {n_resumed} arm(s), "
            f"expected all {total_arms} — the ledger did not serve every arm "
            "from cache, so this is not a genuine resume of the same run."
        )
        # The headline verdict + both means are identical (the property the
        # multi-window launcher's assembled verdict rests on).
        assert dep_f["verdict"] == dep_r["verdict"], (
            f"verdict drifted on resume: fresh={dep_f['verdict']} "
            f"resumed={dep_r['verdict']} — a ledger-replayed arm changed the "
            "§4 verdict vs the fresh one-shot run."
        )
        assert dep_f["candidate_mean"] == dep_r["candidate_mean"]
        assert dep_f["surrogate_mean"] == dep_r["surrogate_mean"]
        # The load-bearing pin: EVERY evidence field (losses, orders, all four
        # provenance dicts, the run-determining config) is byte-identical, so
        # the evidence_hash is too. A ledger round-trip that corrupted any
        # replayed arm's value (a float, a frozen-layer index, a provenance
        # field) breaks this — the assembled scale where the unit-tested
        # _collect_arms mechanics cannot catch a join-level corruption.
        assert dep_f["evidence_hash"] == dep_r["evidence_hash"], (
            "evidence_hash drifted on resume — at least one evidence field "
            "(losses / orders / provenance / run-config) changed when an arm "
            "was served from the ledger vs run fresh (GOAL §7: a resumed "
            "multi-window verdict must reproduce the fresh one-shot hash)."
        )
        for key in EVIDENCE_HASH_KEYS:
            assert dep_f[key] == dep_r[key], (
                f"evidence field {key!r} differs fresh vs resumed — a "
                "ledger-replayed arm did not reproduce its fresh value."
            )

    def test_only_honest_resume_provenance_differs(self, monkeypatch, tmp_path):
        # The two deposits must differ ONLY in the two fields that honestly
        # record HOW the run executed (``ledger_path`` / ``resumed_arm_count``)
        # — never in a verdict, loss, or provenance field. A silent field-level
        # corruption on replay would show up as a third diff here.
        dep_f, dep_r, _ = _resume_pair(monkeypatch, tmp_path)
        diffs = {
            k for k in (set(dep_f) | set(dep_r))
            if dep_f.get(k) != dep_r.get(k)
        }
        assert diffs == {"ledger_path", "resumed_arm_count"}, (
            f"fresh vs resumed deposits differ on {sorted(diffs)} — expected "
            "ONLY the honest ledger_path / resumed_arm_count provenance fields "
            "to differ; every other field must be identical (GOAL §7)."
        )
        # And those two honest fields DO differ (the resume actually happened).
        assert dep_r["resumed_arm_count"] == 6 and dep_f["resumed_arm_count"] == 0
        assert dep_r["ledger_path"] is not None and dep_f["ledger_path"] is None

    def test_direction_ci_faithfully_replayed_from_control_arms(
        self, monkeypatch, tmp_path,
    ):
        # The control-arm mix the harvested heterogeneous full-budget verdict
        # (1224057) used (n_candidate=3 / n_surrogate=3 / n_control=3). The
        # direction-isolation CI (candidate vs the input-contiguous control)
        # is the constitution-P0 attribution the control arm exists for; it
        # must reproduce exactly when the control arms replay from the ledger,
        # not just the headline surrogate A/B.
        dep_f, dep_r, n_resumed = _resume_pair(
            monkeypatch, tmp_path, n_candidate=3, n_surrogate=3, n_control=3,
        )
        assert n_resumed == 9, (
            f"the 9-arm heterogeneous resume replayed {n_resumed} arm(s) — "
            "expected all 9 (3 candidate + 3 surrogate + 3 control)."
        )
        assert dep_f["control_losses"] == dep_r["control_losses"], (
            "control-arm losses drifted on resume — the direction-isolation "
            "control did not reproduce its fresh values from the ledger."
        )
        assert dep_f["control_provenance"] == dep_r["control_provenance"]
        assert dep_f["direction"] == dep_r["direction"], (
            f"direction CI drifted on resume: fresh={dep_f['direction']} "
            f"resumed={dep_r['direction']} — the constitution-P0 direction "
            "attribution is not reproducible when control arms replay."
        )
        # The direction CI is real (not None): control arms ran, so the
        # candidate-vs-control attribution is present in both deposits.
        assert dep_f["direction"] is not None


# ── (11) a fully-banked ledger seals CPU-only — NO 9B model load ───────────────
#
# The multi-window launcher banks each arm to the ``--ledger`` across free-GPU
# windows; that resumable ledger is the architecture that makes a multi-hour 9B
# run survivable. But the FINAL seal (assemble the verdict from the banked arms
# + write the deposit) needs NO GPU — every arm is cached. Before this fix
# ``run_ci_9b`` loaded the 9B model UNCONDITIONALLY (to derive the active scope +
# ``scope_trainable_params``) BEFORE the replay decision in ``_collect_arms``,
# and ``main()``'s CUDA-available + free-memory pre-flight gates ran BEFORE
# ``run_ci_9b``. So a run that banked every arm but crashed before writing its
# deposit (an OOM in the post-arm cleanup, a host kill, a CWD death at the wire)
# could NEVER seal: every re-fire hit the GPU gates, and on a busy / no-GPU
# window it OOM-deferred (exit 75) or CUDA-down'd (exit 2) forever — wasting the
# whole multi-hour run behind the GPU queue for a result that needed zero GPU.
# This proves the seal is now GPU-free: a fully-banked re-fire never calls
# ``load_base_model``, and the sealed deposit reproduces the fresh run's evidence
# byte-for-byte.


class TestAssembledGpuFreeSeal:
    def _common(self, tmp_path, *, ledger_path, **overrides):
        kw = dict(
            cfg=_cpu_cfg(max_steps=6, base_rank=16), device=torch.device("cpu"),
            seq_len=256, train_examples=4, valid_examples=4, total_steps=6,
            warmup_steps=1, depth=1, spacing=2, n_candidate=2, n_surrogate=2,
            base_seed=7, dataset="dolly", max_dataset_rows=12,
            architecture=HETEROGENEOUS, ledger_path=str(ledger_path),
        )
        kw.update(overrides)
        return kw

    def test_fully_banked_seal_never_loads_the_model(self, monkeypatch, tmp_path):
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(mod, "arm_valid_loss_9b", _make_toy_arm(base_seed=7))

        ledger = tmp_path / "ledger.jsonl"
        common = self._common(tmp_path, ledger_path=ledger)
        # Run 1 banks every arm under the real (stubbed) model load.
        fresh = result_to_json(run_ci_9b(**common))
        ledger_lines = [
            ln for ln in Path(ledger).read_text().splitlines() if ln.strip()
        ]
        assert len(ledger_lines) >= 5, "run 1 did not bank a header + 4 arms"

        # Run 2 re-fires against the fully-banked ledger. ``load_base_model`` is
        # now FATAL — the seal must assemble from cached arms and never reach it.
        load_calls = {"n": 0}

        def _boom(cfg):
            load_calls["n"] += 1
            raise AssertionError(
                "the fully-banked seal loaded the 9B model — the seal is still "
                "GPU-gated, so a completed multi-window run cannot finish on a "
                "busy / no-GPU window."
            )

        monkeypatch.setattr(mod, "load_base_model", _boom)
        sealed = result_to_json(run_ci_9b(**common))

        assert load_calls["n"] == 0, (
            "the fully-banked re-fire called load_base_model — the seal is still "
            "GPU-gated, so a completed run that crashed before writing its deposit "
            "would OOM-defer forever behind the GPU queue for a zero-GPU result."
        )
        # The GPU-free seal reproduces the fresh run's evidence byte-for-byte
        # (the §7 fidelity the resume-fidelity invariant pins, in the no-GPU path).
        assert sealed["evidence_hash"] == fresh["evidence_hash"], (
            "the GPU-free seal changed the evidence — a corrupt-but-green deposit."
        )
        for key in EVIDENCE_HASH_KEYS:
            assert sealed[key] == fresh[key], (
                f"evidence field {key!r} differs fresh vs GPU-free-sealed."
            )
        # scope_trainable_params is the ONE model-derived field a replay cannot
        # recompute; it is read from the ledger header, not lost — so the deposit
        # is field-identical, not just evidence-identical.
        assert sealed["scope_trainable_params"] == fresh["scope_trainable_params"]

    def test_incomplete_ledger_still_loads_the_model(self, monkeypatch, tmp_path):
        # A ledger that banks only SOME arms of the requested run must NOT seal
        # GPU-free — the missing arms still need the GPU. ``load_base_model`` is
        # reached (and the toy model served) so the missing arms train for real.
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(mod, "arm_valid_loss_9b", _make_toy_arm(base_seed=7))

        ledger = tmp_path / "ledger.jsonl"
        # Bank only 2 arms (1 candidate + 1 surrogate) of the SAME config — same
        # fingerprint, just fewer arms, so the seal check sees a partial bank.
        run_ci_9b(**self._common(
            tmp_path, ledger_path=ledger, n_candidate=1, n_surrogate=1,
        ))

        load_calls = {"n": 0}

        def _count(cfg):
            load_calls["n"] += 1
            return _tiny_llama(8)

        monkeypatch.setattr(mod, "load_base_model", _count)
        # Re-fire the full 4-arm sweep: 2 cached, 2 missing → must train them.
        run_ci_9b(**self._common(tmp_path, ledger_path=ledger))
        assert load_calls["n"] >= 1, (
            "a partial ledger (arms still missing) sealed GPU-free — the missing "
            "arms would never train, producing a corrupt deposit (GOAL §7)."
        )

    def test_main_seals_fully_banked_ledger_with_cuda_down(self, monkeypatch, tmp_path):
        """The launch path's killer proof: a CUDA-DOWN host (exit 2 forever under
        the pre-seal behavior) seals a fully-banked verdict to rc=0 via main().

        Banks every arm of a real (toy-stubbed) run, then makes CUDA completely
        unavailable and re-fires through ``main`` itself (not ``run_ci_9b``
        directly). With the GPU-free seal, main() recognizes the fully-banked
        ledger, skips the CUDA gate, and assembles the deposit CPU-only — so a
        completed multi-window run that crashed before writing its deposit
        finishes on a host with NO GPU at all. The arm runner is fatal (it must
        not be reached: the seal replays from the ledger) and the deposit's
        evidence is byte-identical to the fresh run (§7 fidelity at the main()
        seam)."""
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(mod, "arm_valid_loss_9b", _make_toy_arm(base_seed=7))

        cfg = _cpu_cfg(max_steps=6, base_rank=16)
        ledger = tmp_path / "ledger.jsonl"
        cfg_path = tmp_path / "cfg.yaml"
        out = tmp_path / "deposit.json"
        OmegaConf.save(cfg, cfg_path)

        # Run 1: bank every arm under the real (stubbed) model load.
        fresh = result_to_json(run_ci_9b(**self._common(tmp_path, ledger_path=ledger)))

        # Run 2: CUDA is now UNAVAILABLE. Pre-seal, main() exits 2 here and the
        # banked run could NEVER finish — the exact multi-window trap the seal
        # closes. The arm runner is made FATAL: the seal must replay, not train.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        def _boom_arm(*a, **kw):  # noqa: ARG001
            raise AssertionError(
                "the arm runner trained on a fully-banked seal — the CUDA gate "
                "was not skipped, so a sealed run still needs the GPU."
            )

        monkeypatch.setattr(mod, "arm_valid_loss_9b", _boom_arm)

        rc = mod.main([
            "--config", str(cfg_path),
            "--seq-len", "256", "--train-examples", "4", "--valid-examples", "4",
            "--total-steps", "6", "--warmup-steps", "1", "--depth", "1",
            "--spacing", "2", "--n-candidate", "2", "--n-surrogate", "2",
            "--base-seed", "7", "--dataset", "dolly", "--max-dataset-rows", "12",
            "--architecture", str(HETEROGENEOUS),
            "--ledger", str(ledger), "--output", str(out),
        ])

        assert rc == 0, (
            f"a CUDA-DOWN host must seal a fully-banked verdict to rc=0, got {rc} "
            "— the CUDA gate blocked a zero-GPU result (the multi-window trap)."
        )
        assert out.exists(), "the sealed deposit was not written to --output."
        sealed = json.loads(out.read_text())
        assert sealed["evidence_hash"] == fresh["evidence_hash"], (
            "main()'s GPU-free seal changed the evidence — a corrupt-but-green deposit."
        )


class TestLedgerSealReadyGate:
    """Unit-pins the seal-decision predicate the GPU-free path + main() gate-skip
    both rely on — the edge cases the integration test above cannot reach
    (config drift, an old pre-seal ledger, no ledger at all)."""

    def _kwargs(self, tmp_path, **overrides):
        from scripts.run_freeze_validloss_ci_9b import _ledger_seal_ready
        kw = dict(
            total_steps=6, warmup_steps=1, depth=1, spacing=2, seq_len=256,
            train_examples=4, valid_examples=4, model="tiny-llama-stub",
            scope_label="last_25_percent", dataset="dolly", max_dataset_rows=12,
            use_local_loss=True, learning_rate=1e-4, base_seed=7,
            architecture=HETEROGENEOUS, lora_r=16, lora_alpha=32, lora_dropout=0.0,
            lora_target_modules="all-linear", n_candidate=2, n_surrogate=2,
            n_control=0, n_baseline=0,
        )
        kw.update(overrides)
        return _ledger_seal_ready, kw

    def test_no_ledger_is_not_seal_ready(self, tmp_path):
        _gate, kw = self._kwargs(tmp_path)
        assert _gate(str(tmp_path / "absent.jsonl"), **kw) is None

    def test_config_drift_is_not_seal_ready(self, monkeypatch, tmp_path):
        # A ledger banked under a DIFFERENT config (here: a different base_rank →
        # different lora_r fingerprint) is not a replay of THIS run.
        _gate, kw = self._kwargs(tmp_path)
        _bank_a_full_ledger(monkeypatch, tmp_path, base_rank=32)  # lora_r=32
        # Requested config has lora_r=16 → drift → None.
        assert _gate(str(tmp_path / "ledger.jsonl"), **kw) is None

    def test_old_ledger_without_scope_trainable_is_not_seal_ready(
        self, monkeypatch, tmp_path,
    ):
        # A ledger written BEFORE the GPU-free-seal field existed (no
        # scope_trainable_params in the header) must NOT seal — fall back to the
        # GPU path rather than seal with a missing model-derived field.
        _gate, kw = self._kwargs(tmp_path)
        _bank_a_full_ledger(monkeypatch, tmp_path, base_rank=16)
        # Strip scope_trainable_params from the header to emulate an old ledger.
        ledger = tmp_path / "ledger.jsonl"
        lines = ledger.read_text().splitlines()
        header = json.loads(lines[0])
        header.pop("scope_trainable_params", None)
        lines[0] = json.dumps(header)
        ledger.write_text("\n".join(lines) + "\n")
        assert _gate(str(ledger), **kw) is None


def _bank_a_full_ledger(monkeypatch, tmp_path, *, base_rank):
    """Bank a complete heterogeneous ledger under ``base_rank`` so a seal-ready
    check has a real (config-matched, every-arm-cached) ledger to reason about."""
    import scripts.run_freeze_validloss_ci_9b as mod

    monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
    monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
    monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())
    monkeypatch.setattr(mod, "arm_valid_loss_9b", _make_toy_arm(base_seed=7))
    run_ci_9b(
        cfg=_cpu_cfg(max_steps=6, base_rank=base_rank), device=torch.device("cpu"),
        seq_len=256, train_examples=4, valid_examples=4, total_steps=6,
        warmup_steps=1, depth=1, spacing=2, n_candidate=2, n_surrogate=2,
        base_seed=7, dataset="dolly", max_dataset_rows=12, architecture=HETEROGENEOUS,
        ledger_path=str(tmp_path / "ledger.jsonl"),
    )


# ── (12) DONE requires a written citable deposit — no silent deposit-less run ──
#
# The exit-code contract the launcher reads says exit 0 = "deposit written →
# DONE" (``launch_freeze_ci_9b_full.classify_exit_code``). The deposit
# (``--output``) is the ENTIRE point of executing ``run_ci_9b`` — the
# multi-hour, multi-arm GPU sweep harvested into ``tests/fixtures/`` and
# re-loaded as JSON by the paper gate. Pre-fix, ``main()`` returned EXIT_DONE
# even when ``--output`` was unset: the verdict was computed (and maybe echoed
# to stdout via ``--json``) but NO citable artifact was persisted, so the
# launcher declared DONE for a run that wrote nothing. A fully-banked ledger
# made it worse: the GPU-free seal (dd0cc03) assembles the verdict CPU-only and
# would have returned DONE on EVERY re-fire without ever writing the deposit —
# the one artifact the whole multi-window architecture exists to produce.
# That is the same "corrupt-but-green" class as the atomic-write (54a4cd8) and
# ``--output``-always-JSON (b8e7ce4) fixes: a GOAL §7 contract the code
# enforces, not one the caller must remember. The guard refuses to reach
# ``run_ci_9b`` without ``--output`` (placed AFTER the CUDA / free-memory gates
# so the gpu_preflight tests, which omit ``--output`` and assert those gates
# fire first, stay green). These prove it at the assembled scale.


class TestAssembledDoneRequiresDeposit:
    def test_seal_ready_run_without_output_is_not_done(self, monkeypatch, tmp_path):
        # The headline corruption: a fully-banked ledger (the GPU-free-seal case)
        # re-fired WITHOUT --output. Pre-fix this assembled CPU-only and returned
        # EXIT_DONE, writing no deposit — and would do so on every re-fire, the
        # multi-window architecture never producing its one artifact. The guard
        # refuses: EXIT_UNEXPECTED (FATAL), not DONE, so the launcher stops and
        # the operator adds --output rather than harvesting nothing.
        import scripts.run_freeze_validloss_ci_9b as mod

        monkeypatch.setattr(mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records())
        monkeypatch.setattr(mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(mod, "arm_valid_loss_9b", _make_toy_arm(base_seed=7))

        cfg = _cpu_cfg(max_steps=6, base_rank=16)
        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(cfg, cfg_path)
        ledger = tmp_path / "ledger.jsonl"
        common = dict(
            cfg=cfg, device=torch.device("cpu"), seq_len=256, train_examples=4,
            valid_examples=4, total_steps=6, warmup_steps=1, depth=1, spacing=2,
            n_candidate=2, n_surrogate=2, base_seed=7, dataset="dolly",
            max_dataset_rows=12, architecture=HETEROGENEOUS, ledger_path=str(ledger),
        )
        # Bank every arm under the real (stubbed) model load.
        run_ci_9b(**common)
        ledger_lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
        assert len(ledger_lines) >= 5, "run 1 did not bank a header + 4 arms"

        # CUDA now down — forces the GPU-free seal path (no model load). NO
        # --output: the deposit the run exists to produce has nowhere to land.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        rc = mod.main([
            "--config", str(cfg_path),
            "--seq-len", "256", "--train-examples", "4", "--valid-examples", "4",
            "--total-steps", "6", "--warmup-steps", "1", "--depth", "1",
            "--spacing", "2", "--n-candidate", "2", "--n-surrogate", "2",
            "--base-seed", "7", "--dataset", "dolly", "--max-dataset-rows", "12",
            "--architecture", str(HETEROGENEOUS),
            "--ledger", str(ledger),
            # NOTE: deliberately NO --output — the silent-corruption case.
        ])
        assert rc == EXIT_UNEXPECTED, (
            f"a deposit-less seal-ready run returned {rc} (DONE={EXIT_DONE}) — "
            "EXIT_DONE's contract is 'deposit written → launcher DONE', but no "
            "--output was given so no citable deposit could be written: the "
            "launcher would declare a completed multi-hour run with no "
            "harvestable artifact (GOAL §7 silent corruption)."
        )

    def test_live_gpu_run_without_output_refuses_before_burning_gpu(
        self, monkeypatch, tmp_path,
    ):
        # The normal (non-seal) path: a live GPU, no ledger, NO --output. The
        # guard must fire BEFORE run_ci_9b so no GPU is burned on a verdict that
        # can't be persisted. The guard sits after the CUDA / free-memory gates
        # so those distinct codes still surface first (pinned by the gpu_preflight
        # suite); this proves the guard then stops the run before the sweep.
        import scripts.run_freeze_validloss_ci_9b as mod

        cfg = _cpu_cfg(max_steps=6, base_rank=16)
        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(cfg, cfg_path)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        reached = {"run_ci_9b": False}

        def _boom_if_reached(**kw):  # noqa: ARG001
            reached["run_ci_9b"] = True
            raise AssertionError(
                "run_ci_9b was reached without --output — the guard did not stop "
                "the run before burning GPU on a verdict that cannot be persisted."
            )

        monkeypatch.setattr(mod, "run_ci_9b", _boom_if_reached)

        rc = mod.main([
            "--config", str(cfg_path),
            "--min-free-gib", "0",  # bypass the free-memory floor (no GPU touch)
            # NO --ledger (so no seal path), NO --output — the guard must fire.
        ])
        assert rc == EXIT_UNEXPECTED, (
            f"a deposit-less live-GPU run returned {rc} — expected the "
            "EXIT_UNEXPECTED refusal, not a silent DONE."
        )
        assert reached["run_ci_9b"] is False, (
            "the guard let run_ci_9b execute without --output — GPU would be "
            "burned on a verdict with no citable sink."
        )

    def test_output_present_still_runs(self, monkeypatch, tmp_path):
        # No false positive: with --output the guard does not fire and the run
        # proceeds (here: reaches run_ci_9b, stubbed to a sentinel). This is the
        # every-real-caller path (every Makefile target / the launcher pass
        # --output), so the guard must not regress it.
        import scripts.run_freeze_validloss_ci_9b as mod

        cfg = _cpu_cfg(max_steps=6, base_rank=16)
        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(cfg, cfg_path)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        class _Reached(RuntimeError):
            pass

        def _reach(**kw):  # noqa: ARG001
            raise _Reached("reached run_ci_9b — guard did not block a --output run")

        monkeypatch.setattr(mod, "run_ci_9b", _reach)

        with pytest.raises(_Reached):
            mod.main([
                "--config", str(cfg_path),
                "--min-free-gib", "0",
                "--output", str(tmp_path / "deposit.json"),
            ])


# ── (13) the LAUNCHER survives a worker run-time OOM (invariant 4's unjoined half) ──
#
# Invariant (4) (TestAssembledOomTempfail above) proves the WORKER half: a run-time
# OOM raised from inside the REAL ``_collect_arms`` arm loop propagates as a
# ``RuntimeError`` ``is_cuda_oom`` recognizes, and the ledger banks the completed
# arm before it raises. ``tests/test_launch_freeze_ci_9b_full.py`` proves the
# LAUNCHER half in ISOLATION: ``classify_exit_code(75) -> RETRY`` and
# ``run_loop`` reaches DONE off a ``_FakeRunner`` that pops a HAND-BUILT
# ``SimpleNamespace(returncode=75)`` — a 75 that never came from a worker. And
# ``tests/test_run_freeze_validloss_ci_9b.py::test_main_classifies_run_time_oom_as_
# tempfail`` proves ``main()`` returns 75 — but by STUBBING ``run_ci_9b`` with a
# ``_boom`` that raises, so no real arm loop, no ledger banking, no launcher.
#
# NEITHER test JOINS the worker to the launcher: a real worker run-time OOM ->
# ``main()`` exit 75 -> the launcher reads that code, classifies RETRY, re-fires,
# and the re-fire completes the run with the banked arm carried across the
# ``--ledger``. That join is the LITERAL directive invariant (4) ("the launcher
# survives") and the one place the worker->launcher exit-code contract
# (``ad8c84a``, pinned only STATICALLY — constant equality + a source grep for
# the string ``return EXIT_GPU_TEMPFAIL``) is never exercised DYNAMICALLY. A
# regression at the join — ``main()``'s ``except RuntimeError`` no longer
# catching the OOM (returns 1 FATAL instead of 75), or a torch version where the
# device-dispatch OOM surfaces as an exception that is NOT a ``RuntimeError``
# sibling — passes EVERY slice above (the ``is_cuda_oom`` unit, the
# ``classify_exit_code(75)`` unit, the ``_FakeRunner`` loop, the static contract
# pin) yet strands the multi-hour run on one contention spike: the exact loss the
# 4afc5e9 tempfail classification exists to prevent.
#
# This drives the REAL ``launch_freeze_ci_9b_full.run_loop`` against the REAL
# worker ``main()`` via an in-process runner (it calls ``main()`` and wraps the
# return in a ``CompletedProcess``-shaped object, so the worker->launcher
# integer-exit-code boundary — the contract — is real, without spawning a
# subprocess or touching a GPU). The worker's real arm loop OOMs on attempt 1
# (after banking 1 arm), the launcher must classify the 75 as RETRY, re-fire, and
# reach DONE; and a non-OOM RuntimeError must reach ``main()``'s reraise -> exit
# 1 -> launcher FATAL (so a real bug is not infinite-looped as retryable).


class TestAssembledLauncherSurvivesWorkerOom:
    def _runner(self, worker_mod, *, attempt_state):
        """A ``run_loop`` runner that invokes the REAL worker ``main()``
        in-process and wraps its return code in a ``CompletedProcess`` stand-in —
        so the integer exit-code boundary the worker->launcher contract is built
        on is exercised for real, without a subprocess or a GPU.

        ``attempt_state`` (a shared dict) lets the toy arm below behave
        differently per launcher attempt (OOM on attempt 1 only)."""
        from types import SimpleNamespace

        def _run(argv, **kwargs):  # noqa: ARG001
            attempt_state["attempt"] += 1
            attempt_state["arm_calls"] = 0
            module_argv = list(argv[3:])  # strip [interpreter, "-m", MODULE]
            try:
                rc = worker_mod.main(module_argv)
            except SystemExit as exc:
                # argparse errors etc. surface as SystemExit in-process; a real
                # subprocess would exit with exc.code.
                rc = int(exc.code) if exc.code is not None else 1
            except BaseException:
                # main() RERAISES a non-OOM RuntimeError (stays fatal); a real
                # subprocess would print a traceback and exit 1. Model that here
                # so the launcher sees the same integer the contract describes.
                rc = 1
            return SimpleNamespace(returncode=rc)

        return _run

    def _common_worker_flags(self, cfg_path, ledger, out):
        return [
            "--config", str(cfg_path), "--device", "cpu",
            "--seq-len", "256", "--train-examples", "4", "--valid-examples", "4",
            "--total-steps", "6", "--warmup-steps", "1", "--depth", "1",
            "--spacing", "2", "--n-candidate", "2", "--n-surrogate", "2",
            "--base-seed", "7", "--dataset", "dolly", "--max-dataset-rows", "12",
            "--architecture", str(HETEROGENEOUS),
            "--min-free-gib", "0",  # bypass the free-memory snapshot gate
            "--ledger", str(ledger), "--output", str(out),
        ]

    def test_worker_oom_then_launcher_retries_to_done(self, monkeypatch, tmp_path):
        # THE join: the real worker arm loop OOMs on attempt 1 -> main() exit 75
        # -> the real launcher reads 75, classifies RETRY, re-fires, and the
        # re-fire completes (DONE) with the OOM-attempt's banked arm carried
        # across the --ledger. A FATAL here (main() returning 1, or the launcher
        # not retrying 75) strands the multi-hour run — invariant (4)'s
        # "launcher survives" half, proven at the assembled scale for the first
        # time.
        import scripts.run_freeze_validloss_ci_9b as worker_mod
        import scripts.launch_freeze_ci_9b_full as launcher_mod

        monkeypatch.setattr(worker_mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(worker_mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(
            worker_mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records(),
        )
        # CUDA "available" so main()'s pre-flight reaches run_ci_9b (the run runs
        # on --device cpu; the toy arm ignores device). The OOM is injected from
        # inside the real arm loop, exactly the TOCTOU run-time contention 4afc5e9
        # re-classifies.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(_cpu_cfg(max_steps=6, base_rank=16), cfg_path)
        ledger = tmp_path / "ledger.jsonl"
        out = tmp_path / "deposit.json"

        good_arm = _make_toy_arm(base_seed=7)
        attempt_state = {"attempt": 0, "arm_calls": 0}

        def _oom_on_attempt1_arm(*a, **kw):
            attempt_state["arm_calls"] += 1
            # Attempt 1: bank arm 1 (call 1 ok), OOM on arm 2 (call 2) — exactly
            # the TestAssembledOomTempfail (4) injection point, reached here
            # through main() instead of run_ci_9b directly.
            if attempt_state["attempt"] == 1 and attempt_state["arm_calls"] > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
            return good_arm(*a, **kw)

        monkeypatch.setattr(worker_mod, "arm_valid_loss_9b", _oom_on_attempt1_arm)

        sleeps: list[float] = []
        result = launcher_mod.run_loop(
            interpreter="/in-process/python",  # never spawned — runner calls main()
            module_argv=self._common_worker_flags(cfg_path, ledger, out),
            max_attempts=3, deadline_seconds=None,
            resume_sleep=0.0, tempfail_sleep=0.0,
            sleep_fn=lambda s: sleeps.append(s),
            now_fn=(lambda: 0.0),
            runner=self._runner(worker_mod, attempt_state=attempt_state),
        )

        assert result.outcome is launcher_mod.Outcome.DONE, (
            f"the launcher did NOT survive a worker run-time OOM — outcome "
            f"{result.outcome.value} (last worker exit {result.last_code}). A "
            "real run-time OOM -> exit 75 must classify RETRY and the re-fire "
            "must complete; stranding here is the multi-hour-run loss invariant "
            "(4) / the 4afc5e9 tempfail exists to prevent (GOAL §7)."
        )
        assert result.last_code == 0
        assert result.attempts == 2, (
            f"the launcher took {result.attempts} attempt(s) — expected exactly 2 "
            "(one OOM-deferred, one completing). A FATAL on the OOM (main() "
            "returning 1 instead of 75) or a missing RETRY would break this."
        )
        assert sleeps == [0.0], (
            f"the launcher backed off {sleeps!r} — expected exactly one tempfail "
            "backoff between the OOM and the completing re-fire (the 'launcher "
            "survives' behavior, not an instant fatal)."
        )
        # The OOM-attempt's banked arm carried across the --ledger: the completing
        # re-fire resumed from it, not from zero — the resume the tempfail exists
        # to enable, now proven joined to the launcher that polls for it.
        assert out.exists(), "the completing re-fire did not write the deposit."
        banked = [
            ln for ln in ledger.read_text().splitlines()
            if ln.strip() and json.loads(ln).get("type") == "arm"
        ]
        assert len(banked) == 4, (
            f"the ledger banks {len(banked)} arm(s) — expected 4 (attempt 1 banked "
            "1 before the OOM, the re-fire banked the other 3). Fewer means the "
            "banked arm did NOT carry across the launcher retry, so a re-fire "
            "would re-burn GPU on arms it already trained."
        )

    def test_worker_non_oom_runtimeerror_is_launcher_fatal_not_retried(
        self, monkeypatch, tmp_path,
    ):
        # The complement of the join: a non-OOM RuntimeError from the real arm
        # loop must reach main()'s reraise -> exit 1 -> launcher FATAL, NOT be
        # mis-retried as tempfail contention. Infinite-looping a real bug
        # (scope drift / a non-OOM device error) behind the retry backoff is the
        # other half of GOAL §7 here: the launcher must DISTINGUISH the retryable
        # 75 from the fatal 1 when fed a code the real worker produced — the
        # slice tests (classify(1) -> FATAL with a bare int; _FakeRunner([1]))
        # never prove the worker actually EMITS that 1 for a non-OOM failure.
        import scripts.run_freeze_validloss_ci_9b as worker_mod
        import scripts.launch_freeze_ci_9b_full as launcher_mod

        monkeypatch.setattr(worker_mod, "load_tokenizer", lambda cfg: _CharTokenizer())
        monkeypatch.setattr(worker_mod, "load_base_model", lambda cfg: _tiny_llama(8))
        monkeypatch.setattr(
            worker_mod, "_load_dolly_records", lambda *a, **kw: _fake_dolly_records(),
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(_cpu_cfg(max_steps=6, base_rank=16), cfg_path)
        ledger = tmp_path / "ledger.jsonl"
        out = tmp_path / "deposit.json"

        def _fatal_arm(*a, **kw):  # noqa: ARG001
            # A non-OOM RuntimeError (is_cuda_oom -> False): main() reraises it,
            # the (real) subprocess would exit 1, and the launcher must STOP.
            raise RuntimeError("synthetic non-OOM failure: device-side assert")

        monkeypatch.setattr(worker_mod, "arm_valid_loss_9b", _fatal_arm)
        attempt_state = {"attempt": 0, "arm_calls": 0}

        sleeps: list[float] = []
        result = launcher_mod.run_loop(
            interpreter="/in-process/python",
            module_argv=self._common_worker_flags(cfg_path, ledger, out),
            max_attempts=3, deadline_seconds=None,
            resume_sleep=0.0, tempfail_sleep=0.0,
            sleep_fn=lambda s: sleeps.append(s),
            now_fn=(lambda: 0.0),
            runner=self._runner(worker_mod, attempt_state=attempt_state),
        )

        assert result.outcome is launcher_mod.Outcome.FATAL, (
            f"a non-OOM worker RuntimeError classified {result.outcome.value} — "
            "expected FATAL. main() must reraise it (exit 1) so the launcher "
            "STOPS rather than infinite-looping a real bug as retryable "
            "contention (GOAL §7: never swallow a non-OOM failure as tempfail)."
        )
        assert result.last_code == 1, (
            f"the worker's non-OOM RuntimeError surfaced as exit {result.last_code} "
            "— expected 1 (EXIT_UNEXPECTED), the contract code the launcher maps "
            "to FATAL."
        )
        assert result.attempts == 1, (
            f"the launcher retried a fatal worker error {result.attempts}x — a "
            "non-OOM RuntimeError must NOT be retried (it would spin forever on a "
            "real bug behind the backoff)."
        )
        assert sleeps == [], (
            f"the launcher backed off {sleeps!r} for a fatal — a FATAL must stop "
            "immediately, never sleep-and-retry."
        )
        # No deposit for a fatal run (the verdict never assembled).
        assert not out.exists()
