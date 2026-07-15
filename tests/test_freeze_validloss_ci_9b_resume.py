"""GPU-free tests for the resumable per-arm ledger in
``scripts/run_freeze_validloss_ci_9b.py``.

The full-budget §4 verdict run (``make freeze-validloss-ci-9b-full``) is ~hours
of 9B GPU (9 arms × 1500 steps) on a shared 12 GB card that is routinely
preempted by concurrent runs (one is holding the GPU this very iteration).
Without the ledger, ``run_ci_9b`` banks every arm only in memory and serializes
the deposit at the very end, so one interruption deep in the run loses ALL
completed arms and the verdict stays blocked indefinitely — the next free-GPU
window restarts from zero. The ledger streams each completed arm to JSONL and
skips it on a re-run whose run-config fingerprint matches, so an interrupted run
RESUMES instead of restarting.

These tests pin the ledger's pure logic without CUDA: spec-building equivalence
to the legacy inline comprehensions, fingerprint invalidation on config change,
ledger roundtrip / stale-ignore / torn-line tolerance, the resume-skip behavior
(mutation-sensitive on the exact runner-call set), the no-ledger byte-identical
default, and the incomplete-resume guard. A stub ``runner`` stands in for the
GPU arm — the real model is never loaded here.
"""

from __future__ import annotations

import pytest

from scripts.run_freeze_validloss_ci_9b import (
    LEDGER_VERSION,
    IncompleteResumeError,
    _arm_specs,
    _collect_arms,
    _config_fingerprint,
    _gather_arm_curves,
    _require_runnable_arms,
    append_arm_to_ledger,
    load_ledger,
)
from scripts.run_freeze_validloss_ci_9b import (
    candidate_order_9b,
    control_order_9b,
)
from src.tg_lora.freeze_schedule import random_freeze_order

SCOPE = {10, 11, 12, 13, 14}
SCOPE_SORTED = sorted(SCOPE)


def _fp(**overrides):
    """A default full-budget fingerprint with one field overridable."""
    kw = dict(
        total_steps=1500, warmup_steps=150, depth=3, spacing=450,
        seq_len=1024, train_examples=600, valid_examples=64,
        model="Qwen/Qwen3.5-9B", scope_label="last_25_percent",
        active_scope=SCOPE_SORTED, dataset="databricks/databricks-dolly-15k",
        use_local_loss=True, base_seed=0,
    )
    kw.update(overrides)
    return _config_fingerprint(**kw)


# --- spec builder equivalence to the legacy comprehensions -------------------

def test_arm_specs_match_legacy_semantics():
    """The refactor must reproduce the exact role/order/seed/depth the inline
    comprehensions assigned — a fresh run with no ledger is identical to before.
    """
    specs = _arm_specs(
        active_indices=SCOPE, scope_sorted=SCOPE_SORTED, base_seed=0, depth=3,
        n_candidate=3, n_surrogate=3, n_control=2, n_baseline=1,
    )
    cand_order = candidate_order_9b(SCOPE)
    ctrl_order = control_order_9b(SCOPE)
    expected = []
    for i in range(3):
        expected.append(("candidate", i, tuple(cand_order), 0 + i, 3))
    for i in range(3):
        expected.append(
            ("surrogate", i, tuple(random_freeze_order(SCOPE_SORTED, 0 + 1000 + i)),
             100 + i, 3)
        )
    for i in range(2):
        expected.append(("control", i, tuple(ctrl_order), 200 + i, 3))
    expected.append(("baseline", 0, tuple(SCOPE_SORTED), 300, 0))

    assert len(specs) == len(expected) == 9
    for spec, (role, idx, order, seed, depth) in zip(specs, expected):
        assert spec["role"] == role
        assert spec["index"] == idx
        assert tuple(spec["order"]) == order
        assert spec["seed"] == seed
        assert spec["depth"] == depth


def test_arm_specs_execution_order_is_candidate_first():
    """Arms execute candidate → surrogate → control → baseline, matching the
    original comprehension order (so the headline A/B completes earliest)."""
    specs = _arm_specs(
        active_indices=SCOPE, scope_sorted=SCOPE_SORTED, base_seed=0, depth=3,
        n_candidate=2, n_surrogate=2, n_control=2, n_baseline=2,
    )
    assert [s["role"] for s in specs] == [
        "candidate", "candidate", "surrogate", "surrogate",
        "control", "control", "baseline", "baseline",
    ]


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_changes_with_total_steps():
    """A 96-step smoke ledger must not seed a 1500-step full run."""
    assert _fp(total_steps=96) != _fp(total_steps=1500)


def test_fingerprint_changes_with_active_scope():
    """Scopes that share a layer count must not collide (the indices pin it)."""
    assert _fp(active_scope=[10, 11, 12]) != _fp(active_scope=[10, 11, 12, 13])


def test_fingerprint_changes_with_use_local_loss():
    assert _fp(use_local_loss=True) != _fp(use_local_loss=False)


def test_fingerprint_carries_ledger_version():
    """A future schema bump reads as stale via the embedded version."""
    assert _fp()["ledger_version"] == LEDGER_VERSION


# --- ledger I/O --------------------------------------------------------------

def test_ledger_roundtrip(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    fp = _fp()
    specs = _arm_specs(
        active_indices=SCOPE, scope_sorted=SCOPE_SORTED, base_seed=0, depth=3,
        n_candidate=3, n_surrogate=3, n_control=0, n_baseline=0,
    )
    for spec in specs:
        append_arm_to_ledger(ledger, fp, spec, 1.0 + spec["index"], {"mark": spec["index"]})

    cached = load_ledger(ledger, fp)
    assert set(cached) == {(r, i) for r in ("candidate", "surrogate") for i in range(3)}
    for (role, idx), (loss, prov) in cached.items():
        assert loss == pytest.approx(1.0 + idx)
        assert prov["mark"] == idx


def test_load_ledger_missing_file_returns_empty(tmp_path):
    assert load_ledger(tmp_path / "absent.jsonl", _fp()) == {}


def test_load_ledger_ignores_stale_fingerprint(tmp_path):
    """A ledger from a 96-step run must be ignored when a 1500-step run resumes."""
    ledger = tmp_path / "ledger.jsonl"
    old_fp = _fp(total_steps=96)
    spec = _arm_specs(
        active_indices=SCOPE, scope_sorted=SCOPE_SORTED, base_seed=0, depth=3,
        n_candidate=1, n_surrogate=0, n_control=0, n_baseline=0,
    )[0]
    append_arm_to_ledger(ledger, old_fp, spec, 1.0, {"stale": True})
    assert load_ledger(ledger, _fp(total_steps=1500)) == {}


def test_load_ledger_tolerates_torn_trailing_line(tmp_path):
    """A partially-flushed arm line (crash mid-append) must not brick the resume."""
    ledger = tmp_path / "ledger.jsonl"
    fp = _fp()
    spec = _arm_specs(
        active_indices=SCOPE, scope_sorted=SCOPE_SORTED, base_seed=0, depth=3,
        n_candidate=1, n_surrogate=0, n_control=0, n_baseline=0,
    )[0]
    append_arm_to_ledger(ledger, fp, spec, 1.0, {"ok": True})
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"arm","role":"candidate","index":1,"valid_loss":')  # truncated

    cached = load_ledger(ledger, fp)
    assert ("candidate", 0) in cached
    assert ("candidate", 1) not in cached  # malformed line skipped, not fatal


# --- _collect_arms resume behavior ------------------------------------------

def _full_specs():
    return _arm_specs(
        active_indices=SCOPE, scope_sorted=SCOPE_SORTED, base_seed=0, depth=3,
        n_candidate=3, n_surrogate=3, n_control=0, n_baseline=0,
    )


def _stub_runner(call_log, loss_for):
    def _runner(spec):
        call_log.append((spec["role"], spec["index"]))
        return (
            loss_for(spec["role"], spec["index"]),
            {"frozen_layers": [], "n_trainable_params": 1,
             "last_train_loss": 0.0, "final_ce_train_loss": 1.5},
        )
    return _runner


def test_collect_arms_fresh_runs_all_when_no_ledger():
    """No ledger ⇒ every spec executes, none resumed (the byte-identical path)."""
    call_log = []
    runner = _stub_runner(call_log, lambda r, i: 1.5 + i * 0.01)
    collected, n_resumed = _collect_arms(_full_specs(), runner)

    assert n_resumed == 0
    assert len(collected["candidate"]) == 3
    assert len(collected["surrogate"]) == 3
    assert call_log == [("candidate", 0), ("candidate", 1), ("candidate", 2),
                        ("surrogate", 0), ("surrogate", 1), ("surrogate", 2)]


def test_collect_arms_resume_skips_banked_arms(tmp_path):
    """A ledger with candidate[0..2] + surrogate[0..1] banked resumes ONLY
    surrogate[2] on the GPU runner — the headline value of the ledger."""
    ledger = tmp_path / "ledger.jsonl"
    fp = _fp()
    specs = _full_specs()
    banked = [("candidate", 0), ("candidate", 1), ("candidate", 2),
              ("surrogate", 0), ("surrogate", 1)]
    by_key = {(s["role"], s["index"]): s for s in specs}
    for role, idx in banked:
        append_arm_to_ledger(ledger, fp, by_key[(role, idx)], 1.6, {"pre": True})

    call_log = []
    runner = _stub_runner(call_log, lambda r, i: 9.99)  # sentinel for the re-run arm
    collected, n_resumed = _collect_arms(
        _full_specs(), runner, ledger_path=ledger, fingerprint=fp,
    )

    assert call_log == [("surrogate", 2)]          # only the missing arm ran
    assert n_resumed == len(banked) == 5
    assert len(collected["candidate"]) == 3
    assert len(collected["surrogate"]) == 3
    # banked arms keep their ledger loss; the freshly-run arm carries the sentinel
    assert [v for v, _ in collected["surrogate"]] == [1.6, 1.6, 9.99]


def test_collect_arms_banks_newly_run_arms_then_second_pass_hits_all(tmp_path):
    """A first pass banks every arm; a second pass with the same ledger reuses
    every arm (idempotent re-verify) — none re-run."""
    ledger = tmp_path / "ledger.jsonl"
    fp = _fp()
    call_log = []
    _collect_arms(
        _full_specs(), _stub_runner(call_log, lambda r, i: 2.0 + i),
        ledger_path=ledger, fingerprint=fp,
    )
    assert len(load_ledger(ledger, fp)) == 6  # all banked

    call_log2 = []
    _, n_resumed2 = _collect_arms(
        _full_specs(), _stub_runner(call_log2, lambda r, i: 99.9),
        ledger_path=ledger, fingerprint=fp,
    )
    assert call_log2 == []
    assert n_resumed2 == 6


def test_collect_arms_stale_ledger_runs_everything(tmp_path):
    """A stale-config ledger is ignored: all arms execute fresh and the ledger
    is rewritten under the new fingerprint."""
    ledger = tmp_path / "ledger.jsonl"
    stale_fp = _fp(total_steps=96)
    spec = _arm_specs(
        active_indices=SCOPE, scope_sorted=SCOPE_SORTED, base_seed=0, depth=3,
        n_candidate=1, n_surrogate=0, n_control=0, n_baseline=0,
    )[0]
    append_arm_to_ledger(ledger, stale_fp, spec, 1.0, {"old": True})

    full_fp = _fp(total_steps=1500)
    call_log = []
    collected, n_resumed = _collect_arms(
        _full_specs(), _stub_runner(call_log, lambda r, i: 1.7),
        ledger_path=ledger, fingerprint=full_fp,
    )
    assert n_resumed == 0
    assert len(call_log) == 6                     # nothing reused from the stale ledger
    # the ledger now holds arms under the NEW fingerprint
    assert len(load_ledger(ledger, full_fp)) == 6


# --- loss-curve capture pipeline (run-log artifact) --------------------------


def test_collect_arms_loss_curve_capture_pipelines_through():
    """The run-log capture pipeline end-to-end with the real ``_collect_arms``:
    the orchestrator seeds each spec with an empty ``_loss_curve``, the runner
    (``arm_valid_loss_9b`` in production) appends each step's loss to it,
    ``_collect_arms`` drives the runner, and ``_gather_arm_curves`` pairs the
    captured curves with results in plan order. Only the GPU model is stubbed —
    the seed → capture → gather glue is exercised, not just the writer.
    """
    specs = _full_specs()
    # Orchestrator seeding (mirrors run_ci_9b when --run-log is set): every
    # planned arm carries an empty _loss_curve the runner forwards as the sink.
    for spec in specs:
        spec["_loss_curve"] = []

    def _runner(spec):
        # Stand-in for arm_valid_loss_9b's per-step append: a short synthetic
        # trajectory so each arm's captured curve is a distinguishable, ordered list.
        spec["_loss_curve"].extend([2.0, 1.8, 1.6])
        return (
            1.5 + spec["index"] * 0.01,
            {"frozen_layers": [14], "n_trainable_params": 1,
             "last_train_loss": 0.0, "final_ce_train_loss": 1.5},
        )

    collected, n_resumed = _collect_arms(specs, _runner)
    assert n_resumed == 0
    arms = _gather_arm_curves(specs, collected)
    # Plan order (candidate block then surrogate block, matching _arm_specs).
    assert [a["role"] for a in arms] == [
        "candidate", "candidate", "candidate",
        "surrogate", "surrogate", "surrogate",
    ]
    # Every planned arm captured its 3-step trajectory through the real loop glue.
    assert all(a["loss_curve"] == [2.0, 1.8, 1.6] for a in arms)
    # The valid_loss is paired from collected (the run result), not the curve stub.
    assert arms[0]["final_valid_loss"] == 1.5
    assert arms[3]["final_valid_loss"] == 1.5  # surrogate index 0


def test_collect_arms_resumed_arm_leaves_empty_curve(tmp_path):
    """A resumed arm (replayed from the ledger, never re-run) keeps the empty
    ``_loss_curve`` the orchestrator seeded — the run log records an honest
    "dynamics unavailable for this arm", never a fabricated trajectory."""
    ledger = tmp_path / "ledger.jsonl"
    fp = _fp()
    specs = _full_specs()
    for spec in specs:
        spec["_loss_curve"] = []
    # Bank candidate[0] up front; the runner never touches it on resume.
    by_key = {(s["role"], s["index"]): s for s in specs}
    append_arm_to_ledger(ledger, fp, by_key[("candidate", 0)], 1.9, {"pre": True})

    def _runner(spec):
        spec["_loss_curve"].extend([2.0, 1.8])
        return (1.5, {"frozen_layers": [14], "n_trainable_params": 1,
                      "last_train_loss": 0.0, "final_ce_train_loss": 1.5})

    collected, n_resumed = _collect_arms(
        specs, _runner, ledger_path=ledger, fingerprint=fp,
    )
    assert n_resumed == 1
    arms = _gather_arm_curves(specs, collected)
    resumed = next(a for a in arms if a["role"] == "candidate" and a["index"] == 0)
    # Replayed from the ledger: valid_loss banked (1.9), but no curve captured.
    assert resumed["final_valid_loss"] == 1.9
    assert resumed["loss_curve"] == []
    # A freshly-run arm DID capture its curve.
    fresh = next(a for a in arms if a["role"] == "candidate" and a["index"] == 1)
    assert fresh["loss_curve"] == [2.0, 1.8]


# --- incomplete-resume guard -------------------------------------------------

def test_require_runnable_arms_raises_when_either_side_empty():
    with pytest.raises(IncompleteResumeError):
        _require_runnable_arms([1.5], [])
    with pytest.raises(IncompleteResumeError):
        _require_runnable_arms([], [1.5])


def test_require_runnable_arms_passes_when_both_present():
    _require_runnable_arms([1.5, 1.6], [1.7])  # no raise


# --- additive deposit keys default gracefully (no-ledger compat) -------------

def test_result_to_json_resume_keys_default_for_ledgerless_result():
    """A result dict built without the ledger (every existing deposit) must
    still serialize — the resume keys are additive with safe defaults."""
    from scripts.run_freeze_validloss_ci_9b import result_to_json
    from src.tg_lora.freeze_surrogate_ci import surrogate_valid_loss_ci

    ci = surrogate_valid_loss_ci([1.5, 1.6], [1.7, 1.8], seed=0)
    result = {
        "ci": ci, "reduced_budget": True, "proxy_scale": False,
        "candidate_losses": [1.5, 1.6], "surrogate_losses": [1.7, 1.8],
        "candidate_order": [14, 13, 12], "device": "cuda", "model": "Qwen/Qwen3.5-9B",
        "dataset": "databricks/databricks-dolly-15k", "total_steps": 96,
        "warmup_steps": 12, "depth": 3, "spacing": 10, "n_candidate": 2,
        "n_surrogate": 2, "base_seed": 0, "active_scope": [12, 13, 14],
        "scope_label": "last_25_percent", "n_active_layers": 3,
        "scope_trainable_params": 1234, "seq_len": 1024, "train_examples": 48,
        "valid_examples": 32, "use_local_loss": True, "cfg_max_steps": 1500,
        "candidate_provenance": [{"final_ce_train_loss": 1.5}],
        "surrogate_provenance": [{"final_ce_train_loss": 1.5}],
        "control_losses": [], "control_order": [10, 11], "n_control": 0,
        "control_provenance": [], "direction_ci": None,
        "baseline_losses": [], "n_baseline": 0, "baseline_provenance": [],
        "baseline_ci": None,
    }
    out = result_to_json(result)
    assert out["resumed_arm_count"] == 0
    assert out["ledger_path"] is None
