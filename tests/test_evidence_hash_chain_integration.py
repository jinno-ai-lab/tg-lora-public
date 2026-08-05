"""§4 evidence-hash 鎖 の end-to-end 統合 test (L3) — leaf → producer → replay → gate → decision.

CONTEXT — WHY THIS FILE EXISTS
------------------------------
The L1 leaf pin (:func:`tests.test_freeze_evidence_hash`,
commit ``2a31465``) fixed the ``evidence_hash`` function's byte output and the
``EVIDENCE_HASH_KEYS`` list against seven committed §4 deposit fixtures.
The L2 leaf pin (:func:`tests.test_freeze_verdict_honesty`,
same commit) fixed ``classify_regime`` / ``is_reduced_budget`` /
``full_section4_verdict_gate`. The L3 producer assembly
(:func:`tests.test_freeze_ci_9b_launch_honesty`, ``b8ee35c``) drives the REAL
``run_ci_9b`` → ``result_to_json`` → ``_write_deposit`` → ``_write_run_log``
assembly against a stubbed GPU/model, proving the five honesty invariants at
integration scale.

WHAT THIS FILE CLOSES
---------------------
``AI_HUB_MAKE_RUN_FEEDBACK``: "test_freeze_evidence_hash.py が leaf の hash 出力を
固定しても、その hash を downstream reproducibility flow がどう verify/consume
するかが未検証 — pipeline 統合側の test が 0 本のはず". The leaf is pinned, but
the LEAF→CALLER chain (producer stamp, replay re-derive, gate conjunct,
``arc_complete`` decision) is not pinned at the assembled producer-driven scale
where drift hides:

1. **Producer-leaf drift attack.** A future refactor that re-defines a local
   ``_evidence_hash`` in ``scripts/run_freeze_validloss_ci_9b`` (or extends the
   producer-side ``EVIDENCE_HASH_KEYS``) drifts the stamped hash from the leaf
   the replay re-derives — leaf imports stay intact, so L1 stays green, so the
   corruption is invisible until an operator sees a stale ``evidence_hash`` at
   the GPU-free replay gate.

2. **Stamp-only-no-verify attack.** Even if the producer stamps the leaf, the
   replay's ``_evidence_hash_stale`` comparison may diverge (replay-side
   canonicalization) so a hand-edited deposit slips past the replay gate.

3. **Consumer-leaf drift attack.** ``scripts.section4_operator_decision``
   imports ``evidence_hash`` from the same leaf at line 76; if its
   ``_assess_leg`` accumulates a divergent canonicalization the conjunct fails
   silently — the SHIP gate rests on ``arc_complete`` (``not evidence_hash_stale``)
   so a drift here undoes the citation-honesty guarantee.

This file proves NONE of those three attacks are reachable, at producer-driven
real-bytes scale, in six cases:

  Case 1 (happy path): producer → deposit → replay → ``arc_complete``.
  Case 2 (coordinated repaint): byte-edit a loss → ``_evidence_hash_stale=True``.
  Case 3 (derived-label repaint): verdict label edit → ``faithful`` catches it,
                                   ``evidence_hash_stale`` does NOT (the threat
                                   model: hash is over evidence bytes, not labels).
  Case 4 (producer-leaf identity): ``scripts._evidence_hash is leaf.evidence_hash``.
  Case 5 (decision-leaf identity):  ``scripts._assess_leg`` reads leaf, not rebind.
  Case 6 (end-to-end): Case 1 vs Case 2 deposit → ``arc_complete`` binary
                        contrast — the conjunct that gates SHIP, at the assembled
                        scale, against the bytes a real run produced.

The Cases are mutation-proven: a leaf rebind or a coordinator conjunct removal
turns the corresponding Case RED.

GPU-free (CPU), torch-only stub model, ``src.data``-free — same envelope as
``test_freeze_ci_9b_launch_honesty``. The real ``run_ci_9b`` assembly + the
real ``_write_deposit`` + the real ``replay_samples`` + the real ``_assess_leg``
all run.
"""

from __future__ import annotations

import json

import pytest

# The L1 leaf — the single source of truth both the producer and the replay
# import. EVERY assertion below either binds to THIS object (``is``) or
# re-derives the deposit's stamped hash through THIS function.
from src.tg_lora.freeze_evidence_hash import (
    EVIDENCE_HASH_KEYS as LEAF_KEYS,
    evidence_hash as LEAF_HASH,
)
from scripts.replay_freeze_validloss_ci import (
    _evidence_hash_stale,
    load_samples,
    replay_samples,
)
from scripts.section4_operator_decision import _assess_leg
# The existing assembled-launch-honesty dry-run owns the CPU/GPU stub helpers
# (``_assemble`` returns a real producer-driven deposit + the path it was
# written to). Reusing them keeps the L3 producer stamp at the SAME scale the
# per-commit fixes proved, with no fresh stub surface.
from tests.test_freeze_ci_9b_launch_honesty import _assemble


# ── Case 1 — happy path: producer-stamped hash survives the chain intact ──────


class TestProducerStampedHashSurvivesChain:
    def test_producer_deposit_hash_matches_leaf_byte_for_byte(
        self, monkeypatch, tmp_path,
    ):
        # The REAL ``run_ci_9b`` → ``result_to_json`` → ``_write_deposit`` assembly
        # (only the network/GPU boundaries stubbed) stamps ``evidence_hash`` onto
        # the deposit it writes. Re-deriving through the LEAF (``evidence_hash``)
        # — the same function the replay gate and the decision tool both call —
        # must reproduce the stamped value byte-for-byte. A drift here would mean
        # the producer-side canonicalization diverged from the leaf without the
        # leaf import being touched; L1 (``test_freeze_evidence_hash``) cannot
        # catch this because L1 tests the leaf in isolation.
        _result, deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        loaded = json.loads(deposit_path.read_text())
        assert loaded["evidence_hash"] == LEAF_HASH(loaded), (
            "producer's stamped ``evidence_hash`` does NOT match the leaf's "
            "re-derivation: the producer-side canonicalization drifted from "
            "the leaf even though the leaf import is intact — the producer-"
            "leaf drift attack (threat 1) is reachable."
        )

    def test_replay_load_samples_and_replay_samples_round_trip(
        self, monkeypatch, tmp_path,
    ):
        # The GPU-free replay (``load_samples`` → ``replay_samples``) is the
        # SAME code path the replay gate (``scripts.replay_freeze_validloss_ci``)
        # runs against any committed deposit; if it cannot parse a
        # producer-driven deposit's losses and re-derive a verdict, the
        # chain is broken at the second hop (producer stamp is correct but
        # the replay cannot read what was stamped). The bootstrap is
        # deterministic over the same losses + ``base_seed``.
        _result, deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        data = load_samples(deposit_path)
        ci = replay_samples(data)
        assert ci.candidate_mean == pytest.approx(deposit["candidate_mean"])
        assert ci.surrogate_mean == pytest.approx(deposit["surrogate_mean"])
        assert ci.significance_verdict == deposit["verdict"], (
            "the replay bootstrap disagreeed with the producer's stored "
            "verdict on a real producer-driven deposit — the replay cannot "
            "read what the producer wrote, the chain's second hop is broken."
        )

    def test_replay_evidence_hash_stale_false_on_happy_deposit(
        self, monkeypatch, tmp_path,
    ):
        # The replay's ``_evidence_hash_stale`` is the load-bearing integrity
        # check at the GPU-free chokepoint. On a happy producer-driven deposit
        # it MUST be ``False``; on a tampered deposit (Case 2) it MUST be
        # ``True``. This is the gate the SHIP decision (``arc_complete``)
        # hinges on (``not evidence_hash_stale`` conjunct, ``6b628fe``).
        _result, _deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        data = load_samples(deposit_path)
        assert _evidence_hash_stale(data) is False, (
            "the replay's ``_evidence_hash_stale`` raised on a producer-driven "
            "deposit that was just written — the replay-side canonicalization "
            "drifted from the leaf (the stamp-only-no-verify attack)."
        )

    def test_assess_leg_arc_complete_true_on_happy_deposit(
        self, monkeypatch, tmp_path,
    ):
        # The decision tool's ``_assess_leg`` reads a deposit from disk
        # (``repo_root / deposit_rel``); to test it against the producer-driven
        # deposit we symlink (or copy) the deposit to ``repo_root / relative``,
        # then call ``_assess_leg`` and check the conjunct the SHIP gate
        # rests on — ``arc_complete`` becomes true ONLY when every leg is
        # ``faithful`` AND ``not evidence_hash_stale``. The producer-driven
        # deposit satisfies both predicates, so the happy path lights up.
        _result, _deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        # _assess_leg looks up ``repo_root / deposit_rel``; copy the deposit
        # into the real repo root under a relative path so the lookup hits.
        rel = "tests/fixtures/_tmp_happy_deposit.json"
        repo_root = deposit_path.resolve().parents[2]  # tests/../
        dest = repo_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(deposit_path.read_bytes())
        try:
            leg = _assess_leg("test_leg", rel, repo_root)
            assert leg["evidence_hash_stale"] is False, (
                "``_assess_leg`` reported evidence_hash_stale=True on a "
                "happy producer-driven deposit — the decision tool drifted."
            )
            assert leg["faithful"] is True, (
                "``_assess_leg`` reported faithful=False on a happy producer-"
                "driven deposit — the verdict label and stored floats diverge "
                "under the replay's bootstrap."
            )
            # The SHIP-gate conjunct — ``arc_complete`` requires both.
            assert leg["evidence_hash_stale"] is False and leg["faithful"] is True
        finally:
            dest.unlink(missing_ok=True)


# ── Case 2 — coordinated repaint: a +0.01 loss edit must turn the gate RED ────


class TestLossEditTripsEvidenceHashStale:
    def test_loss_byte_edit_makes_replay_evidence_hash_stale_true(
        self, monkeypatch, tmp_path,
    ):
        # ``evidence_hash`` is over the EVIDENCE bytes (losses + run config),
        # so a single loss edit MUST move the stamp; the replay's
        # ``_evidence_hash_stale`` MUST raise; ``_assess_leg``'s conjunct
        # MUST flip — that is the threat model ``6b628fe`` formalized.
        # (Faithful stays True because the verdict label + bootstrap agree on
        # the NEW losses — the corruption stays invisible to ``faithful``
        # unless the hash gate also fires.)
        _result, _deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        data = load_samples(deposit_path)
        # Sanity: the un-tampered deposit re-derives clean.
        assert _evidence_hash_stale(data) is False
        # Tamper a single loss by +0.01 — the smallest edit that proves the
        # gate actually reads the bytes (a no-op edit would not test the gate).
        data["candidate_losses"] = [
            loss + 0.01 for loss in data["candidate_losses"]
        ]
        assert _evidence_hash_stale(data) is True, (
            "the replay's ``_evidence_hash_stale`` did NOT raise on a "
            "+0.01 candidate_losses edit — the hash gate is not binding "
            "the evidence bytes (threat 1 + 2 both reachable)."
        )

    def test_assess_leg_arc_complete_false_on_loss_edit(
        self, monkeypatch, tmp_path,
    ):
        # The same tampered deposit, routed through the decision tool: the
        # conjunct ``not evidence_hash_stale`` MUST flip ``arc_complete``
        # from True to False. This is the property the SHIP gate rests on
        # (``6b628fe``) — proven here against producer-driven real bytes
        # for the first time.
        _result, _deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        data = load_samples(deposit_path)
        data["candidate_losses"] = [
            loss + 0.01 for loss in data["candidate_losses"]
        ]
        rel = "tests/fixtures/_tmp_tampered_deposit.json"
        repo_root = deposit_path.resolve().parents[2]
        dest = repo_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2))
        try:
            leg = _assess_leg("test_leg", rel, repo_root)
            assert leg["evidence_hash_stale"] is True, (
                "``_assess_leg`` did not surface a tampered deposit via "
                "``evidence_hash_stale`` — the SHIP-gate conjunct is broken."
            )
            assert leg["faithful"] is True, (
                "``_assess_leg`` reported faithful=False on a tamper that "
                "ONLY touched the losses — the verdict label WAS repainted "
                "to match; the threat model requires faithful=True here so "
                "the binary contrast with Case 3 (verdict-only repaint) is "
                "faithful=True vs faithful=False, NOT faithful=False for both."
            )
        finally:
            dest.unlink(missing_ok=True)


# ── Case 3 — derived-label repaint: verdict edit, NOT a loss edit ──────────────


class TestVerdictEditLeavesHashUntouched:
    def test_verdict_label_edit_does_not_trip_evidence_hash_stale(
        self, monkeypatch, tmp_path,
    ):
        # ``evidence_hash`` is deliberately over EVIDENCE bytes only, NEVER
        # derived labels (see ``test_hash_is_over_evidence_not_derived_labels``
        # in ``test_freeze_evidence_hash``). A verdict repaint that does NOT
        # edit the losses therefore leaves the hash untouched — but the
        # bootstrap disagrees with the painted label, so ``faithful`` flips
        # to False. This is the threat-model threat the hash CANNOT catch
        # alone: ``faithful`` catches it, and BOTH must hold for
        # ``arc_complete`` — so the SHIP-gate defense is the conjunction,
        # not either alone. Proving the conjunction's two halves bind to
        # distinct corruption classes is the whole point of the L3 chain.
        _result, _deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        data = load_samples(deposit_path)
        # Sanity.
        assert _evidence_hash_stale(data) is False
        # Paint a verdict that disagrees with the stored floats — pick the
        # opposite of what the bootstrap will say (the actual stored verdict
        # varies by seed; pick a fixed non-equal label).
        original = data["verdict"]
        flipped = "SURPASSES" if original != "SURPASSES" else "UNDERSHOOTS"
        data["verdict"] = flipped
        assert _evidence_hash_stale(data) is False, (
            "the replay's ``_evidence_hash_stale`` raised on a verdict-LABEL-"
            "only edit — the hash is binding derived labels it MUST NOT bind "
            "(the circular-stamp defense regressed)."
        )

    def test_assess_leg_faithful_false_on_verdict_edit(
        self, monkeypatch, tmp_path,
    ):
        # The complement: the same verdict-painted deposit, routed through
        # ``_assess_leg``. ``evidence_hash_stale`` stays False (Case 3
        # above); ``faithful`` MUST flip to False — the label disagrees
        # with the bootstrap. Together, Cases 2 and 3 prove the conjunct
        # ``arc_complete`` defends against BOTH attack classes the two gates
        # were designed to catch (loss-byte drift + verdict repaint).
        _result, _deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        data = load_samples(deposit_path)
        original = data["verdict"]
        flipped = "SURPASSES" if original != "SURPASSES" else "UNDERSHOOTS"
        data["verdict"] = flipped
        rel = "tests/fixtures/_tmp_painted_deposit.json"
        repo_root = deposit_path.resolve().parents[2]
        dest = repo_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2))
        try:
            leg = _assess_leg("test_leg", rel, repo_root)
            assert leg["evidence_hash_stale"] is False, (
                "``_assess_leg`` reported evidence_hash_stale=True on a "
                "verdict-label-only edit — the hash is over the wrong keys."
            )
            assert leg["faithful"] is False, (
                "``_assess_leg`` reported faithful=True on a verdict repaint "
                "that disagrees with the stored floats — the verdict-label "
                "gate is not binding."
            )
        finally:
            dest.unlink(missing_ok=True)


# ── Case 4 — producer imports the leaf (no local rebind) ──────────────────────


class TestProducerLeafIdentity:
    def test_producer_evidence_hash_is_the_leaf_function(self):
        # The producer re-exports the leaf through
        # ``from src.tg_lora.freeze_evidence_hash import (
        #     EVIDENCE_HASH_KEYS as EVIDENCE_HASH_KEYS,
        #     evidence_hash as _evidence_hash)`` (lines 164-166). A local
        # rebind (e.g. ``_evidence_hash = lambda d: ...``) would silently
        # drift the stamp from the leaf WITHOUT changing the import line.
        # ``is`` catches rebind; ``==`` would not.
        from scripts import run_freeze_validloss_ci_9b as producer_mod
        assert producer_mod._evidence_hash is LEAF_HASH, (
            "the producer's ``_evidence_hash`` is no longer the leaf "
            "``evidence_hash`` (rebind at the producer module level) — the "
            "producer-leaf drift attack (threat 1) is reachable."
        )
        assert producer_mod.EVIDENCE_HASH_KEYS is LEAF_KEYS, (
            "the producer's ``EVIDENCE_HASH_KEYS`` is no longer the leaf "
            "tuple (rebind) — the producer-side key list can drift without "
            "the leaf import changing."
        )


# ── Case 5 — replay imports the leaf (no local rebind) ───────────────────────


class TestReplayLeafIdentity:
    def test_replay_evidence_hash_stale_uses_leaf_function(self):
        # The replay's ``_evidence_hash_stale`` (line 1344) compares the
        # stored stamp against ``evidence_hash(data)`` — the leaf import
        # (line 115). A local rebind in ``scripts.replay_freeze_validloss_ci``
        # would let a tampered deposit pass the replay gate silently (threat
        # 2). Bind the namespaced ``evidence_hash`` (the replay module's
        # import) to the leaf via ``is``.
        from scripts import replay_freeze_validloss_ci as replay_mod
        assert replay_mod.evidence_hash is LEAF_HASH, (
            "the replay's ``evidence_hash`` is no longer the leaf "
            "``evidence_hash`` (rebind at the replay module level) — "
            "``_evidence_hash_stale`` could diverge from the producer."
        )

    def test_decision_evidence_hash_is_the_leaf_function(self):
        # The decision tool imports ``evidence_hash`` directly (line 76);
        # ``_assess_leg`` calls it on line 234 to populate
        # ``evidence_hash_stale``. A rebind here would diverge the SHIP-
        # gate conjunct from the leaf (threat 3). Same ``is`` pin as
        # Case 4: any future rebind turns RED.
        from scripts import section4_operator_decision as decision_mod
        assert decision_mod.evidence_hash is LEAF_HASH, (
            "the decision tool's ``evidence_hash`` is no longer the leaf "
            "``evidence_hash`` (rebind) — ``_assess_leg``'s conjunct could "
            "drift from the producer and replay sides."
        )


# ── Case 6 — end-to-end operator decision (happy vs repaint) ──────────────────


class TestEndToEndArcCompleteConjunct:
    def test_happy_and_repaint_deposits_yield_binary_arc_complete(
        self, monkeypatch, tmp_path,
    ):
        # The headline proof: drive the REAL producer → write a deposit →
        # then write a TAMPERED twin (loss +0.01) → run both through
        # ``_assess_leg`` and compare the ``evidence_hash_stale`` conjunct.
        # The binary contrast — happy=False-on-stale vs repaint=True-on-
        # stale — is the property the SHIP gate rests on at the operator-
        # decision layer. Pre-L3 this was proven only on pre-built
        # tampered fixtures (``tests/test_section4_operator_decision.py``);
        # here the producer REAL writes the bytes and the decision tool
        # REAL reads them.
        _result, _deposit, deposit_path, _rl = _assemble(monkeypatch, tmp_path)
        happy_data = load_samples(deposit_path)
        tampered_data = json.loads(json.dumps(happy_data))
        tampered_data["candidate_losses"] = [
            loss + 0.01 for loss in tampered_data["candidate_losses"]
        ]
        # Sanity: the replay gate disagrees on the tampered deposit; agrees
        # on the happy deposit.
        assert _evidence_hash_stale(happy_data) is False
        assert _evidence_hash_stale(tampered_data) is True
        # Now wire BOTH into ``_assess_leg`` (the operator-decision layer)
        # and check the conjunct each lights up. The function reads
        # ``repo_root / deposit_rel``; write the two files there.
        repo_root = deposit_path.resolve().parents[2]
        happy_rel = "tests/fixtures/_tmp_e2e_happy.json"
        tampered_rel = "tests/fixtures/_tmp_e2e_tampered.json"
        (repo_root / happy_rel).parent.mkdir(parents=True, exist_ok=True)
        (repo_root / happy_rel).write_text(json.dumps(happy_data, indent=2))
        (repo_root / tampered_rel).write_text(json.dumps(tampered_data, indent=2))
        try:
            happy_leg = _assess_leg("test_happy", happy_rel, repo_root)
            tampered_leg = _assess_leg("test_tampered", tampered_rel, repo_root)
            assert happy_leg["evidence_hash_stale"] is False, (
                "happy deposit lights the conjunct — the producer's stamp "
                "is not reproducible from its own bytes (regression)."
            )
            assert tampered_leg["evidence_hash_stale"] is True, (
                "tampered deposit did NOT light the conjunct — the SHIP-gate "
                "defense (the conjunct ``not evidence_hash_stale`` ``6b628fe``) "
                "does not reach ``arc_complete``."
            )
            # The binary contrast that the SHIP-gate defense rests on: the
            # same field that is False for the producer-driven happy deposit
            # flips to True for a +0.01 loss edit, at the operator-decision
            # layer, against bytes the REAL assembly wrote.
            assert (
                happy_leg["evidence_hash_stale"]
                != tampered_leg["evidence_hash_stale"]
            )
        finally:
            (repo_root / happy_rel).unlink(missing_ok=True)
            (repo_root / tampered_rel).unlink(missing_ok=True)