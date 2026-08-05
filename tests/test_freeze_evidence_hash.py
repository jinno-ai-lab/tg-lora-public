"""Unit tests for the §7 reproducibility evidence-hash leaf (``freeze_evidence_hash``).

Why this file exists. ``evidence_hash`` is the **single source of truth**
(SYSTEM_CONSTITUTION Rule #3) for the deposit's frozen evidence bytes: the
producer (``scripts.run_freeze_validloss_ci_9b``) stamps ``evidence_hash`` onto a
deposit at harvest, and the GPU-free replay
(``scripts.replay_freeze_validloss_ci``) re-derives that hash to cross-check any
committed deposit without torch. Keeping the key list + canonicalization in ONE
place means the producer's stamp and the replay's cross-check cannot drift apart.
It is a pure-stdlib (``hashlib`` + ``json``) leaf with no torch / ``src.data``
dependency — that is the whole reason the replay can import it GPU-free.

It is already exercised in ``tests/test_run_freeze_validloss_ci_9b.py`` and
``tests/test_replay_freeze_validloss_ci.py`` — **both of which are un-importable
on the public mirror**: each does a module-level ``import scripts.run_freeze_validloss_ci_9b``
/ ``scripts.replay_freeze_validloss_ci`` that pulls the private ``src.data``
pipeline, so both error at collection time and NONE of their assertions execute
here. Measured coverage of the leaf on this checkout was **0%** (every line),
meaning a broken canonicalization (dropped ``sort_keys``, changed separators, a
key removed from ``EVIDENCE_HASH_KEYS``) would pass this mirror's CI silently.

This file is the **src.data-free, torch-free runnable twin**: it pins the
canonicalization discipline and the evidence-vs-derived-label boundary directly,
mutation-proven, AND — the strongest check — re-derives the hash against every
committed §4 deposit fixture so any drift shows up as RED on real bytes. It is
NOT a second copy of the hash logic; it tests the canonical function the producer
itself calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.tg_lora.freeze_evidence_hash import EVIDENCE_HASH_KEYS, evidence_hash

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Every committed §4 deposit that carries a stamped evidence_hash. These are the
# repo's load-bearing artifacts — the SHIP verdict rests on their evidence bytes
# being immutable (``6b628fe``: the gate now requires ``not evidence_hash_stale``).
# Each must re-derive to its stored hash through THIS leaf, byte-for-byte.
STAMPED_DEPOSIT_FIXTURES = [
    "freeze_validloss_ci_9b_baseline.json",
    "freeze_validloss_ci_9b_direction.json",
    "freeze_validloss_ci_9b_full.json",
    "freeze_validloss_ci_9b_full_heterogeneous.json",
    "freeze_validloss_ci_9b_generalization.json",
    "freeze_validloss_ci_9b_heterogeneous_generalization.json",
    "freeze_validloss_ci_9b_surrogate.json",
]

# Derived labels that the hash MUST exclude — hashing these would make the
# integrity check circular (a verdict painted on the evidence would re-stamp to
# the same hash). Grounded in the leaf's threat model.
DERIVED_LABEL_KEYS = (
    "verdict",
    "gate",
    "regime",
    "citable_as_full_section4_verdict",
    "faithful",
    "evidence_hash_stale",
)

# A representative slice of the real evidence keys (raw measurements + run config)
# that the hash MUST bind — an edit to any of these is exactly the "coordinated
# repaint of the evidence bytes" the stamp exists to catch.
EVIDENCE_BYTE_KEYS = (
    "candidate_losses",
    "surrogate_losses",
    "model",
    "architecture",
    "total_steps",
    "depth",
)


def _deposit() -> dict:
    """A minimal deposit carrying a couple of evidence bytes + derived labels."""
    return {
        "candidate_losses": [1.718, 1.719, 1.717],
        "surrogate_losses": [1.886, 1.887],
        "model": "llama-9b",
        "total_steps": 1500,
        "depth": 4,
        # derived labels (must NOT enter the hash):
        "verdict": "TIES",
        "citable_as_full_section4_verdict": True,
        "faithful": True,
    }


# ---------------------------------------------------------------------------
# Output shape + determinism
# ---------------------------------------------------------------------------


class TestShapeAndDeterminism:
    def test_returns_sha256_hex(self):
        h = evidence_hash(_deposit())
        assert isinstance(h, str)
        assert len(h) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", h)

    def test_deterministic_for_identical_input(self):
        assert evidence_hash(_deposit()) == evidence_hash(_deposit())

    def test_independent_of_outer_dict_insertion_order(self):
        # The deposit is projected onto the FIXED-ORDER EVIDENCE_HASH_KEYS tuple
        # and then sort_keys=True canonicalizes; re-ordering the outer dict (or
        # adding non-evidence keys) cannot perturb the hash.
        d = _deposit()
        reordered = {k: d[k] for k in reversed(list(d))}
        reordered_with_noise = {**reordered, "irrelevant_key": [1, 2, 3]}
        assert evidence_hash(reordered) == evidence_hash(d)
        assert evidence_hash(reordered_with_noise) == evidence_hash(d)


# ---------------------------------------------------------------------------
# The evidence-vs-derived-label boundary (the leaf's core honesty property)
# ---------------------------------------------------------------------------


class TestEvidenceVsDerivedLabels:
    def test_evidence_byte_edit_changes_the_hash(self):
        # An edit to a raw measurement / run-determining config MUST change the
        # hash — otherwise the stamp could not detect a coordinated repaint of
        # the evidence bytes. Mutation: dropping a key from EVIDENCE_HASH_KEYS
        # turns its row red (hash stops reacting to that byte).
        base = _deposit()
        base_hash = evidence_hash(base)
        for key in EVIDENCE_BYTE_KEYS:
            mutated = {**base, key: "MUTATED_VALUE"}
            assert evidence_hash(mutated) != base_hash, (
                f"evidence_hash did not react to an edit of evidence byte '{key}' "
                f"— has it been dropped from EVIDENCE_HASH_KEYS?"
            )

    def test_derived_label_edit_does_not_change_the_hash(self):
        # Editing a verdict/gate/regime label (a DERIVED quantity, never evidence)
        # MUST leave the hash identical: the hash pins only the raw evidence
        # bytes, so a repaint of the derived verdict is invisible here (the
        # verdict-replay cross-check catches that separately). Mutation: wrongly
        # adding any of these to EVIDENCE_HASH_KEYS turns its row red.
        base = _deposit()
        base_hash = evidence_hash(base)
        for key in DERIVED_LABEL_KEYS:
            mutated = {**base, key: "REPAINED_LABEL"}
            assert evidence_hash(mutated) == base_hash, (
                f"evidence_hash leaked derived label '{key}' into the evidence "
                f"bytes — hashing derived labels makes the integrity check circular."
            )

    def test_each_evidence_byte_key_is_in_the_hash_key_set(self):
        # Guards against a key being silently renamed/dropped from the leaf's
        # EVIDENCE_HASH_KEYS tuple (which would un-bind that evidence byte).
        for key in EVIDENCE_BYTE_KEYS:
            assert key in EVIDENCE_HASH_KEYS, (
                f"expected evidence byte '{key}' to be bound by EVIDENCE_HASH_KEYS"
            )

    def test_no_derived_label_is_in_the_hash_key_set(self):
        # The inverse guard: a derived label must never sneak into the key set.
        for key in DERIVED_LABEL_KEYS:
            assert key not in EVIDENCE_HASH_KEYS, (
                f"derived label '{key}' must not be bound by EVIDENCE_HASH_KEYS"
            )


# ---------------------------------------------------------------------------
# Idempotence + missing-key stability
# ---------------------------------------------------------------------------


class TestStampIdempotenceAndMissingKeys:
    def test_stamping_the_hash_into_the_deposit_is_idempotent(self):
        # ``evidence_hash`` is itself absent from EVIDENCE_HASH_KEYS, so writing
        # the computed hash back into the deposit and re-hashing yields the SAME
        # value (the hash never includes itself). Mutation: adding
        # ``"evidence_hash"`` to EVIDENCE_HASH_KEYS makes this non-idempotent.
        d = _deposit()
        once = evidence_hash(d)
        stamped = {**d, "evidence_hash": once}
        assert evidence_hash(stamped) == once

    def test_missing_evidence_keys_contribute_none_and_remain_stable(self):
        # A key missing from an older deposit contributes ``None`` (``.get(k)``);
        # the hash is stable and still well-formed. The committed fixtures have
        # between 18/27 and 27/27 keys present, so this is the live path.
        sparse = {"candidate_losses": [1.7]}  # only 1 of 27 keys present
        h = evidence_hash(sparse)
        assert re.fullmatch(r"[0-9a-f]{64}", h)
        assert evidence_hash(sparse) == h  # stable across calls


# ---------------------------------------------------------------------------
# THE grounded anchor: every committed §4 deposit re-derives to its stamp
# ---------------------------------------------------------------------------


class TestCommittedDepositsReproduceTheirStamp:
    """The strongest possible single-source-of-truth check.

    Each committed §4 deposit was stamped by the producer using THIS leaf; re-
    deriving the hash here must reproduce the stored ``evidence_hash`` byte-for-
    byte. ANY canonicalization drift — dropped ``sort_keys``, changed separators,
    a key added/removed in EVIDENCE_HASH_KEYS — shows up as RED on real bytes,
    on the exact artifacts the SHIP verdict rests on.
    """

    @pytest.mark.parametrize("fixture_name", STAMPED_DEPOSIT_FIXTURES)
    def test_rederived_hash_matches_committed_stamp(self, fixture_name):
        path = FIXTURE_DIR / fixture_name
        data = json.loads(path.read_text())
        stored = data["evidence_hash"]
        assert evidence_hash(data) == stored, (
            f"{fixture_name}: the leaf no longer reproduces the producer's stamped "
            f"evidence_hash — canonicalization or EVIDENCE_HASH_KEYS has drifted."
        )
