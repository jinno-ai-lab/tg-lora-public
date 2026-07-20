"""GOAL §7 reproducibility provenance — the deposit's frozen evidence hash.

This is a **torch-free leaf** imported by BOTH the producer
(:mod:`scripts.run_freeze_validloss_ci_9b`, which stamps ``evidence_hash`` onto a
deposit at harvest) and the GPU-free replay gate
(:mod:`scripts.replay_freeze_validloss_ci`, which re-derives that hash to
cross-check any committed deposit without torch). Keeping the key list and the
canonicalization in ONE place — here — means the producer's stamp and the
replay's cross-check cannot drift apart (single source of truth,
``SYSTEM_CONSTITUTION`` Rule #3); the replay re-derives the same hash from the
same keys the producer stamped, rather than trusting the stored hex.

The threat model (GOAL §7). A deposit's verdict, gate, and regime are all
DERIVED from the raw measurements; the verdict-replay checks guard that
derivation — a verdict painted on that disagrees with the stored floats fails
red. What those checks cannot catch is a COORDINATED repaint: editing the
committed floats, their CI bounds, the verdict label, and the per-arm provenance
TOGETHER so every derived check still passes. :func:`evidence_hash` freezes the
EVIDENCE bytes (the raw measurements a real run produced, never the self-declared
verdict/gate/regime labels) behind a content hash, so any such repaint — or any
accidental byte drift — becomes a visible, reviewable stale-stamp at the replay
chokepoint instead of silent source-of-truth erosion.

This does NOT certify a GPU produced the bytes (only a run log / a fresh
independent reproduction can); it certifies the COMMITTED bytes are the
immutable, auditable record every derived claim rests on, and that the
torch-free replay can verify that binding on its own.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# The fixed, ordered subset of EVIDENCE keys the hash freezes — the raw
# measurements and run-determining config, NEVER the derived verdict/gate/regime
# labels (hashing those would make the integrity check circular). ``evidence_hash``
# itself is absent, so stamping is idempotent; a key missing from an older deposit
# contributes ``None`` and the hash is stable across the stamp's own
# presence/absence.
EVIDENCE_HASH_KEYS = (
    # Raw held-out measurements + the freeze orders they were taken under.
    "candidate_losses", "surrogate_losses", "control_losses", "baseline_losses",
    "candidate_order", "control_order",
    # Per-arm provenance: which layers froze, how many params trained, and the
    # train-CE diagnostics that classify the regime — all run-determined.
    "candidate_provenance", "surrogate_provenance",
    "control_provenance", "baseline_provenance",
    # Run-determining config: identifies WHICH run produced these measurements.
    "model", "architecture", "lora_rank_pattern", "dataset",
    "total_steps", "warmup_steps", "depth", "spacing",
    "active_scope", "seq_len", "train_examples", "valid_examples",
    "n_candidate", "n_surrogate", "n_control", "n_baseline", "base_seed",
)


def evidence_hash(deposit: dict[str, Any]) -> str:
    """SHA-256 hex over the deposit's evidence bytes, for reproducibility pinning.

    Canonicalizes the fixed, ordered subset of EVIDENCE keys (see
    :data:`EVIDENCE_HASH_KEYS`) — the raw measurements and run-determining
    config, never the derived verdict/gate/regime labels — to a stable JSON
    encoding (sorted keys, compact separators) and returns the SHA-256 hex.
    ``evidence_hash`` is itself absent from :data:`EVIDENCE_HASH_KEYS`, so
    stamping it is idempotent: a key missing from an older deposit contributes
    ``None`` and the hash never includes itself.
    """
    payload = {k: deposit.get(k) for k in EVIDENCE_HASH_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
