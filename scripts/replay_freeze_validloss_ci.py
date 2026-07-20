#!/usr/bin/env python
"""Replay recorded valid_loss samples through the GOAL §4 judge — no GPU, no model.

This is the Category-C (GPU) blocker **reduced to a concrete, GPU-free,
executable command** — the move the loop's feedback asked for in place of more
CPU scaffolding. Two roles:

1. **Verifiable recorded evidence.** ``scripts.run_freeze_validloss_ci`` trains
   the proxy and deposits *real* valid_loss samples (candidate + surrogate)
   plus its verdict into a JSON file (``--json --output``). A committed
   recording of that file — e.g. ``tests/fixtures/freeze_validloss_generalize_proxy.json``,
   a real RTX 3060 run — is evidence a reader can re-judge *anywhere* with no
   GPU and no model, and the recomputed verdict must match the one recorded at
   run time. That pins the evidence is faithful rather than painted on: the
   verdict is earned by the stored floats under the deterministic bootstrap,
   not asserted by the recording.

2. **The target-scale drop-in.** The 9B target run (private ``src.data``,
   Category-C) cannot run in this public mirror, but it deposits samples in the
   *same* schema. Dropping a ``proxy_scale: false`` sample file in and running
   this command yields the target-scale §4 verdict with **no code change** —
   the proxy label upgrades to target-scale purely by swapping the sample
   source. ``proxy_scale`` is read from the file and surfaced in the report, so
   a reader always sees which scale a replayed verdict is from.

   **The synthetic-provenance guard.** A recording may carry ``synthetic: true``
   to mark its floats as hand-authored plumbing (a constructed separation that
   exercises this branch), not a measurement. Such a recording is still judged —
   the verdict is faithfully recomputed from the stored floats — but it is never
   *presented* as a citable §4 result: the rendered note withholds the "this
   verdict IS the §4 target-scale result" claim a genuine 9B recording earns and
   instead says plainly "synthetic — do not cite". This converts the feedback's
   "every committed verdict is still proxy-scale and must not be cited as a §4
   target-scale result" warning from prose into a code-enforced contract, so a
   ``proxy_scale: false`` plumbing fixture can never be mistaken for a real 9B
   run. A genuine recording omits ``synthetic`` (or sets it ``false``).

   **The negative-control provenance guard.** A recording may carry
   ``negative_control: true`` to mark one of its arms as deliberately degraded
   on a non-order lever (an asymmetric training budget, ``candidate_total`` or
   ``surrogate_total``). Such a recording is a real measurement and is judged
   faithfully, but its verdict is an apparatus-sensitivity probe (proof the
   gate fires a genuine non-TIES label on a measured loss gap), never a §4
   order result — an additive note names the degraded arm and says so, and the
   ``citable_as_target_scale`` gate withholds the citation claim for it just as
   it does for a synthetic recording. Degrading the candidate fires the
   DOWNWARD label (UNDERSHOOTS); degrading the surrogate fires the UPWARD label
   (SURPASSES) — the symmetric completion, the only real-label direction that
   had never been recorded before this lever existed (the sole prior SURPASSES
   is the synthetic plumbing fixture).

   The same rule is also surfaced as a machine-readable
   ``citable_as_target_scale`` boolean in :func:`replay_to_json` (``True`` only
   for a genuine target-scale recording) so a downstream consumer does not have
   to infer citability from the raw ``proxy_scale`` / ``synthetic`` flags — the
   rendered claim and the machine gate enforce the same contract and cannot
   drift apart.

The replay re-runs *only* :func:`src.tg_lora.freeze_surrogate_ci.surrogate_valid_loss_ci`
— pure numpy over the stored floats, so it is deterministic and device-free.
It does not retrain, does not import torch, and does not assume a label: it
emits whatever the bootstrap CI on the stored samples says
(``SURPASSES`` / ``TIES`` / ``UNDERSHOOTS``).

Usage::

    # Re-judge the committed proxy recording (the make target's default).
    make freeze-replay
    python -m scripts.replay_freeze_validloss_ci \\
        tests/fixtures/freeze_validloss_generalize_proxy.json

    # Assert an expected verdict; exit nonzero on mismatch (CI / gate use).
    python -m scripts.replay_freeze_validloss_ci samples.json --expected TIES

    # The 9B target-scale drop-in: produce samples in the same schema, then:
    python -m scripts.replay_freeze_validloss_ci target_9b_samples.json

    # Override the materiality margin or the bootstrap RNG seed.
    python -m scripts.replay_freeze_validloss_ci samples.json \\
        --material-margin 0.05 --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.tg_lora.freeze_surrogate_ci import (
    SurrogateValidLossCI,
    format_surrogate_valid_loss_ci,
    surrogate_valid_loss_ci,
)
# The §4 citation-gate honesty primitives — the SAME per-axis logic the producer
# (scripts.run_freeze_validloss_ci_9b) uses to stamp
# ``citable_as_full_section4_verdict``, imported here so the GPU-free replay
# re-derives a deposit's budget / regime axes from the shared thresholds and
# rule rather than trusting the stored boolean (single source of truth,
# SYSTEM_CONSTITUTION Rule #3). See :func:`_producer_honesty_axes`.
from src.tg_lora.freeze_verdict_honesty import (
    REGIME_GENERALIZATION,
    classify_regime,
    is_reduced_budget,
)
# The §7 evidence-hash leaf (single source of truth, shared with the producer —
# see :func:`scripts.run_freeze_validloss_ci_9b._evidence_hash`). The replay
# re-derives the deposit's stamped ``evidence_hash`` from the SAME key list +
# canonicalization the producer stamped, so a hand-edited / externally-supplied
# deposit whose committed evidence bytes drifted from its stamp is surfaced loud
# at the GPU-free chokepoint rather than silently trusted.
from src.tg_lora.freeze_evidence_hash import evidence_hash
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

# The three verdicts the §4 judge can emit (imported so ``--expected`` choices
# are the exact labels the bootstrap layer returns, not a parallel vocabulary).
EXPECTED_VERDICTS = (SURPASSES, TIES, UNDERSHOOTS)


def load_samples(path: str | Path) -> dict[str, Any]:
    """Read a recorded-sample JSON and validate the §4 schema.

    Accepts the exact object :func:`scripts.run_freeze_validloss_ci.result_to_json`
    writes with ``--json --output`` (and any future target-scale run that
    deposits samples in the same schema). The two sample lists the judge needs
    are required and must be non-empty; every other field (``verdict``,
    ``base_seed``, ``proxy_scale``, ``synthetic``, ``task``, ...) is optional
    provenance the report surfaces when present.
    """
    p = Path(path)
    with p.open() as fh:
        data = json.load(fh)
    for key in ("candidate_losses", "surrogate_losses"):
        value = data.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"{p}: missing non-empty '{key}' — not a recorded-sample file "
                f"(expected the schema from `run_freeze_validloss_ci --json`)"
            )
    return data


def replay_samples(
    data: dict[str, Any],
    *,
    material_margin: float = 0.0,
    seed: int | None = None,
) -> SurrogateValidLossCI:
    """Re-run the §4 bootstrap judge on stored samples.

    Deterministic and GPU-free: the bootstrap is pure numpy over the stored
    floats. ``seed`` defaults to the file's recorded ``base_seed`` — the seed
    :func:`scripts.run_freeze_validloss_ci.run_ci` used for its own bootstrap —
    so a replay reproduces the recorded verdict bit-for-bit; pass an explicit
    ``seed`` to resample under a different RNG.
    """
    if seed is None:
        seed = int(data.get("base_seed", 0))
    return surrogate_valid_loss_ci(
        data["candidate_losses"],
        data["surrogate_losses"],
        seed=seed,
        material_margin=material_margin,
    )


def _resolve_negative_control_arm(data: dict[str, Any]) -> str | None:
    """Which arm a negative-control recording degraded, or ``None``.

    New recordings carry ``negative_control_arm`` directly. Legacy recordings
    (the committed UNDERSHOOTS fixture predates the field) are resolved by
    comparing each arm's recorded budget against ``total``: a budget that
    diverged was deliberately degraded. Returns ``'candidate'`` / ``'surrogate'``
    / ``'both'`` / ``None`` — so a surrogate-degraded recording (real SURPASSES)
    names the surrogate, not the candidate, keeping the provenance honest.
    """
    arm = data.get("negative_control_arm")
    if arm is not None:
        return arm
    total = data.get("total")
    if total is None:
        return None
    cand = data.get("candidate_total", total)
    surr = data.get("surrogate_total", total)
    cand_div = cand != total
    surr_div = surr != total
    if cand_div and surr_div:
        return "both"
    if cand_div:
        return "candidate"
    if surr_div:
        return "surrogate"
    return None


def _negative_control_active(data: dict[str, Any]) -> bool:
    """Effective negative-control status of a recording (GOAL §7 citation honesty).

    A recording is a negative control when EITHER the operator set the
    ``negative_control`` flag (explicit provenance) OR the gate can detect it
    from the artifact itself: an arm whose recorded training budget diverged
    (``candidate_total`` / ``surrogate_total`` ≠ ``total``) was degraded on a
    non-order lever. The producers (``run_freeze_validloss_ci`` /
    ``run_freeze_validloss_ci_9b``) set the flag from exactly this divergence,
    so the two agree on every recording a producer writes — but the citation
    gate must not trust the stored boolean *alone*: a hand-edited or externally-
    supplied deposit that carries a divergent budget yet an absent/stale flag
    would otherwise read as a citable §4 target-scale result, the operator-set
    label trusted over the machine-checkable artifact reality (the same class
    as the hand-typed ``best_valid_loss`` TASK-0152 closed and the swapped-arm
    guard ``5ed3380``). Budget divergence is authoritative at the gate whether
    the flag says so or not; ``citable_as_target_scale`` / the prose note both
    consult this so the machine gate and the human report cannot drift apart.
    """
    if bool(data.get("negative_control", False)):
        return True
    return _resolve_negative_control_arm(data) is not None


def _seq_len_refutes_full_context(data: dict[str, Any]) -> bool:
    """True when a recorded ``seq_len`` proves the run was NOT full-context.

    A positive numeric ``seq_len < 1024`` is reduced-context (the only config a
    12GB GPU fits, TASK-0152 lines 86-97); absent / non-numeric / ``>= 1024`` is
    not a refutation. ``bool`` is excluded so a stray ``True``/``False`` flag
    written into ``seq_len`` cannot masquerade as a count.
    """
    seq_len = data.get("seq_len")
    if isinstance(seq_len, bool) or not isinstance(seq_len, (int, float)):
        return False
    return 0 < seq_len < 1024


def _full_context_effective(data: dict[str, Any]) -> bool:
    """Effective full-context status of a recording (GOAL §4 citation honesty).

    ``full_context`` answers whether the run trained at the full §4
    ``seq_len=1024``. The producer (``form_freeze_validloss_deposit``) DERIVES the
    flag from ``seq_len`` (full_context = seq_len >= 1024), so the two agree on
    every recording a producer writes — but the citation gate must not trust the
    stored boolean *alone*: a hand-edited or externally-supplied deposit (the
    private-``src.data`` 9B drop-in path this harness replays) that carries a
    recorded ``seq_len < 1024`` yet an absent or stale ``full_context=True`` would
    otherwise read as the *full* §4 verdict, the operator-set label trusted over
    the machine-checkable artifact reality (the same class as the budget-divergence
    negative-control gate :func:`_negative_control_active` / commit ``9dff092``
    and the hand-typed ``best_valid_loss`` TASK-0152 closed). ``seq_len`` is
    authoritative at the gate whether the flag says so or not; both the prose note
    and ``citable_as_full_section4_verdict`` consult this so the machine gate and
    the human report cannot drift apart.
    """
    seq_len = data.get("seq_len")
    # ``seq_len`` is authoritative when it is a positive count: derive the gate's
    # answer from the artifact rather than the (possibly stale/absent) operator
    # flag. ``bool`` is excluded (a stray flag written into seq_len must not
    # count as a context length).
    if not isinstance(seq_len, bool) and isinstance(seq_len, (int, float)) and seq_len > 0:
        return seq_len >= 1024
    return bool(data.get("full_context", True))


def _carries_9b_honesty_schema(data: dict[str, Any]) -> bool:
    """True iff the deposit carries the producer's §4 honesty artifacts.

    A deposit written by :func:`scripts.run_freeze_validloss_ci_9b.result_to_json`
    stamps ``cfg_max_steps``, ``candidate_final_ce_train_loss_mean``, and
    ``regime`` (the four-axis gate's inputs); a proxy / synthetic / legacy
    recording does not. Only the former carries enough information to re-derive
    the producer's four axes, so the replay honors those axes (budget / thin /
    regime) only when this is True and otherwise falls back to the scale + context
    gate alone — the same artifact-when-present-else-flag discipline as
    :func:`_negative_control_active` / :func:`_full_context_effective`. This keeps
    the gate backward-compatible for the proxy / simulated recordings that never
    carried a training budget or a train-CE diagnostic, while closing the
    over-claim on every genuine 9B deposit (which always stamps all three).
    """
    return any(
        k in data
        for k in ("cfg_max_steps", "candidate_final_ce_train_loss_mean", "regime")
    )


def _producer_honesty_axes(
    data: dict[str, Any], ci: SurrogateValidLossCI
) -> tuple[bool, list[str]]:
    """Re-derive the producer's 4-conjunct gate's budget / thin / regime axes from
    a 9B deposit's artifacts (GOAL §4 / §7 citation honesty).

    The producer stamps ``citable_as_full_section4_verdict`` via its four-axis
    gate (:func:`src.tg_lora.freeze_verdict_honesty.full_section4_verdict_gate`);
    the replay must reproduce that answer from the deposit's stored artifacts,
    not trust the stored boolean alone. A hand-edited or externally-supplied 9B
    deposit that is reduced-budget (or thin, or non-generalization) yet carries a
    stale ``citable_as_full_section4_verdict=True`` would otherwise read as the
    COMPLETE §4 verdict — the operator-set label trusted over the
    machine-checkable artifact reality, the same class as
    :func:`_negative_control_active` (``9dff092``) and
    :func:`_full_context_effective` (``bbf6e68``). The three axes derived here
    mirror the producer verbatim via the shared leaf:

    * **budget** — :func:`is_reduced_budget(total_steps, cfg_max_steps)`: the
      deposit stamps both, so the gate never trusts a hardcoded ``reduced_budget``
      flag (a full-length run that reached ``max_steps`` clears the axis);
    * **thin** — ``ci.is_thin_evidence``: the bootstrap's own artifact-derived
      flag (an arm below ``MIN_SAMPLE_FOR_BOOTSTRAP`` seeds cannot anchor a §4
      significance statement);
    * **regime** — :func:`classify_regime(candidate_final_ce_train_loss_mean,
      ci.candidate_mean)`: the deposit stamps the candidate's mean train CE and
      the candidate valid_loss is the recomputed ``ci.candidate_mean`` (identical
      to the producer's, since the bootstrap is deterministic and the losses are
      the stored ones) — so a memorized / overfit / UNKNOWN-regime run is withheld
      even at full budget and full context.

    Returns ``(all_hold, failures)`` where ``failures`` names every axis that
    fails (``'budget'`` / ``'thin'`` / ``'regime=<label>'``) — the boolean drives
    the machine gate; the list drives the prose note so a reader sees *why* a
    target-scale, full-context deposit is still not citable as the complete §4
    verdict. Only meaningful when :func:`_carries_9b_honesty_schema` is True.
    """
    try:
        total_steps = int(data.get("total_steps"))
    except (TypeError, ValueError):
        # A partial 9B deposit missing total_steps reads as reduced — the
        # conservative call that never opens the gate on an unverifiable budget,
        # mirroring the producer (max_steps<=0 or absent -> reduced).
        total_steps = 0
    try:
        cfg_max_steps = int(data.get("cfg_max_steps"))
    except (TypeError, ValueError):
        cfg_max_steps = 0
    reduced = is_reduced_budget(total_steps, cfg_max_steps)
    thin = ci.is_thin_evidence
    regime = classify_regime(
        data.get("candidate_final_ce_train_loss_mean"), ci.candidate_mean
    )
    failures: list[str] = []
    if reduced:
        failures.append("budget")
    if thin:
        failures.append("thin")
    if regime != REGIME_GENERALIZATION:
        failures.append(f"regime={regime}")
    return (not failures), failures


def _citation_label_stale(
    data: dict[str, Any], effective_citable: bool
) -> bool:
    """True when a stored ``citable_as_full_section4_verdict`` boolean disagrees
    with the artifact-rederived effective verdict (GOAL §7 citation honesty).

    A 9B deposit the producer stamps always carries the stored boolean, and the
    replay re-derives the same verdict from the deposit's artifacts (see
    :func:`_producer_honesty_axes`); the two agree on every honest deposit. A
    disagreement means the stored label is stale or hand-edited — the gate treats
    the artifact-derived ``effective_citable`` as authoritative and surfaces the
    contradiction loud (the prose :func:`format_replay` ``CITATION_LABEL_STALE``
    note, mirroring ``BUDGET_DIVERGENCE_UNFLAGGED`` / ``FULL_CONTEXT_FLAG_REFUTED``)
    rather than silently trusting the label. Returns False when the deposit
    carries no stored boolean (a proxy / legacy recording) — there is nothing to
    cross-check, and the effective verdict still governs.
    """
    stored = data.get("citable_as_full_section4_verdict")
    if stored is None:
        return False
    return bool(stored) != effective_citable


def _target_scale_label_stale(data: dict[str, Any]) -> bool:
    """True when a stored ``citable_as_target_scale`` boolean disagrees with the
    artifact-rederived value ``not proxy_scale`` (GOAL §7 citation honesty).

    The producer stamps ``citable_as_target_scale`` as a deterministic function of
    the deposit's own ``proxy_scale`` field — ``not result["proxy_scale"]``
    (:func:`scripts.run_freeze_validloss_ci_9b.result_to_json`, the 1-term Level-1
    citation contract distinct from the replay's own stricter 3-term
    ``citable_as_target_scale`` output, which additionally withholds synthetic /
    negative-control recordings). It is the Level-1 citation boolean a reader checks
    first ("is this a genuine target-scale 9B recording?"), the prerequisite the
    Level-2 ``citable_as_full_section4_verdict`` gate composes on top of.

    :func:`_citation_label_stale` (``d734327``) binds the Level-2 boolean by
    re-deriving the full four-conjunct gate from the artifacts, but it does NOT
    transitively bind this Level-1 boolean: a deposit whose Level-2 label is
    honestly ``False`` (reduced context / thin / non-generalization) yet whose
    Level-1 ``citable_as_target_scale`` was hand-edited to over-claim (e.g. a proxy
    recording stamped ``citable_as_target_scale=True``) passes
    ``citation_label_stale`` — the two fields are independent, computed from
    different conjuncts. ``evidence_hash`` does not reach it either: the evidence
    hash deliberately covers raw measurements + run config and NEVER gate labels
    (see :data:`src.tg_lora.freeze_evidence_hash.EVIDENCE_HASH_KEYS`), and
    ``citable_as_target_scale`` is a gate label. So a hand-edited or
    externally-supplied deposit that flips the Level-1 citation boolean while
    leaving ``proxy_scale`` honest — over-claiming target scale on a proxy run, or
    under-claiming a genuine 9B run — passes every existing gate silently: the
    published deposit JSON carries a corrupt Level-1 citation claim no gate surfaces
    (the proof-of-need :class:`tests.test_replay_freeze_validloss_ci.
    TestTargetScaleLabelBinding` reproduces — a flipped deposit keeps
    ``faithful=True``, ``evidence_hash_stale=False``, ``citation_label_stale=False``
    and every ledger / sub-verdict gate green).

    This binds the stored boolean to ``not proxy_scale`` re-derived from the
    deposit's own ``proxy_scale`` field — the EXACT 1-term invariant the producer
    stamps, so it can never false-positive on an honest producer-stamped deposit
    (verified byte-identical across every committed 9B / proxy fixture) — and
    surfaces a disagreement loud (the prose :func:`format_replay`
    ``TARGET_SCALE_LABEL_STALE`` note, the Level-1 sibling of
    ``CITATION_LABEL_STALE``) rather than silently trusting the label. The threat
    model mirrors :func:`_citation_label_stale` exactly: it catches the simple
    hand-edit (flip the gate label without flipping its input), the same
    stored-boolean-trusted-over-artifact class; a coordinated flip of BOTH
    ``proxy_scale`` and the label is a distinct, harder attack outside this gate's
    scope (as it is outside ``_citation_label_stale``'s, whose ``proxy_scale`` /
    ``cfg_max_steps`` / train-CE inputs are likewise not in the evidence hash).
    Returns False when the deposit carries no stored boolean (a recording that
    predates the field) — there is nothing to cross-check, and the
    artifact-rederived value still governs (artifact-when-present-else-skip, same as
    :func:`_citation_label_stale`).
    """
    stored = data.get("citable_as_target_scale")
    if stored is None:
        return False
    proxy_scale = bool(data.get("proxy_scale", True))
    return bool(stored) != (not proxy_scale)


def _subverdict_rederived(
    data: dict[str, Any], *, losses_key: str
) -> str | None:
    """Re-derive a §4 sub-verdict (direction / baseline) from the deposit's stored
    per-arm losses with the producer's seed (GOAL §7 citation honesty).

    The producer (:func:`scripts.run_freeze_validloss_ci_9b.run_ci_9b`,
    ``run_freeze_validloss_ci_9b.py`` lines ~1715-1727) computes each sub-CI from
    ``candidate_losses`` against the arm's own losses with ``seed=base_seed``: the
    direction-isolation CI from ``control_losses`` (output-contiguous vs
    input-contiguous), the full-backprop baseline CI from ``baseline_losses``
    (progressive-freeze candidate vs no-freeze full-CE). The replay re-derives the
    SAME verdict from the stored floats — the same deterministic bootstrap over the
    same per-arm losses with the same seed — so a deposit whose stored
    ``direction.verdict`` / ``baseline.verdict`` label was hand-edited (or
    externally supplied stale) cannot pass the citation gate silently: the label is
    cross-checked against the artifact reality, the same "stored-label-trusted-over-
    artifact" class as :func:`_citation_label_stale` (``d734327``) and its budget /
    full-context siblings (``9dff092`` / ``bbf6e68``), now extended to the two §4
    condition-(a)/(b) sub-verdicts the producer stamps beside the main verdict.

    Returns the re-derived ``SURPASSES`` / ``TIES`` / ``UNDERSHOOTS`` label, or
    ``None`` when the deposit carries no per-arm losses for that slot (the arm did
    not run — ``control_losses`` / ``baseline_losses`` absent or empty) or no
    candidate losses — there is nothing to re-derive, so the staleness cross-check
    (:func:`_subverdict_stale`) reads False.
    """
    candidate_losses = data.get("candidate_losses")
    arm_losses = data.get(losses_key)
    if not isinstance(candidate_losses, list) or not candidate_losses:
        return None
    if not isinstance(arm_losses, list) or not arm_losses:
        return None
    seed = int(data.get("base_seed", 0))
    return surrogate_valid_loss_ci(
        candidate_losses, arm_losses, seed=seed
    ).significance_verdict


def _subverdict_stale(
    data: dict[str, Any], *, slot: str, losses_key: str
) -> bool:
    """True when a stored §4 sub-verdict label (direction / baseline) disagrees
    with the verdict re-derived from the deposit's per-arm losses (GOAL §7).

    The deposit stamps the sub-CI's verdict as ``direction.verdict`` /
    ``baseline.verdict`` (see :func:`scripts.run_freeze_validloss_ci_9b.
    _direction_ci_to_json` / :func:`_baseline_ci_to_json`); :func:`_subverdict_
    rederived` reproduces that verdict GPU-free from the stored
    ``candidate_losses`` plus the arm's losses with the producer's seed. They agree
    on every honest deposit; a disagreement means the stored label is stale or
    hand-edited, and the citation gate surfaces it loud (the prose
    :func:`format_replay` ``DIRECTION_VERDICT_STALE`` / ``BASELINE_VERDICT_STALE``
    note) rather than silently trusting the nested label — sibling of
    :func:`_citation_label_stale` (``d734327``). Returns False when the slot is
    absent (the arm did not run, stamped ``null``) or carries no verdict label to
    cross-check.
    """
    sub = data.get(slot)
    if not isinstance(sub, dict):
        return False
    stored = sub.get("verdict")
    if stored is None:
        return False
    rederived = _subverdict_rederived(data, losses_key=losses_key)
    if rederived is None:
        return False
    return stored != rederived


# The verdict-critical per-arm loss vectors a deposit stamps, keyed by the ledger
# role name. ``surrogate_valid_loss_ci`` draws the main verdict from the candidate
# and surrogate vectors; the control / baseline arms feed the §4 condition (a)/(b)
# sub-CIs. Each is reproducible from the committed ledger's per-arm ``valid_loss``,
# so the replay can bind the deposit's cited losses to the ledger's ground truth.
_LEDGER_ROLE_LOSS_KEYS = (
    ("candidate", "candidate_losses"),
    ("surrogate", "surrogate_losses"),
    ("control", "control_losses"),
    ("baseline", "baseline_losses"),
)


def _resolve_ledger_path(
    data: dict[str, Any], deposit_path: str | Path
) -> Path | None:
    """Resolve a deposit's committed ledger witness file, or ``None``.

    The producer stamps ``ledger_witness_path`` (a repo-root-relative path to the
    committed per-arm ledger JSONL) only on the citable full-§4 deposits; a proxy
    / synthetic / reduced-budget recording carries none, so this returns ``None``
    and the ledger cross-checks skip (the artifact-when-present-else-skip discipline
    of :func:`_carries_9b_honesty_schema`). When the path IS declared, resolve it
    CWD-relative first (how ``make freeze-replay`` invokes the replay from the repo
    root) and then, for a deposit passed by absolute path, against each ancestor of
    the deposit's own directory — so the binding fires whether the replay is run
    from the repo root or against an absolute deposit path. Returns ``None`` when
    no candidate file exists (the ledger is not committed alongside this deposit;
    the cross-checks skip rather than fail, the same posture as the other
    artifact-derived gates).
    """
    raw = data.get("ledger_witness_path")
    if not raw:
        return None
    witness = Path(raw)
    candidates: list[Path] = [Path.cwd() / witness]
    deposit_dir = Path(deposit_path).resolve().parent
    for ancestor in (deposit_dir, *deposit_dir.parents):
        candidates.append(ancestor / witness)
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_committed_ledger(
    data: dict[str, Any], deposit_path: str | Path
) -> list[dict[str, Any]] | None:
    """Parse a deposit's committed ledger witness, or ``None`` when absent.

    Returns the list of parsed JSONL records (header + per-arm rows) the producer
    wrote alongside the deposit. ``None`` when the deposit declares no
    ``ledger_witness_path``, the file is not committed, or a line fails to parse —
    the cross-checks skip in every case (never trust a half-readable ledger).
    """
    path = _resolve_ledger_path(data, deposit_path)
    if path is None:
        return None
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except (OSError, ValueError):
        return None
    return records


def _ledger_arm_losses_by_role(
    records: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Per-role per-seed ``valid_loss`` vectors, ordered by the arm ``index``.

    The ledger's ground-truth record: one ``valid_loss`` per arm per seed, grouped
    by role (candidate / surrogate / control / baseline) and sorted by the arm's
    ``index`` so the vector aligns with the deposit's ``<role>_losses`` list (which
    the producer fills arm-by-arm in the same seed order).
    """
    by_role: dict[str, list[tuple[int, float]]] = {}
    for rec in records:
        if rec.get("type") != "arm":
            continue
        role = rec.get("role")
        idx = rec.get("index")
        vl = rec.get("valid_loss")
        if role is None or idx is None or vl is None:
            continue
        by_role.setdefault(role, []).append((int(idx), float(vl)))
    return {role: [vl for _, vl in sorted(pairs)] for role, pairs in by_role.items()}


def _ledger_losses_stale(
    data: dict[str, Any], records: list[dict[str, Any]]
) -> list[str]:
    """Roles whose deposit-stamped losses diverge from the ledger's ground truth.

    The verdict (and the direction / baseline sub-verdicts) are computed from the
    deposit's ``candidate_losses`` / ``surrogate_losses`` / ``control_losses`` /
    ``baseline_losses``. The committed ledger records each arm's ground-truth
    ``valid_loss`` at harvest; a hand-edited or externally-supplied deposit that is
    internally self-consistent (``evidence_hash`` matches its own edited fields, the
    verdict matches its edited losses) yet diverges from the ledger would otherwise
    re-derive a corrupt-but-green verdict — the ledger is the primary record the
    deposit was harvested from, and the replay must bind the cited artifact to it
    (the deposit-vs-ledger sibling of the intra-deposit label-vs-losses guards
    :func:`_subverdict_stale` / ``371e934`` and :func:`_citation_label_stale` /
    ``d734327``). Returns the role names whose vectors differ in length or carry
    any element-wise mismatch; empty when every role matches (or the ledger was
    absent — see :func:`_load_committed_ledger`).
    """
    ledger_losses = _ledger_arm_losses_by_role(records)
    stale: list[str] = []
    # No committed ledger to bind against (the deposit declares none, or the file
    # was absent / unreadable) → there is no ground-truth to diverge from, so the
    # cross-check skips (returns clean), the same posture as every other
    # artifact-when-present-else-skip gate. Without this a deposit that simply
    # carries no ledger would read as fully stale on every role.
    if not ledger_losses:
        return stale
    for role, key in _LEDGER_ROLE_LOSS_KEYS:
        deposit_vec = data.get(key) or []
        ledger_vec = ledger_losses.get(role, [])
        # Compare whenever the role appears in EITHER source: empty==empty is a
        # match, but a deposit that dropped (or invented) an arm's losses diverges.
        if not deposit_vec and not ledger_vec:
            continue
        if len(deposit_vec) != len(ledger_vec):
            stale.append(role)
            continue
        if any(float(a) != float(b) for a, b in zip(deposit_vec, ledger_vec)):
            stale.append(role)
    return stale


def _ledger_witness_stale(
    data: dict[str, Any], records: list[dict[str, Any]]
) -> bool:
    """True when a deposit's stamped witness hash diverges from its ledger.

    The producer stamps ``ledger_witness_sha256`` (a canonical SHA-256 over the
    parsed ledger JSONL — ``json.dumps(records, sort_keys=True, separators=...)``,
    the same canonicalization ``TestCommittedLedgerWitness`` pins build-time) so the
    deposit binds to a specific committed ledger by content, not filename. The
    replay re-derives that hash from the ledger it just parsed and flags a deposit
    whose stamp no longer matches — a hand-edited stamp, a ledger rewritten under
    the same path, or a deposit pointed at the wrong ledger. Honored at the replay
    chokepoint; ``False`` when the deposit declares no stamp or no ledger is
    committed (the cross-check skips — same artifact-when-present discipline).
    """
    stored = data.get("ledger_witness_sha256")
    if not stored or not records:
        return False
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest() != stored


def _evidence_hash_stale(data: dict[str, Any]) -> bool:
    """True when a deposit's stamped ``evidence_hash`` diverges from the hash
    re-derived from its OWN evidence bytes (GOAL §7 citation honesty).

    The producer (:func:`scripts.run_freeze_validloss_ci_9b._evidence_hash`, via
    the shared :mod:`src.tg_lora.freeze_evidence_hash` leaf) stamps
    ``evidence_hash`` as a SHA-256 over the deposit's raw measurements +
    run-determining config (the EVIDENCE keys — losses, freeze orders,
    provenance, run config) — never the derived verdict/gate/regime labels — so
    a coordinated repaint that edits the floats, their CI bounds, the verdict
    label, and the per-arm provenance TOGETHER (which passes every DERIVED
    check), or any accidental byte drift, moves the stamp. The replay re-derives
    that hash from the SAME key list + canonicalization and flags a deposit
    whose stamp no longer matches — the deposit-vs-its-own-evidence sibling of
    :func:`_ledger_witness_stale` (which binds the deposit to an EXTERNAL ledger
    by content hash; this binds it to its OWN evidence block). This is the one
    cross-check that reaches the five committed 9B deposits that carry an
    ``evidence_hash`` but no committed ledger (direction / baseline / surrogate
    / generalization / heterogeneous_generalization) — the ledger binding skips
    them, so without this guard their committed bytes have no integrity binding
    at the torch-free chokepoint at all. Honored at the replay gate; ``False``
    when the deposit carries no stamp (a proxy / synthetic / legacy recording —
    the artifact-when-present-else-skip discipline, same as the ledger gate).
    """
    stored = data.get("evidence_hash")
    if not isinstance(stored, str) or not stored:
        return False
    return evidence_hash(data) != stored


# The main verdict's margin-invariant derived statistics a deposit stamps, keyed
# to the attribute on the recomputed ``ci`` (``SurrogateValidLossCI``). The
# producer (:func:`scripts.run_freeze_validloss_ci_9b.result_to_json`) writes
# each straight from the ``ci`` object it computed the verdict from (lines
# ~1995-2000), so on an honest deposit every one is bit-identical to the replay's
# re-derived ``ci`` — the bootstrap is deterministic over the same losses +
# ``base_seed``. ``is_material`` is DELIBERATELY EXCLUDED: it is the one statistic
# that depends on ``material_margin`` (``point_improvement >= material_margin``,
# see :func:`src.tg_lora.freeze_surrogate_ci.SurrogateValidLossCI.is_material`),
# the margin is NOT stamped in the deposit, so binding it strictly would
# false-positive on a producer run that used a non-zero margin. The nine keys
# here are all margin-invariant, so they bind cleanly across every committed
# deposit (verified byte-identical by ``TestCiStatsBinding``).
_CI_STAT_BINDINGS: tuple[tuple[str, str], ...] = (
    ("candidate_mean", "candidate_mean"),
    ("surrogate_mean", "surrogate_mean"),
    ("point_improvement", "point_improvement"),
    ("lower", "lower"),
    ("upper", "upper"),
    ("confidence", "confidence"),
    ("is_thin_evidence", "is_thin_evidence"),
    ("n_candidate", "n_candidate"),
    ("n_surrogate", "n_surrogate"),
)


def _ci_stats_stale(
    data: dict[str, Any], ci: SurrogateValidLossCI
) -> list[str]:
    """Stored derived §4 statistics that disagree with the replay's ``ci``.

    The verdict LABEL is already bound (:func:`replay_samples`' ``faithful``
    cross-check) and the raw EVIDENCE is already bound (:func:`_evidence_hash_stale`
    over the losses + run config). But the verdict's QUANTITATIVE backing — the
    candidate / surrogate means, the point improvement, the bootstrap CI bounds,
    the confidence, the sample sizes, the thin-evidence flag — is stamped in the
    deposit and cited as the §4 result's numbers, yet NEITHER of those gates
    reaches it: ``evidence_hash`` deliberately covers only raw measurements +
    run config (NEVER the derived statistics — see
    :func:`scripts.run_freeze_validloss_ci_9b._evidence_hash` / the producer's
    ``test_hash_is_over_evidence_not_derived_labels``), and ``faithful`` covers
    only the verdict label. A hand-edited or externally-supplied deposit can
    therefore store, say, ``point_improvement=0.05`` / ``lower=0.02`` /
    ``upper=0.08`` (a confidently-positive result) while the stored losses still
    re-derive the honest ``TIES`` verdict bit-for-bit — ``faithful=True``,
    ``evidence_hash_stale=False``, every label / ledger gate green — and the
    corrupt cited numbers pass silently (the proof-of-need the
    ``TestCiStatsBinding`` mutation test reproduces).

    This binds each margin-invariant statistic the deposit stamps to the value
    the replay's ``ci`` recomputed from the SAME losses + seed the producer used,
    surfacing a disagreement loud (the prose :func:`format_replay`
    ``CI_STATS_STALE`` note — the stored-derived-statistic-trusted-over-artifact
    path this guard closes, a distinct class from
    :func:`_citation_label_stale` / ``d734327`` which binds the citability
    BOOLEAN, and :func:`_evidence_hash_stale` / ``79577a5`` which binds the raw
    EVIDENCE hash). Returns the names of the disagreeing statistics; empty when
    every stamped statistic matches, or when the deposit stamps none of them (a
    proxy / legacy recording — the artifact-when-present-else-skip discipline,
    same as :func:`_carries_9b_honesty_schema`). ``is_material`` is intentionally
    not bound (margin-dependent; see :data:`_CI_STAT_BINDINGS`).
    """
    stale: list[str] = []
    for stored_key, ci_attr in _CI_STAT_BINDINGS:
        if stored_key not in data:
            continue
        if data[stored_key] != getattr(ci, ci_attr):
            stale.append(stored_key)
    return stale


def format_replay(path: str | Path, data: dict[str, Any], ci: SurrogateValidLossCI) -> str:
    """Human-readable replay block: scale, the §4 verdict, and faithfulness.

    Faithfulness compares the replayed verdict to the ``verdict`` the recording
    stored at run time: a match is the proof the stored floats earn the verdict
    under the deterministic bootstrap, a mismatch is a warning that the file was
    edited inconsistently. The scale line makes ``proxy_scale`` visible so a
    reader never cites a proxy verdict as target-scale (or vice versa).

    ``synthetic`` (read from the recording, default ``False``) is the
    provenance guard: a hand-authored plumbing recording is judged but never
    presented as a citable §4 result — its note withholds the "this verdict IS
    the §4 target-scale result" claim a genuine recording earns, enforcing the
    feedback's "do not cite as a target-scale result" warning in the rendered
    output rather than relying on prose alone.

    ``negative_control`` is the apparatus-provenance guard: a recording whose
    candidate (or surrogate) arm was degraded on a non-order lever (an
    asymmetric training budget). Such a recording IS a real measurement and is
    judged faithfully, but its verdict is a sensitivity probe, not a §4 order
    result — an additive note says so plainly, keeping a recorded UNDERSHOOTS
    from being misread as an order disadvantage. The status is derived from the
    artifact (budget divergence) as well as the flag (see
    :func:`_negative_control_active`), so a deposit that diverged but left the
    flag unset is still withheld from citation and flagged BUDGET_DIVERGENCE_UNFLAGGED.

    ``full_context`` is derived from ``seq_len`` as well as the flag (see
    :func:`_full_context_effective`): a recording whose ``seq_len < 1024`` is
    reduced-context whether the flag says so or not, so a hand-edited deposit that
    over-claims ``full_context=True`` is still withheld from the full-§4 citation
    and flagged FULL_CONTEXT_FLAG_REFUTED.
    """
    proxy_scale = bool(data.get("proxy_scale", True))
    synthetic = bool(data.get("synthetic", False))
    # The operator-set flag is tracked separately from the EFFECTIVE status: a
    # divergent budget makes a recording a negative control whether the flag was
    # recorded or not (``_negative_control_active``), so the scale-line value
    # reflects reality while the prose note distinguishes a flagged sensitivity
    # probe from an unflagged divergence the gate caught on its own.
    negative_control_flagged = bool(data.get("negative_control", False))
    negative_control = _negative_control_active(data)
    # The operator-set flag is tracked separately from the EFFECTIVE status: a
    # recorded ``seq_len < 1024`` makes a run reduced-context whether the flag was
    # recorded or not (``_full_context_effective``), so the scale-line value
    # reflects reality while a contradicted over-claim surfaces loud below.
    full_context_flag_explicit = (
        "full_context" in data and data.get("full_context") is True
    )
    full_context = _full_context_effective(data)
    full_context_flag_refuted = (
        full_context_flag_explicit and _seq_len_refutes_full_context(data)
    )
    seq_len = data.get("seq_len")
    # The producer's four-axis gate's budget / thin / regime axes — re-derived
    # from the deposit's artifacts when it carries the 9B honesty schema. Drives
    # the target-scale + full-context branch (withhold the strong "this verdict
    # IS" claim when an axis fails) and the trailing CITATION_LABEL_STALE note.
    # ``effective_full_section4`` mirrors the machine gate in :func:`replay_to_json`
    # exactly (same helpers, same order) so the prose claim and the JSON field
    # cannot drift — the invariant ``citable_as_full_section4_verdict is
    # ("this verdict IS" in prose)`` holds across every recording type.
    citable_as_target_scale_eff = (
        not proxy_scale and not synthetic and not negative_control
    )
    producer_axes_hold, producer_axis_failures = (
        _producer_honesty_axes(data, ci)
        if _carries_9b_honesty_schema(data) else (True, [])
    )
    effective_full_section4 = (
        citable_as_target_scale_eff and full_context and producer_axes_hold
    )
    scale = "PROXY" if proxy_scale else "TARGET"
    recorded = data.get("verdict")
    lines = [
        "freeze_replay — GOAL §4 judge on recorded samples (no GPU)",
        f"  source: {path}",
        f"  scale: {scale}_SCALE  (proxy_scale={proxy_scale}, "
        f"synthetic={synthetic}, negative_control={negative_control}, "
        f"full_context={full_context})  "
        f"task={data.get('task', '?')}  architecture={data.get('architecture', '?')}",
        "",
        format_surrogate_valid_loss_ci(ci),
    ]
    if recorded is not None:
        if ci.significance_verdict == recorded:
            lines.append(
                f"  faithfulness: replayed verdict MATCHES recording "
                f"({ci.significance_verdict})"
            )
        else:
            lines.append(
                f"  faithfulness: WARNING replayed {ci.significance_verdict} "
                f"!= recorded {recorded}"
            )
    # Scale-honesty note. ``synthetic`` takes precedence over both PROXY and
    # TARGET: a plumbing recording's floats are not a measurement, so the
    # verdict — though faithfully recomputed — is never citable as a §4 result
    # at any scale. A genuine recording reaches the PROXY/TARGET branches.
    if synthetic:
        lines.append(
            "  note: SYNTHETIC — the stored floats are hand-authored plumbing "
            f"(a constructed separation), not a {scale.lower()}-scale run. The "
            "verdict is faithfully recomputed from those floats but is "
            f"*synthetic* {scale.lower()}-scale evidence; do not cite it as a "
            "§4 result at any scale. A genuine run overwrites this file with "
            "real floats in the same schema and this note drops."
        )
    elif proxy_scale:
        lines.append(
            "  note: PROXY_SCALE — samples are from a 24-hidden proxy run, not "
            "the 9B target. The verdict is faithful to the recorded run but is "
            "a proxy-scale §4 result; do not cite it as target-scale."
        )
    else:
        # Target-scale. Withhold the strong "this verdict IS the §4 target-scale
        # result" claim unless full-context genuine: a negative control is a
        # sensitivity probe; a reduced-context probe IS genuine 9B but not the
        # full seq_len=1024 verdict. The cross-check pins these to the machine
        # gates so prose and JSON cannot drift.
        if negative_control:
            if negative_control_flagged:
                arm = _resolve_negative_control_arm(data)
                arm_clause = (
                    "the candidate arm was" if arm != "surrogate"
                    else "the surrogate arm was"
                )
                lines.append(
                    "  note: TARGET_SCALE — the recording is at target scale (a 9B "
                    "run); however a negative control was applied "
                    f"({arm_clause} deliberately degraded on a non-order lever), so "
                    "the verdict is a sensitivity probe recorded at target scale, "
                    "NOT a citable §4 order result."
                )
            else:
                # Divergent budget the operator flag did not assert: withhold the
                # strong claim from the artifact reality, not the stale label.
                lines.append(
                    "  note: TARGET_SCALE — the recording is at target scale, but "
                    "the arms' budgets diverged while the negative_control flag is "
                    "UNSET; the gate withholds the citable §4 order claim from the "
                    "divergence (see BUDGET_DIVERGENCE_UNFLAGGED), not the flag."
                )
        elif not full_context:
            seq_clause = f" at seq_len={seq_len}" if seq_len is not None else ""
            lines.append(
                "  note: REDUCED CONTEXT — samples are from a 9B run, so the "
                "recording IS at target scale; however it was trained"
                f"{seq_clause}, a reduced-context probe (the only config a 12GB "
                "GPU fits), NOT the full GOAL §4 seq_len=1024 verdict. Do not "
                "cite it as the full §4 verdict."
            )
        else:
            if producer_axes_hold:
                lines.append(
                    "  note: TARGET_SCALE — samples are from a 9B run; this verdict "
                    "IS the §4 target-scale result. The proxy verdict upgrades to "
                    "target-scale by swapping the sample source, with no code change."
                )
            else:
                # Target-scale + full-context, yet a producer honesty axis
                # (budget / thin / regime) failed: a genuine 9B run that is still
                # NOT citable as the COMPLETE §4 verdict. Withhold the strong
                # "this verdict IS" claim (mirrors the reduced-context branch) and
                # name the failing axis so the reader sees *why* the four-axis
                # gate stayed closed despite a target-scale, full-context recording.
                axes_clause = ", ".join(producer_axis_failures)
                lines.append(
                    "  note: TARGET_SCALE — samples are from a 9B run and the "
                    "recording is at target scale and full context; however a §4 "
                    f"producer honesty axis failed ({axes_clause}), so the verdict "
                    "is NOT citable as the COMPLETE §4 verdict. The verdict is "
                    "faithfully recomputed from the stored floats but is a "
                    "reduced-budget / thin / non-generalization probe, not the "
                    "full four-axis §4 result; do not cite it as the complete §4 "
                    "verdict. A full-budget, non-thin, generalizing run clears "
                    "these axes and this note drops."
                )
    # Negative-control provenance (additive — a negative control IS a real
    # measurement, just not of order). The degraded arm is named so the gap's
    # source is honest: a candidate-degraded recording's gap is from
    # undertraining the candidate (a recorded UNDERSHOOTS must not be misread
    # as "the output-first order is worse than random"), while a surrogate-
    # degraded recording's gap makes the candidate look better by construction
    # (a recorded SURPASSES that is a sensitivity probe, not an order win). The
    # verdict itself is recomputed from the stored floats; this note states only
    # why it is never a §4 order result, never which label it earned.
    if negative_control:
        arm = _resolve_negative_control_arm(data)
        if not negative_control_flagged:
            # Budget divergence the operator flag did not assert: the citation
            # gate withholds from the artifact reality, not a label that may be
            # stale, so the inconsistency the flag hid surfaces loud in the report
            # (the silent-corruption path this guard closes — the operator-set
            # label trusted over the machine-checkable budget reality).
            arm_word = (
                "the surrogate arm" if arm == "surrogate"
                else "both arms" if arm == "both"
                else "the candidate arm"
            )
            lines.append(
                "  note: BUDGET_DIVERGENCE_UNFLAGGED — the arms' recorded budgets "
                f"diverge ({arm_word} training total differs) but "
                f"negative_control is {data.get('negative_control', False)!r} "
                "(not asserted). The citation gate treats a divergent-budget "
                "recording as a negative control whether the flag is set or not: "
                "a degraded arm is a sensitivity probe, never a §4 order result. "
                "The stored verdict is faithfully recomputed but is NOT citable "
                "as a §4 order result. Set negative_control: true to record an "
                "intentional sensitivity probe, or correct the budgets to remove "
                "the divergence."
            )
        else:
            if arm == "surrogate":
                arm_phrase = "the surrogate arm was deliberately degraded"
            elif arm == "both":
                arm_phrase = "both arms were deliberately degraded asymmetrically"
            else:  # "candidate" (and the legacy default for pre-arm recordings)
                arm_phrase = "the candidate arm was deliberately degraded"
            lines.append(
                f"  note: NEGATIVE_CONTROL — {arm_phrase} (an asymmetric training "
                "budget, unrelated to freeze order) to inject a real quality gap. "
                "The verdict is faithfully recomputed from the stored floats but is "
                "an apparatus-sensitivity probe, NOT a §4 order result; do not read "
                "it as evidence for or against an output-first order advantage."
            )
    # Full-context over-claim guard (mirrors BUDGET_DIVERGENCE_UNFLAGGED): a
    # recording that explicitly asserts ``full_context=True`` but carries a
    # recorded ``seq_len < 1024`` over-claims the context — the citation gate
    # derives reduced-context from the artifact (seq_len) over the operator-set
    # label, and surfaces the contradiction loud rather than silently trusting
    # the stale flag (the stored-boolean-trusted-over-artifact path this guard
    # closes, sibling of :func:`_negative_control_active` / ``9dff092``).
    if full_context_flag_refuted:
        lines.append(
            "  note: FULL_CONTEXT_FLAG_REFUTED — the recording asserts "
            f"full_context=True but its recorded seq_len={seq_len} (< 1024) "
            "refutes it: the citation gate withholds the full-§4 claim from the "
            "artifact (seq_len), not the stale label. Correct seq_len to >= 1024 "
            "for a genuine full-context run, or set full_context: false to record "
            "the reduced-context probe honestly."
        )
    # §4 verdict-label staleness guard (mirrors FULL_CONTEXT_FLAG_REFUTED /
    # BUDGET_DIVERGENCE_UNFLAGGED): a deposit whose STORED
    # ``citable_as_full_section4_verdict`` boolean disagrees with the artifact-
    # rederived effective verdict has a stale or hand-edited label. The gate
    # derives the answer from the deposit's budget / thin / regime artifacts (the
    # producer's four-axis gate via the shared ``freeze_verdict_honesty`` leaf)
    # over the stored boolean, and surfaces the contradiction loud rather than
    # silently trusting the label — the stored-boolean-trusted-over-artifact path
    # this guard closes, sibling of :func:`_full_context_effective` / ``bbf6e68``
    # and :func:`_negative_control_active` / ``9dff092``.
    if _citation_label_stale(data, effective_full_section4):
        stored = data.get("citable_as_full_section4_verdict")
        lines.append(
            "  note: CITATION_LABEL_STALE — the recording's stored "
            f"citable_as_full_section4_verdict={stored!r} disagrees with the "
            f"effective verdict ({effective_full_section4}) re-derived from its "
            "artifacts (budget / thin / regime axes, the producer's four-axis "
            "gate). The citation gate trusts the artifact-derived answer over the "
            "stored label, so the deposit is treated as "
            f"{'citable' if effective_full_section4 else 'NOT citable'} as the "
            "complete §4 verdict. Re-stamp the deposit from a fresh producer run "
            "(or correct the stored boolean) so the label matches the artifacts."
        )
    # §4 Level-1 target-scale-label staleness guard (the Level-1 sibling of
    # CITATION_LABEL_STALE above): a deposit whose STORED
    # ``citable_as_target_scale`` boolean disagrees with ``not proxy_scale`` — the
    # exact 1-term contract the producer stamps
    # (:func:`scripts.run_freeze_validloss_ci_9b.result_to_json`,
    # ``citable_as_target_scale = not result["proxy_scale"]``) — has a stale or
    # hand-edited Level-1 citation label. ``citation_label_stale`` binds only the
    # Level-2 ``citable_as_full_section4_verdict`` boolean (which can be honestly
    # ``False`` for reasons unrelated to target-scale — reduced context / thin /
    # non-generalization — so it does not transitively bind this Level-1 boolean),
    # and ``evidence_hash`` deliberately never covers gate labels, so a hand-edited
    # deposit that flips the Level-1 boolean (over-claiming target scale on a proxy
    # recording, or under-claiming a genuine 9B run) while leaving ``proxy_scale``
    # honest passes every existing gate silently. The replay re-derives ``not
    # proxy_scale`` from the deposit's own ``proxy_scale`` field and surfaces the
    # contradiction loud rather than silently trusting the label — the
    # stored-boolean-trusted-over-artifact path this guard closes, the Level-1
    # sibling of :func:`_citation_label_stale` (``d734327``, the Level-2 boolean).
    if _target_scale_label_stale(data):
        stored = data.get("citable_as_target_scale")
        proxy_scale = bool(data.get("proxy_scale", True))
        lines.append(
            "  note: TARGET_SCALE_LABEL_STALE — the recording's stored "
            f"citable_as_target_scale={stored!r} disagrees with the value "
            f"(not proxy_scale={not proxy_scale!r}) re-derived from its own "
            f"proxy_scale={proxy_scale!r} (the producer's 1-term Level-1 citation "
            "contract). The citation gate trusts the artifact-derived answer over "
            "the stored label, so the deposit is treated as "
            f"{'target-scale' if (not proxy_scale) else 'proxy-scale'} for "
            "citation. Re-stamp the deposit from a fresh producer run (or correct "
            "the stored boolean) so it matches proxy_scale."
        )
    # §4 sub-verdict-label staleness guards (the §4 condition (a)/(b) siblings of
    # CITATION_LABEL_STALE): a deposit whose stored ``direction.verdict`` /
    # ``baseline.verdict`` label disagrees with the verdict re-derived from its
    # per-arm losses (``control_losses`` / ``baseline_losses``) has a stale or
    # hand-edited sub-verdict. The gate re-derives each from the deposit's floats
    # with the producer's seed (the same candidate-vs-arm bootstrap the producer
    # computed the sub-CI with) and surfaces the contradiction loud rather than
    # silently trusting the nested label — the stored-label-trusted-over-artifact
    # path this guard closes, sibling of :func:`_citation_label_stale`
    # (``d734327``). Fires only when the arm ran (the slot is present and
    # non-null); a deposit that did not run a control / baseline arm carries
    # ``null`` and has nothing to cross-check.
    for slot, losses_key, note in (
        ("direction", "control_losses", "DIRECTION_VERDICT_STALE"),
        ("baseline", "baseline_losses", "BASELINE_VERDICT_STALE"),
    ):
        if _subverdict_stale(data, slot=slot, losses_key=losses_key):
            stored = data[slot]["verdict"]
            rederived = _subverdict_rederived(data, losses_key=losses_key)
            lines.append(
                f"  note: {note} — the recording's stored {slot}.verdict="
                f"{stored!r} disagrees with the verdict ({rederived!r}) "
                f"re-derived from its stored {losses_key} (the same "
                "candidate-vs-arm two-sample bootstrap the producer computed "
                "the sub-CI with, under the deposit's base_seed). The replay "
                f"trusts the artifact-derived answer over the stored label, so "
                f"the {slot} sub-verdict is read as {rederived!r}. Re-stamp the "
                "deposit from a fresh producer run (or correct the stored label) "
                "so it matches the losses."
            )
    # §4 verdict-statistics staleness guard (sibling of CITATION_LABEL_STALE and
    # the sub-verdict guards above): the deposit stamps the main verdict's
    # QUANTITATIVE backing — candidate / surrogate means, point improvement,
    # bootstrap CI bounds, confidence, sample sizes, thin-evidence flag — straight
    # from the ``ci`` the producer computed the verdict from. ``faithful`` binds
    # only the verdict LABEL and ``evidence_hash`` binds only the raw EVIDENCE
    # bytes (losses + config, deliberately NOT the derived statistics), so a
    # hand-edited deposit that repaints the cited CI numbers while leaving the
    # honest losses — and thus the honest verdict label — untouched passes every
    # existing gate. The replay re-derives each margin-invariant statistic from
    # the SAME losses + seed and surfaces a disagreement loud rather than silently
    # trusting the stored numbers — the stored-derived-statistic-trusted-over-
    # artifact path this guard closes, sibling of :func:`_citation_label_stale`
    # (the citability BOOLEAN) and :func:`_evidence_hash_stale` (the raw EVIDENCE
    # hash). Skips statistics the deposit does not stamp (artifact-when-present-
    # else-skip); ``is_material`` is intentionally excluded (margin-dependent,
    # margin not stamped — see :data:`_CI_STAT_BINDINGS`).
    ci_stats_stale = _ci_stats_stale(data, ci)
    if ci_stats_stale:
        stats_clause = ", ".join(ci_stats_stale)
        lines.append(
            "  note: CI_STATS_STALE — the recording's stored §4 statistics "
            f"({stats_clause}) disagree with the values re-derived from its stored "
            "losses (the same deterministic bootstrap the producer computed them "
            "with, under the deposit's base_seed). The verdict label and the raw "
            "losses may still be honest (so `faithful` and `evidence_hash` stay "
            "clean) — only the cited quantitative backing (means, CI bounds, point "
            "improvement) was repainted. The replay trusts the artifact-derived "
            "numbers over the stored ones. Re-stamp the deposit from a fresh "
            "producer run so the statistics match the losses (GOAL §7)."
        )
    # Committed-ledger binding (the deposit-vs-ledger sibling of the intra-deposit
    # label-vs-losses guards above): the verdict is computed from the deposit's
    # per-arm losses, but the committed ledger (``ledger_witness_path``) is the
    # ground-truth record of each arm's valid_loss. A hand-edited deposit that
    # diverges from its ledger would otherwise re-derive a corrupt-but-green
    # verdict; the ledger is parsed and the deposit's losses + witness hash are
    # cross-checked against it, surfaced loud rather than silently trusted. Skipped
    # (no note) when the deposit carries no committed ledger — the same
    # artifact-when-present-else-skip discipline as the 9B honesty-schema gate.
    ledger_records = _load_committed_ledger(data, path)
    ledger_losses_stale = _ledger_losses_stale(data, ledger_records or [])
    if ledger_losses_stale:
        arms = ", ".join(ledger_losses_stale)
        lines.append(
            "  note: LEDGER_LOSSES_STALE — the recording's stored losses for "
            f"({arms}) disagree with the ground-truth per-arm valid_loss in its "
            f"committed ledger (ledger_witness_path="
            f"{data.get('ledger_witness_path')!r}). The verdict is re-derived from "
            "the deposit's losses; a deposit that diverges from its ledger "
            "re-derives a verdict the committed primary record does not support "
            "(corrupt-but-green, GOAL §7). Re-harvest the deposit from a fresh "
            "producer run so its losses match the ledger."
        )
    if _ledger_witness_stale(data, ledger_records or []):
        stored = data.get("ledger_witness_sha256")
        lines.append(
            "  note: LEDGER_WITNESS_STALE — the recording's stamped "
            f"ledger_witness_sha256={stored!r} disagrees with the SHA-256 "
            "re-derived from its committed ledger. The deposit no longer binds to "
            "the ledger by content (a rewritten ledger, a hand-edited stamp, or a "
            "deposit pointed at the wrong ledger). Re-stamp the witness from a "
            "fresh harvest so the hash matches the committed bytes."
        )
    # Evidence-hash staleness guard (the deposit-vs-its-own-evidence sibling of
    # LEDGER_WITNESS_STALE): the producer stamps ``evidence_hash`` over the
    # deposit's raw measurements + run-determining config, so any byte drift in
    # those evidence fields (a hand-edit, a formatter, a botched merge — or a
    # coordinated repaint that touches the floats + verdict + provenance
    # together and passes every DERIVED check) leaves the stamp stale. The
    # replay re-derives the hash from the SAME key list (the shared
    # ``freeze_evidence_hash`` leaf) and surfaces the contradiction loud rather
    # than silently trusting the stale stamp — the one integrity binding that
    # reaches the five committed 9B deposits that carry an ``evidence_hash`` but
    # no committed ledger. Fires only when the deposit carries a stamp; a
    # proxy / synthetic / legacy recording has none and the check skips.
    if _evidence_hash_stale(data):
        lines.append(
            "  note: EVIDENCE_HASH_STALE — the recording's stamped "
            f"evidence_hash={data.get('evidence_hash')!r} disagrees with the "
            "SHA-256 re-derived from its own evidence bytes (the raw measurements "
            "+ run-determining config — losses, freeze orders, provenance, run "
            "config — never the verdict/gate/regime labels). The committed "
            "evidence block has been altered without refreshing the stamp (a "
            "hand-edit, a formatter, a botched merge, or a coordinated repaint "
            "that passes every derived check), so the bytes the verdict rests on "
            "no longer match the pinned record (GOAL §7). Re-stamp the deposit "
            "from a fresh producer run so the hash matches the committed evidence."
        )
    return "\n".join(lines)


def replay_to_json(path: str | Path, data: dict[str, Any], ci: SurrogateValidLossCI) -> dict[str, Any]:
    """Machine-readable replay: the judge output plus the file's provenance.

    Two citation gates, each mirroring a prose claim :func:`format_replay`
    renders so the human claim and the machine gate cannot drift:

    * ``citable_as_target_scale`` — ``True`` for a genuine target-scale recording
      (``proxy_scale=False``, not ``synthetic``, not a negative control). A
      reduced-context (seq_len=256) 9B probe still qualifies — it IS a real 9B
      run, just not at seq_len=1024. Negative-control status is derived from
      budget divergence as well as the flag (:func:`_negative_control_active`),
      so an unflagged degraded arm cannot slip through and be cited as an order
      result.
    * ``citable_as_full_section4_verdict`` — ``True`` only when target-scale AND
      ``full_context`` AND, for a deposit carrying the 9B honesty schema, the
      producer's budget / thin / regime axes (re-derived from the artifacts via
      :func:`_producer_honesty_axes`, never the stored boolean). A reduced-budget,
      thin, or non-generalization 9B deposit is ``False`` even at full context:
      it must not be cited as the *complete* §4 verdict — the same four-axis gate
      the producer stamps, reproduced GPU-free. ``citation_label_stale`` is
      ``True`` when a stored ``citable_as_full_section4_verdict`` boolean disagrees
      with this artifact-rederived value (a stale / hand-edited label; the gate
      trusts the artifacts — see :func:`_citation_label_stale`).
    * ``direction_verdict_stale`` / ``baseline_verdict_stale`` — the §4
      condition-(a)/(b) sub-verdict cross-checks, siblings of
      ``citation_label_stale``: each stored ``direction.verdict`` /
      ``baseline.verdict`` label re-derived from ``control_losses`` /
      ``baseline_losses`` with the producer's seed (see
      :func:`_subverdict_rederived`). ``None`` / ``False`` when the arm did not
      run (the slot is null); ``True`` when the nested label disagrees with the
      losses — a stale sub-verdict the gate overrode from the artifacts.
    """
    proxy_scale = bool(data.get("proxy_scale", True))
    synthetic = bool(data.get("synthetic", False))
    # Derived from the artifact (budget divergence) as well as the flag, so a
    # deposit that diverged but left ``negative_control`` unset cannot be cited
    # as a §4 target-scale result (mirrors :func:`format_replay`; the gate trusts
    # the machine-checkable budget reality over the operator-set label — see
    # ``_negative_control_active``).
    negative_control = _negative_control_active(data)
    # Derived from the artifact (seq_len) as well as the flag, so a deposit that
    # trained at reduced context but left ``full_context`` unset (or stale-True)
    # cannot be cited as the *full* §4 verdict (mirrors :func:`format_replay`;
    # the gate trusts the machine-checkable context length over the operator-set
    # label — see ``_full_context_effective``, the full-context sibling of
    # ``_negative_control_active`` / ``9dff092``).
    full_context = _full_context_effective(data)
    citable_as_target_scale = (
        not proxy_scale and not synthetic and not negative_control
    )
    # The producer's four-axis gate's budget / thin / regime axes, re-derived from
    # the deposit's artifacts — honored ONLY when the deposit carries the 9B
    # honesty schema (cfg_max_steps / candidate_final_ce_train_loss_mean / regime).
    # A proxy / synthetic / legacy recording lacks those artifacts and uses the
    # scale + context gate alone (backward compatible); a genuine 9B deposit
    # always stamps all three, so this closes the over-claim on every one of them.
    # See :func:`_producer_honesty_axes` (single source of truth via the shared
    # ``freeze_verdict_honesty`` leaf — same thresholds/rule the producer stamps).
    producer_axes_hold, producer_axis_failures = (
        _producer_honesty_axes(data, ci)
        if _carries_9b_honesty_schema(data) else (True, [])
    )
    citable_as_full_section4_verdict = (
        citable_as_target_scale and full_context and producer_axes_hold
    )
    # Parse the committed ledger once and reuse it for both ledger-binding fields
    # below (same single read the prose path in :func:`format_replay` uses, so the
    # two JSON fields share one parse and cannot diverge if the file changes
    # between reads). ``[]`` when the deposit declares no committed ledger.
    ledger_records = _load_committed_ledger(data, path) or []
    return {
        "replayed_verdict": ci.significance_verdict,
        "recorded_verdict": data.get("verdict"),
        "faithful": (
            data.get("verdict") is None
            or ci.significance_verdict == data.get("verdict")
        ),
        "source": str(path),
        "proxy_scale": proxy_scale,
        "synthetic": synthetic,
        "negative_control": negative_control,
        # Reduced-context provenance: full §4 seq_len=1024? Derived from seq_len
        # when recorded, else the (backward-compatible) flag default True.
        "full_context": full_context,
        "seq_len": data.get("seq_len"),
        # Which arm a negative control degraded (from ``negative_control_arm`` or
        # inferred from the arms' budgets vs ``total``).
        "negative_control_arm": _resolve_negative_control_arm(data),
        # Citation gate level 1 (target-scale): genuine target-scale (mirrors the
        # prose "samples are from a 9B run").
        "citable_as_target_scale": citable_as_target_scale,
        # Citation gate level 2 (full §4 verdict): target-scale AND full_context
        # AND — for a 9B honesty-schema deposit — the producer's budget/thin/
        # regime axes re-derived from the artifacts (NOT the stored boolean). A
        # reduced-context probe is target-scale but NOT the full verdict
        # (TASK-0152 lines 86-97); a reduced-budget / thin / non-generalization
        # 9B deposit is NOT the complete verdict either — so a 12GB deposit or a
        # reduced run cannot be over-cited.
        "citable_as_full_section4_verdict": citable_as_full_section4_verdict,
        # Which producer honesty axes a 9B-schema deposit failed (empty when the
        # gate holds or the deposit carries no 9B schema). Surfaces *why* the full
        # §4 claim is withheld so a consumer never has to reverse-engineer it.
        "producer_honesty_axis_failures": producer_axis_failures,
        # Cross-check: does the deposit's stored full-§4 boolean agree with the
        # artifact-rederived value? True = stale/hand-edited label the gate
        # overrode from the artifacts (mirrors the prose CITATION_LABEL_STALE note;
        # absent for proxy/legacy recordings that carry no stored boolean).
        "citation_label_stale": _citation_label_stale(
            data, citable_as_full_section4_verdict
        ),
        # Cross-check (the Level-1 sibling of ``citation_label_stale``): does the
        # deposit's stored ``citable_as_target_scale`` boolean agree with ``not
        # proxy_scale`` — the producer's exact 1-term Level-1 citation contract? A
        # deposit whose Level-2 label is honestly ``False`` (reduced context / thin
        # / non-generalization) yet whose Level-1 boolean was hand-edited passes
        # ``citation_label_stale`` (independent fields) and ``evidence_hash`` (gate
        # labels are not evidence), so this is the one gate that flags a flipped
        # Level-1 citation claim (mirrors the prose TARGET_SCALE_LABEL_STALE note;
        # absent for recordings that carry no stored boolean).
        "target_scale_label_stale": _target_scale_label_stale(data),
        # §4 sub-verdict cross-checks (siblings of ``citation_label_stale``): each
        # stored ``direction.verdict`` / ``baseline.verdict`` label re-derived from
        # the deposit's per-arm losses with the producer's seed. A disagreement means
        # a stale/hand-edited sub-verdict label the gate overrode from the artifacts
        # (mirrors the prose DIRECTION_VERDICT_STALE / BASELINE_VERDICT_STALE notes;
        # the two cannot drift — same helper, same args). ``None`` / False when the
        # arm did not run (the slot is null and/or the per-arm losses are absent).
        "direction_verdict_rederived": _subverdict_rederived(
            data, losses_key="control_losses"
        ),
        "direction_verdict_stale": _subverdict_stale(
            data, slot="direction", losses_key="control_losses"
        ),
        "baseline_verdict_rederived": _subverdict_rederived(
            data, losses_key="baseline_losses"
        ),
        "baseline_verdict_stale": _subverdict_stale(
            data, slot="baseline", losses_key="baseline_losses"
        ),
        # Committed-ledger binding (the deposit-vs-ledger sibling of the
        # intra-deposit cross-checks above): the verdict is computed from the
        # deposit's per-arm losses, but the committed ledger is the ground-truth
        # per-arm record. ``ledger_records`` is parsed once before this dict and
        # reused by both fields (mirrors the prose LEDGER_LOSSES_STALE /
        # LEDGER_WITNESS_STALE notes; same helpers, same ledger, one read — the two
        # cannot drift). ``[]`` / False when the deposit carries no committed
        # ledger — the artifact-when-present discipline, same as the 9B honesty
        # schema gate.
        "ledger_losses_stale": _ledger_losses_stale(data, ledger_records),
        "ledger_witness_stale": _ledger_witness_stale(data, ledger_records),
        # Evidence-hash staleness (the deposit-vs-its-own-evidence sibling of the
        # ledger binding above): the producer stamps ``evidence_hash`` over the
        # deposit's raw measurements + run-determining config; the replay
        # re-derives it from the SAME key list (the shared ``freeze_evidence_hash``
        # leaf) and flags a deposit whose stamp no longer matches its committed
        # evidence bytes (mirrors the prose EVIDENCE_HASH_STALE note; same
        # helper — the two cannot drift). ``False`` when the deposit carries no
        # stamp (a proxy / synthetic / legacy recording — the
        # artifact-when-present discipline). This is the one integrity binding
        # that reaches the five committed 9B deposits that carry an
        # ``evidence_hash`` but no committed ledger.
        "evidence_hash_stale": _evidence_hash_stale(data),
        # §4 verdict-statistics staleness (sibling of the label / ledger / evidence
        # bindings above): the deposit stamps the main verdict's QUANTITATIVE
        # backing (means, CI bounds, point improvement, confidence, sample sizes,
        # thin-evidence flag) straight from the producer's ``ci``; the replay
        # re-derives each margin-invariant statistic from the SAME losses + seed
        # and flags a deposit whose stored numbers no longer match the losses it
        # cites (mirrors the prose CI_STATS_STALE note; same helper — the two
        # cannot drift). Empty when every stamped statistic matches or the deposit
        # stamps none (a proxy / legacy recording — the artifact-when-present
        # discipline). ``is_material`` is intentionally excluded (margin-dependent,
        # margin not stamped — see :data:`_CI_STAT_BINDINGS`). This is the one
        # cross-check that reaches the cited CI NUMBERS the verdict label and the
        # raw evidence hash both deliberately skip.
        "ci_stats_stale": _ci_stats_stale(data, ci),
        "candidate_mean": ci.candidate_mean,
        "surrogate_mean": ci.surrogate_mean,
        "point_improvement": ci.point_improvement,
        "lower": ci.lower,
        "upper": ci.upper,
        "confidence": ci.confidence,
        "is_material": ci.is_material,
        "is_thin_evidence": ci.is_thin_evidence,
        "n_candidate": ci.n_candidate,
        "n_surrogate": ci.n_surrogate,
        "seed": ci.seed,
        "material_margin": ci.material_margin,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_freeze_validloss_ci",
        description=(
            "Re-judge recorded valid_loss samples through the GOAL §4 "
            "surrogate_valid_loss_ci judge — no GPU, no model. Reads the JSON "
            "schema `run_freeze_validloss_ci --json --output` writes (and the "
            "same schema a future 9B target run deposits), so a committed "
            "recording is verifiable anywhere and a target-scale sample file "
            "drops straight in."
        ),
    )
    p.add_argument(
        "samples_file",
        help="path to a recorded-sample JSON (from run_freeze_validloss_ci --json).",
    )
    p.add_argument(
        "--material-margin", type=float, default=0.0,
        help="minimum point_improvement for is_material (§7 significance vs "
             "materiality separation); default 0.0.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="bootstrap RNG seed; default is the file's recorded base_seed "
             "(reproduces the recorded verdict bit-for-bit).",
    )
    p.add_argument(
        "--expected", default=None, choices=EXPECTED_VERDICTS,
        help="assert the replayed verdict; exit nonzero (2) on mismatch — for "
             "CI / gate use that pins a recording to its expected outcome.",
    )
    p.add_argument("--json", action="store_true", help="emit the replay as JSON to stdout.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_samples(args.samples_file)
    ci = replay_samples(data, material_margin=args.material_margin, seed=args.seed)
    if args.json:
        print(json.dumps(replay_to_json(args.samples_file, data, ci), indent=2))
    else:
        print(format_replay(args.samples_file, data, ci))
    if args.expected is not None and ci.significance_verdict != args.expected:
        print(
            f"replay: EXPECTED {args.expected} but got {ci.significance_verdict}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
