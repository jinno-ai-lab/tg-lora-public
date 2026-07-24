"""§4 operator-decision surface — the ship / accept-null / pivot call, machine-verified.

The recurring AI-Hub feedback asks to "launch the 9B run through the now-robust
abort→rerun path and report its §4 TIES verdict." That ask is stale here for
three independently-sufficient reasons this command makes *machine-checkable*
rather than prose:

1. **The verdict arc is already COMPLETE.** Both citable full-budget §4 deposits
   — homogeneous (``freeze_validloss_ci_9b_full.json``) and heterogeneous
   (``freeze_validloss_ci_9b_full_heterogeneous.json``) — are committed, are
   ``citable_as_full_section4_verdict=True``, and re-derive to faithful **TIES**
   under the GPU-free bootstrap (:func:`replay_samples`). Re-firing reproduces a
   known TIES and adds zero information.
2. **Two distinct 9B-run paths must not be conflated.** The §4 *verdict* run
   (``make freeze-validloss-ci-9b-full`` → ``scripts.run_freeze_validloss_ci_9b``)
   trains on REAL PUBLIC Dolly (``databricks/databricks-dolly-15k``) via its own
   SFT adapter and imports **no** ``src.data`` — it is architecturally executable
   in this mirror (the committed TIES deposits are proof it was already fired
   here, on the 12 GB RTX 3060). The *recover.py* ``--rerun`` / ``train_tg_lora``
   corrective-re-run path is a DIFFERENT path: ``train_tg_lora`` imports
   ``src.data.build_seed_dataset`` at module scope (line 16), and ``src.data`` is
   **deliberately stripped** from this public mirror — so THAT path dies on
   import, by design. A prior version of this surface keyed ``run_executable_here``
   off the recover path's src.data import and so wrongly reported the verdict run
   as non-executable; this version probes the verdict worker directly.
3. **GPU is a transient runtime factor**, not an architectural one: seq1024
   full-budget fits the 12 GB RTX 3060 under the suffix-only config (both legs
   were fired on exactly that hardware).

Given (1)–(3), the executable operator decision *in this mirror* is **SHIP**
(recommended — adopt the citable relative-verdict TIES as the §4 result; the
verdict run already fired on public Dolly) or **ACCEPT-NULL** (TIES as the honest
null Progressive Freezing does not beat the random-order surrogate at full
budget). **PIVOT** — establishing absolute-loss comparability via the private
``src.data`` quality filter — is *not* a public-mirror action: it is a
private-repo action (``/home/jinno/tg-lora``) because ``src.data`` is stripped
here by design. Executing PIVOT by half-porting ``src.data`` into this mirror
would break the deliberate public/private boundary; the unblock step is to run it
in the private repo, then deposit + replay here.

This command consolidates the two deposits + the verdict-worker executability
probe into a single machine-verifiable snapshot so the operator (and the feedback
loop) can *check* the decision state instead of re-asking whether the run has
fired. It re-uses the existing replay primitive (:func:`replay_samples`) — it
does not re-implement the bootstrap, so it cannot mask producer drift, and it
adds no guard around the (already-locked) metric.

Usage::

    python scripts/section4_operator_decision.py            # snapshot; blocks (exit 3) until a call is landed
    python scripts/section4_operator_decision.py --json     # machine-readable snapshot
    python scripts/section4_operator_decision.py --land accept_null --basis "<why>"  # land the call → exit 0
    python -m scripts.section4_operator_decision --json

Exit contract: ``0`` = an operator decision is landed (done); ``3`` = the verdict
arc is complete but no call is landed yet (BLOCKING); ``2`` = arc incomplete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a bare script (``python scripts/section4_operator_decision.py``):
# a bare invocation puts ``scripts/`` — not the repo root — on sys.path, so make
# the repo root importable so ``scripts.*`` / ``src.*`` resolve without a
# PYTHONPATH wrapper (mirrors ``scripts/recover.py``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_freeze_validloss_ci import load_samples, replay_samples  # noqa: E402
from src.tg_lora.freeze_surrogate_gate import TIES  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402

# The two committed citable full-budget §4 deposits (reified verdict arc). Both
# are seq1024, ``proxy_scale=false``, ``citable_as_full_section4_verdict=True``.
HOMOGENEOUS_DEPOSIT = "tests/fixtures/freeze_validloss_ci_9b_full.json"
HETEROGENEOUS_DEPOSIT = "tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json"

# The ACTUAL entry point of the §4 verdict run. ``make freeze-validloss-ci-9b-full``
# subprocess-invokes this module directly (see the Makefile); it trains on REAL
# PUBLIC Dolly (``databricks/databricks-dolly-15k``) via its own SFT adapter and
# imports NO ``src.data`` — so the verdict run has no architectural src.data
# block (the committed TIES deposits are proof it was already fired here). This
# corrects the prior decision surface that keyed executability off
# ``train_tg_lora``'s src.data import — that is the recover.py ``--rerun`` path
# (a DIFFERENT path, see :data:`RECOVER_RERUN_ENTRYPOINT`), not the verdict run.
VERDICT_WORKER_MODULE = "scripts.run_freeze_validloss_ci_9b"

# The recover.py ``--rerun`` / ``train_tg_lora`` corrective-re-run path — the
# entrypoint that DOES import ``src.data.build_seed_dataset`` at module scope
# (line 16) and therefore IS architecturally non-executable in this mirror
# (src.data deliberately stripped). This is a real block, but on the recover
# path, NOT on the §4 verdict run above. ``present`` ⇒ the recover re-run is
# importable here; absent ⇒ every recover re-run dies on import, GPU or not.
RECOVER_RERUN_ENTRYPOINT = "src/data/build_seed_dataset.py"

# seq1024 full-budget 9B QLoRA VRAM floor under the suffix-only config both legs
# were fired with (RTX 3060, 12 GB). Informational only — a transient runtime
# factor, not an architectural gate (the verdict worker imports cleanly here).
SEQ1024_FULL_BUDGET_VRAM_FLOOR_MIB = 11_000

# --- The decision-landing surface ------------------------------------------------
#
# The feedback's single highest-leverage move for this (already-complete) arc is
# to convert the *documented* accept-null/ship/pivot decision into a *landed*
# one — either a default flip gated behind an explicit operator-approved commit,
# or a single blocking operator prompt — and then STOP surfacing variants until
# the operator's call lands. This implements both at once:
#
#   * ``--land <branch> --basis "<why>"`` writes a committed record (the
#     operator-approved commit/flag that lands the call); once landed, this
#     surface exits 0 and emits no further variants.
#   * until a record is landed, the surface BLOCKS (exit ``EXIT_AWAITING_DECISION``)
#     with ONE operator prompt naming all three branches + the exact land command.
#
# It does NOT re-derive the verdict, add a guard around the (locked) metric, or
# fabricate the efficacy/single-pass levers the feedback also names — those
# (``single-pass`` escape hatch, ``τ=+0.522`` ranking-triage, ``recursive`` /
# ``max_passes``) are absent from this public mirror (see TASK-0167..0177). The
# accept-null/ship/pivot DECISION itself is real and documented; only its landing
# was missing. This tool still does not make the call unilaterally — it blocks
# until the operator records it.

# The committed operator-decision record (repo-root-relative). Absent ⇒ no call
# has landed yet ⇒ the surface blocks. Present + valid ⇒ the call is landed.
LANDING_RECORD_REL = "section4_landed_decision.json"

# The three branches the operator may land. Mirror the ``branches`` dict below;
# landing a fourth is rejected so the decision space cannot drift silently.
VALID_LAND_BRANCHES = ("ship", "accept_null", "pivot")

# Exit contract: 0 = a decision is LANDED (done); 2 = arc incomplete (existing);
# 3 = arc complete but the operator call is un-landed (the blocking state); 4 =
# a ``--land`` was rejected (invalid branch / missing basis / incomplete arc).
EXIT_AWAITING_DECISION = 3
EXIT_LAND_INVALID = 4


def _probe_verdict_worker(python: str | None = None, *, runner=None) -> tuple[str, str]:
    """Honest ground-truth probe of the §4 verdict worker's importability.

    Returns ``(status, reason)`` where ``status`` is one of:

    - ``"executable"`` — the verdict worker imports cleanly (its public-Dolly
      data path + ``src.*`` deps are all present; no stripped-dep block). The
      verdict run is architecturally runnable here — only transient GPU/network
      factors remain (and both committed legs were already fired on exactly this
      hardware).
    - ``"transient_block"`` — the import failed on a transient runtime factor
      (e.g. ``torch`` absent in the probing interpreter), NOT on a stripped dep.
      Architecturally still executable; install the factor and it runs. Reported
      so a torch-free probe cannot masquerade as an architectural block.
    - ``"architectural_block"`` — the import failed on a stripped ``src.*`` dep.
      This is the only status that makes the verdict run architecturally
      non-executable.

    Spawned in a subprocess so this GPU-free script's own process never imports
    the worker's ``torch`` dep; ``runner`` is injectable so the unit tests mock
    it instead of spawning a real interpreter.
    """
    import subprocess

    run = runner or subprocess.run
    proc = run(
        [python or sys.executable, "-c", f"import {VERDICT_WORKER_MODULE}"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),  # so ``scripts``/``src`` resolve regardless of CWD
    )
    rc = getattr(proc, "returncode", 1)
    if rc == 0:
        return (
            "executable",
            "verdict worker imports cleanly (public Dolly; no src.data dep)",
        )
    err = (getattr(proc, "stderr", "") or "").strip()
    last = err.splitlines()[-1] if err else f"import failed (exit {rc}, no stderr)"
    if "No module named 'src" in err:
        return "architectural_block", f"stripped src.* dep: {last}"
    return "transient_block", f"transient runtime factor (torch/etc.): {last}"


def _assess_leg(label: str, deposit_rel: str, repo_root: Path) -> dict[str, Any]:
    """Re-derive one §4 leg's verdict from its committed deposit (GPU-free)."""
    path = repo_root / deposit_rel
    leg: dict[str, Any] = {
        "label": label,
        "deposit": deposit_rel,
        "present": path.exists(),
    }
    if not path.exists():
        leg["citable_as_full_section4_verdict"] = False
        leg["faithful"] = False
        leg["rederived_verdict"] = None
        leg["recorded_verdict"] = None
        return leg

    data = load_samples(path)
    ci = replay_samples(data)
    recorded = data.get("verdict")
    rederived = ci.significance_verdict
    leg.update(
        {
            "recorded_verdict": recorded,
            "rederived_verdict": rederived,
            "citable_as_full_section4_verdict": bool(
                data.get("citable_as_full_section4_verdict", False)
            ),
            # faithful = the recorded verdict is the one the stored floats earn
            # under the deterministic bootstrap (the verdict is not painted on).
            "faithful": rederived == recorded,
            "candidate_mean": data.get("candidate_mean"),
            "surrogate_mean": data.get("surrogate_mean"),
            "ci_lower": data.get("lower"),
            "ci_upper": data.get("upper"),
            "seq_len": data.get("seq_len"),
            "proxy_scale": data.get("proxy_scale"),
            "architecture": data.get("architecture"),
        }
    )
    return leg


def _landing_record_path(repo_root: Path | str | None) -> Path:
    """Resolve the operator-decision landing record under *repo_root*."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / LANDING_RECORD_REL


def _load_landed_decision(repo_root: Path | str | None) -> dict | None:
    """Return the landed operator-decision record, or ``None`` if none is landed.

    ``None`` (no record committed yet) is the BLOCKING state — the surface emits
    its single operator prompt and exits ``EXIT_AWAITING_DECISION``. A record that
    is absent, unreadable, or names a non-branch value is treated as un-landed
    rather than silently mis-read (a malformed record cannot fake a landed call).
    """
    path = _landing_record_path(repo_root)
    if not path.exists():
        return None
    try:
        record = load_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("branch") not in VALID_LAND_BRANCHES:
        return None
    return record


def _write_landed_decision(
    repo_root: Path | str | None,
    *,
    branch: str,
    basis: str,
    pivot_private_repo_only: bool,
) -> dict[str, Any]:
    """Atomically land (commit-ready) the operator's §4 decision record.

    Routed through :func:`src.utils.io.save_json` (the atomic JSON publish point)
    so a kill mid-write cannot leave a half-record that fakes a landed call. The
    record is the operator-approved commit/flag that converts the *documented*
    decision into a *landed* one.
    """
    from datetime import datetime, timezone

    record = {
        "branch": branch,
        "basis": basis,
        "landed": True,
        "pivot_private_repo_only": bool(pivot_private_repo_only),
        "deposits": [HOMOGENEOUS_DEPOSIT, HETEROGENEOUS_DEPOSIT],
        "landed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(record, _landing_record_path(repo_root))
    return record


def _blocking_prompt() -> str:
    """The SINGLE operator prompt emitted while the §4 call is un-landed.

    One prompt — not a stream of variants. Names all three branches and the exact
    ``--land`` command, and states it stops once the operator lands one. This is
    the conversion of the documented accept-null/ship/pivot decision into a
    blocking operator prompt (the feedback's highest-leverage move for this arc).
    """
    lines = [
        "=== ACTION REQUIRED: land your §4 operator decision ===",
        "",
        "The §4 verdict arc is COMPLETE (both legs citable faithful TIES) — the",
        "relative verdict is done and will not change on re-run. What remains is",
        "YOUR call on a TIES (null) result, which this tool does not make for you:",
        "",
        "  ship        — adopt the citable relative-verdict TIES as the §4 result",
        "  accept_null — record the TIES as the honest null (freeze-order gain is null)",
        "  pivot       — absolute-loss comparability (private-repo src.data action)",
        "",
        "Land exactly one (writes a committed record; this prompt then stops):",
        "",
        "  python scripts/section4_operator_decision.py \\",
        '      --land <ship|accept_null|pivot> --basis "<why>"',
        "",
        "Until you land one, this surface blocks (exit 3) and emits no further variants.",
    ]
    return "\n".join(lines)


def assess_section4_decision(
    *,
    repo_root: Path | str | None = None,
    src_data_present: bool | None = None,
    verdict_worker_status: str | None = None,
    worker_probe=None,
    landed: dict | None = None,
) -> dict[str, Any]:
    """Build the machine-verifiable §4 operator-decision snapshot.

    Parameters
    ----------
    repo_root:
        Repository root holding ``tests/fixtures/``. Defaults to this file's
        repo. Override (e.g. an empty ``tmp_path``) to exercise the
        arc-incomplete branch without touching the real deposits.
    src_data_present:
        Override for the ``src.data`` strip invariant (the recover.py ``--rerun``
        / ``train_tg_lora`` block and the PIVOT absolute-loss dependency).
        ``None`` (default) probes the live checkout.
    verdict_worker_status:
        Override for the §4 verdict worker's import probe
        (:func:`_probe_verdict_worker`). ``None`` (default) runs the probe; pass
        ``"executable"`` / ``"transient_block"`` / ``"architectural_block"`` to
        exercise the branch logic deterministically without spawning a probe.
    worker_probe:
        Injectable probe callable (returns ``(status, reason)``) so the unit
        tests mock the verdict-worker import instead of spawning an interpreter.
    landed:
        Override for the operator-decision landing record
        (:func:`_load_landed_decision`). ``None`` (default) reads it from disk
        under *repo_root*; pass a ``dict`` to inject a landed record (tests) or
        ``{}``-falsy via :func:`_load_landed_decision`. A landed record flips the
        surface from the blocking (awaiting) state to done — see
        :data:`EXIT_AWAITING_DECISION`.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT

    legs = [
        _assess_leg("homogeneous", HOMOGENEOUS_DEPOSIT, root),
        _assess_leg("heterogeneous", HETEROGENEOUS_DEPOSIT, root),
    ]

    arc_complete = all(
        leg["present"]
        and leg["citable_as_full_section4_verdict"]
        and leg["faithful"]
        and leg["rederived_verdict"] == TIES
        for leg in legs
    )

    # §4 VERDICT-run executability — keyed off the ACTUAL verdict worker
    # (public Dolly; no src.data), NOT the recover.py --rerun / train_tg_lora
    # path. A stripped src.* dep is the only architectural block; a missing
    # torch is a transient runtime factor that cannot masquerade as one. This
    # corrects the prior surface, which keyed executability off train_tg_lora's
    # src.data import and so wrongly reported the (already-fired, public-Dolly)
    # verdict run as architecturally non-executable.
    if verdict_worker_status is None:
        verdict_worker_status, worker_reason = (worker_probe or _probe_verdict_worker)()
    else:
        worker_reason = f"override: {verdict_worker_status}"
    run_executable_here = verdict_worker_status != "architectural_block"

    # The recover.py --rerun / train_tg_lora corrective-re-run path IS src.data-
    # blocked here — a real block, but on a DIFFERENT path than the verdict run.
    if src_data_present is None:
        src_data_present = (root / RECOVER_RERUN_ENTRYPOINT).exists()
    src_data_status = "present" if src_data_present else "stripped_deliberate"

    # PIVOT (absolute-loss comparability vs the private-repo baseline via the
    # src.data quality filter) is public-doable ONLY when src.data is present.
    pivot_public_doable = bool(src_data_present)

    branches = {
        # SHIP: adopt the citable relative-verdict TIES as the §4 result. The
        # evidence base is the committed arc — executable the moment arc_complete.
        "ship": {
            "executable_here": arc_complete,
            "summary": "adopt the citable full-budget relative-verdict TIES as the §4 result",
        },
        # ACCEPT-NULL: TIES is the honest null — Progressive Freezing does not
        # beat the random-order surrogate at full budget (GOAL §0 quality-parity
        # × 14–35% backward-pass cost reduction holds; the freeze-order gain does
        # not). Always a live, mirror-side reading of the same evidence.
        "accept_null": {
            "executable_here": arc_complete,
            "summary": "TIES as the honest null (freeze-order gain is null at full budget)",
        },
        # PIVOT: establish absolute-loss comparability via the src.data quality
        # filter. Needs src.data → deliberately stripped here → private-repo only.
        "pivot": {
            "executable_here": pivot_public_doable,
            "private_repo_only": not pivot_public_doable,
            "summary": (
                "absolute-loss comparability via src.data — a private-repo action "
                "(src.data is deliberately stripped from this public mirror)"
                if not pivot_public_doable
                else "absolute-loss comparability via the now-present src.data pipeline"
            ),
        },
    }

    # The recommendation keys arc-completeness FIRST: a complete arc means the
    # verdict is already DONE (the full-budget run fired on public Dolly and
    # produced these deposits), so the operator ships the TIES regardless of
    # whether the run could be re-fired (re-firing reproduces TIES — it adds no
    # information). Executability only matters when the arc is INCOMPLETE, to
    # decide fire-it-here vs blocked. (The prior logic let executability flip a
    # complete arc to FIRE_OR_EXTEND, which was wrong for a done verdict.)
    if arc_complete:
        recommendation = "SHIP"
        rationale = (
            "both §4 legs are citable faithful TIES — the verdict arc is COMPLETE "
            "(the full-budget run already fired on public Dolly and produced these "
            "deposits); adopt the TIES as the §4 result. Re-firing reproduces TIES "
            "and adds no information; absolute-loss (vs the private-repo baseline) "
            "remains the only open axis and is a private-repo action."
        )
    elif run_executable_here:
        recommendation = "FIRE_OR_EXTEND"
        rationale = (
            "the §4 verdict arc is incomplete but the verdict worker is executable "
            "in this checkout (public Dolly; no src.data block) — fire "
            "`make freeze-validloss-ci-9b-full[-heterogeneous]` to produce the "
            "missing deposit"
            + (
                " (a transient factor such as GPU/torch is currently unavailable)"
                if verdict_worker_status == "transient_block"
                else ""
            )
        )
    else:
        recommendation = "INCOMPLETE_ARC"
        rationale = (
            "one or both full-budget §4 deposits are missing, non-citable, or not "
            "faithful-TIES, AND the verdict worker is not executable in this "
            f"checkout ({worker_reason})"
        )

    unblock_step = (
        "The §4 RELATIVE verdict arc is COMPLETE (both legs citable faithful TIES) "
        "and the verdict run already fired on public Dolly — no re-fire is needed "
        "(it reproduces TIES). The sole remaining axis is ABSOLUTE-LOSS "
        "comparability vs the private-repo baseline, which needs the private "
        "src.data quality filter: src.data is deliberately stripped from this "
        "public mirror (see the note in scripts/prepare_data.py and the "
        "interface-without-implementation in tests/test_filter_dataset.py + "
        "tests/test_dedup.py). Execute PIVOT in the private repo "
        "(/home/jinno/tg-lora): port/refresh src/data/filter_dataset.py quality "
        "filtering → re-fire the same corpus → form_freeze_validloss_deposit + "
        "freeze-replay → deposit here. Do NOT half-port src.data into this mirror "
        "(it breaks the deliberate boundary and cannot run without the full "
        "private pipeline + corpus). NOTE: the recover.py `--rerun` / "
        "train_tg_lora corrective-re-run path is a SEPARATE, src.data-blocked "
        "path — that block does not apply to the verdict worker."
        if arc_complete and not pivot_public_doable
        else (
            "src.data is present — fire the run directly through "
            "`make freeze-validloss-ci-9b-full` (or `make recover ... --rerun`)."
            if pivot_public_doable
            else "Produce the missing §4 deposit by firing "
            "`make freeze-validloss-ci-9b-full[-heterogeneous]` (the verdict "
            "worker runs on public Dolly; no src.data block)."
        )
    )

    # The operator-decision landing record. ``None`` (default) reads it from disk;
    # a landed record converts the surface from blocking (awaiting) to done. The
    # tool does not make the call — it reports whether the operator has landed one.
    if landed is None:
        landed = _load_landed_decision(root)

    return {
        "arc_complete": arc_complete,
        "landed_decision": landed,
        # blocking state: the verdict arc is DONE but no operator call is recorded.
        "awaiting_operator_decision": bool(arc_complete and landed is None),
        "legs": legs,
        "run_executable_here": run_executable_here,
        "verdict_worker_status": verdict_worker_status,
        "verdict_worker_reason": worker_reason,
        "src_data_status": src_data_status,
        "recover_rerun_blocked_by_src_data": not src_data_present,
        "seq1024_full_budget_vram_floor_mib": SEQ1024_FULL_BUDGET_VRAM_FLOOR_MIB,
        "branches": branches,
        "recommendation": recommendation,
        "rationale": rationale,
        "unblock_step": unblock_step,
    }


def _land_decision(
    repo_root: Path | str | None, branch: str | None, basis: str | None
) -> tuple[int, str]:
    """Land (write the committed record for) the operator's §4 decision.

    Returns ``(exit_code, message)``. Validates the branch is one of the three,
    that a non-empty ``--basis`` records the operator's reasoning, and that the
    verdict arc is complete (you cannot land a call on an incomplete verdict).
    The arc check is deposit-derived (GPU-free); the worker probe is overridden
    so ``--land`` never depends on a torch interpreter being present. On success
    the record is written atomically (:func:`_write_landed_decision`) and the
    surface stops blocking.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    if branch not in VALID_LAND_BRANCHES:
        return EXIT_LAND_INVALID, (
            f"invalid --land branch {branch!r}; choose one of "
            f"{', '.join(VALID_LAND_BRANCHES)}."
        )
    if not (basis and basis.strip()):
        return (
            EXIT_LAND_INVALID,
            "--land requires a non-empty --basis explaining the call.",
        )
    # arc_complete is deposit-derived (GPU-free replay); override the worker probe
    # so --land is torch-free — the probe never gates whether a call may land.
    snap = assess_section4_decision(repo_root=root, verdict_worker_status="executable")
    if not snap["arc_complete"]:
        return EXIT_LAND_INVALID, (
            "cannot land a §4 decision: the verdict arc is incomplete (one or "
            "both deposits are missing / non-citable / not faithful-TIES)."
        )
    pivot_private_repo_only = branch == "pivot" and not snap["branches"]["pivot"].get(
        "executable_here", False
    )
    _write_landed_decision(
        root,
        branch=branch,
        basis=basis.strip(),
        pivot_private_repo_only=pivot_private_repo_only,
    )
    return 0, (
        f"LANDED §4 operator decision: {branch}.\n"
        f"  basis: {basis.strip()}\n"
        f"  record: {_landing_record_path(root)}\n"
        "  This call is landed and binding; `section4_operator_decision` now exits "
        "0 and emits no further variants."
    )


def format_decision(snapshot: dict[str, Any]) -> str:
    """Human-readable rendering of the §4 operator-decision snapshot."""
    lines: list[str] = []
    lines.append("=== §4 operator-decision snapshot ===")
    lines.append("")
    lines.append("Verdict arc (re-derived GPU-free from committed deposits):")
    for leg in snapshot["legs"]:
        if not leg["present"]:
            lines.append(f"  [{leg['label']}] MISSING ({leg['deposit']})")
            continue
        verdict = leg["rederived_verdict"]
        citable = (
            "citable" if leg["citable_as_full_section4_verdict"] else "NON-citable"
        )
        faithful = "faithful" if leg["faithful"] else "STALE-vs-rederived"
        lines.append(
            f"  [{leg['label']}] {verdict} · {citable} · {faithful} · "
            f"cand {leg['candidate_mean']:.4f} vs surr {leg['surrogate_mean']:.4f} "
            f"· CI[{leg['ci_lower']:+.4f}, {leg['ci_upper']:+.4f}] · seq{leg['seq_len']}"
        )
    lines.append(f"  arc_complete = {snapshot['arc_complete']}")
    lines.append("")
    lines.append(
        f"Verdict run executable in this mirror: {snapshot['run_executable_here']} "
        f"({snapshot['verdict_worker_status']})"
    )
    lines.append(f"  src.data status: {snapshot['src_data_status']}")
    lines.append(
        f"  recover.py --rerun / train_tg_lora path blocked by src.data strip: "
        f"{snapshot['recover_rerun_blocked_by_src_data']}"
    )
    lines.append(
        f"  (seq1024 full-budget VRAM floor ~{snapshot['seq1024_full_budget_vram_floor_mib']} MiB "
        "— transient runtime factor, not an architectural gate)"
    )
    lines.append("")
    lines.append("Decision branches (executable_here):")
    for name, branch in snapshot["branches"].items():
        tag = (
            "yes"
            if branch["executable_here"]
            else ("no — private-repo only" if branch.get("private_repo_only") else "no")
        )
        lines.append(f"  {name}: {tag} — {branch['summary']}")
    lines.append("")
    lines.append(f"RECOMMENDATION: {snapshot['recommendation']}")
    lines.append(f"  {snapshot['rationale']}")
    lines.append("")
    lines.append("Unblock step:")
    for sentence in snapshot["unblock_step"].split(". "):
        if sentence.strip():
            lines.append(f"  {sentence.strip()}.")
    lines.append("")
    landed = snapshot.get("landed_decision")
    if landed:
        lines.append("LANDED OPERATOR DECISION (binding — recorded by the operator):")
        lines.append(f"  branch: {landed['branch']}")
        if landed.get("basis"):
            lines.append(f"  basis: {landed['basis']}")
        if landed.get("pivot_private_repo_only"):
            lines.append("  note: pivot recorded as a private-repo-only action")
        lines.append(
            "  This call is landed; this surface exits 0 and emits no further variants."
        )
    elif snapshot.get("awaiting_operator_decision"):
        # The single blocking operator prompt — emitted only while the call is
        # un-landed, never as a stream of variants.
        lines.append(_blocking_prompt())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Machine-verifiable §4 operator-decision snapshot "
            "(ship / accept-null / pivot). GPU- and src.data-free."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the snapshot as JSON (default: human-readable).",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repository root holding tests/fixtures/ (default: this script's repo).",
    )
    parser.add_argument(
        "--land",
        metavar="BRANCH",
        default=None,
        help=(
            "Land (commit-ready) your §4 operator decision: one of "
            "ship/accept_null/pivot. Requires --basis. Writes a committed record "
            "so the decision surface stops blocking (exit 3 → 0)."
        ),
    )
    parser.add_argument(
        "--basis",
        default=None,
        help="Reason recorded alongside --land (required with --land).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.land is not None:
        rc, message = _land_decision(args.repo_root, args.land, args.basis)
        print(message)
        return rc
    snapshot = assess_section4_decision(repo_root=args.repo_root)
    if args.json:
        print(json.dumps(snapshot, indent=2, default=str))
    else:
        print(format_decision(snapshot))
    # exit contract: 0 = an operator decision is LANDED (done); 3 = the verdict
    # arc is complete but no call is landed yet (BLOCKING — emit the single
    # operator prompt, no further variants, until the operator lands one);
    # 2 = arc incomplete. Extends replay's --expected contract with the
    # awaiting-decision state.
    if snapshot["landed_decision"] is not None:
        return 0
    return EXIT_AWAITING_DECISION if snapshot["arc_complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
