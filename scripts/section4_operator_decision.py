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
2. **The 9B run is architecturally non-executable in this public mirror.**
   ``src.training.train_tg_lora`` imports ``src.data.build_seed_dataset`` at
   module scope (line 16), and ``src.data`` is **deliberately stripped** from
   this public mirror — ``scripts/prepare_data.py`` documents it ("the private
   data pipeline that is stripped from this public mirror") and the interface
   survives the strip (``tests/test_filter_dataset.py`` / ``tests/test_dedup.py``
   exist without their implementation). So a launch here dies on import, by
   design, regardless of GPU.
3. **GPU is a transient runtime factor**, not an architectural one: seq1024
   full-budget fits the 12 GB RTX 3060 under the suffix-only config (both legs
   were fired on exactly that hardware). The permanent blocker is (2), not VRAM.

Given (1)–(3), the executable operator decision *in this mirror* is **SHIP**
(recommended — adopt the citable relative-verdict TIES as the §4 result) or
**ACCEPT-NULL** (TIES as the honest null Progressive Freezing does not beat the
random-order surrogate at full budget). **PIVOT** — establishing absolute-loss
comparability via the private ``src.data`` quality filter — is *not* a
public-mirror action: it is a private-repo action (``/home/jinno/tg-lora``)
because ``src.data`` is stripped here by design. Executing PIVOT by half-porting
``src.data`` into this mirror would break the deliberate public/private boundary;
the unblock step is to run it in the private repo, then deposit + replay here.

This command consolidates the two deposits + the executability invariant into a
single machine-verifiable snapshot so the operator (and the feedback loop) can
*check* the decision state instead of re-asking whether the run has fired. It
re-uses the existing replay primitive (:func:`replay_samples`) — it does not
re-implement the bootstrap, so it cannot mask producer drift, and it adds no
guard around the (already-locked) metric.

Usage::

    python scripts/section4_operator_decision.py            # human-readable snapshot
    python scripts/section4_operator_decision.py --json     # machine-readable
    python -m scripts.section4_operator_decision --json
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

# The two committed citable full-budget §4 deposits (reified verdict arc). Both
# are seq1024, ``proxy_scale=false``, ``citable_as_full_section4_verdict=True``.
HOMOGENEOUS_DEPOSIT = "tests/fixtures/freeze_validloss_ci_9b_full.json"
HETEROGENEOUS_DEPOSIT = "tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json"

# ``train_tg_lora``'s module-scope data-pipeline import (line 16). Present → the
# entrypoint is importable; absent (the deliberate public-mirror strip) → every
# launch dies on import before any step trains, GPU or not.
TRAIN_ENTRYPOINT_DATA_IMPORT = "src/data/build_seed_dataset.py"

# seq1024 full-budget 9B QLoRA VRAM floor under the suffix-only config both legs
# were fired with (RTX 3060, 12 GB). Informational only — gated on the permanent
# src.data blocker, not on this transient runtime factor.
SEQ1024_FULL_BUDGET_VRAM_FLOOR_MIB = 11_000


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


def assess_section4_decision(
    *,
    repo_root: Path | str | None = None,
    src_data_present: bool | None = None,
) -> dict[str, Any]:
    """Build the machine-verifiable §4 operator-decision snapshot.

    Parameters
    ----------
    repo_root:
        Repository root holding ``tests/fixtures/``. Defaults to this file's
        repo. Override (e.g. an empty ``tmp_path``) to exercise the
        arc-incomplete branch without touching the real deposits.
    src_data_present:
        Override for the ``src.data`` strip invariant. ``None`` (default) probes
        the live checkout; pass ``True``/``False`` to test the branch logic
        deterministically (e.g. the PIVOT-becomes-public-doable mutation).
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

    # Architectural executability invariant — the permanent blocker.
    if src_data_present is None:
        src_data_present = (root / TRAIN_ENTRYPOINT_DATA_IMPORT).exists()
    src_data_status = "present" if src_data_present else "stripped_deliberate"
    # The hard gate is the module-scope data import; GPU is a transient runtime
    # factor (seq1024 fits 12 GB under the suffix-only config) so it does not
    # gate the *architectural* executability asserted here.
    run_executable_here = bool(src_data_present)

    # PIVOT (absolute-loss comparability via src.data) is public-doable ONLY when
    # src.data is present. In this public mirror it is deliberately stripped, so
    # PIVOT is a private-repo action — this corrects the prior docs-only decision
    # that framed PIVOT as a public-mirror src.data port.
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

    if not arc_complete:
        recommendation = "INCOMPLETE_ARC"
        rationale = "one or both full-budget §4 deposits are missing, non-citable, or not faithful-TIES"
    elif run_executable_here:
        recommendation = "FIRE_OR_EXTEND"
        rationale = "arc complete AND src.data present — a run is executable in this checkout"
    else:
        recommendation = "SHIP"
        rationale = (
            "arc complete (both legs citable faithful TIES) but the run is "
            "architecturally non-executable here (src.data deliberately stripped); "
            "the relative §4 verdict is the citable result, absolute-loss is private-repo"
        )

    unblock_step = (
        "PIVOT / absolute-loss is a PRIVATE-REPO action: src.data is deliberately "
        "stripped from this public mirror (see the note in scripts/prepare_data.py "
        "and the interface-without-implementation in tests/test_filter_dataset.py + "
        "tests/test_dedup.py). Execute in the private repo (/home/jinno/tg-lora): "
        "port/refresh src/data/filter_dataset.py quality filtering → re-fire the same "
        "corpus through the now-robust `make recover ... --rerun` path → "
        "form_freeze_validloss_deposit + freeze-replay → deposit here. Do NOT "
        "half-port src.data into this mirror (it breaks the deliberate boundary and "
        "cannot run without the full private pipeline + corpus)."
        if not pivot_public_doable
        else "src.data is present — fire the run directly through `make recover ... --rerun`."
    )

    return {
        "arc_complete": arc_complete,
        "legs": legs,
        "run_executable_here": run_executable_here,
        "src_data_status": src_data_status,
        "seq1024_full_budget_vram_floor_mib": SEQ1024_FULL_BUDGET_VRAM_FLOOR_MIB,
        "branches": branches,
        "recommendation": recommendation,
        "rationale": rationale,
        "unblock_step": unblock_step,
    }


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
        citable = "citable" if leg["citable_as_full_section4_verdict"] else "NON-citable"
        faithful = "faithful" if leg["faithful"] else "STALE-vs-rederived"
        lines.append(
            f"  [{leg['label']}] {verdict} · {citable} · {faithful} · "
            f"cand {leg['candidate_mean']:.4f} vs surr {leg['surrogate_mean']:.4f} "
            f"· CI[{leg['ci_lower']:+.4f}, {leg['ci_upper']:+.4f}] · seq{leg['seq_len']}"
        )
    lines.append(f"  arc_complete = {snapshot['arc_complete']}")
    lines.append("")
    lines.append(f"Run executable in this mirror: {snapshot['run_executable_here']}")
    lines.append(f"  src.data status: {snapshot['src_data_status']}")
    lines.append(
        f"  (seq1024 full-budget VRAM floor ~{snapshot['seq1024_full_budget_vram_floor_mib']} MiB "
        "— transient runtime factor, not the architectural gate)"
    )
    lines.append("")
    lines.append("Decision branches (executable_here):")
    for name, branch in snapshot["branches"].items():
        tag = "yes" if branch["executable_here"] else (
            "no — private-repo only" if branch.get("private_repo_only") else "no"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = assess_section4_decision(repo_root=args.repo_root)
    if args.json:
        print(json.dumps(snapshot, indent=2, default=str))
    else:
        print(format_decision(snapshot))
    # exit 0 when the arc is complete (the decision is actionable); 2 otherwise —
    # mirrors replay_freeze_validloss_ci's --expected exit contract.
    return 0 if snapshot["arc_complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
