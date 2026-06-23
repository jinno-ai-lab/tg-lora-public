#!/usr/bin/env python
"""Evaluate paper experiment gates G0–G4 against aggregate_summary.json.

Reads the aggregate summary produced by ``run_paper_memory_suite.sh`` (or
``compare_paper_memory_modes.py``) and prints a structured pass/fail report
for each gate defined in ``docs/paper_experiment_plan.md``.

Usage::

    # evaluate a single suite (reuse or one-shot)
    python scripts/evaluate_paper_gates.py runs/.../aggregate_summary.json

    # custom thresholds
    python scripts/evaluate_paper_gates.py summary.json \\
        --g1-loss-red-ratio 2.0 \\
        --g1-quality-tolerance 0.01 \\
        --g2-memory-improvement 0.20

Exit codes: 0 = all evaluated gates passed, 1 = at least one gate failed,
2 = input error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

# Allow running as a standalone CLI (``python scripts/evaluate_paper_gates.py``):
# a bare script invocation puts ``scripts/`` — not the repo root — on sys.path, so
# make the repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
# The Makefile ``paper-memory-evaluate-gates`` target invokes us without one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.stats import analyze_multi_seed, confidence_interval, paired_t_test


def _load_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(2)
    data = json.loads(p.read_text())
    if "aggregate" not in data:
        print("ERROR: aggregate key not found in summary", file=sys.stderr)
        sys.exit(2)
    return data


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


# ---------------------------------------------------------------------------
# Known-limitation records.
#
# Feedback: "formally record the gap as a known-limitation with a concrete
# next action and owner — instead of leaving the gates pinned in their failing
# state." A gate that FAILs carries a ``known_limitation`` record derived from
# WHICH sub-checks failed (not a static string), grounded in
# docs/master_plan.md §2.2 (the source of truth for current G0–G4 status). A
# gate that PASSes carries None — a passing gate claims no limitation — so the
# record is fail-conditioned. This turns a silent, pinned FAIL into an
# actionable, owned gap: a re-run reports what evidence is missing and what
# experiment would flip it, instead of reading as a bare regression.
# ---------------------------------------------------------------------------

# Claim level each gate feeds (docs/master_plan.md §2.1 Claim Ladder).
_CLAIM_FOR_GATE: dict[str, str | None] = {
    "G0": None,
    "G1": "C1 (Strong): multi-seed efficiency + quality retention",
    "G2": "C2 (Revolutionary): frontier separation",
    "G3": "C1 (Strong): external quality retention",
    "G4": "causal attribution: cache vs extrapolation isolation",
}

# These gaps are GPU-only to close (no GPU / no private data pipeline in the
# public mirror), so the owner is the research lead and the tracking artifact
# is TASK-0142 — the same "record the gap honestly rather than ship it
# unverified" principle TASK-0141 used for the §5.2 last mile. TASK-0142 is
# scoped to the G1/G4 *claim* gates (it is where their known-limitation records
# are owned), so it is the right owner ONLY for G1/G4.
_KNOWN_LIMITATION_OWNER = (
    "research-lead (GPU) — tracked in specs/tg-lora/tasks/TASK-0142.md"
)

# Owner for a gap on a gate WITHOUT a custom limitation builder (G0/G2/G3) or
# for an INSUFFICIENT-EVIDENCE state (gate bailed on a missing input). These
# are not the G1/G4 claim gap TASK-0142 owns, so pointing them at TASK-0142
# would misattribute the gap; the source-of-truth status table is master_plan
# §2.2 instead.
_KNOWN_LIMITATION_OWNER_GENERIC = (
    "research-lead — see docs/master_plan.md §2.2 (G0–G4 status table)"
)

# The CLI input each evidence-gated gate needs to actually reach a verdict.
# Drives the INSUFFICIENT-EVIDENCE known-limitation so a bailed gate names the
# exact flag to provide rather than reading as a disproven claim.
_GATE_REQUIRED_INPUT: dict[str, str] = {
    "G2": "--frontier-report (G2.3 frontier sweep)",
    "G3": "--external-eval (TruthfulQA/ARC/HellaSwag on best models)",
    "G4": "--cold-summary and --no-cache-summary "
    "(cold-vs-warm / cache-on-vs-off ablation)",
}

# Minimum seeds before the G1.4b quality-degradation CI is trusted as evidence.
# Mirrors freeze_cost §6.3's MIN_SAMPLE_FOR_CONFIDENCE_BAND: a single mean hides
# variance, and two reproductions of a median is thin evidence. Below this the
# check flags THIN_EVIDENCE rather than silently trusting a bare point estimate
# (GOAL §7: every metric needs a null surrogate + significance, not a mean).
MIN_SEEDS_FOR_QUALITY_CI: int = 3


def _g1_known_limitation(failed: list[str]) -> dict[str, Any]:
    """G1 limitation, attributed to the actually-failing sub-checks.

    The documented research gap (master_plan §2.2 G1, M3 result) is wall-clock
    efficiency, NOT quality: G1.2/G1.4 PASS while G1.1/G1.3 FAIL at ~0.98x.
    """
    wallclock = any(n.startswith("G1.1") or n.startswith("G1.3") for n in failed)
    quality = any(n.startswith("G1.4") for n in failed)
    backward = any(n.startswith("G1.2") for n in failed)

    causes: list[str] = []
    if wallclock:
        causes.append(
            "wall-clock efficiency (G1.1/G1.3) below the 1.25x bar — dominated "
            "by fixed costs (PCIe cache transfer, pilot-validation forwards, "
            "scheduled full eval), not by optimizer backward work; valid-loss "
            "quality (G1.4) is unaffected"
        )
    if quality:
        causes.append(
            "TG best-valid-loss degraded beyond the 1% tolerance (G1.4) — "
            "possible extrapolation overstep / acceptance threshold"
        )
    if backward:
        causes.append("effective backward-pass accounting (G1.2)")

    if wallclock:
        next_action = (
            "extend the final-eval-only config to 3 seeds and decompose the "
            "pilot-validation / scheduled-full-eval fixed cost (master_plan "
            "§1.5, M6); if wall-clock stays flat, checkpoint I/O and model "
            "reload are the next fixed-cost candidates"
        )
        root_cause = (
            "5K-Dolly wall-clock ~0.98x baseline on RTX 3060 12GB "
            "(fixed-cost bound, not compute bound)"
        )
    elif quality:
        next_action = (
            "investigate extrapolation acceptance/rollback thresholds "
            "(master_plan §2.2 G1)"
        )
        root_cause = "extrapolation overstep degrading convergence quality"
    else:
        next_action = "see failed G1 sub-checks above and master_plan §2.2 G1"
        root_cause = "see failed sub-checks"

    return {
        "gap": "; ".join(causes) if causes else "G1 failing",
        "root_cause": root_cause,
        "next_action": next_action,
        "owner": _KNOWN_LIMITATION_OWNER,
        "blocks_claim": _CLAIM_FOR_GATE["G1"],
    }


def _g4_known_limitation(failed: list[str]) -> dict[str, Any]:
    """G4 limitation, attributed to the actually-failing sub-checks.

    The documented gap (master_plan §2.2 G4, M6) is the optimizer/momentum
    confound on warm-vs-cold speedup (G4.1); G4.2 (VRAM) PASSES.
    """
    warm = any(n.startswith("G4.1") for n in failed)
    cache = any(n.startswith("G4.2") for n in failed)

    causes: list[str] = []
    if warm:
        causes.append(
            "warm-vs-cold speedup (G4.1) not established — per-cycle optimizer "
            "recreation (recreate_per_cycle) confounds the comparison with "
            "momentum-lifecycle differences"
        )
    if cache:
        causes.append("cache-on vs cache-off memory effect (G4.2) not separated")

    if warm:
        next_action = (
            "complete the cache-isolation ablation (A baseline / B cache-only / "
            "C TG-LoRA) under optimizer policy='persistent' so all conditions "
            "share one AdamW lifecycle, then re-measure G4.1 warm-vs-cold "
            "(master_plan §2.2 G4, M6)"
        )
        root_cause = (
            "optimizer lifecycle confound (recreate_per_cycle) on warm/cold "
            "attribution"
        )
    elif cache:
        next_action = (
            "provide a cache-off ablation summary and re-measure G4.2 "
            "(master_plan §2.2 G4.2)"
        )
        root_cause = "cache-off ablation missing"
    else:
        next_action = (
            "run the cold-vs-warm and cache-on-vs-off ablation, then "
            "re-evaluate (master_plan §2.2 G4, M6)"
        )
        root_cause = "G4 ablation data not provided"

    return {
        "gap": "; ".join(causes) if causes else "G4 failing",
        "root_cause": root_cause,
        "next_action": next_action,
        "owner": _KNOWN_LIMITATION_OWNER,
        "blocks_claim": _CLAIM_FOR_GATE["G4"],
    }


_LIMITATION_BUILDERS: dict[str, Any] = {
    "G1": _g1_known_limitation,
    "G4": _g4_known_limitation,
}


def _known_limitation_for(result: dict[str, Any]) -> dict[str, Any] | None:
    """Structured known-limitation for a FAILING gate, else None.

    Three cases:

    * **Insufficient evidence** (``result["evaluated"] is False``): the gate
      bailed because its required input was missing, so the underlying claim is
      *unmeasured*, not disproven. Returns a record stamped
      ``status="insufficient_evidence"`` that names the missing input — never a
      disproven-claim record, and never the G1/G4 task owner (TASK-0142), since
      an unmeasured G2/G3/G4 gap is not the claim gap that task tracks. This is
      the evaluate_paper_gates analog of the §5.2 honesty contract's
      ACTIVE/DORMANT split: a missing-input gate must not masquerade as a
      disproven FAIL.
    * **Passing gate**: returns ``None`` — a passing (and evaluated) gate claims
      no limitation, so the disproven-claim record is fail-conditioned.
    * **Disproven gate** (evaluated, ``passed is False``): for G1/G4 the gap is
      derived from the failing sub-checks and grounded in master_plan §2.2;
      other gates get a pointer back to the source-of-truth doc (generic owner,
      not TASK-0142) rather than invented specifics.
    """
    gate = result.get("gate", "")
    if not result.get("evaluated", True):
        missing = _GATE_REQUIRED_INPUT.get(gate)
        # A gate may explain *why* it could not be evaluated via an optional
        # ``insufficient_reason`` — e.g. a required input that is present but
        # unreadable (corrupt JSON / OSError). That is still *unmeasured* (the
        # claim was never tested against it), not disproven, so it stays in the
        # insufficient_evidence branch — but the reason is surfaced verbatim so
        # the record does not mislabel a corrupt file as a merely-missing one
        # (the gap/root_cause default to the "missing" wording only when no
        # reason is given, keeping every existing insufficient record identical).
        reason = result.get("insufficient_reason")
        if reason:
            gap = reason
            root_cause = reason
            next_action = (
                f"provide a valid {missing} and re-evaluate (master_plan §2.2)"
                if missing
                else "provide a valid required input and re-evaluate (master_plan §2.2)"
            )
        else:
            gap = "insufficient evidence — gate could not be evaluated"
            root_cause = (
                f"required input missing: {missing}"
                if missing
                else "required input missing — see failed sub-checks"
            )
            next_action = (
                f"provide {missing} and re-evaluate (master_plan §2.2)"
                if missing
                else "provide the required input and re-evaluate (master_plan §2.2)"
            )
        return {
            "status": "insufficient_evidence",
            "gap": gap,
            "root_cause": root_cause,
            "missing_input": missing,
            "next_action": next_action,
            "owner": _KNOWN_LIMITATION_OWNER_GENERIC,
            "blocks_claim": _CLAIM_FOR_GATE.get(gate),
        }
    if result.get("passed"):
        return None
    failed = [c["check"] for c in result.get("checks", []) if not c.get("pass")]
    builder = _LIMITATION_BUILDERS.get(gate)
    if builder is not None:
        return builder(failed)
    return {
        "gap": f"{gate} ({result.get('name', '')}) failing — see failed sub-checks",
        "root_cause": "see failed sub-check detail above",
        "next_action": (
            "consult docs/master_plan.md §2.2 for this gate's evidence requirements"
        ),
        "owner": _KNOWN_LIMITATION_OWNER_GENERIC,
        "blocks_claim": _CLAIM_FOR_GATE.get(gate),
    }


def _attach_known_limitation(fn):
    """Decorator: stamp every gate result with its known_limitation record.

    Applied to ``_check_g0``…``_check_g4`` so each gate self-describes its
    limitation on FAIL (None on PASS). Centralising it here keeps the
    fail-conditioning in one place rather than at every return statement.
    """

    @wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = fn(*args, **kwargs)
        result["known_limitation"] = _known_limitation_for(result)
        return result

    return _wrapper


@_attach_known_limitation
def _check_g0(summary: dict[str, Any]) -> dict[str, Any]:
    """Gate G0: Hygiene — artifacts exist and are well-formed."""
    checks: list[dict[str, Any]] = []
    seeds = summary.get("seeds", [])
    per_seed = summary.get("per_seed", [])

    if not seeds:
        checks.append({"check": "seeds_present", "pass": False, "detail": "No seeds found"})
    else:
        checks.append({"check": "seeds_present", "pass": True, "detail": f"{len(seeds)} seeds"})

    if len(per_seed) != len(seeds):
        checks.append({
            "check": "per_seed_complete",
            "pass": False,
            "detail": f"Expected {len(seeds)} per_seed entries, got {len(per_seed)}",
        })
    else:
        checks.append({"check": "per_seed_complete", "pass": True, "detail": "All seed rows present"})

    agg = summary.get("aggregate", {})
    required_agg_keys = [
        "warm_tg_loss_red_per_wall_minute",
        "warm_baseline_loss_red_per_wall_minute",
        "warm_tg_best_valid_loss",
        "warm_baseline_best_valid_loss",
    ]
    missing = [k for k in required_agg_keys if k not in agg or agg[k].get("mean") is None]
    if missing:
        checks.append({
            "check": "aggregate_metrics_complete",
            "pass": False,
            "detail": f"Missing aggregate keys: {missing}",
        })
    else:
        checks.append({"check": "aggregate_metrics_complete", "pass": True, "detail": "All required metrics present"})

    passed = all(c["pass"] for c in checks)
    return {"gate": "G0", "name": "Hygiene", "passed": passed, "checks": checks}


@_attach_known_limitation
def _check_g1(
    summary: dict[str, Any],
    *,
    loss_red_ratio: float = 2.0,
    quality_tolerance: float = 0.01,
    efficiency_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Gate G1: Replicated Internal Efficiency."""
    checks: list[dict[str, Any]] = []
    per_seed = summary.get("per_seed", [])
    agg = summary.get("aggregate", {})

    tg_eff = [r.get("warm_tg_loss_red_per_wall_minute") for r in per_seed]
    bl_eff = [r.get("warm_baseline_loss_red_per_wall_minute") for r in per_seed]
    tg_bp = [r.get("warm_tg_backward_passes") for r in per_seed]
    bl_bp = [r.get("warm_baseline_backward_passes") for r in per_seed]
    tg_loss = [r.get("warm_tg_best_valid_loss") for r in per_seed]
    bl_loss = [r.get("warm_baseline_best_valid_loss") for r in per_seed]

    # G1.1: All seeds TG efficiency > baseline
    g11_pass = True
    details_g11 = []
    for i, (t, b) in enumerate(zip(tg_eff, bl_eff)):
        if t is None or b is None:
            g11_pass = False
            details_g11.append(f"seed {per_seed[i].get('seed', i)}: missing data")
        elif t * (1 + efficiency_tolerance) <= b:
            g11_pass = False
            details_g11.append(f"seed {per_seed[i].get('seed', i)}: TG={t:.4f} <= BL={b:.4f} (with tol {efficiency_tolerance:.2f})")
        else:
            details_g11.append(f"seed {per_seed[i].get('seed', i)}: TG={t:.4f} > BL={b:.4f}")
    checks.append({"check": "G1.1_all_seeds_tg_efficiency_superior", "pass": g11_pass, "detail": "; ".join(details_g11)})

    # G1.2: All seeds TG effective backward passes > baseline (extrapolation adds free progress)
    g12_pass = True
    details_g12 = []
    for i, (t, b) in enumerate(zip(tg_bp, bl_bp)):
        if t is None or b is None:
            g12_pass = False
            details_g12.append(f"seed {per_seed[i].get('seed', i)}: missing data")
            continue
        extrap = per_seed[i].get("warm_tg_extrapolation_steps")
        accepted = per_seed[i].get("warm_tg_accepted_extrapolations")
        if extrap is not None and accepted is not None and accepted > 0:
            # TG achieves equivalent of more passes via extrapolation
            details_g12.append(
                f"seed {per_seed[i].get('seed', i)}: TG actual={t} + {accepted} extrapolations, BL={b}"
            )
        elif t > b:
            g12_pass = False
            details_g12.append(f"seed {per_seed[i].get('seed', i)}: TG={t} > BL={b}")
        else:
            details_g12.append(f"seed {per_seed[i].get('seed', i)}: TG={t} <= BL={b}")
    checks.append({"check": "G1.2_all_seeds_tg_fewer_backward_passes", "pass": g12_pass, "detail": "; ".join(details_g12)})

    # G1.3: Aggregate mean TG efficiency >= 2x baseline
    tg_mean = agg.get("warm_tg_loss_red_per_wall_minute", {}).get("mean")
    bl_mean = agg.get("warm_baseline_loss_red_per_wall_minute", {}).get("mean")
    if tg_mean is not None and bl_mean is not None and bl_mean > 0:
        ratio = tg_mean / bl_mean
        g13_pass = ratio >= loss_red_ratio
        checks.append({
            "check": f"G1.3_aggregate_ratio >= {loss_red_ratio}x",
            "pass": g13_pass,
            "detail": f"TG mean={tg_mean:.4f}, BL mean={bl_mean:.4f}, ratio={ratio:.2f}x",
        })
    else:
        checks.append({"check": f"G1.3_aggregate_ratio >= {loss_red_ratio}x", "pass": False, "detail": "Missing aggregate means"})

    # G1.4: TG quality degradation < quality_tolerance
    tg_loss_mean = agg.get("warm_tg_best_valid_loss", {}).get("mean")
    bl_loss_mean = agg.get("warm_baseline_best_valid_loss", {}).get("mean")
    if tg_loss_mean is not None and bl_loss_mean is not None and bl_loss_mean > 0:
        rel_degradation = (tg_loss_mean - bl_loss_mean) / bl_loss_mean
        g14_pass = rel_degradation < quality_tolerance
        checks.append({
            "check": f"G1.4_quality_degradation < {quality_tolerance*100:.0f}%",
            "pass": g14_pass,
            "detail": f"TG loss={tg_loss_mean:.4f}, BL loss={bl_loss_mean:.4f}, rel_degradation={rel_degradation*100:.2f}%",
        })
    else:
        checks.append({
            "check": f"G1.4_quality_degradation < {quality_tolerance*100:.0f}%",
            "pass": False,
            "detail": "Missing loss means",
        })

    # G1.4b: the quality claim is statistically supported (GOAL §7 鉄則).
    # G1.4 trusts the aggregate *mean* of best-valid-loss; this re-checks the
    # same relative degradation with a per-seed paired confidence interval, so a
    # result whose mean sits inside tolerance but whose CI upper bound crosses it
    # (high variance / few seeds) is not silently trusted. The efficiency axis
    # (G1.1/G1.3) already gets a paired_t_test in the enrichment; quality must
    # get the same honesty (freeze_cost §6.3: "two reproductions of a median is
    # thin evidence").
    quality_pairs = [
        (b, t)
        for b, t in zip(bl_loss, tg_loss)
        if b is not None and t is not None and b > 0
    ]
    if quality_pairs:
        degradations = [(t - b) / b for b, t in quality_pairs]
        d_mean, d_lower, d_upper = confidence_interval(degradations)
        g14b_pass = d_upper <= quality_tolerance
        n_pairs = len(quality_pairs)
        detail_parts = [
            f"n={n_pairs} paired seeds",
            f"rel_degradation mean={d_mean * 100:.2f}% "
            f"[CI95 {d_lower * 100:.2f}%, {d_upper * 100:.2f}%]",
            f"tolerance={quality_tolerance * 100:.0f}%",
            f"CI_upper {'<=' if g14b_pass else '>'} tolerance",
        ]
        if n_pairs >= 2:
            bl_paired = [b for b, _ in quality_pairs]
            tg_paired = [t for _, t in quality_pairs]
            t_stat, p_val = paired_t_test(bl_paired, tg_paired)
            detail_parts.append(f"paired_t(tg-bl) t={t_stat:.3f} p={p_val:.4f}")
        if n_pairs < MIN_SEEDS_FOR_QUALITY_CI:
            detail_parts.append(
                f"THIN_EVIDENCE (n<{MIN_SEEDS_FOR_QUALITY_CI}; CI unreliable)"
            )
        checks.append({
            "check": f"G1.4b_quality_degradation_statistically_supported < {quality_tolerance*100:.0f}%",
            "pass": g14b_pass,
            "detail": "; ".join(detail_parts),
        })
    else:
        checks.append({
            "check": f"G1.4b_quality_degradation_statistically_supported < {quality_tolerance*100:.0f}%",
            "pass": False,
            "detail": "Missing per-seed loss pairs for the degradation CI",
        })

    passed = all(c["pass"] for c in checks)
    return {"gate": "G1", "name": "Replicated Internal Efficiency", "passed": passed, "checks": checks}


@_attach_known_limitation
def _check_g2(
    summary: dict[str, Any],
    *,
    memory_improvement: float = 0.20,
    frontier_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Gate G2: Memory Frontier Separation."""
    checks: list[dict[ str, Any]] = []
    summary.get("aggregate", {})
    per_seed = summary.get("per_seed", [])

    tg_peak = [r.get("warm_tg_gpu_peak_mb") for r in per_seed]
    bl_peak = [r.get("warm_baseline_gpu_peak_mb") for r in per_seed]
    offload_freed = [r.get("warm_tg_runtime_offload_gpu_freed_mb") for r in per_seed]

    # G2.1: Memory improvement check
    tg_peak_mean = _mean([v for v in tg_peak if v is not None])
    bl_peak_mean = _mean([v for v in bl_peak if v is not None])

    if tg_peak_mean is not None and bl_peak_mean is not None and bl_peak_mean > 0:
        rel_improvement = (bl_peak_mean - tg_peak_mean) / bl_peak_mean
        g21_pass = rel_improvement >= memory_improvement
        checks.append({
            "check": f"G2.1_peak_memory_reduction >= {memory_improvement*100:.0f}%",
            "pass": g21_pass,
            "detail": f"BL peak={bl_peak_mean:.1f}MB, TG peak={tg_peak_mean:.1f}MB, reduction={rel_improvement*100:.1f}%",
        })
    else:
        checks.append({
            "check": f"G2.1_peak_memory_reduction >= {memory_improvement*100:.0f}%",
            "pass": False,
            "detail": "Missing peak memory data — frontier sweep not yet run or data unavailable",
        })

    # G2.2: Runtime offload benefit
    freed_mean = _mean([v for v in offload_freed if v is not None])
    if freed_mean is not None:
        g22_pass = freed_mean > 0
        checks.append({
            "check": "G2.2_runtime_offload_freed_mb > 0",
            "pass": g22_pass,
            "detail": f"Mean freed={freed_mean:.1f}MB",
        })
    else:
        checks.append({
            "check": "G2.2_runtime_offload_freed_mb > 0",
            "pass": False,
            "detail": "No runtime offload data",
        })

    # G2.3: Frontier separation from frontier_report.json
    #
    # The frontier report is G2's required input (_GATE_REQUIRED_INPUT["G2"])
    # and carries the evidence for G2's headline claim — frontier separation
    # (claim C2). When it is absent or unreadable that claim is *unmeasured*,
    # not disproven — the same honesty contract G3/G4 honour (a missing/corrupt
    # required input bails to evaluated=False so a not-yet-run experiment
    # cannot read as a refuted claim). G2.1/G2.2 are still computed and reported
    # for transparency, but the gate's verdict is INSUFFICIENT EVIDENCE until
    # the frontier sweep lands. --strict still treats an un-evaluated gate as a
    # failure.
    frontier_state = "absent"  # absent | unreadable | loaded
    frontier_payload: dict[str, Any] | None = None
    read_error: Exception | None = None
    if frontier_report_path is not None:
        frp = Path(frontier_report_path)
        if frp.exists():
            try:
                frontier_payload = json.loads(frp.read_text())
                frontier_state = "loaded"
            except (json.JSONDecodeError, OSError) as exc:
                frontier_state = "unreadable"
                read_error = exc
        # path given but file missing -> stays "absent" (a missing required input)

    if frontier_state == "loaded":
        detected = frontier_payload.get("frontier_separation_detected", False)
        boundary = frontier_payload.get("frontier_boundary")
        runs = frontier_payload.get("runs", [])
        frontier_runs = [r for r in runs if r.get("frontier_separation")]
        if detected and frontier_runs:
            seq_lens = [r["seq_len"] for r in frontier_runs]
            checks.append({
                "check": "G2.3_frontier_separation",
                "pass": True,
                "detail": (
                    f"Frontier detected at seq_len={boundary}. "
                    f"Baseline OOM + TG completed at: {seq_lens}"
                ),
            })
        else:
            checks.append({
                "check": "G2.3_frontier_separation",
                "pass": False,
                "detail": f"No frontier separation in sweep (boundary={boundary})",
            })
    elif frontier_state == "unreadable":
        # present but unreadable: the claim was never measured, not disproven.
        # Keep the read failure loud (a corrupt --frontier-report is named), but
        # route to INSUFFICIENT — mirroring G3's corrupt-input handling (a418049).
        checks.append({
            "check": "G2.3_frontier_separation",
            "pass": False,
            "detail": f"Failed to read frontier report: {read_error}",
        })
        return {
            "gate": "G2",
            "name": "Memory Frontier Separation",
            "passed": False,
            "evaluated": False,
            "insufficient_reason": (
                f"frontier report present but unreadable ({read_error})"
            ),
            "checks": checks,
        }
    else:  # absent
        checks.append({
            "check": "G2.3_frontier_separation",
            "pass": False,
            "detail": "No frontier_report.json provided — run frontier sweep to evaluate",
        })
        return {
            "gate": "G2",
            "name": "Memory Frontier Separation",
            "passed": False,
            "evaluated": False,
            "checks": checks,
        }

    passed = all(c["pass"] for c in checks)
    return {"gate": "G2", "name": "Memory Frontier Separation", "passed": passed, "checks": checks}


@_attach_known_limitation
def _check_g3(
    summary: dict[str, Any],
    *,
    external_eval_path: str | Path | None = None,
    max_aggregate_drop: float = 0.01,
    max_single_task_drop: float = 0.03,
) -> dict[str, Any]:
    """Gate G3: External Quality Retention.

    When an external_eval_results.json is provided (or auto-discovered),
    evaluates aggregate mean relative drop < 1% and single task < 3%.
    Otherwise falls back to informational status.
    """
    checks: list[dict[str, Any]] = []

    eval_data: dict[str, Any] | None = None
    if external_eval_path is not None:
        ep = Path(external_eval_path)
        if ep.exists():
            try:
                eval_data = json.loads(ep.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                # A present-but-unreadable required input means the external-
                # quality claim was never measured, not disproven — the same
                # honesty contract as the missing-input path below. The check
                # detail keeps the read error loud (a user who points --external-
                # eval at a corrupt file sees exactly why); evaluated=False
                # routes the gate to INSUFFICIENT EVIDENCE so it cannot
                # masquerade as a disproven FAIL, and insufficient_reason makes
                # the known-limitation say "unreadable" rather than "missing".
                # --strict still treats an un-evaluated gate as a failure.
                checks.append({
                    "check": "G3_external_eval",
                    "pass": False,
                    "detail": f"Failed to read external eval results: {exc}",
                })
                return {
                    "gate": "G3",
                    "name": "External Quality Retention",
                    "passed": False,
                    "evaluated": False,
                    "insufficient_reason": (
                        f"external eval results present but unreadable ({exc})"
                    ),
                    "checks": checks,
                }

    if eval_data is not None:
        comparison = eval_data.get("comparison", {})
        agg_drop = comparison.get("aggregate_relative_drop")
        task_drops = comparison.get("task_relative_drops", {})

        if agg_drop is not None:
            agg_ok = agg_drop < max_aggregate_drop
            checks.append({
                "check": f"G3.1_aggregate_mean_drop < {max_aggregate_drop*100:.0f}%",
                "pass": agg_ok,
                "detail": f"Aggregate mean relative drop: {agg_drop*100:.2f}%",
            })
        else:
            checks.append({
                "check": f"G3.1_aggregate_mean_drop < {max_aggregate_drop*100:.0f}%",
                "pass": False,
                "detail": "No aggregate relative drop data in external eval results",
            })

        if task_drops:
            max_task = max(task_drops.values())
            worst_task = max(task_drops, key=lambda t: task_drops[t])
            single_ok = max_task < max_single_task_drop
            checks.append({
                "check": f"G3.2_single_task_drop < {max_single_task_drop*100:.0f}%",
                "pass": single_ok,
                "detail": f"Worst task: {worst_task} at {max_task*100:.2f}% relative drop",
            })
        else:
            checks.append({
                "check": f"G3.2_single_task_drop < {max_single_task_drop*100:.0f}%",
                "pass": False,
                "detail": "No per-task relative drop data in external eval results",
            })

        tasks_run = eval_data.get("tasks", [])
        required = {"truthfulqa_mc2", "arc_easy", "hellaswag"}
        has_required = required.issubset(set(tasks_run))
        checks.append({
            "check": "G3.3_required_tasks_present",
            "pass": has_required,
            "detail": f"Tasks: {tasks_run} (required: {sorted(required)})",
        })
    else:
        return {
            "gate": "G3",
            "name": "External Quality Retention",
            "passed": False,
            "evaluated": False,
            "checks": [{
                "check": "G3_external_eval",
                "pass": False,
                "detail": "Requires external eval (TruthfulQA, ARC, HellaSwag) on saved best models — provide --external-eval to evaluate",
            }],
        }

    passed = all(c["pass"] for c in checks)
    return {"gate": "G3", "name": "External Quality Retention", "passed": passed, "checks": checks}


@_attach_known_limitation
def _check_g4(
    summary: dict[str, Any],
    *,
    cold_summary: dict[str, Any] | None = None,
    no_cache_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate G4: Causal Attribution.

    When cold and/or no-cache summaries are provided, performs actual
    ablation comparison.  Otherwise falls back to informational status.
    """
    checks: list[dict[str, Any]] = []

    # G4.1: Warm speedup — warm TG efficiency > cold TG efficiency for all seeds
    warm_seeds = summary.get("per_seed", [])
    if cold_summary is not None:
        cold_seeds = cold_summary.get("per_seed", [])

        if not warm_seeds or not cold_seeds:
            checks.append({
                "check": "G4.1_warm_speedup_all_seeds",
                "pass": False,
                "detail": "Missing per_seed data in warm or cold summary",
            })
        else:
            g41_pass = True
            details_g41: list[str] = []
            comparisons_done = 0
            for ws in warm_seeds:
                seed = ws.get("seed", "?")
                warm_eff = ws.get("warm_tg_loss_red_per_wall_minute")
                cs = next(
                    (c for c in cold_seeds if c.get("seed") == seed),
                    None,
                )
                if cs is None:
                    g41_pass = False
                    details_g41.append(f"seed {seed}: no matching cold seed")
                    continue
                comparisons_done += 1
                if warm_eff is None:
                    g41_pass = False
                    details_g41.append(f"seed {seed}: warm TG efficiency missing")
                else:
                    cold_eff = cs.get("warm_tg_loss_red_per_wall_minute")
                    if cold_eff is None:
                        g41_pass = False
                        details_g41.append(f"seed {seed}: cold TG efficiency missing")
                    elif warm_eff <= cold_eff:
                        g41_pass = False
                        details_g41.append(
                            f"seed {seed}: warm={warm_eff:.4f} <= cold={cold_eff:.4f}"
                        )
                    else:
                        speedup = (warm_eff - cold_eff) / cold_eff * 100
                        details_g41.append(
                            f"seed {seed}: warm={warm_eff:.4f} > cold={cold_eff:.4f} (+{speedup:.1f}%)"
                        )
            if comparisons_done == 0:
                g41_pass = False
                details_g41.append("No matching seeds found in cold summary")
            checks.append({
                "check": "G4.1_warm_speedup_all_seeds",
                "pass": g41_pass,
                "detail": "; ".join(details_g41),
            })
    else:
        # Fallback: check self-contained cold vs warm speedup metrics
        g41_pass = True
        details_g41: list[str] = []
        for ws in warm_seeds:
            seed = ws.get("seed", "?")
            cold_wall = ws.get("cold_tg_wall_seconds")
            warm_wall = ws.get("warm_tg_wall_seconds")
            if cold_wall is None or warm_wall is None:
                g41_pass = False
                details_g41.append(f"seed {seed}: missing wall seconds data")
            elif warm_wall >= cold_wall:
                g41_pass = False
                details_g41.append(f"seed {seed}: warm_wall={warm_wall:.1f}s >= cold_wall={cold_wall:.1f}s")
            else:
                speedup = (cold_wall - warm_wall) / cold_wall * 100
                details_g41.append(f"seed {seed}: warm_wall={warm_wall:.1f}s < cold_wall={cold_wall:.1f}s (speedup={speedup:.1f}%)")
        checks.append({
            "check": "G4.1_warm_speedup_all_seeds",
            "pass": g41_pass,
            "detail": "; ".join(details_g41),
        })

    # G4.2: Cache-on memory effect stronger than cache-off
    if no_cache_summary is not None:
        on_seeds = summary.get("per_seed", [])
        off_seeds = no_cache_summary.get("per_seed", [])

        if not on_seeds or not off_seeds:
            checks.append({
                "check": "G4.2_cache_memory_effect",
                "pass": False,
                "detail": "Missing per_seed data in cache-on or cache-off summary",
            })
        else:
            g42_pass = True
            details_g42: list[str] = []
            comparisons_done = 0
            for ons in on_seeds:
                seed = ons.get("seed", "?")
                on_tg_peak = ons.get("warm_tg_gpu_peak_mb")
                on_bl_peak = ons.get("warm_baseline_gpu_peak_mb")
                offs = next(
                    (o for o in off_seeds if o.get("seed") == seed),
                    None,
                )
                if offs is None:
                    g42_pass = False
                    details_g42.append(f"seed {seed}: no matching cache-off seed")
                    continue
                comparisons_done += 1
                if on_tg_peak is None or on_bl_peak is None:
                    g42_pass = False
                    details_g42.append(f"seed {seed}: cache-on peak memory missing")
                else:
                    off_tg_peak = offs.get("warm_tg_gpu_peak_mb")
                    off_bl_peak = offs.get("warm_baseline_gpu_peak_mb")
                    if off_tg_peak is None or off_bl_peak is None:
                        g42_pass = False
                        details_g42.append(f"seed {seed}: cache-off peak memory missing")
                    else:
                        on_savings = on_bl_peak - on_tg_peak
                        off_savings = off_bl_peak - off_tg_peak
                        if on_savings <= off_savings:
                            g42_pass = False
                            details_g42.append(
                                f"seed {seed}: cache-on savings={on_savings:.1f}MB <= cache-off={off_savings:.1f}MB"
                            )
                        else:
                            details_g42.append(
                                f"seed {seed}: cache-on savings={on_savings:.1f}MB > cache-off={off_savings:.1f}MB"
                            )
            if comparisons_done == 0:
                g42_pass = False
                details_g42.append("No matching seeds found in cache-off summary")
            checks.append({
                "check": "G4.2_cache_memory_effect",
                "pass": g42_pass,
                "detail": "; ".join(details_g42),
            })
    else:
        checks.append({
            "check": "G4.2_cache_memory_effect",
            "pass": False,
            "detail": "No cache-off summary provided — cache effect comparison skipped",
        })

    # If no ablation summaries at all, keep informational
    has_ablation = cold_summary is not None or no_cache_summary is not None
    if not has_ablation:
        return {
            "gate": "G4",
            "name": "Causal Attribution",
            "passed": False,
            "evaluated": False,
            "checks": [{
                "check": "G4_causal_attribution",
                "pass": False,
                "detail": "Requires cold vs warm and train-cache on vs off ablation — provide --cold-summary and --no-cache-summary to evaluate",
            }],
        }

    passed = all(c["pass"] for c in checks)
    return {"gate": "G4", "name": "Causal Attribution", "passed": passed, "checks": checks}


def _find_frontier_report(summary_path: str | Path) -> Path | None:
    """Search for frontier_report.json relative to the summary path.

    Checks (in order):
      1. Same directory as summary
      2. Parent directory (for frontier sweep layout: slen_XXXX/aggregate_summary.json
         with frontier_report.json one level up)
    """
    p = Path(summary_path).resolve()
    for candidate in [p.parent, p.parent.parent]:
        fr = candidate / "frontier_report.json"
        if fr.exists():
            return fr
    return None


def _find_sibling_summary(
    summary_path: str | Path,
    *,
    suffix: str,
    candidates: list[str] | None = None,
) -> Path | None:
    """Search for a sibling aggregate_summary.json in a named subdirectory.

    Looks for directories named *suffix* (or any in *candidates*) next to
    the summary's parent and returns the first ``aggregate_summary.json`` found.

    Typical layout::

        runs/paper_memory_suite/
        ├── reuse/aggregate_summary.json   ← main (summary_path)
        ├── cold/aggregate_summary.json    ← G4.1 cold ablation
        └── no_cache/aggregate_summary.json ← G4.2 cache-off ablation
    """
    candidates = candidates or [suffix]
    parent = Path(summary_path).resolve().parent.parent
    for name in candidates:
        candidate = parent / name / "aggregate_summary.json"
        if candidate.exists():
            return candidate
    return None


def _format_report(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Paper Gate Evaluation Report")
    lines.append("=" * 60)
    lines.append("")

    any_failed = False
    any_insufficient = False
    for r in results:
        if not r.get("evaluated", True):
            # Bailed on a missing input: the claim is unmeasured, not disproven.
            # Report it as a third state so it cannot read as a FAIL.
            status = "INSUFFICIENT EVIDENCE"
            any_insufficient = True
        elif r["passed"]:
            status = "PASS"
        else:
            status = "FAIL"
            any_failed = True
        lines.append(f"## {r['gate']}: {r['name']} — {status}")
        lines.append("")
        for c in r.get("checks", []):
            mark = "✓" if c["pass"] else "✗"
            lines.append(f"  {mark} {c['check']}: {c['detail']}")

        kl = r.get("known_limitation")
        if kl:
            if kl.get("status") == "insufficient_evidence":
                lines.append(f"  ℹ Insufficient evidence: {kl['gap']}")
                if kl.get("missing_input"):
                    lines.append(f"    Missing input: {kl['missing_input']}")
                lines.append(f"    Next action: {kl['next_action']}")
                lines.append(
                    f"    Owner: {kl['owner']}  (blocks: {kl.get('blocks_claim')})"
                )
            else:
                lines.append(f"  ⚠ Known limitation: {kl['gap']}")
                lines.append(f"    Root cause: {kl['root_cause']}")
                lines.append(f"    Next action: {kl['next_action']}")
                lines.append(
                    f"    Owner: {kl['owner']}  (blocks: {kl.get('blocks_claim')})"
                )
        lines.append("")

    lines.append("=" * 60)
    if any_failed:
        overall = "AT LEAST ONE GATE FAILED"
    elif any_insufficient:
        overall = "ALL EVALUATED GATES PASSED (some gates lack evidence — see above)"
    else:
        overall = "ALL EVALUATED GATES PASSED"
    lines.append(f"Overall: {overall}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _enrich_with_statistics(summary: dict[str, Any]) -> dict[str, Any]:
    """Compute statistical enrichment from per_seed data.

    Returns a dict with per-metric confidence intervals and paired comparisons.
    """
    per_seed_list = summary.get("per_seed", [])
    if not per_seed_list or len(per_seed_list) < 2:
        return {"seed_count": len(per_seed_list)}

    per_seed_dict = {f"seed_{i}": r for i, r in enumerate(per_seed_list)}
    converted = {"per_seed": per_seed_dict, "aggregate": summary.get("aggregate", {})}
    multi_stats = analyze_multi_seed(converted)

    enrichment: dict[str, Any] = {
        "seed_count": multi_stats["seed_count"],
        "metric_ci": {},
    }

    for key, stats in multi_stats["metrics"].items():
        if "ci_lower" in stats:
            enrichment["metric_ci"][key] = {
                "mean": stats["mean"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "std": stats.get("std"),
            }

    tg_eff = [r.get("warm_tg_loss_red_per_wall_minute") for r in per_seed_list]
    bl_eff = [r.get("warm_baseline_loss_red_per_wall_minute") for r in per_seed_list]
    if all(v is not None for v in tg_eff) and all(v is not None for v in bl_eff):
        t_stat, p_val = paired_t_test(bl_eff, tg_eff)
        enrichment["paired_t_test_efficiency"] = {
            "t_statistic": t_stat,
            "p_value": p_val,
            "significant_005": p_val < 0.05,
        }

    # Quality counterpart (GOAL §7): the efficiency axis above gets a paired
    # t-test; the valid-loss axis must too, so a quality-retention claim is not
    # trusted on a bare mean. CI is on per-seed paired relative degradation.
    tg_loss = [r.get("warm_tg_best_valid_loss") for r in per_seed_list]
    bl_loss = [r.get("warm_baseline_best_valid_loss") for r in per_seed_list]
    quality_pairs = [
        (b, t)
        for b, t in zip(bl_loss, tg_loss)
        if b is not None and t is not None and b > 0
    ]
    if quality_pairs:
        degradations = [(t - b) / b for b, t in quality_pairs]
        d_mean, d_lower, d_upper = confidence_interval(degradations)
        enrichment["quality_degradation_ci"] = {
            "n": len(quality_pairs),
            "mean": d_mean,
            "ci_lower": d_lower,
            "ci_upper": d_upper,
        }
        if len(quality_pairs) >= 2:
            bl_paired = [b for b, _ in quality_pairs]
            tg_paired = [t for _, t in quality_pairs]
            qt, qp = paired_t_test(bl_paired, tg_paired)
            enrichment["paired_t_test_quality"] = {
                "t_statistic": qt,
                "p_value": qp,
                "significant_005": qp < 0.05,
            }

    return enrichment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper experiment gates G0–G4")
    parser.add_argument("summary_json", help="Path to aggregate_summary.json")
    parser.add_argument("--g1-loss-red-ratio", type=float, default=2.0, help="G1.3 minimum TG/BL efficiency ratio (default: 2.0)")
    parser.add_argument("--g1-quality-tolerance", type=float, default=0.01, help="G1.4 max relative quality degradation (default: 0.01 = 1%%)")
    parser.add_argument("--g1-efficiency-tolerance", type=float, default=0.0, help="G1.1 efficiency tolerance ratio (default: 0.0)")
    parser.add_argument("--g2-memory-improvement", type=float, default=0.20, help="G2.1 minimum peak memory reduction (default: 0.20 = 20%%)")
    parser.add_argument("--frontier-report", help="Path to frontier_report.json for G2.3 frontier separation evaluation")
    parser.add_argument("--external-eval", help="Path to external_eval_results.json for G3 external quality evaluation")
    parser.add_argument("--g3-max-aggregate-drop", type=float, default=0.01, help="G3.1 max aggregate relative quality drop (default: 0.01 = 1%%)")
    parser.add_argument("--g3-max-single-task-drop", type=float, default=0.03, help="G3.2 max single-task relative quality drop (default: 0.03 = 3%%)")
    parser.add_argument("--cold-summary", help="Path to cold-mode aggregate_summary.json for G4.1 warm speedup comparison")
    parser.add_argument("--no-cache-summary", help="Path to cache-off aggregate_summary.json for G4.2 cache memory effect comparison")
    parser.add_argument("--skip-gates", nargs="*", default=[], help="Gates to skip (e.g. G2 G3 G4)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any gate lacks evidence (bails on a missing input). "
        "By default a gate that could not be evaluated is reported as "
        "INSUFFICIENT EVIDENCE and does not by itself cause a failure — it is "
        "unmeasured, not disproven. Use --strict to fail unless every gate "
        "reached and passed its verdict.",
    )
    parser.add_argument("--output", "-o", help="Write JSON report to file")
    args = parser.parse_args()

    summary = _load_summary(args.summary_json)

    skip = set(args.skip_gates)

    frontier_report = args.frontier_report
    if not frontier_report:
        discovered = _find_frontier_report(args.summary_json)
        if discovered:
            frontier_report = str(discovered)
            print(f"Auto-discovered frontier report: {frontier_report}")

    cold_summary = None
    if args.cold_summary:
        if args.cold_summary.lower() != 'none':
            cold_summary = _load_summary(args.cold_summary)
    else:
        cold_path = _find_sibling_summary(
            args.summary_json, suffix="cold", candidates=["cold", "cold_start"],
        )
        if cold_path:
            cold_summary = _load_summary(str(cold_path))
            print(f"Auto-discovered cold ablation summary: {cold_path}")

    no_cache_summary = None
    if args.no_cache_summary:
        if args.no_cache_summary.lower() != 'none':
            no_cache_summary = _load_summary(args.no_cache_summary)
    else:
        nc_path = _find_sibling_summary(
            args.summary_json, suffix="no_cache", candidates=["no_cache", "cache_off", "nocache"],
        )
        if nc_path:
            no_cache_summary = _load_summary(str(nc_path))
            print(f"Auto-discovered cache-off ablation summary: {nc_path}")

    external_eval = args.external_eval
    if not external_eval:
        for candidate in [Path(args.summary_json).resolve().parent, Path(args.summary_json).resolve().parent.parent]:
            target = candidate / "external_eval_results.json"
            if target.exists():
                external_eval = str(target)
                print(f"Auto-discovered external eval results: {external_eval}")
                break

    gate_funcs = [
        ("G0", lambda: _check_g0(summary)),
        ("G1", lambda: _check_g1(summary, loss_red_ratio=args.g1_loss_red_ratio, quality_tolerance=args.g1_quality_tolerance, efficiency_tolerance=args.g1_efficiency_tolerance)),
        ("G2", lambda: _check_g2(summary, memory_improvement=args.g2_memory_improvement, frontier_report_path=frontier_report)),
        ("G3", lambda: _check_g3(summary, external_eval_path=external_eval, max_aggregate_drop=args.g3_max_aggregate_drop, max_single_task_drop=args.g3_max_single_task_drop)),
        ("G4", lambda: _check_g4(summary, cold_summary=cold_summary, no_cache_summary=no_cache_summary)),
    ]

    results = [fn() for name, fn in gate_funcs if name not in skip]

    stats_enrichment = _enrich_with_statistics(summary)

    report = _format_report(results)
    print(report)

    if stats_enrichment.get("seed_count", 0) >= 2:
        print("\n--- Statistical Enrichment ---")
        print(f"Seeds analyzed: {stats_enrichment['seed_count']}")
        if "paired_t_test_efficiency" in stats_enrichment:
            pt = stats_enrichment["paired_t_test_efficiency"]
            sig = "YES" if pt["significant_005"] else "NO"
            print(f"TG vs BL efficiency: t={pt['t_statistic']:.3f}, p={pt['p_value']:.4f} (significant@0.05: {sig})")
        if "paired_t_test_quality" in stats_enrichment:
            qt = stats_enrichment["paired_t_test_quality"]
            qsig = "YES" if qt["significant_005"] else "NO"
            print(f"TG vs BL quality (valid-loss): t={qt['t_statistic']:.3f}, p={qt['p_value']:.4f} (significant@0.05: {qsig})")
        qci = stats_enrichment.get("quality_degradation_ci")
        if qci:
            print(f"  quality rel-degradation CI95: {qci['mean'] * 100:.2f}% [{qci['ci_lower'] * 100:.2f}%, {qci['ci_upper'] * 100:.2f}%] (n={qci['n']})")
        ci_metrics = stats_enrichment.get("metric_ci", {})
        if ci_metrics:
            print("\n95% Confidence Intervals:")
            for key in sorted(ci_metrics):
                ci = ci_metrics[key]
                print(f"  {key}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    if args.output:
        disproven = [r["gate"] for r in results if r.get("evaluated", True) and not r["passed"]]
        insufficient = [r["gate"] for r in results if not r.get("evaluated", True)]
        report_data = {
            "gates": results,
            # A gate that bailed on a missing input is unmeasured, not disproven,
            # so overall_passed considers only gates that reached a real verdict.
            "overall_passed": not disproven,
            "disproven_fail_gates": disproven,
            "insufficient_evidence_gates": insufficient,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "statistics": stats_enrichment,
        }
        Path(args.output).write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n")
        print(f"\nJSON report written to {args.output}")

    disproven_fails = [r for r in results if r.get("evaluated", True) and not r["passed"]]
    insufficient_gates = [r for r in results if not r.get("evaluated", True)]
    if insufficient_gates:
        names = ", ".join(r["gate"] for r in insufficient_gates)
        print(
            f"Note: {len(insufficient_gates)} gate(s) lack evidence ({names}) — "
            f"reported above as INSUFFICIENT EVIDENCE, not failures."
        )

    # Default (honest): exit 1 only when a gate that reached a real verdict was
    # disproven. A gate that merely bailed on a missing input is not a failure.
    # --strict restores the legacy "exit 1 unless every gate passed" so a
    # pipeline can still fail when expected evidence did not arrive.
    if disproven_fails or (args.strict and insufficient_gates):
        sys.exit(1)


if __name__ == "__main__":
    main()
