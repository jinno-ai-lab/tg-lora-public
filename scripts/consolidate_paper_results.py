#!/usr/bin/env python
"""Consolidate Stage 2-5 experiment results into paper-ready tables and summaries.

Reads gate evaluation output, frontier report, and aggregate summaries to produce:
- Claim Ladder determination (C0 / C1 / C2)
- Markdown summary table
- LaTeX table for paper inclusion
- Gate PASS / FAIL / INSUFFICIENT overview

Gate results carry an ``evaluated`` flag (set by evaluate_paper_gates when a
gate bails on a missing input). An un-evaluated gate is reported as
``INSUFFICIENT`` — its claim is *unmeasured*, not disproven — and the claim
blocker analysis distinguishes "refuted by evidence" from "pending evidence"
so a run that has not yet produced a gate's input cannot read as a refuted
claim. When every gate is evaluated, output is byte-identical to the
historical pass/fail report.

Usage::

    python scripts/consolidate_paper_results.py \\
        --gate-report runs/.../gate_report.json \\
        --summary runs/.../aggregate_summary.json \\
        --frontier-report runs/.../frontier_report.json \\
        --output-dir paper_output/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def determine_claim_level(gate_results: list[dict[str, Any]]) -> str:
    """Determine the highest achievable claim level from gate results.

    C0: G1 passed (internal efficiency)
    C1: G1 + G3 passed (efficiency + quality retention)
    C2: G1 + G2 + G3 passed (efficiency + frontier separation + quality)
    """
    passed_gates = {r["gate"] for r in gate_results if r.get("passed")}

    if "G1" in passed_gates and "G2" in passed_gates and "G3" in passed_gates:
        return "C2"
    if "G1" in passed_gates and "G3" in passed_gates:
        return "C1"
    if "G1" in passed_gates:
        return "C0"
    return "none"


# Claim ladder in ascending order: each level subsumes its predecessor's gates.
_CLAIM_LADDER: list[tuple[str, tuple[str, ...]]] = [
    ("C0", ("G1",)),
    ("C1", ("G1", "G3")),
    ("C2", ("G1", "G2", "G3")),
]

# Map a 3-state outcome to its human-readable label (mirrors
# evaluate_paper_gates._format_report's PASS / FAIL / INSUFFICIENT EVIDENCE).
_STATUS_LABEL: dict[str, str] = {
    "pass": "PASS",
    "fail": "FAIL",
    "insufficient": "INSUFFICIENT",
}


def _gate_outcome(result: dict[str, Any]) -> str:
    """Three-state outcome for a single gate, mirroring evaluate_paper_gates.

    Returns ``"insufficient"`` for a gate that bailed on a missing input
    (``evaluated is False``) — its underlying claim is *unmeasured*, not
    disproven — ``"pass"`` for an evaluated passing gate, and ``"fail"`` for an
    evaluated gate whose claim was tested and refuted. When every gate is
    evaluated (the historical case pre-TASK-0144) this collapses to
    pass/fail, so existing reports stay byte-identical.
    """
    if not result.get("evaluated", True):
        return "insufficient"
    return "pass" if result.get("passed") else "fail"


def _next_claim_level(achieved: str) -> str | None:
    """Return the claim level strictly above ``achieved``, or None at the top."""
    levels = [level for level, _ in _CLAIM_LADDER]
    if achieved not in levels:
        return "C0"  # achieved == "none"
    idx = levels.index(achieved)
    return levels[idx + 1] if idx + 1 < len(levels) else None


def claim_blockers(gate_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify why the next claim level above the achieved one is not reached.

    For the next claim level, lists its required gates that are not passed,
    split into ``disproven`` (tested and refuted) and ``insufficient``
    (unmeasured). This is the consolidate-side analog of evaluate_paper_gates'
    INSUFFICIENT EVIDENCE third state: a claim held back by an unmeasured gate
    must not read as a refuted claim. A gate absent from ``gate_results`` is
    treated as insufficient (its evidence was never produced).
    """
    achieved = determine_claim_level(gate_results)
    nxt = _next_claim_level(achieved)
    by_gate = {r["gate"]: r for r in gate_results}
    disproven: list[str] = []
    insufficient: list[str] = []
    if nxt is not None:
        required = dict(_CLAIM_LADDER)[nxt]
        for gate in required:
            result = by_gate.get(gate)
            if result is not None and result.get("passed"):
                continue
            # A required gate absent from the results is evidence never produced
            # -> insufficient (unmeasured), not disproven.
            if result is None or _gate_outcome(result) == "insufficient":
                insufficient.append(gate)
            else:
                disproven.append(gate)
    return {
        "achieved": achieved,
        "next": nxt,
        "disproven": disproven,
        "insufficient": insufficient,
    }


def _gate_status(gate_results: list[dict[str, Any]], gate: str) -> str:
    for r in gate_results:
        if r["gate"] == gate:
            return _STATUS_LABEL[_gate_outcome(r)]
    return "SKIP"


def generate_markdown_table(
    gate_results: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    frontier: dict[str, Any] | None,
) -> str:
    """Generate a Markdown summary table of all results."""
    lines: list[str] = []
    lines.append("# Paper Results Summary")
    lines.append("")

    claim = determine_claim_level(gate_results)
    lines.append(f"**Claim Level**: {claim}")
    lines.append("")

    # Gate overview
    lines.append("## Gate Overview")
    lines.append("")
    lines.append("| Gate | Name | Status |")
    lines.append("|------|------|--------|")
    for r in gate_results:
        status = _STATUS_LABEL[_gate_outcome(r)]
        lines.append(f"| {r['gate']} | {r['name']} | {status} |")
    lines.append("")

    # Aggregate metrics
    if summary:
        agg = summary.get("aggregate", {})
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Baseline | TG-LoRA | Improvement |")
        lines.append("|--------|----------|---------|-------------|")

        for tg_key, bl_key, label, unit in [
            ("warm_tg_loss_red_per_wall_minute", "warm_baseline_loss_red_per_wall_minute", "Loss red / wall-min", ""),
            ("warm_tg_best_valid_loss", "warm_baseline_best_valid_loss", "Best valid loss", ""),
            ("warm_tg_gpu_peak_mb", "warm_baseline_gpu_peak_mb", "GPU peak (MB)", " MB"),
            ("warm_tg_runtime_offload_gpu_freed_mb", None, "Runtime offload freed", " MB"),
        ]:
            tg_val = agg.get(tg_key, {}).get("mean")
            bl_val = agg.get(bl_key, {}).get("mean") if bl_key else None
            if tg_val is not None and bl_val is not None and bl_val != 0:
                if "peak_mb" in tg_key:
                    imp = (bl_val - tg_val) / bl_val * 100
                    imp_str = f"{imp:+.1f}%"
                else:
                    ratio = tg_val / bl_val
                    imp_str = f"{ratio:.2f}x"
                lines.append(f"| {label} | {bl_val:.4f}{unit} | {tg_val:.4f}{unit} | {imp_str} |")
            elif tg_val is not None:
                lines.append(f"| {label} | — | {tg_val:.1f}{unit} | — |")
        lines.append("")

    # Frontier separation
    if frontier:
        lines.append("## Frontier Separation")
        lines.append("")
        boundary = frontier.get("frontier_boundary")
        detected = frontier.get("frontier_separation_detected", False)
        avg_savings = frontier.get("avg_memory_savings_pct")
        lines.append(f"- **Boundary**: {boundary}")
        lines.append(f"- **Detected**: {detected}")
        if avg_savings is not None:
            lines.append(f"- **Avg memory savings**: {avg_savings:.1f}%")
        lines.append("")

        if frontier.get("runs"):
            lines.append("### Per-Sequence Results")
            lines.append("")
            lines.append("| Seq Len | Baseline | TG-LoRA | Frontier | Peak Delta |")
            lines.append("|---------|----------|---------|----------|------------|")
            for run in frontier["runs"]:
                fs = "Yes" if run.get("frontier_separation") else "No"
                delta = run.get("memory_delta_mb")
                delta_str = f"{delta:.0f} MB" if delta is not None else "—"
                lines.append(
                    f"| {run['seq_len']} | {run['baseline_status']} | {run['tg_status']} "
                    f"| {fs} | {delta_str} |"
                )
            lines.append("")

    # Claim Ladder explanation
    lines.append("## Claim Ladder")
    lines.append("")
    lines.append("- **C0** (safe paper): G1 passed → internal efficiency improvement")
    lines.append("- **C1** (strong paper): G1 + G3 → efficiency + quality retention")
    lines.append("- **C2** (revolutionary paper): G1 + G2 + G3 → frontier separation")
    lines.append("")
    lines.append(f"**Current status: {claim}**")
    lines.append("")

    # Honest provenance for why the next claim is not reached: distinguish
    # "refuted by evidence" (disproven) from "not yet measured" (insufficient).
    blockers = claim_blockers(gate_results)
    if blockers["next"] is not None:
        if blockers["disproven"]:
            lines.append(
                f"- Next claim **{blockers['next']}** is blocked by disproven "
                f"evidence: {', '.join(blockers['disproven'])}"
            )
        if blockers["insufficient"]:
            lines.append(
                f"- Next claim **{blockers['next']}** is pending evidence "
                f"(unmeasured, not refuted): {', '.join(blockers['insufficient'])}"
            )
        if not blockers["disproven"] and not blockers["insufficient"]:
            lines.append(
                f"- Next claim **{blockers['next']}** is reachable once its "
                "remaining gates pass"
            )
        lines.append("")

    return "\n".join(lines)


def generate_latex_table(
    gate_results: list[dict[str, Any]],
    summary: dict[str, Any] | None,
) -> str:
    """Generate a LaTeX table of aggregate metrics for paper inclusion."""
    if not summary:
        return "% No aggregate summary available"

    agg = summary.get("aggregate", {})
    seeds = summary.get("seeds", [])
    n_seeds = len(seeds)

    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{TG-LoRA vs Baseline Performance Summary}")
    lines.append("\\label{tab:results}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Metric & Baseline & TG-LoRA \\\\")
    lines.append("\\midrule")

    metrics = [
        ("Loss red / wall-min", "warm_baseline_loss_red_per_wall_minute", "warm_tg_loss_red_per_wall_minute"),
        ("Best valid loss", "warm_baseline_best_valid_loss", "warm_tg_best_valid_loss"),
        ("GPU peak (MB)", "warm_baseline_gpu_peak_mb", "warm_tg_gpu_peak_mb"),
    ]

    for label, bl_key, tg_key in metrics:
        bl_val = agg.get(bl_key, {}).get("mean")
        tg_val = agg.get(tg_key, {}).get("mean")
        if bl_val is not None and tg_val is not None:
            lines.append(f"{label} & {bl_val:.2f} & {tg_val:.2f} \\\\")

    # Runtime offload
    freed = agg.get("warm_tg_runtime_offload_gpu_freed_mb", {}).get("mean")
    if freed is not None:
        lines.append(f"Offload freed (MB) & — & {freed:.0f} \\\\")

    lines.append("\\midrule")
    lines.append(f"\\# Seeds & \\multicolumn{{2}}{{c}}{{{n_seeds}}} \\\\")

    claim = determine_claim_level(gate_results)
    claim_labels = {"C0": "Safe", "C1": "Strong", "C2": "Revolutionary"}
    lines.append(f"Claim Level & \\multicolumn{{2}}{{c}}{{{claim_labels.get(claim, 'None')}}} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _find_sibling_json(gate_path: str | Path, filename: str) -> Path | None:
    """Search for a sibling JSON file relative to the gate report path.

    Checks (in order):
      1. Same directory as gate report
      2. Parent directory (for nested layouts like suite_dir/gate_report.json)
    """
    p = Path(gate_path).resolve()
    for candidate in [p.parent, p.parent.parent]:
        target = candidate / filename
        if target.exists():
            return target
    return None


def build_consolidated_report(
    gate_results: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    frontier: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the full consolidated report as a structured dict."""
    claim = determine_claim_level(gate_results)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_level": claim,
        "claim_descriptions": {
            "C0": "G1 passed — internal efficiency improvement",
            "C1": "G1 + G3 passed — efficiency + quality retention",
            "C2": "G1 + G2 + G3 passed — frontier separation + efficiency + quality",
        },
        "gates": {
            r["gate"]: {
                "name": r["name"],
                "passed": r["passed"],
                "evaluated": r.get("evaluated", True),
                "status": _gate_outcome(r),
            }
            for r in gate_results
        },
        "gate_summary": {
            "total": len(gate_results),
            "passed": sum(1 for r in gate_results if r["passed"]),
            "failed": sum(1 for r in gate_results if not r["passed"]),
            "disproven": sum(1 for r in gate_results if _gate_outcome(r) == "fail"),
            "insufficient": sum(
                1 for r in gate_results if _gate_outcome(r) == "insufficient"
            ),
        },
        "claim_blockers": claim_blockers(gate_results),
    }

    if summary:
        agg = summary.get("aggregate", {})
        report["aggregate_metrics"] = agg
        report["n_seeds"] = len(summary.get("seeds", []))

    if frontier:
        report["frontier"] = {
            "boundary": frontier.get("frontier_boundary"),
            "detected": frontier.get("frontier_separation_detected", False),
            "avg_memory_savings_pct": frontier.get("avg_memory_savings_pct"),
            "n_seq_lens": len(frontier.get("seq_lens", [])),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate paper experiment results into tables and summaries",
    )
    parser.add_argument(
        "--gate-report",
        required=True,
        help="Path to gate_report.json from evaluate_paper_gates.py",
    )
    parser.add_argument("--summary", help="Path to aggregate_summary.json")
    parser.add_argument("--frontier-report", help="Path to frontier_report.json")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="paper_output",
        help="Output directory for generated files (default: paper_output/)",
    )
    args = parser.parse_args()

    # Load gate report
    gate_path = Path(args.gate_report)
    if not gate_path.exists():
        print(f"ERROR: gate report not found: {gate_path}", file=sys.stderr)
        sys.exit(2)
    gate_data = json.loads(gate_path.read_text())
    gate_results = gate_data.get("gates", [])

    # Load aggregate summary (explicit or auto-discovered)
    summary: dict[str, Any] | None = None
    summary_path = args.summary
    if not summary_path:
        discovered = _find_sibling_json(args.gate_report, "aggregate_summary.json")
        if discovered:
            summary_path = str(discovered)
            print(f"Auto-discovered aggregate summary: {discovered}")
    if summary_path:
        s_path = Path(summary_path)
        if s_path.exists():
            summary = json.loads(s_path.read_text())
        else:
            print(f"WARNING: summary not found: {s_path}", file=sys.stderr)

    # Load frontier report (explicit or auto-discovered)
    frontier: dict[str, Any] | None = None
    frontier_path = args.frontier_report
    if not frontier_path:
        discovered = _find_sibling_json(args.gate_report, "frontier_report.json")
        if discovered:
            frontier_path = str(discovered)
            print(f"Auto-discovered frontier report: {discovered}")
    if frontier_path:
        f_path = Path(frontier_path)
        if f_path.exists():
            frontier = json.loads(f_path.read_text())
        else:
            print(f"WARNING: frontier report not found: {f_path}", file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    claim = determine_claim_level(gate_results)
    print(f"Claim Level: {claim}")

    # Markdown table
    md = generate_markdown_table(gate_results, summary, frontier)
    md_path = out_dir / "paper_results_summary.md"
    md_path.write_text(md)
    print(f"Markdown summary: {md_path}")

    # LaTeX table
    latex = generate_latex_table(gate_results, summary)
    latex_path = out_dir / "paper_results_table.tex"
    latex_path.write_text(latex)
    print(f"LaTeX table: {latex_path}")

    # Consolidated JSON
    report = build_consolidated_report(gate_results, summary, frontier)
    json_path = out_dir / "consolidated_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Consolidated report: {json_path}")

    # Gate summary
    print("\nGate Summary:")
    for r in gate_results:
        status = _STATUS_LABEL[_gate_outcome(r)]
        print(f"  {r['gate']}: {r['name']} — {status}")

    claim_labels = {"C0": "Safe paper", "C1": "Strong paper", "C2": "Revolutionary paper"}
    print(f"\nClaim Level: {claim} ({claim_labels.get(claim, 'No claim')})")


if __name__ == "__main__":
    main()
