#!/usr/bin/env python
"""Analyze accel param sweep results and generate summary report.

Usage:
    python scripts/analyze_accel_sweep.py <sweep_dir>

    sweep_dir should be the timestamped directory under reports/accel_sweep/
    containing the 4 config run directories.

Output:
    - reports/<sweep_dir>/analysis/summary.md
    - reports/<sweep_dir>/analysis/ranking.json
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

from scripts.compare_runs import find_best_run, gather_runs


def analyze_sweep(sweep_dir: Path) -> dict:
    """Analyze sweep results and return structured summary."""
    runs = gather_runs(sweep_dir)
    if not runs:
        print(f"No runs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    # Augment runs with step-level metrics when footer is missing (in-progress runs)
    from src.utils.run_query import parse_jsonl

    for run in runs:
        jsonl_path = run.get("_jsonl_path")
        if not jsonl_path:
            run["_parsed_records"] = []
            continue
        records = parse_jsonl(jsonl_path)
        steps = [r for r in records if r.get("type") == "step"]
        run["_parsed_records"] = records

        if run.get("best_valid_loss") is None and steps:
            valid_losses = [r.get("loss_valid") for r in steps if r.get("loss_valid") is not None]
            if valid_losses:
                run["best_valid_loss"] = min(valid_losses)
            run["final_train_loss"] = steps[-1].get("loss_train")
            run["total_backward_passes"] = steps[-1].get("total_backward_passes")
            run["total_wall_seconds"] = steps[-1].get("elapsed_seconds")

        run["_convergence"] = compute_loss_trajectory(records)

    # Identify baseline
    baseline = None
    treatments = []
    for run in runs:
        run_id = run.get("run_id", "")
        if "no_accel" in run_id:
            baseline = run
        else:
            treatments.append(run)

    if baseline is None:
        baseline = runs[0]
        treatments = runs[1:]

    best = find_best_run(runs)

    # Use build_comparison_table for efficiency metrics
    from scripts.compare_runs import build_comparison_table

    comp_rows = build_comparison_table(runs)
    row_by_id = {r["run_id"]: r for r in comp_rows}

    # Compute acceptance rate for each run (reuses parsed records from augmentation)
    def _accept_rate(run: dict) -> float | None:
        records = run.get("_parsed_records", [])
        steps = [r for r in records if r.get("type") == "step"]
        if not steps:
            return None
        accepted = sum(1 for s in steps if s.get("tg_lora_accepted"))
        return accepted / len(steps)

    for run in runs:
        run["_accept_rate"] = _accept_rate(run)

    # Compute pairwise deltas
    pairwise = []
    baseline_loss = baseline.get("best_valid_loss") or float("inf")
    bl_accept = baseline.get("_accept_rate")
    for t in treatments:
        t_loss = t.get("best_valid_loss") or float("inf")
        delta = t_loss - baseline_loss
        delta_pct = (delta / baseline_loss * 100) if baseline_loss != 0 and math.isfinite(baseline_loss) else 0
        t_row = row_by_id.get(t.get("run_id", ""), {})
        t_accept = t.get("_accept_rate")
        pairwise.append({
            "run_id": t.get("run_id", "?"),
            "decay": t.get("accel_instability_lr_decay"),
            "boost": t.get("accel_convergence_lr_boost"),
            "best_valid_loss": t_loss,
            "delta_vs_baseline": round(delta, 6),
            "delta_pct": round(delta_pct, 2),
            "efficiency_per_bp": t_row.get("loss_red_per_bp"),
            "loss_red_per_wall_min": t_row.get("loss_red_per_wall_min"),
            "loss_reduction": t_row.get("loss_reduction"),
            "total_backward_passes": t.get("total_backward_passes", 0),
            "wall_seconds": t.get("total_wall_seconds", 0),
            "accept_rate": t_accept,
            "accept_rate_delta": (
                round(t_accept - bl_accept, 4)
                if t_accept is not None and bl_accept is not None
                else None
            ),
            "convergence": t.get("_convergence", {}),
        })

    # Sort by loss (best first)
    pairwise.sort(key=lambda x: x["best_valid_loss"])

    bl_row = row_by_id.get(baseline.get("run_id", ""), {})
    best_row = row_by_id.get(best.get("run_id", ""), {}) if best else {}

    return {
        "baseline": {
            "run_id": baseline.get("run_id", "?"),
            "best_valid_loss": baseline.get("best_valid_loss"),
            "total_backward_passes": baseline.get("total_backward_passes", 0),
            "wall_seconds": baseline.get("total_wall_seconds", 0),
            "loss_reduction": bl_row.get("loss_reduction"),
            "loss_red_per_bp": bl_row.get("loss_red_per_bp"),
            "loss_red_per_wall_min": bl_row.get("loss_red_per_wall_min"),
            "accept_rate": bl_accept,
            "convergence": baseline.get("_convergence", {}),
        },
        "best_run": {
            "run_id": best.get("run_id", "?") if best else None,
            "best_valid_loss": best.get("best_valid_loss") if best else None,
            "loss_red_per_bp": best_row.get("loss_red_per_bp"),
            "loss_red_per_wall_min": best_row.get("loss_red_per_wall_min"),
            "accept_rate": best.get("_accept_rate") if best else None,
            "convergence": best.get("_convergence", {}) if best else {},
        },
        "pairwise": pairwise,
        "total_runs": len(runs),
    }


def generate_summary(analysis: dict) -> str:
    """Generate markdown summary report."""
    md = []
    md.append("# Accel Param Sweep Analysis\n")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md.append(f"_Runs analyzed: {analysis['total_runs']}_\n")

    # Best result
    best = analysis["best_run"]
    md.append("## Optimal Config\n")
    md.append(f"**Best run:** `{best['run_id']}`")
    loss_val = best.get("best_valid_loss")
    md.append(f"**Best valid loss:** {loss_val:.6f}\n" if loss_val is not None else "**Best valid loss:** N/A\n")

    # Baseline
    bl = analysis["baseline"]
    md.append("## Baseline (no_accel)\n")
    md.append(f"- Run ID: `{bl['run_id']}`")
    bl_loss = bl.get("best_valid_loss")
    md.append(f"- Best valid loss: {bl_loss:.6f}" if bl_loss is not None else "- Best valid loss: N/A")
    bl_ar = bl.get("accept_rate")
    md.append(f"- Accept rate: {bl_ar:.1%}" if bl_ar is not None else "- Accept rate: N/A")
    md.append(f"- Backward passes: {bl['total_backward_passes']}")
    md.append(f"- Wall time: {bl['wall_seconds']:.0f}s\n")

    # Pairwise results
    md.append("## Pairwise Comparison (treatment vs no_accel)\n")
    md.append("| Config | decay | boost | Loss | Δ vs baseline | Δ% | Eff/bp | Loss red/min | Accept% | Δ Accept |")
    md.append("|--------|-------|-------|------|---------------|-----|--------|-------------|---------|----------|")
    for p in analysis["pairwise"]:
        delta_str = f"{p['delta_vs_baseline']:+.6f}"
        pct_str = f"{p['delta_pct']:+.2f}%"
        eff_str = f"{p['efficiency_per_bp']:.2e}" if p.get("efficiency_per_bp") is not None else "N/A"
        lrm_str = f"{p['loss_red_per_wall_min']:.5f}" if p.get("loss_red_per_wall_min") is not None else "N/A"
        ar_str = f"{p['accept_rate']:.1%}" if p.get("accept_rate") is not None else "N/A"
        ar_delta = p.get("accept_rate_delta")
        ar_delta_str = f"{ar_delta:+.1%}" if ar_delta is not None else "N/A"
        md.append(
            f"| {p['run_id']} | {p['decay']} | {p['boost']} | "
            f"{p['best_valid_loss']:.6f} | {delta_str} | {pct_str} | {eff_str} | {lrm_str} | {ar_str} | {ar_delta_str} |"
        )
    md.append("")

    # Ranking
    md.append("## Ranking (by best valid loss)\n")
    for i, p in enumerate(analysis["pairwise"], 1):
        marker = " **<- best**" if p["run_id"] == best["run_id"] else ""
        md.append(f"{i}. `{p['run_id']}` — loss={p['best_valid_loss']:.6f}{marker}")
    md.append("")

    # Next actions
    md.append("## Next Actions\n")
    best_treatment = analysis["pairwise"][0] if analysis["pairwise"] else None
    bl_eff = analysis["baseline"].get("loss_red_per_wall_min")
    if best_treatment and best_treatment["delta_vs_baseline"] < 0:
        md.append(f"- **Improvement found**: `{best_treatment['run_id']}` reduces loss by "
                  f"{abs(best_treatment['delta_pct']):.2f}%")
        best_eff = best_treatment.get("loss_red_per_wall_min")
        if best_eff is not None and bl_eff is not None and bl_eff > 0:
            eff_ratio = best_eff / bl_eff
            md.append(f"- **Efficiency**: loss_red_per_wall_min = {best_eff:.5f} "
                      f"({eff_ratio:.1f}x baseline)")
        best_ar = best_treatment.get("accept_rate")
        bl_ar = analysis["baseline"].get("accept_rate")
        if best_ar is not None:
            md.append(f"- **Accept rate**: {best_ar:.1%}" +
                      (f" (baseline: {bl_ar:.1%})" if bl_ar is not None else ""))
        md.append("- Proceed with lm-evaluation-harness on the best config")
        md.append("- Compare TruthfulQA / ARC / HellaSwag accuracy against baseline")
    elif best_treatment and best_treatment["delta_vs_baseline"] > 0:
        md.append("- **No improvement**: all treatments worse than baseline")
        md.append("- Consider different accel param ranges or disabling accel adaptation")
    else:
        md.append("- Results are neutral — no significant difference from baseline")
        md.append("- Consider expanding the parameter grid or increasing training cycles")
    md.append("")

    # Convergence trajectory
    bl_conv = analysis.get("baseline", {}).get("convergence", {})
    if bl_conv and bl_conv.get("slope") is not None:
        md.append("## Convergence Trajectory\n")
        md.append("| Config | Slope | Plateau Start | Half-Red Cycle | Speed (first 25%) |")
        md.append("|--------|-------|---------------|----------------|-------------------|")
        bl_slope = bl_conv.get("slope")
        bl_plateau = bl_conv.get("plateau_start")
        bl_half = bl_conv.get("half_reduction_cycle")
        bl_speed = bl_conv.get("convergence_speed")
        md.append(
            f"| {analysis['baseline']['run_id']} (baseline) | "
            f"{bl_slope:.6f} | {bl_plateau or 'N/A'} | {bl_half or 'N/A'} | "
            f"{f'{bl_speed:.1%}' if bl_speed is not None else 'N/A'} |"
        )
        for p in analysis["pairwise"]:
            conv = p.get("convergence", {})
            if not conv or conv.get("slope") is None:
                continue
            slope = conv.get("slope", 0)
            plateau = conv.get("plateau_start")
            half = conv.get("half_reduction_cycle")
            speed = conv.get("convergence_speed")
            md.append(
                f"| {p['run_id']} | {slope:.6f} | {plateau or 'N/A'} | "
                f"{half or 'N/A'} | {f'{speed:.1%}' if speed is not None else 'N/A'} |"
            )
        md.append("")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Convergence trajectory analysis
# ---------------------------------------------------------------------------


def compute_loss_trajectory(records: list[dict]) -> dict:
    """Compute convergence metrics from step-level records.

    Returns dict with:
        - slope: linear regression slope of valid loss over cycles
        - plateau_start: first cycle where loss stalled (or None)
        - convergence_speed: fraction of total loss reduction in first 25% of training
        - total_reduction: initial_loss - final_loss
        - half_reduction_cycle: cycle at which >=50% of total reduction was achieved
    """
    steps = [r for r in records if r.get("type") == "step"]
    if len(steps) < 2:
        return {
            "slope": None,
            "plateau_start": None,
            "convergence_speed": None,
            "total_reduction": None,
            "half_reduction_cycle": None,
        }

    losses = []
    for s in steps:
        vl = s.get("loss_valid")
        if vl is not None and math.isfinite(vl):
            losses.append((s.get("cycle", len(losses) + 1), vl))

    if len(losses) < 2:
        return {
            "slope": None,
            "plateau_start": None,
            "convergence_speed": None,
            "total_reduction": None,
            "half_reduction_cycle": None,
        }

    cycles = [c for c, _ in losses]
    vals = [v for _, v in losses]
    n = len(vals)

    # Linear regression slope
    mean_x = sum(cycles) / n
    mean_y = sum(vals) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in losses)
    ss_xx = sum((x - mean_x) ** 2 for x in cycles)
    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0

    # Total reduction
    initial = vals[0]
    final = vals[-1]
    total_reduction = initial - final

    # Convergence speed: fraction of total reduction in first 25% of cycles
    quarter = max(2, n // 4)
    quarter_loss = vals[min(quarter, n) - 1]
    quarter_reduction = initial - quarter_loss
    convergence_speed = (
        quarter_reduction / total_reduction
        if total_reduction and total_reduction > 0
        else None
    )

    # Plateau detection: first cycle where improvement < threshold for 5+ consecutive steps
    plateau_start = _detect_plateau(vals, cycles, window=5, threshold=0.001)

    # Half-reduction cycle
    half_reduction_cycle = None
    if total_reduction > 0:
        target = initial - total_reduction * 0.5
        for c, v in losses:
            if v <= target:
                half_reduction_cycle = c
                break

    return {
        "slope": slope,
        "plateau_start": plateau_start,
        "convergence_speed": convergence_speed,
        "total_reduction": total_reduction,
        "half_reduction_cycle": half_reduction_cycle,
    }


def _detect_plateau(
    losses: list[float],
    cycles: list[int],
    window: int = 5,
    threshold: float = 0.001,
) -> int | None:
    """Return the first cycle where loss improvement stalls for `window` steps."""
    for i in range(len(losses) - window):
        segment = losses[i : i + window + 1]
        improvement = segment[0] - segment[-1]
        if improvement < threshold:
            return cycles[i]
    return None


# ---------------------------------------------------------------------------
# Sweep result validation
# ---------------------------------------------------------------------------


def validate_sweep_results(sweep_dir: Path) -> dict:
    """Validate sweep results integrity before analysis.

    Returns dict with:
        - valid: bool
        - errors: list[str]
        - warnings: list[str]
        - runs_checked: int
        - run_details: dict[str, dict]
    """
    errors: list[str] = []
    warnings: list[str] = []
    run_details: dict[str, dict] = {}

    if not sweep_dir.is_dir():
        return {
            "valid": False,
            "errors": [f"Sweep directory not found: {sweep_dir}"],
            "warnings": [],
            "runs_checked": 0,
            "run_details": {},
        }

    metrics_files = sorted(sweep_dir.glob("*/run_metrics.jsonl"))
    if not metrics_files:
        return {
            "valid": False,
            "errors": ["No run_metrics.jsonl files found in sweep directory"],
            "warnings": [],
            "runs_checked": 0,
            "run_details": {},
        }

    for mf in metrics_files:
        run_id = mf.parent.name
        run_errors: list[str] = []
        run_warnings: list[str] = []

        try:
            from src.utils.run_query import parse_jsonl
            records = parse_jsonl(mf)
        except Exception as e:
            run_errors.append(f"Failed to parse: {e}")
            run_details[run_id] = {"errors": run_errors, "warnings": run_warnings}
            continue

        header = next((r for r in records if r.get("type") == "run_header"), None)
        footer = next((r for r in records if r.get("type") == "run_footer"), None)
        steps = [r for r in records if r.get("type") == "step"]

        if header is None:
            run_errors.append("Missing run_header")
        if footer is None:
            run_warnings.append("Missing run_footer (run may be incomplete)")
        if not steps:
            run_errors.append("No step records found")
            run_details[run_id] = {"errors": run_errors, "warnings": run_warnings}
            continue

        # Check for NaN/Inf losses
        nan_steps = []
        for s in steps:
            lt = s.get("loss_train")
            lv = s.get("loss_valid")
            lt_bad = lt is None or not math.isfinite(lt)
            lv_bad = lv is None or not math.isfinite(lv)
            if lt_bad or lv_bad:
                nan_steps.append(s.get("cycle", "?"))
        if nan_steps:
            run_errors.append(f"Non-finite losses at cycles: {nan_steps[:5]}")

        # Check for loss explosion (loss > 10x initial)
        initial_loss = next(
            (s.get("loss_train") for s in steps if s.get("loss_train") is not None),
            None,
        )
        if initial_loss is not None and initial_loss > 0:
            explosions = [
                s.get("cycle", "?")
                for s in steps
                if s.get("loss_train") is not None and s["loss_train"] > initial_loss * 10
            ]
            if explosions:
                run_warnings.append(f"Loss explosion at cycles: {explosions[:5]}")

        run_details[run_id] = {"errors": run_errors, "warnings": run_warnings}

    # Collect all errors/warnings
    for rid, details in run_details.items():
        for e in details["errors"]:
            errors.append(f"[{rid}] {e}")
        for w in details["warnings"]:
            warnings.append(f"[{rid}] {w}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "runs_checked": len(metrics_files),
        "run_details": run_details,
    }


def validate_sweep_configs(config_paths: list[Path]) -> dict:
    """Pre-flight validation of sweep experiment configs.

    Verifies that all configs exist and only differ on decay/boost
    (experimental isolation).

    Returns dict with:
        - valid: bool
        - errors: list[str]
        - warnings: list[str]
    """
    from src.training.config_schema import load_and_validate_config

    errors: list[str] = []
    warnings: list[str] = []

    if len(config_paths) < 2:
        errors.append("Need at least 2 configs for sweep comparison")
        return {"valid": False, "errors": errors, "warnings": warnings}

    configs: dict[str, object] = {}
    for p in config_paths:
        if not p.exists():
            errors.append(f"Config not found: {p}")
            continue
        try:
            configs[p.name] = load_and_validate_config(p)
        except Exception as e:
            errors.append(f"Failed to parse {p.name}: {e}")

    if len(configs) < 2:
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    names = list(configs.keys())
    ref_name = names[0]
    ref = configs[ref_name]

    controlled_fields = [
        ("model.name_or_path", lambda c: c.model.name_or_path),
        ("lora.r", lambda c: c.lora.r),
        ("lora.alpha", lambda c: c.lora.alpha),
        ("training.max_cycles", lambda c: c.training.max_cycles),
        ("training.learning_rate", lambda c: c.training.learning_rate),
        ("tg_lora.K_initial", lambda c: c.tg_lora.K_initial),
        ("tg_lora.N_initial", lambda c: c.tg_lora.N_initial),
        ("tg_lora.alpha_initial", lambda c: c.tg_lora.alpha_initial),
        ("tg_lora.beta_initial", lambda c: c.tg_lora.beta_initial),
        ("tg_lora.enable_random_walk", lambda c: c.tg_lora.enable_random_walk),
        ("tg_lora.lr_explore_prob", lambda c: c.tg_lora.lr_explore_prob),
    ]

    ref_vals = {name: getter(ref) for name, getter in controlled_fields}

    for cfg_name in names[1:]:
        cfg = configs[cfg_name]
        for field_name, getter in controlled_fields:
            ref_val = ref_vals[field_name]
            cfg_val = getter(cfg)
            if ref_val != cfg_val:
                errors.append(
                    f"{cfg_name}: {field_name}={cfg_val} differs from "
                    f"{ref_name} ({ref_val})"
                )

    # Check that decay/boost actually differ across configs
    decay_boost = set()
    for cfg_name, cfg in configs.items():
        pair = (cfg.tg_lora.accel_instability_lr_decay, cfg.tg_lora.accel_convergence_lr_boost)
        if pair in decay_boost:
            warnings.append(
                f"Duplicate decay/boost pair ({pair[0]}, {pair[1]}) in {cfg_name}"
            )
        decay_boost.add(pair)

    if len(decay_boost) == 1:
        warnings.append("All configs have identical decay/boost — no experimental variation")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_accel_sweep.py <sweep_dir>")
        sys.exit(1)

    sweep_dir = Path(sys.argv[1])
    if not sweep_dir.exists():
        print(f"Directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_sweep(sweep_dir)

    out_dir = sweep_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = generate_summary(analysis)
    (out_dir / "summary.md").write_text(summary)
    (out_dir / "ranking.json").write_text(json.dumps(analysis, indent=2, default=str))

    print(summary)
    print(f"\nSummary saved to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
