"""Summarize PSA ablation results with GOAL §3.3 comparison.

Produces:
  - GOAL §3.3 decision table: plain LoRA vs LAWA vs PSA variants
  - Efficiency-ranked table across all γ × regime_reset configs
  - Pairwise comparison: regime-reset ON vs OFF at each γ level
  - Per-layer-type amplification/stability analysis (GOAL §4 step 2 diagnostics)
  - Regime transition statistics
  - Next-action recommendations
"""

import argparse
import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict | None:
    header, records, footer = {}, [], {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("type")
            if t == "run_header":
                header = obj
            elif t == "step":
                records.append(obj)
            elif t == "run_footer":
                footer = obj
    if not footer:
        return None
    return {"header": header, "records": records, "footer": footer}


def _compute_efficiency(run: dict) -> dict:
    initial = run["records"][0].get("loss_train") if run["records"] else None
    best_vl = run["footer"].get("best_valid_loss")
    total_bp = run["records"][-1].get("total_backward_passes", 0) if run["records"] else 0
    wall_sec = run["footer"].get("total_wall_seconds", 0)

    loss_reduction = (initial - best_vl) if (initial is not None and best_vl is not None) else None
    loss_red_per_bp = loss_reduction / total_bp if (loss_reduction is not None and total_bp > 0) else None
    loss_red_per_wall_min = (
        loss_reduction / (wall_sec / 60) if (loss_reduction is not None and wall_sec > 0) else None
    )
    return {
        "loss_reduction": loss_reduction,
        "loss_red_per_bp": loss_red_per_bp,
        "loss_red_per_wall_min": loss_red_per_wall_min,
    }


_KNOWN_LT_METRICS = ("count", "amp_mean", "amp_std", "prior_stability")


def _extract_psa_lt_stats(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Extract per-layer-type amplification/stability time series from run records.

    Returns dict mapping layer_type → {amp_mean: [...], prior_stability: [...]}.
    """
    lt_series: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        for key, val in rec.items():
            if not key.startswith("psa_lt_") or not isinstance(val, (int, float)):
                continue
            suffix = key[len("psa_lt_"):]  # e.g. "out_proj_amp_mean"
            lt_name = None
            metric = None
            for m in _KNOWN_LT_METRICS:
                if suffix.endswith("_" + m) or suffix == m:
                    metric = m
                    lt_name = suffix[: -(len(m) + 1)] if suffix != m else ""
                    break
            if lt_name is None or not lt_name:
                continue
            if lt_name not in lt_series:
                lt_series[lt_name] = {}
            series = lt_series[lt_name]
            if metric not in series:
                series[metric] = []
            series[metric].append(val)
    return lt_series


def _aggregate_lt_stats(lt_series: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    """Aggregate per-layer-type time series into summary statistics."""
    result: dict[str, dict[str, float]] = {}
    for lt_name, metrics in sorted(lt_series.items()):
        entry: dict[str, float] = {}
        for metric_name, vals in metrics.items():
            if not vals:
                continue
            entry[f"{metric_name}_mean"] = sum(vals) / len(vals)
            if len(vals) > 1:
                mean = entry[f"{metric_name}_mean"]
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                entry[f"{metric_name}_std"] = var ** 0.5
            entry[f"{metric_name}_n"] = float(len(vals))
        result[lt_name] = entry
    return result


def classify_run(name: str) -> str:
    """Classify a run directory name into a category for GOAL §3.3 comparison.

    Returns one of: "baseline", "lawa", "psa_default", "psa_gamma", "psa_history",
    "psa_interval", "psa_other", "unknown".
    """
    lower = name.lower()
    if "baseline" in lower or lower == "baseline_plain":
        return "baseline"
    if lower == "lawa_only":
        return "lawa"
    if lower == "psa_default":
        return "psa_default"
    if lower.startswith("gamma_"):
        return "psa_gamma"
    if lower.startswith("history_"):
        return "psa_history"
    if lower.startswith("interval_"):
        return "psa_interval"
    if "psa" in lower:
        return "psa_other"
    return "unknown"


def parse_config_name(name: str) -> dict:
    """Extract γ and regime_reset from run name like 'gamma_0.5_reset_on'."""
    parts = name.split("_")
    gamma = None
    regime_reset = None
    for i, p in enumerate(parts):
        if p == "gamma" and i + 1 < len(parts):
            gamma = float(parts[i + 1])
        if p == "reset" and i + 1 < len(parts):
            regime_reset = parts[i + 1] == "on"
    return {"gamma": gamma, "regime_reset": regime_reset}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PSA ablation results")
    parser.add_argument("--sweep-dir", required=True, help="Ablation output directory")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        print(f"Error: {sweep_dir} not found", file=sys.stderr)
        sys.exit(1)

    raw_runs: dict[str, dict] = {}
    for metrics_file in sorted(sweep_dir.glob("*/run_metrics.jsonl")):
        name = metrics_file.parent.name
        run = load_run(metrics_file)
        if run is None:
            print(f"  Warning: {name} has no footer, skipping")
            continue
        raw_runs[name] = run

    if not raw_runs:
        print("No completed runs found.")
        sys.exit(0)

    # Build per-run summaries
    runs: list[dict] = []
    for name, run in raw_runs.items():
        footer = run["footer"]
        records = run["records"]
        eff = _compute_efficiency(run)
        cfg_info = parse_config_name(name)
        run_type = classify_run(name)

        regime_transitions = records[-1].get("psa_regime_transitions") or 0 if records else 0
        psa_gain_mean = records[-1].get("psa_gain_mean", 0.0) if records else 0.0

        regime_counts: dict[str, int] = {}
        for r in records:
            regime = r.get("psa_regime", "stable")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        lt_series = _extract_psa_lt_stats(records)
        lt_agg = _aggregate_lt_stats(lt_series)

        summary = footer.get("tg_lora_summary", {})
        lawa_loss = summary.get("lawa_loss") if isinstance(summary, dict) else None
        best_lawa_loss = summary.get("best_lawa_loss") if isinstance(summary, dict) else None
        lawa_snapshots = summary.get("lawa_snapshots_recorded") if isinstance(summary, dict) else None
        act_stable = summary.get("activation_regime_stable_fraction") if isinstance(summary, dict) else None
        act_null = summary.get("activation_regime_null_baseline") if isinstance(summary, dict) else None
        act_null_z = act_null.get("stable_fraction_z") if isinstance(act_null, dict) else None

        runs.append({
            "name": name,
            "run_type": run_type,
            "gamma": cfg_info["gamma"],
            "regime_reset": cfg_info["regime_reset"],
            "best_valid": footer.get("best_valid_loss"),
            "final_train": footer.get("final_train_loss", records[-1].get("loss_train") if records else None),
            "total_bp": records[-1].get("total_backward_passes", 0) if records else 0,
            "wall_min": footer.get("total_wall_seconds", 0) / 60,
            "regime_transitions": regime_transitions,
            "psa_gain_mean": psa_gain_mean,
            "regime_counts": regime_counts,
            "lawa_loss": lawa_loss,
            "best_lawa_loss": best_lawa_loss,
            "lawa_snapshots": lawa_snapshots,
            "act_stable": act_stable,
            "act_null_z": act_null_z,
            "lt_stats": lt_agg,
            **eff,
        })

    # === Section 0: GOAL §3.3 Decision Table ===
    baseline_runs = [r for r in runs if r["run_type"] == "baseline"]
    lawa_runs = [r for r in runs if r["run_type"] == "lawa"]
    psa_runs = [r for r in runs if r["run_type"].startswith("psa")]

    print("=" * 90)
    print("GOAL §3.3 Decision: Plain LoRA vs LAWA vs PSA")
    print("=" * 90)
    print(f"{'Condition':<28} {'Best Valid':>10} {'Loss Red':>9} {'Red/bp':>12} {'Total BP':>10}")
    print("-" * 72)

    ref_loss = None
    for label, subset in [("Plain LoRA", baseline_runs), ("LAWA", lawa_runs)]:
        if not subset:
            print(f"{label:<28} {'N/A':>10}")
            continue
        best = min(subset, key=lambda r: r["best_valid"] or float("inf"))
        bv = best["best_valid"]
        if label == "Plain LoRA" and bv is not None:
            ref_loss = bv
        bv_s = f"{bv:.4f}" if bv is not None else "N/A"
        lr = best.get("loss_reduction")
        lr_s = f"{lr:.4f}" if lr is not None else "N/A"
        rb = best.get("loss_red_per_bp")
        rb_s = f"{rb:.2e}" if rb is not None else "N/A"
        bp = str(best["total_bp"])
        print(f"{label:<28} {bv_s:>10} {lr_s:>9} {rb_s:>12} {bp:>10}")

    if psa_runs:
        best_psa = min(psa_runs, key=lambda r: r["best_valid"] or float("inf"))
        bv = best_psa["best_valid"]
        bv_s = f"{bv:.4f}" if bv is not None else "N/A"
        lr = best_psa.get("loss_reduction")
        lr_s = f"{lr:.4f}" if lr is not None else "N/A"
        rb = best_psa.get("loss_red_per_bp")
        rb_s = f"{rb:.2e}" if rb is not None else "N/A"
        bp = str(best_psa["total_bp"])
        print(f"{'PSA (best variant)':<28} {bv_s:>10} {lr_s:>9} {rb_s:>12} {bp:>10}")

        if ref_loss is not None and bv is not None:
            delta = bv - ref_loss
            sign = "+" if delta > 0 else ""
            print(f"\n  PSA vs Plain LoRA: Δ={sign}{delta:.4f}", end="")
            if delta < -0.001:
                print(" → PSA WINS")
            elif delta > 0.001:
                print(" → PSA LOSES")
            else:
                print(" → NEUTRAL")

        # LAWA comparison
        if lawa_runs:
            best_lawa = min(lawa_runs, key=lambda r: r["best_valid"] or float("inf"))
            lawa_vl = best_lawa["best_valid"]
            if lawa_vl is not None and bv is not None:
                delta = bv - lawa_vl
                sign = "+" if delta > 0 else ""
                print(f"  PSA vs LAWA:       Δ={sign}{delta:.4f}", end="")
                if delta < -0.001:
                    print(" → PSA WINS (§3.3 satisfied)")
                elif delta > 0.001:
                    print(" → PSA LOSES (§3.3 FAILED)")
                else:
                    print(" → NEUTRAL")

    print()

    # === Section 1: Per-layer-type diagnostics ===
    psa_with_lt = [r for r in psa_runs if r["lt_stats"]]
    if psa_with_lt:
        print("=" * 90)
        print("Per-Layer-Type Amplification & Stability (GOAL §4 step 2)")
        print("=" * 90)
        print(f"{'Run':<25} {'Layer Type':<12} {'Amp Mean':>9} {'Amp Std':>9} {'Stability':>10}")
        print("-" * 68)

        # Aggregate across all PSA runs for the "out_proj most-stable" test
        lt_overall: dict[str, list[tuple[float, float, float]]] = {}
        for r in psa_with_lt:
            for lt_name, stats in r["lt_stats"].items():
                amp_mean = stats.get("amp_mean_mean", 0)
                amp_std = stats.get("amp_std_mean", 0)
                stab = stats.get("prior_stability_mean", float("nan"))
                lt_overall.setdefault(lt_name, []).append((amp_mean, amp_std, stab))
                amp_s = f"{amp_mean:.3f}"
                std_s = f"{amp_std:.3f}"
                stab_s = f"{stab:.3f}" if stab == stab else "-"
                print(f"{r['name']:<25} {lt_name:<12} {amp_s:>9} {std_s:>9} {stab_s:>10}")

        if lt_overall:
            print("\n  Cross-run summary by layer type:")
            for lt_name in sorted(lt_overall):
                entries = lt_overall[lt_name]
                avg_amp = sum(e[0] for e in entries) / len(entries)
                avg_stab = sum(e[2] for e in entries) / len(entries)
                avg_stab_s = f"{avg_stab:.3f}" if avg_stab == avg_stab else "N/A"
                print(f"    {lt_name:<12}: avg_amp={avg_amp:.3f}  avg_stability={avg_stab_s}  (n_runs={len(entries)})")

            # out_proj most-stable hypothesis test (GOAL §4)
            out_proj_stab = lt_overall.get("out_proj", [])
            mlp_stab = lt_overall.get("mlp", [])
            if out_proj_stab and mlp_stab:
                op_avg = sum(e[2] for e in out_proj_stab) / len(out_proj_stab)
                mlp_avg = sum(e[2] for e in mlp_stab) / len(mlp_stab)
                if op_avg == op_avg and mlp_avg == mlp_avg:
                    if op_avg > mlp_avg:
                        print(f"\n  → out_proj stability ({op_avg:.3f}) > mlp ({mlp_avg:.3f}): hypothesis SUPPORTED")
                    else:
                        print(f"\n  → out_proj stability ({op_avg:.3f}) <= mlp ({mlp_avg:.3f}): hypothesis NOT supported")
        print()

    # === Section 2: Efficiency-ranked overview ===
    runs.sort(key=lambda r: r["best_valid"] or float("inf"))

    print("=" * 90)
    print("PSA Sweep — Efficiency Ranking")
    print("=" * 90)
    hdr = (
        f"{'Name':<25} {'Type':<8} {'γ':>4} {'Reset':>5} {'Best Valid':>10} "
        f"{'Loss Red':>9} {'Red/bp':>10} {'Trans':>5} {'Wall min':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in runs:
        bv = f"{r['best_valid']:.4f}" if r["best_valid"] is not None else "N/A"
        lr = f"{r['loss_reduction']:.4f}" if r["loss_reduction"] is not None else "N/A"
        rb = f"{r['loss_red_per_bp']:.2e}" if r["loss_red_per_bp"] is not None else "N/A"
        rt = f"{r['regime_transitions']}" if r["regime_transitions"] is not None else "-"
        reset = "on" if r["regime_reset"] else "off" if r["regime_reset"] is not None else "-"
        gamma = f"{r['gamma']}" if r["gamma"] is not None else "-"
        wm = f"{r['wall_min']:.1f}"
        print(
            f"{r['name']:<25} {r['run_type']:<8} {gamma:>4} {reset:>5} {bv:>10} {lr:>9} {rb:>10} {rt:>5} {wm:>9}"
        )

    # === Section 3: Regime-reset effect at each γ ===
    print("\n" + "=" * 90)
    print("Regime-Reset Effect: ON vs OFF at each γ")
    print("=" * 90)

    gammas = sorted(set(r["gamma"] for r in runs if r["gamma"] is not None))
    for gamma in gammas:
        on_runs = [r for r in runs if r["gamma"] == gamma and r["regime_reset"] is True]
        off_runs = [r for r in runs if r["gamma"] == gamma and r["regime_reset"] is False]

        on_loss = on_runs[0]["best_valid"] if on_runs else None
        off_loss = off_runs[0]["best_valid"] if off_runs else None

        if on_loss is not None and off_loss is not None:
            delta = on_loss - off_loss
            sign = "+" if delta > 0 else ""
            winner = "reset ON wins" if delta < 0 else "reset OFF wins" if delta > 0 else "tie"
            print(f"  γ={gamma}: ON={on_loss:.4f}  OFF={off_loss:.4f}  Δ={sign}{delta:.4f}  ({winner})")
        elif on_loss is not None:
            print(f"  γ={gamma}: ON={on_loss:.4f}  OFF=N/A")
        elif off_loss is not None:
            print(f"  γ={gamma}: ON=N/A  OFF={off_loss:.4f}")

    # === Section 4: γ effect ===
    print("\n" + "=" * 90)
    print("γ Effect: best loss at each gain level (across reset modes)")
    print("=" * 90)

    gamma_0_baseline = None
    for gamma in gammas:
        gamma_runs = [r for r in runs if r["gamma"] == gamma]
        best_at_gamma = min(gamma_runs, key=lambda r: r["best_valid"] or float("inf"))
        bv = best_at_gamma["best_valid"]
        reset_mode = "on" if best_at_gamma["regime_reset"] else "off"

        if gamma == 0.0:
            gamma_0_baseline = bv
            print(f"  γ={gamma}: {bv:.4f} (reset={reset_mode}) [BASELINE]")
        elif bv is not None and gamma_0_baseline is not None:
            delta = bv - gamma_0_baseline
            sign = "+" if delta > 0 else ""
            print(f"  γ={gamma}: {bv:.4f} (reset={reset_mode}) Δ vs γ=0: {sign}{delta:.4f}")
        else:
            print(f"  γ={gamma}: {bv}")

    # === Section 5: Regime transition statistics ===
    print("\n" + "=" * 90)
    print("Regime Transition Statistics")
    print("=" * 90)
    for r in runs:
        if r["regime_transitions"] is None:
            continue
        total_cycles = sum(r["regime_counts"].values())
        if total_cycles == 0:
            continue
        stable_pct = r["regime_counts"].get("stable", 0) / total_cycles * 100
        plateau_pct = r["regime_counts"].get("plateau", 0) / total_cycles * 100
        transition_pct = r["regime_counts"].get("transition", 0) / total_cycles * 100
        print(
            f"  {r['name']:<25} transitions={r['regime_transitions']:>3}  "
            f"stable={stable_pct:.0f}% plateau={plateau_pct:.0f}% "
            f"transition={transition_pct:.0f}%"
        )

    # === Section 6: Activation regime inventory (GOAL §4 step 1) ===
    act_runs = [r for r in runs if r.get("act_stable") is not None]
    if act_runs:
        print("\n" + "=" * 90)
        print("Activation Regime Inventory (GOAL §4 step 1 — Theoretical Efficiency Ceiling)")
        print("=" * 90)
        print(f"  {'Name':<28} {'Stable %':>8} {'Null z':>8}")
        print(f"  {'-'*46}")
        for r in act_runs:
            stab = r["act_stable"]
            null_z = r.get("act_null_z")
            stab_s = f"{stab:.2f}" if isinstance(stab, float) else "N/A"
            null_s = f"{null_z:.1f}σ" if isinstance(null_z, float) else "N/A"
            print(f"  {r['name']:<28} {stab_s:>8} {null_s:>8}")

        bl_act = [r for r in act_runs if r["run_type"] == "baseline"]
        psa_act = [r for r in act_runs if r["run_type"].startswith("psa")]
        if bl_act:
            bl_stable = bl_act[0]["act_stable"]
            print(f"\n  Baseline stable fraction: {bl_stable:.2f} = theoretical efficiency ceiling")
            if psa_act:
                psa_best = min(psa_act, key=lambda r: r.get("best_valid") or float("inf"))
                psa_stable = psa_best["act_stable"]
                delta = psa_stable - bl_stable
                sign = "+" if delta > 0 else ""
                print(f"  PSA best stable fraction: {psa_stable:.2f} (Δ={sign}{delta:.2f} vs baseline)")

    # === Recommendations ===
    print("\n" + "=" * 90)
    print("Next Actions")
    print("=" * 90)

    if not runs:
        print("  No runs to analyze.")
    else:
        best = runs[0]
        print(f"  Best config: {best['name']} (valid loss={best['best_valid']:.4f})")

        if gamma_0_baseline is not None:
            best_nonzero = [r for r in runs if r["gamma"] is not None and r["gamma"] > 0]
            if best_nonzero:
                best_nz = min(best_nonzero, key=lambda r: r["best_valid"] or float("inf"))
                delta = best_nz["best_valid"] - gamma_0_baseline
                if delta < -0.001:
                    print(f"  - IMPROVEMENT: γ={best_nz['gamma']} reduces loss by {abs(delta):.4f} vs γ=0")
                    print(f"    Regime reset effect: {'included' if best_nz['regime_reset'] else 'disabled'}")
                elif delta > 0.001:
                    print("  - NO IMPROVEMENT: all γ>0 worse than γ=0 ablation baseline")
                    print("    → PSA amplification may not be effective on this task/config")
                else:
                    print("  - NEUTRAL: no significant difference from γ=0 baseline")

        # GOAL §3.3 verdict
        if baseline_runs and psa_runs:
            best_bl = min(baseline_runs, key=lambda r: r["best_valid"] or float("inf"))
            best_psa_r = min(psa_runs, key=lambda r: r["best_valid"] or float("inf"))
            bl_vl = best_bl["best_valid"]
            psa_vl = best_psa_r["best_valid"]
            if bl_vl is not None and psa_vl is not None:
                delta = psa_vl - bl_vl
                if delta < -0.001:
                    print(f"  - §3.3 VERDICT: PSA beats plain LoRA by {abs(delta):.4f}")
                elif delta > 0.001:
                    print(f"  - §3.3 VERDICT: PSA does NOT beat plain LoRA (Δ={delta:+.4f})")

        if lawa_runs and psa_runs:
            best_lawa_r = min(lawa_runs, key=lambda r: r["best_valid"] or float("inf"))
            best_psa_r2 = min(psa_runs, key=lambda r: r["best_valid"] or float("inf"))
            lawa_vl = best_lawa_r["best_valid"]
            psa_vl2 = best_psa_r2["best_valid"]
            lawa_best = best_lawa_r.get("best_lawa_loss")
            if lawa_best is not None:
                print(f"  - LAWA best_lawa_loss: {lawa_best:.4f}")
            if lawa_vl is not None and psa_vl2 is not None:
                delta = psa_vl2 - lawa_vl
                if delta < -0.001:
                    print(f"  - §3.3 VERDICT: PSA beats LAWA by {abs(delta):.4f}")
                elif delta > 0.001:
                    print(f"  - §3.3 VERDICT: PSA does NOT beat LAWA (Δ={delta:+.4f})")

        # Check regime reset value
        reset_on_wins = 0
        reset_off_wins = 0
        for gamma in gammas:
            on_runs_g = [r for r in runs if r["gamma"] == gamma and r["regime_reset"] is True]
            off_runs_g = [r for r in runs if r["gamma"] == gamma and r["regime_reset"] is False]
            if on_runs_g and off_runs_g:
                on_l = on_runs_g[0]["best_valid"]
                off_l = off_runs_g[0]["best_valid"]
                if on_l is not None and off_l is not None:
                    if on_l < off_l:
                        reset_on_wins += 1
                    elif off_l < on_l:
                        reset_off_wins += 1

        if reset_on_wins > reset_off_wins:
            total = reset_on_wins + reset_off_wins
            print(f"  - Regime reset ON wins {reset_on_wins}/{total} comparisons")
            print("    → Keep regime-aware prior reset enabled")
        elif reset_off_wins > reset_on_wins:
            total = reset_on_wins + reset_off_wins
            print(f"  - Regime reset OFF wins {reset_off_wins}/{total} comparisons")
            print("    → Regime reset may be harmful on this task — consider disabling")

    # Save machine-readable summary
    summary_path = sweep_dir / "psa_sweep_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PSA Ablation Summary\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Name':<25} {'Type':<8} {'γ':>4} {'Reset':>5} {'Best Valid':>10} {'Loss Red':>9} {'Trans':>5}\n")
        f.write("-" * 70 + "\n")
        for r in runs:
            bv = f"{r['best_valid']:.4f}" if r["best_valid"] is not None else "N/A"
            lr = f"{r['loss_reduction']:.4f}" if r["loss_reduction"] is not None else "N/A"
            reset = "on" if r["regime_reset"] else "off" if r["regime_reset"] is not None else "-"
            gamma = f"{r['gamma']}" if r["gamma"] is not None else "-"
            trans = f"{r['regime_transitions']}" if r["regime_transitions"] is not None else "-"
            f.write(f"{r['name']:<25} {r['run_type']:<8} {gamma:>4} {reset:>5} {bv:>10} {lr:>9} {trans:>5}\n")

    # Save JSON summary for downstream analysis
    json_path = sweep_dir / "psa_sweep_summary.json"
    json_summary = {
        "runs": [
            {
                "name": r["name"],
                "run_type": r["run_type"],
                "best_valid": r["best_valid"],
                "total_bp": r["total_bp"],
                "loss_reduction": r["loss_reduction"],
                "loss_red_per_bp": r["loss_red_per_bp"],
                "regime_transitions": r["regime_transitions"],
                "lt_stats": r["lt_stats"],
            }
            for r in runs
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=str)

    print(f"\nSummary saved to {summary_path}")
    print(f"JSON summary saved to {json_path}")


if __name__ == "__main__":
    main()
