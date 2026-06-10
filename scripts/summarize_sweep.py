"""Summarize TG-LoRA hyperparameter sweep results.

Produces a ranked efficiency table, pairwise deltas vs baseline, and
next-action recommendations for the accel param sweep (TASK-0094).
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sweep results")
    parser.add_argument("--sweep-dir", required=True, help="Sweep output directory")
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

    # Build per-run summaries with efficiency metrics
    runs: list[dict] = []
    for name, run in raw_runs.items():
        footer = run["footer"]
        records = run["records"]
        accepted = sum(1 for r in records if r.get("tg_lora_accepted"))
        total_cycles = len(records)
        accept_rate = accepted / total_cycles if total_cycles > 0 else 0
        last_train = records[-1].get("loss_train") if records else None
        eff = _compute_efficiency(run)

        runs.append({
            "name": name,
            "best_valid": footer.get("best_valid_loss"),
            "final_train": footer.get("final_train_loss", last_train),
            "accept_rate": accept_rate,
            "total_cycles": total_cycles,
            "wall_min": footer.get("total_wall_seconds", 0) / 60,
            "total_bp": records[-1].get("total_backward_passes", 0) if records else 0,
            "controller": footer.get("tg_lora_summary", {}),
            **eff,
        })

    # Sort by best valid loss (lower is better)
    runs.sort(key=lambda r: r["best_valid"] or float("inf"))

    # Identify baseline
    baseline = next((r for r in runs if "no_accel" in r["name"]), runs[0])
    treatments = [r for r in runs if r["name"] != baseline["name"]]

    # Print efficiency-ranked table
    hdr = (
        f"{'Name':<14} {'Best Valid':>10} {'Loss Red':>9} "
        f"{'Red/bp':>10} {'Red/min':>10} {'Accept%':>8} {'Wall min':>9} {'BP':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in runs:
        bv = f"{r['best_valid']:.4f}" if r["best_valid"] is not None else "N/A"
        lr = f"{r['loss_reduction']:.4f}" if r["loss_reduction"] is not None else "N/A"
        rb = f"{r['loss_red_per_bp']:.2e}" if r["loss_red_per_bp"] is not None else "N/A"
        rm = f"{r['loss_red_per_wall_min']:.5f}" if r["loss_red_per_wall_min"] is not None else "N/A"
        ar = f"{r['accept_rate']:.0%}" if r["accept_rate"] else "N/A"
        wm = f"{r['wall_min']:.1f}"
        print(
            f"{r['name']:<14} {bv:>10} {lr:>9} {rb:>10} {rm:>10} "
            f"{ar:>8} {wm:>9} {r['total_bp']:>6}"
        )

    # Best run details
    best = runs[0]
    print(f"\nBest: {best['name']} (valid loss={best['best_valid']:.4f})")
    ctrl = best.get("controller", {})
    if ctrl:
        print(
            f"  K={ctrl.get('current_K')}, N={ctrl.get('current_N')}, "
            f"alpha={ctrl.get('current_alpha', 0):.4f}, beta={ctrl.get('current_beta')}"
        )
    if best["loss_red_per_wall_min"] is not None:
        print(f"  Loss red/wall-min: {best['loss_red_per_wall_min']:.5f}")
    if best["loss_red_per_bp"] is not None:
        print(f"  Loss red/bp: {best['loss_red_per_bp']:.2e}")

    # Pairwise deltas vs baseline
    if baseline["best_valid"] is not None and treatments:
        print(f"\nPairwise vs {baseline['name']} baseline:")
        bl_loss = baseline["best_valid"]
        for t in sorted(treatments, key=lambda r: r["best_valid"] or float("inf")):
            t_loss = t["best_valid"]
            if t_loss is None:
                continue
            delta = t_loss - bl_loss
            delta_pct = (delta / bl_loss * 100) if bl_loss != 0 else 0
            sign = "+" if delta > 0 else ""
            print(
                f"  {t['name']:<14} delta={sign}{delta:.4f} ({sign}{delta_pct:.2f}%)"
            )

    # Next-action recommendation
    best_treatment = treatments[0] if treatments else None
    print("\nNext actions:")
    if best_treatment and best_treatment["best_valid"] is not None:
        bl_loss = baseline["best_valid"]
        delta = best_treatment["best_valid"] - bl_loss if bl_loss is not None else None
        if delta is not None and delta < -0.001:
            print(f"  - IMPROVEMENT: {best_treatment['name']} reduces loss by {abs(delta):.4f}")
            print("  - Run lm-evaluation-harness on best config")
            print("  - Compare TruthfulQA / ARC / HellaSwag vs baseline")
        elif delta is not None and delta > 0.001:
            print("  - NO IMPROVEMENT: all treatments worse than no_accel baseline")
            print("  - Consider different accel param ranges or disabling accel adaptation")
        else:
            print("  - NEUTRAL: no significant difference from baseline")
            print("  - Consider expanding parameter grid or increasing training cycles")

    # Save summary
    summary_path = sweep_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("TG-LoRA Sweep Summary\n")
        f.write("=" * 78 + "\n\n")
        f.write(
            f"{'Name':<14} {'Best Valid':>10} {'Loss Red':>9} "
            f"{'Red/bp':>10} {'Red/min':>10} {'Accept%':>8}\n"
        )
        f.write("-" * 65 + "\n")
        for r in runs:
            bv = f"{r['best_valid']:.4f}" if r["best_valid"] is not None else "N/A"
            lr = f"{r['loss_reduction']:.4f}" if r["loss_reduction"] is not None else "N/A"
            rb = f"{r['loss_red_per_bp']:.2e}" if r["loss_red_per_bp"] is not None else "N/A"
            rm = f"{r['loss_red_per_wall_min']:.5f}" if r["loss_red_per_wall_min"] is not None else "N/A"
            ar = f"{r['accept_rate']:.0%}" if r["accept_rate"] else "N/A"
            f.write(f"{r['name']:<14} {bv:>10} {lr:>9} {rb:>10} {rm:>10} {ar:>8}\n")

        if baseline["best_valid"] is not None and treatments:
            f.write(f"\nPairwise vs {baseline['name']}:\n")
            bl_loss = baseline["best_valid"]
            for t in sorted(treatments, key=lambda r: r["best_valid"] or float("inf")):
                t_loss = t["best_valid"]
                if t_loss is None:
                    continue
                delta = t_loss - bl_loss
                delta_pct = (delta / bl_loss * 100) if bl_loss != 0 else 0
                f.write(f"  {t['name']}: delta={delta:+.4f} ({delta_pct:+.2f}%)\n")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
