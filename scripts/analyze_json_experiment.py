"""Analyze the JSON-extraction efficiency experiment (Plain vs LAWA vs PSA).

Reads each condition's ``json_eval_log.jsonl`` (written by the TG-LoRA trainer at
every full-eval cycle) and plots JSON-quality metrics vs cumulative backwards and
vs cycle. Writes a text summary stating whether PSA beats plain and LAWA on the
same data-digestion axis (GOAL.md §3.3/§4.3).

Usage:
    python scripts/analyze_json_experiment.py [--runs-dir runs] \\
        [--conditions plain lawa psa] [--targets 0.5 0.7]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# computed_accuracy isolates the arithmetic fields (duration_minutes /
# total_cost) — the discriminating metric where a strong base model lags.
METRICS = ["combined", "field_f1", "exact_match", "computed_accuracy", "strict_valid"]


def load_json_eval(run_dir: Path) -> list[dict]:
    path = run_dir / "json_eval_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def backwards_to_target(rows: list[dict], metric: str, target: float) -> float | None:
    """First cumulative-backward count at which ``metric`` reaches ``target``."""
    for r in rows:
        if r.get(metric, 0.0) >= target:
            return r.get("full_backward_passes")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--conditions", nargs="+", default=["plain", "lawa", "psa"])
    ap.add_argument("--prefix", default="jsonex_")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.9, 0.95],
                    help="combined targets (combined saturates near ceiling; 0.5/0.7 are trivially hit)")
    ap.add_argument("--cacc-targets", nargs="+", type=float, default=[0.7, 0.75, 0.8],
                    help="computed_accuracy targets (plain plateaus ~0.72; does LAWA/PSA break it?)")
    ap.add_argument("--out-dir", default="runs/jsonex_analysis")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, list[dict]] = {}
    for cond in args.conditions:
        rows = load_json_eval(runs_dir / f"{args.prefix}{cond}")
        if not rows:
            print(f"[warn] no json_eval_log.jsonl for {cond} ({runs_dir / f'{args.prefix}{cond}'})")
        data[cond] = rows

    # ---- text summary ----
    lines: list[str] = []
    lines.append("JSON-Extraction Efficiency Experiment — Summary")
    lines.append("=" * 60)
    for cond, rows in data.items():
        if not rows:
            lines.append(f"\n{cond}: NO DATA")
            continue
        last = rows[-1]
        lines.append(
            f"\n{cond}: {len(rows)} eval points, "
            f"final cycle={last.get('cycle')} backwards={last.get('full_backward_passes')}"
        )
        for m in METRICS:
            lines.append(f"  {m:14s} final={last.get(m, float('nan')):.3f}")
        for t in args.targets:
            bw = backwards_to_target(rows, "combined", t)
            lines.append(f"  -> combined>={t}: reached at backwards={bw}")

    # head-to-head. combined saturates near ceiling (format is easy) so it won't
    # separate the conditions; computed_accuracy (arithmetic fields) is the real
    # discriminator — plain plateaus ~0.72, and the question is whether LAWA/PSA
    # break through. Report WIN/LOSS on both, plus LAWA-vs-plain (GOAL §3.3 bar).
    if all(data.get(c) for c in ("plain", "lawa", "psa")):
        lines.append("\n" + "-" * 60)
        for metric in ("combined", "computed_accuracy"):
            final = {c: data[c][-1].get(metric, 0.0) for c in ("plain", "lawa", "psa")}
            lines.append(f"\nFinal {metric}: " + ", ".join(f"{c}={v:.3f}" for c, v in final.items()))
            for a, b in (("psa", "plain"), ("psa", "lawa"), ("lawa", "plain")):
                lines.append(
                    f"  {a} vs {b}: " + ("WIN" if final[a] > final[b] else "LOSS")
                    + f" ({final[a] - final[b]:+.3f})"
                )
        for metric, targets in (
            ("combined", args.targets), ("computed_accuracy", args.cacc_targets)
        ):
            for t in targets:
                bw = {c: backwards_to_target(data[c], metric, t) for c in ("plain", "lawa", "psa")}
                lines.append(
                    f"Backwards to {metric}>={t}: "
                    + ", ".join(f"{c}={bw[c]}" for c in ("plain", "lawa", "psa"))
                )

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "json_experiment_summary.txt").write_text(summary + "\n")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping plots")
        return

    colors = {"plain": "tab:gray", "lawa": "tab:blue", "psa": "tab:red"}

    def _plot(xkey: str, ylabel: str, fname: str):
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.5))
        for ax, m in zip(axes, METRICS):
            for cond, rows in data.items():
                if not rows:
                    continue
                xs = [r[xkey] for r in rows]
                ys = [r.get(m, 0.0) for r in rows]
                ax.plot(xs, ys, marker="o", label=cond, color=colors.get(cond))
            ax.set_title(m)
            ax.set_xlabel({"full_backward_passes": "cumulative backwards", "cycle": "cycle"}[xkey])
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)
        print(f"wrote {out_dir / fname}")

    _plot("full_backward_passes", "score", "json_quality_vs_backwards.png")
    _plot("cycle", "score", "json_quality_vs_cycle.png")


if __name__ == "__main__":
    main()
