"""Post-experiment analysis for the Guard experiment (M10).

Compares two conditions (design 10_guard_experiment.md §5):
  (A) baseline:  dynfreeze_enabled=false  (all-layer TG-LoRA training)
  (B) guard:     dynfreeze_enabled=true   (output-side reversible freeze)

Quality is measured by the JSON-extraction gold metric (gold_combined), NOT
validation loss. The §5.2 two-stage stop is resolved post-hoc:
  - L*  = baseline best valid_full loss;  G* = gold_combined at that cycle
  - B stops at the first cycle where valid_full <= L* + 0.02 AND gold >= G*
  - Speed gate: B's elapsed_seconds at the stop point <= 0.90 * A's total

Produces:
  - guard_loss_vs_wallclock.png
  - gold_vs_wallclock.png
  - freeze_schedule.png
  - gate_decision.txt
  - metrics_timeseries.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

# §5.2 protocol constants (design 10_guard_experiment.md §8)
LOSS_TRIGGER_MARGIN = 0.02  # primary trigger: valid_full <= L* + 0.02
SPEED_GATE_RATIO = 0.90  # speed gate: B_stop <= A_total * 0.90
GOLD_SCORE_KEY = "gold_combined"


def load_run_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "run_metrics.jsonl"
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found")
        return []
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_loss_and_time(records: list[dict]) -> tuple[list[float], list[float], list[int]]:
    times, losses, cycles = [], [], []
    for r in records:
        if r.get("type") == "step" and "loss_valid" in r:
            times.append(r.get("elapsed_seconds", 0.0))
            losses.append(r["loss_valid"])
            cycles.append(r.get("cycle", 0))
    return times, losses, cycles


def extract_freeze_schedule(records: list[dict]) -> dict[int, dict]:
    """Extract per-cycle freeze block info.

    Only the cycle-boundary records carry guard_block_size; the per-step report
    records do not, so skip those to avoid overwriting real freeze state with 0.
    """
    schedule: dict[int, dict] = {}
    for r in records:
        if r.get("type") != "step" or "guard_block_size" not in r:
            continue
        cycle = r.get("cycle", 0)
        block_size = r.get("guard_block_size", 0)
        block_layers = r.get("guard_block_layers", "")
        # Keep the largest block seen at a cycle (robust to ordering).
        if cycle not in schedule or block_size > schedule[cycle]["block_size"]:
            schedule[cycle] = {"block_size": block_size, "block_layers": block_layers}
    return schedule


def extract_r_A_per_layer(records: list[dict], layer_indices: list[int]) -> dict[int, list[tuple[int, float]]]:
    """Extract per-cycle r_A for each layer."""
    r_A_data: dict[int, list[tuple[int, float]]] = {li: [] for li in layer_indices}
    for r in records:
        if r.get("type") != "step":
            continue
        cycle = r.get("cycle", 0)
        for li in layer_indices:
            key = f"guard_r_A_L{li}"
            if key in r:
                r_A_data[li].append((cycle, r[key]))
    return r_A_data


def extract_gold(records: list[dict]) -> list[dict]:
    """Per-cycle gold-eval rows: cycles where gold_combined was recorded.

    Each row carries the cycle's elapsed_seconds and loss_valid alongside the
    gold scores so the §5.2 stop (loss AND gold) can be evaluated jointly.
    """
    rows: list[dict] = []
    for r in records:
        if r.get("type") != "step" or GOLD_SCORE_KEY not in r:
            continue
        rows.append(
            {
                "cycle": r.get("cycle", 0),
                "elapsed": r.get("elapsed_seconds", 0.0),
                "loss": r.get("loss_valid"),
                "gold_combined": r.get("gold_combined"),
                "gold_field_f1": r.get("gold_field_f1"),
                "gold_exact_match": r.get("gold_exact_match"),
                "gold_strict_valid": r.get("gold_strict_valid"),
            }
        )
    return rows


def baseline_targets(
    baseline_losses: list[float],
    baseline_cycles: list[int],
    baseline_times: list[float],
    baseline_gold: list[dict],
) -> tuple[float | None, int | None, float | None, float | None]:
    """§5.1: L* = best valid loss; G* = gold at that cycle; A_total = run length."""
    if not baseline_losses:
        return None, None, None, None
    # L* and its cycle (earliest cycle achieving the min loss)
    best_idx = int(np.argmin(baseline_losses))
    l_star = baseline_losses[best_idx]
    l_star_cycle = baseline_cycles[best_idx] if baseline_cycles else None
    a_total = baseline_times[-1] if baseline_times else None
    # G* = gold_combined at (or just before) the L* cycle
    g_star = None
    if baseline_gold:
        preceding = [g for g in baseline_gold if g["cycle"] <= (l_star_cycle or 0)]
        pool = preceding if preceding else baseline_gold
        g_star = pool[-1]["gold_combined"]
    return l_star, l_star_cycle, g_star, a_total


def guard_stop_point(
    guard_gold: list[dict], l_star: float | None, g_star: float | None
) -> tuple[int | None, float | None]:
    """§5.2: first cycle where valid <= L*+margin AND gold >= G*.

    Returns (stop_cycle, stop_elapsed) or (None, None) if quality never reached.
    """
    if l_star is None or g_star is None or not guard_gold:
        return None, None
    loss_threshold = l_star + LOSS_TRIGGER_MARGIN
    for row in guard_gold:
        loss = row.get("loss")
        gold = row.get("gold_combined")
        if loss is None or gold is None:
            continue
        if loss <= loss_threshold and gold >= g_star:
            return row["cycle"], row["elapsed"]
    return None, None


def plot_loss_vs_wallclock(
    baseline_times: list[float],
    baseline_losses: list[float],
    guard_times: list[float],
    guard_losses: list[float],
    output_path: Path,
    L_star: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if baseline_times:
        ax.plot(baseline_times, baseline_losses, label="Baseline (A)", color="blue", alpha=0.8)
    if guard_times:
        ax.plot(guard_times, guard_losses, label="Guard (B)", color="red", alpha=0.8)
    if L_star is not None:
        ax.axhline(y=L_star, color="green", linestyle="--", alpha=0.5, label=f"L*={L_star:.4f}")
    ax.set_xlabel("Cumulative Wall-Clock Time (s)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Loss vs Wall-Clock: Baseline vs Guard")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gold_vs_wallclock(
    baseline_gold: list[dict],
    guard_gold: list[dict],
    g_star: float | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if baseline_gold:
        ax.plot(
            [g["elapsed"] for g in baseline_gold],
            [g["gold_combined"] for g in baseline_gold],
            label="Baseline (A)", color="blue", marker="o", alpha=0.8,
        )
    if guard_gold:
        ax.plot(
            [g["elapsed"] for g in guard_gold],
            [g["gold_combined"] for g in guard_gold],
            label="Guard (B)", color="red", marker="o", alpha=0.8,
        )
    if g_star is not None:
        ax.axhline(y=g_star, color="green", linestyle="--", alpha=0.5, label=f"G*={g_star:.3f}")
    ax.set_xlabel("Cumulative Wall-Clock Time (s)")
    ax.set_ylabel("Gold combined score")
    ax.set_title("JSON-extraction Gold vs Wall-Clock: Baseline vs Guard")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_freeze_schedule(
    schedule: dict[int, dict],
    layer_indices: list[int],
    output_path: Path,
) -> None:
    if not schedule:
        return
    cycles = sorted(schedule.keys())
    layers = sorted(layer_indices)
    matrix = np.zeros((len(layers), len(cycles)))
    for j, c in enumerate(cycles):
        info = schedule.get(c, {})
        frozen_str = info.get("block_layers", "")
        if frozen_str:
            frozen_set = set(int(x) for x in frozen_str.split(",") if x.strip())
            for i, li in enumerate(layers):
                if li in frozen_set:
                    matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{li}" for li in layers])
    ax.set_xlabel("Cycle")
    ax.set_title("Freeze Schedule (red=frozen)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def write_gate_decision(
    baseline_dir: Path,
    guard_dir: Path,
    baseline_losses: list[float],
    baseline_cycles: list[int],
    baseline_times: list[float],
    baseline_gold: list[dict],
    guard_losses: list[float],
    guard_times: list[float],
    guard_gold: list[dict],
    schedule: dict[int, dict],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("Gate Decision — Guard Experiment (§5.2)")
    lines.append("=" * 50)

    l_star, l_star_cycle, g_star, a_total = baseline_targets(
        baseline_losses, baseline_cycles, baseline_times, baseline_gold
    )
    stop_cycle, stop_elapsed = guard_stop_point(guard_gold, l_star, g_star)
    # A's own time-to-quality (fair baseline for the speed comparison).
    a_t2q_cycle, a_t2q_elapsed = guard_stop_point(baseline_gold, l_star, g_star)

    lines.append(f"\nBaseline (A): {baseline_dir}")
    if baseline_losses:
        lines.append(f"  Cycles: {len(baseline_losses)}")
        lines.append(f"  L* (best valid loss): {l_star:.4f} @ cycle {l_star_cycle}")
        lines.append(f"  G* (gold at L*): {g_star if g_star is not None else 'N/A'}")
        lines.append(f"  Total wall-clock: {a_total:.1f}s" if a_total else "  N/A")

    lines.append(f"\nGuard (B): {guard_dir}")
    if guard_losses:
        lines.append(f"  Cycles: {len(guard_losses)}")
        lines.append(f"  Best valid loss: {min(guard_losses):.4f}")
        lines.append(f"  Total wall-clock: {guard_times[-1]:.1f}s" if guard_times else "  N/A")
        best_guard_gold = max((g["gold_combined"] for g in guard_gold if g["gold_combined"] is not None), default=None)
        lines.append(f"  Best gold_combined: {best_guard_gold if best_guard_gold is not None else 'N/A'}")
        total_cycles = max(schedule.keys()) + 1 if schedule else len(guard_losses)
        frozen_cycles = sum(1 for v in schedule.values() if v.get("block_size", 0) > 0)
        if total_cycles > 0:
            lines.append(f"  Cycles with frozen layers: {frozen_cycles}/{total_cycles}")

    lines.append("\n--- Gate Decision (§5.2 / §7) ---")
    if l_star is not None:
        lines.append(f"  L* = {l_star:.4f}   G* = {g_star if g_star is not None else 'N/A'}")

    # Quality gate (§5.2): gold >= G* reached within the loss trigger margin
    quality_pass = stop_cycle is not None
    if g_star is not None:
        lines.append(
            f"  Quality (gold>=G* @ loss<=L*+{LOSS_TRIGGER_MARGIN}): "
            f"{'PASS' if quality_pass else 'FAIL'}"
            + (f" @ cycle {stop_cycle}" if quality_pass else " (never reached)")
        )
    else:
        lines.append("  Quality: SKIP (no gold metric recorded on baseline)")

    # Speed gate (§7): FAIR comparison = B time-to-quality vs A time-to-quality
    # (both measured at the gold>=G* @ loss<=L*+margin point). Comparing B's
    # stop against A's full run is misleading when A lacks early-stopping.
    fair_pass = False
    if stop_elapsed is not None and a_t2q_elapsed is not None:
        speedup_fair = (a_t2q_elapsed - stop_elapsed) / a_t2q_elapsed * 100
        fair_pass = stop_elapsed <= a_t2q_elapsed * SPEED_GATE_RATIO
        lines.append(
            f"  Speed (fair, time-to-quality): A {a_t2q_elapsed:.1f}s @c{a_t2q_cycle} "
            f"vs B {stop_elapsed:.1f}s @c{stop_cycle} → "
            f"{'PASS' if fair_pass else 'FAIL'} ({speedup_fair:+.1f}% to quality)"
        )
    else:
        lines.append("  Speed (fair): N/A (quality not reached on both runs)")
    # Protocol gate (B_stop vs A full run) — reference only; inflated if A ran past quality.
    if a_total and stop_elapsed is not None:
        speedup_proto = (a_total - stop_elapsed) / a_total * 100
        lines.append(
            f"  Speed (protocol, B_stop vs A_total {a_total:.1f}s): "
            f"{speedup_proto:+.1f}% (reference)"
        )

    overall = quality_pass and fair_pass if g_star is not None else False
    lines.append(f"\n  Overall (quality AND fair speed): {'PASS ✓' if overall else 'FAIL ✗'}")

    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")


def write_metrics_csv(
    records: list[dict],
    layer_indices: list[int],
    output_path: Path,
) -> None:
    step_records = [r for r in records if r.get("type") == "step"]
    if not step_records:
        return
    fieldnames = [
        "cycle", "elapsed_seconds", "loss_valid", "guard_block_size", "guard_block_layers",
        "gold_combined", "gold_field_f1", "gold_exact_match", "gold_strict_valid",
    ]
    for li in layer_indices:
        fieldnames.append(f"guard_r_A_L{li}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in step_records:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Guard experiment results")
    parser.add_argument(
        "--baseline-dir", type=str, default="runs/mlx_9b_jsonex_baseline"
    )
    parser.add_argument(
        "--guard-dir", type=str, default="runs/mlx_9b_jsonex_guard"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    guard_dir = Path(args.guard_dir)
    output_dir = Path(args.output_dir) if args.output_dir else guard_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_indices = list(range(24, 32))

    baseline_records = load_run_metrics(baseline_dir)
    guard_records = load_run_metrics(guard_dir)

    baseline_times, baseline_losses, baseline_cycles = extract_loss_and_time(baseline_records)
    guard_times, guard_losses, _ = extract_loss_and_time(guard_records)

    baseline_gold = extract_gold(baseline_records)
    guard_gold = extract_gold(guard_records)

    l_star, _, g_star, _ = baseline_targets(
        baseline_losses, baseline_cycles, baseline_times, baseline_gold
    )

    plot_loss_vs_wallclock(
        baseline_times, baseline_losses,
        guard_times, guard_losses,
        output_dir / "guard_loss_vs_wallclock.png",
        L_star=l_star,
    )

    if baseline_gold or guard_gold:
        plot_gold_vs_wallclock(
            baseline_gold, guard_gold, g_star,
            output_dir / "gold_vs_wallclock.png",
        )

    schedule = extract_freeze_schedule(guard_records)
    if schedule:
        plot_freeze_schedule(schedule, layer_indices, output_dir / "freeze_schedule.png")

    write_gate_decision(
        baseline_dir, guard_dir,
        baseline_losses, baseline_cycles, baseline_times, baseline_gold,
        guard_losses, guard_times, guard_gold,
        schedule,
        output_dir / "gate_decision.txt",
    )

    write_metrics_csv(guard_records, layer_indices, output_dir / "metrics_timeseries.csv")


if __name__ == "__main__":
    main()
