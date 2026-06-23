"""Compare baseline QLoRA vs TG-LoRA training runs.

Usage:
    # Two-run comparison (legacy)
    python scripts/compare_runs.py --baseline runs/baseline/run_metrics.jsonl --tg-lora runs/tg_lora/run_metrics.jsonl

    # Multi-run dashboard
    python scripts/compare_runs.py dashboard runs/
    python scripts/compare_runs.py dashboard runs/ --format json
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

# Allow running as a standalone CLI (``python scripts/compare_runs.py``): a bare script
# invocation puts ``scripts/`` — not the repo root — on sys.path, so make the repo root
# importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trajectory_artifact_anomalies import \
    summarize_trajectory_artifact_anomalies
from src.utils.mlflow_logger import MLflowLogger
from src.utils.run_query import list_runs, parse_jsonl


def load_run(path: Path) -> tuple[dict, list[dict], dict | None]:
    header = {}
    records = []
    footer = None
    for line in path.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        rec = orjson.loads(line)
        t = rec.get("type", "step")
        if t == "run_header":
            header = rec
        elif t == "run_footer":
            footer = rec
        else:
            records.append(rec)
    return header, records, footer


def _fmt(v, width=10, prec=4):
    if v is None:
        return "N/A".rjust(width)
    if isinstance(v, float):
        return f"{v:.{prec}f}".rjust(width)
    return str(v).rjust(width)


def _pct_delta(base, tg):
    if base is None or tg is None or base == 0:
        return "N/A"
    delta = (tg - base) / abs(base) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"


def _record_fallback_loss(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    first_train = next((r.get("loss_train") for r in records if r.get("loss_train") is not None), None)
    if isinstance(first_train, (int, float)):
        return float(first_train)
    first_valid = next((r.get("loss_valid") for r in records if r.get("loss_valid") is not None), None)
    if isinstance(first_valid, (int, float)):
        return float(first_valid)
    return None


def _header_reference_loss(
    header: dict[str, Any] | None,
    records: list[dict[str, Any]],
    *,
    allow_header_reference: bool = True,
) -> float | None:
    if allow_header_reference:
        comparison_reference = (header or {}).get("comparison_reference") or {}
        loss = comparison_reference.get("loss")
        if isinstance(loss, (int, float)):
            return float(loss)
    return _record_fallback_loss(records)


def _header_reference_kind(header: dict[str, Any] | None) -> str | None:
    comparison_reference = (header or {}).get("comparison_reference") or {}
    kind = comparison_reference.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    return None


def _paired_reference_losses(
    b_header: dict[str, Any],
    b_records: list[dict[str, Any]],
    t_header: dict[str, Any],
    t_records: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    b_header_loss = ((b_header.get("comparison_reference") or {}).get("loss"))
    t_header_loss = ((t_header.get("comparison_reference") or {}).get("loss"))
    b_kind = _header_reference_kind(b_header)
    t_kind = _header_reference_kind(t_header)
    if (
        b_kind is not None
        and b_kind == t_kind
        and isinstance(b_header_loss, (int, float))
        and isinstance(t_header_loss, (int, float))
    ):
        return float(b_header_loss), float(t_header_loss)
    return (
        _header_reference_loss(
            b_header, b_records, allow_header_reference=False
        ),
        _header_reference_loss(
            t_header, t_records, allow_header_reference=False
        ),
    )


def _summary_reference_loss(run: dict[str, Any]) -> float | None:
    loss = run.get("comparison_reference_loss", run.get("initial_loss"))
    if isinstance(loss, (int, float)):
        return float(loss)
    return None


def _use_group_comparison_reference(runs: list[dict[str, Any]]) -> bool:
    if not runs:
        return False
    kinds = {r.get("comparison_reference_kind") for r in runs}
    if len(kinds) != 1:
        return False
    kind = next(iter(kinds))
    if not isinstance(kind, str) or not kind:
        return False
    return all(isinstance(r.get("comparison_reference_loss"), (int, float)) for r in runs)


def _append_artifact_anomaly_lines(
    lines: list[str],
    label: str,
    anomalies: list[dict[str, Any]] | None,
) -> None:
    if not anomalies:
        return
    lines.append("")
    lines.append(f"--- {label} Delta Artifact Anomalies ---")
    for anomaly in anomalies[:5]:
        cycle_or_step = (
            f"cycle={anomaly['cycle']}"
            if anomaly.get("cycle") is not None
            else f"step={anomaly['step']}"
        )
        lines.append(
            f"{anomaly['anchor_kind']}: {cycle_or_step} "
            f"norm={anomaly['delta_total_norm']:.4f} z={anomaly['robust_z_score']:.2f}"
        )
        for example in anomaly.get("source_examples", [])[:3]:
            locator_bits = []
            if example.get("record_id") is not None:
                locator_bits.append(f"id={example['record_id']}")
            locator_bits.append(f"idx={example['dataset_index']}")
            lines.append(
                f"  source [{', '.join(locator_bits)}]: {example['text_preview']}"
            )


def generate_report(
    b_header,
    b_records,
    b_footer,
    t_header,
    t_records,
    t_footer,
    *,
    baseline_artifact_anomalies: list[dict[str, Any]] | None = None,
    tg_artifact_anomalies: list[dict[str, Any]] | None = None,
):
    lines = []
    lines.append("=" * 70)
    lines.append("  TG-LoRA vs Baseline QLoRA: Efficiency Comparison")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Config
    lines.append("")
    lines.append("--- Run Configuration ---")
    lines.append(f"{'':>25s}  {'Baseline':>12s}  {'TG-LoRA':>12s}")
    lines.append(
        f"{'Model':>25s}  {b_header.get('model_name', ''):>12s}  {t_header.get('model_name', ''):>12s}"
    )
    lines.append(
        f"{'LoRA r/alpha':>25s}  {str(b_header.get('lora_r', '')) + '/' + str(b_header.get('lora_alpha', '')):>12s}  {str(t_header.get('lora_r', '')) + '/' + str(t_header.get('lora_alpha', '')):>12s}"
    )
    lines.append(
        f"{'Batch/GA':>25s}  {str(b_header.get('batch_size', '')) + '/' + str(b_header.get('grad_accumulation', '')):>12s}  {str(t_header.get('batch_size', '')) + '/' + str(t_header.get('grad_accumulation', '')):>12s}"
    )
    lines.append(
        f"{'Learning Rate':>25s}  {b_header.get('learning_rate', 0):>12.1e}  {t_header.get('learning_rate', 0):>12.1e}"
    )
    lines.append(
        f"{'Seed':>25s}  {str(b_header.get('seed', '')):>12s}  {str(t_header.get('seed', '')):>12s}"
    )

    # Compute
    lines.append("")
    lines.append("--- Compute Budget ---")
    lines.append(f"{'':>25s}  {'Baseline':>12s}  {'TG-LoRA':>12s}")
    bw = b_footer.get("total_wall_seconds", 0) if b_footer else 0
    tw = t_footer.get("total_wall_seconds", 0) if t_footer else 0
    lines.append(
        f"{'Total Wall Time':>25s}  {_fmt(bw / 60 if bw else None, 10, 1)} min  {_fmt(tw / 60 if tw else None, 10, 1)} min"
    )

    b_bp = b_records[-1]["total_backward_passes"] if b_records else 0
    t_bp = t_records[-1]["total_backward_passes"] if t_records else 0
    lines.append(f"{'Backward Passes':>25s}  {b_bp:>12d}  {t_bp:>12d}")

    t_extrap = sum(r.get("tg_lora_N", 0) or 0 for r in t_records)
    lines.append(f"{'Extrapolation Steps':>25s}  {'-':>12s}  {t_extrap:>12d}")
    lines.append(f"{'Effective Steps':>25s}  {b_bp:>12d}  {t_bp + t_extrap:>12d}")

    # Outcome
    lines.append("")
    lines.append("--- Training Outcome ---")
    lines.append(f"{'':>25s}  {'Baseline':>12s}  {'TG-LoRA':>12s}")
    b_best = b_footer.get("best_valid_loss") if b_footer else None
    t_best = t_footer.get("best_valid_loss") if t_footer else None
    lines.append(f"{'Best Valid Loss':>25s}  {_fmt(b_best, 12)}  {_fmt(t_best, 12)}")

    b_final = b_footer.get("final_train_loss") if b_footer else None
    t_final = t_footer.get("final_train_loss") if t_footer else None
    lines.append(f"{'Final Train Loss':>25s}  {_fmt(b_final, 12)}  {_fmt(t_final, 12)}")

    b_bstep = b_footer.get("best_valid_step") if b_footer else None
    t_bstep = t_footer.get("best_valid_step") if t_footer else None
    lines.append(f"{'Best at Step':>25s}  {str(b_bstep):>12s}  {str(t_bstep):>12s}")

    # Efficiency
    lines.append("")
    lines.append("--- Efficiency Metrics ---")
    lines.append(f"{'':>35s}  {'Baseline':>10s}  {'TG-LoRA':>10s}  {'Delta':>10s}")

    b_init, t_init = _paired_reference_losses(
        b_header, b_records, t_header, t_records
    )
    b_loss_red = (
        (b_init - b_best) if (b_init is not None and b_best is not None) else None
    )
    t_loss_red = (
        (t_init - t_best) if (t_init is not None and t_best is not None) else None
    )

    b_per_bp = b_loss_red / b_bp if (b_loss_red is not None and b_bp) else None
    t_per_bp = t_loss_red / t_bp if (t_loss_red is not None and t_bp) else None
    lines.append(
        f"{'Loss Red. / 100 backward':>35s}  {_fmt(b_per_bp * 100 if b_per_bp else None, 10)}  {_fmt(t_per_bp * 100 if t_per_bp else None, 10)}  {_pct_delta(b_per_bp, t_per_bp):>10s}"
    )

    b_per_min = b_loss_red / (bw / 60) if (b_loss_red is not None and bw) else None
    t_per_min = t_loss_red / (tw / 60) if (t_loss_red is not None and tw) else None
    lines.append(
        f"{'Loss Red. / wall-minute':>35s}  {_fmt(b_per_min, 10, 5)}  {_fmt(t_per_min, 10, 5)}  {_pct_delta(b_per_min, t_per_min):>10s}"
    )

    b_peak = b_footer.get("gpu_peak_mb") if b_footer else None
    t_peak = t_footer.get("gpu_peak_mb") if t_footer else None
    b_gbhr = (
        b_loss_red / (b_peak / 1024 * bw / 3600)
        if (b_loss_red and b_peak and bw)
        else None
    )
    t_gbhr = (
        t_loss_red / (t_peak / 1024 * tw / 3600)
        if (t_loss_red and t_peak and tw)
        else None
    )
    lines.append(
        f"{'Loss Red. / GB-hour':>35s}  {_fmt(b_gbhr, 10, 4)}  {_fmt(t_gbhr, 10, 4)}  {_pct_delta(b_gbhr, t_gbhr):>10s}"
    )
    lines.append(
        f"{'GPU Peak Memory':>35s}  {_fmt(b_peak, 10, 0)} MB  {_fmt(t_peak, 10, 0)} MB  {_pct_delta(b_peak, t_peak):>10s}"
    )

    # TG-LoRA specific
    if t_records:
        accepted = [r for r in t_records if r.get("tg_lora_accepted")]
        cos_sims = [
            r["tg_lora_cosine_sim"]
            for r in t_records
            if r.get("tg_lora_cosine_sim") is not None
        ]
        red_rates = [
            r["tg_lora_reduction_rate"]
            for r in t_records
            if r.get("tg_lora_reduction_rate") is not None
        ]

        lines.append("")
        lines.append("--- TG-LoRA Specific ---")
        lines.append(
            f"{'Acceptance Rate':>25s}  {len(accepted) / len(t_records) * 100:.1f}%"
        )
        if cos_sims:
            lines.append(
                f"{'Avg Cosine Similarity':>25s}  {sum(cos_sims) / len(cos_sims):.4f}"
            )
        if red_rates:
            lines.append(
                f"{'Avg Reduction Rate':>25s}  {sum(red_rates) / len(red_rates):.1%}"
            )
        last = t_records[-1]
        lines.append(f"{'Final K':>25s}  {last.get('tg_lora_K', 'N/A')}")
        lines.append(f"{'Final N':>25s}  {last.get('tg_lora_N', 'N/A')}")
        lines.append(f"{'Final alpha':>25s}  {last.get('tg_lora_alpha', 'N/A')}")

    _append_artifact_anomaly_lines(
        lines,
        "Baseline",
        baseline_artifact_anomalies,
    )
    _append_artifact_anomaly_lines(
        lines,
        "TG-LoRA",
        tg_artifact_anomalies,
    )

    return "\n".join(lines)


def plot_loss_curves(b_records, t_records, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss vs backward passes
    if b_records:
        axes[0].plot(
            [r["total_backward_passes"] for r in b_records],
            [r["loss_train"] for r in b_records],
            label="Baseline",
            alpha=0.8,
        )
    if t_records:
        axes[0].plot(
            [r["total_backward_passes"] for r in t_records],
            [r["loss_train"] for r in t_records],
            label="TG-LoRA",
            alpha=0.8,
        )
    axes[0].set_xlabel("Total Backward Passes")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Train Loss vs Compute")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: loss vs wall-clock
    if b_records:
        axes[1].plot(
            [r["elapsed_seconds"] / 60 for r in b_records],
            [r["loss_train"] for r in b_records],
            label="Baseline",
            alpha=0.8,
        )
    if t_records:
        axes[1].plot(
            [r["elapsed_seconds"] / 60 for r in t_records],
            [r["loss_train"] for r in t_records],
            label="TG-LoRA",
            alpha=0.8,
        )
    axes[1].set_xlabel("Wall-Clock Time (min)")
    axes[1].set_ylabel("Train Loss")
    axes[1].set_title("Train Loss vs Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_acceptance_rate(t_records: list[dict], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping acceptance rate plot")
        return

    accepted = []
    for i, r in enumerate(t_records):
        accepted.append(1 if r.get("tg_lora_accepted") else 0)

    if not accepted:
        print("No TG-LoRA records for acceptance rate plot")
        return

    cumulative = []
    total = 0
    for i, a in enumerate(accepted, 1):
        total += a
        cumulative.append(total / i * 100)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker=".", alpha=0.7)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cumulative Acceptance Rate (%)")
    ax.set_title("TG-LoRA Acceptance Rate Over Cycles")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_reduction_rate(t_records: list[dict], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping reduction rate plot")
        return

    cycles = []
    rates = []
    for i, r in enumerate(t_records, 1):
        rr = r.get("tg_lora_reduction_rate")
        if rr is not None:
            cycles.append(i)
            rates.append(rr)

    if not rates:
        print("No reduction_rate data available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cycles, rates, marker=".", alpha=0.7)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Reduction Rate")
    ax.set_title("TG-LoRA Reduction Rate Over Cycles")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_velocity_magnitude(t_records: list[dict], output_path: Path) -> None:
    """Plot gradient norm (velocity magnitude proxy) over cycles."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping velocity magnitude plot")
        return

    cycles = []
    magnitudes = []
    for i, r in enumerate(t_records, 1):
        gn = r.get("grad_norm")
        if gn is not None:
            cycles.append(i)
            magnitudes.append(gn)

    if not magnitudes:
        print("No grad_norm data available, skipping velocity magnitude plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cycles, magnitudes, marker=".", alpha=0.7)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Velocity Magnitude (Grad Norm) Over Cycles")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_layer_scores(layer_scores: dict[int, float], output_path: Path) -> None:
    """Plot layer score distribution as a bar chart."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping layer scores plot")
        return

    if not layer_scores:
        print("No layer scores data available, skipping plot")
        return

    sorted_indices = sorted(layer_scores.keys())
    scores = [layer_scores[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_indices) * 0.4), 4))
    ax.bar(sorted_indices, scores, alpha=0.7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Score")
    ax.set_title("Layer Score Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_hyperparams(t_records: list[dict], output_path: Path) -> None:
    """Plot K, N, alpha, lr trajectories over cycles."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping hyperparams plot")
        return

    cycles = list(range(1, len(t_records) + 1))
    alpha_vals = [r.get("tg_lora_alpha") for r in t_records]
    lr_vals = [r.get("tg_lora_lr") for r in t_records]
    K_vals = [r.get("tg_lora_K") for r in t_records]
    N_vals = [r.get("tg_lora_N") for r in t_records]

    has_data = any(v is not None for v in alpha_vals + lr_vals + K_vals + N_vals)
    if not has_data:
        print("No hyperparameter data available, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, vals, label in [
        (axes[0, 0], alpha_vals, "alpha"),
        (axes[0, 1], lr_vals, "lr"),
        (axes[1, 0], K_vals, "K"),
        (axes[1, 1], N_vals, "N"),
    ]:
        plotted = [(c, v) for c, v in zip(cycles, vals) if v is not None]
        if plotted:
            xs, ys = zip(*plotted)
            ax.plot(xs, ys, marker=".", alpha=0.7)
        ax.set_xlabel("Cycle")
        ax.set_ylabel(label)
        ax.set_title(f"{label} Over Cycles")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def generate_markdown_report(
    b_header,
    b_records,
    b_footer,
    t_header,
    t_records,
    t_footer,
    *,
    baseline_artifact_anomalies: list[dict[str, Any]] | None = None,
    tg_artifact_anomalies: list[dict[str, Any]] | None = None,
):
    md = []
    layer_scores = {
        i: score
        for i, score in enumerate(
            [r.get("tg_lora_cosine_sim", 0.0) or 0.0 for r in t_records]
        )
        if score != 0.0
    }
    has_velocity_plot = any(r.get("grad_norm") is not None for r in t_records)
    has_reduction_rate_plot = any(
        r.get("tg_lora_reduction_rate") is not None for r in t_records
    )
    has_hyperparams_plot = any(
        r.get(key) is not None
        for r in t_records
        for key in ("tg_lora_alpha", "tg_lora_lr", "tg_lora_K", "tg_lora_N")
    )
    md.append("# TG-LoRA vs Baseline QLoRA: Efficiency Comparison\n")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # Configuration
    md.append("## Configuration\n")
    md.append("| Parameter | Baseline | TG-LoRA |")
    md.append("|-----------|----------|---------|")
    md.append(
        f"| Model | {b_header.get('model_name', '')} | {t_header.get('model_name', '')} |"
    )
    md.append(
        f"| LoRA r/alpha | {b_header.get('lora_r', '')}/{b_header.get('lora_alpha', '')} | {t_header.get('lora_r', '')}/{t_header.get('lora_alpha', '')} |"
    )
    md.append(
        f"| Batch/GA | {b_header.get('batch_size', '')}/{b_header.get('grad_accumulation', '')} | {t_header.get('batch_size', '')}/{t_header.get('grad_accumulation', '')} |"
    )
    md.append(
        f"| Learning Rate | {b_header.get('learning_rate', 0):.1e} | {t_header.get('learning_rate', 0):.1e} |"
    )
    md.append(f"| Seed | {b_header.get('seed', '')} | {t_header.get('seed', '')} |")
    md.append("")

    # Compute Budget
    bw = b_footer.get("total_wall_seconds", 0) if b_footer else 0
    tw = t_footer.get("total_wall_seconds", 0) if t_footer else 0
    b_bp = b_records[-1]["total_backward_passes"] if b_records else 0
    t_bp = t_records[-1]["total_backward_passes"] if t_records else 0
    t_extrap = sum(r.get("tg_lora_N", 0) or 0 for r in t_records)

    md.append("## Compute Budget\n")
    md.append("| Metric | Baseline | TG-LoRA |")
    md.append("|--------|----------|---------|")
    md.append(
        f"| Total Wall Time | {_fmt(bw / 60, 10, 1)} min | {_fmt(tw / 60, 10, 1)} min |"
    )
    md.append(f"| Backward Passes | {b_bp} | {t_bp} |")
    md.append(f"| Extrapolation Steps | - | {t_extrap} |")
    md.append(f"| Effective Steps | {b_bp} | {t_bp + t_extrap} |")
    md.append("")

    # Training Outcome
    b_best = b_footer.get("best_valid_loss") if b_footer else None
    t_best = t_footer.get("best_valid_loss") if t_footer else None
    b_final = b_footer.get("final_train_loss") if b_footer else None
    t_final = t_footer.get("final_train_loss") if t_footer else None
    b_bstep = b_footer.get("best_valid_step") if b_footer else None
    t_bstep = t_footer.get("best_valid_step") if t_footer else None

    md.append("## Training Outcome\n")
    md.append("| Metric | Baseline | TG-LoRA |")
    md.append("|--------|----------|---------|")
    md.append(f"| Best Valid Loss | {_fmt(b_best, 10)} | {_fmt(t_best, 10)} |")
    md.append(f"| Final Train Loss | {_fmt(b_final, 10)} | {_fmt(t_final, 10)} |")
    md.append(
        f"| Best at Step | {b_bstep if b_bstep is not None else 'N/A'} | {t_bstep if t_bstep is not None else 'N/A'} |"
    )
    md.append("")

    # Efficiency Metrics
    b_init, t_init = _paired_reference_losses(
        b_header, b_records, t_header, t_records
    )
    b_loss_red = (
        (b_init - b_best) if (b_init is not None and b_best is not None) else None
    )
    t_loss_red = (
        (t_init - t_best) if (t_init is not None and t_best is not None) else None
    )
    b_per_bp = b_loss_red / b_bp if (b_loss_red is not None and b_bp) else None
    t_per_bp = t_loss_red / t_bp if (t_loss_red is not None and t_bp) else None
    b_per_min = b_loss_red / (bw / 60) if (b_loss_red is not None and bw) else None
    t_per_min = t_loss_red / (tw / 60) if (t_loss_red is not None and tw) else None
    b_peak = b_footer.get("gpu_peak_mb") if b_footer else None
    t_peak = t_footer.get("gpu_peak_mb") if t_footer else None
    b_gbhr = (
        b_loss_red / (b_peak / 1024 * bw / 3600)
        if (b_loss_red and b_peak and bw)
        else None
    )
    t_gbhr = (
        t_loss_red / (t_peak / 1024 * tw / 3600)
        if (t_loss_red and t_peak and tw)
        else None
    )

    md.append("## Efficiency Metrics\n")
    md.append("| Metric | Baseline | TG-LoRA | Delta |")
    md.append("|--------|----------|---------|-------|")
    md.append(
        f"| Loss Red. / 100 backward | {_fmt(b_per_bp * 100 if b_per_bp else None, 10)} | {_fmt(t_per_bp * 100 if t_per_bp else None, 10)} | {_pct_delta(b_per_bp, t_per_bp)} |"
    )
    md.append(
        f"| Loss Red. / wall-minute | {_fmt(b_per_min, 10, 5)} | {_fmt(t_per_min, 10, 5)} | {_pct_delta(b_per_min, t_per_min)} |"
    )
    md.append(
        f"| Loss Red. / GB-hour | {_fmt(b_gbhr, 10, 4)} | {_fmt(t_gbhr, 10, 4)} | {_pct_delta(b_gbhr, t_gbhr)} |"
    )
    md.append(
        f"| GPU Peak Memory | {_fmt(b_peak, 10, 0)} MB | {_fmt(t_peak, 10, 0)} MB | {_pct_delta(b_peak, t_peak)} |"
    )
    md.append("")

    # TG-LoRA Specific
    if t_records:
        accepted = [r for r in t_records if r.get("tg_lora_accepted")]
        cos_sims = [
            r["tg_lora_cosine_sim"]
            for r in t_records
            if r.get("tg_lora_cosine_sim") is not None
        ]
        red_rates = [
            r["tg_lora_reduction_rate"]
            for r in t_records
            if r.get("tg_lora_reduction_rate") is not None
        ]

        md.append("## TG-LoRA Specific Metrics\n")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Acceptance Rate | {len(accepted) / len(t_records) * 100:.1f}% |")
        if cos_sims:
            md.append(
                f"| Avg Cosine Similarity | {sum(cos_sims) / len(cos_sims):.4f} |"
            )
        if red_rates:
            md.append(f"| Avg Reduction Rate | {sum(red_rates) / len(red_rates):.1%} |")
        last = t_records[-1]
        md.append(f"| Final K | {last.get('tg_lora_K', 'N/A')} |")
        md.append(f"| Final N | {last.get('tg_lora_N', 'N/A')} |")
        md.append(f"| Final alpha | {last.get('tg_lora_alpha', 'N/A')} |")
        md.append("")

    for label, anomalies in (
        ("Baseline", baseline_artifact_anomalies),
        ("TG-LoRA", tg_artifact_anomalies),
    ):
        if not anomalies:
            continue
        md.append(f"## {label} Delta Artifact Anomalies\n")
        for anomaly in anomalies[:5]:
            cycle_or_step = (
                f"cycle={anomaly['cycle']}"
                if anomaly.get("cycle") is not None
                else f"step={anomaly['step']}"
            )
            md.append(
                f"- {anomaly['anchor_kind']} at {cycle_or_step}: "
                f"norm={anomaly['delta_total_norm']:.4f}, z={anomaly['robust_z_score']:.2f}"
            )
            for example in anomaly.get("source_examples", [])[:3]:
                locator_bits = []
                if example.get("record_id") is not None:
                    locator_bits.append(f"id={example['record_id']}")
                locator_bits.append(f"idx={example['dataset_index']}")
                md.append(
                    f"  source [{', '.join(locator_bits)}]: {example['text_preview']}"
                )
        md.append("")

    # Plots
    md.append("## Plots\n")
    md.append("### Loss Curves\n")
    md.append("![Loss Comparison](loss_comparison.png)\n")
    if t_records:
        md.append("### TG-LoRA Metrics\n")
        md.append("![Acceptance Rate](acceptance_rate.png)\n")
        if has_velocity_plot:
            md.append("![Velocity Magnitude](velocity_magnitude.png)\n")
        if has_reduction_rate_plot:
            md.append("![Reduction Rate](reduction_rate.png)\n")
        if layer_scores:
            md.append("![Layer Scores](layer_scores.png)\n")
        if has_hyperparams_plot:
            md.append("![Hyperparameters](hyperparams.png)\n")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Multi-run comparison (TASK-0061)
# ---------------------------------------------------------------------------


def gather_runs(run_dir: str | Path) -> list[dict[str, Any]]:
    """Gather detailed run summaries for all runs under *run_dir*."""
    summaries = list_runs(run_dir)
    for s in summaries:
        s.setdefault("parse_warnings", [])
        jsonl_path = Path(s.get("_jsonl_path", ""))
        if not jsonl_path.exists():
            continue
        try:
            records = parse_jsonl(jsonl_path)
            steps = [r for r in records if r.get("type") == "step"]
            s["num_steps"] = len(steps)
            s["fallback_initial_loss"] = _record_fallback_loss(steps)
            if s.get("comparison_reference_loss") is not None:
                s["initial_loss"] = s.get("comparison_reference_loss")
            if steps:
                if s.get("initial_loss") is None:
                    s["initial_loss"] = s.get("fallback_initial_loss")
                s["final_loss"] = steps[-1].get("loss_train")
                s["total_backward_passes"] = steps[-1].get("total_backward_passes")
            s["delta_artifact_anomalies"] = summarize_trajectory_artifact_anomalies(
                jsonl_path.parent,
            )
        except (ValueError, orjson.JSONDecodeError) as e:
            msg = f"Failed to parse {jsonl_path.name}: {e}"
            s["parse_warnings"].append(msg)
            print(f"WARNING: {msg}", file=sys.stderr)
    return summaries


def find_best_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the run with the lowest best_valid_loss (or perplexity as tie-break)."""
    valid = [r for r in runs if r.get("best_valid_loss") is not None]
    if not valid:
        return None
    return min(
        valid, key=lambda r: (r["best_valid_loss"], r.get("perplexity") or float("inf"))
    )


def build_comparison_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a flat comparison table from run summaries.

    Includes efficiency metrics (loss_red/bp, loss_red/wall_min) for
    sweep analysis and best-config identification.
    """
    best = find_best_run(runs)
    best_id = best["run_id"] if best else None
    use_group_reference = _use_group_comparison_reference(runs)
    rows = []
    for r in runs:
        if use_group_reference and isinstance(r.get("comparison_reference_loss"), (int, float)):
            initial = float(r.get("comparison_reference_loss"))
        else:
            initial = r.get("fallback_initial_loss", r.get("initial_loss"))
        best_vl = r.get("best_valid_loss")
        total_bp = r.get("total_backward_passes") or 0
        wall_sec = r.get("total_wall_seconds") or 0

        loss_red = (initial - best_vl) if (initial is not None and best_vl is not None) else None
        loss_red_per_bp = loss_red / total_bp if (loss_red is not None and total_bp > 0) else None
        loss_red_per_wall_min = (
            loss_red / (wall_sec / 60) if (loss_red is not None and wall_sec > 0) else None
        )

        rows.append(
            {
                "run_id": r.get("run_id"),
                "mode": r.get("mode"),
                "model_name": r.get("model_name"),
                "best_valid_loss": r.get("best_valid_loss"),
                "final_train_loss": r.get("final_train_loss"),
                "perplexity": r.get("perplexity"),
                "best_valid_step": r.get("best_valid_step"),
                "total_wall_seconds": r.get("total_wall_seconds"),
                "num_steps": r.get("num_steps"),
                "initial_loss": initial,
                "comparison_reference_loss": r.get("comparison_reference_loss"),
                "comparison_reference_kind": r.get("comparison_reference_kind"),
                "final_loss": r.get("final_loss"),
                "total_backward_passes": r.get("total_backward_passes"),
                "loss_reduction": loss_red,
                "loss_red_per_bp": loss_red_per_bp,
                "loss_red_per_wall_min": loss_red_per_wall_min,
                "is_best": r.get("run_id") == best_id,
                "parse_warnings": r.get("parse_warnings", []),
                "delta_artifact_anomalies": r.get("delta_artifact_anomalies", []),
            }
        )
    return rows


def render_dashboard(runs: list[dict[str, Any]]) -> None:
    """Render rich Table/Panel dashboard to console."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    best = find_best_run(runs)

    # Summary panel
    lines = [f"[bold]Runs found:[/bold] {len(runs)}"]
    if best:
        lines.append(
            f"[bold green]Best run:[/bold green] {best.get('run_id', '?')}  "
            f"(loss={best.get('best_valid_loss', 'N/A')})"
        )
    console.print(
        Panel("\n".join(lines), title="Multi-Run Comparison", border_style="blue")
    )

    # Comparison table
    table = Table(title="Run Metrics", show_lines=True)
    table.add_column("Run ID", style="bold")
    table.add_column("Mode")
    table.add_column("Best Valid Loss", justify="right")
    table.add_column("Final Train Loss", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Wall Time (min)", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Best", justify="center")

    for r in runs:
        is_best = best and r.get("run_id") == best.get("run_id")
        best_mark = "[green]*[/green]" if is_best else ""
        wall = r.get("total_wall_seconds")
        wall_str = f"{wall / 60:.1f}" if wall else "N/A"
        ppl = r.get("perplexity")
        ppl_str = f"{ppl:.4f}" if ppl is not None else "N/A"
        bvl = r.get("best_valid_loss")
        bvl_str = f"{bvl:.4f}" if bvl is not None else "N/A"
        ftl = r.get("final_train_loss")
        ftl_str = f"{ftl:.4f}" if ftl is not None else "N/A"
        style = "green" if is_best else None

        table.add_row(
            str(r.get("run_id", "?")),
            str(r.get("mode", "?")),
            bvl_str,
            ftl_str,
            ppl_str,
            wall_str,
            str(r.get("num_steps", "N/A")),
            best_mark,
            style=style,
        )

    console.print(table)

    all_warnings = [w for r in runs for w in r.get("parse_warnings", [])]
    if all_warnings:
        console.print(Panel(
            "\n".join(f"- {w}" for w in all_warnings),
            title="Parse Warnings",
            border_style="yellow",
        ))


def format_json(runs: list[dict[str, Any]]) -> str:
    """Return JSON string of the comparison table with any parse warnings."""
    rows = build_comparison_table(runs)
    all_warnings = [w for r in rows for w in r.get("parse_warnings", [])]
    result: dict[str, Any] = {"runs": rows}
    if all_warnings:
        result["parse_warnings"] = all_warnings
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


def log_reports_to_mlflow(
    output_dir: Path,
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Log generated comparison reports and plots as MLflow artifacts."""
    mlf = MLflowLogger(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name="compare_runs",
    )
    if not mlf.enabled:
        return
    with mlf:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                mlf.log_artifact(str(path), "comparison_report")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Handle the 'dashboard' subcommand."""
    run_dir = args.run_dir
    runs = gather_runs(run_dir)
    if not runs:
        print(f"No runs found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(format_json(runs))
    else:
        render_dashboard(runs)


def main():
    parser = argparse.ArgumentParser(
        description="Compare training runs — single pair or multi-run dashboard",
    )
    sub = parser.add_subparsers(dest="command")

    # Legacy two-run mode (positional args)
    parser.add_argument("--baseline", help="Path to baseline run_metrics.jsonl")
    parser.add_argument("--tg-lora", help="Path to TG-LoRA run_metrics.jsonl")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--mlflow", action="store_true", help="Log reports to MLflow")
    parser.add_argument(
        "--mlflow-tracking-uri", default=None, help="MLflow tracking URI"
    )
    parser.add_argument(
        "--mlflow-experiment", default=None, help="MLflow experiment name"
    )

    # Dashboard subcommand
    dash = sub.add_parser("dashboard", help="Multi-run comparison dashboard")
    dash.add_argument("run_dir", help="Directory containing run subdirectories")
    dash.add_argument(
        "--format",
        choices=["rich", "json"],
        default="rich",
        help="Output format (default: rich)",
    )

    args = parser.parse_args()

    if args.command == "dashboard":
        cmd_dashboard(args)
        return

    if not args.baseline or not args.tg_lora:
        parser.error("--baseline and --tg-lora are required when not using 'dashboard'")

    b_header, b_records, b_footer = load_run(Path(args.baseline))
    t_header, t_records, t_footer = load_run(Path(args.tg_lora))
    baseline_artifact_anomalies = summarize_trajectory_artifact_anomalies(
        Path(args.baseline).resolve().parent,
    )
    tg_artifact_anomalies = summarize_trajectory_artifact_anomalies(
        Path(args.tg_lora).resolve().parent,
    )

    report = generate_report(
        b_header,
        b_records,
        b_footer,
        t_header,
        t_records,
        t_footer,
        baseline_artifact_anomalies=baseline_artifact_anomalies,
        tg_artifact_anomalies=tg_artifact_anomalies,
    )
    print(report)

    md_report = generate_markdown_report(
        b_header,
        b_records,
        b_footer,
        t_header,
        t_records,
        t_footer,
        baseline_artifact_anomalies=baseline_artifact_anomalies,
        tg_artifact_anomalies=tg_artifact_anomalies,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    md_path = out / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    md_path.write_text(md_report)
    print(f"Markdown report saved to {md_path}")

    if not args.no_plot:
        plot_loss_curves(b_records, t_records, out / "loss_comparison.png")
        if t_records:
            plot_acceptance_rate(t_records, out / "acceptance_rate.png")
            plot_velocity_magnitude(t_records, out / "velocity_magnitude.png")
            plot_reduction_rate(t_records, out / "reduction_rate.png")
            plot_layer_scores(
                {
                    i: s
                    for i, s in enumerate(
                        [r.get("tg_lora_cosine_sim", 0.0) or 0.0 for r in t_records]
                    )
                    if s != 0.0
                },
                out / "layer_scores.png",
            )
            plot_hyperparams(t_records, out / "hyperparams.png")

    if args.mlflow:
        log_reports_to_mlflow(
            out,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
        )


if __name__ == "__main__":
    main()
