#!/usr/bin/env python3
"""Learning phase labeling: detect descent/transition phases from absolute LoRA states.

Processes safetensors checkpoints (cycle-by-cycle) to extract:
  - cos(t): subspace alignment between consecutive cycles (top-k=4)
  - T(t): subspace rotation (sin of principal angles)
  - A(t): Frobenius norm of cumulative ΔW
  - S(t): effective rank (exp of entropy)

Then labels transition candidates and cross-validates with valid_loss.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
import torch

# ── constants ──────────────────────────────────────────────────────────
TOP_K = 4
ATTN_PREFIXES = ("linear_attn.", "self_attn.")
MLP_PREFIX = "mlp."
TRAINABLE_LAYERS = range(24, 32)  # last_25_percent of 32-layer model


# ── SVD ────────────────────────────────────────────────────────────────
def thin_svd_BA(B: torch.Tensor, A: torch.Tensor):
    """Singular values and right singular vectors of ΔW = B·A.  O(d·r²)."""
    Q_B, R_B = torch.linalg.qr(B.float())
    M = R_B @ A.float()
    _, sigma, Vh = torch.linalg.svd(M, full_matrices=False)
    return sigma.numpy().astype(np.float64), Vh.mH.numpy().astype(np.float64)


def frobenius(sigma: np.ndarray) -> float:
    return float(np.sqrt(np.sum(sigma**2)))


def effective_rank(sigma: np.ndarray) -> float:
    p = sigma**2
    total = p.sum()
    if total < 1e-30:
        return 1.0
    p = p / total
    p = p[p > 1e-30]
    return float(np.exp(-np.sum(p * np.log(p))))


def subspace_cosine(V_t: np.ndarray, V_tp1: np.ndarray, k: int = TOP_K) -> float:
    """Mean cos(principal angles) of top-k subspaces."""
    G = V_t[:, :k].T @ V_tp1[:, :k]
    cos_vals = np.linalg.svd(G, compute_uv=False)
    return float(np.mean(np.clip(cos_vals, 0, 1)))


def subspace_rotation(V_t: np.ndarray, V_tp1: np.ndarray, k: int = TOP_K) -> float:
    """Mean sin(principal angles) = T."""
    G = V_t[:, :k].T @ V_tp1[:, :k]
    cos_vals = np.clip(np.linalg.svd(G, compute_uv=False), 0, 1)
    return float(np.mean(np.sqrt(1 - cos_vals**2)))


# ── discovery ──────────────────────────────────────────────────────────
def discover_checkpoints(run_dir: Path) -> list[int]:
    cycles = []
    for d in run_dir.iterdir():
        m = re.match(r"checkpoint-cycle-(\d+)", d.name)
        if m and d.is_dir():
            safetensor = d / "adapter_model.safetensors"
            if safetensor.exists():
                cycles.append(int(m.group(1)))
    return sorted(cycles)


def discover_series(state: dict) -> list[tuple[int, str, str]]:
    """[(layer, module, group)] sorted output→input."""
    pat = re.compile(r"layers\.(\d+)\.(.+?)\.lora_A\.")
    series = set()
    for k in state:
        m = pat.search(k)
        if m and int(m.group(1)) in TRAINABLE_LAYERS:
            li, mod = int(m.group(1)), m.group(2)
            grp = "attn" if any(mod.startswith(p) for p in ATTN_PREFIXES) else "mlp"
            series.add((li, mod, grp))
    return sorted(series, key=lambda x: (-x[0], 0 if x[2] == "mlp" else 1, x[1]))


def series_label(li: int, mod: str) -> str:
    short = mod.replace("linear_attn.", "attn.").replace("self_attn.", "attn.")
    return f"L{li:02d} {short}"


# ── main pipeline ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-surrogates", type=int, default=20)
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "phase_labeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles = discover_checkpoints(run_dir)
    if len(cycles) < 2:
        print(f"ERROR: need >= 2 checkpoints, found {len(cycles)}")
        sys.exit(1)
    print(f"Checkpoints: {len(cycles)} cycles ({cycles[0]}–{cycles[-1]})")

    # Load first checkpoint to discover series
    state0 = load_file(run_dir / f"checkpoint-cycle-{cycles[0]}/adapter_model.safetensors")
    series_list = discover_series(state0)
    n_s = len(series_list)
    n_t = len(cycles)
    print(f"Series: {n_s}")

    # ── extract per-cycle SVD ──────────────────────────────────────────
    print("Extracting SVD from absolute checkpoints …")
    sigma_hist = np.full((n_s, n_t, 16), np.nan)
    V_hist = [[None] * n_t for _ in range(n_s)]
    A_arr = np.full((n_s, n_t), np.nan)
    S_arr = np.full((n_s, n_t), np.nan)

    for ci, cycle in enumerate(cycles):
        path = run_dir / f"checkpoint-cycle-{cycle}/adapter_model.safetensors"
        state = load_file(path)
        for si, (li, mod, _grp) in enumerate(series_list):
            a_key = f"base_model.model.model.layers.{li}.{mod}.lora_A.weight"
            b_key = f"base_model.model.model.layers.{li}.{mod}.lora_B.weight"
            if a_key not in state or b_key not in state:
                continue
            sigma, V = thin_svd_BA(state[b_key], state[a_key])
            sigma_hist[si, ci] = sigma
            V_hist[si][ci] = V
            A_arr[si, ci] = frobenius(sigma)
            S_arr[si, ci] = effective_rank(sigma)
        if (ci + 1) % 10 == 0 or ci == n_t - 1:
            print(f"  cycle {cycle} ({ci + 1}/{n_t})")

    # ── cos and T ──────────────────────────────────────────────────────
    cos_arr = np.full((n_s, n_t - 1), np.nan)
    T_arr = np.full((n_s, n_t - 1), np.nan)
    for si in range(n_s):
        for ci in range(n_t - 1):
            v0, v1 = V_hist[si][ci], V_hist[si][ci + 1]
            if v0 is not None and v1 is not None:
                cos_arr[si, ci] = subspace_cosine(v0, v1)
                T_arr[si, ci] = subspace_rotation(v0, v1)

    # ── surrogates for cos ─────────────────────────────────────────────
    print(f"Computing {args.n_surrogates} surrogates …")
    rng = np.random.default_rng(42)
    cos_surr = np.full((n_s, n_t - 1, args.n_surrogates), np.nan)
    for si in range(n_s):
        valid_idx = [ci for ci in range(n_t) if V_hist[si][ci] is not None]
        if len(valid_idx) < 3:
            continue
        valid_V = [V_hist[si][ci] for ci in valid_idx]
        nv = len(valid_V)
        for sh in range(args.n_surrogates):
            perm = rng.permutation(nv)
            for pi in range(nv - 1):
                i0, i1 = perm[pi], perm[pi + 1]
                cos_surr[si, pi % (n_t - 1), sh] = subspace_cosine(valid_V[i0], valid_V[i1])

    surr_median = np.nanmedian(cos_surr, axis=2)
    surr_p05 = np.nanpercentile(cos_surr, 5, axis=2)
    surr_p95 = np.nanpercentile(cos_surr, 95, axis=2)

    # ── load valid_loss from run_metrics.jsonl ─────────────────────────
    metrics_path = run_dir / "run_metrics.jsonl"
    loss_map = {}
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "step" and "loss_valid" in d:
                    c = d.get("cycle")
                    if c is not None:
                        loss_map[c] = d["loss_valid"]

    # Align loss to checkpoint cycles
    loss_at_cycles = np.array([loss_map.get(c, np.nan) for c in cycles])
    print(f"Valid loss aligned: {sum(~np.isnan(loss_at_cycles))}/{n_t} cycles")

    # ── transition detection ───────────────────────────────────────────
    # Transition candidate: cos drops significantly below local baseline
    # AND T spikes AND/OR S bulges
    transitions_per_series = [[] for _ in range(n_s)]
    for si in range(n_s):
        cos_s = cos_arr[si]
        T_s = T_arr[si]
        S_s = S_arr[si]

        for ci in range(len(cos_s)):
            if np.isnan(cos_s[ci]):
                continue
            # cos drop: below rolling median - 2*std or below surrogate p5
            floor = surr_p05[si, ci] if not np.isnan(surr_p05[si, ci]) else 0.5
            cos_drop = cos_s[ci] < floor
            # T spike: above surrogate p95
            ceiling = surr_p95[si, ci] if not np.isnan(surr_p95[si, ci]) else 0.5
            T_spike = T_s[ci] > ceiling if not np.isnan(T_s[ci]) else False
            # S bulge: S increases by > 1 from previous
            S_bulge = False
            if ci > 0 and not np.isnan(S_s[ci]) and not np.isnan(S_s[ci - 1]):
                S_bulge = (S_s[ci] - S_s[ci - 1]) > 1.0

            if cos_drop and (T_spike or S_bulge):
                transitions_per_series[si].append(ci)

    # ── visualization ──────────────────────────────────────────────────
    labels = [series_label(li, mod) for li, mod, _ in series_list]
    cycle_idx = np.arange(n_t - 1)

    # 1) 4-panel heatmap (cos, T, A, S)
    fig, axes = plt.subplots(2, 2, figsize=(18, max(10, n_s * 0.3)))
    panels = [
        (cos_arr, "cos(t) — subspace alignment", "RdYlGn"),
        (T_arr, "T(t) — subspace rotation", "inferno"),
        (A_arr[:, :-1] if A_arr.shape[1] > n_t - 1 else A_arr, "A(t) — Frobenius norm", "viridis"),
        (S_arr[:, :-1] if S_arr.shape[1] > n_t - 1 else S_arr, "S(t) — effective rank", "plasma"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("cycle pair index")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_yticks(range(n_s))
        ax.set_yticklabels(labels, fontsize=4.5)
    for ax in [axes[0, 1], axes[1, 1]]:
        ax.set_yticks([])

    # X ticks with actual cycle numbers
    for ax in axes.flat:
        ticks = np.arange(0, n_t - 1, max(1, (n_t - 1) // 8))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{cycles[i]}→{cycles[i + 1]}" for i in ticks], fontsize=6, rotation=45)

    fig.suptitle("Learning Phase Structure — Absolute LoRA State\ny: output(L31) → input(L24)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'phase_heatmaps.png'}")

    # 2) Combined cos + loss line plot (main figure)
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()

    # Plot loss on right axis
    valid_loss_idx = [i for i, c in enumerate(cycles) if c in loss_map]
    valid_loss_vals = [loss_map[cycles[i]] for i in valid_loss_idx]
    ax2.plot(valid_loss_idx, valid_loss_vals, "k-", linewidth=2, alpha=0.6, label="valid_loss")
    ax2.set_ylabel("valid_loss", fontsize=11, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Plot cos per series, grouped by attn/mlp
    attn_idx = [si for si, (_, _, g) in enumerate(series_list) if g == "attn"]
    mlp_idx = [si for si, (_, _, g) in enumerate(series_list) if g == "mlp"]

    # Plot mean cos per group with band
    for idx_list, color, name in [(mlp_idx, "#FF5722", "mlp"), (attn_idx, "#2196F3", "attn")]:
        if not idx_list:
            continue
        cos_group = cos_arr[idx_list]
        mean_cos = np.nanmean(cos_group, axis=0)
        std_cos = np.nanstd(cos_group, axis=0)
        ax1.plot(cycle_idx, mean_cos, color=color, linewidth=1.5, label=f"cos mean ({name})")
        ax1.fill_between(cycle_idx, mean_cos - std_cos, mean_cos + std_cos,
                         color=color, alpha=0.15)

    # Mark transition candidates with vertical lines
    all_trans_cycles = set()
    for si, trans_list in enumerate(transitions_per_series):
        for ci in trans_list:
            all_trans_cycles.add(ci)

    for tc in sorted(all_trans_cycles):
        ax1.axvline(x=tc, color="red", linewidth=0.5, alpha=0.4, linestyle="--")

    ax1.set_xlabel("cycle pair index")
    ax1.set_ylabel("cos(t) = subspace alignment", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # X labels
    ticks = np.arange(0, n_t - 1, max(1, (n_t - 1) // 12))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f"{cycles[i]}→{cycles[i + 1]}" for i in ticks], fontsize=7, rotation=45)

    fig.suptitle("Phase Structure: cos(t) + valid_loss (transition candidates = red dashes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "cos_loss_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'cos_loss_overlay.png'}")

    # 3) Per-series cos with transition markers (detailed)
    # Pick 6 representative series: 3 attn + 3 mlp, spread across depths
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ax, (idx_list, grp_name) in zip(axes, [(attn_idx, "attn"), (mlp_idx, "mlp")]):
        n = len(idx_list)
        if n == 0:
            continue
        picks = [0, n // 2, n - 1]
        descs = ["output-most", "middle", "input-most"]
        colors = ["#1f77b4", "#2ca02c", "#d62728"]
        for pi, (mi, desc) in enumerate(zip(picks, descs)):
            si = idx_list[mi]
            li, mod, _ = series_list[si]
            ax.plot(cycle_idx, cos_arr[si], color=colors[pi], linewidth=1.2,
                    label=f"L{li:02d} {series_label(li, mod).split(' ', 1)[1]} ({desc})")
            # Mark transitions
            for tc in transitions_per_series[si]:
                ax.axvline(x=tc, color=colors[pi], linewidth=0.8, alpha=0.3, linestyle=":")

        ax.set_ylabel("cos(t)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{grp_name.upper()} — cos(t) with transition markers")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Overlay loss
    for ax in axes:
        ax2 = ax.twinx()
        ax2.plot(valid_loss_idx, valid_loss_vals, "k-", linewidth=1.5, alpha=0.3)
        ax2.set_ylabel("valid_loss", fontsize=8, alpha=0.5)

    axes[1].set_xlabel("cycle pair index")
    ticks = np.arange(0, n_t - 1, max(1, (n_t - 1) // 12))
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([f"{cycles[i]}→{cycles[i + 1]}" for i in ticks], fontsize=7, rotation=45)

    fig.suptitle("Per-Series Phase Detail: cos(t) + loss overlay", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "cos_per_series_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'cos_per_series_detail.png'}")

    # ── CSV output ─────────────────────────────────────────────────────
    csv_path = output_dir / "phase_quantities.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["series", "layer", "module", "group"]
        for ci in range(n_t):
            header.append(f"A_cycle_{cycles[ci]}")
        for ci in range(n_t):
            header.append(f"S_cycle_{cycles[ci]}")
        for ci in range(n_t - 1):
            header.append(f"cos_{cycles[ci]}→{cycles[ci + 1]}")
        for ci in range(n_t - 1):
            header.append(f"T_{cycles[ci]}→{cycles[ci + 1]}")
        header.append("transition_cycle_pairs")
        writer.writerow(header)

        for si, (li, mod, grp) in enumerate(series_list):
            row = [series_label(li, mod), li, mod, grp]
            row.extend(A_arr[si])
            row.extend(S_arr[si])
            row.extend(cos_arr[si])
            row.extend(T_arr[si])
            row.append(";".join(str(c) for c in transitions_per_series[si]))
            writer.writerow(row)
    print(f"Saved: {csv_path}")

    # ── transition summary ─────────────────────────────────────────────
    print(f"\n=== Transition Summary ===")
    n_trans_total = sum(len(t) for t in transitions_per_series)
    print(f"Total transition candidates: {n_trans_total}")
    trans_by_cycle = defaultdict(int)
    for si, tl in enumerate(transitions_per_series):
        for ci in tl:
            trans_by_cycle[ci] += 1

    if trans_by_cycle:
        print(f"\nTransition density by cycle pair:")
        for ci in sorted(trans_by_cycle.keys()):
            n_series = trans_by_cycle[ci]
            pct = n_series / n_s * 100
            c0, c1 = cycles[ci], cycles[ci + 1] if ci + 1 < n_t else "?"
            loss_str = ""
            if c1 in loss_map:
                loss_str = f"  loss={loss_map[c1]:.4f}"
            print(f"  {c0}→{c1}: {n_series}/{n_s} series ({pct:.0f}%){loss_str}")

    # ── cross-validation with loss ─────────────────────────────────────
    print(f"\n=== Loss Cross-Validation ===")
    for ci in sorted(trans_by_cycle.keys()):
        c0, c1 = cycles[ci], cycles[min(ci + 1, n_t - 1)]
        if c1 in loss_map and (c1 - 1) in loss_map:
            l_before = loss_map.get(c0, np.nan)
            l_at = loss_map.get(c1, np.nan)
            l_after = loss_map.get(c1 + 1, np.nan) if c1 + 1 in loss_map else np.nan
            if not np.isnan(l_before) and not np.isnan(l_at):
                delta = l_at - l_before
                stalled = abs(delta) < 0.005 or delta > 0
                status = "STALL/UP" if stalled else "descent"
                print(f"  {c0}→{c1}: loss {l_before:.4f}→{l_at:.4f} (Δ={delta:+.4f}) → {status}")

    print(f"\n{'='*50}")
    print(f"PRELIMINARY — {n_t} checkpoints analyzed so far")
    print(f"Run is still in progress. Re-run after completion for full 120-cycle analysis.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
