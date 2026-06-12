#!/usr/bin/env python3
"""TG-LoRA 層内特異構造の時系列可視化 (M9 Run 後処理).

Progressive freezing frontier analysis via singular structure evolution.
Pure post-processing — no training interference.

Physical quantities per (series, cycle):
  A(s,t)  = Frobenius norm of ΔW (total LoRA strength)
  S(s,t)  = exp(entropy of σ²) — effective rank (1=rank-1 dominant, r=isotropic)
  T(s,t)  = mean sin(principal angles) of top-k subspaces between t and t+1
  dA/dt, dS/dt, dT/dt = raw adjacent differences

Surrogates: shuffled-cycle T distribution for noise floor estimation.
"""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── constants ──────────────────────────────────────────────────────────
RANK = 16
TOP_K = 4
N_SURROGATE_SHUFFLES = 20
SURROGATE_SEED = 12345
ARTIFACT_PREFIX = "tg_lora_after_pilot_cycle_"

# series classification: attn-family vs mlp-family
ATTN_PREFIXES = ("linear_attn.", "self_attn.")
MLP_PREFIX = "mlp."


# ── artifact I/O ───────────────────────────────────────────────────────
def discover_cycles(artifact_dir: Path) -> list[int]:
    """Return sorted list of available cycle indices."""
    cycles: set[int] = set()
    for f in artifact_dir.iterdir():
        if f.name.startswith(ARTIFACT_PREFIX) and f.name.endswith(".pt"):
            c = int(f.name.replace(ARTIFACT_PREFIX, "").replace(".pt", ""))
            cycles.add(c)
    return sorted(cycles)


def load_artifact(artifact_dir: Path, cycle: int) -> dict:
    path = artifact_dir / f"{ARTIFACT_PREFIX}{cycle:06d}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def discover_series(artifact_dir: Path) -> list[tuple[int, str, str]]:
    """Return [(layer_idx, module_name, series_group)] sorted output→input.

    series_group is 'attn' or 'mlp'.
    """
    data = load_artifact(artifact_dir, discover_cycles(artifact_dir)[0])
    series_set: set[tuple[int, str, str]] = set()
    pat = re.compile(r"layers\.(\d+)\.(.+?)\.lora_A\.")
    for key in data["delta_tensors"]:
        m = pat.search(key)
        if not m:
            continue
        li, mod = int(m.group(1)), m.group(2)
        group = "attn" if any(mod.startswith(p) for p in ATTN_PREFIXES) else "mlp"
        series_set.add((li, mod, group))

    # Sort: layer descending (31→24), then mlp before attn within layer
    # so that y-axis top = output-most mlp, bottom = input-most attn
    def sort_key(x):
        li, mod, grp = x
        grp_order = 0 if grp == "mlp" else 1
        return (-li, grp_order, mod)

    return sorted(series_set, key=sort_key)


# ── SVD via QR ─────────────────────────────────────────────────────────
def thin_svd_BA(B: torch.Tensor, A: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Singular values σ and right singular vectors V of ΔW = B·A.

    B: (d_out, r), A: (r, d_in).  Returns σ(r,) and V(d_in, r).
    Never forms the full (d_out, d_in) product.
    """
    Q_B, R_B = torch.linalg.qr(B)  # Q_B(d_out,r), R_B(r,r)
    M = R_B @ A  # (r, d_in)
    U_M, sigma, Vh_M = torch.linalg.svd(M, full_matrices=False)
    # Right singular vectors of ΔW = Q_B·U_M · sigma · Vh_M
    # Vh_M is (r, d_in), so right SVs as columns: V = Vh_M.T
    V = Vh_M.mH  # (d_in, r) — right singular vectors as columns
    return sigma.numpy().astype(np.float64), V.numpy().astype(np.float64)


# ── physical quantities ────────────────────────────────────────────────
def compute_frobenius(sigma: np.ndarray) -> float:
    """A = ||ΔW||_F via σ."""
    return float(np.sqrt(np.sum(sigma**2)))


def compute_effective_rank(sigma: np.ndarray) -> float:
    """S = exp(H) where H = -Σ p_i ln p_i, p_i = σ_i²/Σσ_j²."""
    p = sigma**2
    total = p.sum()
    if total < 1e-30:
        return 1.0
    p = p / total
    p = p[p > 1e-30]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def compute_time_twist(V_t: np.ndarray, V_tp1: np.ndarray, k: int = TOP_K) -> float:
    """Mean sin(principal angles) of top-k subspaces between consecutive cycles."""
    V1 = V_t[:, :k]  # (d, k)
    V2 = V_tp1[:, :k]
    # Gram matrix
    G = V1.T @ V2  # (k, k)
    # SVD to get cosines of principal angles
    s = np.linalg.svd(G, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    sin_angles = np.sqrt(1.0 - s**2)
    return float(np.mean(sin_angles))


# ── main computation pipeline ──────────────────────────────────────────
def extract_all_quantities(
    artifact_dir: Path,
    series_list: list[tuple[int, str, str]],
    cycles: list[int],
) -> dict:
    """Return {quantity_name: np.ndarray(series, cycle)} for A, S, T, dA, dS, dT."""
    n_s = len(series_list)
    n_t = len(cycles)

    A_arr = np.full((n_s, n_t), np.nan)
    S_arr = np.full((n_s, n_t), np.nan)

    # Store V history for T computation
    V_history = [[None] * n_t for _ in range(n_s)]

    print(f"Extracting SVD for {n_s} series × {n_t} cycles …")
    for ci, cycle in enumerate(cycles):
        data = load_artifact(artifact_dir, cycle)
        tensors = data["delta_tensors"]
        for si, (layer, mod, _grp) in enumerate(series_list):
            # Build A and B key patterns
            key_a = None
            key_b = None
            for k in tensors:
                if f"layers.{layer}.{mod}.lora_A." in k:
                    key_a = k
                if f"layers.{layer}.{mod}.lora_B." in k:
                    key_b = k
            if key_a is None or key_b is None:
                continue

            A_tensor = tensors[key_a].float()  # (r, d_in)
            B_tensor = tensors[key_b].float()  # (d_out, r)

            sigma, V = thin_svd_BA(B_tensor, A_tensor)
            A_arr[si, ci] = compute_frobenius(sigma)
            S_arr[si, ci] = compute_effective_rank(sigma)
            V_history[si][ci] = V

        if (ci + 1) % 10 == 0 or ci == n_t - 1:
            print(f"  cycle {cycle} ({ci+1}/{n_t})")

    # Compute T (time twist)
    T_arr = np.full((n_s, n_t), np.nan)
    for si in range(n_s):
        for ci in range(n_t - 1):
            v0 = V_history[si][ci]
            v1 = V_history[si][ci + 1]
            if v0 is not None and v1 is not None:
                T_arr[si, ci] = compute_time_twist(v0, v1, TOP_K)

    # Derivatives (raw adjacent differences)
    dA = np.full_like(A_arr, np.nan)
    dS = np.full_like(S_arr, np.nan)
    dT = np.full_like(T_arr, np.nan)
    dA[:, :-1] = np.diff(A_arr, axis=1)
    dS[:, :-1] = np.diff(S_arr, axis=1)
    dT[:, :-1] = np.diff(T_arr, axis=1)

    return {
        "A": A_arr,
        "S": S_arr,
        "T": T_arr,
        "dA_dt": dA,
        "dS_dt": dS,
        "dT_dt": dT,
        "V_history": V_history,
    }


# ── surrogates ─────────────────────────────────────────────────────────
def compute_surrogates(
    V_history: list[list[np.ndarray | None]],
    n_s: int,
    n_t: int,
    n_shuffles: int = N_SURROGATE_SHUFFLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return T_surrogate_mean, T_surrogate_p05, T_surrogate_p95.

    Shape: (n_s, n_t-1) for each.
    """
    rng = np.random.default_rng(SURROGATE_SEED)
    n_pairs = n_t - 1
    T_surr = np.full((n_s, n_pairs, n_shuffles), np.nan)

    print(f"Computing {n_shuffles} surrogates …")
    for si in range(n_s):
        # Collect valid V matrices
        valid_indices = [ci for ci in range(n_t) if V_history[si][ci] is not None]
        if len(valid_indices) < 2:
            continue
        valid_V = [V_history[si][ci] for ci in valid_indices]
        n_valid = len(valid_V)

        for sh in range(n_shuffles):
            perm = rng.permutation(n_valid)
            for pi in range(n_valid - 1):
                i0, i1 = perm[pi], perm[pi + 1]
                t_val = compute_time_twist(valid_V[i0], valid_V[i1], TOP_K)
                # Map to original time position
                orig_ci = min(valid_indices[pi], n_pairs - 1)
                T_surr[si, orig_ci, sh] = t_val

    surr_mean = np.nanmedian(T_surr, axis=2)
    surr_p05 = np.nanpercentile(T_surr, 5, axis=2)
    surr_p95 = np.nanpercentile(T_surr, 95, axis=2)
    return surr_mean, surr_p05, surr_p95


# ── convergence detection ──────────────────────────────────────────────
def detect_convergence_cycles(
    T: np.ndarray, surr_p05: np.ndarray, n_s: int, n_t: int
) -> list[int | None]:
    """For each series, find first cycle where T < surrogate 5th percentile.

    Must stay below for at least 3 consecutive cycles to confirm.
    """
    convergence: list[int | None] = []
    for si in range(n_s):
        found = None
        consecutive = 0
        for ci in range(n_t - 1):
            t_val = T[si, ci]
            floor = surr_p05[si, ci]
            if np.isnan(t_val) or np.isnan(floor):
                consecutive = 0
                continue
            if t_val < floor:
                consecutive += 1
                if consecutive >= 3 and found is None:
                    found = ci - 2  # first cycle of the run
            else:
                consecutive = 0
        convergence.append(found)
    return convergence


# ── visualization ──────────────────────────────────────────────────────
def make_series_labels(series_list: list[tuple[int, str, str]]) -> list[str]:
    labels = []
    for li, mod, grp in series_list:
        short = mod.replace("linear_attn.", "attn.").replace("self_attn.", "attn.")
        labels.append(f"L{li:02d} {short}")
    return labels


def plot_heatmaps(
    quantities: dict,
    series_labels: list[str],
    cycles: list[int],
    output_path: Path,
):
    """6-panel heatmap: 2 rows × 3 cols."""
    panels = [
        ("A", "A(s,t) = ||ΔW||_F  [Frobenius]", "coolwarm"),
        ("S", "S(s,t) = exp(H)  [effective rank]", "viridis"),
        ("dS_dt", "dS/dt  [Δ effective rank]", "RdBu_r"),
        ("T", "T(s,t) = subspace rotation  [time twist]", "inferno"),
        ("dT_dt", "dT/dt  [rotation acceleration]", "RdBu_r"),
        ("dA_dt", "dA/dt  [Δ Frobenius]", "RdBu_r"),
    ]

    n_s = len(series_labels)
    fig, axes = plt.subplots(2, 3, figsize=(20, max(12, n_s * 0.35)))
    axes = axes.flatten()

    for ax, (key, title, cmap) in zip(axes, panels):
        data = quantities[key]
        # Symmetric colormap for derivative panels
        if key in ("dA_dt", "dS_dt", "dT_dt"):
            vmax = np.nanmax(np.abs(data)) or 1.0
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("series" if ax in axes[:3] else "")
        ax.set_xlabel("cycle")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # Y-axis labels on leftmost panels
    for ax in [axes[0], axes[3]]:
        ax.set_yticks(range(n_s))
        ax.set_yticklabels(series_labels, fontsize=5)
    for ax in axes:
        if ax not in [axes[0], axes[3]]:
            ax.set_yticks([])

    # X-axis: show cycle numbers
    n_t = len(cycles)
    for ax in axes:
        tick_idx = np.linspace(0, n_t - 1, min(10, n_t), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(cycles[i]) for i in tick_idx], fontsize=7)

    fig.suptitle(
        "TG-LoRA Singular Structure Evolution (M9)\n"
        "y-axis: output → input  |  Progressive Freezing Frontier Analysis",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {output_path}")


def plot_line_overlay(
    quantities: dict,
    surr_mean: np.ndarray,
    surr_p05: np.ndarray,
    surr_p95: np.ndarray,
    series_list: list[tuple[int, str, str]],
    cycles: list[int],
    output_path: Path,
):
    """Line plots of T with surrogate bands, grouped by attn/mlp.

    Per group, show 3 representative depths: most-output, middle, most-input.
    """
    T = quantities["T"]
    n_t = len(cycles)
    t_axis = np.arange(n_t - 1)

    groups = {"attn": [], "mlp": []}
    for si, (li, mod, grp) in enumerate(series_list):
        groups[grp].append((si, li, mod))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 3))

    for ax, (grp_name, members) in zip(axes, groups.items()):
        n_members = len(members)
        if n_members == 0:
            continue

        # Pick 3 representative depths
        pick_indices = [0, n_members // 2, n_members - 1]
        labels_desc = ["output-most", "middle", "input-most"]

        for color_i, (mi, desc) in enumerate(zip(pick_indices, labels_desc)):
            si, li, mod = members[mi]
            short = mod.replace("linear_attn.", "attn.").replace("self_attn.", "attn.")
            t_vals = T[si, : n_t - 1]
            sm = surr_mean[si]
            sp5 = surr_p05[si]
            sp95 = surr_p95[si]

            ax.plot(t_axis, t_vals, color=colors[color_i], linewidth=1.5,
                    label=f"L{li:02d} {short} ({desc})")
            ax.fill_between(t_axis, sp5, sp95, color=colors[color_i], alpha=0.12)
            ax.plot(t_axis, sm, color=colors[color_i], linewidth=0.6, linestyle="--", alpha=0.5)

        ax.set_title(f"{grp_name.upper()} — T(s,t) with surrogate floor", fontweight="bold")
        ax.set_xlabel("cycle index")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("T = mean sin(principal angles)")
    fig.suptitle(
        "Time Twist T with Surrogate Noise Floor (dashed = surrogate median, band = p5–p95)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved line overlay: {output_path}")


def plot_convergence_summary(
    convergence: list[int | None],
    series_list: list[tuple[int, str, str]],
    cycles: list[int],
    output_path: Path,
):
    """Bar chart of convergence cycle per series."""
    n_s = len(series_list)
    labels = make_series_labels(series_list)
    conv_arr = np.array([c if c is not None else np.nan for c in convergence])

    fig, ax = plt.subplots(figsize=(max(10, n_s * 0.3), 5))
    x = np.arange(n_s)
    colors = ["#2196F3" if not np.isnan(c) else "#BBBBBB" for c in conv_arr]
    bars = ax.bar(x, conv_arr, color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Convergence cycle (first T < surrogate p5)")
    ax.set_title("Progressive Freezing Frontier — Convergence Detection per Series")
    ax.axhline(y=np.nanmedian(conv_arr), color="red", linestyle="--", linewidth=0.8,
               label=f"median = {np.nanmedian(conv_arr):.0f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence summary: {output_path}")


# ── CSV output ─────────────────────────────────────────────────────────
def save_csv(
    quantities: dict,
    convergence: list[int | None],
    series_list: list[tuple[int, str, str]],
    cycles: list[int],
    output_dir: Path,
):
    """Save full time-series table and convergence summary as CSV."""
    import csv

    n_s = len(series_list)
    n_t = len(cycles)
    labels = make_series_labels(series_list)

    # Full time-series
    ts_path = output_dir / "singular_structure_timeseries.csv"
    header = ["series", "layer", "module", "group"]
    for c in cycles:
        header.append(f"A_cycle_{c}")
    for c in cycles:
        header.append(f"S_cycle_{c}")
    for c in cycles[: n_t - 1]:
        header.append(f"T_cycle_{c}_{c+1}")
    for c in cycles[: n_t - 1]:
        header.append(f"dA_cycle_{c}_{c+1}")
    for c in cycles[: n_t - 1]:
        header.append(f"dS_cycle_{c}_{c+1}")
    for c in cycles[: n_t - 1]:
        header.append(f"dT_cycle_{c}_{c+1}")

    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for si, (li, mod, grp) in enumerate(series_list):
            row = [labels[si], li, mod, grp]
            row.extend(quantities["A"][si])
            row.extend(quantities["S"][si])
            row.extend(quantities["T"][si, : n_t - 1])
            row.extend(quantities["dA_dt"][si, : n_t - 1])
            row.extend(quantities["dS_dt"][si, : n_t - 1])
            row.extend(quantities["dT_dt"][si, : n_t - 1])
            writer.writerow(row)
    print(f"Saved timeseries CSV: {ts_path}")

    # Convergence summary
    conv_path = output_dir / "convergence_summary.csv"
    with open(conv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "layer", "module", "group", "convergence_cycle", "converged"])
        for si, (li, mod, grp) in enumerate(series_list):
            c = convergence[si]
            writer.writerow([labels[si], li, mod, grp, c if c is not None else "", c is not None])
    print(f"Saved convergence CSV: {conv_path}")


# ── gate decision ──────────────────────────────────────────────────────
def evaluate_gate(
    convergence: list[int | None],
    series_list: list[tuple[int, str, str]],
) -> str:
    """Evaluate GOAL §7 gate criteria."""
    # Separate convergence cycles by depth (layer index)
    attn_conv = [(li, c) for (li, _, grp), c in zip(series_list, convergence) if grp == "attn" and c is not None]
    mlp_conv = [(li, c) for (li, _, grp), c in zip(series_list, convergence) if grp == "mlp" and c is not None]

    if not attn_conv and not mlp_conv:
        return ("FAIL: No series shows convergence above surrogate floor. "
                "No convergence signal exists — freezing strategy needs rethinking.")

    all_conv = attn_conv + mlp_conv
    n_converged = len(all_conv)
    n_total = len(series_list)

    if n_converged < n_total * 0.3:
        return (f"WEAK: Only {n_converged}/{n_total} series converge. "
                "Insufficient for reliable freeze ordering.")

    # Check monotonic depth dependence: output layers should converge earlier
    layers = np.array([li for li, _ in all_conv])
    conv_cycles = np.array([c for _, c in all_conv])
    if len(layers) >= 3:
        corr = np.corrcoef(layers, conv_cycles)[0, 1]
        if corr > 0.3:
            return (f"PASS: Depth-dependent convergence detected (r={corr:.2f}). "
                    "Output layers converge earlier — progressive freezing frontier confirmed. "
                    "Proceed to Phase 0 → freezing implementation.")
        elif corr < -0.3:
            return (f"INVERTED: Input layers converge earlier (r={corr:.2f}). "
                    "Opposite of expected — front-loading input side may work better.")
        else:
            return (f"FLAT: No clear depth dependence (r={corr:.2f}). "
                    f"{n_converged}/{n_total} converge but not depth-ordered. "
                    "Freezing order has no empirical basis.")
    else:
        return f"INSUFFICIENT: Only {n_converged} converged series — cannot assess depth dependence."


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TG-LoRA singular structure visualization")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/tg_lora_9b_m9"),
                        help="Path to M9 run directory")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/tg_lora_9b_m9/singular_structure"),
                        help="Output directory for plots and CSVs")
    parser.add_argument("--n-shuffles", type=int, default=N_SURROGATE_SHUFFLES)
    args = parser.parse_args()

    artifact_dir = args.run_dir / "trajectory_delta_artifacts"
    if not artifact_dir.exists():
        print(f"ERROR: artifact dir not found: {artifact_dir}")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover data
    cycles = discover_cycles(artifact_dir)
    series_list = discover_series(artifact_dir)
    print(f"Cycles: {len(cycles)} ({cycles[0]}–{cycles[-1]})")
    print(f"Series: {len(series_list)} (layers {series_list[-1][0]}–{series_list[0][0]})")
    print()

    # Extract physical quantities
    quantities = extract_all_quantities(artifact_dir, series_list, cycles)
    n_s = len(series_list)
    n_t = len(cycles)

    # Surrogates
    surr_mean, surr_p05, surr_p95 = compute_surrogates(
        quantities["V_history"], n_s, n_t, args.n_shuffles
    )

    # Convergence detection
    convergence = detect_convergence_cycles(quantities["T"], surr_p05, n_s, n_t)

    # Visualization
    series_labels = make_series_labels(series_list)
    plot_heatmaps(quantities, series_labels, cycles, output_dir / "singular_heatmaps.png")
    plot_line_overlay(
        quantities, surr_mean, surr_p05, surr_p95,
        series_list, cycles, output_dir / "time_twist_lines.png",
    )
    plot_convergence_summary(convergence, series_list, cycles, output_dir / "convergence_summary.png")

    # CSV output
    save_csv(quantities, convergence, series_list, cycles, output_dir)

    # Gate decision
    gate = evaluate_gate(convergence, series_list)
    print(f"\n{'='*60}")
    print(f"GATE DECISION: {gate}")
    print(f"{'='*60}")

    # Save gate result
    gate_path = output_dir / "gate_decision.txt"
    gate_path.write_text(gate + "\n")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
