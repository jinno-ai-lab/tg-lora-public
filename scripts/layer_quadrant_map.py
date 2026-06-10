#!/usr/bin/env python3
"""Layer-level 3-classification quadrant map (Continue / Freeze / Shake).

Uses existing r=2 trajectory snapshots (11 cycles) to classify each layer per phase
based on two axes:
  - S_ℓ: direction stability (mean adjacent-step cosine of ΔW velocity)
  - N_ℓ: amplitude (ΔW norm, with decay-rate correction for phase 2)

Null baseline via randomized-surrogate direction shuffling ensures S_ℓ significance.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ── phase boundaries (cycle indices) ──────────────────────────────────
PHASES = {
    "phase1": list(range(0, 6)),   # cycles 0-5
    "transition": [6],             # cycle 6
    "phase2": list(range(7, 11)),  # cycles 7-10
}

MODULE_COLORS = {
    "self_attn.q_proj": "#e41a1c",
    "self_attn.k_proj": "#377eb8",
    "self_attn.v_proj": "#4daf4a",
    "self_attn.o_proj": "#984ea3",
    "mlp.gate_proj":   "#ff7f00",
    "mlp.up_proj":     "#ffff33",
    "mlp.down_proj":   "#a65628",
    "linear_attn.in_proj_a":  "#f781bf",
    "linear_attn.in_proj_b":  "#999999",
    "linear_attn.in_proj_qkv": "#66c2a5",
    "linear_attn.in_proj_z":  "#fc8d62",
    "linear_attn.out_proj":   "#8da0cb",
}

CLASS_COLORS = {"continue": "#2ca02c", "freeze": "#1f77b4", "shake": "#d62728"}


def _parse_module(key: str) -> str:
    """Extract module name from a full parameter key."""
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return ".".join(parts[i + 2 : i + 4])
    return "unknown"


def _parse_layer(key: str) -> int:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return -1


def _lora_type(key: str) -> str:
    if ".lora_A." in key:
        return "A"
    if ".lora_B." in key:
        return "B"
    return "?"


def load_snapshots(art_dir: str):
    """Load trajectory delta snapshots, return dict[cycle] -> dict[key] -> tensor."""
    files = sorted(Path(art_dir).glob("*.pt"))
    snapshots = {}
    for f in files:
        name = f.stem
        # extract cycle number
        cyc_str = name.split("_")[-1]
        cyc = int(cyc_str)
        t = torch.load(str(f), map_location="cpu", weights_only=False)
        deltas = t["delta_tensors"]
        # keep original shapes for SVD, store as float
        data = {k: v.float() for k, v in deltas.items()}
        snapshots[cyc] = data
    return snapshots


def compute_velocity(snapshots, keys, cycles_in_phase):
    """Compute velocity (ΔW difference) between consecutive cycles for given phase."""
    vels = {}
    sorted_cyc = sorted(cycles_in_phase)
    for i in range(len(sorted_cyc) - 1):
        c0, c1 = sorted_cyc[i], sorted_cyc[i + 1]
        for k in keys:
            v = snapshots[c1][k] - snapshots[c0][k]
            vels.setdefault(k, []).append(v)
    return vels


def _rank1_direction(tensor_2d, shape):
    """Compute rank-1 SVD direction from a tensor.

    Returns a normalized 1-D direction vector (the dominant row or column).
    """
    if tensor_2d.norm().item() < 1e-12:
        return torch.zeros(max(shape))
    mat = tensor_2d.reshape(shape) if tensor_2d.ndim == 1 else tensor_2d
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    # Return the longer singular vector for more stable direction tracking
    if shape[0] >= shape[1]:
        return u[:, 0]  # left singular vector, shape [rows]
    else:
        return vh[0, :]  # right singular vector, shape [cols]


def compute_S_and_N(snapshots, keys, phase_cycles):
    """Compute direction stability S and amplitude N for each key in a phase.

    S_ℓ (rank-1 direction drift): tracks how the rank-1 principal direction
    of cumulative ΔW drifts between cycles.
    For each cycle, compute SVD of ΔW → top singular vector.
    S = mean cos(u1[cycle_i], u1[cycle_i+1]) for consecutive cycles.
    High → consistent learning direction (signal).
    Low → direction wanders (noise).

    N_ℓ: norm of cumulative ΔW at end of phase (how far the layer moved).
    For phase2, decay-rate distinguishes still-moving vs plateaued.

    Also returns per-tensor original shapes for SVD computation.
    """
    sorted_cyc = sorted(phase_cycles)
    # Get original shapes from first snapshot
    original_shapes = {k: tuple(snapshots[sorted_cyc[0]][k].shape) for k in keys
                      if snapshots[sorted_cyc[0]][k].ndim >= 2}
    # For any 1-D tensors (shouldn't happen), use as-is
    for k in keys:
        if k not in original_shapes:
            original_shapes[k] = tuple(snapshots[sorted_cyc[0]][k].shape)

    if len(sorted_cyc) < 2:
        norms = {}
        for k in keys:
            norms[k] = snapshots[sorted_cyc[0]][k].norm().item()
        S = {k: 0.0 for k in keys}
        return S, norms, {}, original_shapes

    # rank-1 direction at each cycle
    directions = {k: [] for k in keys}
    for c in sorted_cyc:
        for k in keys:
            d = _rank1_direction(snapshots[c][k], original_shapes[k])
            directions[k].append(d)

    # S: mean cosine between consecutive rank-1 directions
    S = {}
    for k in keys:
        ds = directions[k]
        if len(ds) < 2:
            S[k] = 0.0
            continue
        cosines = []
        for i in range(len(ds) - 1):
            n0, n1 = ds[i].norm().item(), ds[i + 1].norm().item()
            if n0 < 1e-12 or n1 < 1e-12:
                cosines.append(0.0)
                continue
            cos = (ds[i] @ ds[i + 1]).item() / (n0 * n1)
            cosines.append(max(-1.0, min(1.0, cos)))
        S[k] = float(np.mean(cosines)) if cosines else 0.0

    # N: norm of cumulative ΔW (at the last cycle of the phase)
    N_raw = {}
    for k in keys:
        N_raw[k] = snapshots[sorted_cyc[-1]][k].norm().item()

    # Decay rate: compare ΔW norm growth between first and second half
    decay_rate = {}
    mid = len(sorted_cyc) // 2
    for k in keys:
        first_half_norms = [snapshots[c][k].norm().item() for c in sorted_cyc[:mid + 1]]
        second_half_norms = [snapshots[c][k].norm().item() for c in sorted_cyc[mid:]]
        fh_growth = first_half_norms[-1] - first_half_norms[0] if len(first_half_norms) > 1 else 0
        sh_growth = second_half_norms[-1] - second_half_norms[0] if len(second_half_norms) > 1 else 0
        if abs(fh_growth) > 1e-12:
            decay_rate[k] = sh_growth / fh_growth
        else:
            decay_rate[k] = 0.0

    return S, N_raw, decay_rate, original_shapes


def compute_null_S(snapshots, keys, phase_cycles, original_shapes,
                   n_surrogates=200, seed=42):
    """Compute null distribution of S (rank-1 direction drift) by permuting directions.

    Null: randomly permute the order of rank-1 directions across cycles,
    then recompute the mean consecutive cosine. This destroys temporal structure
    while preserving the set of observed directions.
    """
    rng = np.random.RandomState(seed)
    sorted_cyc = sorted(phase_cycles)
    if len(sorted_cyc) < 3:
        return {k: {"mean": 0.0, "std": 1.0, "p95": 0.0} for k in keys}

    # compute rank-1 directions at each cycle
    directions = {k: [] for k in keys}
    for c in sorted_cyc:
        for k in keys:
            d = _rank1_direction(snapshots[c][k], original_shapes[k])
            directions[k].append(d)

    null_S = {k: [] for k in keys}
    for _ in range(n_surrogates):
        for k in keys:
            ds = directions[k]
            if len(ds) < 2:
                null_S[k].append(0.0)
                continue
            perm = rng.permutation(len(ds)).tolist()
            shuffled = [ds[i] for i in perm]
            cosines = []
            for i in range(len(shuffled) - 1):
                n0, n1 = shuffled[i].norm().item(), shuffled[i + 1].norm().item()
                if n0 < 1e-12 or n1 < 1e-12:
                    cosines.append(0.0)
                    continue
                cos = (shuffled[i] @ shuffled[i + 1]).item() / (n0 * n1)
                cosines.append(max(-1.0, min(1.0, cos)))
            null_S[k].append(float(np.mean(cosines)) if cosines else 0.0)

    null_stats = {}
    for k in keys:
        vals = np.array(null_S[k])
        null_stats[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()) if len(vals) > 1 else 1.0,
            "p95": float(np.percentile(vals, 95)) if len(vals) > 1 else 0.0,
        }
    return null_stats


def classify_layer(S_val, N_norm, null_p95, decay_rate, N_median, phase_name):
    """Classify a layer into continue / freeze / shake.

    Rules:
      - If S < null_p95 → shake (direction is noise-level)
      - If S >= null_p95 (significant direction):
          - If amplitude N is above median AND decay < 1 (still shrinking) → continue
          - If amplitude N is below median OR decay ≈ 1 (plateaued) → freeze
    """
    if S_val < null_p95:
        return "shake"
    # Direction is significant
    # For phase2: use decay rate to distinguish continue vs freeze
    if phase_name == "phase2":
        if decay_rate < 0.95:
            # still shrinking noticeably → continue
            return "continue"
        else:
            return "freeze"
    # For phase1: use amplitude relative to median
    if N_norm >= N_median:
        return "continue"
    else:
        return "freeze"


def main():
    parser = argparse.ArgumentParser(description="Layer quadrant map: 3-classification")
    parser.add_argument("--art-dir",
                        default="runs/p1_r2_spectrum/trajectory_delta_artifacts",
                        help="Directory with trajectory delta snapshots")
    parser.add_argument("--output-dir", default="runs/p1_r2_layer_quadrant_map",
                        help="Output directory")
    parser.add_argument("--n-surrogates", type=int, default=200,
                        help="Number of null surrogate samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading snapshots...")
    snapshots = load_snapshots(args.art_dir)
    all_cycles = sorted(snapshots.keys())
    print(f"  Loaded {len(all_cycles)} cycles: {all_cycles}")

    keys = sorted(snapshots[all_cycles[0]].keys())
    print(f"  {len(keys)} tensors")

    # ── per-phase computation ──────────────────────────────────────────
    results = {}
    all_S = {}
    all_N_raw = {}
    all_null = {}
    all_decay = {}

    original_shapes = None

    for phase_name, phase_cycles in PHASES.items():
        actual = [c for c in phase_cycles if c in snapshots]
        if not actual:
            continue
        print(f"\nProcessing {phase_name} (cycles {actual})...")

        S, N_raw, decay, shapes = compute_S_and_N(snapshots, keys, actual)
        if original_shapes is None:
            original_shapes = shapes
        null_stats = compute_null_S(snapshots, keys, actual, original_shapes,
                                    args.n_surrogates)

        all_S[phase_name] = S
        all_N_raw[phase_name] = N_raw
        all_null[phase_name] = null_stats
        all_decay[phase_name] = decay

    # ── normalize N across all phases for comparability ────────────────
    all_N_vals = []
    for phase_name in PHASES:
        if phase_name in all_N_raw:
            all_N_vals.extend(all_N_raw[phase_name].values())
    N_global_max = max(all_N_vals) if all_N_vals else 1.0

    # ── classify ───────────────────────────────────────────────────────
    classifications = {}  # (key, phase) -> class
    for phase_name in PHASES:
        if phase_name not in all_S:
            continue
        S = all_S[phase_name]
        N_raw = all_N_raw[phase_name]
        null = all_null[phase_name]
        decay = all_decay[phase_name]

        N_norm = {k: v / N_global_max for k, v in N_raw.items()}
        N_vals = sorted(N_norm.values())
        N_median = N_vals[len(N_vals) // 2] if N_vals else 0.5

        for k in keys:
            cls = classify_layer(
                S.get(k, 0.0), N_norm.get(k, 0.0),
                null.get(k, {}).get("p95", 0.0),
                decay.get(k, 1.0),
                N_median,
                phase_name,
            )
            classifications[(k, phase_name)] = cls

    # ── build structured output ────────────────────────────────────────
    layer_data = []
    for k in keys:
        module = _parse_module(k)
        layer = _parse_layer(k)
        lt = _lora_type(k)
        entry = {
            "key": k,
            "module": module,
            "layer": layer,
            "lora_type": lt,
        }
        for phase_name in PHASES:
            if phase_name not in all_S:
                continue
            entry[f"S_{phase_name}"] = all_S[phase_name].get(k, 0.0)
            entry[f"N_raw_{phase_name}"] = all_N_raw[phase_name].get(k, 0.0)
            entry[f"N_norm_{phase_name}"] = all_N_raw[phase_name].get(k, 0.0) / N_global_max
            entry[f"null_p95_{phase_name}"] = all_null[phase_name].get(k, {}).get("p95", 0.0)
            entry[f"decay_{phase_name}"] = all_decay[phase_name].get(k, 1.0)
            entry[f"class_{phase_name}"] = classifications.get((k, phase_name), "shake")
        layer_data.append(entry)

    # ── Figure 1: quadrant scatter per phase ───────────────────────────
    for phase_name in ["phase1", "phase2"]:
        if phase_name not in all_S:
            continue
        fig, ax = plt.subplots(figsize=(12, 8))

        # plot null threshold line
        null_p95_vals = [all_null[phase_name].get(k, {}).get("p95", 0.0) for k in keys]
        null_mean = float(np.mean(null_p95_vals))
        ax.axvline(null_mean, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Null p95 (mean={null_mean:.3f})")

        for k in keys:
            module = _parse_module(k)
            S_val = all_S[phase_name].get(k, 0.0)
            N_val = all_N_raw[phase_name].get(k, 0.0) / N_global_max
            cls = classifications.get((k, phase_name), "shake")
            color = MODULE_COLORS.get(module, "#333333")
            marker = {"continue": "o", "freeze": "s", "shake": "x"}[cls]
            size = 60 if cls == "continue" else (50 if cls == "freeze" else 40)
            ax.scatter(S_val, N_val, c=color, marker=marker, s=size,
                      alpha=0.8, edgecolors="black", linewidths=0.5)

        # legend
        module_patches = [mpatches.Patch(color=c, label=m)
                         for m, c in sorted(MODULE_COLORS.items())
                         if any(_parse_module(k) == m for k in keys)]
        class_markers = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=10, label="Continue"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                       markersize=10, label="Freeze"),
            plt.Line2D([0], [0], marker="x", color="gray",
                       markersize=10, label="Shake"),
        ]
        leg1 = ax.legend(handles=module_patches, title="Module",
                         loc="upper left", fontsize=7, ncol=2)
        leg2 = ax.legend(handles=class_markers, title="Classification",
                         loc="upper right", fontsize=8)
        ax.add_artist(leg1)

        ax.set_xlabel("Direction Stability S (cosine)", fontsize=12)
        ax.set_ylabel("Normalized Amplitude N", fontsize=12)
        ax.set_title(f"Layer Quadrant Map — {phase_name} (cycles "
                     f"{PHASES[phase_name][0]}-{PHASES[phase_name][-1]})", fontsize=14)
        ax.set_xlim(-0.3, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

        # add quadrant annotations
        ax.text(0.7, 0.9, "CONTINUE\n(stable + large)", fontsize=9, color="green",
                ha="center", va="center", alpha=0.6,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.2))
        ax.text(0.7, 0.15, "FREEZE\n(stable + small)", fontsize=9, color="blue",
                ha="center", va="center", alpha=0.6,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.2))
        ax.text(0.08, 0.5, "SHAKE\n(unstable)", fontsize=9, color="red",
                ha="center", va="center", alpha=0.6,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.2))

        plt.tight_layout()
        path = os.path.join(args.output_dir, f"fig1_{phase_name}_quadrant.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    # ── Figure 2: layer × phase heatmap ────────────────────────────────
    # Aggregate A+B per module-layer pair
    module_layers = sorted(set((e["module"], e["layer"]) for e in layer_data))
    phase_names = [p for p in ["phase1", "transition", "phase2"] if p in all_S]

    # Build matrix: row = (module, layer), col = phase
    class_to_int = {"continue": 2, "freeze": 1, "shake": 0}
    matrix = np.full((len(module_layers), len(phase_names)), -1, dtype=int)
    row_labels = []
    ml_class = {}  # (module, layer, phase) -> dominant class

    for (module, layer), _ in sorted(
        defaultdict(list).items()
    ):
        pass

    # For each (module, layer), aggregate across A and B tensors
    for ri, (module, layer) in enumerate(module_layers):
        row_labels.append(f"L{layer} {module[:20]}")
        for pi, phase_name in enumerate(phase_names):
            classes = [e[f"class_{phase_name}"]
                      for e in layer_data
                      if e["module"] == module and e["layer"] == layer
                      and f"class_{phase_name}" in e]
            if not classes:
                continue
            # dominant class
            counts = Counter(classes)
            dominant = counts.most_common(1)[0][0]
            # but if A and B disagree, mark as shake (conflicting signal)
            if len(set(classes)) > 1:
                # use the lower-energy class (shake > freeze > continue for safety)
                if "shake" in classes:
                    dominant = "shake"
                elif "freeze" in classes:
                    dominant = "freeze"
            matrix[ri, pi] = class_to_int[dominant]
            ml_class[(module, layer, phase_name)] = dominant

    fig, ax = plt.subplots(figsize=(8, max(8, len(module_layers) * 0.3)))
    cmap = matplotlib.colors.ListedColormap(
        [CLASS_COLORS["shake"], CLASS_COLORS["freeze"], CLASS_COLORS["continue"]]
    )
    mask = matrix >= 0
    if mask.any():
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Layer × Phase Classification Map", fontsize=14)
    ax.set_xlabel("Phase", fontsize=12)
    ax.set_ylabel("Layer (Module)", fontsize=12)

    # add text labels
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            if matrix[ri, ci] >= 0:
                label = ["Shake", "Freeze", "Continue"][matrix[ri, ci]]
                ax.text(ci, ri, label[0], ha="center", va="center",
                       fontsize=8, fontweight="bold", color="white")

    # legend
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c.capitalize())
                      for c in ["continue", "freeze", "shake"]]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
             bbox_to_anchor=(1.25, 1.0))

    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig2_layer_phase_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  CLASSIFICATION SUMMARY")
    print("=" * 80)

    summary = {}
    for phase_name in phase_names:
        phase_classes = [e[f"class_{phase_name}"]
                        for e in layer_data if f"class_{phase_name}" in e]
        counts = Counter(phase_classes)
        summary[phase_name] = {
            "total": len(phase_classes),
            "counts": dict(counts),
            "by_module": defaultdict(Counter),
        }
        for e in layer_data:
            if f"class_{phase_name}" in e:
                summary[phase_name]["by_module"][e["module"]][e[f"class_{phase_name}"]] += 1

        print(f"\n  {phase_name}: {dict(counts)}")
        for mod in sorted(summary[phase_name]["by_module"].keys()):
            mc = dict(summary[phase_name]["by_module"][mod])
            print(f"    {mod:40s} {mc}")

    # ── Pre-registered criteria evaluation ─────────────────────────────
    print("\n" + "=" * 80)
    print("  PRE-REGISTERED CRITERIA EVALUATION")
    print("=" * 80)

    # Criterion 1: classification correlates with module type
    for phase_name in phase_names:
        module_class_entropy = []
        for mod in summary[phase_name]["by_module"]:
            mc = summary[phase_name]["by_module"][mod]
            total = sum(mc.values())
            if total < 2:
                continue
            probs = [v / total for v in mc.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            module_class_entropy.append((mod, entropy, dict(mc)))
        module_class_entropy.sort(key=lambda x: x[1])
        print(f"\n  {phase_name} — Module purity (lower entropy = cleaner):")
        for mod, ent, mc in module_class_entropy:
            print(f"    {mod:40s} H={ent:.2f} bits  {mc}")

    # Criterion 2: phase transitions (how layers move between phases)
    if "phase1" in all_S and "phase2" in all_S:
        print("\n  Phase transitions (phase1 → phase2):")
        transitions = Counter()
        for e in layer_data:
            c1 = e.get("class_phase1", "?")
            c2 = e.get("class_phase2", "?")
            transitions[(c1, c2)] += 1
        for (c1, c2), count in sorted(transitions.items()):
            print(f"    {c1:10s} → {c2:10s}: {count:3d} tensors")

    # ── Save JSON results ──────────────────────────────────────────────
    output = {
        "layer_data": layer_data,
        "summary": {phase: {
            "counts": data["counts"],
            "by_module": {mod: dict(cnt) for mod, cnt in data["by_module"].items()},
        } for phase, data in summary.items()},
        "null_stats": {phase: {k: v for k, v in stats.items()}
                       for phase, stats in all_null.items()
                       for k, v in [("", None)]  # will rebuild below
                       },
        "N_global_max": N_global_max,
        "phases": {k: v for k, v in PHASES.items()},
    }
    # proper null_stats
    output["null_stats"] = {
        phase: {k: v for k, v in stats.items()}
        for phase, stats in all_null.items()
    }

    json_path = os.path.join(args.output_dir, "quadrant_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved {json_path}")

    # ── Criterion 3 note: C-level descent evidence ─────────────────────
    print("\n  ── C-level descent observations (§6) ──")
    for e in layer_data:
        module = e["module"]
        lt = e["lora_type"]
        for phase_name in ["phase1", "phase2"]:
            sk = f"S_{phase_name}"
            nk = f"N_raw_{phase_name}"
            if sk in e and nk in e:
                # Check if A and B within same module/layer show large S divergence
                pass
    # Check A vs B stability divergence per module-layer
    ab_divergence = []
    for module, layer in module_layers:
        for phase_name in ["phase1", "phase2"]:
            s_a = [e[f"S_{phase_name}"] for e in layer_data
                   if e["module"] == module and e["layer"] == layer
                   and e["lora_type"] == "A" and f"S_{phase_name}" in e]
            s_b = [e[f"S_{phase_name}"] for e in layer_data
                   if e["module"] == module and e["layer"] == layer
                   and e["lora_type"] == "B" and f"S_{phase_name}" in e]
            if s_a and s_b:
                div = abs(s_a[0] - s_b[0])
                ab_divergence.append((module, layer, phase_name, s_a[0], s_b[0], div))

    ab_divergence.sort(key=lambda x: -x[5])
    print("  Top A/B stability divergences:")
    for mod, layer, phase, sa, sb, div in ab_divergence[:10]:
        print(f"    L{layer} {mod:35s} {phase}: A={sa:.3f} B={sb:.3f} Δ={div:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
