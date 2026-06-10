"""Layer×Mode structure visualization and eigenmode analysis.

Produces 4 figures + B-filter surrogate test + SNR map.
Input: r=2 P1 run (11 cycles) trajectory deltas. No additional training.
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARTIFACT_DIR = Path("runs/p1_r2_spectrum/trajectory_delta_artifacts")
OUTPUT_DIR = Path("runs/p1_r2_layer_mode_analysis")
N_SURROGATES = 1000
RNG = np.random.RandomState(42)


def top_left_sv_lowrank(B_np, A_np):
    """Top left singular vector of M = B @ A using 2x2 core (r=2)."""
    G = A_np @ A_np.T  # [2, 2]
    eigenvalues, Q = np.linalg.eigh(G)
    v = B_np @ Q[:, -1]  # largest eigenvalue direction
    return v / (np.linalg.norm(v) + 1e-12)


def participation_ratio(sigma):
    """PR = (Σσ²)² / Σσ⁴. Returns effective number of modes."""
    s2 = sigma ** 2
    return float((s2.sum()) ** 2 / (s2 ** 2).sum()) if (s2 ** 2).sum() > 1e-15 else 0.0


def marchenko_pastour_pr_null(m, n, n_surr=500):
    """Expected PR for random [m, n] matrix (m < n)."""
    prs = []
    for _ in range(n_surr):
        M = RNG.randn(m, n)
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        prs.append(participation_ratio(s))
    return np.mean(prs), np.std(prs)


def norm_preserving_surrogate(vectors, n_surr=1000):
    """Shuffle cycle order to get null for temporal stability metrics."""
    n = len(vectors)
    cos_vals = []
    for _ in range(n_surr):
        perm = RNG.permutation(n)
        cos_list = []
        for t in range(1, n):
            a = vectors[perm[t]].flatten()
            b = vectors[perm[t - 1]].flatten()
            cos_list.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos_vals.append(np.mean(cos_list))
    return np.array(cos_vals)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(ARTIFACT_DIR.glob("*.pt"))
    pilot_files = [f for f in files if "after_pilot" in f.name]
    if not pilot_files:
        pilot_files = files
    n_cycles = len(pilot_files)
    print(f"=== Layer×Mode Eigenmode Analysis ===")
    print(f"Cycles: {n_cycles}, Surrogates: {N_SURROGATES}")

    # Load all deltas
    print("Loading...")
    art0 = torch.load(pilot_files[0], map_location="cpu", weights_only=False)
    tnames = sorted(art0["delta_tensors"].keys())
    del art0
    n_t = len(tnames)

    all_dA = {tn: [] for tn in tnames}
    all_dB = {tn: [] for tn in tnames}
    for f in pilot_files:
        art = torch.load(f, map_location="cpu", weights_only=False)
        for tn in tnames:
            all_dA[tn].append(art["delta_tensors"][tn].float())
            all_dB[tn].append(art["delta_tensors"][tn.replace("lora_A", "lora_B")].float())
        del art

    # Parse tensor metadata
    def parse(tn):
        parts = tn.split(".")
        layer = module = lora_type = None
        for i, p in enumerate(parts):
            if p == "layers":
                layer = int(parts[i + 1])
                rest = parts[i + 2:]
                for j, q in enumerate(rest):
                    if q.startswith("lora_"):
                        module = ".".join(rest[:j])
                        break
                break
        lora_type = "A" if "lora_A" in tn else "B"
        return layer, module, lora_type

    # ─── Compute metrics per tensor ───
    print("Computing per-tensor metrics...")
    results = {}

    # A-side targets (lora_A only)
    a_tensors = [tn for tn in tnames if "lora_A" in tn]

    for tn in a_tensors:
        layer, module, lt = parse(tn)
        tn_B = tn.replace("lora_A", "lora_B")

        # Stack ΔA as [n_cycles, 2*4096] for temporal SVD
        dA_flat = np.array([d.flatten().numpy() for d in all_dA[tn]])  # [11, 8192]
        _, s_time_A, _ = np.linalg.svd(dA_flat, full_matrices=False)
        pr_time_A = participation_ratio(s_time_A)

        # Vel_cos per cycle pair
        dirs_A = []
        for d in all_dA[tn]:
            _, _, Vh = np.linalg.svd(d.numpy(), full_matrices=False)
            dirs_A.append(Vh[0])  # [4096] feature-space direction

        vel_cos_A = []
        for t in range(1, n_cycles):
            c = np.dot(dirs_A[t], dirs_A[t - 1]) / (
                np.linalg.norm(dirs_A[t]) * np.linalg.norm(dirs_A[t - 1]) + 1e-12
            )
            vel_cos_A.append(float(c))
        vel_cos_A = np.array(vel_cos_A)

        # Early vel_cos (cycles 0-3, pairs 0-2)
        early_vel = vel_cos_A[:3].mean() if len(vel_cos_A) >= 3 else vel_cos_A.mean()
        late_vel = vel_cos_A[5:].mean() if len(vel_cos_A) > 5 else 0.0

        # Norms per cycle
        norms_A = np.array([d.norm().item() for d in all_dA[tn]])

        # B-filtered A_signal stability
        B_cum = torch.zeros_like(all_dB[tn][0])
        a_sig_dirs = []
        for c in range(n_cycles):
            B_prev = B_cum.clone()
            A_sig = B_prev @ all_dA[tn][c]
            if A_sig.norm() > 1e-10:
                a_sig_dirs.append(top_left_sv_lowrank(B_prev.numpy(), all_dA[tn][c].numpy()))
            else:
                a_sig_dirs.append(None)
            B_cum = B_cum + all_dB[tn][c]

        a_sig_cos = []
        for t in range(1, len(a_sig_dirs)):
            if a_sig_dirs[t] is not None and a_sig_dirs[t - 1] is not None:
                c = np.dot(a_sig_dirs[t], a_sig_dirs[t - 1]) / (
                    np.linalg.norm(a_sig_dirs[t]) * np.linalg.norm(a_sig_dirs[t - 1]) + 1e-12
                )
                a_sig_cos.append(float(c))
        a_sig_cos = np.array(a_sig_cos)
        a_sig_stability = np.abs(a_sig_cos).mean() if len(a_sig_cos) > 0 else 0.0

        # B-side metrics
        dB_flat = np.array([d.flatten().numpy() for d in all_dB[tn]])
        _, s_time_B, _ = np.linalg.svd(dB_flat, full_matrices=False)
        pr_time_B = participation_ratio(s_time_B)

        # Per-cycle rank-1 dominance and stable rank
        r1_B_per_cycle = []
        sr_B_per_cycle = []
        for d in all_dB[tn]:
            s = np.linalg.svd(d.numpy(), compute_uv=False)
            s2 = s ** 2
            total = s2.sum().item()
            r1 = float(s2[0]) / total if total > 1e-15 else 0
            r1_B_per_cycle.append(r1)
            sr = total / (float(s[0]) ** 2) if float(s[0]) > 1e-15 else 0
            sr_B_per_cycle.append(float(sr))

        # Module type short name
        mod_short = module.split(".")[0] if module else "unknown"
        # Distinguish attention subtypes
        if module:
            if "out_proj" in module:
                mod_detail = "attn.out_proj"
            elif "in_proj_qkv" in module:
                mod_detail = "attn.in_qkv"
            elif "in_proj_a" in module:
                mod_detail = "attn.in_a"
            elif "in_proj_b" in module:
                mod_detail = "attn.in_b"
            elif "in_proj_z" in module:
                mod_detail = "attn.in_z"
            elif "o_proj" in module:
                mod_detail = "self_attn.o"
            elif "q_proj" in module:
                mod_detail = "self_attn.q"
            elif "k_proj" in module:
                mod_detail = "self_attn.k"
            elif "v_proj" in module:
                mod_detail = "self_attn.v"
            elif "gate_proj" in module:
                mod_detail = "mlp.gate"
            elif "up_proj" in module:
                mod_detail = "mlp.up"
            elif "down_proj" in module:
                mod_detail = "mlp.down"
            else:
                mod_detail = mod_short
        else:
            mod_detail = "?"

        results[tn] = {
            "layer": layer,
            "module": module,
            "mod_detail": mod_detail,
            "mod_short": mod_short,
            "pr_time_A": pr_time_A,
            "pr_time_B": pr_time_B,
            "early_vel": early_vel,
            "late_vel": late_vel,
            "vel_cos_all": vel_cos_A,
            "norms_A": norms_A,
            "r1_B_mean": np.mean(r1_B_per_cycle),
            "r1_B_per_cycle": np.array(r1_B_per_cycle),
            "sr_B_mean": np.mean(sr_B_per_cycle),
            "sr_B_per_cycle": np.array(sr_B_per_cycle),
            "a_sig_stability": a_sig_stability,
            "a_sig_cos": a_sig_cos,
            "dirs_A": dirs_A,
        }

    # ─── Null baselines ───
    print("Computing null baselines...")
    mp_pr_null_A, mp_pr_std_A = marchenko_pastour_pr_null(n_cycles, 2 * 4096, 200)
    mp_pr_null_B, mp_pr_std_B = marchenko_pastour_pr_null(n_cycles, 8192 * 2, 200)

    # Vel_cos null (cycle shuffle)
    vel_nulls = []
    for tn in a_tensors[:10]:
        null_dist = norm_preserving_surrogate(results[tn]["dirs_A"], 200)
        vel_nulls.append(null_dist.mean())
    vel_null_mean = np.mean(vel_nulls)

    # ─── B-filter surrogate test ───
    # Test on representative tensors only (3 strongest out_proj + 3 in_proj_qkv) for speed
    print("B-filter surrogate test (representative tensors, n=50)...")
    b_filter_test_tensors = [
        tn for tn in a_tensors
        if "out_proj" in (results[tn]["mod_detail"] or "")
    ][:3] + [
        tn for tn in a_tensors
        if "in_proj_qkv" in (results[tn]["mod_detail"] or "")
    ][:3]

    real_all = []
    shuf_all = []
    for tn in b_filter_test_tensors:
        r = results[tn]
        real_all.append(r["a_sig_stability"])

        # Surrogate: shuffle the ORDER of dB accumulation
        shuffled_stabs = []
        for si in range(50):
            perm = RNG.permutation(n_cycles)
            B_cum_s = torch.zeros_like(all_dB[tn][0])
            sig_cos_s = []
            prev_dir = None
            for c in range(n_cycles):
                B_prev_s = B_cum_s.clone()
                A_sig_s = B_prev_s @ all_dA[tn][c]
                B_cum_s = B_cum_s + all_dB[tn][perm[c]]  # shuffled B growth
                if A_sig_s.norm() > 1e-10:
                    cur_dir = top_left_sv_lowrank(B_cum_s.numpy(), all_dA[tn][c].numpy())
                    if prev_dir is not None:
                        cos_v = abs(np.dot(cur_dir, prev_dir) / (
                            np.linalg.norm(cur_dir) * np.linalg.norm(prev_dir) + 1e-12))
                        sig_cos_s.append(float(cos_v))
                    prev_dir = cur_dir
                else:
                    prev_dir = None
            if sig_cos_s:
                shuffled_stabs.append(np.mean(sig_cos_s))
        shuf_all.extend(shuffled_stabs)

    real_global = np.mean(real_all) if real_all else 0
    shuf_global = np.mean(shuf_all) if shuf_all else 0
    shuf_std = np.std(shuf_all) if len(shuf_all) > 1 else 1e-6
    z_filter = (real_global - shuf_global) / (shuf_std + 1e-12)

    print(f"\n{'='*60}")
    print("B-FILTER SURROGATE TEST")
    print(f"{'='*60}")
    print(f"  Real A_signal stability:    {real_global:.4f}")
    print(f"  B-shuffled stability:       {shuf_global:.4f}")
    print(f"  Δ = {real_global - shuf_global:+.4f}  z = {z_filter:.1f}σ")
    if z_filter > 2.0:
        print(f"  → 真フィルタ. BとΔAの協調あり. 増幅ギミックの本命.")
    else:
        print(f"  → B慣性. フィルタ仮説棄却.")

    # ─── FIGURE 1: A layer profile ───
    print("\nGenerating figures...")
    layers = sorted(set(r["layer"] for r in results.values()))
    attn_modules = {"attn.out_proj", "attn.in_qkv", "self_attn.o"}
    mlp_modules = {"mlp.gate", "mlp.up", "mlp.down"}

    # Aggregate per layer×module_type
    layer_attn = defaultdict(lambda: {"pr": [], "early_vel": [], "dir_cos_early": []})
    layer_mlp = defaultdict(lambda: {"pr": [], "early_vel": [], "dir_cos_early": []})

    for tn, r in results.items():
        l = r["layer"]
        d = r["mod_detail"]
        target = layer_attn if any(m in d for m in attn_modules) else layer_mlp
        target[l]["pr"].append(r["pr_time_A"])
        target[l]["early_vel"].append(r["early_vel"])
        # early_dir_cos from SNR map = early vel
        target[l]["dir_cos_early"].append(r["early_vel"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Figure 1: A Layer Profile (r=2, 11 cycles)", fontsize=14)

    metrics = [
        ("pr", "Temporal Participation Ratio (A)", "PR"),
        ("early_vel", "Early vel_cos (cycles 0-3)", "cos"),
        ("dir_cos_early", "Early dir_cos", "cos"),
    ]

    for ax_idx, (key, title, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        attn_means = [np.mean(layer_attn[l][key]) for l in layers]
        attn_stds = [np.std(layer_attn[l][key]) for l in layers]
        mlp_means = [np.mean(layer_mlp[l][key]) for l in layers]
        mlp_stds = [np.std(layer_mlp[l][key]) for l in layers]

        ax.errorbar(layers, attn_means, yerr=attn_stds, marker="o", label="Attention",
                     capsize=3, linewidth=2, markersize=8)
        ax.errorbar(layers, mlp_means, yerr=mlp_stds, marker="s", label="MLP",
                     capsize=3, linewidth=2, markersize=8)

        if key == "pr":
            ax.axhline(mp_pr_null_A, color="gray", linestyle="--", label=f"MP null ({mp_pr_null_A:.1f})")
            ax.axhline(n_cycles, color="lightgray", linestyle=":", alpha=0.5, label=f"Flat ({n_cycles})")
            ax.set_ylim(0, n_cycles + 1)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Layer")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_A_layer_profile.png", dpi=150)
    plt.close()
    print(f"  Saved fig1")

    # ─── FIGURE 2: B layer profile ───
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Figure 2: B Layer Profile (r=2, 11 cycles)", fontsize=14)

    b_metrics = {
        "pr_time_B": ("Temporal PR (B)", [np.mean([results[tn]["pr_time_B"]
             for tn in a_tensors if results[tn]["layer"] == l]) for l in layers]),
        "r1_B": ("Rank-1 Dominance (B)", [np.mean([results[tn]["r1_B_mean"]
             for tn in a_tensors if results[tn]["layer"] == l]) for l in layers]),
        "a_sig_stab": ("B-filtered A_signal Stability", [np.mean([results[tn]["a_sig_stability"]
             for tn in a_tensors if results[tn]["layer"] == l]) for l in layers]),
    }

    for ax_idx, (key, (title, vals)) in enumerate(b_metrics.items()):
        ax = axes[ax_idx]
        colors = []
        for l in layers:
            has_attn = any("attn" in results[tn]["mod_detail"]
                          for tn in a_tensors if results[tn]["layer"] == l)
            colors.append("#4477AA" if has_attn else "#EE7733")
        ax.bar(layers, vals, color=colors, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

        if key == "a_sig_stab":
            ax.axhline(shuf_global, color="red", linestyle="--",
                       label=f"B-shuffled null ({shuf_global:.3f})")
            ax.legend()

    axes[-1].set_xlabel("Layer")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_B_layer_profile.png", dpi=150)
    plt.close()
    print(f"  Saved fig2")

    # ─── FIGURE 3: Time × Layer phase map ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle("Figure 3: Time × Layer Phase Map", fontsize=14)

    # 3a: Norm heatmap
    norm_map = np.zeros((len(layers), n_cycles))
    for li, l in enumerate(layers):
        tns = [tn for tn in a_tensors if results[tn]["layer"] == l]
        for c in range(n_cycles):
            norms = [results[tn]["norms_A"][c] for tn in tns]
            norm_map[li, c] = np.mean(norms)

    ax = axes[0]
    im = ax.imshow(norm_map, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Cycle")
    ax.set_title("||ΔA|| mean")
    plt.colorbar(im, ax=ax)

    # 3b: vel_cos heatmap (per layer, cycle pairs)
    vel_map = np.zeros((len(layers), n_cycles - 1))
    for li, l in enumerate(layers):
        tns = [tn for tn in a_tensors if results[tn]["layer"] == l]
        for c in range(n_cycles - 1):
            vals = [results[tn]["vel_cos_all"][c] for tn in tns if c < len(results[tn]["vel_cos_all"])]
            vel_map[li, c] = np.mean(vals) if vals else 0

    ax = axes[1]
    vmax = max(abs(vel_map.min()), abs(vel_map.max()))
    im = ax.imshow(vel_map, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Cycle pair")
    ax.set_title("ΔA direction cos")
    plt.colorbar(im, ax=ax)
    # Mark cycle 6
    ax.axvline(x=5.5, color="yellow", linestyle="--", alpha=0.7, linewidth=2)

    # 3c: B rank-1 dominance heatmap
    r1_map = np.zeros((len(layers), n_cycles))
    for li, l in enumerate(layers):
        tns = [tn for tn in a_tensors if results[tn]["layer"] == l]
        for c in range(n_cycles):
            vals = [results[tn]["r1_B_per_cycle"][c] for tn in tns]
            r1_map[li, c] = np.mean(vals)

    ax = axes[2]
    im = ax.imshow(r1_map, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Cycle")
    ax.set_title("B rank-1 dominance")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_time_layer_phase_map.png", dpi=150)
    plt.close()
    print(f"  Saved fig3")

    # ─── FIGURE 4: Attention zoom ───
    # Pick strongest layers: out_proj with highest early_vel
    zoom_candidates = [(tn, r) for tn, r in results.items() if "out_proj" in (r["mod_detail"] or "")]
    zoom_candidates.sort(key=lambda x: x[1]["early_vel"], reverse=True)
    zoom_top = zoom_candidates[:3]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Figure 4: Attention Zoom (top out_proj layers)", fontsize=14)

    # 4a: Cycle-by-cycle direction cosine
    ax = axes[0, 0]
    for tn, r in zoom_top:
        short = f"L{r['layer']} {r['mod_detail']}"
        ax.plot(range(1, n_cycles), r["vel_cos_all"], marker="o", label=short, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(vel_null_mean, color="red", linestyle=":", alpha=0.7, label=f"Shuffle null ({vel_null_mean:.3f})")
    ax.axvspan(0.5, 3.5, alpha=0.1, color="green", label="Early phase")
    ax.set_xlabel("Cycle pair")
    ax.set_ylabel("ΔA direction cos")
    ax.set_title("4a: Raw A-direction persistence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4b: Norms cycle-by-cycle
    ax = axes[0, 1]
    for tn, r in zoom_top:
        short = f"L{r['layer']} {r['mod_detail']}"
        ax.plot(range(n_cycles), r["norms_A"], marker="s", label=short, linewidth=2)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("||ΔA||")
    ax.set_title("4b: ΔA norm trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4c: B-filtered A_signal stability
    ax = axes[1, 0]
    for tn, r in zoom_top:
        short = f"L{r['layer']} {r['mod_detail']}"
        if len(r["a_sig_cos"]) > 0:
            ax.plot(range(2, 2 + len(r["a_sig_cos"])), np.abs(r["a_sig_cos"]),
                    marker="D", label=short, linewidth=2)
    ax.axhline(shuf_global, color="red", linestyle="--", alpha=0.7,
               label=f"B-shuffled null ({shuf_global:.3f})")
    ax.set_xlabel("Cycle pair (B@dA)")
    ax.set_ylabel("|cos| of B-filtered A_signal")
    ax.set_title("4c: B-filtered A_signal direction stability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    # 4d: Pairwise direction matrix for strongest tensor
    ax = axes[1, 1]
    tn_best, r_best = zoom_top[0]
    dirs = r_best["dirs_A"]
    pw = np.zeros((n_cycles, n_cycles))
    for i in range(n_cycles):
        for j in range(n_cycles):
            pw[i, j] = np.dot(dirs[i], dirs[j]) / (
                np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j]) + 1e-12
            )
    im = ax.imshow(pw, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_cycles))
    ax.set_yticks(range(n_cycles))
    ax.set_xticklabels(range(n_cycles))
    ax.set_yticklabels(range(n_cycles))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cycle")
    short_best = f"L{r_best['layer']} {r_best['mod_detail']}"
    ax.set_title(f"4d: Pairwise ΔA direction cos ({short_best})")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_attention_zoom.png", dpi=150)
    plt.close()
    print(f"  Saved fig4")

    # ─── Summary output ───
    print(f"\n{'='*70}")
    print("JUDGMENTS (pre-registered)")
    print(f"{'='*70}")

    # Judgment 1: A temporal mode count
    attn_prs = [results[tn]["pr_time_A"] for tn in a_tensors
                if any(m in results[tn]["mod_detail"] for m in attn_modules)]
    mlp_prs = [results[tn]["pr_time_A"] for tn in a_tensors
               if any(m in results[tn]["mod_detail"] for m in mlp_modules)]

    print(f"\n[1] A temporal mode count:")
    print(f"  MP null PR: {mp_pr_null_A:.2f} ± {mp_pr_std_A:.2f}")
    print(f"  Attention PR: mean={np.mean(attn_prs):.2f}  range=[{np.min(attn_prs):.2f}, {np.max(attn_prs):.2f}]")
    print(f"  MLP PR:       mean={np.mean(mlp_prs):.2f}  range=[{np.min(mlp_prs):.2f}, {np.max(mlp_prs):.2f}]")
    attn_z = (mp_pr_null_A - np.mean(attn_prs)) / (mp_pr_std_A + 1e-12)
    if attn_z > 2.0:
        print(f"  → Attention A軌跡は低モード(z={attn_z:.1f}σ): 外挿有望")
    else:
        print(f"  → Attention A軌跹はフラットに近い(z={attn_z:.1f}σ): 外挿困難")

    # Judgment 2: B-filter (already printed above)
    print(f"\n[2] B-filter hypothesis: (see above)")

    # Judgment 3: Phase map - cycle 6 sync
    print(f"\n[3] Phase sync at cycle 6:")
    cycle6_norms = norm_map[:, 5:7]  # cycles 5-6
    drops = (cycle6_norms[:, 1] / (cycle6_norms[:, 0] + 1e-12))
    print(f"  Norm ratio cycle6/cycle5: {[f'{d:.2f}' for d in drops]}")
    all_drop = (drops < 0.7).all()
    if all_drop:
        print(f"  → 全層同期: 学習全体のphase遷移")
    else:
        print(f"  → 層ごとに非同期: 層間位相の可能性")

    # Save data
    import json
    summary = {
        "mp_null_pr_A": float(mp_pr_null_A),
        "attn_pr_mean": float(np.mean(attn_prs)),
        "mlp_pr_mean": float(np.mean(mlp_prs)),
        "real_a_sig_stability": float(real_global),
        "shuffled_a_sig_stability": float(shuf_global),
        "b_filter_z": float(z_filter),
        "vel_null_mean": float(vel_null_mean),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("DONE")


if __name__ == "__main__":
    main()
