"""Per-tensor time series analysis of LoRA trajectory deltas.

Loads trajectory delta artifacts and analyzes temporal structure
at the per-tensor (per-layer, per-module) level.

Usage:
  python scripts/per_tensor_timeseries.py
  python scripts/per_tensor_timeseries.py --artifact-dir runs/p1_r2_spectrum/trajectory_delta_artifacts
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def autocorr(x, lag):
    """Autocorrelation at given lag."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < lag + 2:
        return 0.0
    m = x.mean()
    v = ((x - m) ** 2).sum()
    if v < 1e-15:
        return 0.0
    return float(((x[lag:] - m) * (x[:-lag] - m)).sum() / v)


def tensor_cos(a, b):
    """Cosine similarity between two tensors."""
    af = a.float().flatten()
    bf = b.float().flatten()
    d = torch.dot(af, bf).item()
    na = af.norm().item()
    nb = bf.norm().item()
    denom = na * nb
    return d / denom if denom > 1e-12 else 0.0


def angular_dist(v1, v2):
    """Angular distance (radians) between unit vectors."""
    cos = torch.dot(v1.flatten(), v2.flatten()).item()
    cos = max(-1.0, min(1.0, cos))
    return math.acos(cos)


def parse_tensor_name(tname):
    """Extract layer, module, lora_type from tensor name."""
    parts = tname.split(".")
    layer = module = lora_type = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            layer = int(parts[i + 1])
            rest = parts[i + 2 :]
            for j, q in enumerate(rest):
                if q.startswith("lora_"):
                    module = ".".join(rest[:j])
                    break
            break
    lora_type = "A" if "lora_A" in tname else "B"
    return layer, module, lora_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="runs/tg_lora_9b_m9/trajectory_delta_artifacts",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        run_name = Path(args.artifact_dir).parent.name
        args.output = f"runs/{run_name}_per_tensor_ts.json"

    # ─── Load file list (pilot only) ───
    all_files = sorted(Path(args.artifact_dir).glob("*.pt"))
    # Use only pilot artifacts for training trajectory analysis
    pilot_files = [f for f in all_files if "after_pilot" in f.name]
    if not pilot_files:
        pilot_files = all_files  # fallback if no pilot/speculative distinction

    n = len(pilot_files)
    print(f"=== Per-Tensor Time Series Analysis ===")
    print(f"Artifact dir: {args.artifact_dir}")
    print(f"Total files: {len(all_files)}, Pilot files: {n}")

    # Get tensor names from first file
    art0 = torch.load(pilot_files[0], map_location="cpu", weights_only=False)
    tnames = sorted(art0["delta_tensors"].keys())
    del art0
    print(f"Tensors per cycle: {len(tnames)}")

    # ─── Phase 1: Streaming extraction ───
    # Keep: norms, rank-1 ratios, feature-space directions, velocity cosine
    ts_norms = {tn: [] for tn in tnames}
    ts_r1ratios = {tn: [] for tn in tnames}
    ts_feat_dirs = {tn: [] for tn in tnames}
    ts_vel_cos = {tn: [] for tn in tnames}

    prev_deltas = None

    print("\nPhase 1: Extracting per-tensor stats...")
    for ci, f in enumerate(pilot_files):
        art = torch.load(f, map_location="cpu", weights_only=False)
        deltas = art["delta_tensors"]

        for tn in tnames:
            t = deltas[tn].float()
            norm = t.norm().item()
            ts_norms[tn].append(norm)

            # Rank-1 analysis via SVD
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            total_var = (S**2).sum().item()
            r1_ratio = (S[0] ** 2).item() / total_var if total_var > 1e-15 else 0.0
            ts_r1ratios[tn].append(r1_ratio)

            # Feature-space rank-1 direction
            if "lora_A" in tn:
                feat_dir = Vh[0].clone()  # [in_dim]
            else:
                feat_dir = U[:, 0].clone()  # [out_dim]
            ts_feat_dirs[tn].append(feat_dir)

            # Velocity autocorrelation: cos(delta_t, delta_{t-1})
            if prev_deltas is not None and tn in prev_deltas:
                cos_val = tensor_cos(t, prev_deltas[tn])
                ts_vel_cos[tn].append(cos_val)

        prev_deltas = {tn: deltas[tn].float().clone() for tn in tnames}
        del art

        if (ci + 1) % 10 == 0 or ci == n - 1:
            print(f"  {ci + 1}/{n}")

    # ─── Phase 2: Time series analysis ───
    print("\nPhase 2: Analyzing time series...")

    results = {}
    for tn in tnames:
        norms = np.array(ts_norms[tn])
        r1s = np.array(ts_r1ratios[tn])
        dirs = ts_feat_dirs[tn]
        vel_cos = np.array(ts_vel_cos[tn])

        layer, module, lora_type = parse_tensor_name(tn)

        # Norm time series
        norm_trend = float(np.polyfit(np.arange(len(norms)), norms, 1)[0])
        norm_ac = {lag: autocorr(norms, lag) for lag in [1, 2, 5, 10, 20]}

        # Rank-1 ratio time series
        r1_ac = {lag: autocorr(r1s, lag) for lag in [1, 5, 10]}

        # Direction drift: angular velocity and direction cosine
        angvels = []
        dir_cos_vals = []
        for t in range(1, len(dirs)):
            cos_val = torch.dot(dirs[t].flatten(), dirs[t - 1].flatten()).item() / (
                dirs[t].norm().item() * dirs[t - 1].norm().item() + 1e-12
            )
            cos_val = max(-1.0, min(1.0, cos_val))
            angvels.append(math.acos(cos_val))
            dir_cos_vals.append(cos_val)

        angvels = np.array(angvels)
        dir_cos_vals = np.array(dir_cos_vals)

        # Direction autocorrelation at lag > 1
        dir_ac = {}
        for lag in [1, 2, 5]:
            if len(dir_cos_vals) > lag:
                dir_ac[lag] = autocorr(dir_cos_vals, lag)
            else:
                dir_ac[lag] = 0.0

        # Phase analysis: early / mid / late
        n3 = len(norms) // 3
        phases = {
            "early": slice(0, n3),
            "mid": slice(n3, 2 * n3),
            "late": slice(2 * n3, None),
        }
        phase_r1 = {ph: float(r1s[s].mean()) for ph, s in phases.items()}
        phase_norm = {ph: float(norms[s].mean()) for ph, s in phases.items()}
        phase_vel = {}
        for ph, s in phases.items():
            idx_start = s.start if s.start < len(vel_cos) else 0
            idx_stop = min(s.stop, len(vel_cos)) if s.stop else len(vel_cos)
            if idx_start < idx_stop:
                phase_vel[ph] = float(vel_cos[idx_start:idx_stop].mean())
            else:
                phase_vel[ph] = 0.0
        phase_angvel = {}
        for ph, s in phases.items():
            idx_start = max(s.start - 1, 0)
            idx_stop = min(s.stop - 1 if s.stop else len(angvels), len(angvels))
            if idx_start < idx_stop:
                phase_angvel[ph] = float(angvels[idx_start:idx_stop].mean())
            else:
                phase_angvel[ph] = 0.0

        results[tn] = {
            "layer": layer,
            "module": module,
            "lora_type": lora_type,
            # Norm
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_cv": float(norms.std() / (norms.mean() + 1e-12)),
            "norm_trend": norm_trend,
            "norm_ac": norm_ac,
            # Rank-1 ratio
            "r1_mean": float(r1s.mean()),
            "r1_std": float(r1s.std()),
            "r1_min": float(r1s.min()),
            "r1_max": float(r1s.max()),
            "r1_ac": r1_ac,
            # Direction drift
            "angvel_mean": float(angvels.mean()) if len(angvels) > 0 else 0.0,
            "angvel_std": float(angvels.std()) if len(angvels) > 0 else 0.0,
            "dir_cos_mean": float(dir_cos_vals.mean()) if len(dir_cos_vals) > 0 else 0.0,
            "dir_cos_std": float(dir_cos_vals.std()) if len(dir_cos_vals) > 0 else 0.0,
            "dir_ac": dir_ac,
            # Velocity autocorrelation
            "vel_cos_mean": float(vel_cos.mean()) if len(vel_cos) > 0 else 0.0,
            "vel_cos_std": float(vel_cos.std()) if len(vel_cos) > 0 else 0.0,
            "vel_ac": {lag: autocorr(vel_cos, lag) for lag in [1, 2, 5]},
            # Phase analysis
            "phase_r1": phase_r1,
            "phase_norm": phase_norm,
            "phase_vel": phase_vel,
            "phase_angvel": phase_angvel,
        }

    # Free large direction vectors
    del ts_feat_dirs, prev_deltas

    # ─── Phase 3: Summary ───
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # --- By lora type ---
    for lt in ["A", "B"]:
        group = {k: v for k, v in results.items() if v["lora_type"] == lt}
        if not group:
            continue
        print(f"\n--- lora_{lt} (n={len(group)}) ---")
        for field in ["r1_mean", "dir_cos_mean", "vel_cos_mean", "angvel_mean"]:
            vals = [v[field] for v in group.values()]
            print(
                f"  {field:20s}: {np.mean(vals):+.4f} ± {np.std(vals):.4f}"
            )
        print(f"  norm_ac(1):           {np.mean([v['norm_ac'][1] for v in group.values()]):+.4f}")
        print(f"  vel_ac(1):            {np.mean([v['vel_ac'][1] for v in group.values()]):+.4f}")

    # --- By layer ---
    print(f"\n--- Per-layer summary ---")
    layers = sorted(set(v["layer"] for v in results.values() if v["layer"] is not None))
    print(f"{'Layer':>6s} {'r1_A':>7s} {'r1_B':>7s} {'dir_cos_A':>10s} {'dir_cos_B':>10s} {'vel_A':>7s} {'vel_B':>7s} {'angvel_B':>9s}")
    for layer in layers:
        gA = [v for v in results.values() if v["layer"] == layer and v["lora_type"] == "A"]
        gB = [v for v in results.values() if v["layer"] == layer and v["lora_type"] == "B"]
        r1A = np.mean([v["r1_mean"] for v in gA]) if gA else 0
        r1B = np.mean([v["r1_mean"] for v in gB]) if gB else 0
        dcA = np.mean([v["dir_cos_mean"] for v in gA]) if gA else 0
        dcB = np.mean([v["dir_cos_mean"] for v in gB]) if gB else 0
        vA = np.mean([v["vel_cos_mean"] for v in gA]) if gA else 0
        vB = np.mean([v["vel_cos_mean"] for v in gB]) if gB else 0
        avB = np.mean([v["angvel_mean"] for v in gB]) if gB else 0
        print(
            f"{layer:>6d} {r1A:>7.3f} {r1B:>7.3f} {dcA:>10.4f} {dcB:>10.4f} {vA:>7.4f} {vB:>7.4f} {avB:>9.4f}"
        )

    # --- By module type ---
    print(f"\n--- Per-module-type summary ---")
    mod_types = defaultdict(list)
    for v in results.values():
        if v["module"]:
            mod_short = v["module"].split(".")[0]
            mod_types[mod_short].append(v)

    print(f"{'Module':>15s} {'n':>4s} {'r1_mean':>8s} {'dir_cos':>8s} {'vel_cos':>8s} {'angvel':>8s}")
    for mtype in sorted(mod_types.keys()):
        g = mod_types[mtype]
        print(
            f"{mtype:>15s} {len(g):>4d} {np.mean([v['r1_mean'] for v in g]):>8.3f} "
            f"{np.mean([v['dir_cos_mean'] for v in g]):>8.4f} "
            f"{np.mean([v['vel_cos_mean'] for v in g]):>8.4f} "
            f"{np.mean([v['angvel_mean'] for v in g]):>8.4f}"
        )

    # --- Phase comparison ---
    print(f"\n--- Phase comparison (early vs late) ---")
    for phase in ["early", "mid", "late"]:
        r1 = [v["phase_r1"][phase] for v in results.values()]
        vel = [v["phase_vel"][phase] for v in results.values()]
        ang = [v["phase_angvel"][phase] for v in results.values()]
        print(
            f"  {phase:>5s}: r1={np.mean(r1):.3f}  vel_cos={np.mean(vel):+.4f}  angvel={np.mean(ang):.4f}"
        )

    # --- Top tensors by velocity autocorrelation ---
    print(f"\n--- Top 10 tensors by |vel_cos_mean| ---")
    sorted_by_vel = sorted(results.items(), key=lambda x: abs(x[1]["vel_cos_mean"]), reverse=True)
    for tn, r in sorted_by_vel[:10]:
        short = tn.replace("base_model.model.model.", "")
        print(
            f"  {short[:55]:55s} vel={r['vel_cos_mean']:+.4f}  r1={r['r1_mean']:.3f}  "
            f"dir_cos={r['dir_cos_mean']:.4f}  L{r['layer']}_{r['lora_type']}"
        )

    # --- Top tensors by direction stability ---
    print(f"\n--- Top 10 tensors by dir_cos_mean (lora_B only) ---")
    b_tensors = {k: v for k, v in results.items() if v["lora_type"] == "B"}
    sorted_by_dir = sorted(b_tensors.items(), key=lambda x: x[1]["dir_cos_mean"], reverse=True)
    for tn, r in sorted_by_dir[:10]:
        short = tn.replace("base_model.model.model.", "")
        print(
            f"  {short[:55]:55s} dir_cos={r['dir_cos_mean']:.4f}  vel={r['vel_cos_mean']:+.4f}  "
            f"r1={r['r1_mean']:.3f}  angvel={r['angvel_mean']:.4f}  L{r['layer']}"
        )

    # --- Statistical significance of vel_cos ---
    print(f"\n--- Velocity autocorrelation significance ---")
    all_vel_cos = [v["vel_cos_mean"] for v in results.values()]
    vel_arr = np.array(all_vel_cos)
    # Null: vel_cos ~ N(0, 1/sqrt(T)) where T is number of cycles
    T = n - 1  # vel_cos has T-1 values
    null_std = 1.0 / math.sqrt(T)
    global_mean_vel = vel_arr.mean()
    z_score = global_mean_vel / (null_std + 1e-12)
    print(f"  Global mean vel_cos: {global_mean_vel:+.4f}")
    print(f"  Null std (1/sqrt({T})): {null_std:.4f}")
    print(f"  Z-score: {z_score:.2f}σ")
    print(f"  Positive fraction: {(vel_arr > 0).sum()}/{len(vel_arr)} ({(vel_arr > 0).mean()*100:.1f}%)")

    # Per lora type
    for lt in ["A", "B"]:
        vals = np.array([v["vel_cos_mean"] for v in results.values() if v["lora_type"] == lt])
        z = vals.mean() / (null_std + 1e-12)
        print(
            f"  lora_{lt}: mean={vals.mean():+.4f}  z={z:.2f}σ  "
            f"pos={int((vals > 0).sum())}/{len(vals)}"
        )

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
