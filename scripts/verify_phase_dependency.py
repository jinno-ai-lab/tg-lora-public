"""Verify phase dependency: is early vel_cos genuinely special or small-sample artifact?

Test: sliding window of size k across all cycles, compute per-tensor vel_cos
in each window. If early is genuinely special, earliest windows should be
consistent outliers vs same-size later windows.

Controls:
  1. Same window size across all positions (eliminates n-bias)
  2. Surrogate: shuffle cycle order and recompute (null distribution)
  3. Per-tensor: avoid global aggregation masking individual variation
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def tensor_cos(a, b):
    af = a.float().flatten()
    bf = b.float().flatten()
    d = torch.dot(af, bf).item()
    na = af.norm().item()
    nb = bf.norm().item()
    return d / (na * nb) if na * nb > 1e-12 else 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=str,
                        default="runs/p1_r2_spectrum/trajectory_delta_artifacts")
    parser.add_argument("--n-surrogates", type=int, default=1000)
    args = parser.parse_args()

    files = sorted(Path(args.artifact_dir).glob("*.pt"))
    pilot_files = [f for f in files if "after_pilot" in f.name]
    if not pilot_files:
        pilot_files = files
    n = len(pilot_files)
    print(f"=== Phase Dependency Verification ===")
    print(f"Cycles: {n}, Surrogates: {args.n_surrogates}")

    # Load all deltas
    print("Loading deltas...")
    tnames = None
    all_deltas = []
    for f in pilot_files:
        art = torch.load(f, map_location="cpu", weights_only=False)
        deltas = art["delta_tensors"]
        if tnames is None:
            tnames = sorted(deltas.keys())
        all_deltas.append({tn: deltas[tn].float() for tn in tnames})
        del art

    # ─── Compute vel_cos matrix: [n_tensors, n-1] ───
    n_t = len(tnames)
    vel_matrix = np.zeros((n_t, n - 1))  # per-tensor, per-pair
    for t in range(1, n):
        for ti, tn in enumerate(tnames):
            vel_matrix[ti, t - 1] = tensor_cos(all_deltas[t][tn], all_deltas[t - 1][tn])

    # ─── Sliding window analysis ───
    print(f"\n=== Sliding Window vel_cos (per-tensor mean) ===")
    for w_size in [3, 4, 5, 6]:
        if w_size >= n:
            continue
        n_windows = n - w_size  # w_size+1 cycles → w_size vel pairs, but vel has n-1 entries
        # Actually: w_size+1 consecutive cycles give w_size velocity pairs
        # But vel_matrix has n-1 entries (for n cycles)
        # A window of w_size+1 cycles [i, i+w_size] corresponds to vel entries [i, i+w_size-1]
        n_windows = n - w_size  # cycles i..i+w_size, vel pairs i..i+w_size-1

        window_means = []
        for start in range(n_windows):
            # vel pairs for cycles [start, start+w_size]
            window_vel = vel_matrix[:, start:start + w_size]  # [n_t, w_size]
            mean_vel = window_vel.mean()  # mean across tensors AND time
            window_means.append(mean_vel)

        window_means = np.array(window_means)
        print(f"\n  Window size {w_size} ({w_size+1} cycles):")
        print(f"  {'Position':>10s}  {'Cycles':>12s}  {'mean_vel_cos':>13s}  {'rank':>6s}")

        ranked = np.argsort(-window_means)
        rank_map = {pos: r + 1 for r, pos in enumerate(ranked)}

        for i, m in enumerate(window_means):
            marker = " <<<" if i == 0 else ""
            print(f"  {i:>10d}  [{i:>2d}..{i+w_size:>2d}]     {m:+.4f}  {rank_map[i]:>6d}{marker}")

        # Is position 0 (earliest) an outlier?
        pos0_val = window_means[0]
        rest_vals = window_means[1:]
        if len(rest_vals) > 0:
            z_vs_rest = (pos0_val - rest_vals.mean()) / (rest_vals.std() + 1e-12)
            print(f"  Early vs rest: z={z_vs_rest:.2f}σ  "
                  f"(early={pos0_val:+.4f}, rest_mean={rest_vals.mean():+.4f})")

    # ─── Surrogate test: shuffle cycle order ───
    print(f"\n=== Surrogate Test (shuffle cycle order, n={args.n_surrogates}) ===")
    # For each surrogate, shuffle the cycle order, recompute vel_cos, measure early window
    w_size = 4  # Use 4 as primary window
    n_windows = n - w_size

    # Observed early window mean
    obs_early = vel_matrix[:, :w_size].mean()

    surrogate_earlies = []
    rng = np.random.RandomState(42)
    for si in range(args.n_surrogates):
        perm = rng.permutation(n)
        # Recompute vel_matrix with permuted order
        surr_vel = np.zeros((n_t, n - 1))
        for t in range(1, n):
            pt = perm[t]
            pt_prev = perm[t - 1]
            for ti, tn in enumerate(tnames):
                surr_vel[ti, t - 1] = tensor_cos(all_deltas[pt][tn], all_deltas[pt_prev][tn])
        surr_early = surr_vel[:, :w_size].mean()
        surrogate_earlies.append(surr_early)

    surrogate_earlies = np.array(surrogate_earlies)
    p_value = (surrogate_earlies >= obs_early).mean()

    print(f"  Observed early vel_cos (w={w_size}): {obs_early:+.4f}")
    print(f"  Surrogate null: mean={surrogate_earlies.mean():+.4f}  "
          f"std={surrogate_earlies.std():.4f}")
    print(f"  Z-score: {(obs_early - surrogate_earlies.mean()) / (surrogate_earlies.std() + 1e-12):.2f}σ")
    print(f"  P-value (surrogate >= observed): {p_value:.4f}")
    if p_value < 0.05:
        print(f"  *** SIGNIFICANT: early phase is genuinely special (p={p_value:.4f}) ***")
    else:
        print(f"  NOT SIGNIFICANT: early advantage is consistent with random ordering")

    # ─── Bootstrap: CI for early vs late ───
    print(f"\n=== Bootstrap CI for early vs late vel_cos ===")
    n_bootstrap = 10000
    early_size = 4  # cycles 0-4
    late_start = n - early_size - 1  # last 4 pairs

    early_vels = vel_matrix[:, :early_size]  # [n_t, early_size]
    late_vels = vel_matrix[:, late_start:]   # [n_t, late_size]

    early_flat = early_vels.flatten()
    late_flat = late_vels.flatten()

    rng2 = np.random.RandomState(123)
    diff_dist = []
    for _ in range(n_bootstrap):
        e = rng2.choice(early_flat, size=len(early_flat), replace=True).mean()
        l = rng2.choice(late_flat, size=len(late_flat), replace=True).mean()
        diff_dist.append(e - l)
    diff_dist = np.array(diff_dist)

    print(f"  Early mean: {early_flat.mean():+.4f} (n={len(early_flat)})")
    print(f"  Late mean:  {late_flat.mean():+.4f} (n={len(late_flat)})")
    print(f"  Diff: {early_flat.mean() - late_flat.mean():+.4f}")
    print(f"  Bootstrap 95% CI for diff: [{np.percentile(diff_dist, 2.5):+.4f}, "
          f"{np.percentile(diff_dist, 97.5):+.4f}]")
    if np.percentile(diff_dist, 2.5) > 0:
        print(f"  *** CI excludes 0: early > late is robust ***")
    else:
        print(f"  CI includes 0: early > late is NOT significant")

    # ─── Half-cycle test: first half vs second half ───
    print(f"\n=== Half-Cycle Test ===")
    half = (n - 1) // 2
    first_half = vel_matrix[:, :half].flatten()
    second_half = vel_matrix[:, half:].flatten()
    print(f"  First half (pairs 0-{half-1}): mean={first_half.mean():+.4f} n={len(first_half)}")
    print(f"  Second half (pairs {half}-{n-2}): mean={second_half.mean():+.4f} n={len(second_half)}")
    # t-test
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(first_half, second_half, equal_var=False)
    print(f"  Welch t-test: t={t_stat:.3f}  p={p_val:.4f}")

    # ─── Per-tensor half-cycle test ───
    print(f"\n=== Per-Tensor Half-Cycle (which tensors drive the phase effect?) ===")
    per_tensor_early = vel_matrix[:, :half].mean(axis=1)  # [n_t]
    per_tensor_late = vel_matrix[:, half:].mean(axis=1)    # [n_t]
    diff = per_tensor_early - per_tensor_late
    n_positive = (diff > 0).sum()

    print(f"  Tensors where early > late: {n_positive}/{n_t} ({n_positive/n_t*100:.0f}%)")
    print(f"  Mean diff: {diff.mean():+.4f}")

    # Top tensors
    top_idx = np.argsort(-diff)[:10]
    for rank, ti in enumerate(top_idx):
        tn = tnames[ti]
        short = tn.replace("base_model.model.model.", "")
        print(f"  {rank+1}. {short[:50]:50s} "
              f"early={per_tensor_early[ti]:+.3f} late={per_tensor_late[ti]:+.3f} "
              f"diff={diff[ti]:+.3f}")


if __name__ == "__main__":
    main()
