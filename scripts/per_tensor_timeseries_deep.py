"""Deep time series analysis: AR fitting, spectral, cross-correlation, change-point.

Extends per_tensor_timeseries.py with deeper temporal structure analysis.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def autocorr(x, lag):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < lag + 2:
        return 0.0
    m = x.mean()
    v = ((x - m) ** 2).sum()
    if v < 1e-15:
        return 0.0
    return float(((x[lag:] - m) * (x[:-lag] - m)).sum() / v)


def tensor_cos(a, b):
    af = a.float().flatten()
    bf = b.float().flatten()
    d = torch.dot(af, bf).item()
    na = af.norm().item()
    nb = bf.norm().item()
    return d / (na * nb) if na * nb > 1e-12 else 0.0


def fit_ar1(series):
    """Fit AR(1) model: x_t = phi * x_{t-1} + c + eps. Returns phi, c, sigma_eps."""
    x = np.asarray(series, dtype=np.float64)
    if len(x) < 3:
        return 0.0, x.mean() if len(x) > 0 else 0.0, 0.0
    # phi = cov(x_t, x_{t-1}) / var(x_{t-1})
    xm = x[1:]
    ym = x[:-1]
    mm = xm.mean()
    my = ym.mean()
    cov = ((xm - mm) * (ym - my)).sum()
    var = ((ym - my) ** 2).sum()
    phi = cov / var if var > 1e-15 else 0.0
    c = mm - phi * my
    resid = xm - phi * ym - c
    sigma = resid.std()
    return float(phi), float(c), float(sigma)


def spectral_peak(series, dt=1.0):
    """Find dominant frequency via FFT. Returns (freq, power_ratio)."""
    x = np.asarray(series, dtype=np.float64)
    if len(x) < 4:
        return 0.0, 0.0
    x = x - x.mean()
    # Window
    n = len(x)
    fft = np.fft.rfft(x)
    power = np.abs(fft[1:]) ** 2  # skip DC
    if power.sum() < 1e-15:
        return 0.0, 0.0
    peak_idx = power.argmax()
    freqs = np.fft.rfftfreq(n, d=dt)[1:]
    return float(freqs[peak_idx]), float(power[peak_idx] / power.sum())


def cross_corr(x, y, max_lag=5):
    """Normalized cross-correlation at lags -max_lag..+max_lag."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y))
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    vx = (x ** 2).sum()
    vy = (y ** 2).sum()
    denom = math.sqrt(vx * vy) if vx * vy > 1e-15 else 1.0
    result = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = (x[lag:] * y[: n - lag]).sum()
        else:
            c = (x[: n + lag] * y[-lag:]).sum()
        result[lag] = float(c / denom)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=str,
                        default="runs/p1_r2_spectrum/trajectory_delta_artifacts")
    parser.add_argument("--ts-json", type=str,
                        default="runs/p1_r2_spectrum_per_tensor_ts.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        run_name = Path(args.artifact_dir).parent.name
        args.output = f"runs/{run_name}_deep_ts.json"

    # Load basic time series data
    files = sorted(Path(args.artifact_dir).glob("*.pt"))
    pilot_files = [f for f in files if "after_pilot" in f.name]
    if not pilot_files:
        pilot_files = files
    n = len(pilot_files)
    print(f"=== Deep Time Series Analysis ===")
    print(f"Cycles: {n}")

    art0 = torch.load(pilot_files[0], map_location="cpu", weights_only=False)
    tnames = sorted(art0["delta_tensors"].keys())
    del art0

    # Load basic results
    with open(args.ts_json) as f:
        basic = json.load(f)

    # ─── Phase 1: Load all deltas ───
    print("Loading deltas...")
    all_deltas = []  # list of dicts
    for f in pilot_files:
        art = torch.load(f, map_location="cpu", weights_only=False)
        all_deltas.append({tn: art["delta_tensors"][tn].float() for tn in tnames})
        del art
    print(f"Loaded {len(all_deltas)} cycles")

    # ─── Phase 2: Per-tensor AR(1) + spectral analysis ───
    print("\n=== AR(1) Fitting ===")
    ar_results = {}
    for tn in tnames:
        norms = np.array([d[tn].norm().item() for d in all_deltas])
        phi_n, c_n, sig_n = fit_ar1(norms)

        # AR(1) on velocity cosine (norm series)
        # Also AR(1) on rank-1 direction cosine series
        # Compute rank-1 feature dirs
        feat_dirs = []
        for d in all_deltas:
            t = d[tn]
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            if "lora_A" in tn:
                feat_dirs.append(Vh[0].clone())
            else:
                feat_dirs.append(U[:, 0].clone())

        # Direction cosine series
        dir_cos = []
        for t in range(1, len(feat_dirs)):
            cos = torch.dot(feat_dirs[t].flatten(), feat_dirs[t-1].flatten()).item() / (
                feat_dirs[t].norm().item() * feat_dirs[t-1].norm().item() + 1e-12)
            dir_cos.append(max(-1, min(1, cos)))
        dir_cos = np.array(dir_cos)

        phi_d, c_d, sig_d = fit_ar1(dir_cos)

        # Spectral on norms
        freq_n, pwr_n = spectral_peak(norms)
        # Spectral on direction cosines
        freq_d, pwr_d = spectral_peak(dir_cos)

        # Norm spectral at various lags
        norm_ac_full = [autocorr(norms, lag) for lag in range(1, min(n, 11))]
        dir_ac_full = [autocorr(dir_cos, lag) for lag in range(1, min(n-1, 11))]

        ar_results[tn] = {
            "norm_ar_phi": phi_n,
            "norm_ar_sigma": sig_n,
            "dir_cos_ar_phi": phi_d,
            "dir_cos_ar_sigma": sig_d,
            "norm_spectral_freq": freq_n,
            "norm_spectral_power": pwr_n,
            "dir_spectral_freq": freq_d,
            "dir_spectral_power": pwr_d,
            "norm_ac_full": norm_ac_full,
            "dir_ac_full": dir_ac_full,
        }

    # Free direction vectors
    del feat_dirs

    # ─── Phase 3: Cross-layer correlation ───
    print("\n=== Cross-Layer Correlation ===")

    # Group tensors by (layer, module, lora_type) - use norm series
    layer_norms = defaultdict(dict)  # layer -> {tn: norm_series}
    for tn in tnames:
        parts = tn.split(".")
        for i, p in enumerate(parts):
            if p == "layers":
                layer = int(parts[i+1])
                layer_norms[layer][tn] = np.array([d[tn].norm().item() for d in all_deltas])
                break

    # Cross-correlation between layers (using mean norm per layer)
    layers = sorted(layer_norms.keys())
    layer_mean_norms = {}
    for l in layers:
        norms_list = list(layer_norms[l].values())
        if norms_list:
            layer_mean_norms[l] = np.mean(norms_list, axis=0)

    print(f"\nLayer-to-layer norm correlation (lag 0):")
    print(f"{'':>6s}", end="")
    for l2 in layers:
        print(f"  L{l2:>2d}", end="")
    print()
    for l1 in layers:
        print(f"L{l1:>2d}  ", end="")
        for l2 in layers:
            if l1 == l2:
                print(f"   1.0", end="")
            else:
                c = np.corrcoef(layer_mean_norms[l1], layer_mean_norms[l2])[0, 1]
                print(f"  {c:+.2f}", end="")
        print()

    # Cross-correlation at lag 1 (does layer L1 lead L2?)
    print(f"\nLag-1 cross-correlation (L1_t vs L2_{{t+1}}):")
    for l1 in layers:
        for l2 in layers:
            if l1 >= l2:
                continue
            x = layer_mean_norms[l1][:-1]
            y = layer_mean_norms[l2][1:]
            if len(x) > 2:
                c = np.corrcoef(x, y)[0, 1]
                if abs(c) > 0.3:
                    print(f"  L{l1} -> L{l2}: {c:+.3f}")

    # ─── Phase 4: Cross-correlation between lora_A and lora_B within same module ───
    print(f"\n=== lora_A vs lora_B Cross-Correlation (same module) ===")
    # Pair A and B tensors
    pairs = {}
    for tn in tnames:
        base = tn.replace("lora_A", "lora_X").replace("lora_B", "lora_X")
        pairs.setdefault(base, []).append(tn)

    pair_corrs = []
    for base, tns in pairs.items():
        if len(tns) == 2:
            tnA = [t for t in tns if "lora_A" in t][0]
            tnB = [t for t in tns if "lora_B" in t][0]
            nA = np.array([d[tnA].norm().item() for d in all_deltas])
            nB = np.array([d[tnB].norm().item() for d in all_deltas])
            c = np.corrcoef(nA, nB)[0, 1]
            pair_corrs.append(c)

    print(f"  Mean |corr(A,B)|: {np.mean(np.abs(pair_corrs)):.3f}")
    print(f"  Mean corr(A,B): {np.mean(pair_corrs):+.3f}")
    print(f"  Positive fraction: {np.mean(np.array(pair_corrs) > 0)*100:.0f}%")

    # ─── Phase 5: Change-point detection (norm series) ───
    print(f"\n=== Change-Point Detection (CUSUM on norms) ===")
    for layer in layers[:4]:  # First 4 layers
        norms = layer_mean_norms[layer]
        mean = norms.mean()
        std = norms.std()
        if std < 1e-10:
            continue
        cusum = np.cumsum(norms - mean) / std
        # Find max deviation
        max_idx = np.argmax(np.abs(cusum))
        print(f"  Layer {layer}: CUSUM peak at cycle {max_idx} (cusum={cusum[max_idx]:+.2f}σ)")

    # ─── Phase 6: Per-layer velocity autocorrelation with holdout ───
    print(f"\n=== Per-Layer Holdout Velocity AC ===")
    for layer in layers:
        # Compute vel_cos per layer using delta tensors
        vel_cos_layer = []
        for t in range(1, len(all_deltas)):
            # Per-layer global delta
            layer_d_t = torch.cat([all_deltas[t][tn].flatten() for tn in sorted(layer_norms[layer].keys())])
            layer_d_t1 = torch.cat([all_deltas[t-1][tn].flatten() for tn in sorted(layer_norms[layer].keys())])
            cos = tensor_cos(layer_d_t, layer_d_t1)
            vel_cos_layer.append(cos)
        vel_cos_layer = np.array(vel_cos_layer)
        print(f"  Layer {layer}: vel_cos={vel_cos_layer.mean():+.4f}  "
              f"pos={int((vel_cos_layer > 0).sum())}/{len(vel_cos_layer)}")

    # ─── Summary of key findings ───
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    # AR(1) phi summary
    phis_dir = [ar_results[tn]["dir_cos_ar_phi"] for tn in tnames]
    phis_norm = [ar_results[tn]["norm_ar_phi"] for tn in tnames]
    print(f"  AR(1) phi (dir_cos): mean={np.mean(phis_dir):+.3f}  "
          f"pos={int(np.mean(np.array(phis_dir) > 0)*100)}%")
    print(f"  AR(1) phi (norm):    mean={np.mean(phis_norm):+.3f}  "
          f"pos={int(np.mean(np.array(phis_norm) > 0)*100)}%")

    # Spectral
    freqs_n = [ar_results[tn]["norm_spectral_freq"] for tn in tnames if ar_results[tn]["norm_spectral_power"] > 0.3]
    if freqs_n:
        print(f"  Norm spectral peak freq (power>0.3): {[f'{f:.3f}' for f in freqs_n[:5]]}")

    # Save
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ar_results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
