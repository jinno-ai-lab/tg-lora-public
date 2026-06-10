"""Trace A-direction trajectory at finest resolution.

Focus: lora_A of out_proj/in_proj_qkv, cycles 0-10 (r=2, 11 cycles).
Decompose weight change into B-growth and A-direction.
Track A-direction cycle by cycle, NOT averaged.

Key question: after subtracting B=0 startup transient, does the A-direction
show genuine persistence that can be exploited for extrapolation?
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    artifact_dir = Path("runs/p1_r2_spectrum/trajectory_delta_artifacts")
    files = sorted(artifact_dir.glob("*.pt"))
    n = len(files)
    print(f"=== A-Direction Trajectory Trace ===")
    print(f"Cycles: {n}")

    # Load all deltas
    all_deltas = []
    all_meta = []
    for f in files:
        art = torch.load(f, map_location="cpu", weights_only=False)
        all_deltas.append(art["delta_tensors"])
        all_meta.append(art["metadata"])

    # Target tensors: lora_A of out_proj and in_proj_qkv (strongest signal)
    tnames = sorted(all_deltas[0].keys())
    targets = [tn for tn in tnames if "lora_A" in tn and
               ("out_proj" in tn or "in_proj_qkv" in tn)]
    print(f"Target tensors: {len(targets)}")
    for tn in targets:
        short = tn.replace("base_model.model.model.", "")
        shape = all_deltas[0][tn].shape
        print(f"  {short[:55]:55s} {list(shape)}")

    # ─── Reconstruct cumulative B (B_0 = 0 in standard LoRA init) ───
    # For each target lora_A, find matching lora_B
    def get_B_name(A_name):
        return A_name.replace("lora_A", "lora_B")

    # ─── Cycle-by-cycle A-direction analysis ───
    print(f"\n{'='*90}")
    print("CYCLE-BY-CYCLE A-DIRECTION TRAJECTORY")
    print(f"{'='*90}")

    # For each target tensor: track ΔA direction, cumulative B, weight decomposition
    for tn in targets:
        short = tn.replace("base_model.model.model.", "")
        tn_B = get_B_name(tn)
        layer = short.split("layers.")[1].split(".")[0]

        print(f"\n--- {short[:60]} (L{layer}) ---")
        print(f"{'Cycle':>5s}  {'||dA||':>8s}  {'||dB||':>8s}  {'||B_cum||':>9s}  "
              f"{'r1_A':>6s}  {'r1_B':>6s}  {'A_dir_cos':>10s}  "
              f"{'||B@dA||':>9s}  {'||dB@dA||':>9s}  {'A_contribution':>15s}")

        prev_A_dir = None
        B_cum = None  # cumulative B (starts at 0)
        cum_A_dir = []

        for c in range(n):
            dA = all_deltas[c][tn].float()
            dB = all_deltas[c][tn_B].float()

            # Cumulative B: B_t = B_0 + sum(dB_0..dB_t), B_0 = 0
            if B_cum is None:
                B_cum = dB.clone()
            else:
                B_cum = B_cum + dB

            # SVD of dA [2, 4096] → rank-1 direction
            U_A, S_A, Vh_A = torch.linalg.svd(dA, full_matrices=False)
            total_var_A = (S_A ** 2).sum().item()
            r1_A = (S_A[0] ** 2).item() / total_var_A if total_var_A > 1e-15 else 0
            A_dir = Vh_A[0]  # [4096] - feature space direction

            # SVD of dB
            U_B, S_B, Vh_B = torch.linalg.svd(dB, full_matrices=False)
            total_var_B = (S_B ** 2).sum().item()
            r1_B = (S_B[0] ** 2).item() / total_var_B if total_var_B > 1e-15 else 0

            # Direction cosine with previous cycle
            if prev_A_dir is not None:
                cos_val = torch.dot(A_dir, prev_A_dir).item() / (
                    A_dir.norm().item() * prev_A_dir.norm().item() + 1e-12)
                cos_val = max(-1, min(1, cos_val))
            else:
                cos_val = float('nan')

            # Weight change decomposition
            # B_{t-1} @ dA_t: A-direction contribution
            B_prev = B_cum - dB  # B_{t-1}
            A_contrib = B_prev @ dA  # [out_dim, 4096]
            a_contrib_norm = A_contrib.norm().item()

            # Cross term: dB_t @ dA_t
            cross = dB @ dA
            cross_norm = cross.norm().item()

            # Total weight delta norm (approximate: dB @ A_{t-1} + B_{t-1} @ dA + cross)
            # We have B_{t-1} @ dA and cross, but not dB @ A_{t-1}
            # So A_contribution = ||B_{t-1} @ dA|| / (||B_{t-1} @ dA|| + ||cross||)
            total_known = a_contrib_norm + cross_norm
            a_frac = a_contrib_norm / total_known if total_known > 1e-12 else 0

            print(f"{c:>5d}  {dA.norm().item():>8.4f}  {dB.norm().item():>8.4f}  "
                  f"{B_cum.norm().item():>9.4f}  "
                  f"{r1_A:>6.3f}  {r1_B:>6.3f}  {cos_val:>10.4f}  "
                  f"{a_contrib_norm:>9.4f}  {cross_norm:>9.4f}  "
                  f"{a_frac:>14.3f}")

            prev_A_dir = A_dir.clone()
            cum_A_dir.append(A_dir)

        # ─── Pairwise direction matrix for this tensor ───
        print(f"\n  Pairwise A-direction cosine matrix:")
        header = "        " + "".join(f"C{c:>5d}" for c in range(n))
        print(f"  {header}")
        for i in range(n):
            row = f"  C{i:>3d}  "
            for j in range(n):
                cos = torch.dot(cum_A_dir[i], cum_A_dir[j]).item() / (
                    cum_A_dir[i].norm().item() * cum_A_dir[j].norm().item() + 1e-12)
                cos = max(-1, min(1, cos))
                row += f"{cos:>6.3f}"
            print(row)

    # ─── Cross-tensor direction coherence ───
    print(f"\n{'='*90}")
    print("CROSS-TENSOR DIRECTION COHERENCE (same cycle, different tensors)")
    print(f"{'='*90}")

    # For cycle 0-3, compare A-direction across different target tensors
    for c in range(min(4, n)):
        print(f"\n  Cycle {c}:")
        dirs = {}
        for tn in targets:
            dA = all_deltas[c][tn].float()
            U, S, Vh = torch.linalg.svd(dA, full_matrices=False)
            dirs[tn] = Vh[0]

        # Pairwise cos
        tlist = list(dirs.keys())
        for i in range(len(tlist)):
            for j in range(i + 1, len(tlist)):
                s1 = tlist[i].replace("base_model.model.model.", "")[:35]
                s2 = tlist[j].replace("base_model.model.model.", "")[:35]
                cos = torch.dot(dirs[tlist[i]], dirs[tlist[j]]).item() / (
                    dirs[tlist[i]].norm().item() * dirs[tlist[j]].norm().item() + 1e-12)
                print(f"    {s1:>35s} x {s2:>35s}: {cos:+.4f}")

    # ─── Residual analysis: subtract B-growth, isolate pure A-direction ───
    print(f"\n{'='*90}")
    print("RESIDUAL ANALYSIS: B-growth subtracted A-direction persistence")
    print(f"{'='*90}")

    # The weight change B_{t-1} @ dA_t represents the A-direction signal
    # through existing B. When B is small (early), this is weak.
    # When B is established, this captures the A-direction evolution.

    for tn in targets[:4]:  # Top 4 targets
        short = tn.replace("base_model.model.model.", "")
        tn_B = get_B_name(tn)
        print(f"\n  {short[:55]}:")

        B_cum = torch.zeros_like(all_deltas[0][tn_B].float())
        prev_A_signal_dir = None

        print(f"  {'Cycle':>5s}  {'||B_cum||':>9s}  {'||A_signal||':>12s}  "
              f"{'A_sig_dir_cos':>14s}  {'B_ratio':>8s}")

        for c in range(n):
            dA = all_deltas[c][tn].float()
            dB = all_deltas[c][tn_B].float()

            B_prev = B_cum.clone()
            A_signal = B_prev @ dA  # The A-direction contribution to weight change
            a_sig_norm = A_signal.norm().item()

            if a_sig_norm > 1e-10:
                U_as, S_as, Vh_as = torch.linalg.svd(A_signal, full_matrices=False)
                A_sig_dir = U_as[:, 0]  # dominant direction in output space

                if prev_A_signal_dir is not None:
                    cos_val = torch.dot(A_sig_dir, prev_A_signal_dir).item() / (
                        A_sig_dir.norm().item() * prev_A_signal_dir.norm().item() + 1e-12)
                    cos_val = max(-1, min(1, cos_val))
                else:
                    cos_val = float('nan')
                prev_A_signal_dir = A_sig_dir.clone()
            else:
                cos_val = float('nan')
                prev_A_signal_dir = None

            B_cum = B_cum + dB

            # B-growth ratio: ||dB|| / (||dB|| + ||dA||) as proxy
            total_param_change = dB.norm().item() + dA.norm().item()
            b_ratio = dB.norm().item() / total_param_change if total_param_change > 1e-12 else 0

            print(f"  {c:>5d}  {B_cum.norm().item():>9.4f}  {a_sig_norm:>12.6f}  "
                  f"{cos_val if not math.isnan(cos_val) else 0:>14.4f}  {b_ratio:>8.3f}")

    # ─── Layer-wise SNR map ───
    print(f"\n{'='*90}")
    print("LAYER-WISE SNR MAP (A-direction stability × phase)")
    print(f"{'='*90}")

    # For ALL lora_A tensors, compute early/late A-direction stability
    all_A_targets = [tn for tn in tnames if "lora_A" in tn]
    half = n // 2

    print(f"\n{'Layer':>6s} {'Module':>20s} {'early_dir_cos':>14s} {'late_dir_cos':>14s} "
          f"{'early_r1':>9s} {'late_r1':>9s} {'B_cum_ratio':>12s}")

    for tn in sorted(all_A_targets):
        parts = tn.split(".")
        layer = module = None
        for i, p in enumerate(parts):
            if p == "layers":
                layer = int(parts[i+1])
                rest = parts[i+2:]
                for j, q in enumerate(rest):
                    if q.startswith("lora_"):
                        module = ".".join(rest[:j])
                        break
                break

        tn_B = tn.replace("lora_A", "lora_B")

        # Compute A-direction cosines
        dirs = []
        B_cum_local = torch.zeros_like(all_deltas[0][tn_B].float())
        for c in range(n):
            dA = all_deltas[c][tn].float()
            dB = all_deltas[c][tn_B].float()
            U, S, Vh = torch.linalg.svd(dA, full_matrices=False)
            dirs.append(Vh[0])
            B_cum_local = B_cum_local + dB

        # Direction cosines
        early_cos = []
        late_cos = []
        for t in range(1, len(dirs)):
            cos = torch.dot(dirs[t], dirs[t-1]).item() / (
                dirs[t].norm().item() * dirs[t-1].norm().item() + 1e-12)
            if t <= half:
                early_cos.append(cos)
            else:
                late_cos.append(cos)

        # Rank-1 ratios
        early_r1 = []
        late_r1 = []
        for c in range(n):
            dA = all_deltas[c][tn].float()
            U, S, Vh = torch.linalg.svd(dA, full_matrices=False)
            tv = (S**2).sum().item()
            r1 = (S[0]**2).item() / tv if tv > 1e-15 else 0
            if c <= half:
                early_r1.append(r1)
            else:
                late_r1.append(r1)

        # B cumulative ratio at half point
        B_at_half = torch.zeros_like(all_deltas[0][tn_B].float())
        for c in range(half):
            B_at_half = B_at_half + all_deltas[c][tn_B].float()
        b_ratio = B_at_half.norm().item()

        ec = np.mean(early_cos) if early_cos else 0
        lc = np.mean(late_cos) if late_cos else 0
        er = np.mean(early_r1) if early_r1 else 0
        lr = np.mean(late_r1) if late_r1 else 0

        print(f"{layer:>6d} {module[:20]:>20s} {ec:>+14.4f} {lc:>+14.4f} "
              f"{er:>9.3f} {lr:>9.3f} {b_ratio:>12.4f}")


if __name__ == "__main__":
    main()
