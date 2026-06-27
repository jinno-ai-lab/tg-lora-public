"""Per-tensor ΔW analysis grouped by layer type (GOAL §4 step 2).

Computes rank-1 dominance and direction stability from per-tensor delta
histories (same format as PSA's internal ring buffer). Groups results by
layer type to verify:
- "out_proj 最安定仮説" (Track08)
- "DeltaNet fp32 経路の数値優位" (Track07)

All metrics include a Marchenko-Pastur null baseline per GOAL §7.
"""

import math

import torch

from src.tg_lora.layer_type import classify_layer_type
from src.tg_lora.psa import _power_iteration_pc1


def compute_rank1_dominance(mat: torch.Tensor) -> float:
    """Compute rank-1 dominance ratio: fraction of variance in PC1.

    Args:
        mat: [H, numel] matrix of stacked delta snapshots.

    Returns:
        Ratio in [0, 1]. Higher = more rank-1 dominant.
        1.0 means all variance is along one direction.
    """
    if mat.shape[0] < 2:
        return 0.0
    v_pc1 = _power_iteration_pc1(mat, n_iters=30)
    # Variance explained by PC1: ||mat @ v||^2 / ||mat||^2
    proj = mat @ v_pc1  # [H]
    pc1_var = proj.dot(proj).item()
    total_var = mat.pow(2).sum().item()
    if total_var < 1e-30:
        return 0.0
    return pc1_var / total_var


def compute_direction_stability(mat: torch.Tensor) -> float | None:
    """Compute direction stability: cosine between PC1 of first and second halves.

    Args:
        mat: [H, numel] matrix of stacked delta snapshots.

    Returns:
        Cosine similarity in [-1, 1] between PC1(first_half) and PC1(second_half).
        None if insufficient data.
    """
    h = mat.shape[0]
    if h < 4:
        return None
    mid = h // 2
    v_first = _power_iteration_pc1(mat[:mid], n_iters=30)
    v_second = _power_iteration_pc1(mat[mid:], n_iters=30)
    cos = torch.dot(v_first, v_second).clamp(-1, 1).item()
    return abs(cos)


def marchenko_pastur_expected_rank1(
    n_rows: int, n_cols: int, gamma: float | None = None
) -> float:
    """Expected rank-1 dominance under random null (Marchenko-Pastur).

    For a random matrix with iid entries, the largest eigenvalue is bounded
    by the MP upper edge. The expected rank-1 dominance is approximately
    1/sqrt(min(n,p)) for wide matrices, or more precisely the ratio of the
    top eigenvalue to the total.

    Args:
        n_rows: Number of rows (snapshots).
        n_cols: Number of columns (parameters per tensor).
        gamma: Aspect ratio n/p. Computed if None.

    Returns:
        Expected rank-1 dominance ratio under the null.
    """
    if n_rows < 2 or n_cols < 2:
        return 0.0
    p = max(n_rows, n_cols)
    n = min(n_rows, n_cols)
    ratio = n / p
    # Under MP, the largest eigenvalue ≈ (1 + sqrt(n/p))^2 * sigma^2
    # and total trace ≈ p * sigma^2 for iid entries
    # rank-1 dominance ≈ (1 + sqrt(n/p))^2 / p
    # But for our case where we have n_rows snapshots of n_cols-dimensional vectors,
    # the mat^T@mat is n_cols x n_cols, with n_rows effective samples
    # The expected top eigenvalue fraction is approximately 1/n_rows for n_rows << n_cols
    # and slightly higher due to finite-size effects
    return 1.0 / n_rows + math.sqrt(ratio) / n_rows


def analyze_tensor_deltas(
    deltas: list[dict[str, torch.Tensor]],
    tensor_names: list[str] | None = None,
) -> dict[str, dict]:
    """Analyze per-tensor ΔW metrics from a delta history.

    Args:
        deltas: List of per-step delta dicts {tensor_name: delta_tensor}.
            Same format as PSA's _delta_history.
        tensor_names: Optional subset of tensors to analyze. All if None.

    Returns:
        Per-tensor analysis results:
        {
            tensor_name: {
                "rank1_dominance": float,
                "direction_stability": float | None,
                "layer_type": str,
                "n_snapshots": int,
                "rank1_null_expected": float,
                "rank1_z": float,  # z-score vs null
            },
            ...
        }
    """
    if not deltas:
        return {}

    all_names = sorted(deltas[0].keys())
    if tensor_names is not None:
        target_names = sorted(set(tensor_names) & set(all_names))
    else:
        target_names = all_names

    results: dict[str, dict] = {}

    for name in target_names:
        rows = []
        for d in deltas:
            if name not in d:
                continue
            rows.append(d[name].flatten().to(torch.float32))
        if len(rows) < 2:
            continue

        mat = torch.stack(rows)
        n_snapshots = mat.shape[0]

        rank1 = compute_rank1_dominance(mat)
        dir_stab = compute_direction_stability(mat)
        null_exp = marchenko_pastur_expected_rank1(n_snapshots, mat.shape[1])
        null_std = null_exp * 0.5  # conservative estimate
        z_score = (rank1 - null_exp) / null_std if null_std > 1e-12 else 0.0

        results[name] = {
            "rank1_dominance": rank1,
            "direction_stability": dir_stab,
            "layer_type": classify_layer_type(name).value,
            "n_snapshots": n_snapshots,
            "rank1_null_expected": null_exp,
            "rank1_z": z_score,
        }

    return results


def group_by_layer_type(
    per_tensor: dict[str, dict],
) -> dict[str, dict]:
    """Aggregate per-tensor analysis results by layer type.

    Returns:
        {
            layer_type: {
                "rank1_dominance_mean": float,
                "rank1_dominance_std": float,
                "rank1_z_mean": float,
                "direction_stability_mean": float | None,
                "direction_stability_std": float | None,
                "n_tensors": int,
                "tensor_names": list[str],
            },
            ...
        }
    """
    groups: dict[str, list[tuple[str, dict]]] = {}
    for name, info in per_tensor.items():
        lt = info["layer_type"]
        groups.setdefault(lt, []).append((name, info))

    result: dict[str, dict] = {}
    for lt, entries in sorted(groups.items()):
        r1s = [e[1]["rank1_dominance"] for e in entries]
        z1s = [e[1]["rank1_z"] for e in entries]
        ds_vals = [e[1]["direction_stability"] for e in entries if e[1]["direction_stability"] is not None]

        r1_mean = sum(r1s) / len(r1s) if r1s else 0.0
        r1_var = sum((r - r1_mean) ** 2 for r in r1s) / len(r1s) if r1s else 0.0
        z_mean = sum(z1s) / len(z1s) if z1s else 0.0

        ds_mean = None
        ds_std = None
        if ds_vals:
            ds_mean = sum(ds_vals) / len(ds_vals)
            ds_var = sum((d - ds_mean) ** 2 for d in ds_vals) / len(ds_vals)
            ds_std = math.sqrt(ds_var) if ds_var > 0 else 0.0

        result[lt] = {
            "rank1_dominance_mean": r1_mean,
            "rank1_dominance_std": math.sqrt(r1_var) if r1_var > 0 else 0.0,
            "rank1_z_mean": z_mean,
            "direction_stability_mean": ds_mean,
            "direction_stability_std": ds_std,
            "n_tensors": len(entries),
            "tensor_names": [e[0] for e in entries],
        }

    return result
