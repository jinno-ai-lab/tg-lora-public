"""Analyze trajectory delta artifacts for GOAL §4 step 2.

Loads cumulative LoRA weight deltas from baseline ablation run artifacts,
computes per-step increments (consecutive differences), then uses
layer_delta_analysis to extract per-tensor and per-layer-type metrics:
- rank-1 dominance (with Marchenko-Pastur null)
- direction stability
- layer-type grouping

Output: printed summary table + JSON to the run directory.

Usage:
    python scripts/analyze_trajectory_deltas.py \
        runs/psa_ablation_*/baseline_plain/trajectory_delta_artifacts/
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tg_lora.layer_delta_analysis import (
    analyze_tensor_deltas,
    group_by_layer_type,
)
from src.tg_lora.layer_type import LayerType, classify_layer_type


def find_artifacts(artifact_dir: Path) -> list[tuple[int, Path]]:
    """Find and sort artifact files by step number."""
    pattern = re.compile(r"step_(\d+)\.pt$")
    entries = []
    for p in sorted(artifact_dir.glob("*.pt")):
        m = pattern.search(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda x: x[0])
    return entries


def load_artifact(path: Path) -> dict:
    """Load a single artifact (metadata + delta_tensors)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["metadata"], data["delta_tensors"]


def compute_incremental_deltas(
    artifacts: list[tuple[int, Path]],
    start_step: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """Compute per-step incremental deltas from consecutive cumulative artifacts.

    Loads one pair at a time to bound memory.
    """
    increments = []
    prev_meta, prev_tensors = None, None

    for step, path in artifacts:
        meta, tensors = load_artifact(path)

        if step < start_step:
            prev_meta, prev_tensors = meta, tensors
            continue

        if prev_tensors is None:
            prev_meta, prev_tensors = meta, tensors
            continue

        # Incremental delta = current cumulative - previous cumulative
        incr = {}
        for name in tensors:
            if name in prev_tensors:
                incr[name] = tensors[name] - prev_tensors[name]
            else:
                incr[name] = tensors[name].clone()

        increments.append(incr)
        prev_meta, prev_tensors = meta, tensors

    return increments


def compute_regime_inventory(
    increments: list[dict[str, torch.Tensor]],
    per_tensor_results: dict[str, dict],
) -> dict:
    """Compute regime inventory: fraction of stable vs transition steps.

    Uses direction cosine between consecutive increments as a proxy.
    """
    if not increments or len(increments) < 2:
        return {"n_steps": 0}

    # Pick a few representative tensors for regime detection
    target_types = {
        LayerType.ATTENTION_OUT.value,
        LayerType.ATTENTION_V.value,
        LayerType.DELTANET.value,
    }
    rep_tensors = [
        name
        for name, info in per_tensor_results.items()
        if info["layer_type"] in target_types
    ]
    # Take up to 5 per type
    selected = []
    seen_types = {}
    for name in rep_tensors:
        lt = per_tensor_results[name]["layer_type"]
        seen_types.setdefault(lt, []).append(name)
    for lt, names in seen_types.items():
        selected.extend(names[:5])

    if not selected:
        return {"n_steps": len(increments)}

    # Compute per-step direction norm and consecutive cosine
    step_cosines = []
    for name in selected:
        vecs = [inc[name].flatten().float() for inc in increments if name in inc]
        for i in range(1, len(vecs)):
            n1, n2 = vecs[i - 1].norm().item(), vecs[i].norm().item()
            if n1 > 1e-10 and n2 > 1e-10:
                cos = torch.dot(vecs[i - 1], vecs[i]).item() / (n1 * n2)
                step_cosines.append(cos)

    if not step_cosines:
        return {"n_steps": len(increments)}

    # Classify: cos > 0.5 = stable, 0.0-0.5 = plateau, < 0.0 = transition
    n_stable = sum(1 for c in step_cosines if c > 0.5)
    n_plateau = sum(1 for c in step_cosines if 0.0 <= c <= 0.5)
    n_transition = sum(1 for c in step_cosines if c < 0.0)
    total = len(step_cosines)

    return {
        "n_steps": len(increments),
        "n_tensors_sampled": len(selected),
        "stable_fraction": n_stable / total,
        "plateau_fraction": n_plateau / total,
        "transition_fraction": n_transition / total,
        "mean_cosine": sum(step_cosines) / total,
    }


def print_summary(
    per_tensor: dict[str, dict],
    by_type: dict[str, dict],
    regime: dict,
) -> None:
    """Print human-readable summary table."""
    print("\n" + "=" * 80)
    print("TRAJECTORY DELTA ANALYSIS — GOAL §4 step 2")
    print("=" * 80)

    # Per-layer-type table
    print(f"\n{'Layer Type':<20} {'N':>4} {'Rank1 Dom':>10} {'±std':>8} "
          f"{'Dir Stab':>10} {'±std':>8} {'Z(mean)':>8}")
    print("-" * 80)

    type_order = [
        LayerType.ATTENTION_OUT.value,
        LayerType.ATTENTION_V.value,
        LayerType.ATTENTION_OTHER.value,
        LayerType.DELTANET.value,
        LayerType.MLP.value,
    ]
    for lt in type_order:
        if lt not in by_type:
            continue
        d = by_type[lt]
        ds_mean = d["direction_stability_mean"]
        ds_std = d["direction_stability_std"]
        print(f"{lt:<20} {d['n_tensors']:>4} "
              f"{d['rank1_dominance_mean']:>10.4f} {d['rank1_dominance_std']:>8.4f} "
              f"{ds_mean if ds_mean is not None else float('nan'):>10.4f} "
              f"{ds_std if ds_std is not None else float('nan'):>8.4f} "
              f"{d['rank1_z_mean']:>8.2f}")

    # Other types
    for lt, d in sorted(by_type.items()):
        if lt in type_order:
            continue
        ds_mean = d["direction_stability_mean"]
        ds_std = d["direction_stability_std"]
        print(f"{lt:<20} {d['n_tensors']:>4} "
              f"{d['rank1_dominance_mean']:>10.4f} {d['rank1_dominance_std']:>8.4f} "
              f"{ds_mean if ds_mean is not None else float('nan'):>10.4f} "
              f"{ds_std if ds_std is not None else float('nan'):>8.4f} "
              f"{d['rank1_z_mean']:>8.2f}")

    # Top-5 most rank-1 dominant tensors
    print(f"\n{'— Top 5 rank-1 dominant tensors —':^80}")
    top5 = sorted(per_tensor.items(), key=lambda x: x[1]["rank1_dominance"], reverse=True)[:5]
    for name, info in top5:
        print(f"  {name}")
        print(f"    rank1={info['rank1_dominance']:.4f}  z={info['rank1_z']:.2f}  "
              f"dir_stab={info['direction_stability']}  type={info['layer_type']}")

    # Top-5 most direction-stable tensors
    print(f"\n{'— Top 5 direction-stable tensors —':^80}")
    with_stab = [(n, i) for n, i in per_tensor.items() if i["direction_stability"] is not None]
    top5_stab = sorted(with_stab, key=lambda x: x[1]["direction_stability"], reverse=True)[:5]
    for name, info in top5_stab:
        print(f"  {name}")
        print(f"    dir_stab={info['direction_stability']:.4f}  rank1={info['rank1_dominance']:.4f}  "
              f"z={info['rank1_z']:.2f}  type={info['layer_type']}")

    # Regime inventory
    if "stable_fraction" in regime:
        print(f"\n{'— Regime Inventory —':^80}")
        print(f"  Steps: {regime['n_steps']}, Sampled tensors: {regime['n_tensors_sampled']}")
        print(f"  Stable:    {regime['stable_fraction']:.1%}")
        print(f"  Plateau:   {regime['plateau_fraction']:.1%}")
        print(f"  Transition:{regime['transition_fraction']:.1%}")
        print(f"  Mean cosine: {regime['mean_cosine']:.4f}")
        print(f"  → Theoretical efficiency ceiling = {regime['stable_fraction']:.1%} of steps")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory delta artifacts")
    parser.add_argument(
        "artifact_dir",
        type=Path,
        help="Directory containing trajectory_delta_artifacts/*.pt",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=2,
        help="First step with non-zero cumulative delta (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSON (default: artifact_dir/../)",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    if not artifact_dir.exists():
        print(f"Error: {artifact_dir} does not exist")
        sys.exit(1)

    artifacts = find_artifacts(artifact_dir)
    print(f"Found {len(artifacts)} artifacts (steps {artifacts[0][0]}–{artifacts[-1][0]})")

    # Compute incremental deltas (peak memory: 2 artifacts)
    print("Computing per-step incremental deltas...")
    increments = compute_incremental_deltas(artifacts, start_step=args.start_step)
    print(f"  {len(increments)} incremental steps computed")

    if not increments:
        print("No incremental deltas produced. Check start-step or artifact contents.")
        sys.exit(1)

    # Run per-tensor analysis
    print("Running per-tensor analysis...")
    per_tensor = analyze_tensor_deltas(increments)
    print(f"  Analyzed {len(per_tensor)} tensors")

    # Group by layer type
    by_type = group_by_layer_type(per_tensor)
    print(f"  Grouped into {len(by_type)} layer types: {sorted(by_type.keys())}")

    # Compute regime inventory
    print("Computing regime inventory...")
    regime = compute_regime_inventory(increments, per_tensor)

    # Print summary
    print_summary(per_tensor, by_type, regime)

    # Write JSON output
    output_dir = args.output_dir or artifact_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory_delta_analysis.json"

    # Convert to serializable format
    json_out = {
        "per_tensor": per_tensor,
        "by_layer_type": by_type,
        "regime_inventory": regime,
        "n_incremental_steps": len(increments),
        "source_artifact_dir": str(artifact_dir),
    }
    with open(output_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\nJSON output: {output_path}")


if __name__ == "__main__":
    main()
