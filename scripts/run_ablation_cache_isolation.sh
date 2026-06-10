#!/usr/bin/env bash
# G4 Cache Isolation Ablation: 3-condition × 3-seed experiment
#
# Conditions:
#   A: Baseline (no cache, no extrapolation)
#   B: Baseline+Cache (cache, no extrapolation) — N=0 with prefix feature cache
#   C: TG-LoRA (cache + extrapolation) — reuses existing results if available
#
# Isolates:
#   Cache effect = B - A  (memory savings, wall-clock)
#   TG effect    = C - B  (convergence improvement from trajectory extrapolation)
#
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
TARGET_BP="${TARGET_BP:-240}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-32}"
EVAL_POINTS="${EVAL_POINTS:-3}"
SEEDS="${SEEDS:-42 43 44}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/ablation_cache_isolation_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline_suffix_only_last25.yaml}"
BASELINE_CACHE_CONFIG="${BASELINE_CACHE_CONFIG:-configs/9b_baseline_with_prefix_cache.yaml}"
TG_CONFIG="${TG_CONFIG:-configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml}"
CACHE_BASE="${CACHE_BASE:-.cache/prefix_feature_cache_ablation}"
EXISTING_TG_SUITE="${EXISTING_TG_SUITE:-}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
DRY_RUN="${DRY_RUN:-false}"

read -r -a SEED_ARRAY <<<"${SEEDS}"

echo "=============================================="
echo "  G4 Cache Isolation Ablation"
echo "  Seeds: ${SEEDS}"
echo "  Target backward passes: ${TARGET_BP}"
echo "  Output: ${OUTPUT_BASE}"
echo "  A: Baseline (no cache)     → ${BASELINE_CONFIG}"
echo "  B: Baseline+Cache (N=0)    → ${BASELINE_CACHE_CONFIG}"
echo "  C: TG-LoRA (N=1, cache)    → ${TG_CONFIG}"
if [[ -n "${EXISTING_TG_SUITE}" ]]; then
echo "  Reusing TG results from: ${EXISTING_TG_SUITE}"
fi
if [[ "${DRY_RUN}" == "true" ]]; then
echo "  Mode: DRY RUN"
fi
echo "=============================================="

# --- Pre-flight validation ---
DRY_RUN_ERRORS=0

for cfg in "${BASELINE_CONFIG}" "${BASELINE_CACHE_CONFIG}" "${TG_CONFIG}"; do
    if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: Config not found: ${cfg}" >&2
        DRY_RUN_ERRORS=$((DRY_RUN_ERRORS + 1))
    else
        echo "  [OK] ${cfg}"
    fi
done

# Validate N=0 in baseline+cache config
if [[ -f "${BASELINE_CACHE_CONFIG}" ]]; then
    if grep -q "N_initial: *0" "${BASELINE_CACHE_CONFIG}"; then
        echo "  [OK] N_initial=0 confirmed in baseline+cache config"
    else
        echo "WARNING: baseline+cache config should have N_initial: 0" >&2
    fi
    if grep -q "prefix_feature_cache_experimental: *true" "${BASELINE_CACHE_CONFIG}"; then
        echo "  [OK] prefix_feature_cache_experimental: true confirmed"
    else
        echo "WARNING: baseline+cache config should have prefix_feature_cache_experimental: true" >&2
    fi
fi

if [[ "${DRY_RUN}" == "true" && ${DRY_RUN_ERRORS} -gt 0 ]]; then
    echo "" >&2
    echo "DRY RUN FAILED: ${DRY_RUN_ERRORS} validation error(s)" >&2
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"

# --- Helper: patch seed into config ---
_patch_seed() {
    local config_path="$1"
    local seed="$2"
    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${config_path}')
cfg.experiment.seed = int('${seed}')
# Remove extra fields that pydantic BaselineConfig rejects
if 'paper_experiment' in cfg.experiment:
    del cfg.experiment.paper_experiment
if 'paper_experiment_id' in cfg.experiment:
    del cfg.experiment.paper_experiment_id
OmegaConf.save(cfg, '${config_path}')
"
}

# --- Run a single condition via benchmark_prefix_cache.py ---
_run_condition() {
    local label="$1"
    local baseline_cfg="$2"
    local tg_cfg="$3"
    local seed="$4"
    local output_dir="$5"
    local cache_dir="$6"

    echo ""
    echo "--- [${label}] seed=${seed} ---"

    local config_dir="${output_dir}/configs"
    mkdir -p "${config_dir}"

    cp "${baseline_cfg}" "${config_dir}/baseline.yaml"
    cp "${tg_cfg}" "${config_dir}/tg_cache.yaml"
    _patch_seed "${config_dir}/baseline.yaml" "${seed}"
    _patch_seed "${config_dir}/tg_cache.yaml" "${seed}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [DRY RUN] would run benchmark_prefix_cache.py for seed=${seed}"
        return 0
    fi

    ${VENV_PYTHON} scripts/benchmark_prefix_cache.py \
        --budget "${TARGET_BP}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --quick-eval-examples "${QUICK_EVAL_EXAMPLES}" \
        --eval-points "${EVAL_POINTS}" \
        --baseline-config "${config_dir}/baseline.yaml" \
        --tg-config "${config_dir}/tg_cache.yaml" \
        --cache-dir "${cache_dir}" \
        --output-base "${output_dir}/coldwarm" \
        $(if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then printf -- '--cuda-visible-devices %q ' "${CUDA_VISIBLE_DEVICES_VALUE}"; fi) \
        $(if [[ "${MLFLOW_ENABLED}" == "true" ]]; then echo --mlflow-enabled; fi)
}

# --- Collect results from a condition ---
_collect_summary() {
    local cond_dir="$1"
    local summary_path="${cond_dir}/coldwarm/summary.json"
    if [[ -f "${summary_path}" ]]; then
        echo "${summary_path}"
        return 0
    fi
    # Fallback: look for individual run metrics
    echo "" >&2
    echo "WARNING: No summary.json found at ${summary_path}" >&2
    return 1
}

# ============================================================
# Main experiment loop
# ============================================================

for seed in "${SEED_ARRAY[@]}"; do
    seed_root="${OUTPUT_BASE}/seed_${seed}"
    cache_dir="${CACHE_BASE}/seed_${seed}"

    # --- Condition A: Baseline (no cache) vs dummy ---
    # We need baseline-only (no cache) as condition A.
    # Use the existing baseline from the standard paper suite comparison.
    cond_a_dir="${seed_root}/condition_a_baseline"
    cond_b_dir="${seed_root}/condition_b_baseline_cache"
    cond_c_dir="${seed_root}/condition_c_tg_lora"

    # --- Condition B: Baseline+Cache (N=0) ---
    # Run as: baseline=nocache (dummy baseline) + tg=baseline+cache(N=0)
    # This uses benchmark_prefix_cache.py where "tg" side is actually baseline+cache
    # and "baseline" side is the standard no-cache baseline.
    # The warm tg_lora result IS condition B.
    echo ""
    echo "=== Seed ${seed}: A vs B (baseline no-cache vs baseline+cache) ==="

    # For A vs B comparison, use:
    #   baseline → standard baseline (no cache)
    #   tg_lora  → baseline_with_prefix_cache (N=0)
    _run_condition "A-vs-B" \
        "${BASELINE_CONFIG}" \
        "${BASELINE_CACHE_CONFIG}" \
        "${seed}" \
        "${cond_b_dir}" \
        "${cache_dir}"

    # --- Condition C: TG-LoRA (reuse or re-run) ---
    if [[ -n "${EXISTING_TG_SUITE}" ]]; then
        # Reuse existing TG-LoRA results
        existing_seed_dir="${EXISTING_TG_SUITE}/seed_${seed}/coldwarm"
        if [[ -d "${existing_seed_dir}" ]]; then
            echo "  Reusing existing TG-LoRA results from ${existing_seed_dir}"
            if [[ "${DRY_RUN}" != "true" ]]; then
                mkdir -p "${cond_c_dir}"
                ln -sf "$(realpath "${existing_seed_dir}")" "${cond_c_dir}/coldwarm" 2>/dev/null || {
                    cp -r "${existing_seed_dir}" "${cond_c_dir}/coldwarm"
                }
            fi
        else
            echo "WARNING: Existing TG suite seed_${seed} not found, will re-run" >&2
            _run_condition "C" \
                "${BASELINE_CONFIG}" \
                "${TG_CONFIG}" \
                "${seed}" \
                "${cond_c_dir}" \
                "${cache_dir}"
        fi
    else
        echo ""
        echo "=== Seed ${seed}: B vs C (baseline+cache vs TG-LoRA) ==="
        _run_condition "B-vs-C" \
            "${BASELINE_CACHE_CONFIG}" \
            "${TG_CONFIG}" \
            "${seed}" \
            "${cond_c_dir}" \
            "${cache_dir}"
    fi
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "  DRY RUN complete: all validations passed"
    echo "=============================================="
    exit 0
fi

# ============================================================
# Aggregate 3-condition comparison
# ============================================================
echo ""
echo "=== Generating 3-condition ablation summary ==="

${VENV_PYTHON} - <<'PY'
import json
import statistics
from pathlib import Path

base = Path('${OUTPUT_BASE}')
conditions = {
    "A_baseline": "No cache, no extrapolation",
    "B_baseline_cache": "Cache, no extrapolation (N=0)",
    "C_tg_lora": "Cache + extrapolation (N=1)",
}

cond_map = {
    "A_baseline": "condition_a_baseline",
    "B_baseline_cache": "condition_b_baseline_cache",
    "C_tg_lora": "condition_c_tg_lora",
}

rows = []
for seed_dir in sorted(base.glob("seed_*")):
    seed = int(seed_dir.name.replace("seed_", ""))
    row = {"seed": seed}
    for cond_key, dir_name in cond_map.items():
        summary_path = seed_dir / dir_name / "coldwarm" / "summary.json"
        if not summary_path.exists():
            # For condition A, extract from the B comparison (warm baseline)
            if cond_key == "A_baseline":
                alt_path = seed_dir / "condition_b_baseline_cache" / "coldwarm" / "summary.json"
                if alt_path.exists():
                    summary = json.loads(alt_path.read_text())
                    row["A_baseline_wall_seconds"] = summary["warm"]["baseline"]["wall_seconds"]
                    row["A_baseline_gpu_peak_mb"] = summary["warm"]["baseline"].get("gpu_peak_mb")
                    row["A_baseline_best_valid_loss"] = summary["warm"]["baseline"]["best_valid_loss"]
                    row["A_baseline_loss_red_per_wall_minute"] = summary["warm"]["baseline"].get("loss_red_per_wall_minute")
                    row["A_baseline_backward_passes"] = summary["warm"]["baseline"].get("total_backward_passes")
                    continue
            print(f"WARNING: Missing {cond_key} for seed {seed}")
            continue
        summary = json.loads(summary_path.read_text())
        # For B and C, the "tg_lora" side contains the condition results
        if cond_key == "B_baseline_cache":
            side = summary["warm"]["tg_lora"]
        elif cond_key == "C_tg_lora":
            side = summary["warm"]["tg_lora"]
        else:
            side = summary["warm"]["baseline"]

        prefix = cond_key
        row[f"{prefix}_wall_seconds"] = side.get("wall_seconds")
        row[f"{prefix}_gpu_peak_mb"] = side.get("gpu_peak_mb")
        row[f"{prefix}_best_valid_loss"] = side.get("best_valid_loss")
        row[f"{prefix}_loss_red_per_wall_minute"] = side.get("loss_red_per_wall_minute")
        row[f"{prefix}_backward_passes"] = side.get("total_backward_passes")
        if cond_key in ("B_baseline_cache", "C_tg_lora"):
            row[f"{prefix}_offload_gpu_freed_mb"] = side.get("prefix_feature_cache_runtime_offload_gpu_freed_mb")
    rows.append(row)

# Compute deltas
for row in rows:
    # Cache effect: B - A
    a_loss = row.get("A_baseline_best_valid_loss")
    b_loss = row.get("B_baseline_cache_best_valid_loss")
    if a_loss is not None and b_loss is not None:
        row["cache_effect_valid_loss_delta"] = b_loss - a_loss
    a_peak = row.get("A_baseline_gpu_peak_mb")
    b_peak = row.get("B_baseline_cache_gpu_peak_mb")
    if a_peak is not None and b_peak is not None and a_peak > 0:
        row["cache_effect_gpu_reduction_pct"] = (a_peak - b_peak) / a_peak * 100

    # TG effect: C - B
    b_loss = row.get("B_baseline_cache_best_valid_loss")
    c_loss = row.get("C_tg_lora_best_valid_loss")
    if b_loss is not None and c_loss is not None:
        row["tg_effect_valid_loss_delta"] = c_loss - b_loss

# Aggregate
aggregate = {
    "experiment": "G4_cache_isolation_ablation",
    "seeds": [r["seed"] for r in rows],
    "conditions": conditions,
    "per_seed": rows,
}

out_path = base / "ablation_summary.json"
out_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")

# Markdown report
lines = [
    "# G4 Cache Isolation Ablation Summary",
    "",
    f"- Seeds: {aggregate['seeds']}",
    f"- Target backward passes: ${TARGET_BP}",
    "",
    "## Per-Seed Results",
    "",
    "| Seed | A: Loss | B: Loss | C: Loss | Cache Effect (B-A) | TG Effect (C-B) | A: Peak MB | B: Peak MB | Cache GPU Red % |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(
        f"| {r['seed']} "
        f"| {r.get('A_baseline_best_valid_loss', 'N/A'):.4f} "
        f"| {r.get('B_baseline_cache_best_valid_loss', 'N/A'):.4f} "
        f"| {r.get('C_tg_lora_best_valid_loss', 'N/A'):.4f} "
        f"| {r.get('cache_effect_valid_loss_delta', float('nan')):.4f} "
        f"| {r.get('tg_effect_valid_loss_delta', float('nan')):.4f} "
        f"| {r.get('A_baseline_gpu_peak_mb', 'N/A'):.1f} "
        f"| {r.get('B_baseline_cache_gpu_peak_mb', 'N/A'):.1f} "
        f"| {r.get('cache_effect_gpu_reduction_pct', float('nan')):.1f}% |"
    )

lines.append("")
lines.append("## Attribution")
lines.append("")
lines.append("- **Cache effect** (B - A): Memory savings, wall-clock change from prefix feature cache alone")
lines.append("- **TG-LoRA effect** (C - B): Convergence improvement from trajectory extrapolation alone")

md_path = base / "ablation_summary.md"
md_path.write_text("\n".join(lines) + "\n")

print(json.dumps(aggregate, indent=2, ensure_ascii=False))
print(f"Ablation summary written to {out_path}")
print(f"Markdown summary written to {md_path}")
PY

echo ""
echo "Ablation artifacts written to ${OUTPUT_BASE}"
