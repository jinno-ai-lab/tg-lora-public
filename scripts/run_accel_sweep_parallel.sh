#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Parallel accel param sweep — runs 2 configs simultaneously on 2 GPUs.
#
# Round 1: no_accel (cuda:0) + conservative (cuda:1)
# Round 2: balanced   (cuda:0) + aggressive  (cuda:1)
#
# Usage:
#   bash scripts/run_accel_sweep_parallel.sh [max_cycles]
# ---------------------------------------------------------------------------
set -euo pipefail

MAX_CYCLES="${1:-50}"
VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
OUTPUT_BASE="reports/accel_sweep/$(date +%Y%m%d_%H%M%S)"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"

echo "=== Parallel Accel Sweep: max_cycles=${MAX_CYCLES}, output=${OUTPUT_BASE} ==="
mkdir -p "${OUTPUT_BASE}"

run_config() {
    local config="$1"
    local gpu="$2"
    local shift_arg="$3"

    if [[ ! -f "${config}" ]]; then
        echo "WARNING: ${config} not found, skipping"
        return
    fi

    local experiment_name
    experiment_name="$(grep '^  name:' "${config}" | head -1 | sed 's/.*: *//')"
    local run_dir="${OUTPUT_BASE}/${experiment_name}"
    mkdir -p "${run_dir}"
    cp "${config}" "${run_dir}/config.yaml"

    # Patch config: override device, cycles, mlflow
    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.model.device = '${gpu}'
cfg.training.max_cycles = ${MAX_CYCLES}
cfg.logging.run_dir = '${run_dir}'
if 'mlflow' not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = '${MLFLOW_ENABLED}'.lower() == 'true'
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

    echo "[${gpu}] Starting: ${experiment_name}"
    ${VENV_PYTHON} -m src.training.train_tg_lora --config "${run_dir}/config.yaml" \
        > "${run_dir}/train.log" 2>&1 && echo "[${gpu}] DONE: ${experiment_name}" \
        || echo "[${gpu}] ERROR: ${experiment_name} (check ${run_dir}/train.log)" &
}

# Round 1: no_accel + conservative
echo ""
echo "=== Round 1: no_accel (cuda:0) + conservative (cuda:1) ==="
run_config "configs/9b_tg_lora_accel_no_accel.yaml" "cuda:0" "no_accel"
run_config "configs/9b_tg_lora_accel_conservative.yaml" "cuda:1" "conservative"
wait
echo "=== Round 1 complete ==="

# Round 2: balanced + aggressive
echo ""
echo "=== Round 2: balanced (cuda:0) + aggressive (cuda:1) ==="
run_config "configs/9b_tg_lora_accel_balanced.yaml" "cuda:0" "balanced"
run_config "configs/9b_tg_lora_accel_aggressive.yaml" "cuda:1" "aggressive"
wait
echo "=== Round 2 complete ==="

# --- Post-sweep analysis ---
echo ""
echo "=== Generating Analysis ==="

REPORT_DIR="${OUTPUT_BASE}/report"
mkdir -p "${REPORT_DIR}"

# Pairwise comparison
BASELINE_METRICS=""
TREATMENT_DIRS=()
for dir in "${OUTPUT_BASE}"/tg_lora_9b_accel_*; do
    metrics_file="${dir}/run_metrics.jsonl"
    [[ -f "${metrics_file}" ]] || continue
    name="$(basename "${dir}")"
    if [[ "${name}" == *"no_accel"* ]]; then
        BASELINE_METRICS="${metrics_file}"
    else
        TREATMENT_DIRS+=("${dir}")
    fi
done

if [[ -n "${BASELINE_METRICS}" ]]; then
    for dir in "${TREATMENT_DIRS[@]}"; do
        metrics_file="${dir}/run_metrics.jsonl"
        name="$(basename "${dir}")"
        echo "  Comparing: ${name} vs no_accel baseline"
        ${VENV_PYTHON} scripts/compare_runs.py \
            --baseline "${BASELINE_METRICS}" \
            --tg-lora "${metrics_file}" \
            --output-dir "${REPORT_DIR}/${name}" || true
    done
fi

# Sweep summary
${VENV_PYTHON} -m scripts.summarize_sweep --sweep-dir "${OUTPUT_BASE}" || true

# Detailed analysis
${VENV_PYTHON} scripts/analyze_accel_sweep.py "${OUTPUT_BASE}" || true

# HTML dashboard
if [[ -f "${OUTPUT_BASE}/analysis/ranking.json" ]]; then
    ${VENV_PYTHON} scripts/generate_sweep_dashboard.py "${OUTPUT_BASE}" || true
fi

# --- Summary ---
echo ""
echo "=== Sweep Complete ==="
for dir in "${OUTPUT_BASE}"/tg_lora_9b_accel_*; do
    name="$(basename "${dir}")"
    metrics="${dir}/run_metrics.jsonl"
    if [[ -f "${metrics}" ]]; then
        echo "  [OK] ${name}"
    else
        echo "  [--] ${name} (no metrics)"
    fi
done
echo "=== Results in: ${OUTPUT_BASE} ==="
