#!/usr/bin/env bash
# Run remaining 3 accel configs sequentially on cuda:0 after no_accel finishes.
# Also runs full analysis pipeline after all configs complete.
set -euo pipefail

SWEEP_DIR="reports/accel_sweep/20260524_134239"
VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
MAX_CYCLES=25

echo "=== Running remaining accel configs ==="
echo "Sweep dir: ${SWEEP_DIR}"

CONFIGS=(
    "configs/9b_tg_lora_accel_conservative.yaml"
    "configs/9b_tg_lora_accel_balanced.yaml"
    "configs/9b_tg_lora_accel_aggressive.yaml"
)

for config in "${CONFIGS[@]}"; do
    if [[ ! -f "${config}" ]]; then
        echo "WARNING: ${config} not found, skipping"
        continue
    fi

    experiment_name="$(grep '^  name:' "${config}" | head -1 | sed 's/.*: *//')"
    run_dir="${SWEEP_DIR}/${experiment_name}"

    # Skip if already completed
    if [[ -f "${run_dir}/run_metrics.jsonl" ]]; then
        existing=$(wc -l < "${run_dir}/run_metrics.jsonl")
        if [[ "${existing}" -gt 5 ]]; then
            echo "SKIP: ${experiment_name} already has ${existing} lines"
            continue
        fi
    fi

    mkdir -p "${run_dir}"
    cp "${config}" "${run_dir}/config.yaml"

    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.model.device = 'cuda:0'
cfg.training.max_cycles = ${MAX_CYCLES}
cfg.logging.run_dir = '${run_dir}'
if 'mlflow' not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = False
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

    echo ""
    echo "--- Running: ${experiment_name} ---"
    ${VENV_PYTHON} -m src.training.train_tg_lora --config "${run_dir}/config.yaml" \
        2>&1 | tee "${run_dir}/train.log" || echo "ERROR: ${experiment_name} failed"
done

# --- Post-sweep analysis ---
echo ""
echo "=== Running Analysis ==="

REPORT_DIR="${SWEEP_DIR}/report"
mkdir -p "${REPORT_DIR}"

BASELINE_METRICS=""
TREATMENT_DIRS=()
for dir in "${SWEEP_DIR}"/tg_lora_9b_accel_*; do
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

${VENV_PYTHON} -m scripts.summarize_sweep --sweep-dir "${SWEEP_DIR}" || true
${VENV_PYTHON} scripts/analyze_accel_sweep.py "${SWEEP_DIR}" || true

if [[ -f "${SWEEP_DIR}/analysis/ranking.json" ]]; then
    ${VENV_PYTHON} scripts/generate_sweep_dashboard.py "${SWEEP_DIR}" || true
fi

# --- lm-evaluation-harness on best config ---
echo ""
echo "=== Running lm-evaluation-harness on best config ==="
bash scripts/run_best_config_eval.sh "${SWEEP_DIR}" || echo "Warning: lm-eval failed"

echo ""
echo "=== Sweep Complete ==="
for dir in "${SWEEP_DIR}"/tg_lora_9b_accel_*; do
    name="$(basename "${dir}")"
    metrics="${dir}/run_metrics.jsonl"
    if [[ -f "${metrics}" ]]; then
        cycles=$(${VENV_PYTHON} -c "
import json
steps = 0
with open('${metrics}') as f:
    for line in f:
        if json.loads(line).get('type') == 'step':
            steps += 1
print(steps)
")
        echo "  [OK] ${name} (${cycles} cycles)"
    else
        echo "  [--] ${name} (no metrics)"
    fi
done
echo "=== Results in: ${SWEEP_DIR} ==="
