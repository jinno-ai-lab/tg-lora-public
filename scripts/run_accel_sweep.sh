#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Accel Param Sweep — experiment design
#
# PURPOSE: Isolate the effect of acceleration-based lr/K adaptation by sweeping
#   accel_instability_lr_decay × accel_convergence_lr_boost while holding all
#   other TG-LoRA hyperparameters constant.
#
# CONTROLLED VARIABLES (identical across all 4 configs):
#   - Model: Qwen3.5-9B, 4-bit QLoRA (r=16, alpha=32)
#   - Training: batch=1, grad_accum=8, lr=2e-4, max_cycles=500
#   - TG-LoRA: K_initial=3, N_initial=5, alpha_initial=0.3, beta_initial=0.8
#   - enable_random_walk: false (no exploration noise)
#   - enable_convergence_adaptation: true (trend-based K/lr adjustment)
#   - All explore probs: 0.0 (deterministic proposals only)
#   - confident_skip_cos: 0.0 (never skip eval)
#
# SWEPT VARIABLE (the only difference between configs):
#   accel_instability_lr_decay  — lr multiplier when velocity accelerates
#   accel_convergence_lr_boost  — lr multiplier when velocity decelerates
#
# CONFIG GRID:
#   conservative:  decay=0.3  boost=1.1  (strong instability brake, modest recovery)
#   aggressive:    decay=0.9  boost=2.0  (weak instability brake, strong recovery)
#   balanced:      decay=0.5  boost=1.5  (middle ground)
#   no_accel:      decay=0.99 boost=1.01 (near-identity ablation baseline)
#
# ANALYSIS: Pairwise comparison of each treatment vs no_accel baseline.
#   Uses scripts/compare_runs.py for text/markdown reports and plots.
# ---------------------------------------------------------------------------
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
OUTPUT_BASE="${OUTPUT_BASE:-reports/accel_sweep/$(date +%Y%m%d_%H%M%S)}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"

CONFIGS=(
  configs/9b_tg_lora_accel_conservative.yaml
  configs/9b_tg_lora_accel_aggressive.yaml
  configs/9b_tg_lora_accel_balanced.yaml
  configs/9b_tg_lora_accel_no_accel.yaml
)

echo "=== Accel Param Sweep: ${#CONFIGS[@]} configs ==="
echo "=== Output: ${OUTPUT_BASE} ==="

mkdir -p "${OUTPUT_BASE}"

CONFIG_DIRS=()

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "WARNING: ${config} not found, skipping"
    continue
  fi

  experiment_name="$(grep '^  name:' "${config}" | head -1 | sed 's/.*: *//')"
  run_dir="${OUTPUT_BASE}/${experiment_name}"
  CONFIG_DIRS+=("${run_dir}")

  echo ""
  echo "--- Running: ${experiment_name} (${config}) ---"
  mkdir -p "${run_dir}"
  cp "${config}" "${run_dir}/config.yaml"

  ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.logging.run_dir = '${run_dir}'
if 'mlflow' not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = '${MLFLOW_ENABLED}'.lower() == 'true'
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

  ${VENV_PYTHON} -m src.training.train_tg_lora --config "${run_dir}/config.yaml" || {
    echo "ERROR: ${experiment_name} failed, continuing to next config"
    continue
  }
done

# --- Generate comparison report ---
echo ""
echo "--- Generating Sweep Comparison Report ---"

REPORT_DIR="${OUTPUT_BASE}/report"
mkdir -p "${REPORT_DIR}"

if command -v ${VENV_PYTHON} &>/dev/null && [[ -f scripts/compare_runs.py ]]; then
  # Find the no-accel baseline metrics
  baseline_metrics=""
  treatment_dirs=()
  for dir in "${CONFIG_DIRS[@]}"; do
    metrics_file="${dir}/run_metrics.jsonl"
    if [[ ! -f "${metrics_file}" ]]; then
      continue
    fi
    name="$(basename "${dir}")"
    if [[ "${name}" == *"no_accel"* ]]; then
      baseline_metrics="${metrics_file}"
    else
      treatment_dirs+=("${dir}")
    fi
  done

  # Pairwise comparison: each treatment vs no-accel baseline
  if [[ -n "${baseline_metrics}" ]]; then
    for dir in "${treatment_dirs[@]}"; do
      metrics_file="${dir}/run_metrics.jsonl"
      name="$(basename "${dir}")"
      echo "  Comparing: ${name} vs no_accel baseline"
      ${VENV_PYTHON} scripts/compare_runs.py \
        --baseline "${baseline_metrics}" \
        --tg-lora "${metrics_file}" \
        --output-dir "${REPORT_DIR}/${name}" || true
    done
  else
    echo "  WARNING: no_accel baseline not found, using first available as baseline"
    first_metrics=""
    for dir in "${CONFIG_DIRS[@]}"; do
      metrics_file="${dir}/run_metrics.jsonl"
      if [[ -f "${metrics_file}" ]]; then
        if [[ -z "${first_metrics}" ]]; then
          first_metrics="${metrics_file}"
        else
          name="$(basename "${dir}")"
          ${VENV_PYTHON} scripts/compare_runs.py \
            --baseline "${first_metrics}" \
            --tg-lora "${metrics_file}" \
            --output-dir "${REPORT_DIR}/${name}" || true
        fi
      fi
    done
  fi

  # Multi-run dashboard for overview
  ${VENV_PYTHON} scripts/compare_runs.py dashboard "${OUTPUT_BASE}" --format json \
    > "${REPORT_DIR}/dashboard.json" 2>/dev/null || true

  # Sweep summary with efficiency ranking and next-action recommendations
  ${VENV_PYTHON} -m scripts.summarize_sweep --sweep-dir "${OUTPUT_BASE}" || true

  # Detailed accel analysis with pairwise deltas and efficiency metrics
  ${VENV_PYTHON} scripts/analyze_accel_sweep.py "${OUTPUT_BASE}" || true

  # HTML dashboard generation
  if [[ -f "${OUTPUT_BASE}/analysis/ranking.json" ]]; then
    ${VENV_PYTHON} scripts/generate_sweep_dashboard.py "${OUTPUT_BASE}" || true
  fi
fi

# --- Summary ---
echo ""
echo "=== Sweep Complete ==="
echo "=== Configs run: ${#CONFIG_DIRS[@]} ==="
echo "=== Results in: ${OUTPUT_BASE} ==="

for dir in "${CONFIG_DIRS[@]}"; do
  name="$(basename "${dir}")"
  metrics="${dir}/run_metrics.jsonl"
  if [[ -f "${metrics}" ]]; then
    echo "  [OK] ${name}"
  else
    echo "  [--] ${name} (no metrics)"
  fi
done
