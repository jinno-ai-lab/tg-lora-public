#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PSA γ (Gain) Sweep — ablation framework (GOAL §4 step 3)
#
# PURPOSE: Measure PSA effectiveness by sweeping the amplification gain (γ)
#   with and without regime-aware prior reset. This isolates:
#   1. Whether PSA amplification helps at all (γ=0 ablation baseline)
#   2. The optimal gain value
#   3. Whether regime-aware prior reset improves results
#
# CONTROLLED VARIABLES (identical across all configs):
#   - Model: Qwen3.5-9B, 4-bit QLoRA (r=16, alpha=32)
#   - Training: batch=1, grad_accum=8, lr=2e-4, max_cycles=120
#   - PSA: history_length=6, update_interval=3, warmup_steps=4, l2_reg=0.01
#   - All other TG-LoRA params from configs/9b_tg_lora_psa.yaml
#
# SWEPT VARIABLES:
#   psa_gain           ∈ {0.0, 0.5, 1.0, 2.0}
#   psa_regime_reset   ∈ {true, false}
#
# CONFIG GRID (4×2 = 8 runs):
#   gamma_0.0_reset_on   | γ=0.0, regime reset enabled  (no-amplification baseline)
#   gamma_0.0_reset_off  | γ=0.0, regime reset disabled
#   gamma_0.5_reset_on   | γ=0.5, regime reset enabled  (current default)
#   gamma_0.5_reset_off  | γ=0.5, regime reset disabled
#   gamma_1.0_reset_on   | γ=1.0, regime reset enabled
#   gamma_1.0_reset_off  | γ=1.0, regime reset disabled
#   gamma_2.0_reset_on   | γ=2.0, regime reset enabled
#   gamma_2.0_reset_off  | γ=2.0, regime reset disabled
#
# ANALYSIS: scripts/summarize_psa_sweep.py — PSA-specific metrics including
#   regime transition counts, per-γ efficiency, and regime reset effect size.
# ---------------------------------------------------------------------------
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
OUTPUT_BASE="${OUTPUT_BASE:-reports/psa_gamma_sweep/$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="${BASE_CONFIG:-configs/9b_tg_lora_psa.yaml}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
SEED="${SEED:-42}"

# Sweep grid: gamma × regime_reset
GAMMA_VALUES=(0.0 0.5 1.0 2.0)
REGIME_RESET_VALUES=(true false)

total=${#GAMMA_VALUES[@]}
run_count=0

echo "=== PSA γ Sweep: ${#GAMMA_VALUES[@]} γ × ${#REGIME_RESET_VALUES[@]} regime = $(( total * ${#REGIME_RESET_VALUES[@]} )) configs ==="
echo "=== Output: ${OUTPUT_BASE} ==="
echo "=== Base config: ${BASE_CONFIG} ==="

mkdir -p "${OUTPUT_BASE}"

CONFIG_DIRS=()

for gamma in "${GAMMA_VALUES[@]}"; do
  for regime_reset in "${REGIME_RESET_VALUES[@]}"; do
    reset_tag="on"
    if [[ "${regime_reset}" == "false" ]]; then
      reset_tag="off"
    fi
    experiment_name="gamma_${gamma}_reset_${reset_tag}"
    run_dir="${OUTPUT_BASE}/${experiment_name}"
    CONFIG_DIRS+=("${run_dir}")
    run_count=$((run_count + 1))

    echo ""
    echo "--- [${run_count}/$(( total * ${#REGIME_RESET_VALUES[@]} ))] Running: ${experiment_name} (γ=${gamma}, regime_reset=${regime_reset}) ---"
    mkdir -p "${run_dir}"
    cp "${BASE_CONFIG}" "${run_dir}/config.yaml"

    # Apply sweep overrides
    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.experiment.name = '${experiment_name}'
cfg.experiment.seed = ${SEED}
cfg.tg_lora.psa_gain = ${gamma}
cfg.tg_lora.psa_regime_reset_enabled = ${regime_reset}
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
done

# --- Generate PSA-specific analysis ---
echo ""
echo "--- Generating PSA Sweep Analysis ---"

REPORT_DIR="${OUTPUT_BASE}/report"
mkdir -p "${REPORT_DIR}"

if [[ -f scripts/summarize_psa_sweep.py ]]; then
  ${VENV_PYTHON} scripts/summarize_psa_sweep.py --sweep-dir "${OUTPUT_BASE}" || true
fi

# Also run the general sweep summary for cross-comparison
if [[ -f scripts/summarize_sweep.py ]]; then
  ${VENV_PYTHON} -m scripts.summarize_sweep --sweep-dir "${OUTPUT_BASE}" || true
fi

# --- Summary ---
echo ""
echo "=== PSA γ Sweep Complete ==="
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
