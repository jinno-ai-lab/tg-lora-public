#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
SWEEP_BUDGET="${SWEEP_BUDGET:-100}"
SWEEP_DIR="${SWEEP_DIR:-runs/sweep_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/9b_tg_lora.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ENV=()

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

# Sweep grid: name|override1,override2,...
# 9 configs × ~13 min each ≈ 2 hours total (quick_eval=32)
SWEEP_GRID=(
  "lr_1e-4|training.learning_rate=0.0001"
  "lr_2e-4|training.learning_rate=0.0002"
  "lr_5e-4|training.learning_rate=0.0005"
  "rt_0.001|eval.rollback_tolerance=0.001"
  "rt_0.005|eval.rollback_tolerance=0.005"
  "rt_0.01|eval.rollback_tolerance=0.01"
  "K2_N3|tg_lora.K_initial=2,tg_lora.N_initial=3"
  "K3_N5|tg_lora.K_initial=3,tg_lora.N_initial=5"
  "K5_N10|tg_lora.K_initial=5,tg_lora.N_initial=10"
)

TOTAL=${#SWEEP_GRID[@]}
CURRENT=0

echo "=== TG-LoRA Hyperparameter Sweep ==="
echo "=== Budget: ${SWEEP_BUDGET} backward passes | ${TOTAL} configs ==="
echo "=== Output: ${SWEEP_DIR} ==="
echo "=== CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE} ==="
echo ""

for entry in "${SWEEP_GRID[@]}"; do
  CURRENT=$((CURRENT + 1))
  name="${entry%%|*}"
  overrides="${entry#*|}"

  run_dir="${SWEEP_DIR}/${name}"
  mkdir -p "${run_dir}"

  echo "[${CURRENT}/${TOTAL}] Sweep: ${name}"

  cp "${CONFIG}" "${run_dir}/config.yaml"

  ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.data.max_seq_len = 1024
cfg.eval.quick_eval_examples = 32
cfg.eval.full_eval_every_cycles = 9999
cfg.logging.save_every_cycles = 9999
cfg.logging.run_dir = '${run_dir}'
cfg.logging.mlflow.enabled = False
cfg.model.device_map = None
cfg.model.device = None
for pair in '${overrides}'.split(','):
    k, v = pair.split('=', 1)
    OmegaConf.update(cfg, k.strip(), __import__('ast').literal_eval(v.strip()))
cfg.training.max_cycles = max(1, ${SWEEP_BUDGET} // (cfg.tg_lora.K_initial * cfg.training.grad_accumulation))
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

  env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora --config "${run_dir}/config.yaml"
  echo ""
done

echo "=== Sweep Summary ==="
${VENV_PYTHON} scripts/summarize_sweep.py --sweep-dir "${SWEEP_DIR}"
echo ""
echo "=== Sweep complete. Results in ${SWEEP_DIR} ==="
