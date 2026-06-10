#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
BUDGET_PASSES="${BUDGET_PASSES:-1500}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/comparison_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline.yaml}"
TG_LORA_CONFIG="${TG_LORA_CONFIG:-configs/9b_tg_lora.yaml}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-64}"
EVAL_POINTS="${EVAL_POINTS:-4}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
K_INITIAL_OVERRIDE="${K_INITIAL:-}"
TG_PREFIX_CACHE_DIR="${TG_PREFIX_CACHE_DIR:-}"
TG_PREFIX_FORCE_REBUILD="${TG_PREFIX_FORCE_REBUILD:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ENV=()

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

BASELINE_DIR="${OUTPUT_BASE}/baseline"
TG_LORA_DIR="${OUTPUT_BASE}/tg_lora"

echo "=== Fair Comparison: budget=${BUDGET_PASSES} backward passes ==="
echo "=== Baseline config: ${BASELINE_CONFIG} ==="
echo "=== TG-LoRA config: ${TG_LORA_CONFIG} ==="
echo "=== max_seq_len=${MAX_SEQ_LEN} quick_eval_examples=${QUICK_EVAL_EXAMPLES} eval_points=${EVAL_POINTS} mlflow=${MLFLOW_ENABLED} ==="
echo "=== CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE} ==="
if [[ -n "${TG_PREFIX_CACHE_DIR}" ]]; then
    echo "=== tg_prefix_cache_dir=${TG_PREFIX_CACHE_DIR} force_rebuild=${TG_PREFIX_FORCE_REBUILD:-false} ==="
fi

# --- Run Baseline ---
echo ""
echo "--- [1/3] Running Baseline QLoRA ---"
mkdir -p "${BASELINE_DIR}"
cp "${BASELINE_CONFIG}" "${BASELINE_DIR}/config.yaml"
${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${BASELINE_DIR}/config.yaml')
steps = max(1, ${BUDGET_PASSES} // cfg.training.grad_accumulation)
eval_every = max(1, steps // ${EVAL_POINTS})
cfg.training.max_steps = steps
cfg.training.save_every_steps = 9999
cfg.training.early_stopping_patience = None
cfg.data.max_seq_len = ${MAX_SEQ_LEN}
cfg.eval.quick_eval_examples = ${QUICK_EVAL_EXAMPLES}
cfg.eval.full_eval_every_steps = eval_every
cfg.logging.run_dir = '${BASELINE_DIR}'
cfg.model.device_map = None
cfg.model.device = None
if 'mlflow' not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = '${MLFLOW_ENABLED}'.lower() == 'true'
OmegaConf.save(cfg, '${BASELINE_DIR}/config.yaml')
"
env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora --config "${BASELINE_DIR}/config.yaml"

# --- Run TG-LoRA ---
echo ""
echo "--- [2/3] Running TG-LoRA ---"
mkdir -p "${TG_LORA_DIR}"
cp "${TG_LORA_CONFIG}" "${TG_LORA_DIR}/config.yaml"
${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${TG_LORA_DIR}/config.yaml')
if '${K_INITIAL_OVERRIDE}':
    cfg.tg_lora.K_initial = int('${K_INITIAL_OVERRIDE}')
planned_cycles = max(1, ${BUDGET_PASSES} // (cfg.training.grad_accumulation * cfg.tg_lora.K_initial))
eval_every = max(1, planned_cycles // ${EVAL_POINTS})
cfg.training.max_cycles = planned_cycles
cfg.training.early_stopping_patience = None
cfg.data.max_seq_len = ${MAX_SEQ_LEN}
cfg.eval.quick_eval_examples = ${QUICK_EVAL_EXAMPLES}
cfg.logging.save_every_cycles = 9999
cfg.eval.full_eval_every_cycles = eval_every
cfg.logging.run_dir = '${TG_LORA_DIR}'
cfg.model.device_map = None
cfg.model.device = None
if 'mlflow' not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = '${MLFLOW_ENABLED}'.lower() == 'true'
if '${TG_PREFIX_CACHE_DIR}':
    cfg.training.prefix_feature_cache_dir = '${TG_PREFIX_CACHE_DIR}'
if '${TG_PREFIX_FORCE_REBUILD}':
    cfg.training.prefix_feature_cache_force_rebuild = '${TG_PREFIX_FORCE_REBUILD}'.lower() == 'true'
OmegaConf.save(cfg, '${TG_LORA_DIR}/config.yaml')
"
env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora --config "${TG_LORA_DIR}/config.yaml"

# --- Compare ---
echo ""
echo "--- [3/3] Generating Comparison Report ---"
REPORT_DIR="${OUTPUT_BASE}/reports"
${VENV_PYTHON} scripts/compare_runs.py \
    --baseline "${BASELINE_DIR}/run_metrics.jsonl" \
    --tg-lora "${TG_LORA_DIR}/run_metrics.jsonl" \
    --output-dir "${REPORT_DIR}"

echo ""
echo "=== Comparison complete. Report at ${REPORT_DIR}/ ==="
