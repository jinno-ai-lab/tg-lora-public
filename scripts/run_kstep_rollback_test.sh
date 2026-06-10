#!/usr/bin/env bash
# Test K-step intermediate rollback mechanism.
# High lr + large K: pilot diverges at step N but rollback finds best intermediate point.
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
BUDGET="${BUDGET:-400}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/kstep_test_$(date +%Y%m%d_%H%M%S)}"
TG_CONFIG="${TG_CONFIG:-configs/9b_tg_lora.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline.yaml}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ENV=()

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

echo "=== K-step Intermediate Rollback Test ==="
echo "=== Budget: ${BUDGET} backward passes per run ==="
echo "=== CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE} ==="
echo ""

_run_tg() {
    local name="$1"
    local lr_init="$2"
    local lr_max="$3"
    local K="$4"
    local run_dir="${OUTPUT_BASE}/${name}"
    local n_cycles=$((BUDGET / (K * GRAD_ACCUM)))
    if [ "${n_cycles}" -lt 1 ]; then
        n_cycles=1
    fi
    local eval_every=$(( n_cycles / 4 ))
    if [ "${eval_every}" -lt 1 ]; then
        eval_every=1
    fi

    echo "--- Running ${name}: lr_init=${lr_init}, lr_max=${lr_max}, K=${K}, cycles=${n_cycles} ---"
    mkdir -p "${run_dir}"
    cp "${TG_CONFIG}" "${run_dir}/config.yaml"
    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.training.max_cycles = ${n_cycles}
cfg.tg_lora.K_initial = ${K}
cfg.tg_lora.lr_initial = ${lr_init}
cfg.tg_lora.lr_max = ${lr_max}
cfg.tg_lora.lr_accept_boost = 1.05
cfg.tg_lora.enable_random_walk = True
cfg.tg_lora.enable_convergence_adaptation = True
cfg.tg_lora.force_top_layers_only = False
cfg.tg_lora.active_layer_strategy = 'last_25_percent_plus_random_2'
cfg.training.grad_accumulation = ${GRAD_ACCUM}
cfg.data.max_seq_len = 1024
cfg.eval.quick_eval_examples = 64
cfg.eval.full_eval_every_cycles = ${eval_every}
cfg.eval.moving_avg_window = 3
cfg.eval.soft_accept_temperature = 0.0
cfg.logging.save_every_cycles = 9999
cfg.logging.run_dir = '${run_dir}'
cfg.model.device_map = None
cfg.model.device = None
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"
    env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora --config "${run_dir}/config.yaml" 2>&1 | tail -5
    echo ""
}

_run_baseline() {
    local name="$1"
    local lr="$2"
    local run_dir="${OUTPUT_BASE}/${name}"
    local steps=$((BUDGET / GRAD_ACCUM))
    if [ "${steps}" -lt 1 ]; then
        steps=1
    fi
    local eval_every=$(( steps / 4 ))
    if [ "${eval_every}" -lt 1 ]; then
        eval_every=1
    fi

    echo "--- Running ${name}: lr=${lr}, steps=${steps} ---"
    mkdir -p "${run_dir}"
    cp "${BASELINE_CONFIG}" "${run_dir}/config.yaml"
    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.training.max_steps = ${steps}
cfg.training.learning_rate = ${lr}
cfg.training.grad_accumulation = ${GRAD_ACCUM}
cfg.data.max_seq_len = 1024
cfg.eval.quick_eval_examples = 64
cfg.training.save_every_steps = 9999
cfg.eval.full_eval_every_steps = ${eval_every}
cfg.logging.run_dir = '${run_dir}'
cfg.model.device_map = None
cfg.model.device = None
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"
    env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora --config "${run_dir}/config.yaml" 2>&1 | tail -5
    echo ""
}

# TG-LoRA: safe lr_init → auto-boost to lr_max (tests intermediate rollback at high lr)
# Dynamic lr: accept→lr×1.2, reject→lr×0.5. ~15 accepts to reach lr_max from lr_init.
_run_tg "tg_k5_auto"   "0.0002" "0.005" 5
_run_tg "tg_k8_auto"   "0.0002" "0.005" 8
_run_tg "tg_k5_max2e-3" "0.0002" "0.002" 5

# Baseline for comparison
_run_baseline "b_safe_lr"   "0.0002"
_run_baseline "b_lr2e-3"    "0.002"

# Summary
echo "=== Summary ==="
for d in "${OUTPUT_BASE}"/*/; do
    name=$(basename "${d}")
    footer=$(${VENV_PYTHON} -c "
import orjson, sys
best, final = None, None
with open('${d}/run_metrics.jsonl', 'rb') as f:
    for line in f:
        r = orjson.loads(line)
        if r['type'] == 'run_footer':
            best = r.get('best_valid_loss', 'N/A')
            final = r.get('final_train_loss', 'N/A')
            ppl = r.get('perplexity', 'N/A')
            wall = r.get('total_wall_seconds', 0)
            print(f'${name}  best_valid={best}  final_train={final}  ppl={ppl}  wall={wall:.0f}s')
            break
" 2>/dev/null || echo "${name}  (no footer yet)")
    echo "  ${footer}"
done
