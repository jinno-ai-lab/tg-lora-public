#!/usr/bin/env bash
# Run lm-evaluation-harness on the best config from the accel sweep.
#
# Usage:
#   bash scripts/run_best_config_eval.sh <sweep_dir>
#
# Reads analysis/ranking.json to find the best run, then runs lm-eval
# on its best_model checkpoint.
set -euo pipefail

SWEEP_DIR="${1:?Usage: $0 <sweep_dir>}"
VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
RANKING_JSON="${SWEEP_DIR}/analysis/ranking.json"

if [[ ! -f "${RANKING_JSON}" ]]; then
    echo "Error: ${RANKING_JSON} not found. Run analyze_accel_sweep.py first."
    exit 1
fi

# Extract best run ID from ranking.json
BEST_RUN=$(${VENV_PYTHON} -c "
import json
with open('${RANKING_JSON}') as f:
    data = json.load(f)
best = data.get('best_run', {})
print(best.get('run_id', ''))
" 2>/dev/null)

if [[ -z "${BEST_RUN}" ]]; then
    echo "Error: Could not determine best run from ranking.json"
    exit 1
fi

echo "Best run: ${BEST_RUN}"

# Find the run directory
RUN_DIR=""
for dir in "${SWEEP_DIR}"/tg_lora_9b_accel_*; do
    name=$(basename "$dir")
    if [[ "${name}" == *"${BEST_RUN}"* ]]; then
        RUN_DIR="$dir"
        break
    fi
done

if [[ -z "${RUN_DIR}" ]]; then
    # Try matching by experiment name pattern
    for dir in "${SWEEP_DIR}"/tg_lora_9b_accel_*; do
        metrics="${dir}/run_metrics.jsonl"
        if [[ -f "$metrics" ]]; then
            run_id=$(${VENV_PYTHON} -c "
import json
with open('${metrics}') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('type') == 'run_header':
            print(obj.get('run_id', ''))
            break
" 2>/dev/null)
            if [[ "${run_id}" == "${BEST_RUN}" ]]; then
                RUN_DIR="$dir"
                break
            fi
        fi
    done
fi

if [[ -z "${RUN_DIR}" ]]; then
    echo "Error: Could not find run directory for ${BEST_RUN}"
    exit 1
fi

echo "Run directory: ${RUN_DIR}"

# Check for best_model checkpoint
BEST_MODEL="${RUN_DIR}/best_model"
if [[ ! -d "${BEST_MODEL}" ]]; then
    echo "No best_model checkpoint found. Looking for latest checkpoint..."
    # Use the latest checkpoint
    latest=$(ls -td "${RUN_DIR}"/checkpoint-cycle-* 2>/dev/null | head -1)
    if [[ -n "${latest}" ]]; then
        BEST_MODEL="${latest}"
        echo "Using latest checkpoint: ${BEST_MODEL}"
    else
        echo "Error: No checkpoints found in ${RUN_DIR}"
        exit 1
    fi
fi

# Run lm-evaluation-harness
EVAL_DIR="${SWEEP_DIR}/eval"
mkdir -p "${EVAL_DIR}"

BASE_MODEL="Qwen/Qwen3.5-9B"
TASKS="arc_easy,hellaswag,gsm8k,truthfulqa_mc2"

echo ""
echo "=== Running lm-evaluation-harness ==="
echo "Base model: ${BASE_MODEL}"
echo "LoRA adapter: ${BEST_MODEL}"
echo "Tasks: ${TASKS}"
echo "Output: ${EVAL_DIR}"

${VENV_PYTHON} -m lm_eval \
    --model hf \
    --model_args "pretrained=${BASE_MODEL},peft=${BEST_MODEL},dtype=float16,load_in_4bit=True" \
    --tasks "${TASKS}" \
    --batch_size auto \
    --output_path "${EVAL_DIR}/lm_eval_results.json" \
    --log_samples || {
    echo "Warning: lm-eval with peft loading failed. Trying without 4-bit..."
    ${VENV_PYTHON} -m lm_eval \
        --model hf \
        --model_args "pretrained=${BASE_MODEL},peft=${BEST_MODEL},dtype=float16" \
        --tasks "${TASKS}" \
        --batch_size auto \
        --output_path "${EVAL_DIR}/lm_eval_results.json" \
        --log_samples || {
        echo "Error: lm-eval failed"
        exit 1
    }
}

echo ""
echo "=== Evaluation Results ==="
${VENV_PYTHON} -c "
import json
with open('${EVAL_DIR}/lm_eval_results.json') as f:
    data = json.load(f)
if isinstance(data, dict) and 'results' in data:
    results = data['results']
    for task, metrics in results.items():
        acc = metrics.get('acc,none', metrics.get('acc_norm,none', 'N/A'))
        print(f'  {task}: {acc}')
else:
    print(json.dumps(data, indent=2))
"

echo ""
echo "Results saved to: ${EVAL_DIR}/lm_eval_results.json"
