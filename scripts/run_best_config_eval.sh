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

# Extract best run ID from ranking.json. Delegate to best_run_reader.py so a
# malformed or wrong-shape ranking.json fails LOUD (stderr cause + non-zero
# exit) instead of the old inline reader's silent swallow: a malformed file
# raised behind `2>/dev/null` and `set -e` killed the shell with ZERO output,
# while a valid-JSON-but-non-object file was silently coerced to `{}` and
# surfaced only as a misleading "could not determine best run".
BEST_RUN=$("${VENV_PYTHON}" "$(dirname "$0")/best_run_reader.py" "${RANKING_JSON}") || {
    # best_run_reader.py already wrote the cause to stderr.
    echo "Error: Could not determine best run from ${RANKING_JSON}" >&2
    exit 1
}

if [[ -z "${BEST_RUN}" ]]; then
    echo "Error: ${RANKING_JSON}: best_run.run_id resolved empty" >&2
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
    # Try matching by the run_id inside each candidate's run_metrics.jsonl.
    # Delegate to run_metrics_reader.py so a malformed metrics file fails LOUD
    # (stderr cause + non-zero exit) instead of the old inline reader's
    # `2>/dev/null` swallow: a corrupt line raised JSONDecodeError, the
    # traceback was eaten, run_id came back empty, and the operator saw only a
    # misleading "Could not find run directory" — never the corrupt file.
    # `|| continue` keeps searching sibling dirs; the helper has already written
    # the cause to stderr, so the failure is VISIBLE (not silent) without
    # aborting the whole sweep on one bad sibling.
    for dir in "${SWEEP_DIR}"/tg_lora_9b_accel_*; do
        metrics="${dir}/run_metrics.jsonl"
        [[ -f "$metrics" ]] || continue
        run_id=$("${VENV_PYTHON}" "$(dirname "$0")/run_metrics_reader.py" "${metrics}") || continue
        if [[ "${run_id}" == "${BEST_RUN}" ]]; then
            RUN_DIR="$dir"
            break
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
