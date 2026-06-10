#!/usr/bin/env bash
# Run lm-evaluation-harness benchmarks against a trained model.
#
# Usage:
#   ./scripts/run_eval.sh <model_path> [--tasks arc_easy,hellaswag,truthfulqa_mc2]
#
# Defaults to MLX backend (--model mlx). Set LM_EVAL_MODEL to override.
#
# Prerequisites:
#   pip install -e ~/lm-evaluation-harness

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> [--tasks task1,task2,...]}"
shift

# Default tasks for initial validation
TASKS="arc_easy,hellaswag,gsm8k,truthfulqa_mc2"
OUTPUT_DIR="reports/eval"
BATCH_SIZE="1"
BACKEND="mlx"
ADAPTER_PATH=""
NUM_FEWSHOT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --adapter)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --num-fewshot)
            NUM_FEWSHOT="--num_fewshot $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

LM_EVAL_HARNESS="${LM_EVAL_HARNESS:-$HOME/lm-evaluation-harness}"
PYTHONPATH="${LM_EVAL_HARNESS}:${PYTHONPATH:-}"

echo "============================================"
echo "  lm-evaluation-harness (backend: ${BACKEND})"
echo "  Model: $MODEL_PATH"
if [ -n "$ADAPTER_PATH" ]; then
    echo "  Adapter: $ADAPTER_PATH"
fi
echo "  Tasks: $TASKS"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

if [ "$BACKEND" = "mlx" ]; then
    MODEL_ARGS="model=${MODEL_PATH}"
    if [ -n "$ADAPTER_PATH" ]; then
        MODEL_ARGS="${MODEL_ARGS},adapter_path=${ADAPTER_PATH}"
    fi
    PYTHONPATH="$PYTHONPATH" python -m lm_eval \
        --model mlx \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$OUTPUT_DIR" \
        ${NUM_FEWSHOT}
else
    # HF backend (CUDA)
    if ! command -v lm_eval &> /dev/null; then
        echo "ERROR: lm_eval not found. Install with:"
        echo "  pip install -e ~/lm-evaluation-harness"
        exit 1
    fi
    lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL_PATH},dtype=float16" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$OUTPUT_DIR" \
        ${NUM_FEWSHOT} \
        --log_samples
fi

echo ""
echo "Evaluation complete. Results in: $OUTPUT_DIR"
