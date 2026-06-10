#!/usr/bin/env bash
# Run lm-evaluation-harness for a LoRA adapter merged with base model.
#
# Usage:
#   ./scripts/run_eval_lora.sh <base_model> <adapter_path> [--tasks ...]
#
# This script:
# 1. Merges LoRA adapter with base model (temporary)
# 2. Runs lm-evaluation-harness
# 3. Cleans up merged model

set -euo pipefail

BASE_MODEL="${1:?Usage: $0 <base_model> <adapter_path> [--tasks ...]}"
ADAPTER_PATH="${2:?Usage: $0 <base_model> <adapter_path> [--tasks ...]}"
shift 2

TASKS="arc_easy,hellaswag,gsm8k,truthfulqa_mc2"
OUTPUT_DIR="reports/eval"
MERGED_DIR="/tmp/tg-lora-merged-$$"

cleanup() {
    if [ -d "${MERGED_DIR:-}" ]; then
        rm -rf "$MERGED_DIR"
    fi
}
trap cleanup EXIT INT TERM HUP

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  LoRA Eval Pipeline"
echo "  Base:    $BASE_MODEL"
echo "  Adapter: $ADAPTER_PATH"
echo "  Tasks:   $TASKS"
echo "============================================"

# Merge adapter
echo "Step 1: Merging LoRA adapter..."
python - "$BASE_MODEL" "$ADAPTER_PATH" "$MERGED_DIR" <<'PYEOF'
import sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model, adapter_path, merged_dir = sys.argv[1], sys.argv[2], sys.argv[3]

base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, adapter_path)
merged = model.merge_and_unload()
merged.save_pretrained(merged_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(merged_dir)
print(f"Merged model saved to {merged_dir}")
PYEOF

# Run eval
echo "Step 2: Running evaluation..."
./scripts/run_eval.sh "$MERGED_DIR" --tasks "$TASKS" --output-dir "$OUTPUT_DIR"

echo "Done. Results in: $OUTPUT_DIR"
