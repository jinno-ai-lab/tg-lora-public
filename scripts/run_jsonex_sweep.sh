#!/usr/bin/env bash
# Full 3-condition JSON-extraction efficiency sweep: plain vs LAWA vs PSA.
# Runs each condition to MAX_CYCLES, logs separately, then analyzes.
# All three share identical backward accounting & data-digestion (GOAL.md §3.3);
# conditions differ only by the enable_psa / enable_lawa toggles in each config.
#
# Usage: bash scripts/run_jsonex_sweep.sh [MAX_CYCLES]
set -u
cd /home/jinno/tg-lora
MAX=${1:-40}
CONDS=(plain lawa psa)

echo "=== sweep start $(date) max_cycles=$MAX ==="
for cond in "${CONDS[@]}"; do
  echo "=== $(date) START $cond ==="
  rm -rf "runs/jsonex_$cond"
  python -m src.training.train_tg_lora \
    --config "configs/jsonex_${cond}.yaml" \
    --override "training.max_cycles=${MAX}" \
    > "runs_jsonex_${cond}.log" 2>&1
  ec=$?
  echo "=== $(date) END $cond (exit $ec) ==="
done

echo "=== $(date) ANALYSIS ==="
python scripts/analyze_json_experiment.py 2>&1 | tee runs/jsonex_analysis/sweep_stdout.txt
echo "=== sweep done $(date) ==="
