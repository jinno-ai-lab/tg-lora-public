#!/usr/bin/env bash
set -euo pipefail

# High-LR Comparison: Baseline vs TG-LoRA at aggressive learning rates
# Demonstrates TG-LoRA's stability advantage: rollback prevents divergence
# even at 10-25x normal lr.

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
BUDGET="${BUDGET:-500}"
COMPARISON_MODE="${COMPARISON_MODE:-stability}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/high_lr_compare_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline.yaml}"
TG_LORA_CONFIG="${TG_LORA_CONFIG:-configs/9b_tg_lora.yaml}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-64}"
BASELINE_PRETRAIN_REFERENCE_EVAL_ENABLED="${BASELINE_PRETRAIN_REFERENCE_EVAL_ENABLED:-true}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN_ENV=()

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

read -r BASELINE_GRAD_ACCUM TG_CYCLE_BACKWARD_PASSES COMPARABLE_BUDGET_UNIT <<EOF
$({ BASELINE_CONFIG="${BASELINE_CONFIG}" TG_LORA_CONFIG="${TG_LORA_CONFIG}" ${VENV_PYTHON} - <<'PY'
import math
import os
from omegaconf import OmegaConf

b_cfg = OmegaConf.load(os.environ['BASELINE_CONFIG'])
t_cfg = OmegaConf.load(os.environ['TG_LORA_CONFIG'])
b_ga = int(b_cfg.training.grad_accumulation)
t_cycle_bp = int(t_cfg.training.grad_accumulation) * int(t_cfg.tg_lora.K_initial)
print(b_ga, t_cycle_bp, math.lcm(b_ga, t_cycle_bp))
PY
};)
EOF

ACTUAL_BUDGET_PASSES="${BUDGET}"
if [[ "${COMPARISON_MODE}" == "parity" ]]; then
  if (( BUDGET < COMPARABLE_BUDGET_UNIT )); then
    echo "ERROR: parity mode requires BUDGET >= ${COMPARABLE_BUDGET_UNIT} backward passes" >&2
    exit 2
  fi
  ACTUAL_BUDGET_PASSES=$(( BUDGET / COMPARABLE_BUDGET_UNIT * COMPARABLE_BUDGET_UNIT ))
  if (( ACTUAL_BUDGET_PASSES != BUDGET )); then
    echo "Normalized requested budget ${BUDGET} -> comparable budget ${ACTUAL_BUDGET_PASSES} backward passes"
  fi
elif [[ "${COMPARISON_MODE}" != "stability" ]]; then
  echo "ERROR: COMPARISON_MODE must be 'stability' or 'parity'" >&2
  exit 2
fi

echo "=============================================="
echo "  High-LR Stability Comparison"
echo "  Budget: ${BUDGET} backward passes"
echo "  Comparison mode: ${COMPARISON_MODE}"
if [[ "${COMPARISON_MODE}" == "parity" ]]; then
  echo "  Comparable budget unit: ${COMPARABLE_BUDGET_UNIT} backward passes"
  echo "  Effective comparable budget: ${ACTUAL_BUDGET_PASSES} backward passes"
else
  echo "  Effective budget: ${ACTUAL_BUDGET_PASSES} backward passes"
  echo "  Note: stability mode does not enforce equal backward-pass budgets"
fi
echo "  Output: ${OUTPUT_BASE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "=============================================="
echo ""

# --- Experiment definitions ---
# name|type|config_overrides
declare -a EXPERIMENTS=(
  "b_safe_lr|baseline|training.learning_rate=0.0002,training.schedule_type=cosine"
  "b_2e-3|baseline|training.learning_rate=0.002,training.schedule_type=cosine"
  "b_5e-3|baseline|training.learning_rate=0.005,training.schedule_type=cosine"
  "tg_2e-3|tg_lora|tg_lora.lr_initial=0.002"
  "tg_5e-3|tg_lora|tg_lora.lr_initial=0.005"
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for entry in "${EXPERIMENTS[@]}"; do
  CURRENT=$((CURRENT + 1))
  IFS='|' read -r name exp_type overrides <<< "$entry"

  run_dir="${OUTPUT_BASE}/${name}"
  mkdir -p "${run_dir}"

  if [ "$exp_type" = "baseline" ]; then
    src_config="${BASELINE_CONFIG}"
  else
    src_config="${TG_LORA_CONFIG}"
  fi

  echo "[${CURRENT}/${TOTAL}] ${name} (${exp_type}, overrides=${overrides})"
  cp "${src_config}" "${run_dir}/config.yaml"

  ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
import ast
cfg = OmegaConf.load('${run_dir}/config.yaml')
cfg.data.max_seq_len = ${MAX_SEQ_LEN}
cfg.eval.quick_eval_examples = ${QUICK_EVAL_EXAMPLES}
cfg.eval.baseline_pretrain_reference_eval_enabled = '${BASELINE_PRETRAIN_REFERENCE_EVAL_ENABLED}'.lower() == 'true'
cfg.logging.mlflow.enabled = False
cfg.logging.run_dir = '${run_dir}'
cfg.model.device_map = None
cfg.model.device = None
for pair in '${overrides}'.split(','):
    k, v = pair.split('=', 1)
    k, v = k.strip(), v.strip()
    try:
        parsed = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        parsed = v
    OmegaConf.update(cfg, k, parsed)
if '${exp_type}' == 'baseline':
    cfg.training.max_steps = max(1, ${ACTUAL_BUDGET_PASSES} // cfg.training.grad_accumulation)
    cfg.training.save_every_steps = 9999
    cfg.eval.full_eval_every_steps = cfg.training.max_steps
else:
    cfg.training.max_cycles = max(1, ${ACTUAL_BUDGET_PASSES} // (cfg.tg_lora.K_initial * cfg.training.grad_accumulation))
    cfg.eval.full_eval_every_cycles = 9999
    cfg.logging.save_every_cycles = 9999
    # Ensure lr_max >= lr_initial for high-lr experiments
    if hasattr(cfg.tg_lora, 'lr_initial') and hasattr(cfg.tg_lora, 'lr_max'):
        if cfg.tg_lora.lr_initial > cfg.tg_lora.lr_max:
            cfg.tg_lora.lr_max = cfg.tg_lora.lr_initial * 2
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

  # Run and capture exit code (may fail with divergence)
  set +e
  env "${RUN_ENV[@]}" "PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF_VALUE}" ${VENV_PYTHON} -m src.training.$([ "$exp_type" = "baseline" ] && echo "train_baseline_qlora" || echo "train_tg_lora") --config "${run_dir}/config.yaml" 2>&1 | tail -5
  exit_code=$?
  set -e

  if [ ${exit_code} -ne 0 ]; then
    echo "  → DIVERGED (exit code ${exit_code})"
    echo "{\"type\": \"run_footer\", \"diverged\": true, \"exit_code\": ${exit_code}}" >> "${run_dir}/run_metrics.jsonl"
  else
    echo "  → Completed"
  fi
  echo ""
done

# --- Summary ---
echo "=============================================="
echo "  Results Summary"
echo "=============================================="

OUTPUT_BASE="${OUTPUT_BASE}" ${VENV_PYTHON} - <<'PY'
import json
import os
from pathlib import Path

results = []
base = Path(os.environ["OUTPUT_BASE"])
for d in sorted(base.iterdir()):
  if not d.is_dir():
    continue
  metrics = d / "run_metrics.jsonl"
  name = d.name
  if not metrics.exists():
    results.append((name, "NO_METRICS", "", "", ""))
    continue

  lines = [json.loads(l) for l in metrics.read_text().strip().split("\n")]
  footer = next((l for l in reversed(lines) if l.get("type") == "run_footer"), None)
  steps = [l for l in lines if l.get("type") == "step"]
  total_bp = steps[-1].get("total_backward_passes", "") if steps else ""

  if footer and footer.get("diverged"):
    results.append((name, "DIVERGED", "", "", str(total_bp)))
    continue

  best = footer.get("best_valid_loss") if footer else None
  final = footer.get("final_train_loss") if footer else None
  wall = footer.get("total_wall_seconds", 0) / 60 if footer else 0

  if best is not None:
    results.append((name, f"{best:.4f}", f"{final:.4f}" if final else "", f"{wall:.1f}min", str(total_bp)))
  else:
    last_loss = steps[-1].get("loss_train", "N/A") if steps else "N/A"
    results.append((name, "NO_EVAL", str(last_loss), f"{wall:.1f}min", str(total_bp)))

print(f"{'Name':<15} {'Best Valid':>10} {'Train Loss':>10} {'Wall Time':>10} {'BP':>8}")
print("-" * 60)
for name, bv, tl, wt, bp in results:
  print(f"{name:<15} {bv:>10} {tl:>10} {wt:>10} {bp:>8}")
PY

echo ""
echo "Output: ${OUTPUT_BASE}"
