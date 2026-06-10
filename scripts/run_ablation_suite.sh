#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
TARGET_BP="${TARGET_BP:-3600}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-64}"
EVAL_POINTS="${EVAL_POINTS:-4}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/ablation_suite_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ENV=()

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline.yaml}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/9b_tg_lora_paper_poc.yaml}"
ADAPTIVE_CONFIG="${ADAPTIVE_CONFIG:-configs/9b_tg_lora_adaptive_k5.yaml}"
ADAPTIVE_NO_CONV_CONFIG="${ADAPTIVE_NO_CONV_CONFIG:-configs/9b_tg_lora_adaptive_k5_no_conv.yaml}"

echo "=============================================="
echo "  TG-LoRA Ablation Suite"
echo "  Target backward passes: ${TARGET_BP}"
echo "  Max seq len: ${MAX_SEQ_LEN}"
echo "  Quick eval examples: ${QUICK_EVAL_EXAMPLES}"
echo "  Output: ${OUTPUT_BASE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "=============================================="
echo ""
echo "NOTE: adaptive K runs use planned cycles from K_initial, so actual total backward"
echo "passes may drift. Always compare against reported total_backward_passes, not only"
echo "the requested target budget."
echo ""

_run_baseline() {
    local name="$1"
    local config_path="$2"
    local run_dir="${OUTPUT_BASE}/${name}"

    mkdir -p "${run_dir}"
    cp "${config_path}" "${run_dir}/config.yaml"

    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
steps = max(1, ${TARGET_BP} // cfg.training.grad_accumulation)
eval_every = max(1, steps // ${EVAL_POINTS})
cfg.training.max_steps = steps
cfg.training.save_every_steps = 9999
cfg.training.early_stopping_patience = None
cfg.data.max_seq_len = ${MAX_SEQ_LEN}
cfg.eval.quick_eval_examples = ${QUICK_EVAL_EXAMPLES}
cfg.eval.full_eval_every_steps = eval_every
cfg.logging.run_dir = '${run_dir}'
cfg.logging.mlflow.enabled = False
cfg.model.device_map = None
cfg.model.device = None
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

    echo "--- Running ${name} (baseline) ---"
    env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora --config "${run_dir}/config.yaml" 2>&1 | tail -5
    echo ""
}

_run_tg() {
    local name="$1"
    local config_path="$2"
    local run_dir="${OUTPUT_BASE}/${name}"

    mkdir -p "${run_dir}"
    cp "${config_path}" "${run_dir}/config.yaml"

    ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
planned_cycles = max(1, ${TARGET_BP} // (cfg.training.grad_accumulation * cfg.tg_lora.K_initial))
eval_every = max(1, planned_cycles // ${EVAL_POINTS})
cfg.training.max_cycles = planned_cycles
cfg.training.early_stopping_patience = None
cfg.data.max_seq_len = ${MAX_SEQ_LEN}
cfg.eval.quick_eval_examples = ${QUICK_EVAL_EXAMPLES}
cfg.eval.full_eval_every_cycles = eval_every
cfg.logging.save_every_cycles = 9999
cfg.logging.run_dir = '${run_dir}'
cfg.logging.mlflow.enabled = False
cfg.model.device_map = None
cfg.model.device = None
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

    echo "--- Running ${name} (tg-lora) ---"
    env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora --config "${run_dir}/config.yaml" 2>&1 | tail -5
    echo ""
}

_run_baseline "baseline_lr2e-4" "${BASELINE_CONFIG}"
_run_tg "tg_paper_poc" "${PAPER_CONFIG}"
_run_tg "tg_adaptive_k5" "${ADAPTIVE_CONFIG}"
_run_tg "tg_adaptive_k5_no_conv" "${ADAPTIVE_NO_CONV_CONFIG}"

echo "=============================================="
echo "  Summary"
echo "=============================================="

OUTPUT_BASE="${OUTPUT_BASE}" ${VENV_PYTHON} - <<'PY'
import orjson
import os
from pathlib import Path

base = Path(os.environ["OUTPUT_BASE"])
rows = []
for run_dir in sorted(base.iterdir()):
    if not run_dir.is_dir():
        continue
    path = run_dir / "run_metrics.jsonl"
    if not path.exists():
        rows.append((run_dir.name, "NO_METRICS", "", "", ""))
        continue

    records = [orjson.loads(line) for line in path.open("rb")]
    footer = next((r for r in reversed(records) if r.get("type") == "run_footer"), None)
    last_step = next((r for r in reversed(records) if r.get("type") == "step"), None)

    best = footer.get("best_valid_loss") if footer else None
    wall = footer.get("total_wall_seconds", 0.0) if footer else 0.0
    total_bp = last_step.get("total_backward_passes") if last_step else None
    summary = footer.get("tg_lora_summary") if footer else None
    accept = summary.get("acceptance_rate") if isinstance(summary, dict) else None

    rows.append(
        (
            run_dir.name,
            f"{best:.4f}" if isinstance(best, (float, int)) else "NO_FOOTER",
            str(total_bp) if total_bp is not None else "N/A",
            f"{accept:.2%}" if isinstance(accept, (float, int)) else "-",
            f"{wall/60:.1f}m" if wall else "-",
        )
    )

print(f"{'Name':<24} {'Best Valid':>12} {'Total BP':>10} {'Accept':>10} {'Wall':>10}")
print("-" * 72)
for row in rows:
    print(f"{row[0]:<24} {row[1]:>12} {row[2]:>10} {row[3]:>10} {row[4]:>10}")
PY

echo ""
echo "Artifacts written to ${OUTPUT_BASE}"