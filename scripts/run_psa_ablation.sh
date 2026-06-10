#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PSA Full Ablation — GOAL §4 step 3
#
# PURPOSE: Compare PSA against plain LoRA and LAWA on the same backward-pass
#   budget, then sweep PSA hyperparameters to determine optimal configuration.
#
# AUTO-RESUME: Re-running this script skips completed runs and resumes
#   interrupted ones from the latest checkpoint. Safe to re-run after a crash.
#
# RUNS (in order):
#   1. Plain LoRA baseline (no PSA, no LAWA)
#   2. LAWA-only (weight averaging, no PSA)
#   3. PSA default (γ=0.5, regime reset ON)
#   4. PSA γ sweep:   {0.0, 0.3, 0.5, 1.0} × regime_reset {on, off}
#   5. PSA history sweep:  {3, 6, 10} at best γ
#   6. PSA interval sweep: {1, 3, 5}  at best γ
#
# CONTROLLED: All runs share the same BUDGET_PASSES, model, LoRA config
#   (r=16, alpha=32, dropout=0.0, trainable_lora_scope=last_25_percent),
#   learning rate, and evaluation protocol.
#
# FAIR COMPARISON (GOAL §3.3): The baseline uses 9b_baseline_suffix_only_last25
#   which is parity-matched to the PSA config on dropout (0.0) and
#   trainable_lora_scope (last_25_percent). Using 9b_baseline.yaml instead
#   would confound the comparison (dropout 0.05, scope=all).
#
# USAGE:
#   BUDGET_PASSES=1500 ./scripts/run_psa_ablation.sh
#   BUDGET_PASSES=3600 SEED=43 ./scripts/run_psa_ablation.sh
# ---------------------------------------------------------------------------
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
BUDGET_PASSES="${BUDGET_PASSES:-1500}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/psa_ablation_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline_suffix_only_last25.yaml}"
PSA_CONFIG="${PSA_CONFIG:-configs/9b_tg_lora_psa.yaml}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL="${QUICK_EVAL:-32}"
EVAL_POINTS="${EVAL_POINTS:-4}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

RUN_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

mkdir -p "${OUTPUT_BASE}"
echo "=============================================="
echo "  PSA Full Ablation (GOAL §4 step 3)"
echo "  Budget: ${BUDGET_PASSES} backward passes"
echo "  Seed: ${SEED}"
echo "  Output: ${OUTPUT_BASE}"
echo "=============================================="

# --- Resume helpers ---

_is_completed() {
  local metrics="${1}/run_metrics.jsonl"
  [[ -f "${metrics}" ]] && grep -q '"type":"run_footer"' "${metrics}"
}

# Find latest checkpoint with training_state.pt (baseline: checkpoint-*/training_state.pt)
_find_baseline_resume() {
  local run_dir="$1"
  local latest=""
  local latest_step=0
  for ckpt in "${run_dir}"/checkpoint-*/training_state.pt; do
    [[ -f "${ckpt}" ]] || continue
    local step
    step=$(basename "$(dirname "${ckpt}")" | grep -o '[0-9]*$')
    if [[ "${step}" -gt "${latest_step}" ]]; then
      latest_step="${step}"
      latest="${ckpt}"
    fi
  done
  # Also check oom_checkpoint
  if [[ -f "${run_dir}/oom_checkpoint/training_state.pt" ]]; then
    echo "${run_dir}/oom_checkpoint/training_state.pt"
    return
  fi
  if [[ -n "${latest}" ]]; then
    echo "${latest}"
  fi
}

# Find latest checkpoint with training_state.pt (tg-lora: checkpoint-cycle-*/training_state.pt)
_find_tglora_resume() {
  local run_dir="$1"
  local latest=""
  local latest_cycle=0
  for ckpt in "${run_dir}"/checkpoint-cycle-*/training_state.pt; do
    [[ -f "${ckpt}" ]] || continue
    local cycle
    cycle=$(basename "$(dirname "${ckpt}")" | grep -o '[0-9]*$')
    if [[ "${cycle}" -gt "${latest_cycle}" ]]; then
      latest_cycle="${cycle}"
      latest="${ckpt}"
    fi
  done
  # Also check oom_checkpoint
  if [[ -f "${run_dir}/oom_checkpoint/training_state.pt" ]]; then
    echo "${run_dir}/oom_checkpoint/training_state.pt"
    return
  fi
  if [[ -n "${latest}" ]]; then
    echo "${latest}"
  fi
}

# --- Helper: compute training length from budget ---
# baseline uses max_steps = budget / grad_accum
# tg-lora uses max_cycles = budget / (grad_accum * K_initial)

_run_baseline() {
  local name="$1"
  local run_dir="${OUTPUT_BASE}/${name}"

  if _is_completed "${run_dir}"; then
    echo "--- Skipping ${name} (already completed) ---"
    return
  fi

  mkdir -p "${run_dir}"
  cp "${BASELINE_CONFIG}" "${run_dir}/config.yaml"

  ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
steps = max(1, ${BUDGET_PASSES} // cfg.training.grad_accumulation)
eval_every = 999999 if '${SKIP_EVAL}'.lower() == 'true' else max(1, steps // ${EVAL_POINTS})
cfg.experiment.name = '${name}'
cfg.experiment.seed = ${SEED}
cfg.training.max_steps = steps
cfg.training.save_every_steps = max(1, steps // 7)
cfg.training.early_stopping_patience = None
cfg.data.max_seq_len = ${MAX_SEQ_LEN}
cfg.eval.quick_eval_examples = ${QUICK_EVAL}
cfg.eval.full_eval_every_steps = eval_every
cfg.logging.run_dir = '${run_dir}'
if 'mlflow' not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = '${MLFLOW_ENABLED}'.lower() == 'true'
cfg.model.device_map = None
cfg.model.device = None
cfg.training.activation_regime_enabled = True
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

  local resume_arg=""
  local resume_pt
  resume_pt=$(_find_baseline_resume "${run_dir}")
  if [[ -n "${resume_pt}" ]]; then
    resume_arg="--resume ${resume_pt}"
    echo "--- Resuming ${name} from ${resume_pt} ---"
  else
    echo "--- Running ${name} (baseline) ---"
  fi
  env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora --config "${run_dir}/config.yaml" ${resume_arg} 2>&1 | tail -5
}

_run_psa() {
  local name="$1"
  local gamma="${2}"
  local regime_reset="${3}"
  local history="${4:-6}"
  local interval="${5:-3}"
  local enable_lawa="${6:-false}"
  local run_dir="${OUTPUT_BASE}/${name}"

  if _is_completed "${run_dir}"; then
    echo "--- Skipping ${name} (already completed) ---"
    return
  fi

  mkdir -p "${run_dir}"
  cp "${PSA_CONFIG}" "${run_dir}/config.yaml"

  ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${run_dir}/config.yaml')
K = cfg.tg_lora.K_initial
planned_cycles = max(1, ${BUDGET_PASSES} // (cfg.training.grad_accumulation * K))
eval_every = 999999 if '${SKIP_EVAL}'.lower() == 'true' else max(1, planned_cycles // ${EVAL_POINTS})
cfg.experiment.name = '${name}'
cfg.experiment.seed = ${SEED}
cfg.training.max_cycles = planned_cycles
cfg.training.early_stopping_patience = None
cfg.data.max_seq_len = ${MAX_SEQ_LEN}
cfg.tg_lora.enable_psa = ${gamma} > 0
cfg.tg_lora.psa_gain = ${gamma}
cfg.tg_lora.psa_regime_reset_enabled = ${regime_reset^}
cfg.tg_lora.psa_history_length = ${history}
cfg.tg_lora.psa_update_interval = ${interval}
cfg.tg_lora.psa_warmup_steps = 4
cfg.tg_lora.psa_l2_reg = 0.01
cfg.tg_lora.enable_lawa = ${enable_lawa^}
cfg.tg_lora.activation_regime_enabled = True
cfg.eval.quick_eval_examples = ${QUICK_EVAL}
cfg.eval.full_eval_every_cycles = eval_every
cfg.logging.save_every_cycles = max(1, planned_cycles // 6)
cfg.logging.run_dir = '${run_dir}'
if 'mlflow' not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = '${MLFLOW_ENABLED}'.lower() == 'true'
cfg.model.device_map = None
cfg.model.device = None
OmegaConf.save(cfg, '${run_dir}/config.yaml')
"

  local resume_arg=""
  local resume_pt
  resume_pt=$(_find_tglora_resume "${run_dir}")
  if [[ -n "${resume_pt}" ]]; then
    resume_arg="--resume ${resume_pt}"
    echo "--- Resuming ${name} from ${resume_pt} ---"
  else
    echo "--- Running ${name} ---"
  fi
  env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora --config "${run_dir}/config.yaml" ${resume_arg} 2>&1 | tail -5
}

# --- 1. Plain LoRA baseline ---
echo ""
_run_baseline "baseline_plain"

# --- 2. LAWA-only (PSA disabled, LAWA enabled) ---
echo ""
_run_psa "lawa_only" 0.0 true 6 3 true
# Note: LAWA uses _run_psa with gamma=0 and enable_lawa=true, which disables PSA and enables LAWA

# --- 3. PSA default ---
echo ""
_run_psa "psa_default" 0.5 true 6 3

# --- 4. PSA γ sweep ---
GAMMAS=(0.0 0.3 0.5 1.0)
RESETS=(true false)
for gamma in "${GAMMAS[@]}"; do
  for reset in "${RESETS[@]}"; do
    reset_tag="on"
    [[ "${reset}" == "false" ]] && reset_tag="off"
    # Skip default (already run) and LAWA overlap (gamma=0, lawa=true)
    if [[ "${gamma}" == "0.5" && "${reset}" == "true" ]]; then
      echo "--- Skipping gamma_0.5_reset_on (already run as psa_default) ---"
      continue
    fi
    echo ""
    _run_psa "gamma_${gamma}_reset_${reset_tag}" "${gamma}" "${reset}" 6 3
  done
done

# --- 5. PSA history sweep (γ=0.5, reset=on) ---
for history in 3 10; do
  echo ""
  _run_psa "history_${history}" 0.5 true "${history}" 3
done

# --- 6. PSA interval sweep (γ=0.5, reset=on, history=6) ---
for interval in 1 5; do
  echo ""
  _run_psa "interval_${interval}" 0.5 true 6 "${interval}"
done

# --- Generate comparison report ---
echo ""
echo "=============================================="
echo "  PSA Ablation Summary"
echo "=============================================="

REPORT_DIR="${OUTPUT_BASE}/report"
mkdir -p "${REPORT_DIR}"

export OUTPUT_BASE
${VENV_PYTHON} - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["OUTPUT_BASE"])

rows = []
for run_dir in sorted(base.iterdir()):
    if not run_dir.is_dir():
        continue
    metrics = run_dir / "run_metrics.jsonl"
    if not metrics.exists():
        continue

    records = []
    footer = {}
    with open(metrics) as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("type")
            if t == "step":
                records.append(obj)
            elif t == "run_footer":
                footer = obj

    if not footer:
        rows.append((run_dir.name, "NO_FOOTER", "", "", "", "", ""))
        continue

    best_vl = footer.get("best_valid_loss")
    total_bp = records[-1].get("total_backward_passes", 0) if records else 0
    wall_sec = footer.get("total_wall_seconds", 0)
    summary = footer.get("tg_lora_summary", {})
    accept = summary.get("acceptance_rate") if isinstance(summary, dict) else None
    final_train = footer.get("final_train_loss", records[-1].get("loss_train") if records else None)

    # PSA metrics
    psa_gain_mean = records[-1].get("psa_gain_mean") if records else None
    regime_transitions = records[-1].get("psa_regime_transitions") if records else None

    # Activation regime inventory (nested inside tg_lora_summary)
    act_stable = summary.get("activation_regime_stable_fraction") if isinstance(summary, dict) else None
    act_null_z = None
    act_null = summary.get("activation_regime_null_baseline") if isinstance(summary, dict) else None
    if isinstance(act_null, dict):
        act_null_z = act_null.get("stable_fraction_z")

    # LAWA
    lawa_snapshots = summary.get("lawa_snapshots_recorded") if isinstance(summary, dict) else None

    rows.append((
        run_dir.name,
        f"{best_vl:.4f}" if isinstance(best_vl, (float, int)) else "N/A",
        f"{final_train:.4f}" if isinstance(final_train, (float, int)) else "N/A",
        str(total_bp),
        f"{accept:.2%}" if isinstance(accept, (float, int)) else "-",
        f"{wall_sec/60:.1f}m" if wall_sec else "-",
        f"stable={act_stable:.2f}" if isinstance(act_stable, float) else "-",
    ))

print(f"{'Name':<28} {'Best Valid':>10} {'Final Trn':>10} {'Total BP':>10} {'Accept':>10} {'Wall':>8} {'Act Regime':>12}")
print("-" * 92)
for row in rows:
    print(f"{row[0]:<28} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>8} {row[6]:>12}")
PY

# --- Enhanced PSA sweep analysis (§3.3 decision table, per-layer-type diagnostics) ---
if [[ -f scripts/summarize_psa_sweep.py ]]; then
  echo ""
  echo "--- Enhanced PSA Ablation Analysis (GOAL §3.3) ---"
  ${VENV_PYTHON} scripts/summarize_psa_sweep.py --sweep-dir "${OUTPUT_BASE}" || true
fi

echo ""
echo "Artifacts written to ${OUTPUT_BASE}"
echo ""
echo "Next steps (GOAL §4.3):"
echo "  1. Compare 'baseline_plain' vs 'lawa_only' vs 'psa_default' — does PSA beat LAWA?"
echo "  2. Check γ sweep — which gain is optimal?"
echo "  3. Check history/interval sweeps — further tuning opportunity"
echo "  4. Verify activation_regime_stable_fraction — theoretical efficiency ceiling"
