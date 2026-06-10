#!/usr/bin/env bash
# Component 2 runtime ablation:
#   A: baseline persistent Adam, fixed lr, no extrapolation
#   B: TG-LoRA fixed N, persistent Adam, fixed lr
#   C: TG-LoRA cosine-driven N, persistent Adam, fixed lr
#
# The summary reports reduction_rate, wall-clock, selected-N distribution,
# rollback frequency, validation-forward count, and runtime consistency.
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
TARGET_BP="${TARGET_BP:-240}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-32}"
EVAL_POINTS="${EVAL_POINTS:-3}"
SEEDS="${SEEDS:-42 43 44}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/cosine_n_ablation_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline_suffix_only_last25.yaml}"
FIXED_CONFIG="${FIXED_CONFIG:-configs/9b_tg_lora_fixed_n_persistent.yaml}"
COSINE_CONFIG="${COSINE_CONFIG:-configs/9b_tg_lora_cosine_n_persistent.yaml}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-false}"
CONFIDENT_SKIP_COS="${CONFIDENT_SKIP_COS:-0.0}"
CONFIDENT_SKIP_MIN_CYCLES="${CONFIDENT_SKIP_MIN_CYCLES:-10}"
ACCEPT_EVAL_EXAMPLES="${ACCEPT_EVAL_EXAMPLES:-}"
VALIDATION_SKIP_ENABLED="${VALIDATION_SKIP_ENABLED:-}"
VALIDATION_SKIP_HIGH_COS="${VALIDATION_SKIP_HIGH_COS:-}"
VALIDATION_SKIP_MID_COS="${VALIDATION_SKIP_MID_COS:-}"
VALIDATION_SKIP_MID_EVAL_EVERY="${VALIDATION_SKIP_MID_EVAL_EVERY:-}"
VALIDATION_SKIP_MIN_CYCLES="${VALIDATION_SKIP_MIN_CYCLES:-}"
VALIDATION_SKIP_FORCE_EVAL_N="${VALIDATION_SKIP_FORCE_EVAL_N:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-false}"

read -r -a SEED_ARRAY <<<"${SEEDS}"
RUN_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

echo "=============================================="
echo "  Cosine-N Runtime Ablation"
echo "  Seeds: ${SEEDS}"
echo "  Target backward passes: ${TARGET_BP}"
echo "  Max seq len: ${MAX_SEQ_LEN}"
echo "  Output: ${OUTPUT_BASE}"
echo "  Baseline: ${BASELINE_CONFIG}"
echo "  Fixed-N:  ${FIXED_CONFIG}"
echo "  Cosine-N: ${COSINE_CONFIG}"
echo "  Save trajectory artifacts: ${SAVE_TRAJECTORY}"
echo "  Confident skip cos: ${CONFIDENT_SKIP_COS}"
if [[ -n "${ACCEPT_EVAL_EXAMPLES}" ]]; then
    echo "  Accept eval examples: ${ACCEPT_EVAL_EXAMPLES}"
fi
if [[ -n "${VALIDATION_SKIP_ENABLED}" ]]; then
    echo "  Validation skip enabled: ${VALIDATION_SKIP_ENABLED}"
fi
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  Mode: DRY RUN"
fi
echo "=============================================="

for cfg in "${BASELINE_CONFIG}" "${FIXED_CONFIG}" "${COSINE_CONFIG}"; do
    if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: missing config ${cfg}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_BASE}"

_patch_config() {
    local mode="$1"
    local path="$2"
    local seed="$3"
    local run_dir="$4"
    ${VENV_PYTHON} - <<PY
from omegaconf import OmegaConf

mode = "${mode}"
path = "${path}"
cfg = OmegaConf.load(path)
cfg.experiment.seed = int("${seed}")
cfg.experiment.name = f"{cfg.experiment.name}_{mode}_seed_${seed}"
cfg.data.max_seq_len = int("${MAX_SEQ_LEN}")
cfg.eval.quick_eval_examples = int("${QUICK_EVAL_EXAMPLES}")
cfg.logging.run_dir = "${run_dir}"
if "mlflow" not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = "${MLFLOW_ENABLED}".lower() == "true"

if mode == "baseline":
    steps = max(1, int("${TARGET_BP}") // int(cfg.training.grad_accumulation))
    cfg.training.max_steps = steps
    cfg.training.optimizer_lifecycle = "persistent"
    cfg.training.save_every_steps = 999999
    cfg.training.early_stopping_patience = None
    cfg.eval.full_eval_every_steps = max(1, steps // int("${EVAL_POINTS}"))
else:
    K = int(cfg.tg_lora.K_initial)
    cycles = max(1, int("${TARGET_BP}") // (int(cfg.training.grad_accumulation) * K))
    cfg.training.max_cycles = cycles
    cfg.training.save_every_steps = 999999
    cfg.logging.save_every_cycles = 999999
    cfg.training.early_stopping_patience = None
    cfg.training.save_trajectory_delta_artifacts = "${SAVE_TRAJECTORY}".lower() == "true"
    cfg.eval.full_eval_every_cycles = max(1, cycles // int("${EVAL_POINTS}"))
    cfg.tg_lora.enable_random_walk = False
    cfg.tg_lora.enable_convergence_adaptation = False
    cfg.tg_lora.k_explore_prob = 0.0
    cfg.tg_lora.n_explore_prob = 0.0
    cfg.tg_lora.beta_explore_prob = 0.0
    cfg.tg_lora.strategy_explore_prob = 0.0
    cfg.tg_lora.lr_explore_prob = 0.0
    cfg.tg_lora.active_layer_strategy = "last_25_percent"
    cfg.tg_lora.random_middle_layers = 0
    cfg.tg_lora.force_top_layers_only = True
    cfg.tg_lora.confident_skip_cos = float("${CONFIDENT_SKIP_COS}")
    cfg.tg_lora.confident_skip_min_cycles = int("${CONFIDENT_SKIP_MIN_CYCLES}")
    if "${ACCEPT_EVAL_EXAMPLES}":
        cfg.eval.accept_eval_examples = int("${ACCEPT_EVAL_EXAMPLES}")
    if "${VALIDATION_SKIP_ENABLED}":
        cfg.tg_lora.validation_skip_enabled = "${VALIDATION_SKIP_ENABLED}".lower() == "true"
    if "${VALIDATION_SKIP_HIGH_COS}":
        cfg.tg_lora.validation_skip_high_cos = float("${VALIDATION_SKIP_HIGH_COS}")
    if "${VALIDATION_SKIP_MID_COS}":
        cfg.tg_lora.validation_skip_mid_cos = float("${VALIDATION_SKIP_MID_COS}")
    if "${VALIDATION_SKIP_MID_EVAL_EVERY}":
        cfg.tg_lora.validation_skip_mid_eval_every = int("${VALIDATION_SKIP_MID_EVAL_EVERY}")
    if "${VALIDATION_SKIP_MIN_CYCLES}":
        cfg.tg_lora.validation_skip_min_cycles = int("${VALIDATION_SKIP_MIN_CYCLES}")
    if "${VALIDATION_SKIP_FORCE_EVAL_N}":
        cfg.tg_lora.validation_skip_force_eval_N = int("${VALIDATION_SKIP_FORCE_EVAL_N}")

OmegaConf.save(cfg, path)
PY
}

for seed in "${SEED_ARRAY[@]}"; do
    seed_root="${OUTPUT_BASE}/seed_${seed}"
    config_dir="${seed_root}/configs"
    mkdir -p "${config_dir}"

    cp "${BASELINE_CONFIG}" "${config_dir}/baseline.yaml"
    cp "${FIXED_CONFIG}" "${config_dir}/fixed_n.yaml"
    cp "${COSINE_CONFIG}" "${config_dir}/cosine_n.yaml"

    _patch_config "baseline" "${config_dir}/baseline.yaml" "${seed}" "${seed_root}/baseline"
    _patch_config "fixed_n" "${config_dir}/fixed_n.yaml" "${seed}" "${seed_root}/fixed_n"
    _patch_config "cosine_n" "${config_dir}/cosine_n.yaml" "${seed}" "${seed_root}/cosine_n"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY RUN] seed=${seed}: configs written under ${config_dir}"
        continue
    fi

    echo ""
    echo "--- Seed ${seed}: baseline ---"
    env "${RUN_ENV[@]}" "${VENV_PYTHON}" -m src.training.train_baseline_qlora \
        --config "${config_dir}/baseline.yaml"

    echo ""
    echo "--- Seed ${seed}: fixed-N TG-LoRA ---"
    env "${RUN_ENV[@]}" "${VENV_PYTHON}" -m src.training.train_tg_lora \
        --config "${config_dir}/fixed_n.yaml"

    echo ""
    echo "--- Seed ${seed}: cosine-N TG-LoRA ---"
    env "${RUN_ENV[@]}" "${VENV_PYTHON}" -m src.training.train_tg_lora \
        --config "${config_dir}/cosine_n.yaml"
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY RUN complete: ${OUTPUT_BASE}"
    exit 0
fi

"${VENV_PYTHON}" scripts/summarize_cosine_n_ablation.py "${OUTPUT_BASE}"
