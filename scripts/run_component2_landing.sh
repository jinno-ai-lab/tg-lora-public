#!/usr/bin/env bash
# Run the Component 2 landing comparison:
#   baseline: suffix-only last25, full backward every optimizer step
#   proposal: zero-order alpha-line + intermittent full backward
#
# This script is intentionally configurable so it can run both a short smoke
# and the 3-seed paper-facing run with the same code path.
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
SEEDS="${SEEDS:-42 123 7}"
TARGET_BP="${TARGET_BP:-240}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-32}"
EVAL_POINTS="${EVAL_POINTS:-1}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/component2_landing_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline_suffix_only_last25.yaml}"
PROPOSAL_CONFIG="${PROPOSAL_CONFIG:-configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml}"
TRAIN_PATH="${TRAIN_PATH:-data/train.jsonl}"
VALID_QUICK_PATH="${VALID_QUICK_PATH:-data/valid_quick.jsonl}"
VALID_FULL_PATH="${VALID_FULL_PATH:-data/valid_full.jsonl}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
ALPHA_M_STEPS="${ALPHA_M_STEPS:-19}"
ALPHA_LR="${ALPHA_LR:-0.01}"
K_INITIAL="${K_INITIAL:-1}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
PREFIX_CACHE_DIR="${PREFIX_CACHE_DIR:-.cache/prefix_feature_cache_paper_poc}"
PREFIX_FORCE_REBUILD="${PREFIX_FORCE_REBUILD:-false}"
PREFIX_CACHE_SHARE_ACROSS_SEEDS="${PREFIX_CACHE_SHARE_ACROSS_SEEDS:-true}"
FUTURE_WORK_METRICS_ENABLED="${FUTURE_WORK_METRICS_ENABLED:-false}"
FUTURE_WORK_INTERNAL_METRICS_ENABLED="${FUTURE_WORK_INTERNAL_METRICS_ENABLED:-false}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-false}"

read -r -a SEED_ARRAY <<<"${SEEDS}"
RUN_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

for cfg in "${BASELINE_CONFIG}" "${PROPOSAL_CONFIG}"; do
    if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: missing config ${cfg}" >&2
        exit 1
    fi
done
for data_path in "${TRAIN_PATH}" "${VALID_QUICK_PATH}" "${VALID_FULL_PATH}"; do
    if [[ ! -f "${data_path}" ]]; then
        echo "ERROR: missing data file ${data_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_BASE}"

echo "=============================================="
echo "  Component 2 Landing"
echo "  Seeds: ${SEEDS}"
echo "  Target backward passes: ${TARGET_BP}"
echo "  Max seq len: ${MAX_SEQ_LEN}"
echo "  Eval points: ${EVAL_POINTS}"
echo "  Alpha steps M: ${ALPHA_M_STEPS}"
echo "  Output: ${OUTPUT_BASE}"
echo "  Baseline: ${BASELINE_CONFIG}"
echo "  Proposal: ${PROPOSAL_CONFIG}"
echo "  Train data: ${TRAIN_PATH}"
echo "  Valid quick data: ${VALID_QUICK_PATH}"
echo "  Valid full data: ${VALID_FULL_PATH}"
echo "  Prefix cache dir: ${PREFIX_CACHE_DIR}"
echo "  Prefix cache share across seeds: ${PREFIX_CACHE_SHARE_ACROSS_SEEDS}"
echo "  Dry run: ${DRY_RUN}"
echo "=============================================="

_patch_baseline_config() {
    local path="$1"
    local seed="$2"
    local run_dir="$3"
    "${VENV_PYTHON}" - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${path}")
cfg.experiment.seed = int("${seed}")
cfg.experiment.name = f"{cfg.experiment.name}_component2_baseline_seed_${seed}"
cfg.data.train_path = "${TRAIN_PATH}"
cfg.data.valid_quick_path = "${VALID_QUICK_PATH}"
cfg.data.valid_full_path = "${VALID_FULL_PATH}"
cfg.data.max_seq_len = int("${MAX_SEQ_LEN}")
cfg.training.grad_accumulation = int("${GRAD_ACCUMULATION}")
steps = max(1, int("${TARGET_BP}") // int(cfg.training.grad_accumulation))
cfg.training.max_steps = steps
cfg.training.save_every_steps = 999999
cfg.training.early_stopping_patience = None
cfg.training.save_trajectory_delta_artifacts = False
cfg.eval.quick_eval_examples = int("${QUICK_EVAL_EXAMPLES}")
cfg.eval.full_eval_every_steps = max(1, steps // int("${EVAL_POINTS}"))
cfg.logging.run_dir = "${run_dir}"
if "mlflow" not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = "${MLFLOW_ENABLED}".lower() == "true"
cfg.model.device_map = None
cfg.model.device = None
OmegaConf.save(cfg, "${path}")
PY
}

_patch_proposal_config() {
    local path="$1"
    local seed="$2"
    local run_dir="$3"
    local prefix_cache_dir="$4"
    "${VENV_PYTHON}" - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${path}")
cfg.experiment.seed = int("${seed}")
cfg.experiment.name = f"{cfg.experiment.name}_component2_alpha_line_seed_${seed}"
cfg.data.train_path = "${TRAIN_PATH}"
cfg.data.valid_quick_path = "${VALID_QUICK_PATH}"
cfg.data.valid_full_path = "${VALID_FULL_PATH}"
cfg.data.max_seq_len = int("${MAX_SEQ_LEN}")
cfg.training.grad_accumulation = int("${GRAD_ACCUMULATION}")
cfg.training.optimizer_lifecycle = "recreate_per_cycle"
cfg.training.save_trajectory_delta_artifacts = False
cfg.training.early_stopping_patience = None
cfg.training.prefix_feature_cache_dir = "${prefix_cache_dir}"
cfg.training.prefix_feature_cache_force_rebuild = "${PREFIX_FORCE_REBUILD}".lower() == "true"
cfg.training.prefix_feature_cache_share_across_seeds = "${PREFIX_CACHE_SHARE_ACROSS_SEEDS}".lower() == "true"
cfg.training.prefix_feature_cache_train = True
cfg.training.prefix_feature_cache_valid_quick = True
cfg.training.prefix_feature_cache_valid_full = True
cycles = max(1, int("${TARGET_BP}") // (int(cfg.training.grad_accumulation) * int("${K_INITIAL}")))
cfg.training.max_cycles = cycles
cfg.logging.run_dir = "${run_dir}"
cfg.logging.save_every_cycles = 999999
if "mlflow" not in cfg.logging:
    cfg.logging.mlflow = {}
cfg.logging.mlflow.enabled = "${MLFLOW_ENABLED}".lower() == "true"
cfg.eval.quick_eval_examples = int("${QUICK_EVAL_EXAMPLES}")
cfg.eval.full_eval_every_cycles = max(1, cycles // int("${EVAL_POINTS}"))
cfg.tg_lora.K_initial = int("${K_INITIAL}")
cfg.tg_lora.K_candidates = [int("${K_INITIAL}")]
cfg.tg_lora.N_initial = 1
cfg.tg_lora.N_candidates = [1]
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
cfg.alpha_line = {
    "alpha_line_enabled": True,
    "alpha_line_order": 0,
    "b_logical": 32,
    "b_heavy": 4,
    "b_light": 16,
    "m_alpha_steps": int("${ALPHA_M_STEPS}"),
    "alpha_init": 0.0,
    "alpha_lr": float("${ALPHA_LR}"),
    "v_update_every": 1,
    "alpha_line_max_consecutive_reject": 3,
    "alpha_line_finite_diff_eps": 0.001,
    "future_work_metrics_enabled": "${FUTURE_WORK_METRICS_ENABLED}".lower() == "true",
    "future_work_internal_metrics_enabled": "${FUTURE_WORK_INTERNAL_METRICS_ENABLED}".lower() == "true",
}
cfg.model.device_map = None
cfg.model.device = None
OmegaConf.save(cfg, "${path}")
PY
}

BASELINE_RUNS=()
PROPOSAL_RUNS=()

for seed in "${SEED_ARRAY[@]}"; do
    seed_root="${OUTPUT_BASE}/seed_${seed}"
    config_dir="${seed_root}/configs"
    mkdir -p "${config_dir}"

    baseline_config="${config_dir}/baseline.yaml"
    proposal_config="${config_dir}/proposal.yaml"
    baseline_run="${seed_root}/baseline"
    proposal_run="${seed_root}/proposal"
    proposal_cache_dir="${PREFIX_CACHE_DIR//\{seed\}/${seed}}"

    cp "${BASELINE_CONFIG}" "${baseline_config}"
    cp "${PROPOSAL_CONFIG}" "${proposal_config}"
    _patch_baseline_config "${baseline_config}" "${seed}" "${baseline_run}"
    _patch_proposal_config "${proposal_config}" "${seed}" "${proposal_run}" "${proposal_cache_dir}"

    BASELINE_RUNS+=("${baseline_run}")
    PROPOSAL_RUNS+=("${proposal_run}")

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY RUN] seed=${seed}: configs written under ${config_dir}"
        continue
    fi

    echo ""
    echo "--- Seed ${seed}: baseline ---"
    env "${RUN_ENV[@]}" "${VENV_PYTHON}" -m src.training.train_baseline_qlora \
        --config "${baseline_config}"

    echo ""
    echo "--- Seed ${seed}: zero-order alpha-line ---"
    env "${RUN_ENV[@]}" "${VENV_PYTHON}" -m src.training.train_tg_lora \
        --config "${proposal_config}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY RUN complete: ${OUTPUT_BASE}"
    exit 0
fi

"${VENV_PYTHON}" scripts/summarize_component2_landing.py \
    --baseline-runs "${BASELINE_RUNS[@]}" \
    --proposal-runs "${PROPOSAL_RUNS[@]}" \
    --output-dir "${OUTPUT_BASE}/summary"

echo "Component 2 landing summary: ${OUTPUT_BASE}/summary/component2_landing_summary.md"
