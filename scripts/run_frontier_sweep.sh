#!/usr/bin/env bash
set -euo pipefail

# Stage 3: Memory Frontier Sweep
# Runs baseline and TG as separate arms so that baseline OOM does not prevent TG.

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
TARGET_BP="${TARGET_BP:-240}"
SEEDS="${SEEDS:-42 43 44}"
SEQS="${SEQS:-1536 2048 3072}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-32}"
EVAL_POINTS="${EVAL_POINTS:-3}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/frontier_sweep_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline_suffix_only_last25.yaml}"
TG_CONFIG="${TG_CONFIG:-configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml}"
CACHE_BASE="${CACHE_BASE:-.cache/prefix_feature_cache_frontier}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    RUN_ENV=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

RUN_ARGS=()

echo "=============================================="
echo "  TG-LoRA Frontier Sweep (arm-separated)"
echo "  Seq lens: ${SEQS}"
echo "  Seeds: ${SEEDS}"
echo "  Target backward passes: ${TARGET_BP}"
echo "  Output: ${OUTPUT_BASE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "=============================================="

mkdir -p "${OUTPUT_BASE}"

run_arm() {
    local name="$1"
    local config="$2"
    local arm_dir="$3"
    local log_path="${arm_dir}/${name}.log"

    (
        set +e
        ${VENV_PYTHON} -m "src.training.train_${name}" \
            --config "${config}" \
            --override "experiment.seed=${seed}" \
            --override "data.max_seq_len=${seq_len}" \
            --override "training.max_steps=${arm_steps}" \
            --override "eval.quick_eval_examples=${QUICK_EVAL_EXAMPLES}" \
            --override "logging.run_dir=${arm_dir}" \
            > "${log_path}" 2>&1
        echo $? > "${arm_dir}/${name}_exit_code"
    )
}

for seq_len in ${SEQS}; do
    run_dir="${OUTPUT_BASE}/slen_${seq_len}"
    mkdir -p "${run_dir}"

    echo ""
    echo "--- Frontier sweep: MAX_SEQ_LEN=${seq_len} ---"

    baseline_status="not_run"
    tg_status="not_run"

    for seed in ${SEEDS}; do
        seed_dir="${run_dir}/seed_${seed}"
        bl_dir="${seed_dir}/baseline"
        tg_dir="${seed_dir}/tg_lora"
        mkdir -p "${bl_dir}" "${tg_dir}"

        # --- Baseline arm ---
        arm_steps=$((TARGET_BP))
        echo "  seed=${seed} baseline (steps=${arm_steps})..."
        (
            set +e
            env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_baseline_qlora \
                --config "${BASELINE_CONFIG}" \
                --override "experiment.seed=${seed}" \
                --override "experiment.paper_experiment_id=frontier_sweep" \
                --override "data.max_seq_len=${seq_len}" \
                --override "training.max_steps=${arm_steps}" \
                --override "eval.quick_eval_examples=${QUICK_EVAL_EXAMPLES}" \
                --override "model.device_map=null" \
                --override "model.device=null" \
                --override "logging.run_dir=${bl_dir}" \
                > "${bl_dir}/train.log" 2>&1
            echo $? > "${bl_dir}/exit_code"
        ) || true

        bl_exit=1
        if [[ -f "${bl_dir}/exit_code" ]]; then
            bl_exit=$(cat "${bl_dir}/exit_code")
        fi
        if [[ ${bl_exit} -ne 0 ]]; then
            echo "  seed=${seed} baseline FAILED (exit=${bl_exit})"
            if grep -i -q -E "(CUDA|out.of.memory)" "${bl_dir}/train.log" 2>/dev/null; then
                echo "    -> OOM detected"
            fi
        else
            echo "  seed=${seed} baseline OK"
        fi

        # --- TG arm (always runs regardless of baseline outcome) ---
        tg_cycles=$((TARGET_BP / 3))  # default K=3
        if [[ ${tg_cycles} -lt 1 ]]; then tg_cycles=1; fi
        echo "  seed=${seed} tg_lora (cycles=${tg_cycles})..."
        (
            set +e
            env "${RUN_ENV[@]}" ${VENV_PYTHON} -m src.training.train_tg_lora \
                --config "${TG_CONFIG}" \
                --override "experiment.seed=${seed}" \
                --override "experiment.paper_experiment_id=frontier_sweep" \
                --override "data.max_seq_len=${seq_len}" \
                --override "training.max_cycles=${tg_cycles}" \
                --override "eval.quick_eval_examples=${QUICK_EVAL_EXAMPLES}" \
                --override "model.device_map=null" \
                --override "model.device=null" \
                --override "logging.run_dir=${tg_dir}" \
                > "${tg_dir}/train.log" 2>&1
            echo $? > "${tg_dir}/exit_code"
        ) || true

        tg_exit=1
        if [[ -f "${tg_dir}/exit_code" ]]; then
            tg_exit=$(cat "${tg_dir}/exit_code")
        fi
        if [[ ${tg_exit} -ne 0 ]]; then
            echo "  seed=${seed} tg_lora FAILED (exit=${tg_exit})"
        else
            echo "  seed=${seed} tg_lora OK"
        fi
    done

    # Write structured metadata
    "${VENV_PYTHON}" -c "
import json, glob
from pathlib import Path

base = Path('${run_dir}')
seeds = []
for d in sorted(base.glob('seed_*')):
    bl_exit_f = d / 'baseline' / 'exit_code'
    tg_exit_f = d / 'tg_lora' / 'exit_code'
    bl_exit = int(bl_exit_f.read_text().strip()) if bl_exit_f.exists() else 1
    tg_exit = int(tg_exit_f.read_text().strip()) if tg_exit_f.exists() else 1
    seeds.append({
        'seed': int(d.name.replace('seed_', '')),
        'baseline_exit': bl_exit,
        'tg_exit': tg_exit,
        'baseline_oom': bl_exit != 0 and 'out of memory' in (d / 'baseline' / 'train.log').read_text(errors='ignore').lower() if (d / 'baseline' / 'train.log').exists() else False,
        'tg_oom': tg_exit != 0 and 'out of memory' in (d / 'tg_lora' / 'train.log').read_text(errors='ignore').lower() if (d / 'tg_lora' / 'train.log').exists() else False,
    })

metadata = {
    'seq_len': ${seq_len},
    'seeds': seeds,
    'baseline_all_passed': all(s['baseline_exit'] == 0 for s in seeds),
    'tg_all_passed': all(s['tg_exit'] == 0 for s in seeds),
    'frontier_separation': any(s['baseline_exit'] != 0 and s['tg_exit'] == 0 for s in seeds),
}
metadata['summary_exists'] = (base / 'aggregate_summary.json').exists()

Path('${run_dir}/run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
"

    RUN_ARGS+=("${seq_len}:${run_dir}")

    echo "  Done"
done

echo ""
echo "--- Generating frontier_report.json ---"

"${VENV_PYTHON}" scripts/frontier_report.py \
    --runs "${RUN_ARGS[@]}" \
    --output "${OUTPUT_BASE}/frontier_report.json"

echo ""
echo "Frontier sweep complete. Artifacts in ${OUTPUT_BASE}"
