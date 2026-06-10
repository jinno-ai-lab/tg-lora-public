#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
TARGET_BP="${TARGET_BP:-240}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
QUICK_EVAL_EXAMPLES="${QUICK_EVAL_EXAMPLES:-32}"
EVAL_POINTS="${EVAL_POINTS:-3}"
SEEDS="${SEEDS:-42 43 44}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/paper_memory_suite_$(date +%Y%m%d_%H%M%S)}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/9b_baseline_suffix_only_last25.yaml}"
TG_CONFIG="${TG_CONFIG:-configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml}"
CACHE_BASE="${CACHE_BASE:-.cache/prefix_feature_cache_paper_suite}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
DRY_RUN="${DRY_RUN:-false}"

read -r -a SEED_ARRAY <<<"${SEEDS}"

echo "=============================================="
echo "  TG-LoRA Paper Memory Suite"
echo "  Seeds: ${SEEDS}"
echo "  Target backward passes: ${TARGET_BP}"
echo "  Max seq len: ${MAX_SEQ_LEN}"
echo "  Quick eval examples: ${QUICK_EVAL_EXAMPLES}"
echo "  Output: ${OUTPUT_BASE}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}"
fi
if [[ "${DRY_RUN}" == "true" ]]; then
echo "  Mode: DRY RUN (validation only)"
fi
echo "=============================================="

# --- Pre-flight config validation ---
DRY_RUN_ERRORS=0

if [[ ! -f "${BASELINE_CONFIG}" ]]; then
    echo "ERROR: Baseline config not found: ${BASELINE_CONFIG}" >&2
    DRY_RUN_ERRORS=$((DRY_RUN_ERRORS + 1))
fi

if [[ ! -f "${TG_CONFIG}" ]]; then
    echo "ERROR: TG config not found: ${TG_CONFIG}" >&2
    DRY_RUN_ERRORS=$((DRY_RUN_ERRORS + 1))
fi

# Validate prefix_feature_cache_train in TG config
if [[ -f "${TG_CONFIG}" ]]; then
    if ! grep -q "prefix_feature_cache_train: *true" "${TG_CONFIG}"; then
        echo "WARNING: TG config does not have prefix_feature_cache_train: true — required for paper-memory suite" >&2
        if [[ "${DRY_RUN}" == "true" ]]; then
            DRY_RUN_ERRORS=$((DRY_RUN_ERRORS + 1))
        fi
    else
        echo "  [OK] prefix_feature_cache_train: true found in TG config"
    fi
fi

if [[ "${DRY_RUN}" == "true" && ${DRY_RUN_ERRORS} -gt 0 ]]; then
    echo "" >&2
    echo "DRY RUN FAILED: ${DRY_RUN_ERRORS} validation error(s) detected" >&2
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"

for seed in "${SEED_ARRAY[@]}"; do
    seed_root="${OUTPUT_BASE}/seed_${seed}"
    config_dir="${seed_root}/configs"
    cache_dir="${CACHE_BASE}/seed_${seed}"
    mkdir -p "${config_dir}"

    if [[ -f "${BASELINE_CONFIG}" ]]; then
        cp "${BASELINE_CONFIG}" "${config_dir}/baseline.yaml"
    fi
    if [[ -f "${TG_CONFIG}" ]]; then
        cp "${TG_CONFIG}" "${config_dir}/tg_cache.yaml"
    fi

    # Seed patching via OmegaConf
    if [[ -f "${config_dir}/baseline.yaml" && -f "${config_dir}/tg_cache.yaml" ]]; then
        ${VENV_PYTHON} -c "
from omegaconf import OmegaConf
for path, suffix in [
    ('${config_dir}/baseline.yaml', 'baseline_seed_${seed}'),
    ('${config_dir}/tg_cache.yaml', 'tg_cache_seed_${seed}'),
]:
    cfg = OmegaConf.load(path)
    cfg.experiment.seed = int('${seed}')
    cfg.experiment.name = f'{cfg.experiment.name}_{suffix}'
    OmegaConf.save(cfg, path)
"
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [OK] seed=${seed}: configs patched, output dir ready at ${seed_root}"
        continue
    fi

    echo ""
    echo "--- Running seed ${seed} cold/warm paper memory benchmark ---"
    ${VENV_PYTHON} scripts/benchmark_prefix_cache.py \
        --budget "${TARGET_BP}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --quick-eval-examples "${QUICK_EVAL_EXAMPLES}" \
        --eval-points "${EVAL_POINTS}" \
        --baseline-config "${config_dir}/baseline.yaml" \
        --tg-config "${config_dir}/tg_cache.yaml" \
        --cache-dir "${cache_dir}" \
        --output-base "${seed_root}/coldwarm" \
        $(if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then printf -- '--cuda-visible-devices %q ' "${CUDA_VISIBLE_DEVICES_VALUE}"; fi) \
        $(if [[ "${MLFLOW_ENABLED}" == "true" ]]; then echo --mlflow-enabled; fi)
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "  DRY RUN complete: all validations passed"
    echo "  Seeds validated: ${SEEDS}"
    echo "  Baseline config: ${BASELINE_CONFIG}"
    echo "  TG config: ${TG_CONFIG}"
    echo "  Output base: ${OUTPUT_BASE}"
    echo "=============================================="
    exit 0
fi

${VENV_PYTHON} - <<PY
import json
import statistics
from pathlib import Path

base = Path('${OUTPUT_BASE}')
rows = []
for summary_path in sorted(base.glob('seed_*/coldwarm/summary.json')):
    seed = summary_path.parents[1].name.replace('seed_', '')
    summary = json.loads(summary_path.read_text())
    cold = summary['cold']
    warm = summary['warm']
    rows.append(
        {
            'seed': int(seed),
            'cold_baseline_wall_seconds': cold['baseline']['wall_seconds'],
            'cold_tg_wall_seconds': cold['tg_lora']['wall_seconds'],
            'cold_tg_gpu_peak_mb': cold['tg_lora'].get('gpu_peak_mb'),
            'warm_baseline_wall_seconds': warm['baseline']['wall_seconds'],
            'warm_tg_wall_seconds': warm['tg_lora']['wall_seconds'],
            'warm_baseline_gpu_peak_mb': warm['baseline'].get('gpu_peak_mb'),
            'warm_tg_gpu_peak_mb': warm['tg_lora'].get('gpu_peak_mb'),
            'warm_baseline_best_valid_loss': warm['baseline']['best_valid_loss'],
            'warm_tg_best_valid_loss': warm['tg_lora']['best_valid_loss'],
            'warm_baseline_loss_red_per_wall_minute': warm['baseline']['loss_red_per_wall_minute'],
            'warm_tg_loss_red_per_wall_minute': warm['tg_lora']['loss_red_per_wall_minute'],
            'warm_tg_backward_passes': warm['tg_lora']['total_backward_passes'],
            'warm_baseline_backward_passes': warm['baseline']['total_backward_passes'],
            'warm_tg_extrapolation_steps': warm['tg_lora'].get('extrapolation_steps'),
            'warm_tg_accepted_extrapolations': warm['tg_lora'].get('accepted_extrapolations'),
            'warm_tg_acceptance_rate': warm['tg_lora'].get('acceptance_rate'),
            'tg_cache_build_seconds': cold['tg_lora'].get('prefix_feature_cache_total_build_seconds'),
            'tg_cache_load_seconds': warm['tg_lora'].get('prefix_feature_cache_total_load_seconds'),
            'warm_tg_runtime_offload_gpu_allocated_mb_before': warm['tg_lora'].get('prefix_feature_cache_runtime_offload_gpu_allocated_mb_before'),
            'warm_tg_runtime_offload_gpu_allocated_mb_after': warm['tg_lora'].get('prefix_feature_cache_runtime_offload_gpu_allocated_mb_after'),
            'warm_tg_runtime_offload_gpu_freed_mb': warm['tg_lora'].get('prefix_feature_cache_runtime_offload_gpu_freed_mb'),
            'tg_cache_warm_speedup_pct': summary['delta'].get('tg_wall_speedup_pct'),
            'tg_loss_red_per_wall_minute_delta_pct': summary['delta'].get('tg_loss_red_per_wall_minute_delta_pct'),
        }
    )

if not rows:
    raise SystemExit('No seed summaries were produced')

def series(key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return {'values': [], 'mean': None, 'stdev': None}
    return {
        'values': values,
        'mean': sum(values) / len(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
    }

aggregate = {
    'seeds': [row['seed'] for row in rows],
    'per_seed': rows,
    'aggregate': {
        'warm_baseline_wall_seconds': series('warm_baseline_wall_seconds'),
        'warm_tg_wall_seconds': series('warm_tg_wall_seconds'),
        'warm_baseline_gpu_peak_mb': series('warm_baseline_gpu_peak_mb'),
        'warm_tg_gpu_peak_mb': series('warm_tg_gpu_peak_mb'),
        'warm_baseline_best_valid_loss': series('warm_baseline_best_valid_loss'),
        'warm_tg_best_valid_loss': series('warm_tg_best_valid_loss'),
        'warm_baseline_loss_red_per_wall_minute': series('warm_baseline_loss_red_per_wall_minute'),
        'warm_tg_loss_red_per_wall_minute': series('warm_tg_loss_red_per_wall_minute'),
        'tg_cache_build_seconds': series('tg_cache_build_seconds'),
        'tg_cache_load_seconds': series('tg_cache_load_seconds'),
        'warm_tg_runtime_offload_gpu_allocated_mb_before': series('warm_tg_runtime_offload_gpu_allocated_mb_before'),
        'warm_tg_runtime_offload_gpu_allocated_mb_after': series('warm_tg_runtime_offload_gpu_allocated_mb_after'),
        'warm_tg_runtime_offload_gpu_freed_mb': series('warm_tg_runtime_offload_gpu_freed_mb'),
        'tg_cache_warm_speedup_pct': series('tg_cache_warm_speedup_pct'),
        'tg_loss_red_per_wall_minute_delta_pct': series('tg_loss_red_per_wall_minute_delta_pct'),
    },
}

json_path = base / 'aggregate_summary.json'
json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + '\n')

lines = []
lines.append('# Paper Memory Suite Summary')
lines.append('')
lines.append(f'- seeds: {aggregate["seeds"]}')
lines.append(f'- target backward passes: ${TARGET_BP}')
lines.append(f'- max seq len: ${MAX_SEQ_LEN}')
lines.append('')
lines.append('| Seed | Warm Baseline Wall (s) | Warm TG Wall (s) | Warm Baseline Peak MB | Warm TG Peak MB | Warm Baseline Loss | Warm TG Loss | TG Warm Speedup vs Cold |')
lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for row in rows:
    lines.append(
        f"| {row['seed']} | {row['warm_baseline_wall_seconds']:.1f} | {row['warm_tg_wall_seconds']:.1f} | "
        f"{row['warm_baseline_gpu_peak_mb']:.1f} | {row['warm_tg_gpu_peak_mb']:.1f} | "
        f"{row['warm_baseline_best_valid_loss']:.4f} | {row['warm_tg_best_valid_loss']:.4f} | "
        f"{row['tg_cache_warm_speedup_pct']:.2f}% |"
    )
lines.append('')
lines.append('## Aggregate Means')
lines.append('')
for key, value in aggregate['aggregate'].items():
    if value['mean'] is None:
        continue
    lines.append(f'- {key}: mean={value["mean"]:.4f}, stdev={value["stdev"]:.4f}, values={value["values"]}')

md_path = base / 'aggregate_summary.md'
md_path.write_text('\n'.join(lines) + '\n')

print(json.dumps(aggregate, indent=2, ensure_ascii=False))
print(f'Aggregate summary written to {json_path}')
print(f'Markdown summary written to {md_path}')
PY

echo ""
echo "Artifacts written to ${OUTPUT_BASE}"
