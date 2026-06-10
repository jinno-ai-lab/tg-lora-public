#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Accel Param Sweep — auto-wait wrapper
#
# Waits for GPU 0 to become free (< 500 MiB used), then runs the full
# accel param sweep via run_accel_sweep.sh.
#
# Usage:
#   nohup bash scripts/run_accel_sweep_auto.sh &
#
# Environment:
#   VENV_PYTHON  — Python binary (default: /home/jinno/tg-lora/.venv/bin/python)
#   SWEEP_DIR    — Working directory (default: repo root where this script lives)
# ---------------------------------------------------------------------------
set -euo pipefail

SWEEP_DIR="${SWEEP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SWEEP_DIR"

VENV_PYTHON="${VENV_PYTHON:-/home/jinno/tg-lora/.venv/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"
GPU_THRESHOLD_MB="${GPU_THRESHOLD_MB:-500}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
LOG_FILE="${SWEEP_DIR}/reports/accel_sweep_auto.log"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== Accel Sweep Auto-Start ==="
log "SWEEP_DIR: ${SWEEP_DIR}"
log "VENV_PYTHON: ${VENV_PYTHON}"
log "GPU_INDEX: ${GPU_INDEX}"
log "Threshold: ${GPU_THRESHOLD_MB} MiB"

# Wait for GPU to become free
while true; do
    used_mb=$(nvidia-smi -i "${GPU_INDEX}" \
        --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "99999")
    if [[ "${used_mb}" -lt "${GPU_THRESHOLD_MB}" ]]; then
        log "GPU ${GPU_INDEX} is free (${used_mb} MiB used). Starting sweep."
        break
    fi
    log "GPU ${GPU_INDEX} still busy (${used_mb} MiB). Waiting ${POLL_INTERVAL}s..."
    sleep "${POLL_INTERVAL}"
done

# Run the sweep
log "Launching run_accel_sweep.sh"
export VENV_PYTHON
export MLFLOW_ENABLED="${MLFLOW_ENABLED:-false}"

bash scripts/run_accel_sweep.sh 2>&1 | tee -a "$LOG_FILE"

log "=== Sweep Complete ==="
