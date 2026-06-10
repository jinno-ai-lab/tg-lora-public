#!/usr/bin/env bash
# Setup development environment for TG-LoRA.
#
# Creates conda environment, installs all dependencies including
# latest transformers from main (required for Qwen3.5).
#
# Usage:
#   bash scripts/setup_env.sh
#   bash scripts/setup_env.sh --python 3.12
#   bash scripts/setup_env.sh --cuda 124

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-121}"
ENV_NAME="${ENV_NAME:-tg-lora}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        --name) ENV_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  TG-LoRA Environment Setup"
echo "  Python:  ${PYTHON_VERSION}"
echo "  CUDA:    ${CUDA_VERSION}"
echo "  Env:     ${ENV_NAME}"
echo "============================================"

# --- Conda ---
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "[1/6] Creating conda environment..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y 2>/dev/null || true

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# --- PyTorch ---
echo ""
echo "[2/6] Installing PyTorch (CUDA ${CUDA_VERSION})..."
pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"

# --- Core dependencies ---
echo ""
echo "[3/6] Installing core ML dependencies..."
pip install \
    bitsandbytes \
    datasets \
    safetensors \
    sentencepiece \
    tokenizers

# --- Latest transformers + peft + accelerate (required for Qwen3.5) ---
echo ""
echo "[4/6] Installing latest transformers/peft/accelerate from main (Qwen3.5 support)..."
pip install \
    "transformers @ git+https://github.com/huggingface/transformers.git@main" \
    "peft @ git+https://github.com/huggingface/peft.git@main" \
    "accelerate @ git+https://github.com/huggingface/accelerate.git@main"

# --- Project & utilities ---
echo ""
echo "[5/6] Installing project and utilities..."
pip install -e ".[dev]"

pip install \
    hydra-core \
    omegaconf \
    mlflow \
    orjson \
    jsonlines \
    tqdm \
    rich \
    rapidfuzz \
    sentence-transformers \
    faiss-cpu

# --- Verification ---
echo ""
echo "[6/6] Verifying installation..."

python -c "
import torch
import transformers
import peft
import bitsandbytes

print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT:         {peft.__version__}')
print(f'  Bitsandbytes: {bitsandbytes.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:        {props.name} ({props.total_mem // 1024**2} MB)')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate with:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Next steps:"
echo "    make inspect          # Check model architecture & LoRA targets"
echo "    make download-data    # Download training data"
echo "    make prepare-data     # Prepare train/valid splits"
echo "============================================"
