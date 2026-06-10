# TG-LoRA development environment
# CUDA 12.1 + Python 3.11 for RTX3060 12GB training
#
# Usage:
#   docker compose build
#   docker compose run --rm tg-lora make test

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch first (cache layer)
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
COPY pyproject.toml ./
RUN pip install -e ".[dev]" || true

# Copy source
COPY . .

# Install project (ensures editable install sees all source)
RUN pip install -e ".[dev]"

# Default: run tests
CMD ["pytest", "tests/", "-v"]
