#!/usr/bin/env python
"""Run mlx_lm.lora with local guards for pre-#3524 MLX shape overflow."""

from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx.src.utils.shape_guard import install  # noqa: E402


def _cleanup_metal() -> None:
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            mx.synchronize()
            mx.clear_cache()
    except Exception:
        pass


def _patch_mlx_lm_for_long_training(lora_module: object) -> None:
    if os.environ.get("TG_LORA_MLX_SKIP_VALIDATION") != "1":
        return

    original_train = lora_module.train

    def train_without_validation(*args, **kwargs):
        kwargs["val_dataset"] = None
        return original_train(*args, **kwargs)

    lora_module.train = train_without_validation


def main() -> None:
    install()
    import mlx_lm.lora as mlx_lora

    _patch_mlx_lm_for_long_training(mlx_lora)
    try:
        mlx_lora.main()
    finally:
        _cleanup_metal()


if __name__ == "__main__":
    main()
