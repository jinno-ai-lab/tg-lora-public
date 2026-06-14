#!/usr/bin/env python
"""Convert a HuggingFace model to MLX format with optional 4-bit quantization.

Usage:
    python scripts/convert_mlx_model.py --model Qwen/Qwen3.5-9B --quantize
    python scripts/convert_mlx_model.py --model Qwen/Qwen3.5-9B --quantize --bits 4 --group-size 64
    python scripts/convert_mlx_model.py --model Qwen/Qwen3.5-9B  # fp16 (no quantization)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF model to MLX format")
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument(
        "--bits", type=int, default=4, help="Quantization bits (default: 4)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: .cache/mlx_models/<model-name>)",
    )
    args = parser.parse_args()

    model_name = args.model.replace("/", "--")
    output_dir = Path(args.output or f".cache/mlx_models/{model_name}")

    print(f"Converting: {args.model}")
    print(f"Output:     {output_dir}")
    print(
        f"Quantize:   {args.quantize} (bits={args.bits}, group_size={args.group_size})"
    )

    if output_dir.exists():
        print(f"[WARN] Output directory already exists, removing: {output_dir}")
        shutil.rmtree(output_dir)

    from mlx_lm.convert import convert

    convert(
        hf_path=args.model,
        mlx_path=str(output_dir),
        quantize=args.quantize,
        q_bits=args.bits if args.quantize else None,
        q_group_size=args.group_size if args.quantize else None,
    )

    print(f"\nDone. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
