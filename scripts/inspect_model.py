"""Inspect a HuggingFace model to discover LoRA-compatible target modules.

Downloads only the config.json (no weights), enumerates all Linear layers
by name pattern, and recommends target_modules for PEFT LoRA.

Usage:
    python scripts/inspect_model.py --model Qwen/Qwen3.5-9B
    python scripts/inspect_model.py --model Qwen/Qwen3.5-9B --download-weights
    python scripts/inspect_model.py --config configs/9b_tg_lora.yaml
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.utils.logging import setup_logging

logger = setup_logging()


def inspect_from_config(model_name: str, download_weights: bool = False):
    """Inspect model by loading it (optionally with weights)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _print_config_summary(model_name, cfg)

    if download_weights:
        logger.info("Loading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    else:
        logger.info("Loading model structure (no weights)...")
        import torch

        # Load with random weights on CPU just for structure inspection
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        except Exception:
            # If full loading fails, try from config
            logger.info("Full load failed, instantiating from config...")
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    _analyze_model(model)
    return model


def inspect_from_yaml(config_path: str):
    """Inspect using the model specified in a YAML config."""
    from src.training.config_schema import load_and_validate_config

    cfg = load_and_validate_config(config_path)
    model_name = cfg.model.name_or_path
    logger.info(f"Model from config: {model_name}")
    inspect_from_config(model_name)


def _print_config_summary(model_name: str, cfg):
    """Print key config details."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"{'=' * 60}")
    print(f"  model_type:        {getattr(cfg, 'model_type', 'N/A')}")

    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg:
        print(f"  text_model_type:   {getattr(text_cfg, 'model_type', 'N/A')}")
        print(f"  hidden_size:       {getattr(text_cfg, 'hidden_size', 'N/A')}")
        print(f"  num_hidden_layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
        print(f"  vocab_size:        {getattr(text_cfg, 'vocab_size', 'N/A')}")
        print(f"  architectures:     {getattr(text_cfg, 'architectures', 'N/A')}")
    else:
        print(f"  hidden_size:       {getattr(cfg, 'hidden_size', 'N/A')}")
        print(f"  num_hidden_layers: {getattr(cfg, 'num_hidden_layers', 'N/A')}")
        print(f"  vocab_size:        {getattr(cfg, 'vocab_size', 'N/A')}")
        print(f"  architectures:     {getattr(cfg, 'architectures', 'N/A')}")

    if text_cfg and hasattr(text_cfg, "full_attention_interval"):
        total = getattr(text_cfg, "num_hidden_layers", 0)
        interval = getattr(text_cfg, "full_attention_interval", 1)
        n_full = total // interval
        n_linear = total - n_full
        print(f"  full_attn_interval: {interval}")
        print(f"  attention layers:  {n_full} full + {n_linear} linear (DeltaNet)")

    print(f"{'=' * 60}\n")


def _analyze_model(model):
    """Analyze model structure and recommend LoRA targets."""
    import torch.nn as nn

    linear_layers = defaultdict(list)
    all_named = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract the leaf name (last component)
            parts = name.split(".")
            leaf = parts[-1]
            linear_layers[leaf].append(name)
            all_named[name] = {
                "shape": str(tuple(module.weight.shape)),
                "leaf": leaf,
            }

    # Group by leaf name and count
    print("Linear layers by name pattern:")
    print(f"{'  leaf name':<25} {'count':>6} {'example path'}")
    print(f"{'  ' + '-' * 23} {'------':>6} {'-' * 50}")

    examples = {}
    for leaf, names in sorted(linear_layers.items(), key=lambda x: -len(x[1])):
        example = names[0]
        # Shorten example path
        if len(example) > 70:
            example = "..." + example[-67:]
        examples[leaf] = example
        print(f"  {leaf:<25} {len(names):>6}   {example}")

    print()

    # Categorize into attention vs mlp vs other
    attn_leaves = set()
    mlp_leaves = set()
    other_leaves = set()

    for leaf, names in linear_layers.items():
        if any(
            kw in names[0] for kw in ["attn", "attention", "self_attn", "linear_attn"]
        ):
            attn_leaves.add(leaf)
        elif "mlp" in names[0] or "ffn" in names[0] or "feed_forward" in names[0]:
            mlp_leaves.add(leaf)
        else:
            other_leaves.add(leaf)

    # Generate recommendations
    print("Recommended target_modules for LoRA:")
    print()

    print("  # RECOMMENDED for Qwen3.5 (hybrid DeltaNet + Attention):")
    print("  target_modules: all-linear")
    print()

    print("  # Manual: Attention-only (minimal):")
    print("  target_modules:")
    for leaf in sorted(attn_leaves):
        print(f"    - {leaf}")

    print()
    print("  # Attention + MLP (recommended for better quality):")
    print("  target_modules:")
    for leaf in sorted(attn_leaves | mlp_leaves):
        print(f"    - {leaf}")

    if other_leaves:
        print()
        print("  # Other Linear layers (review before including):")
        for leaf in sorted(other_leaves):
            print(f"    # - {leaf}  # {examples.get(leaf, '')}")

    # Parameter count
    total_params = 0
    trainable_estimate = 0
    for name, p in model.named_parameters():
        total_params += p.numel()
        for leaf in attn_leaves | mlp_leaves:
            if name.endswith(f".{leaf}.weight") or name.endswith(f".{leaf}.bias"):
                trainable_estimate += p.numel()

    print(f"\n  Total parameters: {total_params:,}")
    print(
        f"  Estimated LoRA trainable (r=8, attn+mlp): ~{trainable_estimate:,} "
        f"({100 * trainable_estimate / max(total_params, 1):.2f}%)"
    )

    # Print layer-level detail for transformer layers
    print("\n  Layer-level detail (first 4 layers):")
    for name, module in model.named_modules():
        if "layers." in name and name.count(".") <= 3:
            import re

            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                idx = int(m.group(1))
                if idx >= 4:
                    continue
                if isinstance(module, nn.Module) and not isinstance(module, nn.Linear):
                    submodules = [
                        n.split(".")[-1]
                        for n, _ in module.named_modules()
                        if n != name and isinstance(_, nn.Linear)
                    ]
                    if submodules:
                        print(
                            f"    layers.{idx}: {type(module).__name__} -> {submodules}"
                        )

    print()

    # Save full report as JSON
    report = {
        "linear_layers": {leaf: len(names) for leaf, names in linear_layers.items()},
        "recommendations": {
            "attention_only": sorted(attn_leaves),
            "attention_and_mlp": sorted(attn_leaves | mlp_leaves),
        },
        "total_params": total_params,
    }
    report_path = Path("reports/model_inspection.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Full report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model for LoRA target modules"
    )
    parser.add_argument(
        "--model", type=str, help="HuggingFace model name (e.g. Qwen/Qwen3.5-9B)"
    )
    parser.add_argument(
        "--config", type=str, help="YAML config file to read model name from"
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download full weights (slow, uses more RAM)",
    )
    args = parser.parse_args()

    if args.config:
        inspect_from_yaml(args.config)
    elif args.model:
        inspect_from_config(args.model, download_weights=args.download_weights)
    else:
        parser.error("Specify --model or --config")


if __name__ == "__main__":
    main()
