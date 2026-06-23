"""TG-LoRA diagnostic health check — automates runbook procedures.

Usage:
    python scripts/diagnose.py                        # full check
    python scripts/diagnose.py --gpu                  # GPU status only
    python scripts/diagnose.py --checkpoint <path>    # checkpoint integrity
    python scripts/diagnose.py --config <path>        # config validation
    python scripts/diagnose.py --logs <dir>           # log analysis
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow running as a standalone CLI (``python scripts/diagnose.py``): a bare
# script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the
# repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.tensor_artifact import load_tensor_artifact


@dataclass
class CheckResult:
    status: str  # "ok", "warn", "error"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def check_gpu() -> list[CheckResult]:
    """Check GPU availability, memory, and device status."""
    results: list[CheckResult] = []
    try:
        import torch
    except ImportError:
        results.append(CheckResult("error", "PyTorch not installed"))
        return results

    from src.utils.device import (detect_device, gpu_memory_allocated_mb)

    device = detect_device()

    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory / (1024 * 1024)
                allocated_mb = torch.cuda.memory_allocated(i) / (1024 * 1024)
                free_mb = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024 * 1024)

                results.append(
                    CheckResult(
                        "ok",
                        f"GPU {i}: {name}",
                        {
                            "total_mb": round(total_mb, 1),
                            "allocated_mb": round(allocated_mb, 1),
                            "free_mb": round(free_mb, 1),
                        },
                    )
                )
            except Exception as e:
                results.append(
                    CheckResult(
                        "warn",
                        f"GPU {i}: Error querying properties ({str(e)})",
                    )
                )
                continue

            if total_mb < 10_000:
                results.append(
                    CheckResult(
                        "warn",
                        f"GPU {i} has {total_mb:.0f}MB — RTX 3060 12GB recommended minimum",
                        {
                            "recommendation": "Use max_seq_len=1024 and grad_accumulation=4 for <12GB GPUs",
                        },
                    )
                )

        cuda_version = torch.version.cuda
        if cuda_version:
            results.append(CheckResult("ok", f"CUDA version: {cuda_version}"))
        else:
            results.append(CheckResult("warn", "CUDA version not detected"))

    elif device.type == "mps":
        results.append(CheckResult("ok", "Apple MPS (Metal Performance Shaders)"))
        alloc = gpu_memory_allocated_mb(device)
        if alloc is not None:
            results.append(
                CheckResult("ok", f"MPS allocated memory: {alloc:.0f} MB")
            )

    else:
        results.append(
            CheckResult("warn", "No GPU detected — training will run on CPU (very slow)")
        )

    return results


def check_checkpoint(checkpoint_dir: str) -> list[CheckResult]:
    """Verify checkpoint integrity: files exist, no NaN/Inf in weights."""
    results: list[CheckResult] = []
    path = Path(checkpoint_dir)

    if not path.exists():
        results.append(
            CheckResult("error", f"Checkpoint directory not found: {checkpoint_dir}")
        )
        return results

    # Check required files
    required_files = ["adapter_config.json"]
    found_any_weight = False
    for weight_file in ["adapter_model.safetensors", "adapter_model.bin"]:
        if (path / weight_file).exists():
            found_any_weight = True
            break

    for f in required_files:
        if not (path / f).exists():
            results.append(CheckResult("warn", f"Missing {f} in checkpoint"))

    if not found_any_weight:
        results.append(
            CheckResult(
                "error", "No adapter weights found (adapter_model.safetensors or .bin)"
            )
        )
    else:
        results.append(CheckResult("ok", "Adapter weights file present"))

    # Check training_state.pt if present
    state_path = path / "training_state.pt"
    if state_path.exists():
        try:
            state = load_tensor_artifact(state_path)
            keys = list(state.keys()) if isinstance(state, dict) else ["<non-dict>"]
            results.append(
                CheckResult(
                    "ok",
                    f"training_state.pt loaded ({len(keys)} keys)",
                    {
                        "keys": keys[:10],
                    },
                )
            )

            # Check for NaN/Inf in tensors
            non_finite_count = 0
            for k, v in state.items():
                if hasattr(v, "isfinite") and not v.isfinite().all():
                    non_finite_count += 1
                    results.append(
                        CheckResult(
                            "warn", f"Non-finite values in training_state key: {k}"
                        )
                    )
            if non_finite_count == 0:
                results.append(
                    CheckResult("ok", "All tensors in training_state.pt are finite")
                )
        except Exception as e:
            results.append(
                CheckResult("error", f"Failed to load training_state.pt: {e}")
            )
    else:
        results.append(
            CheckResult("warn", "No training_state.pt — resume will start fresh")
        )

    # Check safetensors for NaN/Inf
    safetensor_path = path / "adapter_model.safetensors"
    if safetensor_path.exists():
        try:
            from safetensors import safe_open

            non_finite = []
            with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if not tensor.isfinite().all():
                        non_finite.append(key)
            if non_finite:
                results.append(
                    CheckResult(
                        "error",
                        f"NaN/Inf in adapter weights: {non_finite}",
                        {
                            "affected_keys": non_finite,
                        },
                    )
                )
            else:
                results.append(
                    CheckResult("ok", "All adapter weight tensors are finite")
                )
        except ImportError:
            results.append(
                CheckResult(
                    "warn", "safetensors not installed — skipping weight validation"
                )
            )
        except Exception as e:
            results.append(CheckResult("error", f"Failed to inspect safetensors: {e}"))

    return results


# Recommended config values from the runbook
_RECOMMENDED_DEFAULTS = {
    "tg_lora.K_initial": (2, 8),
    "tg_lora.N_initial": (1, 20),
    "tg_lora.alpha_initial": (0.03, 1.5),
    "tg_lora.beta_initial": (0.5, 0.95),
    "tg_lora.lr_initial": (1e-5, 1e-3),
    "tg_lora.relative_update_cap": (0.001, 0.01),
    "training.grad_accumulation": (1, 16),
    "data.max_seq_len": (128, 4096),
}


def _get_nested(cfg: Any, dotted_key: str) -> Any:
    """Get a nested value from an OmegaConf/dict using dotted notation."""
    keys = dotted_key.split(".")
    obj = cfg
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            obj = getattr(obj, k, None)
        if obj is None:
            return None
    return obj


def check_config(config_path: str) -> list[CheckResult]:
    """Validate training config against recommended ranges from the runbook."""
    results: list[CheckResult] = []

    path = Path(config_path)
    if not path.exists():
        results.append(CheckResult("error", f"Config file not found: {config_path}"))
        return results

    try:
        from src.training.config_schema import load_and_validate_config

        validated = load_and_validate_config(config_path)
        cfg = validated.model_dump()
        results.append(CheckResult("ok", f"Config loaded and validated: {config_path}"))
    except Exception as e:
        results.append(
            CheckResult("warn", f"Pydantic validation failed: {e}; falling back to raw load")
        )
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(config_path)
        except Exception as e2:
            results.append(CheckResult("error", f"Failed to load config: {e2}"))
            return results

    results.append(CheckResult("ok", f"Config loaded: {config_path}"))

    for key, (lo, hi) in _RECOMMENDED_DEFAULTS.items():
        val = _get_nested(cfg, key)
        if val is None:
            continue
        try:
            val_num = float(val)
        except (TypeError, ValueError):
            continue

        if val_num < lo or val_num > hi:
            results.append(
                CheckResult(
                    "warn",
                    f"{key}={val_num} outside recommended [{lo}, {hi}]",
                    {
                        "key": key,
                        "value": val_num,
                        "recommended_range": [lo, hi],
                    },
                )
            )
        else:
            results.append(CheckResult("ok", f"{key}={val_num} (in range)"))

    # Check critical settings for 12GB GPU
    seq_len = _get_nested(cfg, "data.max_seq_len")
    if seq_len and int(seq_len) > 2048:
        results.append(
            CheckResult(
                "warn",
                f"max_seq_len={seq_len} may cause OOM on 12GB GPU",
                {
                    "recommendation": "Use max_seq_len=1024 for RTX 3060 12GB",
                },
            )
        )

    gc = _get_nested(cfg, "training.gradient_checkpointing")
    if gc is False:
        results.append(
            CheckResult(
                "warn",
                "gradient_checkpointing is disabled — enables 30-40% memory savings",
                {
                    "recommendation": "Set training.gradient_checkpointing: true",
                },
            )
        )

    return results


# Error patterns from the runbook
_ERROR_PATTERNS = {
    "oom": (re.compile(r"OutOfMemoryError", re.IGNORECASE), "error", "OOM detected"),
    "cuda_error": (
        re.compile(r"CUDA\s+error", re.IGNORECASE),
        "error",
        "CUDA error detected",
    ),
    "nan_loss": (
        re.compile(r"non-finite|NaN|Inf.*loss|loss.*NaN", re.IGNORECASE),
        "error",
        "Non-finite loss detected",
    ),
    "instability": (
        re.compile(r"NumericalInstabilityError|instability", re.IGNORECASE),
        "warn",
        "Numerical instability detected",
    ),
    "checkpoint_saved": (
        re.compile(r"fault.*checkpoint.*saved|saved.*fault.*checkpoint", re.IGNORECASE),
        "warn",
        "Fault checkpoint was saved",
    ),
}


def check_logs(log_dir: str) -> list[CheckResult]:
    """Analyze training logs for error patterns documented in the runbook."""
    results: list[CheckResult] = []
    path = Path(log_dir)

    if not path.exists():
        results.append(CheckResult("error", f"Log directory not found: {log_dir}"))
        return results

    log_files = sorted(path.rglob("*.log")) + sorted(path.rglob("train.log"))
    if not log_files:
        # Also check for any text files
        log_files = sorted(path.rglob("*.txt")) + sorted(path.rglob("*.out"))

    if not log_files:
        results.append(CheckResult("warn", f"No log files found in {log_dir}"))
        return results

    results.append(CheckResult("ok", f"Found {len(log_files)} log file(s)"))

    error_summary: dict[str, list[tuple[str, int]]] = {}
    for log_file in log_files:
        try:
            text = log_file.read_text(errors="replace")
        except Exception as e:
            results.append(CheckResult("error", f"Cannot read {log_file.name}: {e}"))
            continue

        for pattern_name, (regex, severity, label) in _ERROR_PATTERNS.items():
            matches = regex.findall(text)
            if matches:
                # Get line numbers
                lines_with_matches = []
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        lines_with_matches.append((log_file.name, i))
                error_summary.setdefault(pattern_name, []).extend(
                    lines_with_matches[:10]
                )

    if error_summary:
        for pattern_name, occurrences in error_summary.items():
            _, severity, label = _ERROR_PATTERNS[pattern_name]
            count = len(occurrences)
            results.append(
                CheckResult(
                    severity,
                    f"{label} ({count} occurrences)",
                    {
                        "locations": [
                            f"{name}:{line}" for name, line in occurrences[:5]
                        ],
                    },
                )
            )
    else:
        results.append(CheckResult("ok", "No error patterns found in logs"))

    return results


def _format_result(r: CheckResult, verbose: bool = False) -> str:
    icon = {"ok": "+", "warn": "!", "error": "X"}[r.status]
    line = f"  [{icon}] {r.message}"
    if verbose and r.details:
        for k, v in r.details.items():
            line += f"\n      {k}: {v}"
    return line


def run_all_checks(
    gpu: bool = False,
    checkpoint: str | None = None,
    config: str | None = None,
    logs: str | None = None,
    verbose: bool = False,
) -> dict[str, list[CheckResult]]:
    """Run selected diagnostic checks and return results by category."""
    all_results: dict[str, list[CheckResult]] = {}
    do_all = not (gpu or checkpoint or config or logs)

    if gpu or do_all:
        all_results["GPU"] = check_gpu()
    if checkpoint or do_all:
        default_ckpt = "runs"
        ckpt_path = checkpoint if checkpoint else default_ckpt
        all_results["Checkpoint"] = check_checkpoint(ckpt_path)
    if config:
        all_results["Config"] = check_config(config)
    elif do_all:
        all_results["Config"] = [
            CheckResult("ok", "No config specified — use --config <path>")
        ]
    if logs or do_all:
        default_logs = "runs"
        log_path = logs if logs else default_logs
        all_results["Logs"] = check_logs(log_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TG-LoRA diagnostic health check")
    parser.add_argument("--gpu", action="store_true", help="Check GPU status only")
    parser.add_argument(
        "--checkpoint", type=str, help="Verify checkpoint integrity at <path>"
    )
    parser.add_argument("--config", type=str, help="Validate training config at <path>")
    parser.add_argument("--logs", type=str, help="Analyze training logs in <dir>")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    args = parser.parse_args()

    results = run_all_checks(
        gpu=args.gpu,
        checkpoint=args.checkpoint,
        config=args.config,
        logs=args.logs,
        verbose=args.verbose,
    )

    if args.json:
        output = {}
        for category, checks in results.items():
            output[category] = [
                {"status": r.status, "message": r.message, "details": r.details}
                for r in checks
            ]
        print(json.dumps(output, indent=2, default=str))
    else:
        for category, checks in results.items():
            print(f"\n=== {category} ===")
            for r in checks:
                print(_format_result(r, verbose=args.verbose))
                if r.status == "error":
                    pass

        print()

    # Exit with error if any check failed
    any_error = any(r.status == "error" for checks in results.values() for r in checks)
    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
