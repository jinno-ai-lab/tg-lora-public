"""TG-LoRA automated fault recovery — makes runbook procedures executable.

Usage:
    python scripts/recover.py --analyze <run_dir>                # diagnose fault type
    python scripts/recover.py --sanitize <checkpoint_dir>        # clean NaN/Inf from weights
    python scripts/recover.py --fix-config <config.yaml>         # generate recovery config
    python scripts/recover.py --remediate <run_dir> <config.yaml>  # full automated recovery
    python scripts/recover.py --remediate <run_dir> <config.yaml> --rerun  # ... + auto-launch the corrective re-run
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Allow running as a standalone CLI (``python scripts/recover.py``): a bare
# script invocation puts ``scripts/`` — not the repo root — on sys.path, so make the
# repo root importable so ``src.*`` resolves without a PYTHONPATH wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.atomic_save import _atomic_torch_save
from src.utils.tensor_artifact import load_tensor_artifact

_FAULT_PATTERNS = {
    "oom": re.compile(r"OutOfMemoryError", re.IGNORECASE),
    "cuda_error": re.compile(r"CUDA\s+error|device-side\s+assert", re.IGNORECASE),
    "nan_loss": re.compile(r"non-finite|NaN|Inf.*loss|loss.*NaN", re.IGNORECASE),
    "instability": re.compile(r"NumericalInstabilityError|instability", re.IGNORECASE),
}

_RECOMMENDED_RANGES = {
    "data.max_seq_len": (128, 2048),
    "training.batch_size": (1, 4),
    "training.grad_accumulation": (4, 16),
    "tg_lora.alpha_initial": (0.03, 1.0),
    "tg_lora.relative_update_cap": (0.001, 0.005),
    "tg_lora.lr_initial": (1e-5, 5e-4),
}


class RecoveryResult:
    """Structured result for a recovery action."""

    def __init__(
        self,
        action: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        self.action = action
        self.status = status  # "ok", "warn", "error", "skipped"
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def analyze_fault(run_dir: str) -> list[RecoveryResult]:
    """Analyze a run directory to determine fault type from logs and checkpoints.

    Checks:
    - Log files for OOM, CUDA error, NaN patterns
    - training_state.pt for presence (indicates interrupted training)
    - oom_checkpoint/ for OOM recovery checkpoint
    """
    results: list[RecoveryResult] = []
    run_path = Path(run_dir)

    if not run_path.exists():
        return [
            RecoveryResult("analyze", "error", f"Run directory not found: {run_dir}")
        ]

    # Check for fault indicators in the directory structure
    has_oom_checkpoint = (run_path / "oom_checkpoint").exists()
    has_training_state = (run_path / "training_state.pt").exists()

    if has_oom_checkpoint:
        results.append(
            RecoveryResult(
                "analyze",
                "ok",
                "OOM checkpoint found",
                {"path": str(run_path / "oom_checkpoint")},
            )
        )

    if has_training_state:
        results.append(
            RecoveryResult(
                "analyze",
                "ok",
                "Training state found — interrupted training detected",
                {"path": str(run_path / "training_state.pt")},
            )
        )

    # Analyze log files
    log_files = list(run_path.rglob("*.log")) + list(run_path.rglob("*.txt"))
    if not log_files:
        results.append(
            RecoveryResult("analyze", "warn", "No log files found in run directory")
        )
        return results

    detected_faults: dict[str, list[str]] = {}
    for log_file in log_files:
        try:
            text = log_file.read_text(errors="replace")
        except Exception as e:
            results.append(
                RecoveryResult("analyze", "error", f"Cannot read {log_file.name}: {e}")
            )
            continue

        for fault_name, pattern in _FAULT_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detected_faults.setdefault(fault_name, []).extend(
                    [
                        f"{log_file.name}:{i}"
                        for i, line in enumerate(text.splitlines(), 1)
                        if pattern.search(line)
                    ][:5],
                )

    if detected_faults:
        for fault_name, locations in detected_faults.items():
            results.append(
                RecoveryResult(
                    "analyze",
                    "warn",
                    f"Fault detected: {fault_name} ({len(locations)} occurrences)",
                    {"locations": locations},
                )
            )
    else:
        results.append(
            RecoveryResult("analyze", "ok", "No fault patterns detected in logs")
        )

    # Determine primary fault type
    if "oom" in detected_faults:
        results.append(
            RecoveryResult(
                "analyze",
                "ok",
                "Recommended action: OOM recovery",
                {
                    "remediation": "reduce_seq_len",
                    "config_changes": {
                        "data.max_seq_len": "reduce by 50%",
                        "training.gradient_checkpointing": True,
                    },
                },
            )
        )
    elif "nan_loss" in detected_faults or "instability" in detected_faults:
        results.append(
            RecoveryResult(
                "analyze",
                "ok",
                "Recommended action: NaN/instability recovery",
                {
                    "remediation": "reduce_alpha_and_cap",
                    "config_changes": {
                        "tg_lora.alpha_initial": "halve",
                        "tg_lora.relative_update_cap": "halve",
                    },
                },
            )
        )
    elif "cuda_error" in detected_faults:
        results.append(
            RecoveryResult(
                "analyze",
                "ok",
                "Recommended action: CUDA error recovery",
                {"remediation": "clean_restart"},
            )
        )

    return results


def sanitize_checkpoint(
    checkpoint_dir: str, output_dir: str | None = None
) -> list[RecoveryResult]:
    """Remove NaN/Inf from adapter weights in a checkpoint directory.

    Creates a sanitized copy in output_dir, or overwrites in-place if output_dir is None.
    """
    results: list[RecoveryResult] = []
    src_path = Path(checkpoint_dir)

    if not src_path.exists():
        return [
            RecoveryResult(
                "sanitize", "error", f"Checkpoint directory not found: {checkpoint_dir}"
            )
        ]

    dst_path = Path(output_dir) if output_dir else src_path
    dst_path.mkdir(parents=True, exist_ok=True)

    # Copy config files as-is
    for cfg_file in src_path.glob("*.json"):
        if output_dir:
            import shutil

            shutil.copy2(cfg_file, dst_path / cfg_file.name)
        results.append(
            RecoveryResult("sanitize", "ok", f"Config copied: {cfg_file.name}")
        )

    # Handle safetensors
    safetensor_src = src_path / "adapter_model.safetensors"
    if safetensor_src.exists():
        try:
            import torch
            from safetensors.torch import load_file, save_file

            tensors = load_file(safetensor_src)
            sanitized_keys = []
            for key in list(tensors.keys()):
                t = tensors[key]
                if not torch.isfinite(t).all():
                    tensors[key] = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
                    sanitized_keys.append(key)

            save_file(tensors, dst_path / "adapter_model.safetensors")

            if sanitized_keys:
                results.append(
                    RecoveryResult(
                        "sanitize",
                        "ok",
                        f"Sanitized NaN/Inf in {len(sanitized_keys)} tensor(s)",
                        {"sanitized_keys": sanitized_keys},
                    )
                )
            else:
                results.append(
                    RecoveryResult("sanitize", "ok", "All tensors already finite")
                )
        except ImportError:
            results.append(
                RecoveryResult("sanitize", "error", "safetensors package required")
            )
        except Exception as e:
            results.append(
                RecoveryResult("sanitize", "error", f"Failed to sanitize: {e}")
            )
    else:
        results.append(
            RecoveryResult("sanitize", "warn", "No adapter_model.safetensors found")
        )

    # Handle training_state.pt
    state_src = src_path / "training_state.pt"
    if state_src.exists():
        try:
            import torch

            state = load_tensor_artifact(state_src)
            non_finite = []
            if isinstance(state, dict):
                for key in list(state.keys()):
                    val = state[key]
                    if isinstance(val, dict):
                        for k2, v2 in val.items():
                            if hasattr(v2, "isfinite") and not v2.isfinite().all():
                                state[key][k2] = torch.nan_to_num(
                                    v2, nan=0.0, posinf=1e6, neginf=-1e6
                                )
                                non_finite.append(f"{key}.{k2}")
                    elif hasattr(val, "isfinite") and not val.isfinite().all():
                        state[key] = torch.nan_to_num(
                            val, nan=0.0, posinf=1e6, neginf=-1e6
                        )
                        non_finite.append(key)

            # Route through the atomic helper so an OOM kill / SIGINT mid-dump
            # never leaves a torn ``training_state.pt`` — the resume-critical
            # artifact this sanitize step exists to keep loadable.
            _atomic_torch_save(state, dst_path / "training_state.pt")
            if non_finite:
                results.append(
                    RecoveryResult(
                        "sanitize",
                        "ok",
                        f"Sanitized {len(non_finite)} non-finite tensor(s) in training_state",
                        {"sanitized_keys": non_finite},
                    )
                )
            else:
                results.append(
                    RecoveryResult(
                        "sanitize", "ok", "training_state tensors are finite"
                    )
                )
        except Exception as e:
            results.append(
                RecoveryResult(
                    "sanitize", "warn", f"Could not process training_state: {e}"
                )
            )

    return results


def _get_nested(cfg: Any, dotted_key: str) -> Any:
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


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    obj = cfg
    for k in keys[:-1]:
        if k not in obj or not isinstance(obj[k], dict):
            obj[k] = {}
        obj = obj[k]
    obj[keys[-1]] = value


def generate_recovery_config(
    config_path: str, fault_type: str, output_path: str
) -> RecoveryResult:
    """Generate a recovery config by adjusting parameters for the given fault type.

    Applies runbook-specified adjustments:
    - OOM: reduce max_seq_len by 50%, enable gradient_checkpointing
    - NaN/instability: halve alpha_initial and relative_update_cap
    - CUDA: enable gradient_checkpointing, reduce batch_size
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return RecoveryResult("fix-config", "error", f"Config not found: {config_path}")

    try:
        from src.training.config_schema import load_and_validate_config

        validated = load_and_validate_config(config_path)
        data = validated.model_dump()
    except Exception as e:
        return RecoveryResult("fix-config", "error", f"Config validation failed: {e}")

    changes: dict[str, Any] = {}

    if fault_type == "oom":
        seq_len = _get_nested(data, "data.max_seq_len")
        if seq_len and int(seq_len) > 512:
            new_len = max(512, int(seq_len) // 2)
            _set_nested(data, "data.max_seq_len", new_len)
            changes["data.max_seq_len"] = f"{seq_len} → {new_len}"

        gc = _get_nested(data, "training.gradient_checkpointing")
        if gc is not True:
            _set_nested(data, "training.gradient_checkpointing", True)
            changes["training.gradient_checkpointing"] = f"{gc} → True"

    elif fault_type in ("nan_loss", "instability"):
        alpha = _get_nested(data, "tg_lora.alpha_initial")
        if alpha:
            new_alpha = float(alpha) / 2
            _set_nested(data, "tg_lora.alpha_initial", new_alpha)
            changes["tg_lora.alpha_initial"] = f"{alpha} → {new_alpha}"

        cap = _get_nested(data, "tg_lora.relative_update_cap")
        if cap:
            new_cap = float(cap) / 2
            _set_nested(data, "tg_lora.relative_update_cap", new_cap)
            changes["tg_lora.relative_update_cap"] = f"{cap} → {new_cap}"

    elif fault_type == "cuda_error":
        gc = _get_nested(data, "training.gradient_checkpointing")
        if gc is not True:
            _set_nested(data, "training.gradient_checkpointing", True)
            changes["training.gradient_checkpointing"] = f"{gc} → True"

        bs = _get_nested(data, "training.batch_size")
        ga = _get_nested(data, "training.grad_accumulation")
        if bs and int(bs) > 1:
            new_bs = max(1, int(bs) // 2)
            _set_nested(data, "training.batch_size", new_bs)
            if ga:
                new_ga = int(ga) * 2
                _set_nested(data, "training.grad_accumulation", new_ga)
                changes["training.batch_size"] = f"{bs} → {new_bs}"
                changes["training.grad_accumulation"] = (
                    f"{ga} → {new_ga} (preserves effective batch)"
                )
            else:
                changes["training.batch_size"] = f"{bs} → {new_bs}"

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from omegaconf import OmegaConf

        OmegaConf.save(OmegaConf.create(data), output)
    except Exception:
        import yaml

        with open(output, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    return RecoveryResult(
        "fix-config",
        "ok",
        f"Recovery config written to {output_path}",
        {"fault_type": fault_type, "changes": changes},
    )


def _build_rerun_command(
    recovery_config: str, run_dir: str, python: str | None = None
) -> list[str]:
    """Build the corrective re-run command for an interrupted 9B arm.

    Resumes from the killed run's ``training_state.pt`` using the
    reduced-footprint ``recovery_config`` produced by
    :func:`generate_recovery_config` (halved ``max_seq_len``, enabled
    ``gradient_checkpointing``, and — for CUDA faults — halved ``batch_size``
    with doubled ``grad_accumulation``). The whole point of a recovery config
    that trims the memory footprint is to re-fire the run that OOM-killed /
    was SIGKILL'd by the OOM-killer; this closes the gap where remediation
    stopped at *printing* a ``Resume with: ...`` hint and left the operator to
    copy, paste and adjust it by hand.
    """
    state_path = str(Path(run_dir) / "training_state.pt")
    return [
        python or sys.executable,
        "-m",
        "src.training.train_tg_lora",
        "--config",
        recovery_config,
        "--resume",
        state_path,
    ]


def _default_rerun_launcher(cmd: list[str]) -> int:
    """Launch the corrective re-run as a child process and return its exit code.

    Kept as a tiny, injectable seam (:func:`apply_remediation` accepts any
    ``launcher`` callable) so the auto-relaunch path is unit-testable without
    GPU / private ``src.data`` — the test injects a recorder instead of
    spawning a real training process.
    """
    import subprocess

    return subprocess.run(cmd).returncode


def apply_remediation(
    run_dir: str,
    config_path: str,
    *,
    rerun: bool = False,
    launcher=None,
) -> list[RecoveryResult]:
    """Full automated remediation: analyze → sanitize → generate recovery config.

    Returns all recovery results for logging/display.
    """
    results: list[RecoveryResult] = []
    run_path = Path(run_dir)

    # Step 1: Analyze
    results.append(RecoveryResult("remediate", "ok", "Starting automated remediation"))
    analysis = analyze_fault(run_dir)
    results.extend(analysis)

    # Determine primary fault
    fault_type = None
    for r in analysis:
        if r.status == "warn" and "Fault detected" in r.message:
            for known_fault in ("oom", "nan_loss", "instability", "cuda_error"):
                if known_fault in r.message:
                    fault_type = known_fault
                    break
            break

    if not fault_type:
        results.append(
            RecoveryResult(
                "remediate",
                "ok",
                "No fault detected — no remediation needed",
            )
        )
        return results

    # Step 2: Sanitize OOM checkpoint if present
    oom_ckpt = run_path / "oom_checkpoint"
    if oom_ckpt.exists():
        results.append(RecoveryResult("remediate", "ok", "Sanitizing OOM checkpoint"))
        results.extend(sanitize_checkpoint(str(oom_ckpt)))

    # Step 3: Generate recovery config
    recovery_config_path = str(run_path / "recovery_config.yaml")
    result = generate_recovery_config(config_path, fault_type, recovery_config_path)
    results.append(result)

    if result.status == "ok":
        results.append(
            RecoveryResult(
                "remediate",
                "ok",
                f"Remediation complete. Resume with: python -m src.training.train_tg_lora --config {recovery_config_path} --resume {run_dir}/training_state.pt",
                {"recovery_config": recovery_config_path, "fault_type": fault_type},
            )
        )

        # Connect abort-detection to corrective re-run (feedback: 'report' →
        # 'automatic launch'). Opt-in via ``rerun`` so plain remediation still
        # just diagnoses + emits the recovery config. The launch reuses the
        # reduced-footprint recovery config so the OOM/SIGKILL that killed the
        # original arm is not re-triggered on retry.
        if rerun:
            cmd = _build_rerun_command(recovery_config_path, run_dir)
            launch = launcher or _default_rerun_launcher
            try:
                rc = launch(cmd)
                status = "ok" if rc == 0 else "warn"
                results.append(
                    RecoveryResult(
                        "remediate",
                        status,
                        f"Corrective re-run launched (exit {rc})",
                        {
                            "rerun_command": cmd,
                            "returncode": rc,
                            "fault_type": fault_type,
                        },
                    )
                )
            except Exception as e:
                results.append(
                    RecoveryResult(
                        "remediate",
                        "error",
                        f"Failed to launch corrective re-run: {e}",
                        {"rerun_command": cmd, "fault_type": fault_type},
                    )
                )

    return results


def _format_result(r: RecoveryResult, verbose: bool = False) -> str:
    icon = {"ok": "+", "warn": "!", "error": "X", "skipped": "-"}[r.status]
    line = f"  [{icon}] [{r.action}] {r.message}"
    if verbose and r.details:
        for k, v in r.details.items():
            line += f"\n      {k}: {v}"
    return line


def main():
    parser = argparse.ArgumentParser(description="TG-LoRA automated fault recovery")
    parser.add_argument(
        "--analyze",
        type=str,
        metavar="RUN_DIR",
        help="Analyze run directory for fault type",
    )
    parser.add_argument(
        "--sanitize",
        type=str,
        metavar="CHECKPOINT_DIR",
        help="Sanitize NaN/Inf in checkpoint weights",
    )
    parser.add_argument(
        "--sanitize-output",
        type=str,
        metavar="DIR",
        help="Output directory for sanitized checkpoint (default: in-place)",
    )
    parser.add_argument(
        "--fix-config",
        type=str,
        metavar="CONFIG_PATH",
        help="Generate recovery config for detected fault",
    )
    parser.add_argument(
        "--fault-type",
        type=str,
        choices=["oom", "nan_loss", "instability", "cuda_error"],
        help="Override fault type for config generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recovery_config.yaml",
        help="Output path for recovery config (default: recovery_config.yaml)",
    )
    parser.add_argument(
        "--remediate",
        nargs=2,
        metavar=("RUN_DIR", "CONFIG_PATH"),
        help="Full automated remediation: analyze + sanitize + fix-config",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="With --remediate: auto-launch the corrective re-run (resume from "
        "training_state.pt with the reduced-footprint recovery config) instead "
        "of only printing the resume command. Default: off.",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    args = parser.parse_args()

    all_results: list[RecoveryResult] = []

    if args.analyze:
        all_results.extend(analyze_fault(args.analyze))
    elif args.sanitize:
        all_results.extend(sanitize_checkpoint(args.sanitize, args.sanitize_output))
    elif args.fix_config:
        fault = args.fault_type or "oom"
        all_results.append(
            generate_recovery_config(args.fix_config, fault, args.output)
        )
    elif args.remediate:
        all_results.extend(
            apply_remediation(
                args.remediate[0], args.remediate[1], rerun=args.rerun
            )
        )
    else:
        parser.print_help()
        sys.exit(1)

    if args.json:
        print(json.dumps([r.to_dict() for r in all_results], indent=2, default=str))
    else:
        for r in all_results:
            print(_format_result(r, verbose=args.verbose))
        print()

    has_error = any(r.status == "error" for r in all_results)
    sys.exit(1 if has_error else 0)


if __name__ == "__main__":
    main()
