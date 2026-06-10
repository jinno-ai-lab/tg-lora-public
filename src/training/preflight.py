"""Pre-flight validation for training runs (TASK-0018).

Checks configuration, data paths, and environment before expensive GPU
training starts.  All checks are CPU-only and GPU-independent.
"""

from pathlib import Path

from src.training.config_schema import BaselineConfig, TGLoRAConfig


class PreflightError(Exception):
    """Raised when training prerequisites are not met."""


def validate_training_prerequisites(
    cfg: BaselineConfig | TGLoRAConfig,
    config_path: str | Path | None = None,
) -> list[str]:
    """Validate training prerequisites and return a list of warning messages.

    Raises PreflightError if any critical prerequisite fails.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Data file existence
    for field_name in ("train_path", "valid_quick_path"):
        path = Path(getattr(cfg.data, field_name))
        if not path.exists():
            errors.append(f"Data file not found: {path} (data.{field_name})")

    valid_full = Path(cfg.data.valid_full_path)
    if cfg.data.valid_full_path and not valid_full.exists():
        warnings.append(
            f"Full validation data not found: {valid_full} (data.valid_full_path)"
        )

    # run_dir writability
    run_dir = Path(cfg.logging.run_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        test_file = run_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as exc:
        errors.append(f"run_dir is not writable: {run_dir} ({exc})")

    # max_seq_len sanity
    if cfg.data.max_seq_len < 32:
        errors.append(f"max_seq_len too small: {cfg.data.max_seq_len}")

    # Learning rate positivity (already enforced by Pydantic but double-check)
    if cfg.training.learning_rate <= 0:
        errors.append(f"learning_rate must be positive: {cfg.training.learning_rate}")

    # MPS + bitsandbytes 4bit is unsupported
    if cfg.model.load_in_4bit:
        from src.utils.device import detect_device
        device = detect_device()
        if device.type == "mps":
            errors.append(
                "load_in_4bit=True is not supported on MPS (bitsandbytes requires CUDA). "
                "Set model.load_in_4bit=false for MPS training."
            )

    if errors:
        raise PreflightError("\n".join(errors))

    return warnings


def validate_max_seq_len(
    cfg: BaselineConfig | TGLoRAConfig,
    tokenizer,
) -> None:
    """Check that max_seq_len does not exceed the tokenizer's model_max_length.

    Call this after loading the tokenizer (inside the training scripts).
    """
    max_allowed = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_allowed, int) and cfg.data.max_seq_len > max_allowed:
        raise PreflightError(
            f"max_seq_len ({cfg.data.max_seq_len}) exceeds tokenizer "
            f"model_max_length ({max_allowed})"
        )
