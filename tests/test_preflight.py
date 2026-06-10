"""Tests for pre-flight training validation (TASK-0018)."""

import os
from pathlib import Path

import pytest

from src.training.config_schema import BaselineConfig, DataConfig
from src.training.preflight import (
    PreflightError,
    validate_training_prerequisites,
    validate_max_seq_len,
)


def _make_baseline_cfg(
    tmp_path: Path,
    *,
    train_exists: bool = True,
    valid_exists: bool = True,
    valid_full_exists: bool = True,
    max_seq_len: int = 2048,
    run_dir: str | None = None,
) -> BaselineConfig:
    """Build a BaselineConfig with data paths inside tmp_path."""
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid_quick.jsonl"
    valid_full_path = tmp_path / "valid_full.jsonl"

    if train_exists:
        train_path.write_text("{}")
    if valid_exists:
        valid_path.write_text("{}")
    if valid_full_exists:
        valid_full_path.write_text("{}")

    data = DataConfig(
        train_path=str(train_path),
        valid_quick_path=str(valid_path),
        valid_full_path=str(valid_full_path),
        max_seq_len=max_seq_len,
    )
    return BaselineConfig(
        experiment={"name": "test", "seed": 42},
        model={"name_or_path": "test"},
        lora={"r": 16, "alpha": 32, "dropout": 0.05},
        data=data,
        training={
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 2e-4,
            "max_steps": 100,
        },
        logging={"run_dir": run_dir or str(tmp_path / "runs")},
    )


class TestValidateTrainingPrerequisites:
    def test_all_ok_no_errors(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path)
        warnings = validate_training_prerequisites(cfg)
        assert isinstance(warnings, list)

    def test_train_path_missing_raises(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path, train_exists=False)
        with pytest.raises(PreflightError, match="train_path"):
            validate_training_prerequisites(cfg)

    def test_valid_quick_path_missing_raises(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path, valid_exists=False)
        with pytest.raises(PreflightError, match="valid_quick_path"):
            validate_training_prerequisites(cfg)

    def test_valid_full_missing_is_warning(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path, valid_full_exists=False)
        warnings = validate_training_prerequisites(cfg)
        assert any("valid_full_path" in w for w in warnings)

    def test_run_dir_not_writable_raises(self, tmp_path):
        run_dir = tmp_path / "readonly_runs"
        run_dir.mkdir()
        os.chmod(run_dir, 0o444)
        cfg = _make_baseline_cfg(tmp_path, run_dir=str(run_dir))
        try:
            with pytest.raises(PreflightError, match="not writable"):
                validate_training_prerequisites(cfg)
        finally:
            os.chmod(run_dir, 0o755)

    def test_max_seq_len_too_small_rejected_by_schema(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="max_seq_len"):
            DataConfig(
                train_path="a",
                valid_quick_path="b",
                valid_full_path="c",
                max_seq_len=16,
            )

    def test_multiple_errors_all_reported(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path, train_exists=False, valid_exists=False)
        with pytest.raises(PreflightError) as exc_info:
            validate_training_prerequisites(cfg)
        msg = str(exc_info.value)
        assert "train_path" in msg
        assert "valid_quick_path" in msg

    def test_schema_validation_then_preflight(self, tmp_path):
        """Verify the integration order: schema first, then preflight."""
        from src.training.config_schema import load_and_validate_config

        yaml_content = f"""
experiment:
  name: test
  seed: 42
model:
  name_or_path: test
lora:
  r: 16
  alpha: 32
  dropout: 0.05
data:
  train_path: {tmp_path / "train.jsonl"}
  valid_quick_path: {tmp_path / "valid_quick.jsonl"}
  valid_full_path: {tmp_path / "valid_full.jsonl"}
  max_seq_len: 2048
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2.0e-4
  max_steps: 100
logging:
  run_dir: {tmp_path / "runs"}
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        # Create data files so preflight passes
        (tmp_path / "train.jsonl").write_text("{}")
        (tmp_path / "valid_quick.jsonl").write_text("{}")
        (tmp_path / "valid_full.jsonl").write_text("{}")

        cfg = load_and_validate_config(yaml_path)
        warnings = validate_training_prerequisites(cfg)
        assert isinstance(warnings, list)


class TestValidateMaxSeqLen:
    def test_within_limit_no_error(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path, max_seq_len=2048)
        fake_tokenizer = type("Tok", (), {"model_max_length": 4096})()
        validate_max_seq_len(cfg, fake_tokenizer)

    def test_exceeds_limit_raises(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path, max_seq_len=8192)
        fake_tokenizer = type("Tok", (), {"model_max_length": 4096})()
        with pytest.raises(PreflightError, match="exceeds tokenizer"):
            validate_max_seq_len(cfg, fake_tokenizer)

    def test_no_model_max_length_attr_skips(self, tmp_path):
        cfg = _make_baseline_cfg(tmp_path, max_seq_len=8192)
        fake_tokenizer = type("Tok", (), {})()
        validate_max_seq_len(cfg, fake_tokenizer)

    def test_max_seq_len_defense_in_depth(self, tmp_path):
        """Preflight catches max_seq_len < 32 even if Pydantic is bypassed."""
        cfg = _make_baseline_cfg(tmp_path)
        cfg.data.max_seq_len = 10  # bypass Pydantic validation
        with pytest.raises(PreflightError, match="max_seq_len too small"):
            validate_training_prerequisites(cfg)

    def test_negative_lr_defense_in_depth(self, tmp_path):
        """Preflight catches learning_rate <= 0 even if Pydantic is bypassed."""
        cfg = _make_baseline_cfg(tmp_path)
        cfg.training.learning_rate = -1.0  # bypass Pydantic validation
        with pytest.raises(PreflightError, match="learning_rate must be positive"):
            validate_training_prerequisites(cfg)
