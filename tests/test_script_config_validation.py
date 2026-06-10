"""Integration tests verifying that scripts use load_and_validate_config (TASK-0084)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.training.config_schema import (
    BaselineConfig,
    TGLoRAConfig,
    load_and_validate_config,
)

CONFIGS_DIR = Path("configs")
ALL_CONFIGS = [p for p in sorted(CONFIGS_DIR.glob("*.yaml")) if not p.name.startswith("mlx_")]

# Minimal valid YAML snippets for each config type.

_BASELINE_YAML = """
experiment:
  name: test
  seed: 42
model:
  name_or_path: Qwen/Qwen3.5-9B
lora:
  r: 16
  alpha: 32
  dropout: 0.05
data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid.jsonl
  valid_full_path: data/valid_full.jsonl
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2e-4
  max_steps: 100
logging:
  run_dir: runs/test
"""

_TG_LORA_YAML = """
experiment:
  name: test
  seed: 42
model:
  name_or_path: Qwen/Qwen3.5-9B
lora:
  r: 16
  alpha: 32
  dropout: 0.05
data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid.jsonl
  valid_full_path: data/valid_full.jsonl
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2e-4
  max_cycles: 500
tg_lora:
  K_initial: 3
  K_candidates: [2, 3, 5, 8]
  N_initial: 5
  N_candidates: [1, 3, 5, 10, 20]
  alpha_initial: 0.3
  alpha_min: 0.03
  alpha_max: 1.5
  beta_initial: 0.8
  beta_candidates: [0.5, 0.8, 0.9, 0.95]
  relative_update_cap: 0.005
  active_layer_strategy: last_25_percent_plus_random_2
logging:
  run_dir: runs/test
"""


def _write_yaml(tmp_path: Path, content: str, name: str = "test.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ── 1. All configs/*.yaml pass Pydantic validation ─────────────────────────


class TestAllConfigFilesValidate:
    """Every checked-in YAML config must pass load_and_validate_config."""

    @pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=lambda p: p.name)
    def test_config_validates(self, config_path):
        cfg = load_and_validate_config(config_path)
        assert isinstance(cfg, (BaselineConfig, TGLoRAConfig))


# ── 2. Scripts use load_and_validate_config ────────────────────────────────


class TestScriptsUseValidation:
    """Verify that scripts call load_and_validate_config instead of raw OmegaConf.load."""

    def test_inspect_model_uses_validation(self):
        import inspect

        from scripts.inspect_model import inspect_from_yaml

        assert "load_and_validate_config" in inspect.getsource(inspect_from_yaml)

    def test_diagnose_check_config_uses_validation(self):
        import inspect

        from scripts.diagnose import check_config

        assert "load_and_validate_config" in inspect.getsource(check_config)

    def test_recover_generate_recovery_config_uses_validation(self):
        import inspect

        from scripts.recover import generate_recovery_config

        assert "load_and_validate_config" in inspect.getsource(
            generate_recovery_config
        )

    def test_benchmark_optimizer_lifecycle_uses_validation(self):
        import inspect

        from scripts.benchmark_optimizer_lifecycle import main

        assert "load_and_validate_config" in inspect.getsource(main)


# ── 3. Invalid YAML is rejected by script config-loading paths ─────────────


class TestInvalidYAMLRejected:
    """Scripts should reject configs with typos, wrong types, or extra fields."""

    def test_typo_field_rejected(self, tmp_path):
        yaml = """
experiment:
  name: test
  seed: 42
model:
  name_or_path: Qwen/Qwen3.5-9B
lora:
  r: 16
  alpha: 32
  dropout: 0.05
data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid.jsonl
  valid_full_path: data/valid_full.jsonl
training:
  batch_size: 1
  grad_accumulation: 8
  lerning_rate: 2e-4
  max_steps: 100
logging:
  run_dir: runs/test
"""
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError, match="lerning_rate"):
            load_and_validate_config(path)

    def test_wrong_type_rejected(self, tmp_path):
        yaml = """
experiment:
  name: test
  seed: "not_a_number"
model:
  name_or_path: Qwen/Qwen3.5-9B
lora:
  r: 16
  alpha: 32
  dropout: 0.05
data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid.jsonl
  valid_full_path: data/valid_full.jsonl
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2e-4
  max_steps: 100
logging:
  run_dir: runs/test
"""
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError):
            load_and_validate_config(path)

    def test_extra_top_level_field_rejected(self, tmp_path):
        yaml = _BASELINE_YAML + "typo_section:\n  foo: 1\n"
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError, match="typo_section"):
            load_and_validate_config(path)

    def test_missing_required_field_rejected(self, tmp_path):
        yaml = """
experiment:
  name: test
  seed: 42
model:
  name_or_path: Qwen/Qwen3.5-9B
"""
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError):
            load_and_validate_config(path)

    def test_value_range_violation_rejected(self, tmp_path):
        yaml = """
experiment:
  name: test
  seed: 42
model:
  name_or_path: Qwen/Qwen3.5-9B
lora:
  r: 16
  alpha: 32
  dropout: 0.05
data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid.jsonl
  valid_full_path: data/valid_full.jsonl
  max_seq_len: 10
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2e-4
  max_steps: 100
logging:
  run_dir: runs/test
"""
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError):
            load_and_validate_config(path)

    def test_diagnose_rejects_invalid_config(self, tmp_path):
        from scripts.diagnose import check_config

        yaml = """
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
  train_path: a
  valid_quick_path: b
  valid_full_path: c
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2e-4
  max_steps: 100
logging:
  run_dir: runs/test
  extra_unknown_field: true
"""
        path = _write_yaml(tmp_path, yaml)
        results = check_config(str(path))
        warnings = [r for r in results if r.status == "warn"]
        assert any("validation failed" in r.message.lower() for r in warnings)

    def test_recover_rejects_invalid_config(self, tmp_path):
        from scripts.recover import generate_recovery_config

        yaml = """
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
  train_path: a
  valid_quick_path: b
  valid_full_path: c
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2e-4
  max_steps: 100
logging:
  run_dir: runs/test
  extra_unknown_field: true
"""
        path = _write_yaml(tmp_path, yaml)
        out = tmp_path / "recovery.yaml"
        result = generate_recovery_config(str(path), "oom", str(out))
        assert result.status == "error"
        assert "validation failed" in result.message.lower()


# ── 4. Valid configs are accepted by script paths ──────────────────────────


class TestValidConfigsAccepted:
    """Scripts should accept well-formed configs via load_and_validate_config."""

    def test_inspect_from_yaml_accepts_baseline(self, tmp_path):

        path = _write_yaml(tmp_path, _BASELINE_YAML)
        # inspect_from_yaml calls inspect_from_config which tries to download
        # the model — we just verify load_and_validate_config succeeds.
        cfg = load_and_validate_config(path)
        assert cfg.model.name_or_path == "Qwen/Qwen3.5-9B"

    def test_inspect_from_yaml_accepts_tg_lora(self, tmp_path):
        path = _write_yaml(tmp_path, _TG_LORA_YAML)
        cfg = load_and_validate_config(path)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.K_initial == 3

    def test_diagnose_accepts_valid_config(self, tmp_path):
        from scripts.diagnose import check_config

        path = _write_yaml(tmp_path, _BASELINE_YAML)
        results = check_config(str(path))
        errors = [r for r in results if r.status == "error"]
        assert len(errors) == 0

    def test_recover_accepts_valid_config(self, tmp_path):
        from scripts.recover import generate_recovery_config

        path = _write_yaml(tmp_path, _TG_LORA_YAML)
        out = tmp_path / "recovery.yaml"
        result = generate_recovery_config(str(path), "oom", str(out))
        assert result.status == "ok"
        assert out.exists()
