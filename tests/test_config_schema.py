"""Tests for Pydantic config schema validation (TASK-0016)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.training.config_schema import (AlphaLineConfig, BaselineConfig, DataConfig, EvalConfig,
                                        ExperimentConfig, LoRAConfig,
                                        ModelConfig, TGLoRAConfig,
                                        TGLoRAParams, TrainingConfig,
                                        load_and_validate_config,
                                        validate_config_data)

# ── Fixtures ──────────────────────────────────────────────────────────────

BASELINE_YAML = """
experiment:
  name: qlora_9b_baseline
  seed: 42

model:
  name_or_path: Qwen/Qwen3.5-9B
  dtype: bfloat16
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  device_map: auto

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid_quick.jsonl
  valid_full_path: data/valid_full.jsonl
  gold_test_path: data/gold_test.jsonl
  max_seq_len: 2048

training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2.0e-4
  weight_decay: 0.0
  max_steps: 1500
  gradient_checkpointing: true
  max_grad_norm: 1.0
  warmup_steps: 50
  save_every_steps: 250

eval:
  quick_eval_examples: 64
  full_eval_every_steps: 250

logging:
  backend: mlflow
  log_every_steps: 10
  run_dir: runs/test_baseline
"""

TG_LORA_YAML = """
experiment:
  name: tg_lora_9b_mvp
  seed: 42

model:
  name_or_path: Qwen/Qwen3.5-9B
  dtype: bfloat16
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  device_map: auto

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid_quick.jsonl
  valid_full_path: data/valid_full.jsonl
  gold_test_path: data/gold_test.jsonl
  max_seq_len: 2048

training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2.0e-4
  weight_decay: 0.0
  max_cycles: 500
  gradient_checkpointing: true
  max_grad_norm: 1.0

tg_lora:
  K_initial: 3
  K_candidates: [2, 3, 5, 8]
  N_initial: 5
  N_candidates: [1, 3, 5, 10, 20]
  alpha_initial: 0.3
  alpha_min: 0.03
  alpha_max: 1.5
  alpha_log_sigma: 0.15
  beta_initial: 0.8
  beta_candidates: [0.5, 0.8, 0.9, 0.95]
  relative_update_cap: 0.005
  active_layer_strategy: last_25_percent_plus_random_2
  random_middle_layers: 2

eval:
  quick_eval_examples: 64
  full_eval_every_cycles: 10
  rollback_tolerance: 0.005

logging:
  backend: mlflow
  log_every_cycles: 1
  save_every_cycles: 25
  run_dir: runs/test_tg_lora
"""


def _write_yaml(tmp_path: Path, content: str, name: str = "test.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ── ExperimentConfig ──────────────────────────────────────────────────────


class TestExperimentConfig:
    def test_valid(self):
        cfg = ExperimentConfig(name="test", seed=42)
        assert cfg.name == "test"
        assert cfg.seed == 42

    def test_negative_seed_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", seed=-1)


# ── ModelConfig ───────────────────────────────────────────────────────────


class TestModelConfig:
    def test_valid_with_defaults(self):
        cfg = ModelConfig(name_or_path="Qwen/Qwen3.5-9B")
        assert cfg.dtype == "bfloat16"
        assert cfg.load_in_4bit is True
        assert cfg.device_map is None

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig()


# ── LoRAConfig ────────────────────────────────────────────────────────────


class TestLoRAConfig:
    def test_valid(self):
        cfg = LoRAConfig(r=16, alpha=32, dropout=0.05)
        assert cfg.r == 16

    def test_r_must_be_positive(self):
        with pytest.raises(ValidationError):
            LoRAConfig(r=0, alpha=32, dropout=0.05)

    def test_dropout_range(self):
        with pytest.raises(ValidationError):
            LoRAConfig(r=16, alpha=32, dropout=1.0)


# ── DataConfig ────────────────────────────────────────────────────────────


class TestDataConfig:
    def test_valid(self):
        cfg = DataConfig(
            train_path="a",
            valid_quick_path="b",
            valid_full_path="c",
        )
        assert cfg.max_seq_len == 2048

    def test_missing_train_path_rejected(self):
        with pytest.raises(ValidationError):
            DataConfig(valid_quick_path="b", valid_full_path="c")

    def test_max_seq_len_below_32_rejected(self):
        with pytest.raises(ValidationError):
            DataConfig(
                train_path="a",
                valid_quick_path="b",
                valid_full_path="c",
                max_seq_len=16,
            )

    def test_max_seq_len_31_rejected(self):
        with pytest.raises(ValidationError):
            DataConfig(
                train_path="a",
                valid_quick_path="b",
                valid_full_path="c",
                max_seq_len=31,
            )

    def test_max_seq_len_32_accepted(self):
        cfg = DataConfig(
            train_path="a",
            valid_quick_path="b",
            valid_full_path="c",
            max_seq_len=32,
        )
        assert cfg.max_seq_len == 32


# ── TrainingConfig ────────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_valid_with_max_steps(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_steps=1500,
        )
        assert cfg.max_steps == 1500

    def test_valid_with_max_cycles(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_cycles=500,
        )
        assert cfg.max_cycles == 500

    def test_zero_learning_rate_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=0.0,
                max_steps=100,
            )

    def test_schedule_type_default_is_linear(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=1,
            learning_rate=1e-4,
            max_steps=100,
        )
        assert cfg.schedule_type == "linear"

    def test_schedule_type_cosine_accepted(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=1,
            learning_rate=1e-4,
            max_steps=100,
            schedule_type="cosine",
        )
        assert cfg.schedule_type == "cosine"

    def test_schedule_type_invalid_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=1,
                learning_rate=1e-4,
                max_steps=100,
                schedule_type="polynomial",
            )


# ── EvalConfig ────────────────────────────────────────────────────────────


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.quick_eval_examples == 64
        assert cfg.rollback_tolerance == 0.005

    def test_negative_rollback_tolerance_rejected(self):
        with pytest.raises(ValidationError):
            EvalConfig(rollback_tolerance=-0.1)

    def test_save_predictions_field_removed(self):
        """save_predictions was removed from schema (TASK-0037)."""
        with pytest.raises(ValidationError, match="save_predictions"):
            EvalConfig(save_predictions=True)

    def test_backward_compat_ignores_save_predictions(self, tmp_path):
        """Old YAML with save_predictions should fail cleanly, not crash."""
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
eval:
  save_predictions: true
logging:
  run_dir: runs/test
"""
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError, match="save_predictions"):
            load_and_validate_config(path)


# ── TGLoRAParams ─────────────────────────────────────────────────────────


class TestTGLoRAParams:
    _BASE = dict(
        K_initial=3,
        K_candidates=[3],
        N_initial=5,
        N_candidates=[5],
        alpha_initial=0.3,
        alpha_min=0.03,
        alpha_max=1.5,
        beta_initial=0.8,
        beta_candidates=[0.8],
        relative_update_cap=0.005,
        active_layer_strategy="last_25_percent",
    )

    def test_valid(self):
        p = TGLoRAParams(
            K_initial=3,
            K_candidates=[2, 3, 5, 8],
            N_initial=5,
            N_candidates=[1, 3, 5, 10, 20],
            alpha_initial=0.3,
            alpha_min=0.03,
            alpha_max=1.5,
            beta_initial=0.8,
            beta_candidates=[0.5, 0.8, 0.9, 0.95],
            relative_update_cap=0.005,
            active_layer_strategy="last_25_percent_plus_random_2",
        )
        assert p.K_initial == 3

    def test_alpha_min_greater_than_max_rejected(self):
        with pytest.raises(ValidationError):
            TGLoRAParams(
                K_initial=3,
                K_candidates=[2, 3],
                N_initial=5,
                N_candidates=[1, 3],
                alpha_initial=0.3,
                alpha_min=1.5,
                alpha_max=0.03,
                beta_initial=0.8,
                beta_candidates=[0.5],
                relative_update_cap=0.005,
                active_layer_strategy="last_25_percent_plus_random_2",
            )

    def test_zero_K_rejected(self):
        with pytest.raises(ValidationError):
            TGLoRAParams(
                K_initial=0,
                K_candidates=[2],
                N_initial=1,
                N_candidates=[1],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=0.8,
                beta_candidates=[0.5],
                relative_update_cap=0.005,
                active_layer_strategy="last_25_percent",
            )

    def test_beta_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            TGLoRAParams(
                K_initial=3,
                K_candidates=[2],
                N_initial=5,
                N_candidates=[1],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=1.5,
                beta_candidates=[0.5],
                relative_update_cap=0.005,
                active_layer_strategy="last_25_percent",
            )

    def test_empty_K_candidates_rejected(self):
        with pytest.raises(ValidationError):
            TGLoRAParams(
                K_initial=3,
                K_candidates=[],
                N_initial=5,
                N_candidates=[1],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=0.8,
                beta_candidates=[0.5],
                relative_update_cap=0.005,
                active_layer_strategy="last_25_percent",
            )

    def test_relative_update_cap_range(self):
        with pytest.raises(ValidationError):
            TGLoRAParams(
                K_initial=3,
                K_candidates=[2],
                N_initial=5,
                N_candidates=[1],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=0.8,
                beta_candidates=[0.5],
                relative_update_cap=1.5,
                active_layer_strategy="last_25_percent",
            )

    # ── Cross-field: initial values must belong to candidate lists ─────

    @pytest.mark.parametrize("overrides,match", [
        (dict(K_initial=4, K_candidates=[2, 3, 5, 8], N_candidates=[1, 5], beta_candidates=[0.5, 0.8]), "K_initial"),
        (dict(N_initial=7, N_candidates=[1, 3, 5, 10]), "N_initial"),
        (dict(beta_initial=0.7, beta_candidates=[0.5, 0.8, 0.9]), "beta_initial"),
        (dict(alpha_initial=0.01), "alpha_initial"),
        (dict(alpha_initial=2.0), "alpha_initial"),
    ])
    def test_initial_value_out_of_range_rejected(self, overrides, match):
        with pytest.raises(ValidationError, match=match):
            TGLoRAParams(**{**self._BASE, **overrides})

    @pytest.mark.parametrize("overrides,expected", [
        (dict(alpha_initial=0.03, alpha_min=0.03), 0.03),
        (dict(alpha_initial=1.5, alpha_max=1.5), 1.5),
    ])
    def test_alpha_initial_at_boundary_accepted(self, overrides, expected):
        p = TGLoRAParams(**{**self._BASE, **overrides})
        assert p.alpha_initial == expected

    # ── Adaptive LR validation ────────────────────────────────────────

    def test_lr_defaults_valid(self):
        p = TGLoRAParams(
            K_initial=3,
            K_candidates=[2, 3],
            N_initial=3,
            N_candidates=[1, 3],
            alpha_initial=0.3,
            alpha_min=0.03,
            alpha_max=1.5,
            beta_initial=0.5,
            beta_candidates=[0.5],
            relative_update_cap=0.005,
            active_layer_strategy="last_25_percent",
        )
        assert p.lr_initial == 5e-4
        assert p.lr_min == 1e-5
        assert p.lr_max == 1e-3
        assert p.lr_accept_boost == 1.2
        assert p.lr_reject_decay == 0.5

    def test_lr_explicit_values_valid(self):
        p = TGLoRAParams(
            K_initial=3,
            K_candidates=[3],
            N_initial=5,
            N_candidates=[5],
            alpha_initial=0.3,
            alpha_min=0.03,
            alpha_max=1.5,
            beta_initial=0.8,
            beta_candidates=[0.8],
            relative_update_cap=0.005,
            active_layer_strategy="last_25_percent",
            lr_initial=0.0005,
            lr_min=1e-5,
            lr_max=0.001,
            lr_accept_boost=1.2,
            lr_reject_decay=0.5,
        )
        assert p.lr_initial == 0.0005

    @pytest.mark.parametrize("overrides", [
        dict(K_candidates=[2], N_candidates=[1], beta_candidates=[0.5], lr_min=0.01, lr_max=0.001),
        dict(K_candidates=[2], N_candidates=[1], beta_candidates=[0.5], lr_initial=1e-6, lr_min=1e-5, lr_max=1e-3),
        dict(K_candidates=[2], N_candidates=[1], beta_candidates=[0.5], lr_initial=0.01, lr_min=1e-5, lr_max=1e-3),
        dict(K_candidates=[2], N_candidates=[1], beta_candidates=[0.5], lr_initial=-0.001),
        dict(K_candidates=[2], N_candidates=[1], beta_candidates=[0.5], lr_accept_boost=0.8),
        dict(K_candidates=[2], N_candidates=[1], beta_candidates=[0.5], lr_reject_decay=1.5),
    ])
    def test_lr_validation_rejected(self, overrides):
        with pytest.raises(ValidationError):
            TGLoRAParams(**{**self._BASE, **overrides})

    # ── layer_sample_temperature ────────────────────────────────────────

    def test_layer_sample_temperature_default(self):
        p = TGLoRAParams(**self._BASE)
        assert p.layer_sample_temperature == 1.0

    def test_layer_sample_temperature_explicit(self):
        p = TGLoRAParams(**{**self._BASE, "active_layer_strategy": "lisa_like_weighted", "layer_sample_temperature": 0.5})
        assert p.layer_sample_temperature == 0.5

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_layer_sample_temperature_invalid_rejected(self, value):
        with pytest.raises(ValidationError):
            TGLoRAParams(**{**self._BASE, "layer_sample_temperature": value})

    def test_layer_sample_temperature_in_tg_lora_yaml(self):
        config_path = Path("configs/9b_tg_lora.yaml")
        if not config_path.exists():
            pytest.skip("configs/9b_tg_lora.yaml not found")
        cfg = load_and_validate_config(config_path)
        assert cfg.tg_lora.layer_sample_temperature == 1.0


# ── load_and_validate_config ──────────────────────────────────────────────


class TestLoadAndValidateConfig:
    def test_baseline_yaml_loads(self, tmp_path):
        path = _write_yaml(tmp_path, BASELINE_YAML)
        cfg = load_and_validate_config(path)
        assert isinstance(cfg, BaselineConfig)
        assert not isinstance(cfg, TGLoRAConfig)
        assert cfg.experiment.name == "qlora_9b_baseline"
        assert cfg.training.max_steps == 1500

    def test_tg_lora_yaml_loads(self, tmp_path):
        path = _write_yaml(tmp_path, TG_LORA_YAML)
        cfg = load_and_validate_config(path)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.K_initial == 3
        assert cfg.training.max_cycles == 500
        assert cfg.alpha_line.alpha_line_enabled is False


class TestAlphaLineConfig:
    def test_defaults_are_disabled_and_valid(self):
        cfg = AlphaLineConfig()
        assert cfg.alpha_line_enabled is False
        assert cfg.alpha_line_order == 0
        assert cfg.b_logical == 32
        assert cfg.b_heavy == 4
        assert cfg.b_light == 16
        assert cfg.alpha_line_max_consecutive_reject == 3
        assert cfg.alpha_line_finite_diff_eps == pytest.approx(1e-3)
        assert cfg.future_work_metrics_enabled is False
        assert cfg.future_work_internal_metrics_enabled is False

    def test_rejects_non_divisible_heavy_batch(self):
        with pytest.raises(ValidationError, match="b_logical must be divisible"):
            AlphaLineConfig(b_logical=30, b_heavy=8, b_light=10)

    def test_rejects_non_divisible_light_batch(self):
        with pytest.raises(ValidationError, match="b_logical must be divisible"):
            AlphaLineConfig(b_logical=32, b_heavy=4, b_light=12)

    def test_missing_field_raises(self, tmp_path):
        yaml = """
experiment:
  name: test
  seed: 42
model:
  name_or_path: test
"""
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError):
            load_and_validate_config(path)

    def test_wrong_type_raises(self, tmp_path):
        yaml = """
experiment:
  name: test
  seed: "not_a_number"
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
eval:
  {}
logging:
  run_dir: runs/test
"""
        path = _write_yaml(tmp_path, yaml)
        with pytest.raises(ValidationError):
            load_and_validate_config(path)

    def test_omegaconf_variable_resolved(self, tmp_path):
        yaml = """
experiment:
  name: my_experiment
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
eval:
  {}
logging:
  run_dir: runs/${experiment.name}
"""
        path = _write_yaml(tmp_path, yaml)
        cfg = load_and_validate_config(path)
        assert cfg.logging.run_dir == "runs/my_experiment"

    def test_actual_baseline_config_file(self):
        config_path = Path("configs/9b_baseline.yaml")
        if not config_path.exists():
            pytest.skip("configs/9b_baseline.yaml not found")
        cfg = load_and_validate_config(config_path)
        assert isinstance(cfg, BaselineConfig)
        assert cfg.experiment.name == "qlora_9b_baseline"

    def test_actual_tg_lora_config_file(self):
        config_path = Path("configs/9b_tg_lora.yaml")
        if not config_path.exists():
            pytest.skip("configs/9b_tg_lora.yaml not found")
        cfg = load_and_validate_config(config_path)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.K_initial == 3
        assert cfg.tg_lora.lr_initial == 0.0005
        assert cfg.tg_lora.lr_min == 1e-5
        assert cfg.tg_lora.lr_max == 0.001

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_and_validate_config("/nonexistent/path.yaml")

    def test_actual_psa_config_file(self):  # TC-EDGE-211
        config_path = Path("configs/9b_tg_lora_psa.yaml")
        if not config_path.exists():
            pytest.skip("configs/9b_tg_lora_psa.yaml not found")
        cfg = load_and_validate_config(config_path)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.enable_psa is True


# ── PSA gain validation (TC-EDGE-210) ──────────────────────────────────────


class TestPSAGainValidation:
    """psa_gain uses Field(ge=0.0): negatives are rejected, but 0.0 is allowed
    because the gamma sweep (TC-281-01) uses gamma=0.0 as the no-amplification
    baseline. This matches PSAPrior's own gain < 0 guard (TC-EDGE-203)."""

    @staticmethod
    def _base_kwargs() -> dict:
        return dict(
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=5,
            N_candidates=[1, 3, 5],
            alpha_initial=0.3,
            alpha_min=0.03,
            alpha_max=1.5,
            beta_initial=0.8,
            beta_candidates=[0.5, 0.8, 0.9],
            relative_update_cap=0.005,
            active_layer_strategy="last_25_percent",
        )

    def test_negative_psa_gain_rejected(self):  # TC-EDGE-210 (negative half)
        with pytest.raises(ValidationError):
            TGLoRAParams(**self._base_kwargs(), psa_gain=-0.1)

    def test_zero_psa_gain_accepted(self):  # TC-EDGE-210 (0.0 allowed)
        params = TGLoRAParams(**self._base_kwargs(), psa_gain=0.0)
        assert params.psa_gain == 0.0


# ── validate_config_data ──────────────────────────────────────────────────


class TestValidateConfigData:
    """Tests for the validate_config_data helper (post-override validation)."""

    def _make_tg_lora_dict(self, **tg_overrides) -> dict:
        base = {
            "experiment": {"name": "test", "seed": 42},
            "model": {"name_or_path": "test"},
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
            "data": {
                "train_path": "a",
                "valid_quick_path": "b",
                "valid_full_path": "c",
            },
            "training": {
                "batch_size": 1,
                "grad_accumulation": 8,
                "learning_rate": 2e-4,
                "max_cycles": 500,
            },
            "tg_lora": {
                "K_initial": 3,
                "K_candidates": [2, 3, 5, 8],
                "N_initial": 5,
                "N_candidates": [1, 3, 5, 10, 20],
                "alpha_initial": 0.3,
                "alpha_min": 0.03,
                "alpha_max": 1.5,
                "beta_initial": 0.8,
                "beta_candidates": [0.5, 0.8, 0.9, 0.95],
                "relative_update_cap": 0.005,
                "active_layer_strategy": "last_25_percent_plus_random_2",
            },
            "eval": {},
            "logging": {"run_dir": "runs/test"},
        }
        base["tg_lora"].update(tg_overrides)
        return base

    def test_valid_dict_passes(self):
        data = self._make_tg_lora_dict()
        cfg = validate_config_data(data)
        assert isinstance(cfg, TGLoRAConfig)

    def test_invalid_lr_accept_boost_caught(self):
        data = self._make_tg_lora_dict(lr_accept_boost=0.8)
        with pytest.raises(ValidationError):
            validate_config_data(data)

    def test_invalid_lr_reject_decay_caught(self):
        data = self._make_tg_lora_dict(lr_reject_decay=1.5)
        with pytest.raises(ValidationError):
            validate_config_data(data)

    def test_invalid_lr_range_caught(self):
        data = self._make_tg_lora_dict(lr_min=0.01, lr_max=0.001)
        with pytest.raises(ValidationError):
            validate_config_data(data)

    def test_baseline_dict_without_tg_lora(self):
        data = {
            "experiment": {"name": "test", "seed": 42},
            "model": {"name_or_path": "test"},
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
            "data": {
                "train_path": "a",
                "valid_quick_path": "b",
                "valid_full_path": "c",
            },
            "training": {
                "batch_size": 1,
                "grad_accumulation": 8,
                "learning_rate": 2e-4,
                "max_steps": 100,
            },
            "eval": {},
            "logging": {"run_dir": "runs/test"},
        }
        cfg = validate_config_data(data)
        assert isinstance(cfg, BaselineConfig)
        assert not isinstance(cfg, TGLoRAConfig)


# ── Integration: CLI override schema rejection ────────────────────────────


class TestCLIOverrideSchemaRejection:
    """End-to-end tests verifying that both training scripts reject
    invalid config values introduced via CLI --override flags."""

    @staticmethod
    def _touch_data_files(tmp_path: Path) -> dict[str, str]:
        """Create empty dummy data files and return paths for YAML interpolation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        paths = {}
        for key in (
            "train_path",
            "valid_quick_path",
            "valid_full_path",
            "gold_test_path",
        ):
            p = data_dir / f"{key}.jsonl"
            p.write_text("{}")
            paths[key] = str(p)
        return paths

    def _make_tg_lora_yaml(self, tmp_path: Path) -> Path:
        dp = self._touch_data_files(tmp_path)
        yaml = f"""
experiment:
  name: tg_lora_9b_mvp
  seed: 42

model:
  name_or_path: Qwen/Qwen3.5-9B
  dtype: bfloat16
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  device_map: auto

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

data:
  train_path: {dp["train_path"]}
  valid_quick_path: {dp["valid_quick_path"]}
  valid_full_path: {dp["valid_full_path"]}
  gold_test_path: {dp["gold_test_path"]}
  max_seq_len: 2048

training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2.0e-4
  weight_decay: 0.0
  max_cycles: 500
  gradient_checkpointing: true
  max_grad_norm: 1.0

tg_lora:
  K_initial: 3
  K_candidates: [2, 3, 5, 8]
  N_initial: 5
  N_candidates: [1, 3, 5, 10, 20]
  alpha_initial: 0.3
  alpha_min: 0.03
  alpha_max: 1.5
  alpha_log_sigma: 0.15
  beta_initial: 0.8
  beta_candidates: [0.5, 0.8, 0.9, 0.95]
  relative_update_cap: 0.005
  active_layer_strategy: last_25_percent_plus_random_2
  random_middle_layers: 2

eval:
  quick_eval_examples: 64
  full_eval_every_cycles: 10
  rollback_tolerance: 0.005

logging:
  backend: mlflow
  log_every_cycles: 1
  save_every_cycles: 25
  run_dir: {tmp_path}/runs/test_tg_lora
"""
        return _write_yaml(tmp_path, yaml, name="tg_lora.yaml")

    def _make_baseline_yaml(self, tmp_path: Path) -> Path:
        dp = self._touch_data_files(tmp_path)
        yaml = f"""
experiment:
  name: qlora_9b_baseline
  seed: 42

model:
  name_or_path: Qwen/Qwen3.5-9B
  dtype: bfloat16
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  device_map: auto

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

data:
  train_path: {dp["train_path"]}
  valid_quick_path: {dp["valid_quick_path"]}
  valid_full_path: {dp["valid_full_path"]}
  gold_test_path: {dp["gold_test_path"]}
  max_seq_len: 2048

training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2.0e-4
  weight_decay: 0.0
  max_steps: 1500
  gradient_checkpointing: true
  max_grad_norm: 1.0
  warmup_steps: 50
  save_every_steps: 250

eval:
  quick_eval_examples: 64
  full_eval_every_steps: 250

logging:
  backend: mlflow
  log_every_steps: 10
  run_dir: {tmp_path}/runs/test_baseline
"""
        return _write_yaml(tmp_path, yaml, name="baseline.yaml")

    # -- train_tg_lora.py --

    def test_tg_lora_rejects_invalid_lr_accept_boost_override(self, tmp_path):
        """If --override sets lr_accept_boost <= 1, the post-override
        validation in train_tg_lora.main() must reject it."""
        import subprocess
        import sys

        config_path = self._make_tg_lora_yaml(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.training.train_tg_lora",
                "--config",
                str(config_path),
                "--override",
                "tg_lora.lr_accept_boost=0.5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "validation error" in result.stderr.lower()

    def test_tg_lora_rejects_negative_lr_override(self, tmp_path):
        import subprocess
        import sys

        config_path = self._make_tg_lora_yaml(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.training.train_tg_lora",
                "--config",
                str(config_path),
                "--override",
                "training.learning_rate=-0.001",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "validation error" in result.stderr.lower()

    # -- train_baseline_qlora.py --

    def test_baseline_rejects_negative_lr_override(self, tmp_path):
        """If --override sets a negative learning_rate, the post-override
        validation in train_baseline_qlora.main() must reject it."""
        import subprocess
        import sys

        config_path = self._make_baseline_yaml(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.training.train_baseline_qlora",
                "--config",
                str(config_path),
                "--override",
                "training.learning_rate=-0.001",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "validation error" in result.stderr.lower()

    def test_baseline_rejects_zero_seq_len_override(self, tmp_path):
        import subprocess
        import sys

        config_path = self._make_baseline_yaml(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.training.train_baseline_qlora",
                "--config",
                str(config_path),
                "--override",
                "data.max_seq_len=0",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "validation error" in result.stderr.lower()


# ── ActiveLayerStrategy enum validation ───────────────────────────────────


class TestActiveLayerStrategy:
    def test_valid_strategies_accepted(self):
        for strategy in [
            "last_25_percent",
            "last_25_percent_plus_random_2",
            "middle_random",
            "lisa_like_weighted",
        ]:
            p = TGLoRAParams(
                K_initial=3,
                K_candidates=[3],
                N_initial=5,
                N_candidates=[5],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=0.8,
                beta_candidates=[0.8],
                relative_update_cap=0.005,
                active_layer_strategy=strategy,
            )
            assert p.active_layer_strategy == strategy

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValidationError):
            TGLoRAParams(
                K_initial=3,
                K_candidates=[3],
                N_initial=5,
                N_candidates=[5],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=0.8,
                beta_candidates=[0.8],
                relative_update_cap=0.005,
                active_layer_strategy="invalid_strategy",
            )

    def test_typo_strategy_rejected(self):
        with pytest.raises(ValidationError):
            TGLoRAParams(
                K_initial=3,
                K_candidates=[3],
                N_initial=5,
                N_candidates=[5],
                alpha_initial=0.3,
                alpha_min=0.03,
                alpha_max=1.5,
                beta_initial=0.8,
                beta_candidates=[0.8],
                relative_update_cap=0.005,
                active_layer_strategy="last_25_percent_plus_random",
            )


# ── BnbQuantType validation ───────────────────────────────────────────────


class TestBnbQuantType:
    def test_nf4_accepted(self):
        cfg = ModelConfig(name_or_path="test", bnb_4bit_quant_type="nf4")
        assert cfg.bnb_4bit_quant_type == "nf4"

    def test_fp4_accepted(self):
        cfg = ModelConfig(name_or_path="test", bnb_4bit_quant_type="fp4")
        assert cfg.bnb_4bit_quant_type == "fp4"

    def test_invalid_quant_type_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(name_or_path="test", bnb_4bit_quant_type="int8")


# ── ModelConfig.device field ─────────────────────────────────────────────


class TestModelConfigDevice:
    def test_device_optional(self):
        cfg = ModelConfig(name_or_path="test")
        assert cfg.device is None

    def test_device_accepted(self):
        cfg = ModelConfig(name_or_path="test", device="cuda:0")
        assert cfg.device == "cuda:0"

    def test_device_cpu(self):
        cfg = ModelConfig(name_or_path="test", device="cpu")
        assert cfg.device == "cpu"


# ── TrainingConfig budget validator ──────────────────────────────────────


class TestTrainingConfigBudget:
    def test_max_steps_only_valid(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_steps=100,
        )
        assert cfg.max_steps == 100

    def test_max_cycles_only_valid(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_cycles=500,
        )
        assert cfg.max_cycles == 500

    def test_both_budgets_valid(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_steps=100,
            max_cycles=500,
        )
        assert cfg.max_steps == 100
        assert cfg.max_cycles == 500

    def test_no_budget_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=2e-4,
            )


# ── TrainingConfig early stopping fields ─────────────────────────────────


class TestTrainingConfigEarlyStopping:
    def test_defaults(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_steps=100,
        )
        assert cfg.early_stopping_patience is None
        assert cfg.min_cycles_before_stop == 10
        assert cfg.min_steps_before_stop == 100

    def test_early_stopping_patience_set(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_steps=100,
            early_stopping_patience=5,
        )
        assert cfg.early_stopping_patience == 5

    def test_zero_patience_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=2e-4,
                max_steps=100,
                early_stopping_patience=0,
            )

    def test_negative_patience_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=2e-4,
                max_steps=100,
                early_stopping_patience=-1,
            )

    def test_min_steps_zero_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=2e-4,
                max_steps=100,
                min_steps_before_stop=0,
            )

    def test_min_cycles_zero_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=2e-4,
                max_steps=100,
                min_cycles_before_stop=0,
            )

    def test_early_stopping_params_in_baseline_yaml(self):
        config_path = Path("configs/9b_baseline.yaml")
        if not config_path.exists():
            pytest.skip("configs/9b_baseline.yaml not found")
        cfg = load_and_validate_config(config_path)
        assert cfg.training.early_stopping_patience is None
        assert cfg.training.min_steps_before_stop == 100

    def test_early_stopping_params_in_tg_lora_yaml(self):
        config_path = Path("configs/9b_tg_lora.yaml")
        if not config_path.exists():
            pytest.skip("configs/9b_tg_lora.yaml not found")
        cfg = load_and_validate_config(config_path)
        assert cfg.training.early_stopping_patience is None
        assert cfg.training.min_cycles_before_stop == 3
        assert cfg.training.min_steps_before_stop == 100


class TestTrainingConfigExperimentalScopes:
    def test_trainable_lora_scope_defaults_to_all(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_steps=100,
        )
        assert cfg.trainable_lora_scope == "all"

    def test_prefix_cache_loader_options_valid(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_cycles=10,
            trainable_lora_scope="last_25_percent",
            prefix_feature_cache_experimental=True,
            prefix_feature_cache_num_workers=2,
            prefix_feature_cache_persistent_workers=True,
            prefix_feature_cache_prefetch_factor=2,
        )
        assert cfg.prefix_feature_cache_num_workers == 2
        assert cfg.prefix_feature_cache_prefetch_factor == 2

    def test_prefetch_without_workers_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=2e-4,
                max_cycles=10,
                prefix_feature_cache_prefetch_factor=2,
            )

    def test_prefix_cache_dir_and_force_rebuild_fields(self):
        cfg = TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_cycles=10,
            prefix_feature_cache_dir=".cache/prefix",
            prefix_feature_cache_force_rebuild=True,
        )
        assert cfg.prefix_feature_cache_dir == ".cache/prefix"
        assert cfg.prefix_feature_cache_force_rebuild is True

    def test_prefix_cache_runtime_offload_requires_experimental(self):
        with pytest.raises(ValidationError, match="prefix_feature_cache_experimental"):
            TrainingConfig(
                batch_size=1,
                grad_accumulation=8,
                learning_rate=2e-4,
                max_cycles=10,
                prefix_feature_cache_offload_prefix_to_cpu=True,
            )

    def test_suffix_only_configs_load(self):
        for path_str in [
            "configs/9b_tg_lora_prefix_feature_cache_experimental.yaml",
            "configs/9b_tg_lora_prefix_feature_cache_async.yaml",
            "configs/9b_baseline_suffix_only_last25.yaml",
        ]:
            config_path = Path(path_str)
            if not config_path.exists():
                pytest.skip(f"{path_str} not found")
            cfg = load_and_validate_config(config_path)
            assert cfg.training.trainable_lora_scope == "last_25_percent"

    def test_async_cache_requires_device(self):
        with pytest.raises(ValidationError, match="async_device"):
            TrainingConfig(
                batch_size=1, grad_accumulation=1, learning_rate=1e-4,
                max_cycles=10,
                prefix_feature_cache_experimental=True,
                prefix_feature_cache_async=True,
                prefix_feature_cache_async_device=None,
            )

    def test_async_cache_requires_experimental(self):
        with pytest.raises(ValidationError, match="prefix_feature_cache_experimental"):
            TrainingConfig(
                batch_size=1, grad_accumulation=1, learning_rate=1e-4,
                max_cycles=10,
                prefix_feature_cache_async=True,
                prefix_feature_cache_async_device="cuda:1",
            )


# ── DtypeLiteral validation (REQ-058) ─────────────────────────────────────


class TestDtypeLiteral:
    def test_invalid_dtype_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(name_or_path="test", dtype="invalid_dtype")

    def test_invalid_bnb_compute_dtype_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(name_or_path="test", bnb_4bit_compute_dtype="test")

    def test_valid_dtypes_accepted(self):
        for dt in ("bfloat16", "float16", "float32"):
            cfg = ModelConfig(name_or_path="test", dtype=dt)
            assert cfg.dtype == dt

    def test_valid_bnb_compute_dtypes_accepted(self):
        for dt in ("bfloat16", "float16", "float32"):
            cfg = ModelConfig(name_or_path="test", bnb_4bit_compute_dtype=dt)
            assert cfg.bnb_4bit_compute_dtype == dt


# ── Extra-field rejection (RISK-0015/0016: YAML typo detection) ────────────


class TestExtraFieldsRejected:
    """Unknown keys in config dicts should raise ValidationError.

    This prevents silent misconfiguration from YAML typos (e.g.,
    'lerning_rate' instead of 'learning_rate').
    """

    @pytest.mark.parametrize("setup_fn,match", [
        (lambda d: (d["training"].__setitem__("lerning_rate", 1e-4), d["training"].__delitem__("learning_rate")), "lerning_rate"),
        (lambda d: (d["model"].__setitem__("name_or_pat", "Qwen/Qwen3.5-9B"), d["model"].__delitem__("name_or_path")), "name_or_pat"),
        (lambda d: (d["lora"].__setitem__("dropot", 0.05), d["lora"].__delitem__("dropout")), "dropot"),
    ])
    def test_typo_rejected(self, setup_fn, match):
        data = _baseline_dict()
        setup_fn(data)
        with pytest.raises(ValidationError, match=match):
            validate_config_data(data)

    @pytest.mark.parametrize("modifier,match", [
        (lambda d: d.__setitem__("typo_section", {"foo": 1}), "typo_section"),
        (lambda d: (d["tg_lora"].__setitem__("K_init", 3), d["tg_lora"].__delitem__("K_initial")), "K_init"),
        (lambda d: d.__setitem__("eval", {"quick_eval_examples": 64, "eval_every": 100}), "eval_every"),
        (lambda d: d["model"].__setitem__("quant_bits", 4), "quant_bits"),
    ])
    def test_extra_field_rejected(self, modifier, match):
        data = _baseline_dict() if "tg_lora" not in str(modifier.__code__.co_consts) else _tg_lora_dict()
        if "tg_lora" in str(match) or match == "K_init":
            data = _tg_lora_dict()
        else:
            data = _baseline_dict()
        modifier(data)
        with pytest.raises(ValidationError, match=match):
            validate_config_data(data)


# ── Malformed YAML defense ─────────────────────────────────────────────────


class TestMalformedYAML:
    """load_and_validate_config should raise on non-mapping YAML."""

    def test_empty_yaml_rejected(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValidationError):
            load_and_validate_config(p)

    def test_list_yaml_rejected(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2")
        with pytest.raises(ValueError, match="did not resolve"):
            load_and_validate_config(p)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _baseline_dict() -> dict:
    return {
        "experiment": {"name": "test", "seed": 42},
        "model": {"name_or_path": "Qwen/Qwen3.5-9B"},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "data": {
            "train_path": "data/train.jsonl",
            "valid_quick_path": "data/valid.jsonl",
            "valid_full_path": "data/valid_full.jsonl",
        },
        "training": {
            "batch_size": 1,
            "grad_accumulation": 8,
            "learning_rate": 2e-4,
            "max_steps": 1500,
        },
        "logging": {"backend": "mlflow", "run_dir": "runs/test"},
    }


def _tg_lora_dict() -> dict:
    d = _baseline_dict()
    del d["training"]["max_steps"]
    d["training"]["max_cycles"] = 500
    d["tg_lora"] = {
        "K_initial": 3,
        "K_candidates": [2, 3, 5, 8],
        "N_initial": 5,
        "N_candidates": [1, 3, 5, 10, 20],
        "alpha_initial": 0.3,
        "alpha_min": 0.03,
        "alpha_max": 1.5,
        "beta_initial": 0.8,
        "beta_candidates": [0.5, 0.8, 0.9, 0.95],
        "relative_update_cap": 0.005,
        "active_layer_strategy": "last_25_percent_plus_random_2",
    }
    return d


# --- accel param config validation ---


class TestAccelParamConfig:
    """Tests for accel_instability_lr_decay and accel_convergence_lr_boost fields."""

    def test_accel_params_default(self):
        from src.training.config_schema import TGLoRAParams
        params = TGLoRAParams(**_tg_lora_dict()["tg_lora"])
        assert params.accel_instability_lr_decay == 0.7
        assert params.accel_convergence_lr_boost == 1.1

    def test_accel_params_explicit(self):
        from src.training.config_schema import TGLoRAParams
        d = _tg_lora_dict()["tg_lora"]
        d["accel_instability_lr_decay"] = 0.5
        d["accel_convergence_lr_boost"] = 1.3
        params = TGLoRAParams(**d)
        assert params.accel_instability_lr_decay == 0.5
        assert params.accel_convergence_lr_boost == 1.3

    @pytest.mark.parametrize("field,value,match", [
        ("accel_instability_lr_decay", 0.0, "accel_instability_lr_decay"),
        ("accel_instability_lr_decay", 1.0, "accel_instability_lr_decay"),
        ("accel_instability_lr_decay", float("nan"), None),
        ("accel_instability_lr_decay", float("inf"), None),
        ("accel_convergence_lr_boost", 0.9, "accel_convergence_lr_boost"),
        ("accel_convergence_lr_boost", 1.0, "accel_convergence_lr_boost"),
        ("accel_convergence_lr_boost", float("nan"), None),
        ("accel_convergence_lr_boost", float("inf"), None),
    ])
    def test_accel_param_invalid_rejected(self, field, value, match):
        d = _tg_lora_dict()["tg_lora"]
        d[field] = value
        with pytest.raises(ValidationError, match=match):
            TGLoRAParams(**d)


class TestPhase56ConfigSurfaces:
    """REQ-225/231: Phase 56 config surface validation."""

    def test_one_shot_poc_config_loads(self):
        config_path = Path("configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml")
        if not config_path.exists():
            pytest.skip("one_shot_poc config not found")
        cfg = load_and_validate_config(config_path)
        assert cfg.training.prefix_feature_cache_experimental is True
        assert cfg.training.prefix_feature_cache_mode == "one_shot"

    def test_baseline_suffix_only_config_loads(self):
        config_path = Path("configs/9b_baseline_suffix_only_last25.yaml")
        if not config_path.exists():
            pytest.skip("baseline_suffix_only config not found")
        cfg = load_and_validate_config(config_path)
        assert cfg.training.trainable_lora_scope == "last_25_percent"
