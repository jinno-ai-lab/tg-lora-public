from __future__ import annotations

from pathlib import Path

import pytest

from tg_lora.config import TGLoraConfig

# ---------------------------------------------------------------------------
# Default initialization
# ---------------------------------------------------------------------------


def test_default_initialization():
    cfg = TGLoraConfig()
    assert cfg.model_name == "Qwen/Qwen2.5-0.5B"
    assert cfg.adapter_rank == 8
    assert cfg.adapter_alpha == 16
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.lr == 1e-4
    assert cfg.batch_size == 4
    assert cfg.grad_accum == 1
    assert cfg.max_steps == 1000
    assert cfg.warmup_steps == 100
    assert cfg.K_initial == 3
    assert cfg.N_initial == 5
    assert cfg.alpha_initial == 1.0
    assert cfg.beta_initial == 0.9
    assert cfg.relative_update_cap == 0.5
    assert cfg.rollback_tolerance == 0.0
    assert cfg.active_layer_strategy == "last_25_percent_plus_random_2"
    assert cfg.trainable_lora_scope == "all"
    assert cfg.output_dir == "runs/tg_lora"
    assert cfg.log_every == 10
    assert cfg.eval_every == 100
    assert cfg.save_every == 500


# ---------------------------------------------------------------------------
# Validation: positive integer parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,bad_value", [
    ("adapter_rank", 0),
    ("adapter_rank", -1),
    ("adapter_alpha", 0),
    ("adapter_alpha", -1),
    ("batch_size", 0),
    ("batch_size", -1),
    ("grad_accum", 0),
    ("grad_accum", -1),
    ("max_steps", 0),
    ("max_steps", -1),
    ("K_initial", 0),
    ("K_initial", -1),
    ("N_initial", 0),
    ("N_initial", -1),
])
def test_reject_nonpositive_integers(field, bad_value):
    with pytest.raises(ValueError):
        TGLoraConfig(**{field: bad_value})


# ---------------------------------------------------------------------------
# Validation: lr > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_lr", [0.0, -1e-4])
def test_reject_nonpositive_lr(bad_lr):
    with pytest.raises(ValueError, match="lr"):
        TGLoraConfig(lr=bad_lr)


# ---------------------------------------------------------------------------
# Validation: alpha_initial in (0, 2.0]
# ---------------------------------------------------------------------------


def test_reject_alpha_initial_zero():
    with pytest.raises(ValueError, match="alpha_initial"):
        TGLoraConfig(alpha_initial=0.0)


def test_reject_alpha_initial_negative():
    with pytest.raises(ValueError, match="alpha_initial"):
        TGLoraConfig(alpha_initial=-0.1)


def test_reject_alpha_initial_above_two():
    with pytest.raises(ValueError, match="alpha_initial"):
        TGLoraConfig(alpha_initial=2.1)


def test_accept_alpha_initial_at_two():
    TGLoraConfig(alpha_initial=2.0)


# ---------------------------------------------------------------------------
# Validation: beta_initial in (0, 1.0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_beta", [0.0, -0.1, 1.0, 1.5])
def test_reject_invalid_beta(bad_beta):
    with pytest.raises(ValueError, match="beta_initial"):
        TGLoraConfig(beta_initial=bad_beta)


# ---------------------------------------------------------------------------
# Validation: relative_update_cap in (0, 1.0]
# ---------------------------------------------------------------------------


def test_reject_relative_update_cap_zero():
    with pytest.raises(ValueError, match="relative_update_cap"):
        TGLoraConfig(relative_update_cap=0.0)


def test_reject_relative_update_cap_negative():
    with pytest.raises(ValueError, match="relative_update_cap"):
        TGLoraConfig(relative_update_cap=-0.1)


def test_reject_relative_update_cap_above_one():
    with pytest.raises(ValueError, match="relative_update_cap"):
        TGLoraConfig(relative_update_cap=1.1)


def test_accept_relative_update_cap_at_one():
    TGLoraConfig(relative_update_cap=1.0)


# ---------------------------------------------------------------------------
# Validation: rollback_tolerance >= 0
# ---------------------------------------------------------------------------


def test_reject_negative_rollback_tolerance():
    with pytest.raises(ValueError, match="rollback_tolerance"):
        TGLoraConfig(rollback_tolerance=-0.01)


def test_accept_zero_rollback_tolerance():
    TGLoraConfig(rollback_tolerance=0.0)


# ---------------------------------------------------------------------------
# Validation: active_layer_strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valid_strategy", [
    "last_25_percent",
    "last_25_percent_plus_random_2",
    "middle_random",
    "lisa_like_weighted",
])
def test_accept_valid_strategies(valid_strategy):
    TGLoraConfig(active_layer_strategy=valid_strategy)


def test_reject_invalid_strategy():
    with pytest.raises(ValueError, match="active_layer_strategy"):
        TGLoraConfig(active_layer_strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# Validation: trainable_lora_scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valid_scope", ["all", "last_25_percent"])
def test_accept_valid_lora_scope(valid_scope):
    TGLoraConfig(trainable_lora_scope=valid_scope)


def test_reject_invalid_lora_scope():
    with pytest.raises(ValueError, match="trainable_lora_scope"):
        TGLoraConfig(trainable_lora_scope="middle")


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_returns_all_fields():
    cfg = TGLoraConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["model_name"] == "Qwen/Qwen2.5-0.5B"
    assert d["adapter_rank"] == 8
    assert d["adapter_alpha"] == 16
    assert d["target_modules"] == ["q_proj", "v_proj"]
    assert d["lr"] == 1e-4
    assert d["batch_size"] == 4
    assert d["grad_accum"] == 1
    assert d["max_steps"] == 1000
    assert d["warmup_steps"] == 100
    assert d["K_initial"] == 3
    assert d["N_initial"] == 5
    assert d["alpha_initial"] == 1.0
    assert d["beta_initial"] == 0.9
    assert d["relative_update_cap"] == 0.5
    assert d["rollback_tolerance"] == 0.0
    assert d["active_layer_strategy"] == "last_25_percent_plus_random_2"
    assert d["trainable_lora_scope"] == "all"
    assert d["output_dir"] == "runs/tg_lora"
    assert d["log_every"] == 10
    assert d["eval_every"] == 100
    assert d["save_every"] == 500


def test_to_dict_custom_values():
    cfg = TGLoraConfig(adapter_rank=16, lr=5e-4, K_initial=5)
    d = cfg.to_dict()
    assert d["adapter_rank"] == 16
    assert d["lr"] == 5e-4
    assert d["K_initial"] == 5


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_is_string():
    cfg = TGLoraConfig()
    s = cfg.summary()
    assert isinstance(s, str)
    assert "TGLoraConfig" in s
    assert "model_name" in s
    assert "Qwen/Qwen2.5-0.5B" in s


def test_summary_contains_all_groups():
    cfg = TGLoraConfig()
    s = cfg.summary()
    assert "adapter_rank" in s
    assert "adapter_alpha" in s
    assert "target_modules" in s
    assert "lr" in s
    assert "batch_size" in s
    assert "K_initial" in s
    assert "N_initial" in s
    assert "alpha_initial" in s
    assert "beta_initial" in s
    assert "active_layer_strategy" in s
    assert "trainable_lora_scope" in s
    assert "output_dir" in s


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip(tmp_path: Path):
    cfg = TGLoraConfig(adapter_rank=32, lr=3e-4)
    path = tmp_path / "config.json"
    cfg.save_json(path)

    loaded = TGLoraConfig.from_json(path)
    assert loaded.adapter_rank == 32
    assert loaded.lr == 3e-4
    assert loaded.model_name == "Qwen/Qwen2.5-0.5B"
    assert loaded.target_modules == ["q_proj", "v_proj"]


def test_json_round_trip_preserves_all_fields(tmp_path: Path):
    cfg = TGLoraConfig()
    path = tmp_path / "config.json"
    cfg.save_json(path)
    loaded = TGLoraConfig.from_json(path)

    d_orig = cfg.to_dict()
    d_loaded = loaded.to_dict()
    assert d_orig == d_loaded


def test_from_json_with_string_path(tmp_path: Path):
    cfg = TGLoraConfig()
    path = str(tmp_path / "config.json")
    cfg.save_json(path)
    loaded = TGLoraConfig.from_json(path)
    assert loaded.model_name == cfg.model_name


def test_from_json_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TGLoraConfig.from_json(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


def test_yaml_round_trip(tmp_path: Path):
    cfg = TGLoraConfig(adapter_rank=16, beta_initial=0.95)
    path = tmp_path / "config.yaml"
    cfg.save_yaml(path)

    loaded = TGLoraConfig.from_yaml(path)
    assert loaded.adapter_rank == 16
    assert loaded.beta_initial == pytest.approx(0.95)
    assert loaded.model_name == "Qwen/Qwen2.5-0.5B"


def test_yaml_round_trip_preserves_all_fields(tmp_path: Path):
    cfg = TGLoraConfig()
    path = tmp_path / "config.yaml"
    cfg.save_yaml(path)
    loaded = TGLoraConfig.from_yaml(path)

    d_orig = cfg.to_dict()
    d_loaded = loaded.to_dict()
    assert d_orig == d_loaded


def test_from_yaml_with_string_path(tmp_path: Path):
    cfg = TGLoraConfig()
    path = str(tmp_path / "config.yaml")
    cfg.save_yaml(path)
    loaded = TGLoraConfig.from_yaml(path)
    assert loaded.model_name == cfg.model_name


def test_from_yaml_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TGLoraConfig.from_yaml(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Consistency with RandomWalkController defaults
# ---------------------------------------------------------------------------


def test_defaults_match_random_walk_controller():
    from tg_lora.random_walk_controller import RandomWalkController

    cfg = TGLoraConfig()
    ctrl = RandomWalkController()

    assert cfg.K_initial == ctrl.state.K
    assert cfg.N_initial == ctrl.state.N
    assert cfg.active_layer_strategy == ctrl.state.active_layer_strategy


# ---------------------------------------------------------------------------
# Custom values preserved through round-trip
# ---------------------------------------------------------------------------


def test_custom_config_json_round_trip(tmp_path: Path):
    cfg = TGLoraConfig(
        model_name="custom/model",
        adapter_rank=64,
        adapter_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lr=3e-4,
        batch_size=8,
        grad_accum=4,
        max_steps=5000,
        warmup_steps=200,
        K_initial=5,
        N_initial=10,
        alpha_initial=0.5,
        beta_initial=0.95,
        relative_update_cap=0.3,
        rollback_tolerance=0.01,
        active_layer_strategy="middle_random",
        trainable_lora_scope="last_25_percent",
        output_dir="custom/output",
        log_every=5,
        eval_every=50,
        save_every=250,
    )
    path = tmp_path / "custom.json"
    cfg.save_json(path)
    loaded = TGLoraConfig.from_json(path)

    assert loaded.model_name == "custom/model"
    assert loaded.adapter_rank == 64
    assert loaded.adapter_alpha == 128
    assert loaded.target_modules == ["q_proj", "v_proj", "k_proj"]
    assert loaded.lr == 3e-4
    assert loaded.batch_size == 8
    assert loaded.grad_accum == 4
    assert loaded.max_steps == 5000
    assert loaded.warmup_steps == 200
    assert loaded.K_initial == 5
    assert loaded.N_initial == 10
    assert loaded.alpha_initial == 0.5
    assert loaded.beta_initial == pytest.approx(0.95)
    assert loaded.relative_update_cap == pytest.approx(0.3)
    assert loaded.rollback_tolerance == pytest.approx(0.01)
    assert loaded.active_layer_strategy == "middle_random"
    assert loaded.trainable_lora_scope == "last_25_percent"
    assert loaded.output_dir == "custom/output"
    assert loaded.log_every == 5
    assert loaded.eval_every == 50
    assert loaded.save_every == 250


def test_custom_config_yaml_round_trip(tmp_path: Path):
    cfg = TGLoraConfig(
        model_name="custom/model-yaml",
        adapter_rank=32,
        K_initial=8,
        active_layer_strategy="lisa_like_weighted",
    )
    path = tmp_path / "custom.yaml"
    cfg.save_yaml(path)
    loaded = TGLoraConfig.from_yaml(path)

    assert loaded.model_name == "custom/model-yaml"
    assert loaded.adapter_rank == 32
    assert loaded.K_initial == 8
    assert loaded.active_layer_strategy == "lisa_like_weighted"
