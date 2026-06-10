"""Config validation tests for accel adaptation experiment configs (TASK-0092)."""

from pathlib import Path

import pytest

from src.training.config_schema import TGLoRAConfig, load_and_validate_config

CONFIGS_DIR = Path("configs")

ACCEL_CONFIGS = {
    "conservative": {
        "path": CONFIGS_DIR / "9b_tg_lora_accel_conservative.yaml",
        "accel_instability_lr_decay": 0.3,
        "accel_convergence_lr_boost": 1.1,
    },
    "aggressive": {
        "path": CONFIGS_DIR / "9b_tg_lora_accel_aggressive.yaml",
        "accel_instability_lr_decay": 0.9,
        "accel_convergence_lr_boost": 2.0,
    },
    "balanced": {
        "path": CONFIGS_DIR / "9b_tg_lora_accel_balanced.yaml",
        "accel_instability_lr_decay": 0.5,
        "accel_convergence_lr_boost": 1.5,
    },
    "no_accel": {
        "path": CONFIGS_DIR / "9b_tg_lora_accel_no_accel.yaml",
        "accel_instability_lr_decay": 0.99,
        "accel_convergence_lr_boost": 1.01,
    },
}


@pytest.fixture(params=list(ACCEL_CONFIGS.keys()), ids=list(ACCEL_CONFIGS.keys()))
def accel_config(request):
    return ACCEL_CONFIGS[request.param]


@pytest.fixture(scope="module")
def loaded_configs():
    configs = {}
    for name, spec in ACCEL_CONFIGS.items():
        p = spec["path"]
        if not p.exists():
            pytest.skip(f"{p} not found")
        configs[name] = load_and_validate_config(p)
    return configs


class TestAccelConfigValidation:
    def test_config_loads_as_tg_lora(self, accel_config, loaded_configs):
        cfg = loaded_configs[accel_config if isinstance(accel_config, str) else next(k for k, v in ACCEL_CONFIGS.items() if v is accel_config)]
        assert isinstance(cfg, TGLoRAConfig)

    def test_accel_params_match_expected(self, loaded_configs):
        for name, spec in ACCEL_CONFIGS.items():
            cfg = loaded_configs[name]
            assert cfg.tg_lora.accel_instability_lr_decay == spec["accel_instability_lr_decay"], (
                f"{name}: expected decay={spec['accel_instability_lr_decay']}, got {cfg.tg_lora.accel_instability_lr_decay}"
            )
            assert cfg.tg_lora.accel_convergence_lr_boost == spec["accel_convergence_lr_boost"], (
                f"{name}: expected boost={spec['accel_convergence_lr_boost']}, got {cfg.tg_lora.accel_convergence_lr_boost}"
            )

    def test_accel_decay_in_valid_range(self, loaded_configs):
        for name, cfg in loaded_configs.items():
            decay = cfg.tg_lora.accel_instability_lr_decay
            assert 0.0 < decay < 1.0, f"{name}: decay {decay} not in (0, 1)"

    def test_accel_boost_gt_one(self, loaded_configs):
        for name, cfg in loaded_configs.items():
            boost = cfg.tg_lora.accel_convergence_lr_boost
            assert boost > 1.0, f"{name}: boost {boost} not > 1.0"


class TestAccelConfigFairComparison:
    def test_same_model_across_configs(self, loaded_configs):
        models = {name: cfg.model.name_or_path for name, cfg in loaded_configs.items()}
        assert len(set(models.values())) == 1, f"Models differ: {models}"

    def test_same_data_paths(self, loaded_configs):
        train_paths = {name: cfg.data.train_path for name, cfg in loaded_configs.items()}
        assert len(set(train_paths.values())) == 1, f"Train paths differ: {train_paths}"

    def test_same_lora_params(self, loaded_configs):
        for name, cfg in loaded_configs.items():
            assert cfg.lora.r == 16, f"{name}: lora.r differs"
            assert cfg.lora.alpha == 32, f"{name}: lora.alpha differs"
            assert cfg.lora.dropout == 0.05, f"{name}: lora.dropout differs"

    def test_same_training_budget(self, loaded_configs):
        budgets = {name: cfg.training.max_cycles for name, cfg in loaded_configs.items()}
        assert len(set(budgets.values())) == 1, f"Max cycles differ: {budgets}"

    def test_same_tg_lora_core_params(self, loaded_configs):
        for name, cfg in loaded_configs.items():
            assert cfg.tg_lora.K_initial == 3, f"{name}: K_initial differs"
            assert cfg.tg_lora.N_initial == 5, f"{name}: N_initial differs"
            assert cfg.tg_lora.alpha_initial == 0.3, f"{name}: alpha_initial differs"
            assert cfg.tg_lora.beta_initial == 0.8, f"{name}: beta_initial differs"

    def test_unique_experiment_names(self, loaded_configs):
        names = [cfg.experiment.name for cfg in loaded_configs.values()]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"
