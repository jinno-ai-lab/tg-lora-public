"""Tests for LR exploration config explicitness and pipeline (TASK-0087).

Verifies that lr_explore_prob/lr_log_sigma are explicitly declared in all
TG-LoRA config YAMLs and that the Config → Controller → Optimizer LR
pipeline works correctly.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tg_lora.random_walk_controller import RandomWalkController
from src.training.config_schema import (
    TGLoRAConfig,
    load_and_validate_config,
)

CONFIGS_DIR = Path("configs")

TG_LORA_CONFIGS = {
    "9b_tg_lora.yaml": {
        "lr_explore_prob": 0.0,
        "lr_log_sigma": 0.1,
        "enable_random_walk": False,
    },
    "9b_tg_lora_paper_poc.yaml": {
        "lr_explore_prob": 0.0,
        "lr_log_sigma": 0.1,
        "enable_random_walk": False,
    },
    "9b_tg_lora_adaptive_k5.yaml": {
        "lr_explore_prob": 0.3,
        "lr_log_sigma": 0.1,
        "enable_random_walk": True,
    },
    "9b_tg_lora_adaptive_k5_no_conv.yaml": {
        "lr_explore_prob": 0.3,
        "lr_log_sigma": 0.1,
        "enable_random_walk": True,
    },
    "9b_tg_lora_optimizer_reuse_experimental.yaml": {
        "lr_explore_prob": 0.0,
        "lr_log_sigma": 0.1,
        "enable_random_walk": False,
    },
    "9b_tg_lora_prefix_feature_cache_experimental.yaml": {
        "lr_explore_prob": 0.3,
        "lr_log_sigma": 0.1,
        "enable_random_walk": True,
    },
    "9b_tg_lora_prefix_feature_cache_async.yaml": {
        "lr_explore_prob": 0.3,
        "lr_log_sigma": 0.1,
        "enable_random_walk": True,
    },
}

BASELINE_CONFIGS = [
    "9b_baseline.yaml",
    "9b_baseline_suffix_only_last25.yaml",
]


def _load_raw_yaml(filename: str) -> dict:
    path = CONFIGS_DIR / filename
    raw = OmegaConf.load(path)
    return OmegaConf.to_container(raw, resolve=True)


# ── Criterion 1: lr_explore_prob/lr_log_sigma explicitly present ────────


class TestLRExploreFieldsExplicit:
    """All TG-LoRA YAML files must explicitly contain lr_explore_prob and
    lr_log_sigma keys in the tg_lora section (no implicit schema defaults)."""

    @pytest.mark.parametrize("filename", TG_LORA_CONFIGS.keys())
    def test_lr_explore_prob_present_in_yaml(self, filename: str):
        raw = _load_raw_yaml(filename)
        tg = raw["tg_lora"]
        assert "lr_explore_prob" in tg, (
            f"{filename}: lr_explore_prob not explicitly declared"
        )

    @pytest.mark.parametrize("filename", TG_LORA_CONFIGS.keys())
    def test_lr_log_sigma_present_in_yaml(self, filename: str):
        raw = _load_raw_yaml(filename)
        tg = raw["tg_lora"]
        assert "lr_log_sigma" in tg, (
            f"{filename}: lr_log_sigma not explicitly declared"
        )


# ── Criterion 2 & 3: lr_explore_prob consistency with enable_random_walk ─


class TestLRExploreProbConsistency:
    """lr_explore_prob must be 0.0 when random walk is disabled, and a
    positive value when random walk is enabled."""

    @pytest.mark.parametrize(
        "filename,expected",
        [(k, v) for k, v in TG_LORA_CONFIGS.items() if not v["enable_random_walk"]],
    )
    def test_no_random_walk_lr_explore_prob_is_zero(
        self, filename: str, expected: dict
    ):
        cfg = load_and_validate_config(CONFIGS_DIR / filename)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.lr_explore_prob == expected["lr_explore_prob"]

    @pytest.mark.parametrize(
        "filename,expected",
        [(k, v) for k, v in TG_LORA_CONFIGS.items() if v["enable_random_walk"]],
    )
    def test_random_walk_lr_explore_prob_positive(
        self, filename: str, expected: dict
    ):
        cfg = load_and_validate_config(CONFIGS_DIR / filename)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.lr_explore_prob > 0.0
        assert cfg.tg_lora.lr_explore_prob == expected["lr_explore_prob"]


# ── Criterion 4: All configs pass Pydantic validation ────────────────────


class TestAllConfigsPydanticValidation:
    """Every config YAML (including baselines) must load without error."""

    ALL_CONFIGS = list(TG_LORA_CONFIGS.keys()) + BASELINE_CONFIGS

    @pytest.mark.parametrize("filename", ALL_CONFIGS)
    def test_config_loads_and_validates(self, filename: str):
        path = CONFIGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        cfg = load_and_validate_config(path)
        assert cfg is not None

    @pytest.mark.parametrize("filename", TG_LORA_CONFIGS.keys())
    def test_tg_lora_params_reflect_lr_explore_values(self, filename: str):
        expected = TG_LORA_CONFIGS[filename]
        cfg = load_and_validate_config(CONFIGS_DIR / filename)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.lr_explore_prob == expected["lr_explore_prob"]
        assert cfg.tg_lora.lr_log_sigma == expected["lr_log_sigma"]


# ── Criterion 5: Config → Controller → Optimizer LR pipeline ─────────────


def _make_controller_from_config(filename: str) -> RandomWalkController:
    """Construct a RandomWalkController exactly as train_tg_lora.py does."""
    cfg = load_and_validate_config(CONFIGS_DIR / filename)
    assert isinstance(cfg, TGLoRAConfig)
    tg = cfg.tg_lora
    return RandomWalkController(
        K_initial=tg.K_initial,
        K_candidates=tg.K_candidates,
        N_initial=tg.N_initial,
        N_candidates=tg.N_candidates,
        alpha_initial=tg.alpha_initial,
        alpha_min=tg.alpha_min,
        alpha_max=tg.alpha_max,
        alpha_log_sigma=tg.alpha_log_sigma,
        beta_initial=tg.beta_initial,
        beta_candidates=tg.beta_candidates,
        lr_initial=tg.lr_initial,
        lr_min=tg.lr_min,
        lr_max=tg.lr_max,
        lr_accept_boost=tg.lr_accept_boost,
        lr_reject_decay=tg.lr_reject_decay,
        active_layer_strategy=tg.active_layer_strategy,
        relative_update_cap=tg.relative_update_cap,
        rollback_tolerance=cfg.eval.rollback_tolerance,
        enable_random_walk=tg.enable_random_walk,
        enable_convergence_adaptation=tg.enable_convergence_adaptation,
        k_explore_prob=tg.k_explore_prob,
        n_explore_prob=tg.n_explore_prob,
        beta_explore_prob=tg.beta_explore_prob,
        strategy_explore_prob=tg.strategy_explore_prob,
        lr_explore_prob=tg.lr_explore_prob,
        lr_log_sigma=tg.lr_log_sigma,
    )


class TestConfigToControllerLRPipeline:
    """Verify that LR values flow correctly from config YAML through
    Pydantic validation into the RandomWalkController."""

    @pytest.mark.parametrize("filename", TG_LORA_CONFIGS.keys())
    def test_controller_initial_lr_matches_config(self, filename: str):
        cfg = load_and_validate_config(CONFIGS_DIR / filename)
        assert isinstance(cfg, TGLoRAConfig)
        ctrl = _make_controller_from_config(filename)
        assert ctrl.state.lr == cfg.tg_lora.lr_initial

    @pytest.mark.parametrize("filename", TG_LORA_CONFIGS.keys())
    def test_controller_lr_explore_prob_matches_config(self, filename: str):
        expected = TG_LORA_CONFIGS[filename]
        ctrl = _make_controller_from_config(filename)
        assert ctrl.lr_explore_prob == expected["lr_explore_prob"]

    @pytest.mark.parametrize("filename", TG_LORA_CONFIGS.keys())
    def test_controller_lr_log_sigma_matches_config(self, filename: str):
        expected = TG_LORA_CONFIGS[filename]
        ctrl = _make_controller_from_config(filename)
        assert ctrl.lr_log_sigma == expected["lr_log_sigma"]

    @pytest.mark.parametrize(
        "filename",
        [k for k, v in TG_LORA_CONFIGS.items() if not v["enable_random_walk"]],
    )
    def test_no_random_walk_propose_preserves_lr(self, filename: str):
        ctrl = _make_controller_from_config(filename)
        initial_lr = ctrl.state.lr
        for _ in range(20):
            proposal = ctrl.propose()
            assert proposal.lr == initial_lr

    @pytest.mark.parametrize(
        "filename",
        [k for k, v in TG_LORA_CONFIGS.items() if v["enable_random_walk"]],
    )
    def test_random_walk_propose_lr_within_bounds(self, filename: str):
        cfg = load_and_validate_config(CONFIGS_DIR / filename)
        assert isinstance(cfg, TGLoRAConfig)
        ctrl = _make_controller_from_config(filename)
        for _ in range(100):
            proposal = ctrl.propose()
            assert cfg.tg_lora.lr_min <= proposal.lr <= cfg.tg_lora.lr_max

    @pytest.mark.parametrize(
        "filename",
        [k for k, v in TG_LORA_CONFIGS.items() if v["enable_random_walk"]],
    )
    def test_random_walk_propose_lr_actually_explores(self, filename: str):
        """With lr_explore_prob > 0, repeated propose() calls should
        produce at least one lr different from the initial value."""
        ctrl = _make_controller_from_config(filename)
        initial_lr = ctrl.state.lr
        saw_different = False
        for _ in range(200):
            proposal = ctrl.propose()
            if proposal.lr != initial_lr:
                saw_different = True
                break
        assert saw_different, (
            f"{filename}: lr never changed in 200 propose() calls"
        )

    @pytest.mark.parametrize("filename", TG_LORA_CONFIGS.keys())
    def test_proposal_lr_feeds_optimizer(self, filename: str):
        """Simulate the controller → optimizer lr handoff that
        train_tg_lora.py performs each cycle."""
        ctrl = _make_controller_from_config(filename)
        proposal = ctrl.propose()
        lr_for_optimizer = proposal.lr
        cfg = load_and_validate_config(CONFIGS_DIR / filename)
        assert isinstance(cfg, TGLoRAConfig)
        assert cfg.tg_lora.lr_min <= lr_for_optimizer <= cfg.tg_lora.lr_max
