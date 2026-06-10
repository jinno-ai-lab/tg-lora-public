"""Pydantic config schema validation for YAML training configs (TASK-0016).

Catches configuration errors (typos, missing fields, wrong types) before
expensive GPU training starts.
"""

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

ActiveLayerStrategy = Literal[
    "last_25_percent",
    "last_25_percent_plus_random_2",
    "middle_random",
    "lisa_like_weighted",
]

BnbQuantType = Literal["nf4", "fp4"]

DtypeLiteral = Literal["bfloat16", "float16", "float32"]

ScheduleType = Literal["linear", "cosine"]
TrainableLoraScope = Literal["all", "last_25_percent"]

OptimizerLifecyclePolicy = Literal[
    "recreate_per_cycle",
    "reuse_state_reset_experimental",
    "persistent",
]

PrefixFeatureCacheMode = Literal["reuse", "one_shot"]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    seed: int = Field(ge=0)
    paper_experiment: bool = False
    paper_experiment_id: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name_or_path: str
    dtype: DtypeLiteral = "bfloat16"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: BnbQuantType = "nf4"
    bnb_4bit_compute_dtype: DtypeLiteral = "bfloat16"
    device_map: str | dict | None = None
    device: str | None = None


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r: int = Field(ge=1)
    alpha: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: str | list[str] = "all-linear"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_path: str
    valid_quick_path: str
    valid_full_path: str
    gold_test_path: str = ""
    max_seq_len: int = Field(ge=32, default=2048)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(ge=1)
    grad_accumulation: int = Field(ge=1)
    train_on_prompt: bool = False
    deterministic_data_order: bool = True
    save_batch_plan_manifest: bool = True
    save_trajectory_delta_artifacts: bool = False
    trajectory_delta_artifact_interval: int = Field(default=1, ge=1)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0, default=0.0)
    max_steps: int | None = Field(default=None, ge=1)
    max_cycles: int | None = Field(default=None, ge=1)
    gradient_checkpointing: bool = True
    max_grad_norm: float = Field(gt=0.0, default=1.0)
    optimizer_lifecycle: OptimizerLifecyclePolicy = "recreate_per_cycle"
    trainable_lora_scope: TrainableLoraScope = "all"
    prefix_feature_cache_experimental: bool = False
    prefix_feature_cache_train: bool = True
    prefix_feature_cache_valid_quick: bool = True
    prefix_feature_cache_valid_full: bool = True
    prefix_feature_cache_num_workers: int = Field(default=0, ge=0)
    prefix_feature_cache_pin_memory: bool = False
    prefix_feature_cache_persistent_workers: bool = False
    prefix_feature_cache_prefetch_factor: int | None = Field(default=None, ge=1)
    prefix_feature_cache_dir: str = ".cache/prefix_feature_cache"
    prefix_feature_cache_force_rebuild: bool = False
    prefix_feature_cache_mode: PrefixFeatureCacheMode = "reuse"
    prefix_feature_cache_share_across_seeds: bool = False
    prefix_feature_cache_offload_prefix_to_cpu: bool = False
    prefix_feature_cache_async: bool = False
    prefix_feature_cache_async_device: str | None = None
    warmup_steps: int = Field(default=0, ge=0)
    save_every_steps: int = Field(default=250, ge=1)
    schedule_type: ScheduleType = "linear"
    early_stopping_patience: int | None = Field(default=None, ge=1)
    min_cycles_before_stop: int = Field(default=10, ge=1)
    min_steps_before_stop: int = Field(default=100, ge=1)
    measurement_noise_samples: int = Field(default=0, ge=0)
    measurement_save_per_step_deltas: bool = False

    @model_validator(mode="after")
    def budget_must_be_set(self) -> "TrainingConfig":
        if self.max_steps is None and self.max_cycles is None:
            raise ValueError(
                "At least one of max_steps or max_cycles must be specified"
            )
        return self

    @model_validator(mode="after")
    def prefix_cache_loader_params_valid(self) -> "TrainingConfig":
        if (
            self.prefix_feature_cache_prefetch_factor is not None
            and self.prefix_feature_cache_num_workers == 0
        ):
            raise ValueError(
                "prefix_feature_cache_prefetch_factor requires prefix_feature_cache_num_workers > 0"
            )
        if (
            self.prefix_feature_cache_persistent_workers
            and self.prefix_feature_cache_num_workers == 0
        ):
            raise ValueError(
                "prefix_feature_cache_persistent_workers requires prefix_feature_cache_num_workers > 0"
            )
        return self

    @model_validator(mode="after")
    def async_cache_valid(self) -> "TrainingConfig":
        if (
            self.prefix_feature_cache_async
            and not self.prefix_feature_cache_async_device
        ):
            raise ValueError(
                "prefix_feature_cache_async requires prefix_feature_cache_async_device"
            )
        if (
            self.prefix_feature_cache_async
            and not self.prefix_feature_cache_experimental
        ):
            raise ValueError(
                "prefix_feature_cache_async requires prefix_feature_cache_experimental"
            )
        return self

    @model_validator(mode="after")
    def prefix_runtime_offload_valid(self) -> "TrainingConfig":
        if (
            self.prefix_feature_cache_offload_prefix_to_cpu
            and not self.prefix_feature_cache_experimental
        ):
            raise ValueError(
                "prefix_feature_cache_offload_prefix_to_cpu requires prefix_feature_cache_experimental"
            )
        return self


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quick_eval_examples: int = Field(default=64, ge=1)
    accept_eval_examples: int | None = Field(default=None, ge=1)
    baseline_pretrain_reference_eval_enabled: bool = True
    full_eval_every_steps: int = Field(default=250, ge=1)
    full_eval_every_cycles: int = Field(default=10, ge=1)
    rollback_tolerance: float = Field(default=0.005, ge=0.0)
    moving_avg_window: int = Field(default=3, ge=1)
    soft_accept_temperature: float = Field(default=0.0, ge=0.0)
    eval_batch_size: int = Field(default=16, ge=1)


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    tracking_uri: str = ""
    experiment_name: str = ""


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = "mlflow"
    log_every_steps: int = Field(default=10, ge=1)
    log_every_cycles: int = Field(default=1, ge=1)
    save_every_cycles: int = Field(default=25, ge=1)
    run_dir: str
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


class TGLoRAParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    K_initial: int = Field(ge=1)
    K_candidates: list[int] = Field(min_length=1)
    N_initial: int = Field(ge=1)
    N_candidates: list[int] = Field(min_length=1)
    alpha_initial: float = Field(gt=0.0)
    alpha_min: float = Field(gt=0.0)
    alpha_max: float = Field(gt=0.0)
    alpha_log_sigma: float = Field(default=0.15, gt=0.0)
    beta_initial: float = Field(gt=0.0, lt=1.0)
    beta_candidates: list[float] = Field(min_length=1)
    relative_update_cap: float = Field(gt=0.0, lt=1.0)
    active_layer_strategy: ActiveLayerStrategy
    random_middle_layers: int = Field(default=2, ge=0)
    layer_sample_temperature: float = Field(default=1.0, gt=0.0)
    force_top_layers_only: bool = False
    enable_random_walk: bool = True
    enable_convergence_adaptation: bool = True

    lr_initial: float = Field(default=5e-4, gt=0.0)
    lr_min: float = Field(default=1e-5, gt=0.0)
    lr_max: float = Field(default=1e-3, gt=0.0)
    lr_accept_boost: float = Field(default=1.2, gt=1.0)
    lr_reject_decay: float = Field(default=0.5, gt=0.0, lt=1.0)

    # Exploration probabilities
    k_explore_prob: float = Field(default=0.4, ge=0.0, lt=1.0)
    n_explore_prob: float = Field(default=0.4, ge=0.0, lt=1.0)
    beta_explore_prob: float = Field(default=0.15, ge=0.0, lt=1.0)
    strategy_explore_prob: float = Field(default=0.08, ge=0.0, lt=1.0)
    lr_explore_prob: float = Field(default=0.3, ge=0.0, lt=1.0)
    lr_log_sigma: float = Field(default=0.1, gt=0.0)

    # Acceleration-based adaptation (adapt_to_acceleration)
    accel_instability_lr_decay: float = Field(default=0.7, gt=0.0, lt=1.0)
    accel_convergence_lr_boost: float = Field(default=1.1, gt=1.0, lt=100.0)

    # Confident-skip: auto-accept without eval when velocity direction is stable
    confident_skip_cos: float = Field(default=0.0, ge=0.0, le=1.0)
    confident_skip_min_cycles: int = Field(default=10, ge=1)

    # Probe-validation skip: remove fixed post-extrapolation eval cost when the
    # short/long EMA consistency indicates a stable dominant direction.
    validation_skip_enabled: bool = False
    validation_skip_high_cos: float = Field(default=0.85, ge=0.0, le=1.0)
    validation_skip_mid_cos: float = Field(default=0.70, ge=0.0, le=1.0)
    validation_skip_mid_eval_every: int = Field(default=3, ge=1)
    validation_skip_min_cycles: int = Field(default=0, ge=0)
    validation_skip_min_acceptance_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    validation_skip_force_eval_N: int = Field(default=20, ge=0)

    # Cosine-driven N selection: choose speculative horizon from short/long
    # lr-normalized velocity EMA consistency before extrapolation.
    cosine_n_selection_enabled: bool = False
    cosine_n_selection_short_window: int = Field(default=3, ge=1)
    cosine_n_selection_long_window: int = Field(default=10, ge=2)
    cosine_n_selection_thresholds: dict[int, float] = Field(
        default_factory=lambda: {
            1: 0.0,
            3: 0.70,
            5: 0.75,
            10: 0.80,
            20: 0.90,
        }
    )

    # Local linearity guard: when speculative behavior stops looking locally
    # linear, skip extrapolation and keep only the pilot update (baseline-like).
    linearity_guard_enabled: bool = True
    linearity_guard_warmup_cycles: int = Field(default=2, ge=0)
    linearity_guard_min_acceptance_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    linearity_guard_pilot_margin: float = Field(default=0.01, ge=0.0, lt=1.0)
    linearity_guard_max_positive_acceleration: float = Field(default=0.0)

    # Subspace zeroth-order steps: replace blind velocity addition with
    # forward-only line searches inside the short/long velocity subspace.
    subspace_zo_enabled: bool = False
    subspace_zo_tau_dim: float = Field(default=0.15, ge=0.0, le=1.0)
    subspace_zo_tau_cos: float = Field(default=0.70, ge=0.0, le=1.0)
    subspace_zo_mu_ratio: float = Field(default=0.001, gt=0.0)
    subspace_zo_eps_curv: float = Field(default=1e-8, gt=0.0)
    subspace_zo_eta_fallback_ratio: float = Field(default=1e-2, gt=0.0)
    subspace_zo_max_step_ratio: float = Field(default=0.02, gt=0.0)
    subspace_zo_max_steps_per_cycle: int = Field(default=10, ge=0)
    subspace_zo_force_dim: int = Field(default=0, ge=0, le=2)
    subspace_zo_disable_curvature: bool = False
    subspace_zo_stop_on_positive_g1: bool = True
    subspace_zo_g1_stop_epsilon: float = Field(default=0.0, ge=0.0)

    # Prior-based Subspace Learning (M9) parameters
    subspace_m9_enabled: bool = False
    subspace_m9_fd_eps: float = Field(default=1e-3, gt=0.0)
    subspace_m9_lr: float = Field(default=0.5, gt=0.0)
    subspace_m9_steps: int = Field(default=1, ge=1)
    warmup_release_cos: float = Field(default=0.75, ge=0.0, le=1.0)
    warmup_release_count: int = Field(default=0, ge=0)
    accept_after_sgd_steps: int = Field(default=0, ge=0)
    accept_after_sgd_lr: float | None = Field(default=None)
    shadow_extrapolation_enabled: bool = Field(default=False)

    # Prior-based Subspace Amplification (PSA)
    enable_psa: bool = False
    psa_history_length: int = Field(default=6, ge=2)
    psa_gain: float = Field(default=0.5, ge=0.0)
    psa_update_interval: int = Field(default=3, ge=1)
    psa_warmup_steps: int = Field(default=4, ge=0)
    psa_l2_reg: float = Field(default=0.01, ge=0.0)
    psa_regime_reset_enabled: bool = True
    psa_regime_window: int = Field(default=8, ge=3)
    psa_regime_plateau_eps: float = Field(default=1e-4, gt=0.0)
    psa_regime_transition_z: float = Field(default=2.0, gt=0.0)
    psa_regime_plateau_gain: float = Field(default=0.5, ge=0.0, le=1.0)

    # LAWA weight averaging baseline (GOAL §3.3)
    enable_lawa: bool = False
    lawa_window_size: int = Field(default=5, ge=2)
    lawa_start_cycle: int = Field(default=10, ge=0)

    # Activation-fingerprint regime inventory (GOAL §4 step 1)
    activation_regime_enabled: bool = False
    activation_regime_window: int = Field(default=10, ge=3)
    activation_regime_stable_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    activation_regime_chaotic_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    activation_regime_transition_drop_z: float = Field(default=2.0, gt=0.0)
    activation_regime_min_history: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def psa_m9_exclusive(self) -> "TGLoRAParams":
        if self.enable_psa and self.subspace_m9_enabled:
            raise ValueError("PSA (enable_psa) and M9 (subspace_m9_enabled) are mutually exclusive")
        return self

    @model_validator(mode="after")
    def alpha_range_valid(self) -> "TGLoRAParams":
        if self.alpha_min >= self.alpha_max:
            raise ValueError(
                f"alpha_min ({self.alpha_min}) must be less than alpha_max ({self.alpha_max})"
            )
        if not (self.alpha_min <= self.alpha_initial <= self.alpha_max):
            raise ValueError(
                f"alpha_initial ({self.alpha_initial}) must be between "
                f"alpha_min ({self.alpha_min}) and alpha_max ({self.alpha_max})"
            )
        return self

    @model_validator(mode="after")
    def initial_values_in_candidates(self) -> "TGLoRAParams":
        if self.K_initial not in self.K_candidates:
            raise ValueError(
                f"K_initial ({self.K_initial}) must be one of K_candidates ({self.K_candidates})"
            )
        if self.N_initial not in self.N_candidates:
            raise ValueError(
                f"N_initial ({self.N_initial}) must be one of N_candidates ({self.N_candidates})"
            )
        if self.beta_initial not in self.beta_candidates:
            raise ValueError(
                f"beta_initial ({self.beta_initial}) must be one of beta_candidates ({self.beta_candidates})"
            )
        return self

    @model_validator(mode="after")
    def lr_range_valid(self) -> "TGLoRAParams":
        if self.lr_min >= self.lr_max:
            raise ValueError(
                f"lr_min ({self.lr_min}) must be less than lr_max ({self.lr_max})"
            )
        if not (self.lr_min <= self.lr_initial <= self.lr_max):
            raise ValueError(
                f"lr_initial ({self.lr_initial}) must be between lr_min ({self.lr_min}) and lr_max ({self.lr_max})"
            )
        return self

    @model_validator(mode="after")
    def cosine_n_selection_valid(self) -> "TGLoRAParams":
        if self.cosine_n_selection_short_window >= self.cosine_n_selection_long_window:
            raise ValueError(
                "cosine_n_selection_short_window must be smaller than "
                "cosine_n_selection_long_window"
            )
        if not self.cosine_n_selection_thresholds:
            raise ValueError("cosine_n_selection_thresholds must not be empty")
        invalid_n = [n for n in self.cosine_n_selection_thresholds if n <= 0]
        if invalid_n:
            raise ValueError(f"cosine N thresholds must use positive N: {invalid_n}")
        invalid_c = [
            c
            for c in self.cosine_n_selection_thresholds.values()
            if not (0.0 <= c <= 1.0)
        ]
        if invalid_c:
            raise ValueError(f"cosine N thresholds must be in [0, 1]: {invalid_c}")
        return self

    @model_validator(mode="after")
    def validation_skip_thresholds_valid(self) -> "TGLoRAParams":
        if self.validation_skip_high_cos < self.validation_skip_mid_cos:
            raise ValueError(
                "validation_skip_high_cos must be greater than or equal to "
                "validation_skip_mid_cos"
            )
        return self


class AlphaLineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha_line_enabled: bool = False
    alpha_line_order: int = Field(default=0, ge=0, le=1)
    b_logical: int = Field(default=32, ge=1)
    b_heavy: int = Field(default=4, ge=1)
    b_light: int = Field(default=16, ge=1)
    m_alpha_steps: int = Field(default=19, ge=0)
    alpha_init: float = 0.0
    alpha_lr: float = Field(default=1e-2, gt=0.0)
    v_update_every: int = Field(default=1, ge=1)
    alpha_line_max_consecutive_reject: int = Field(default=3, ge=1)
    alpha_line_finite_diff_eps: float = Field(default=1e-3, gt=0.0)
    future_work_metrics_enabled: bool = False
    future_work_internal_metrics_enabled: bool = False

    @model_validator(mode="after")
    def batch_divisibility_valid(self) -> "AlphaLineConfig":
        if self.b_logical % self.b_heavy != 0:
            raise ValueError("b_logical must be divisible by b_heavy")
        if self.b_logical % self.b_light != 0:
            raise ValueError("b_logical must be divisible by b_light")
        return self


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: ExperimentConfig
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    training: TrainingConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig


class TGLoRAConfig(BaselineConfig):
    model_config = ConfigDict(extra="forbid")
    tg_lora: TGLoRAParams
    alpha_line: AlphaLineConfig = Field(default_factory=AlphaLineConfig)


def validate_config_data(data: dict) -> BaselineConfig | TGLoRAConfig:
    """Validate a raw config dict against the Pydantic schema.

    Use this when config values come from sources other than a file
    (e.g., after applying CLI overrides).
    """
    if "tg_lora" in data:
        return TGLoRAConfig(**data)
    return BaselineConfig(**data)


def load_and_validate_config(config_path: str | Path) -> BaselineConfig | TGLoRAConfig:
    """Load a YAML config and validate it against the Pydantic schema."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = OmegaConf.load(config_path)
    data = OmegaConf.to_container(raw, resolve=True)

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path} did not resolve to a mapping (got {type(data).__name__})"
        )

    if "experiment_plan" in data and "experiment" not in data:
        plan = data["experiment_plan"]
        if not isinstance(plan, dict):
            raise ValueError("experiment_plan must resolve to a mapping")
        referenced = plan.get("tg_config") or plan.get("baseline_config")
        if not isinstance(referenced, str) or not referenced:
            raise ValueError("experiment_plan must define tg_config or baseline_config")
        referenced_path = Path(referenced)
        if not referenced_path.is_absolute():
            parent_relative = config_path.parent / referenced_path
            referenced_path = (
                parent_relative
                if parent_relative.exists()
                else Path.cwd() / referenced_path
            )
        return load_and_validate_config(referenced_path)

    return validate_config_data(data)


def load_validate_and_build_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> tuple[BaselineConfig | TGLoRAConfig, DictConfig]:
    """Load config, apply CLI overrides, validate, and rebuild typed OmegaConf.

    OmegaConf dotlist parsing preserves scalar types from CLI overrides, while the
    Pydantic round-trip ensures the training loop receives fully validated values.
    """

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_cfg = OmegaConf.load(config_path)
    override_cfg = OmegaConf.from_dotlist(overrides or [])
    merged_cfg = OmegaConf.merge(base_cfg, override_cfg)
    data = OmegaConf.to_container(merged_cfg, resolve=True)

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path} did not resolve to a mapping (got {type(data).__name__})"
        )

    validated = validate_config_data(data)
    typed_cfg = OmegaConf.create(validated.model_dump(mode="python"))
    return validated, typed_cfg
