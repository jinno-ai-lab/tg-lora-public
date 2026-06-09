from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

_VALID_STRATEGIES = {"last_25_percent", "last_25_percent_plus_random_2", "middle_random", "lisa_like_weighted"}
_VALID_SCOPES = {"all", "last_25_percent"}


@dataclass
class TGLoraConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    adapter_rank: int = 8
    adapter_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    lr: float = 1e-4
    batch_size: int = 4
    grad_accum: int = 1
    max_steps: int = 1000
    warmup_steps: int = 100

    # TG-LoRA
    K_initial: int = 3
    N_initial: int = 5
    alpha_initial: float = 1.0
    beta_initial: float = 0.9
    relative_update_cap: float = 0.5
    rollback_tolerance: float = 0.0

    # Layer selection
    active_layer_strategy: str = "last_25_percent_plus_random_2"
    trainable_lora_scope: str = "all"

    # Output
    output_dir: str = "runs/tg_lora"
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500

    def __post_init__(self) -> None:
        if self.adapter_rank <= 0:
            raise ValueError(f"adapter_rank must be positive, got {self.adapter_rank}")
        if self.adapter_alpha <= 0:
            raise ValueError(f"adapter_alpha must be positive, got {self.adapter_alpha}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.grad_accum <= 0:
            raise ValueError(f"grad_accum must be positive, got {self.grad_accum}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.K_initial <= 0:
            raise ValueError(f"K_initial must be positive, got {self.K_initial}")
        if self.N_initial <= 0:
            raise ValueError(f"N_initial must be positive, got {self.N_initial}")
        if not (0 < self.alpha_initial <= 2.0):
            raise ValueError(f"alpha_initial must be in (0, 2.0], got {self.alpha_initial}")
        if not (0 < self.beta_initial < 1.0):
            raise ValueError(f"beta_initial must be in (0, 1.0), got {self.beta_initial}")
        if not (0 < self.relative_update_cap <= 1.0):
            raise ValueError(f"relative_update_cap must be in (0, 1.0], got {self.relative_update_cap}")
        if self.rollback_tolerance < 0:
            raise ValueError(f"rollback_tolerance must be non-negative, got {self.rollback_tolerance}")
        if self.active_layer_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"active_layer_strategy must be one of {sorted(_VALID_STRATEGIES)}, "
                f"got '{self.active_layer_strategy}'"
            )
        if self.trainable_lora_scope not in _VALID_SCOPES:
            raise ValueError(
                f"trainable_lora_scope must be one of {sorted(_VALID_SCOPES)}, "
                f"got '{self.trainable_lora_scope}'"
            )

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        d = self.to_dict()
        lines = ["TGLoraConfig:"]
        for key, value in d.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> TGLoraConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save_yaml(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML support. Install it with: pip install pyyaml"
            ) from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TGLoraConfig:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML support. Install it with: pip install pyyaml"
            ) from exc
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
