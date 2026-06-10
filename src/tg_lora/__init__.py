"""TG-LoRA core algorithm — velocity tracking, extrapolation, layer sampling."""

from src.tg_lora.velocity import OrthonormalBasis, Velocity
from src.tg_lora.extrapolator import (
    ZerothOrderStepStats,
    apply_extrapolation,
    cap_update,
    subspace_zeroth_order_step,
)
from src.tg_lora.delta_tracker import DeltaTracker, compute_mean_delta
from src.tg_lora.cycle_state import CycleState
from src.tg_lora.layer_sampler import select_active_layers, get_num_layers, StrategyName
from src.tg_lora.rollback_manager import RollbackManager
from src.tg_lora.random_walk_controller import RandomWalkController
from src.tg_lora.lora_state import snapshot_lora, load_lora_snapshot, diff_lora
from src.tg_lora.metrics import cosine_similarity, total_norm, per_layer_norms
from src.tg_lora.trajectory import (
    TrajectoryAnalyzer,
    TrajectoryPoint,
    ConvergenceEstimate,
    EarlyStopAdvice,
    TrajectoryReport,
)
from src.tg_lora.trajectory_controller import (
    TrajectoryController,
    CycleDecision,
    TrajectoryControllerConfig,
)

__all__ = [
    "Velocity",
    "OrthonormalBasis",
    "apply_extrapolation",
    "cap_update",
    "subspace_zeroth_order_step",
    "ZerothOrderStepStats",
    "DeltaTracker",
    "compute_mean_delta",
    "CycleState",
    "select_active_layers",
    "get_num_layers",
    "StrategyName",
    "RollbackManager",
    "RandomWalkController",
    "snapshot_lora",
    "load_lora_snapshot",
    "diff_lora",
    "cosine_similarity",
    "total_norm",
    "per_layer_norms",
    "TrajectoryAnalyzer",
    "TrajectoryPoint",
    "ConvergenceEstimate",
    "EarlyStopAdvice",
    "TrajectoryReport",
    "TrajectoryController",
    "CycleDecision",
    "TrajectoryControllerConfig",
]
