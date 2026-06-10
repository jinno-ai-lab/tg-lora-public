"""Verify that src/tg_lora's public API (__all__) is importable and well-formed."""

import importlib

import pytest


@pytest.fixture
def tg_lora_module():
    return importlib.import_module("src.tg_lora")


class TestPublicAPIExports:
    """Every name in __all__ must be importable and match the documented API."""

    EXPECTED_ALL = [
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

    def test_all_list_matches_expected(self, tg_lora_module):
        """__all__ must contain exactly the documented public exports."""
        actual = set(tg_lora_module.__all__)
        expected = set(self.EXPECTED_ALL)
        assert actual == expected, (
            f"Missing from __all__: {expected - actual}\n"
            f"Extra in __all__: {actual - expected}"
        )

    def test_all_exports_are_callable_or_class(self, tg_lora_module):
        """Every export must be a class or function (not a module or constant)."""
        for name in tg_lora_module.__all__:
            obj = getattr(tg_lora_module, name)
            assert callable(obj) or isinstance(obj, type), (
                f"{name} is {type(obj)}, expected callable or class"
            )

    def test_each_export_accessible_from_package(self, tg_lora_module):
        """Each __all__ entry is accessible via getattr on the package."""
        for name in tg_lora_module.__all__:
            obj = getattr(tg_lora_module, name)
            assert obj is not None, f"{name} is None"

    def test_no_extra_public_names_without_underscore(self, tg_lora_module):
        """No public names (no underscore prefix) should exist outside __all__."""
        public_attrs = {
            name
            for name in dir(tg_lora_module)
            if not name.startswith("_")
            and not isinstance(getattr(tg_lora_module, name), type(importlib))
        }
        all_set = set(tg_lora_module.__all__)
        extra = public_attrs - all_set - {"__version__", "__doc__"}
        # Filter out module-level attributes that are standard
        extra = {n for n in extra if not isinstance(getattr(tg_lora_module, n), type)}
        assert len(extra) == 0, f"Public names not in __all__: {extra}"

    def test_class_exports_have_public_methods(self, tg_lora_module):
        """Key classes should have their documented public methods."""
        # Velocity
        v = tg_lora_module.Velocity()
        assert hasattr(v, "update")
        assert hasattr(v, "reset")
        assert hasattr(v, "is_magnitude_anomalous")
        assert hasattr(v, "magnitude_trend")

        # DeltaTracker
        dt = tg_lora_module.DeltaTracker()
        assert hasattr(dt, "compute_and_record")
        assert hasattr(dt, "is_anomalous")
        assert hasattr(dt, "convergence_trend")

        # CycleState
        cs = tg_lora_module.CycleState()
        assert hasattr(cs, "record_cycle")
        assert hasattr(cs, "should_stop")

        # RollbackManager
        rm = tg_lora_module.RollbackManager()
        assert hasattr(rm, "save")
        assert hasattr(rm, "rollback")
        assert hasattr(rm, "clear")

    def test_cap_update_handles_nan(self, tg_lora_module):
        """cap_update must return zeros for NaN input (runbook: §1.3)."""
        import torch

        result = tg_lora_module.cap_update(
            torch.tensor([float("nan"), 1.0]),
            torch.ones(2),
        )
        assert torch.all(result == 0)

    def test_cap_update_caps_norm(self, tg_lora_module):
        """cap_update must cap updates exceeding max_ratio (runbook: relative_update_cap)."""
        import torch

        update = torch.ones(4) * 10.0
        ref = torch.ones(4)
        result = tg_lora_module.cap_update(update, ref, max_ratio=0.01)
        assert result.norm().item() <= 0.01 * ref.norm().item() + 1e-6

    def test_velocity_ema_smoothing(self, tg_lora_module):
        """Velocity.update applies EMA with beta (runbook: §2.1 beta parameter)."""
        import torch

        v = tg_lora_module.Velocity()
        delta1 = {"w": torch.ones(2)}
        v.update(delta1, beta=0.8)
        assert v.state is not None
        assert len(v.magnitudes) == 1

    def test_cosine_similarity_bounds(self, tg_lora_module):
        """cosine_similarity must return value in [-1, 1]."""
        import torch

        a = torch.randn(10)
        b = torch.randn(10)
        sim = tg_lora_module.cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0
