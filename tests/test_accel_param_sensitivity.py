"""Tests for accel param sensitivity — REQ-169 (TC-169-01, TC-169-02).

Verify that accel_instability_lr_decay and accel_convergence_lr_boost values
produce proportional changes in lr trajectory through adapt_to_acceleration().
"""

import pytest

from src.tg_lora.random_walk_controller import RandomWalkController


# ---------------------------------------------------------------------------
# TC-169-01: accel_instability_lr_decay値によるlr減衰率変化
# ---------------------------------------------------------------------------


class TestInstabilityDecaySensitivity:
    """TC-169-01: Different decay values produce proportionally different lr reduction."""

    @pytest.mark.parametrize("decay", [0.3, 0.5, 0.7, 0.9])
    def test_lr_reduces_proportionally(self, decay):
        ctrl = RandomWalkController(
            lr_initial=5e-4,
            lr_min=1e-8,
            lr_max=1e-2,
            enable_random_walk=True,
            accel_instability_lr_decay=decay,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        lr_before = ctrl.state.lr
        ctrl.adapt_to_acceleration(acceleration=1.0)
        expected = lr_before * decay
        assert ctrl.state.lr == pytest.approx(expected)

    def test_lower_decay_means_larger_reduction(self):
        """decay=0.3 reduces lr more than decay=0.9 under identical conditions."""
        results = {}
        for decay in [0.3, 0.5, 0.7, 0.9]:
            ctrl = RandomWalkController(
                lr_initial=5e-4,
                lr_min=1e-8,
                lr_max=1e-2,
                enable_random_walk=True,
                accel_instability_lr_decay=decay,
                k_explore_prob=0.0,
                n_explore_prob=0.0,
                beta_explore_prob=0.0,
                strategy_explore_prob=0.0,
                lr_explore_prob=0.0,
            )
            ctrl.adapt_to_acceleration(acceleration=1.0)
            results[decay] = ctrl.state.lr

        assert results[0.3] < results[0.5] < results[0.7] < results[0.9]

    def test_cumulative_effect_over_multiple_calls(self):
        """Multiple instability calls compound the decay proportionally."""
        trajectories = {}
        for decay in [0.3, 0.5, 0.7, 0.9]:
            ctrl = RandomWalkController(
                lr_initial=5e-4,
                lr_min=1e-8,
                lr_max=1e-2,
                enable_random_walk=True,
                accel_instability_lr_decay=decay,
                k_explore_prob=0.0,
                n_explore_prob=0.0,
                beta_explore_prob=0.0,
                strategy_explore_prob=0.0,
                lr_explore_prob=0.0,
            )
            lrs = [ctrl.state.lr]
            for _ in range(5):
                ctrl.adapt_to_acceleration(acceleration=1.0)
                lrs.append(ctrl.state.lr)
            trajectories[decay] = lrs

        for i in range(1, 6):
            assert trajectories[0.3][i] < trajectories[0.5][i] < trajectories[0.7][i] < trajectories[0.9][i]


# ---------------------------------------------------------------------------
# TC-169-02: accel_convergence_lr_boost値によるlr回復率変化
# ---------------------------------------------------------------------------


class TestConvergenceBoostSensitivity:
    """TC-169-02: Different boost values produce proportionally different lr recovery."""

    @pytest.mark.parametrize("boost", [1.1, 1.3, 1.5, 2.0])
    def test_lr_boosts_proportionally(self, boost):
        ctrl = RandomWalkController(
            lr_initial=5e-4,
            lr_min=1e-8,
            lr_max=1e2,
            enable_random_walk=True,
            accel_convergence_lr_boost=boost,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        lr_before = ctrl.state.lr
        ctrl.adapt_to_acceleration(acceleration=-0.5)
        expected = lr_before * boost
        assert ctrl.state.lr == pytest.approx(expected)

    def test_higher_boost_means_larger_recovery(self):
        """boost=2.0 increases lr more than boost=1.1 under identical conditions."""
        results = {}
        for boost in [1.1, 1.3, 1.5, 2.0]:
            ctrl = RandomWalkController(
                lr_initial=5e-4,
                lr_min=1e-8,
                lr_max=1e2,
                enable_random_walk=True,
                accel_convergence_lr_boost=boost,
                k_explore_prob=0.0,
                n_explore_prob=0.0,
                beta_explore_prob=0.0,
                strategy_explore_prob=0.0,
                lr_explore_prob=0.0,
            )
            ctrl.adapt_to_acceleration(acceleration=-0.5)
            results[boost] = ctrl.state.lr

        assert results[1.1] < results[1.3] < results[1.5] < results[2.0]

    def test_cumulative_effect_over_multiple_calls(self):
        """Multiple convergence calls compound the boost proportionally."""
        trajectories = {}
        for boost in [1.1, 1.3, 1.5, 2.0]:
            ctrl = RandomWalkController(
                lr_initial=5e-4,
                lr_min=1e-8,
                lr_max=1e2,
                enable_random_walk=True,
                accel_convergence_lr_boost=boost,
                k_explore_prob=0.0,
                n_explore_prob=0.0,
                beta_explore_prob=0.0,
                strategy_explore_prob=0.0,
                lr_explore_prob=0.0,
            )
            lrs = [ctrl.state.lr]
            for _ in range(5):
                ctrl.adapt_to_acceleration(acceleration=-0.5)
                lrs.append(ctrl.state.lr)
            trajectories[boost] = lrs

        for i in range(1, 6):
            assert trajectories[1.1][i] < trajectories[1.3][i] < trajectories[1.5][i] < trajectories[2.0][i]
