import math
from pathlib import Path
from unittest.mock import patch

import pytest

from tg_lora.random_walk_controller import RandomWalkController


def test_initial_state():
    ctrl = RandomWalkController(
        K_initial=3,
        N_initial=5,
        alpha_initial=0.3,
        beta_initial=0.8,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.state.K == 3
    assert ctrl.state.N == 5
    assert ctrl.state.alpha == 0.3
    assert ctrl.state.beta == 0.8


def test_propose_alpha_in_range():
    ctrl = RandomWalkController(
        alpha_initial=0.3,
        alpha_min=0.03,
        alpha_max=1.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for _ in range(100):
        proposal = ctrl.propose()
        assert proposal.alpha >= 0.03
        assert proposal.alpha <= 1.5


def test_accept_and_reward():
    ctrl = RandomWalkController(
        rollback_tolerance=0.005,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )

    assert ctrl.accept(loss_pilot=1.0, loss_after=0.9) is True
    assert ctrl.accept(loss_pilot=1.0, loss_after=1.004) is True
    assert ctrl.accept(loss_pilot=1.0, loss_after=1.01) is False

    ctrl.reward(loss_pilot=1.0, loss_after=0.9)
    assert ctrl.state.accepted_count == 1
    assert ctrl.state.alpha > 0.3  # boosted


def test_penalize():
    ctrl = RandomWalkController(
        alpha_initial=0.3,
        N_initial=5,
        N_candidates=[1, 3, 5, 10, 20],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.penalize(loss_pilot=1.0, loss_after=1.1)
    assert ctrl.state.rolled_back_count == 1
    assert ctrl.state.alpha < 0.3  # decayed


def test_acceptance_rate():
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.reward(1.0, 0.9)
    ctrl.reward(1.0, 0.9)
    ctrl.penalize(1.0, 1.1)

    rate = ctrl.acceptance_rate()
    assert abs(rate - 2.0 / 3.0) < 1e-6


def test_summary():
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.reward(1.0, 0.9)
    s = ctrl.summary()
    assert s["total_cycles"] == 1
    assert s["accepted"] == 1
    assert s["acceptance_rate"] == 1.0


def test_acceptance_rate_zero_cycles():
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.acceptance_rate() == 0.0


def test_update_layer_scores():
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.update_layer_scores([0, 4, 8], reward=1.5)
    assert ctrl.state.layer_scores[0] == 1.5
    assert ctrl.state.layer_scores[4] == 1.5
    assert ctrl.state.layer_scores[8] == 1.5
    assert 1 not in ctrl.state.layer_scores


def test_update_layer_scores_accumulates():
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.update_layer_scores([0], reward=1.0)
    ctrl.update_layer_scores([0], reward=2.0)
    assert ctrl.state.layer_scores[0] == 3.0


def test_penalize_decreases_n():
    ctrl = RandomWalkController(
        N_initial=5,
        N_candidates=[1, 3, 5, 10, 20],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    # random calls: N decrease check (0.3), strategy switch (0.5)
    with patch("random.random", side_effect=[0.3, 0.5]):
        ctrl.penalize(1.0, 1.1)
    assert ctrl.state.N == 3


def test_penalize_switches_strategy():
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    original = ctrl.state.active_layer_strategy
    # random calls: N decrease (0.5), strategy switch (0.05)
    with patch("random.random", side_effect=[0.5, 0.05]):
        ctrl.penalize(1.0, 1.1)
    assert ctrl.state.active_layer_strategy != original


def test_reward_increases_n():
    ctrl = RandomWalkController(
        N_initial=5,
        N_candidates=[1, 3, 5, 10, 20],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    # random calls: K decrease check (0.5), N increase check (0.1)
    with patch("random.random", side_effect=[0.5, 0.1]):
        ctrl.reward(1.0, 0.9)
    assert ctrl.state.N == 10  # bumped from 5 (idx=2) to 10 (idx=3)


def test_lr_initial_state():
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=0.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    assert ctrl.state.lr == 5e-4
    proposal = ctrl.propose()
    assert proposal.lr == 5e-4


def test_reward_boosts_lr():
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_max=1e-3,
        lr_accept_boost=1.2,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    with patch("random.random", side_effect=[0.5, 0.5]):
        ctrl.reward(1.0, 0.9)
    assert ctrl.state.lr > 5e-4
    assert ctrl.state.lr <= 1e-3


def test_penalize_decays_lr():
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_reject_decay=0.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    # random calls: N decrease (0.5), strategy (0.5)
    with patch("random.random", side_effect=[0.5, 0.5]):
        ctrl.penalize(1.0, 1.1)
    assert ctrl.state.lr == 2.5e-4


def test_reward_decreases_K():
    ctrl = RandomWalkController(
        K_initial=5,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    # random calls: K decrease check (0.1 < 0.2), N increase check (0.5)
    with patch("random.random", side_effect=[0.1, 0.5]):
        ctrl.reward(1.0, 0.9)
    assert ctrl.state.K == 3


def test_penalize_increases_K_deterministic():
    ctrl = RandomWalkController(
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    # K increases deterministically on reject, no random call for K
    # random calls: N decrease (0.5), strategy (0.5)
    with patch("random.random", side_effect=[0.5, 0.5]):
        ctrl.penalize(1.0, 1.1)
    assert ctrl.state.K == 5


def test_adapt_to_convergence_stalling():
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.state.total_cycles = 5
    ctrl.adapt_to_convergence(convergence_trend=0.01)  # positive = stalling
    assert ctrl.state.lr < 5e-4
    assert ctrl.state.K == 5  # increased


def test_adapt_to_convergence_healthy():
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        K_initial=3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.state.total_cycles = 5
    original_lr = ctrl.state.lr
    ctrl.adapt_to_convergence(convergence_trend=-0.5)  # negative = healthy
    assert ctrl.state.lr == original_lr  # no change
    assert ctrl.state.K == 3  # no change


def test_adapt_to_convergence_can_be_disabled_independently():
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        enable_random_walk=True,
        enable_convergence_adaptation=False,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.state.total_cycles = 5
    original_lr = ctrl.state.lr
    ctrl.adapt_to_convergence(convergence_trend=0.01)
    assert ctrl.state.lr == original_lr
    assert ctrl.state.K == 3


def test_adapt_to_convergence_active_when_random_walk_disabled():
    """Convergence adaptation should work independently of random walk.

    Phase 43 experiment configs set enable_random_walk=False but
    enable_convergence_adaptation=True — convergence adaptation must
    still be active.
    """
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        enable_random_walk=False,
        enable_convergence_adaptation=True,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.state.total_cycles = 5
    ctrl.adapt_to_convergence(convergence_trend=0.01)  # positive = stalling
    assert ctrl.state.lr < 5e-4, "lr should decay when convergence stalls"
    assert ctrl.state.K == 5, "K should increase when convergence stalls"


def test_lr_clamps_at_lr_min_under_repeated_rejects():
    """Repeated penalize cycles must never push lr below lr_min."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_reject_decay=0.5,
        N_initial=5,
        K_initial=3,
        N_candidates=[1, 3, 5, 10, 20],
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for _ in range(50):
        # fix random so N doesn't go below index 0 and strategy doesn't error
        with patch("random.random", side_effect=[0.5, 0.5]):
            ctrl.penalize(1.0, 1.1)
    assert ctrl.state.lr >= ctrl.lr_min
    assert ctrl.state.lr == ctrl.lr_min  # clamped at floor


def test_lr_clamps_at_lr_max_under_repeated_accepts():
    """Repeated reward cycles must never push lr above lr_max."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_accept_boost=1.2,
        K_initial=5,
        K_candidates=[2, 3, 5, 8],
        N_initial=1,
        N_candidates=[1, 3, 5, 10, 20],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for _ in range(50):
        with patch("random.random", side_effect=[0.5, 0.5]):
            ctrl.reward(1.0, 0.9)
    assert ctrl.state.lr <= ctrl.lr_max
    assert ctrl.state.lr == ctrl.lr_max  # clamped at ceiling


def test_lr_alternating_accept_reject_stays_in_bounds():
    """Alternating accept/reject cycles keep lr within [lr_min, lr_max]."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_accept_boost=1.5,
        lr_reject_decay=0.5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        N_initial=5,
        N_candidates=[1, 3, 5, 10, 20],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for i in range(100):
        if i % 2 == 0:
            with patch("random.random", side_effect=[0.5, 0.5]):
                ctrl.reward(1.0, 0.9)
        else:
            with patch("random.random", side_effect=[0.5, 0.5]):
                ctrl.penalize(1.0, 1.1)
        assert ctrl.lr_min <= ctrl.state.lr <= ctrl.lr_max, (
            f"Cycle {i}: lr={ctrl.state.lr} out of [{ctrl.lr_min}, {ctrl.lr_max}]"
        )


def test_accept_rejects_nan_loss():
    """NaN loss must always be rejected."""
    ctrl = RandomWalkController(
        rollback_tolerance=0.005,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accept(loss_pilot=1.0, loss_after=float("nan")) is False
    assert ctrl.accept(loss_pilot=float("nan"), loss_after=0.9) is False
    assert ctrl.accept(loss_pilot=float("nan"), loss_after=float("nan")) is False


def test_accept_rejects_inf_loss():
    """Inf loss must always be rejected."""
    ctrl = RandomWalkController(
        rollback_tolerance=0.005,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accept(loss_pilot=1.0, loss_after=float("inf")) is False
    assert ctrl.accept(loss_pilot=float("inf"), loss_after=1.0) is False
    assert ctrl.accept(loss_pilot=1.0, loss_after=float("-inf")) is False


def test_layer_score_feedback_loop_integration():
    """REQ-055: layer scores are updated based on accept/reject feedback for active layers."""
    import sys

    from tg_lora.layer_sampler import select_active_layers

    sys.path.insert(0, str(Path(__file__).parent))
    from test_layer_sampler import FakeTransformerModel

    model = FakeTransformerModel(12)
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )

    # Simulate 3 accept cycles with the same layers
    active_names, active_indices = select_active_layers(model, "last_25_percent")
    for _ in range(3):
        ctrl.update_layer_scores(list(active_indices), 1.0)

    # Last 3 layers should have score 3.0, others 0
    assert ctrl.state.layer_scores[9] == 3.0
    assert ctrl.state.layer_scores[10] == 3.0
    assert ctrl.state.layer_scores[11] == 3.0
    assert 0 not in ctrl.state.layer_scores

    # Simulate a reject cycle — penalize the same layers
    ctrl.update_layer_scores(list(active_indices), -1.0)
    assert ctrl.state.layer_scores[9] == 2.0

    # Verify scores can feed back into weighted selection
    _, weighted_indices = select_active_layers(
        model,
        "lisa_like_weighted",
        layer_scores=ctrl.state.layer_scores,
        temperature=1.0,
    )
    assert len(weighted_indices) > 0


# --- Input validation tests ---


def test_reject_negative_K_candidates():
    import pytest

    with pytest.raises(ValueError, match="K_candidates must be positive"):
        RandomWalkController(
            K_candidates=[2, -1, 5],
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_zero_N_candidates():
    import pytest

    with pytest.raises(ValueError, match="N_candidates must be positive"):
        RandomWalkController(
            N_candidates=[1, 0, 5],
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_lr_min_ge_lr_max():
    import pytest

    with pytest.raises(ValueError, match="lr_min .* must be < lr_max"):
        RandomWalkController(
            lr_min=1e-3,
            lr_max=1e-3,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_alpha_min_ge_alpha_max():
    import pytest

    with pytest.raises(ValueError, match="alpha_min .* must be < alpha_max"):
        RandomWalkController(
            alpha_min=1.5,
            alpha_max=1.5,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_nonpositive_K_initial():
    with pytest.raises(ValueError, match="K_initial must be positive"):
        RandomWalkController(
            K_initial=0,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_negative_K_initial():
    with pytest.raises(ValueError, match="K_initial must be positive"):
        RandomWalkController(
            K_initial=-1,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_nonpositive_N_initial():
    with pytest.raises(ValueError, match="N_initial must be positive"):
        RandomWalkController(
            N_initial=0,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_alpha_initial_below_min():
    with pytest.raises(ValueError, match="alpha_initial .* must be in"):
        RandomWalkController(
            alpha_initial=0.01,
            alpha_min=0.03,
            alpha_max=1.5,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_alpha_initial_above_max():
    with pytest.raises(ValueError, match="alpha_initial .* must be in"):
        RandomWalkController(
            alpha_initial=2.0,
            alpha_min=0.03,
            alpha_max=1.5,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_beta_initial_below_zero():
    with pytest.raises(ValueError, match="beta_initial must be in"):
        RandomWalkController(
            beta_initial=-0.1,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_beta_initial_above_one():
    with pytest.raises(ValueError, match="beta_initial must be in"):
        RandomWalkController(
            beta_initial=1.1,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_accept_beta_initial_at_boundaries():
    RandomWalkController(
        beta_initial=0.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    RandomWalkController(
        beta_initial=1.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )


def test_reject_lr_initial_below_min():
    with pytest.raises(ValueError, match="lr_initial .* must be in"):
        RandomWalkController(
            lr_initial=1e-6,
            lr_min=1e-5,
            lr_max=1e-3,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_lr_initial_above_max():
    with pytest.raises(ValueError, match="lr_initial .* must be in"):
        RandomWalkController(
            lr_initial=2e-3,
            lr_min=1e-5,
            lr_max=1e-3,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_nonpositive_relative_update_cap():
    with pytest.raises(ValueError, match="relative_update_cap must be positive"):
        RandomWalkController(
            relative_update_cap=0.0,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_negative_rollback_tolerance():
    with pytest.raises(ValueError, match="rollback_tolerance must be non-negative"):
        RandomWalkController(
            rollback_tolerance=-0.01,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_accept_zero_rollback_tolerance():
    RandomWalkController(
        rollback_tolerance=0.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )


def test_reject_nonpositive_alpha_log_sigma():
    with pytest.raises(ValueError, match="alpha_log_sigma must be positive"):
        RandomWalkController(
            alpha_log_sigma=0.0,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


# --- Exploration probability parameter tests ---


def test_explore_prob_defaults():
    """Exploration probability parameters default to class constants when None."""
    ctrl = RandomWalkController()
    assert ctrl.k_explore_prob == RandomWalkController._DEFAULT_K_EXPLORE_PROB
    assert ctrl.n_explore_prob == RandomWalkController._DEFAULT_N_EXPLORE_PROB
    assert ctrl.beta_explore_prob == RandomWalkController._DEFAULT_BETA_EXPLORE_PROB
    assert (
        ctrl.strategy_explore_prob
        == RandomWalkController._DEFAULT_STRATEGY_EXPLORE_PROB
    )
    assert ctrl.lr_explore_prob == RandomWalkController._DEFAULT_LR_EXPLORE_PROB
    assert ctrl.lr_log_sigma == RandomWalkController._LR_LOG_SIGMA


def test_explore_prob_custom_values():
    """Exploration probability parameters accept custom values."""
    ctrl = RandomWalkController(
        k_explore_prob=0.1,
        n_explore_prob=0.2,
        beta_explore_prob=0.3,
        strategy_explore_prob=0.4,
        lr_explore_prob=0.25,
        lr_log_sigma=0.2,
    )
    assert ctrl.k_explore_prob == 0.1
    assert ctrl.n_explore_prob == 0.2
    assert ctrl.beta_explore_prob == 0.3
    assert ctrl.strategy_explore_prob == 0.4
    assert ctrl.lr_explore_prob == 0.25
    assert ctrl.lr_log_sigma == 0.2


@pytest.mark.parametrize("label,extra_kwargs,attr,initial", [
    ("K", dict(K_initial=3, K_candidates=[2, 3, 5, 8]), "K", 3),
    ("N", dict(N_initial=5, N_candidates=[1, 3, 5, 10, 20]), "N", 5),
    ("beta", dict(beta_initial=0.8, beta_candidates=[0.5, 0.8, 0.9, 0.95]), "beta", 0.8),
    ("strategy", {}, "active_layer_strategy", None),
])
def test_zero_explore_prob_freezes_param(label, extra_kwargs, attr, initial):
    ctrl = RandomWalkController(
        **extra_kwargs,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    expected = initial if initial is not None else ctrl.state.active_layer_strategy
    for _ in range(200):
        assert getattr(ctrl.propose(), attr) == expected


@pytest.mark.parametrize("label,extra_kwargs,explore_key,attr,initial,n_trials,min_changes", [
    ("K", dict(K_initial=3, K_candidates=[2, 3, 5, 8]), "k_explore_prob", "K", 3, 100, 100),
    ("N", dict(N_initial=5, N_candidates=[1, 3, 5, 10, 20]), "n_explore_prob", "N", 5, 100, 100),
    ("beta", dict(beta_initial=0.8, beta_candidates=[0.5, 0.8, 0.9, 0.95]), "beta_explore_prob", "beta", 0.8, 300, 187),
    ("strategy", {}, "strategy_explore_prob", "active_layer_strategy", None, 200, 200),
])
def test_full_explore_prob_changes_param(label, extra_kwargs, explore_key, attr, initial, n_trials, min_changes):
    probs = dict(k_explore_prob=0.0, n_explore_prob=0.0, beta_explore_prob=0.0,
                 strategy_explore_prob=0.0, lr_explore_prob=0.0)
    probs[explore_key] = 1.0
    ctrl = RandomWalkController(**extra_kwargs, **probs)
    expected = initial if initial is not None else ctrl.state.active_layer_strategy
    changed = sum(1 for _ in range(n_trials) if getattr(ctrl.propose(), attr) != expected)
    assert changed >= min_changes


# --- Config-to-controller integration tests ---


def _make_controller_from_config_dict(tg_lora_dict: dict) -> RandomWalkController:
    """Mirror the controller construction logic from train_tg_lora.py."""
    from omegaconf import OmegaConf

    tg_cfg = OmegaConf.create(tg_lora_dict)
    return RandomWalkController(
        K_initial=tg_cfg.K_initial,
        K_candidates=list(tg_cfg.K_candidates),
        N_initial=tg_cfg.N_initial,
        N_candidates=list(tg_cfg.N_candidates),
        alpha_initial=tg_cfg.alpha_initial,
        alpha_min=tg_cfg.alpha_min,
        alpha_max=tg_cfg.alpha_max,
        alpha_log_sigma=tg_cfg.get("alpha_log_sigma", 0.15),
        beta_initial=tg_cfg.beta_initial,
        beta_candidates=list(tg_cfg.beta_candidates),
        lr_initial=tg_cfg.get("lr_initial", 5e-4),
        lr_min=tg_cfg.get("lr_min", 1e-5),
        lr_max=tg_cfg.get("lr_max", 1e-3),
        lr_accept_boost=tg_cfg.get("lr_accept_boost", 1.2),
        lr_reject_decay=tg_cfg.get("lr_reject_decay", 0.5),
        active_layer_strategy=tg_cfg.active_layer_strategy,
        relative_update_cap=tg_cfg.relative_update_cap,
        k_explore_prob=tg_cfg.get("k_explore_prob", None),
        n_explore_prob=tg_cfg.get("n_explore_prob", None),
        beta_explore_prob=tg_cfg.get("beta_explore_prob", None),
        strategy_explore_prob=tg_cfg.get("strategy_explore_prob", None),
        lr_explore_prob=tg_cfg.get("lr_explore_prob", None),
        lr_log_sigma=tg_cfg.get("lr_log_sigma", None),
    )


def _base_tg_lora_dict(**overrides) -> dict:
    base = {
        "K_initial": 3,
        "K_candidates": [2, 3, 5, 8],
        "N_initial": 5,
        "N_candidates": [1, 3, 5, 10, 20],
        "alpha_initial": 0.3,
        "alpha_min": 0.03,
        "alpha_max": 1.5,
        "alpha_log_sigma": 0.15,
        "beta_initial": 0.8,
        "beta_candidates": [0.5, 0.8, 0.9, 0.95],
        "lr_initial": 5e-4,
        "lr_min": 1e-5,
        "lr_max": 1e-3,
        "lr_accept_boost": 1.2,
        "lr_reject_decay": 0.5,
        "relative_update_cap": 0.005,
        "active_layer_strategy": "last_25_percent_plus_random_2",
    }
    base.update(overrides)
    return base


def test_config_to_controller_explicit_explore_probs():
    """Config with explicit explore prob values → controller receives those values."""
    ctrl = _make_controller_from_config_dict(
        _base_tg_lora_dict(
            k_explore_prob=0.1,
            n_explore_prob=0.2,
            beta_explore_prob=0.3,
            strategy_explore_prob=0.07,
        )
    )
    assert ctrl.k_explore_prob == 0.1
    assert ctrl.n_explore_prob == 0.2
    assert ctrl.beta_explore_prob == 0.3
    assert ctrl.strategy_explore_prob == 0.07


def test_config_to_controller_defaults_when_omitted():
    """Config without explore prob fields → controller falls back to class defaults."""
    ctrl = _make_controller_from_config_dict(_base_tg_lora_dict())
    assert ctrl.k_explore_prob == RandomWalkController._DEFAULT_K_EXPLORE_PROB
    assert ctrl.n_explore_prob == RandomWalkController._DEFAULT_N_EXPLORE_PROB
    assert ctrl.beta_explore_prob == RandomWalkController._DEFAULT_BETA_EXPLORE_PROB
    assert (
        ctrl.strategy_explore_prob
        == RandomWalkController._DEFAULT_STRATEGY_EXPLORE_PROB
    )
    assert ctrl.lr_explore_prob == RandomWalkController._DEFAULT_LR_EXPLORE_PROB
    assert ctrl.lr_log_sigma == RandomWalkController._LR_LOG_SIGMA


def test_enable_random_walk_false_freezes_hyperparameters():
    ctrl = RandomWalkController(
        K_initial=3,
        N_initial=5,
        alpha_initial=0.3,
        beta_initial=0.8,
        lr_initial=5e-4,
        enable_random_walk=False,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    proposal = ctrl.propose()
    assert proposal.K == 3
    assert proposal.N == 5
    assert proposal.alpha == 0.3
    assert proposal.beta == 0.8
    assert proposal.lr == 5e-4

    ctrl.reward(1.0, 0.9)
    assert ctrl.state.accepted_count == 1
    assert ctrl.state.total_cycles == 1
    assert ctrl.state.K == 3
    assert ctrl.state.N == 5
    assert ctrl.state.alpha == 0.3
    assert ctrl.state.lr == 5e-4

    ctrl.penalize(1.0, 1.1)
    assert ctrl.state.rolled_back_count == 1
    assert ctrl.state.total_cycles == 2
    assert ctrl.state.K == 3
    assert ctrl.state.N == 5
    assert ctrl.state.alpha == 0.3
    assert ctrl.state.lr == 5e-4


def test_config_custom_explore_probs_affect_propose():
    """Custom explore probs from config actually change propose() behavior.

    K at non-edge index, both neighbors differ → P(change|explore)=1.0.
    P(change)=0.99, n=200 → E=198, σ≈1.41.
    Threshold at E−5σ≈191 → false-positive rate < 0.0001%.
    """
    # High explore prob for K (0.99), zero for others
    ctrl = _make_controller_from_config_dict(
        _base_tg_lora_dict(
            k_explore_prob=0.99,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
        )
    )
    original_k = ctrl.state.K
    k_changed = sum(1 for _ in range(200) if ctrl.propose().K != original_k)
    assert k_changed > 191  # E−5σ=198−7; false-positive < 0.0001%

    # N should never change
    original_n = ctrl.state.N
    for _ in range(200):
        assert ctrl.propose().N == original_n


# --- ControllerState serialization tests ---


def test_controller_state_summary_round_trip():
    """summary() → from_dict() preserves all fields."""
    from tg_lora.random_walk_controller import ControllerState as CS

    state = CS(
        K=5,
        N=10,
        alpha=0.3,
        beta=0.8,
        lr=5e-4,
        active_layer_strategy="last_25_percent",
        relative_update_cap=0.005,
        layer_scores={0: 1.5, 4: 2.1},
        total_cycles=7,
        accepted_count=5,
        rolled_back_count=2,
        alpha_accept_boost=1.2,
        alpha_reject_decay=0.4,
        lr_accept_boost=1.3,
        lr_reject_decay=0.6,
    )
    restored = CS.from_dict(state.summary())
    assert restored.K == state.K
    assert restored.N == state.N
    assert restored.alpha == pytest.approx(state.alpha)
    assert restored.beta == pytest.approx(state.beta)
    assert restored.lr == pytest.approx(state.lr)
    assert restored.active_layer_strategy == state.active_layer_strategy
    assert restored.relative_update_cap == pytest.approx(state.relative_update_cap)
    assert restored.layer_scores == state.layer_scores
    assert restored.total_cycles == state.total_cycles
    assert restored.accepted_count == state.accepted_count
    assert restored.rolled_back_count == state.rolled_back_count
    assert restored.alpha_accept_boost == pytest.approx(state.alpha_accept_boost)
    assert restored.alpha_reject_decay == pytest.approx(state.alpha_reject_decay)
    assert restored.lr_accept_boost == pytest.approx(state.lr_accept_boost)
    assert restored.lr_reject_decay == pytest.approx(state.lr_reject_decay)


def test_controller_state_from_dict_with_defaults():
    """from_dict fills defaults for optional fields."""
    from tg_lora.random_walk_controller import ControllerState as CS

    data = {
        "K": 3,
        "N": 5,
        "alpha": 0.3,
        "beta": 0.8,
        "lr": 5e-4,
        "active_layer_strategy": "middle_random",
        "relative_update_cap": 0.005,
    }
    restored = CS.from_dict(data)
    assert restored.layer_scores == {}
    assert restored.total_cycles == 0
    assert restored.accepted_count == 0
    assert restored.rolled_back_count == 0
    assert restored.alpha_accept_boost == 1.1
    assert restored.alpha_reject_decay == 0.5
    assert restored.lr_accept_boost == 1.2
    assert restored.lr_reject_decay == 0.5


def test_propose_recovers_from_zero_alpha():
    """propose() must not crash when alpha is zero (e.g. via corrupted state)."""
    from tg_lora.random_walk_controller import ControllerState as CS

    ctrl = RandomWalkController(
        alpha_initial=0.3,
        alpha_min=0.03,
        alpha_max=1.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.state = CS(
        K=3, N=5, alpha=0.0, beta=0.8, lr=5e-4,
        active_layer_strategy="last_25_percent_plus_random_2",
        relative_update_cap=0.005,
    )
    proposal = ctrl.propose()
    assert proposal.alpha == pytest.approx(ctrl.alpha_min)


def test_propose_recovers_from_negative_alpha():
    """propose() must not crash when alpha is negative."""
    from tg_lora.random_walk_controller import ControllerState as CS

    ctrl = RandomWalkController(
        alpha_initial=0.3,
        alpha_min=0.03,
        alpha_max=1.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.state = CS(
        K=3, N=5, alpha=-1.0, beta=0.8, lr=5e-4,
        active_layer_strategy="last_25_percent_plus_random_2",
        relative_update_cap=0.005,
    )
    proposal = ctrl.propose()
    assert proposal.alpha == pytest.approx(ctrl.alpha_min)


# --- Learning rate exploration tests ---


def test_lr_explore_prob_zero_freezes_lr():
    """lr_explore_prob=0.0 means lr never changes in propose()."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=0.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    for _ in range(200):
        assert ctrl.propose().lr == 5e-4


def test_lr_explore_prob_one_always_perturbs():
    """lr_explore_prob=1.0 means lr is perturbed every propose() call.

    Log-normal walk is continuous so the probability of landing exactly on the
    current value is effectively zero. We just verify perturbation happens >95%.
    """
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=1.0,
        lr_log_sigma=0.15,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    changed = sum(1 for _ in range(200) if ctrl.propose().lr != 5e-4)
    assert changed > 190


def test_lr_exploration_stays_in_bounds():
    """Proposed lr must always stay within [lr_min, lr_max]."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=1.0,
        lr_log_sigma=0.3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    for _ in range(500):
        proposal = ctrl.propose()
        assert ctrl.lr_min <= proposal.lr <= ctrl.lr_max, (
            f"lr={proposal.lr} out of [{ctrl.lr_min}, {ctrl.lr_max}]"
        )


def test_lr_exploration_does_not_modify_state():
    """propose() must not change controller.state.lr — only reward/penalize do."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=1.0,
        lr_log_sigma=0.3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    for _ in range(50):
        ctrl.propose()
    assert ctrl.state.lr == 5e-4


def test_lr_exploration_disabled_when_random_walk_off():
    """lr exploration must not happen when enable_random_walk=False."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=1.0,
        enable_random_walk=False,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    proposal = ctrl.propose()
    assert proposal.lr == 5e-4


def test_lr_exploration_near_min_clamps_to_lr_min():
    """When lr is at lr_min, downward perturbation must clamp."""
    from tg_lora.random_walk_controller import ControllerState as CS

    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=1.0,
        lr_log_sigma=0.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    ctrl.state = CS(
        K=3, N=5, alpha=0.3, beta=0.8, lr=1e-5,
        active_layer_strategy="last_25_percent_plus_random_2",
        relative_update_cap=0.005,
    )
    for _ in range(100):
        assert ctrl.propose().lr >= ctrl.lr_min


def test_lr_exploration_near_max_clamps_to_lr_max():
    """When lr is at lr_max, upward perturbation must clamp."""
    from tg_lora.random_walk_controller import ControllerState as CS

    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=1.0,
        lr_log_sigma=0.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    ctrl.state = CS(
        K=3, N=5, alpha=0.3, beta=0.8, lr=1e-3,
        active_layer_strategy="last_25_percent_plus_random_2",
        relative_update_cap=0.005,
    )
    for _ in range(100):
        assert ctrl.propose().lr <= ctrl.lr_max


def test_lr_exploration_recovers_from_zero_lr():
    """propose() must handle lr=0 gracefully (skip exploration, return 0)."""
    from tg_lora.random_walk_controller import ControllerState as CS

    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        lr_explore_prob=1.0,
        lr_log_sigma=0.3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    ctrl.state = CS(
        K=3, N=5, alpha=0.3, beta=0.8, lr=0.0,
        active_layer_strategy="last_25_percent_plus_random_2",
        relative_update_cap=0.005,
    )
    # With lr=0, the log-normal walk is skipped; lr stays at 0 (from state)
    proposal = ctrl.propose()
    assert proposal.lr == 0.0


def test_propose_lr_in_range_statistical():
    """Over many samples, the median proposed lr should be near the current lr.

    Log-normal walk with σ=0.1 → sample-median std ≈ σ·√(π/(2n)).
    n=10000 → SE ≈ 0.00125 in log-space. Tolerance of ±50% in linear space
    corresponds to ±0.693 in log-space ≈ 554σ → false-positive ≈ 0.
    """
    import statistics

    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-6,
        lr_max=1e-2,
        lr_explore_prob=1.0,
        lr_log_sigma=0.1,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    proposed_lrs = [ctrl.propose().lr for _ in range(10000)]
    median_lr = statistics.median(proposed_lrs)
    # Median should be within 50% of the initial lr
    assert 2.5e-4 < median_lr < 1e-3, f"Median lr {median_lr} too far from 5e-4"


# --- math.exp overflow protection tests ---


def test_propose_no_overflow_with_extreme_alpha_sigma():
    """propose() must not raise OverflowError even with huge alpha_log_sigma."""
    ctrl = RandomWalkController(
        alpha_initial=1.0,
        alpha_min=0.01,
        alpha_max=2.0,
        alpha_log_sigma=100.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for _ in range(100):
        proposal = ctrl.propose()
        assert math.isfinite(proposal.alpha)
        assert ctrl.alpha_min <= proposal.alpha <= ctrl.alpha_max


def test_propose_no_overflow_with_extreme_lr_sigma():
    """propose() must not raise OverflowError even with huge lr_log_sigma."""
    ctrl = RandomWalkController(
        lr_initial=1e-3,
        lr_min=1e-6,
        lr_max=1e-1,
        lr_explore_prob=1.0,
        lr_log_sigma=100.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    for _ in range(100):
        proposal = ctrl.propose()
        assert math.isfinite(proposal.lr)
        assert ctrl.lr_min <= proposal.lr <= ctrl.lr_max


# --- adapt_to_acceleration tests ---


def test_adapt_to_acceleration_positive_reduces_lr_and_increases_K():
    """Positive acceleration (instability) reduces lr and increases K."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.adapt_to_acceleration(acceleration=0.5)
    assert ctrl.state.lr < 5e-4
    assert ctrl.state.K == 5  # bumped from 3 to 5


def test_adapt_to_acceleration_negative_boosts_lr():
    """Negative acceleration (healthy convergence) slightly boosts lr."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.adapt_to_acceleration(acceleration=-0.3)
    assert ctrl.state.lr > 5e-4


def test_adapt_to_acceleration_zero_no_change():
    """Zero acceleration means no adjustment."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    original_lr = ctrl.state.lr
    original_K = ctrl.state.K
    ctrl.adapt_to_acceleration(acceleration=0.0)
    assert ctrl.state.lr == original_lr
    assert ctrl.state.K == original_K


def test_adapt_to_acceleration_small_positive_deadzone_no_change():
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    original_lr = ctrl.state.lr
    original_K = ctrl.state.K
    ctrl.adapt_to_acceleration(acceleration=0.005)
    assert ctrl.state.lr == original_lr
    assert ctrl.state.K == original_K


def test_adapt_to_acceleration_disabled_when_random_walk_off():
    """adapt_to_acceleration does nothing when both enable_random_walk=False
    and enable_convergence_adaptation=False."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        enable_random_walk=False,
        enable_convergence_adaptation=False,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    original_lr = ctrl.state.lr
    original_K = ctrl.state.K
    ctrl.adapt_to_acceleration(acceleration=0.5)
    assert ctrl.state.lr == original_lr
    assert ctrl.state.K == original_K


def test_adapt_to_acceleration_lr_clamps_at_min():
    """Repeated instability signals clamp lr at lr_min."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        K_initial=5,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for _ in range(50):
        ctrl.adapt_to_acceleration(acceleration=1.0)
    assert ctrl.state.lr >= ctrl.lr_min
    assert ctrl.state.lr == ctrl.lr_min


def test_adapt_to_acceleration_lr_clamps_at_max():
    """Repeated convergence signals clamp lr at lr_max."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for _ in range(200):
        ctrl.adapt_to_acceleration(acceleration=-1.0)
    assert ctrl.state.lr <= ctrl.lr_max
    assert ctrl.state.lr == ctrl.lr_max


def test_adapt_to_acceleration_K_clamps_at_max():
    """Repeated instability signals clamp K at max candidate."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        K_initial=3,
        K_candidates=[2, 3, 5, 8],
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    for _ in range(10):
        ctrl.adapt_to_acceleration(acceleration=1.0)
    assert ctrl.state.K == 8  # clamped at max


def test_adapt_to_acceleration_large_negative_is_convergence():
    """Acceleration below the deadzone is treated as convergence."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.adapt_to_acceleration(acceleration=-0.03)
    assert ctrl.state.lr > 5e-4  # boosted


def test_adapt_to_acceleration_tiny_negative_above_threshold_no_change():
    """Tiny negative acceleration inside the deadzone causes no change."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    original_lr = ctrl.state.lr
    ctrl.adapt_to_acceleration(acceleration=-1e-13)
    assert ctrl.state.lr == original_lr  # no change (not < -1e-12)


# --- configurable accel params ---


def test_accel_params_default_values():
    """Default accel params match the class constants."""
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_instability_lr_decay == 0.7
    assert ctrl.accel_convergence_lr_boost == 1.1


def test_accel_params_custom_values():
    """Custom accel params are accepted and used."""
    ctrl = RandomWalkController(
        accel_instability_lr_decay=0.5,
        accel_convergence_lr_boost=1.3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_instability_lr_decay == 0.5
    assert ctrl.accel_convergence_lr_boost == 1.3


def test_custom_instability_decay_applied():
    """accel_instability_lr_decay controls lr reduction on positive acceleration."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        accel_instability_lr_decay=0.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.adapt_to_acceleration(acceleration=1.0)
    assert ctrl.state.lr == pytest.approx(5e-4 * 0.5)


def test_custom_convergence_boost_applied():
    """accel_convergence_lr_boost controls lr increase on negative acceleration."""
    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_min=1e-5,
        lr_max=1e-3,
        accel_convergence_lr_boost=1.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.adapt_to_acceleration(acceleration=-0.5)
    assert ctrl.state.lr == pytest.approx(5e-4 * 1.5)


def test_accel_params_none_uses_defaults():
    """Passing None explicitly falls back to defaults."""
    ctrl = RandomWalkController(
        accel_instability_lr_decay=None,
        accel_convergence_lr_boost=None,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_instability_lr_decay == 0.7
    assert ctrl.accel_convergence_lr_boost == 1.1


# --- accel param constructor validation ---


@pytest.mark.parametrize("value", [0.0, -0.5, 1.0, 1.5])
def test_accel_instability_lr_decay_rejects_invalid(value):
    with pytest.raises(ValueError, match="accel_instability_lr_decay must be in"):
        RandomWalkController(
            accel_instability_lr_decay=value,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_accel_instability_lr_decay_accepts_boundary():
    ctrl = RandomWalkController(
        accel_instability_lr_decay=0.99,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_instability_lr_decay == 0.99


@pytest.mark.parametrize("value", [1.0, 0.5, 0.0, -0.1])
def test_accel_convergence_lr_boost_rejects_invalid(value):
    with pytest.raises(ValueError, match="accel_convergence_lr_boost must be > 1.0"):
        RandomWalkController(
            accel_convergence_lr_boost=value,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_accel_convergence_lr_boost_accepts_above_one():
    ctrl = RandomWalkController(
        accel_convergence_lr_boost=1.01,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_convergence_lr_boost == 1.01


def test_accel_params_none_skips_validation():
    """None values should be accepted (they use defaults)."""
    ctrl = RandomWalkController(
        accel_instability_lr_decay=None,
        accel_convergence_lr_boost=None,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_instability_lr_decay == 0.7
    assert ctrl.accel_convergence_lr_boost == 1.1


# --- accel bounds edge-case tests ---


@pytest.mark.parametrize("kwargs,attr,expected", [
    (dict(alpha_initial=0.03, alpha_min=0.03, alpha_max=1.5), "alpha", 0.03),
    (dict(alpha_initial=1.5, alpha_min=0.03, alpha_max=1.5), "alpha", 1.5),
    (dict(lr_initial=1e-5, lr_min=1e-5, lr_max=1e-3), "lr", 1e-5),
    (dict(lr_initial=1e-3, lr_min=1e-5, lr_max=1e-3), "lr", 1e-3),
])
def test_initial_at_boundary_accepted(kwargs, attr, expected):
    ctrl = RandomWalkController(
        **kwargs,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert getattr(ctrl.state, attr) == expected


@pytest.mark.parametrize("kwargs,match", [
    (dict(alpha_initial=-0.1, alpha_min=0.03, alpha_max=1.5), "alpha_initial .* must be in"),
    (dict(lr_initial=-1e-4, lr_min=1e-5, lr_max=1e-3), "lr_initial .* must be in"),
])
def test_reject_negative_initial(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RandomWalkController(
            **kwargs,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


def test_reject_negative_alpha_min():
    """Negative alpha_min should still allow construction if alpha_initial is in range."""
    # alpha_min can be negative in theory, but alpha_initial must be >= alpha_min
    # This tests that the range check is [alpha_min, alpha_max]
    ctrl = RandomWalkController(
        alpha_initial=0.03,
        alpha_min=-0.1,
        alpha_max=1.5,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.alpha_min == -0.1
    assert ctrl.state.alpha == 0.03


def test_reject_negative_lr_min():
    """Negative lr_min is accepted by construction but lr_initial must be in range."""
    ctrl = RandomWalkController(
        lr_initial=1e-4,
        lr_min=-1e-3,
        lr_max=1e-3,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.lr_min == -1e-3


@pytest.mark.parametrize("kwargs,match", [
    (dict(alpha_initial=0.5, alpha_min=0.5, alpha_max=0.5), "alpha_min .* must be < alpha_max"),
    (dict(lr_initial=5e-4, lr_min=5e-4, lr_max=5e-4), "lr_min .* must be < lr_max"),
    (dict(alpha_initial=0.5, alpha_min=1.5, alpha_max=0.03), "alpha_min .* must be < alpha_max"),
    (dict(lr_initial=5e-4, lr_min=1e-3, lr_max=1e-5), "lr_min .* must be < lr_max"),
])
def test_reject_invalid_range(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RandomWalkController(
            **kwargs,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


# --- accel NaN edge-case tests ---


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_accel_instability_lr_decay_rejects_nonfinite(value):
    with pytest.raises(ValueError, match="accel_instability_lr_decay must be in"):
        RandomWalkController(
            accel_instability_lr_decay=value,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_accel_convergence_lr_boost_rejects_nonfinite(value):
    with pytest.raises(ValueError, match="accel_convergence_lr_boost"):
        RandomWalkController(
            accel_convergence_lr_boost=value,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


# --- runtime non-finite input guards ---


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_adapt_to_acceleration_ignores_nonfinite(value):
    ctrl = RandomWalkController(
        enable_random_walk=True,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    lr_before = ctrl.state.lr
    ctrl.adapt_to_acceleration(value)
    assert ctrl.state.lr == lr_before
    assert ctrl.last_accel_action == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_adapt_to_convergence_ignores_nonfinite(value):
    ctrl = RandomWalkController(
        enable_convergence_adaptation=True,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    lr_before = ctrl.state.lr
    ctrl.state.total_cycles = 5
    ctrl.adapt_to_convergence(value)
    assert ctrl.state.lr == lr_before


# --- lr_log_sigma validation ---


@pytest.mark.parametrize("value", [
    float("nan"), 0.0, -0.1, float("inf"),
])
def test_lr_log_sigma_rejects_invalid(value):
    with pytest.raises(ValueError, match="lr_log_sigma must be positive and finite"):
        RandomWalkController(
            lr_log_sigma=value,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )


# --- last_accel_action observability ---


@pytest.mark.parametrize("accel,enable_rw,enable_conv,expected", [
    (None, True, True, 0),        # default, no action taken
    (1.0, True, True, 1),         # positive → instability
    (-0.5, True, True, -1),       # negative → convergence
    (0.0, True, True, 0),         # zero → stable
    (1.0, False, False, 0),       # disabled → stays 0
])
def test_last_accel_action(accel, enable_rw, enable_conv, expected):
    ctrl = RandomWalkController(
        enable_random_walk=enable_rw,
        enable_convergence_adaptation=enable_conv,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    if accel is not None:
        ctrl.adapt_to_acceleration(acceleration=accel)
    assert ctrl.last_accel_action == expected


def test_last_accel_action_in_summary():
    """summary() includes last_accel_action."""
    ctrl = RandomWalkController(
        enable_random_walk=True,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.adapt_to_acceleration(acceleration=1.0)
    s = ctrl.summary()
    assert "last_accel_action" in s
    assert s["last_accel_action"] == 1


# --- restore_state tests ---


def test_restore_state_replaces_state():
    """restore_state replaces K, N, alpha, beta, lr and counts."""
    from tg_lora.random_walk_controller import ControllerState

    ctrl = RandomWalkController(
        K_initial=3,
        N_initial=5,
        alpha_initial=0.3,
        beta_initial=0.8,
        lr_initial=5e-4,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    saved = ControllerState(
        K=8,
        N=20,
        alpha=0.9,
        beta=0.95,
        lr=1e-4,
        active_layer_strategy="middle_random",
        relative_update_cap=0.01,
        total_cycles=42,
        accepted_count=30,
        rolled_back_count=12,
    )
    ctrl.restore_state(saved)

    assert ctrl.state.K == 8
    assert ctrl.state.N == 20
    assert ctrl.state.alpha == 0.9
    assert ctrl.state.beta == 0.95
    assert ctrl.state.lr == 1e-4
    assert ctrl.state.active_layer_strategy == "middle_random"
    assert ctrl.state.total_cycles == 42
    assert ctrl.state.accepted_count == 30
    assert ctrl.state.rolled_back_count == 12


def test_restore_state_resets_last_accel_action():
    """restore_state resets last_accel_action to 0."""
    from tg_lora.random_walk_controller import ControllerState

    ctrl = RandomWalkController(
        enable_random_walk=True,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    ctrl.adapt_to_acceleration(acceleration=1.0)
    assert ctrl.last_accel_action == 1

    ctrl.restore_state(ControllerState(
        K=3, N=5, alpha=0.3, beta=0.8, lr=5e-4,
        active_layer_strategy="last_25_percent_plus_random_2",
        relative_update_cap=0.005,
    ))
    assert ctrl.last_accel_action == 0


def test_restore_state_preserves_config():
    """restore_state keeps controller config (candidates, bounds)."""
    from tg_lora.random_walk_controller import ControllerState

    ctrl = RandomWalkController(
        K_candidates=[2, 4, 8],
        N_candidates=[1, 3, 7],
        alpha_min=0.01,
        alpha_max=2.0,
        lr_min=1e-6,
        lr_max=1e-2,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    saved = ControllerState(
        K=8, N=7, alpha=1.5, beta=0.9, lr=5e-3,
        active_layer_strategy="lisa_like_weighted",
        relative_update_cap=0.003,
    )
    ctrl.restore_state(saved)

    # Config is preserved
    assert ctrl.K_candidates == [2, 4, 8]
    assert ctrl.N_candidates == [1, 3, 7]
    assert ctrl.alpha_min == 0.01
    assert ctrl.alpha_max == 2.0
    assert ctrl.lr_min == 1e-6
    assert ctrl.lr_max == 1e-2
    # State is from saved
    assert ctrl.state.K == 8
    assert ctrl.state.alpha == 1.5


def test_restore_state_propose_uses_restored_lr():
    """After restore_state, propose() uses the restored lr."""
    from tg_lora.random_walk_controller import ControllerState

    ctrl = RandomWalkController(
        lr_initial=5e-4,
        lr_explore_prob=0.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
    )
    saved = ControllerState(
        K=3, N=5, alpha=0.3, beta=0.8, lr=7e-4,
        active_layer_strategy="last_25_percent_plus_random_2",
        relative_update_cap=0.005,
    )
    ctrl.restore_state(saved)

    proposal = ctrl.propose()
    assert proposal.lr == 7e-4


# --- Coverage gap: lines 206, 372-377 ---


def test_accel_deadzone_custom_value():
    """Line 206: custom accel_deadzone is stored."""
    ctrl = RandomWalkController(
        accel_deadzone=0.05,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_deadzone == 0.05


def test_accel_deadzone_default():
    """Default accel_deadzone is _DEFAULT_ACCEL_DEADZONE."""
    ctrl = RandomWalkController(
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_deadzone == RandomWalkController._DEFAULT_ACCEL_DEADZONE


def test_commit_proposal_adopted():
    """Lines 372-377: commit_proposal writes proposal fields into state."""
    from tg_lora.random_walk_controller import Proposal

    ctrl = RandomWalkController(
        K_initial=3,
        N_initial=5,
        alpha_initial=0.3,
        beta_initial=0.8,
        lr_initial=5e-4,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    proposal = Proposal(
        K=8,
        N=10,
        alpha=1.0,
        beta=0.95,
        lr=8e-4,
        active_layer_strategy="last_25_percent",
        relative_update_cap=0.01,
    )
    ctrl.commit_proposal(proposal)
    assert ctrl.state.K == 8
    assert ctrl.state.N == 10
    assert ctrl.state.alpha == 1.0
    assert ctrl.state.beta == 0.95
    assert ctrl.state.lr == 8e-4
    assert ctrl.state.active_layer_strategy == "last_25_percent"


def test_accel_deadzone_zero_accepted():
    """accel_deadzone=0.0 is valid (finite, non-negative)."""
    ctrl = RandomWalkController(
        accel_deadzone=0.0,
        k_explore_prob=0.0,
        n_explore_prob=0.0,
        beta_explore_prob=0.0,
        strategy_explore_prob=0.0,
        lr_explore_prob=0.0,
    )
    assert ctrl.accel_deadzone == 0.0


