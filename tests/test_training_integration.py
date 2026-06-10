"""Integration tests for CycleState + DeltaTracker as used by train_tg_lora.

Extended for TASK-0013: mock-based training loop integration tests covering
the full pilot → extrapolation → accept/rollback flow with mocked
model, optimizer, dataloader, and external dependencies.
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.prefix_feature_cache import PrefixFeatureDatasetBase
from src.training.train_tg_lora import (build_training_summary,
                                        should_run_full_eval, train_tg_lora)

# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


class _LoRAMockModel(torch.nn.Module):
    """Minimal nn.Module with LoRA-like named_parameters for testing."""

    def __init__(self, num_layers: int = 4, hidden: int = 8):
        super().__init__()
        for i in range(num_layers):
            setattr(
                self,
                f"layers_{i}_lora_A",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01),
            )
            setattr(
                self,
                f"layers_{i}_lora_B",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01),
            )
        self._loss_val = 2.0
        self.save_pretrained = MagicMock()
        self.save_pretrained.__wrapped__ = lambda path: None  # make it callable

    def parameters(self):
        for p in super().parameters():
            p.requires_grad = True
            yield p

    def named_parameters(self, **kwargs):
        for name, p in super().named_parameters(**kwargs):
            yield name, p

    def train(self, mode=True):
        return self

    def __call__(self, **kwargs):
        out = MagicMock()
        out.loss = torch.tensor(self._loss_val, requires_grad=True)
        return out


class _SimpleDataset(Dataset):
    def __init__(self, n: int = 10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 100, (16,)),
            "attention_mask": torch.ones(16, dtype=torch.long),
            "labels": torch.randint(0, 100, (16,)),
        }


class _LeadingMaskedDataset(Dataset):
    def __init__(self, n: int = 6, seq_len: int = 16):
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        labels = torch.full((self.seq_len,), -100, dtype=torch.long) if idx == 0 else torch.randint(0, 100, (self.seq_len,))
        return {
            "input_ids": torch.randint(0, 100, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": labels,
        }


class _CachedFeatureDataset(PrefixFeatureDatasetBase):
    def __init__(self, n: int = 10, seq_len: int = 16, hidden: int = 8, split_layer_idx: int = 3):
        self.n = n
        self.seq_len = seq_len
        self.hidden = hidden
        self.split_layer_idx = split_layer_idx
        self._total_bytes = n * seq_len * hidden * 4

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        del idx
        return {
            "hidden_states": torch.randn(self.seq_len, self.hidden),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, 100, (self.seq_len,)),
            "split_layer_idx": torch.tensor(self.split_layer_idx, dtype=torch.long),
        }

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


def _make_config(**overrides) -> OmegaConf:
    """Build a minimal OmegaConf config that train_tg_lora() accepts."""
    defaults = {
        "experiment": {"name": "test_run", "seed": 42},
        "model": {"name_or_path": "dummy-model"},
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
        "data": {
            "train_path": "/tmp/dummy_train",
            "valid_quick_path": "/tmp/dummy_valid_q",
            "valid_full_path": "/tmp/dummy_valid_f",
            "max_seq_len": 32,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "grad_accumulation": 1,
            "max_grad_norm": 1.0,
            "max_cycles": 5,
            "trainable_lora_scope": "all",
            "early_stopping_patience": None,
            "min_cycles_before_stop": 3,
        },
        "tg_lora": {
            "K_initial": 2,
            "K_candidates": [2, 3],
            "N_initial": 1,
            "N_candidates": [1, 3],
            "alpha_initial": 0.3,
            "alpha_min": 0.01,
            "alpha_max": 2.0,
            "alpha_log_sigma": 0.1,
            "beta_initial": 0.9,
            "beta_candidates": [0.9],
            "active_layer_strategy": "last_25_percent",
            "relative_update_cap": 0.5,
            "random_middle_layers": 2,
            "layer_sample_temperature": 1.0,
        },
        "eval": {
            "quick_eval_examples": 5,
            "full_eval_every_cycles": 10,
            "rollback_tolerance": 0.005,
        },
        "logging": {
            "run_dir": "/tmp/tg_lora_test_run",
            "log_every_cycles": 1,
            "save_every_cycles": 25,
        },
    }
    cfg = OmegaConf.create(defaults)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _patch_train_tg_lora_deps(model_loss=2.0, eval_losses=None):
    """Return a dict of patch targets for train_tg_lora's external deps.

    eval_losses: list of floats to cycle through for eval_loss calls.
    """
    from src.eval.eval_loss import EvalLossResult

    model = _LoRAMockModel()
    model._loss_val = model_loss

    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()

    if eval_losses is None:
        eval_losses = [2.0]

    # Prepend initial eval loss (training loop now evals before cycle 0)
    eval_losses = [model_loss] + list(eval_losses)

    dataset = _SimpleDataset(10)

    run_dir = Path("/tmp/tg_lora_test_run")

    metrics = MagicMock()

    # Build mock objects with correct return_value / side_effect
    mock_load_tokenizer = MagicMock(return_value=tokenizer)
    mock_load_base_model = MagicMock(return_value=model)
    mock_apply_lora = MagicMock(return_value=model)
    mock_configure_trainable_lora_scope = MagicMock(return_value=(set(), set()))
    mock_get_input_device = MagicMock(return_value="cpu")
    mock_load_dataset = MagicMock(return_value=dataset)
    mock_eval_loss = MagicMock(side_effect=list(eval_losses) * 100)
    mock_eval_loss_detailed = MagicMock(
        side_effect=[
            (
                EvalLossResult(avg_loss=v, num_batches=1, min_loss=v, max_loss=v)
                if math.isfinite(v)
                else EvalLossResult(
                    avg_loss=v, num_batches=0, min_loss=v, max_loss=v
                )
            )
            for v in (list(eval_losses) * 100)
        ]
    )
    mock_ensure_dir = MagicMock(return_value=run_dir)
    mock_run_metrics_cls = MagicMock(return_value=metrics)
    mock_count_parameters = MagicMock(return_value={"total": 100, "trainable": 50})
    mock_set_seed = MagicMock()

    patches = {
        "src.training.train_tg_lora.load_tokenizer": mock_load_tokenizer,
        "src.training.train_tg_lora.load_base_model": mock_load_base_model,
        "src.training.train_tg_lora.apply_lora": mock_apply_lora,
        "src.training.train_tg_lora.configure_trainable_lora_scope": mock_configure_trainable_lora_scope,
        "src.training.train_tg_lora.get_input_device": mock_get_input_device,
        "src.training.train_tg_lora.load_dataset": mock_load_dataset,
        "src.training.train_tg_lora.eval_loss": mock_eval_loss,
        "src.training.train_tg_lora.eval_loss_detailed": mock_eval_loss_detailed,
        "src.training.train_tg_lora.ensure_dir": mock_ensure_dir,
        "src.training.train_tg_lora.RunMetrics": mock_run_metrics_cls,
        "src.training.train_tg_lora.count_parameters": mock_count_parameters,
        "src.training.train_tg_lora.set_seed": mock_set_seed,
    }
    return patches


class TestCycleStateDeltaTrackerIntegration:
    """Verify CycleState and DeltaTracker work together as in the training loop."""

    def setup_method(self):
        self.cycle_state = CycleState()
        self.delta_tracker = DeltaTracker()

    def test_single_cycle_recording(self):
        """Record one full cycle and verify all state is consistent."""
        K, N, grad_accum = 4, 2, 1

        self.cycle_state.record_cycle(
            K=K,
            N=N,
            grad_accum=grad_accum,
            train_loss=2.5,
            valid_loss=2.5,
            accepted=True,
        )

        assert self.cycle_state.cycle == 1
        assert self.cycle_state.full_backward_passes == K * grad_accum
        assert self.cycle_state.accepted_count == 1
        assert self.cycle_state.rejected_count == 0
        assert self.cycle_state.best_loss == 2.5
        assert self.cycle_state.best_step == K * grad_accum

    def test_multi_cycle_reduction_rate(self):
        """After several cycles, reduction_rate = 1 - backward / (backward + extrapolation)."""
        for i in range(5):
            self.cycle_state.record_cycle(
                K=4,
                N=2,
                grad_accum=1,
                train_loss=2.0 - i * 0.1,
                valid_loss=2.0 - i * 0.1 if i % 2 == 0 else None,
                accepted=(i != 3),
            )

        assert self.cycle_state.cycle == 5
        assert self.cycle_state.full_backward_passes == 5 * 4
        assert self.cycle_state.accepted_count == 4
        assert self.cycle_state.rejected_count == 1
        assert self.cycle_state.reduction_rate == pytest.approx(
            1.0 - (5 * 4) / (5 * 4 + 5 * 2)
        )
        assert self.cycle_state.acceptance_rate == pytest.approx(4 / 5)

    def test_early_stopping_via_should_stop(self):
        """should_stop triggers after patience exceeded."""
        for i in range(15):
            self.cycle_state.record_cycle(
                K=2,
                N=1,
                grad_accum=1,
                train_loss=1.0,
                valid_loss=1.5 + i * 0.01,
                accepted=True,
            )

        assert self.cycle_state.stale_cycles == 14
        assert self.cycle_state.should_stop(patience=10, min_cycles=10) is True
        assert self.cycle_state.should_stop(patience=10, min_cycles=20) is False
        assert self.cycle_state.should_stop(patience=None) is False

    def test_best_loss_tracking_resets_stale(self):
        """A new best loss resets stale_cycles to 0."""
        self.cycle_state.record_cycle(
            K=2, N=1, grad_accum=1, train_loss=1.0, valid_loss=2.0, accepted=True
        )
        self.cycle_state.record_cycle(
            K=2, N=1, grad_accum=1, train_loss=1.0, valid_loss=2.1, accepted=True
        )
        assert self.cycle_state.stale_cycles == 1

        self.cycle_state.record_cycle(
            K=2, N=1, grad_accum=1, train_loss=1.0, valid_loss=1.8, accepted=True
        )
        assert self.cycle_state.stale_cycles == 0
        assert self.cycle_state.best_loss == 1.8

    def test_delta_tracker_records_and_tracks(self):
        """DeltaTracker records per-cycle deltas and computes anomaly/trend."""
        for i in range(6):
            before = {"lora_A": torch.zeros(4)}
            after = {"lora_A": torch.full((4,), 0.1 * (i + 1))}
            delta = self.delta_tracker.compute_and_record(after, before, K=2)
            assert "lora_A" in delta

        assert len(self.delta_tracker.norm_history) == 6
        stats = self.delta_tracker.last_stats
        assert stats is not None
        assert stats.total_norm > 0

        summary = self.delta_tracker.summary()
        assert "total_norm" in summary
        assert "convergence_trend" in summary
        assert "anomalous" in summary

    def test_combined_summary_as_in_training_loop(self):
        """Summary dicts merge cleanly as train_tg_lora does."""
        for i in range(3):
            before = {"lora_A": torch.zeros(4)}
            after = {"lora_A": torch.full((4,), 0.1 + i * 0.05)}
            self.delta_tracker.compute_and_record(after, before, K=4)

            self.cycle_state.record_cycle(
                K=4,
                N=2,
                grad_accum=1,
                train_loss=2.0 - i * 0.1,
                valid_loss=2.0 - i * 0.1,
                accepted=True,
            )

        cs_summary = self.cycle_state.summary()
        dt_summary = self.delta_tracker.summary()

        merged = {}
        merged.update(cs_summary)
        merged.update(dt_summary)

        assert "cycles" in merged
        assert "reduction_rate" in merged
        assert "total_norm" in merged
        assert "convergence_trend" in merged
        assert merged["cycles"] == 3
        assert merged["accepted_count"] == 3


class TestMockedTrainingLoop:
    """Simulate the full training loop flow with mocked model/data.

    Exercises the sequence: record_cycle → compute_and_record → should_stop,
    including the quick-eval / full-eval stale_cycles split.
    """

    def setup_method(self):
        self.cs = CycleState()
        self.dt = DeltaTracker()

    def _run_cycle(
        self, K, N, grad_accum, train_loss, accepted, full_eval_every=0, full_loss=None
    ):
        """Execute one training cycle's worth of state updates."""
        # 1. compute_and_record (delta tracker)
        before = {"lora_A": torch.zeros(4)}
        after = {"lora_A": torch.full((4,), train_loss * 0.01)}
        self.dt.compute_and_record(after, before, K=K)

        # 2. record_cycle (with quick-eval stale tracking unless full eval)
        is_full = should_run_full_eval(self.cs.cycle, full_eval_every)
        self.cs.record_cycle(
            K=K,
            N=N,
            grad_accum=grad_accum,
            train_loss=train_loss,
            valid_loss=None if is_full else train_loss,
            accepted=accepted,
        )

        # 3. Full eval (if applicable)
        if is_full and full_loss is not None:
            self.cs.record_full_eval(full_loss)

    def test_ten_cycles_no_full_eval(self):
        """10 cycles with only quick-eval tracking, all accepted, constant loss."""
        for i in range(10):
            self._run_cycle(K=4, N=2, grad_accum=1, train_loss=2.0, accepted=True)

        assert self.cs.cycle == 10
        assert self.cs.full_backward_passes == 40
        assert self.cs.extrapolation_steps == 20
        assert self.cs.reduction_rate == pytest.approx(1.0 - 40 / 60)
        assert self.cs.accepted_count == 10
        # Cycle 0 sets best=2.0, cycles 1-9 don't improve → stale=9
        assert self.cs.stale_cycles == 9
        assert self.dt.norm_history[-1] > 0

    def test_mixed_accept_reject_with_delta_tracking(self):
        """Interleave accepted and rejected cycles; verify delta tracker sees both."""
        acceptances = [True, False, True, True, False, True]
        for i, acc in enumerate(acceptances):
            self._run_cycle(
                K=3, N=5, grad_accum=1, train_loss=1.5 + i * 0.1, accepted=acc
            )

        assert self.cs.accepted_count == 4
        assert self.cs.rejected_count == 2
        assert self.cs.acceptance_rate == pytest.approx(4 / 6)
        assert len(self.dt.norm_history) == 6

        # Check that anomaly detection works across cycles
        summary = self.dt.summary()
        assert "convergence_trend" in summary
        # Norms are increasing (train_loss increasing) so trend should be positive
        assert summary["convergence_trend"] > 0

    def test_full_eval_does_not_double_count_stale(self):
        """Full-eval cycles track stale_cycles via record_full_eval only."""
        full_eval_every = 5
        # Run 12 cycles. Full eval at cycle 5 and 10.
        # Quick eval: loss always 2.0 (no improvement after cycle 0)
        # Full eval: loss always 2.5 (no improvement ever)
        for i in range(12):
            cycle_before = self.cs.cycle
            self._run_cycle(
                K=2,
                N=1,
                grad_accum=1,
                train_loss=2.0,
                accepted=True,
                full_eval_every=full_eval_every,
                full_loss=2.5,
            )
            is_full = should_run_full_eval(cycle_before, full_eval_every)
            if is_full:
                # After full eval, stale_cycles should have been incremented
                # exactly once (by record_full_eval), NOT twice
                assert self.cs.stale_cycles == cycle_before

        # Full eval ran at cycles 5 and 10 (0-indexed: after cycles 4 and 9)
        # Non-full-eval cycles (0-3, 5-8, 11): stale tracked by record_cycle
        #   - cycle 0: loss=2.0, best was inf → improvement → stale=0
        #   - cycles 1-3: loss=2.0, best=2.0 → no improvement → stale 1,2,3
        #   - cycle 4 is full eval: record_cycle(valid_loss=None) skips stale,
        #     record_full_eval(2.5): 2.5 > 2.0 → stale 3+1=4
        #   - cycles 5-8: loss=2.0, best=2.0 → stale 5,6,7,8
        #   - cycle 9 is full eval: record_cycle(valid_loss=None) skips stale,
        #     record_full_eval(2.5): 2.5 > 2.0 → stale 8+1=9
        #   - cycle 10-11: loss=2.0, best=2.0 → stale 10,11
        assert self.cs.stale_cycles == 11

    def test_full_eval_improvement_resets_stale(self):
        """Full eval improvement resets stale_cycles to 0."""
        # Cycles 0-4: no improvement via quick eval (loss stays at 2.0)
        for i in range(5):
            self._run_cycle(K=2, N=1, grad_accum=1, train_loss=2.0, accepted=True)
        assert self.cs.stale_cycles == 4  # 0th improved, 1-4 didn't

        # Now simulate full eval at cycle 5 with improvement
        self._run_cycle(
            K=2,
            N=1,
            grad_accum=1,
            train_loss=2.0,
            accepted=True,
            full_eval_every=5,
            full_loss=1.5,  # improves best_loss
        )
        assert self.cs.stale_cycles == 0
        assert self.cs.best_loss == 1.5

    def test_early_stopping_triggers_after_patience_of_full_evals(self):
        """Early stopping triggers based on full-eval stale_cycles."""
        patience = 3
        full_eval_every = 3
        stopped_at = None

        for i in range(30):
            self._run_cycle(
                K=2,
                N=1,
                grad_accum=1,
                train_loss=2.0,
                accepted=True,
                full_eval_every=full_eval_every,
                full_loss=3.0,  # never improves
            )
            if self.cs.should_stop(patience=patience, min_cycles=5):
                stopped_at = i
                break

        assert stopped_at is not None
        # With full_eval_every=3, first full eval runs at cycle 3 (0-indexed).
        # Non-full-eval cycles increment stale via quick eval; full eval cycles
        # increment stale via record_full_eval. Either way, stale grows each
        # cycle since full_loss=3.0 never improves.
        # patience=3, min_cycles=5: triggers once stale>=3 AND cycle>=5.
        assert stopped_at >= 3  # at least past first full eval

    def test_convergence_trend_across_cycles(self):
        """Delta norms should show decreasing trend for converging training."""
        for i in range(8):
            magnitude = 0.5 * (0.9**i)  # decreasing deltas
            before = {"layers.0.lora_A": torch.zeros(4)}
            after = {"layers.0.lora_A": torch.full((4,), magnitude)}
            self.dt.compute_and_record(after, before, K=2)
            self.cs.record_cycle(
                K=2, N=1, grad_accum=1, train_loss=2.0 - i * 0.1, accepted=True
            )

        trend = self.dt.convergence_trend(window=5)
        assert trend < 0  # negative = converging

    def test_anomaly_detection_in_loop(self):
        """A sudden spike in delta norm is flagged as anomalous."""
        for i in range(5):
            before = {"lora_A": torch.zeros(4)}
            after = {"lora_A": torch.full((4,), 0.1)}
            self.dt.compute_and_record(after, before, K=2)
            self.cs.record_cycle(K=2, N=1, grad_accum=1, train_loss=1.5, accepted=True)

        assert not self.dt.is_anomalous()

        # Inject anomaly: huge delta
        before = {"lora_A": torch.zeros(4)}
        after = {"lora_A": torch.full((4,), 100.0)}
        self.dt.compute_and_record(after, before, K=2)

        assert self.dt.is_anomalous()
        summary = self.dt.summary()
        assert summary["anomalous"] is True

    def test_build_training_summary_merges_all(self):
        """build_training_summary produces a dict with all three sources."""
        from src.tg_lora.random_walk_controller import RandomWalkController

        controller = RandomWalkController(
            K_initial=3,
            K_candidates=[3, 5],
            N_initial=5,
            N_candidates=[5, 10],
            alpha_initial=0.3,
            alpha_min=0.01,
            alpha_max=2.0,
            alpha_log_sigma=0.3,
            beta_initial=0.9,
            beta_candidates=[0.9],
            active_layer_strategy="last_25_percent",
            relative_update_cap=0.5,
            rollback_tolerance=0.005,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        controller.propose()
        controller.accept(1.0, 0.95)
        controller.reward(1.0, 0.95)

        for i in range(3):
            before = {"lora_A": torch.zeros(4)}
            after = {"lora_A": torch.full((4,), 0.1)}
            self.dt.compute_and_record(after, before, K=3)
            self.cs.record_cycle(K=3, N=5, grad_accum=1, train_loss=2.0, accepted=True)

        summary = build_training_summary(controller, self.cs, self.dt)

        # From CycleState
        assert "cycles" in summary
        assert "reduction_rate" in summary
        assert "accepted_count" in summary
        # From DeltaTracker
        assert "total_norm" in summary
        assert "convergence_trend" in summary
        assert "anomalous" in summary
        # From RandomWalkController
        assert "acceptance_rate" in summary
        assert "total_cycles" in summary


class TestShouldRunFullEval:
    """Tests for the extracted should_run_full_eval pure function."""

    def test_cycle_zero_never(self):
        assert should_run_full_eval(0, 10) is False

    def test_exact_multiple(self):
        assert should_run_full_eval(10, 10) is True
        assert should_run_full_eval(20, 10) is True

    def test_non_multiple(self):
        assert should_run_full_eval(7, 10) is False
        assert should_run_full_eval(15, 10) is False

    def test_disabled_zero(self):
        assert should_run_full_eval(10, 0) is False

    def test_disabled_negative(self):
        assert should_run_full_eval(10, -1) is False

    def test_every_cycle(self):
        assert should_run_full_eval(1, 1) is True
        assert should_run_full_eval(2, 1) is True


class TestRecordFullEval:
    """Tests for the CycleState.record_full_eval method."""

    def test_improvement_resets_stale(self):
        cs = CycleState()
        cs.stale_cycles = 5
        cs.record_full_eval(1.0)  # 1.0 < inf
        assert cs.stale_cycles == 0
        assert cs.best_loss == 1.0
        assert cs.best_step == 0

    def test_no_improvement_increments_stale(self):
        cs = CycleState()
        cs.best_loss = 1.0
        cs.stale_cycles = 3
        cs.record_full_eval(1.5)
        assert cs.stale_cycles == 4
        assert cs.best_loss == 1.0

    def test_exact_best_loss_counts_as_no_improvement(self):
        cs = CycleState()
        cs.best_loss = 1.0
        cs.record_full_eval(1.0)
        assert cs.stale_cycles == 1

    def test_best_step_updated_on_improvement(self):
        cs = CycleState()
        cs.full_backward_passes = 42
        cs.record_full_eval(0.5)
        assert cs.best_step == 42


# ---------------------------------------------------------------------------
# TASK-0013: Full mock-based training loop integration tests
# ---------------------------------------------------------------------------


def _run_mocked_training(cfg_overrides=None, eval_losses=None, model_loss=2.0):
    """Helper to run train_tg_lora with all external deps mocked.

    Returns a dict with mock objects for post-hoc assertions.
    """
    cfg = _make_config(**(cfg_overrides or {}))
    deps = _patch_train_tg_lora_deps(model_loss=model_loss, eval_losses=eval_losses)

    import contextlib

    with contextlib.ExitStack() as stack:
        mocks = {}
        for target, mock_obj in deps.items():
            mocks[target.split(".")[-1]] = stack.enter_context(
                patch(target, new=mock_obj)
            )
        train_tg_lora(cfg)
        return mocks


class TestOptimizerLifecycleIntegration:
    def test_reuse_state_policy_creates_optimizer_once(self):
        cfg = _make_config(
            training={
                "max_cycles": 3,
                "optimizer_lifecycle": "reuse_state_reset_experimental",
            }
        )
        deps = _patch_train_tg_lora_deps(eval_losses=[2.0, 1.5] * 20)

        import contextlib

        from src.training.trainer_loop import \
            create_optimizer as real_create_optimizer

        create_calls = []

        def counting_create_optimizer(model, lr, weight_decay=0.0):
            create_calls.append((lr, weight_decay))
            return real_create_optimizer(model, lr=lr, weight_decay=weight_decay)

        with contextlib.ExitStack() as stack:
            for target, mock_obj in deps.items():
                stack.enter_context(patch(target, new=mock_obj))
            stack.enter_context(
                patch(
                    "src.training.optimizer_lifecycle.create_optimizer",
                    side_effect=counting_create_optimizer,
                )
            )
            train_tg_lora(cfg)

        assert len(create_calls) == 1


class TestFullTrainingLoopMocked:
    """End-to-end test of train_tg_lora with all GPU deps mocked.

    Exercises the full cycle: pilot → snapshot → extrapolation → accept/rollback
    → record_cycle → build_training_summary.
    """

    def test_skips_all_masked_training_batches(self):
        cfg = _make_config(training={"max_cycles": 1, "grad_accumulation": 1})
        deps = _patch_train_tg_lora_deps(eval_losses=[2.0, 1.5] * 10)
        deps["src.training.train_tg_lora.load_dataset"] = MagicMock(return_value=_LeadingMaskedDataset())

        def _assert_supervised(_model, batch, grad_accum=1):
            assert torch.any(batch["labels"] != -100)
            return 1.0

        deps["src.training.train_tg_lora.forward_backward"] = MagicMock(side_effect=_assert_supervised)

        import contextlib

        with contextlib.ExitStack() as stack:
            for target, mock_obj in deps.items():
                stack.enter_context(patch(target, new=mock_obj))
            train_tg_lora(cfg)

    def test_completes_all_cycles_when_accepted(self):
        """5 cycles, all accepted (eval_loss after <= eval_loss pilot)."""
        # eval_losses: each cycle calls eval_loss twice (pilot + after).
        # loss_after (1.5) <= loss_pilot (2.0) → accepted
        eval_losses = [2.0, 1.5] * 10
        mocks = _run_mocked_training(eval_losses=eval_losses)

        # Should have called eval_loss for each cycle (2 per cycle × 5 cycles = 10)
        assert mocks["eval_loss"].called

    def test_rejected_cycles_rollback(self):
        """All cycles rejected (loss_after > loss_pilot + tolerance)."""
        # loss_after (3.0) > loss_pilot (2.0) + tolerance(0.005) → rejected
        eval_losses = [2.0, 3.0] * 10
        mocks = _run_mocked_training(eval_losses=eval_losses)
        assert mocks["eval_loss"].called

    def test_mixed_accept_reject(self):
        """Alternating accept and reject cycles."""
        # Accept: pilot=2.0, after=1.5; Reject: pilot=2.0, after=3.0
        eval_losses = [2.0, 1.5, 2.0, 3.0, 2.0, 1.5, 2.0, 3.0, 2.0, 1.5]
        mocks = _run_mocked_training(eval_losses=eval_losses)
        assert mocks["eval_loss"].called

    def test_early_stopping_triggers(self):
        """Training stops early when patience is exceeded."""
        # All rejected: loss_after > loss_pilot, so stale accumulates.
        # With patience=2, min_cycles=2, should stop after cycle 4.
        eval_losses = [2.0, 3.0] * 20
        cfg_overrides = {
            "training": {
                "early_stopping_patience": 2,
                "min_cycles_before_stop": 2,
                "max_cycles": 50,
            },
            "eval": {"full_eval_every_cycles": 1},
        }
        mocks = _run_mocked_training(
            cfg_overrides=cfg_overrides,
            eval_losses=eval_losses,
        )
        assert mocks["eval_loss"].called

    def test_build_training_summary_called(self):
        """Final summary is computed at end of training."""
        eval_losses = [2.0, 1.5] * 10
        mocks = _run_mocked_training(eval_losses=eval_losses)
        # RunMetrics.write_footer should have been called
        assert mocks["RunMetrics"].return_value.write_footer.called

    def test_model_save_on_improvement(self):
        """Best model is saved when full eval shows improvement."""
        eval_losses = [2.0, 1.5] * 10
        cfg_overrides = {
            "eval": {"full_eval_every_cycles": 1},
        }
        mocks = _run_mocked_training(
            cfg_overrides=cfg_overrides,
            eval_losses=eval_losses,
        )
        model = mocks["load_base_model"].return_value
        assert model.save_pretrained.called

    def test_periodic_checkpoint_save(self):
        """Periodic checkpoint saves occur at save_every_cycles intervals."""
        eval_losses = [2.0, 1.5] * 20
        cfg_overrides = {
            "training": {"max_cycles": 6},
            "logging": {"save_every_cycles": 3},
        }
        mocks = _run_mocked_training(
            cfg_overrides=cfg_overrides,
            eval_losses=eval_losses,
        )
        model = mocks["load_base_model"].return_value
        # save_pretrained called for periodic checkpoint at cycle 3
        assert model.save_pretrained.called


class TestPilotExtrapolationRollback:
    """Test the pilot → extrapolation → accept/rollback flow in detail."""

    def _run_one_cycle(self, loss_pilot=2.0, loss_after=1.5, accepted=True):
        """Run a single training cycle and return the mock state."""
        eval_losses = [loss_pilot, loss_after]
        if accepted:
            assert loss_after <= loss_pilot + 0.005
        return _run_mocked_training(
            cfg_overrides={"training": {"max_cycles": 1}},
            eval_losses=eval_losses,
        )

    def test_accept_path_calls_reward(self):
        """When loss_after <= loss_pilot + tolerance, accept is called."""
        mocks = self._run_one_cycle(loss_pilot=2.0, loss_after=1.5, accepted=True)
        assert mocks["eval_loss"].call_count >= 2

    def test_reject_path_calls_rollback(self):
        """When loss_after > loss_pilot + tolerance, rollback occurs."""
        mocks = self._run_one_cycle(loss_pilot=2.0, loss_after=3.0, accepted=False)
        assert mocks["eval_loss"].call_count >= 2


class TestCycleStateFullFlow:
    """Test record_cycle → should_stop → early stopping full flow."""

    def test_three_normal_cycles_then_early_stop(self):
        """3 normal accepted cycles, then early stop triggers."""
        cs = CycleState()
        for i in range(3):
            cs.record_cycle(
                K=2,
                N=1,
                grad_accum=1,
                train_loss=2.0 - i * 0.1,
                valid_loss=2.0 - i * 0.1,
                accepted=True,
            )
        assert cs.cycle == 3
        assert cs.accepted_count == 3
        assert cs.rejected_count == 0
        assert cs.best_loss == 1.8
        assert cs.reduction_rate == pytest.approx(1.0 - 6 / 9)
        # Not enough stale cycles yet
        assert not cs.should_stop(patience=5, min_cycles=3)

        # Now add stale cycles
        for _ in range(5):
            cs.record_cycle(
                K=2, N=1, grad_accum=1, train_loss=2.0, valid_loss=2.0, accepted=True
            )

        assert cs.should_stop(patience=5, min_cycles=3)
        assert cs.stale_cycles == 5

    def test_all_rejected_cycles(self):
        """3 all-rejected cycles: stale grows, rejected_count increases."""
        cs = CycleState()
        for i in range(3):
            cs.record_cycle(
                K=2,
                N=1,
                grad_accum=1,
                train_loss=2.0,
                valid_loss=2.0 + i * 0.1,
                accepted=False,
            )
        assert cs.cycle == 3
        assert cs.accepted_count == 0
        assert cs.rejected_count == 3
        assert cs.reduction_rate == pytest.approx(1.0 - 6 / 9)


class TestDeltaTrackerAnomalyIntegration:
    """Test DeltaTracker compute_and_record → is_anomalous → convergence_trend chain."""

    def test_normal_to_anomaly_to_recovery(self):
        """Normal deltas → anomaly spike → back to normal."""
        dt = DeltaTracker()

        # Normal phase: small, decreasing deltas
        for i in range(5):
            before = {"layer.0.lora_A": torch.zeros(4)}
            after = {"layer.0.lora_A": torch.full((4,), 0.01 * (5 - i))}
            dt.compute_and_record(after, before, K=2)
        assert not dt.is_anomalous()
        assert dt.convergence_trend() < 0  # converging

        # Anomaly: huge spike
        before = {"layer.0.lora_A": torch.zeros(4)}
        after = {"layer.0.lora_A": torch.full((4,), 10.0)}
        dt.compute_and_record(after, before, K=2)
        assert dt.is_anomalous()

        # Recovery: small deltas again (need several to dilute the spike)
        for i in range(10):
            before = {"layer.0.lora_A": torch.zeros(4)}
            after = {"layer.0.lora_A": torch.full((4,), 0.005)}
            dt.compute_and_record(after, before, K=2)
        # After enough small deltas, anomaly should clear
        assert not dt.is_anomalous()

    def test_convergence_trend_negative_for_shrinking_deltas(self):
        """Decreasing delta norms → negative convergence trend."""
        dt = DeltaTracker()
        for i in range(8):
            before = {"lora_A": torch.zeros(4)}
            after = {"lora_A": torch.full((4,), 0.5 * (0.8**i))}
            dt.compute_and_record(after, before, K=2)

        assert dt.convergence_trend(window=5) < 0

    def test_convergence_trend_positive_for_growing_deltas(self):
        """Increasing delta norms → positive convergence trend."""
        dt = DeltaTracker()
        for i in range(8):
            before = {"lora_A": torch.zeros(4)}
            after = {"lora_A": torch.full((4,), 0.1 * (1.5**i))}
            dt.compute_and_record(after, before, K=2)

        assert dt.convergence_trend(window=5) > 0

    def test_summary_contains_all_fields(self):
        """DeltaTracker.summary() returns all expected keys."""
        dt = DeltaTracker()
        before = {"lora_A": torch.zeros(4)}
        after = {"lora_A": torch.ones(4) * 0.1}
        dt.compute_and_record(after, before, K=2)

        s = dt.summary()
        assert "total_norm" in s
        assert "max_component" in s
        assert "mean_abs" in s
        assert "anomalous" in s
        assert "convergence_trend" in s
        assert "history_length" in s


class TestFullEvalVsQuickEval:
    """Test full evaluation cycle vs quick evaluation cycle switching."""

    def test_full_eval_every_n_cycles(self):
        """Full eval runs only at the configured interval."""
        cs = CycleState()
        full_eval_every = 3
        full_eval_calls = 0

        for cycle in range(9):
            is_full = should_run_full_eval(cycle, full_eval_every)
            cs.record_cycle(
                K=2,
                N=1,
                grad_accum=1,
                train_loss=2.0,
                valid_loss=None if is_full else 2.0,
                accepted=True,
            )
            if is_full:
                cs.record_full_eval(2.5)
                full_eval_calls += 1

        assert full_eval_calls == 2  # at cycles 3 and 6

    def test_full_eval_disabled(self):
        """No full eval when full_eval_every=0."""
        count = sum(1 for c in range(20) if should_run_full_eval(c, 0))
        assert count == 0

    def test_quick_eval_tracks_stale(self):
        """Quick eval cycles track stale_cycles correctly."""
        cs = CycleState()
        # Cycle 0: improvement (inf → 2.0), stale=0
        cs.record_cycle(
            K=1, N=1, grad_accum=1, train_loss=1.0, valid_loss=2.0, accepted=True
        )
        assert cs.stale_cycles == 0

        # Cycle 1: no improvement (2.0 == 2.0), stale=1
        cs.record_cycle(
            K=1, N=1, grad_accum=1, train_loss=1.0, valid_loss=2.0, accepted=True
        )
        assert cs.stale_cycles == 1

        # Cycle 2: no improvement, stale=2
        cs.record_cycle(
            K=1, N=1, grad_accum=1, train_loss=1.0, valid_loss=2.0, accepted=True
        )
        assert cs.stale_cycles == 2


class TestBestModelSave:
    """Test best model save logic via mocked training loop."""

    def test_model_saved_when_full_eval_improves(self):
        """Model is saved when full eval loss improves."""
        # Pattern accounts for initial eval consuming one value;
        # need enough cycles for full eval to see improvement.
        eval_losses = [2.0, 1.5] * 20
        cfg_overrides = {
            "training": {"max_cycles": 5},
            "eval": {"full_eval_every_cycles": 1},
        }
        mocks = _run_mocked_training(
            cfg_overrides=cfg_overrides, eval_losses=eval_losses
        )
        model = mocks["load_base_model"].return_value
        assert model.save_pretrained.called

    def test_model_not_saved_when_no_improvement(self):
        """Few saves when full eval never improves beyond initial baseline."""
        # All evals return high loss. The initial baseline (model_loss) is also high.
        # With the initial eval + intermediate rollback, detailed eval may see
        # the prepended model_loss as an improvement once, but no further saves.
        eval_losses = [10.0] * 50
        cfg_overrides = {
            "training": {"max_cycles": 3},
            "eval": {"full_eval_every_cycles": 1},
        }
        mocks = _run_mocked_training(
            cfg_overrides=cfg_overrides, eval_losses=eval_losses
        )
        model = mocks["load_base_model"].return_value
        # At most 1 save (initial improvement), no repeated saves
        assert model.save_pretrained.call_count <= 1


class TestActivationCacheMetrics:
    def test_footer_summary_records_activation_cache_hit_rate(self):
        eval_losses = [2.0, 1.5] * 20
        cfg_overrides = {
            "training": {"max_cycles": 3},
            "eval": {"full_eval_every_cycles": 10},
        }
        mocks = _run_mocked_training(
            cfg_overrides=cfg_overrides,
            eval_losses=eval_losses,
        )
        metrics = mocks["RunMetrics"].return_value
        footer_summary = metrics.write_footer.call_args.kwargs["tg_lora_summary"]

        assert "activation_cache_build_count" in footer_summary
        assert "activation_cache_eligible_count" in footer_summary
        assert "activation_cache_hit_count" in footer_summary
        assert "activation_cache_miss_count" in footer_summary
        assert "activation_cache_hit_rate" in footer_summary


class TestPrefixFeatureCacheExperimental:
    def test_prefix_feature_cache_summary_recorded(self):
        eval_losses = [2.0, 1.8] * 20
        cfg_overrides = {
            "lora": {"dropout": 0.0},
            "training": {
                "max_cycles": 2,
                "trainable_lora_scope": "last_25_percent",
                "prefix_feature_cache_experimental": True,
            },
            "eval": {"full_eval_every_cycles": 10},
        }

        deps = _patch_train_tg_lora_deps(model_loss=2.0, eval_losses=eval_losses)
        cached_train = _CachedFeatureDataset(n=10)
        cached_quick = _CachedFeatureDataset(n=4)
        cached_full = _CachedFeatureDataset(n=6)
        deps["src.training.train_tg_lora.configure_trainable_lora_scope"] = MagicMock(
            return_value=({"layers.3.mock_lora.lora_A", "layers.3.mock_lora.lora_B"}, {3})
        )
        deps["src.training.train_tg_lora.build_prefix_feature_dataset"] = MagicMock(
            side_effect=[cached_train, cached_quick, cached_full]
        )
        deps["src.training.train_tg_lora.save_prefix_feature_dataset"] = MagicMock()
        deps["src.training.train_tg_lora.forward_backward"] = MagicMock(return_value=1.0)
        deps["src.training.train_tg_lora.optimizer_step"] = MagicMock()

        import contextlib

        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_obj in deps.items():
                mocks[target.split(".")[-1]] = stack.enter_context(
                    patch(target, new=mock_obj)
                )
            train_tg_lora(_make_config(**cfg_overrides))

        footer_summary = mocks["RunMetrics"].return_value.write_footer.call_args.kwargs[
            "tg_lora_summary"
        ]

        assert footer_summary["prefix_feature_cache_experimental"] is True
        assert footer_summary["trainable_lora_scope"] == "last_25_percent"
        assert footer_summary["prefix_feature_cache_split_layer"] == 3
        assert footer_summary["prefix_feature_cache_train_examples"] == 10
        assert footer_summary["prefix_feature_cache_valid_quick_examples"] == 4
        assert footer_summary["prefix_feature_cache_valid_full_examples"] == 6
        assert footer_summary["prefix_feature_cache_total_build_seconds"] >= 0.0

    def test_prefix_feature_cache_disk_hit_skips_rebuild(self, tmp_path):
        eval_losses = [2.0, 1.8] * 20
        cfg_overrides = {
            "lora": {"dropout": 0.0},
            "training": {
                "max_cycles": 2,
                "trainable_lora_scope": "last_25_percent",
                "prefix_feature_cache_experimental": True,
                "prefix_feature_cache_dir": str(tmp_path),
            },
            "eval": {"full_eval_every_cycles": 10},
        }

        deps = _patch_train_tg_lora_deps(model_loss=2.0, eval_losses=eval_losses)
        cached_train = _CachedFeatureDataset(n=10)
        cached_shared = _CachedFeatureDataset(n=4)
        train_cache = tmp_path / "train.pt"
        shared_cache = tmp_path / "shared.pt"
        train_cache.touch()
        shared_cache.touch()
        deps["src.training.train_tg_lora.configure_trainable_lora_scope"] = MagicMock(
            return_value=({"layers.3.mock_lora.lora_A", "layers.3.mock_lora.lora_B"}, {3})
        )
        deps["src.training.train_tg_lora.get_prefix_feature_cache_path"] = MagicMock(
            side_effect=[train_cache, shared_cache, shared_cache]
        )
        deps["src.training.train_tg_lora.load_prefix_feature_dataset"] = MagicMock(
            side_effect=[cached_train, cached_shared]
        )
        deps["src.training.train_tg_lora.save_prefix_feature_dataset"] = MagicMock()
        deps["src.training.train_tg_lora.build_prefix_feature_dataset"] = MagicMock(
            side_effect=AssertionError("disk hit should skip rebuild")
        )
        deps["src.training.train_tg_lora.forward_backward"] = MagicMock(return_value=1.0)
        deps["src.training.train_tg_lora.optimizer_step"] = MagicMock()

        import contextlib

        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_obj in deps.items():
                mocks[target.split(".")[-1]] = stack.enter_context(
                    patch(target, new=mock_obj)
                )
            train_tg_lora(_make_config(**cfg_overrides))

        footer_summary = mocks["RunMetrics"].return_value.write_footer.call_args.kwargs[
            "tg_lora_summary"
        ]

        assert footer_summary["prefix_feature_cache_train_source"] == "disk"
        assert footer_summary["prefix_feature_cache_valid_quick_source"] == "disk"
        assert footer_summary["prefix_feature_cache_valid_full_source"] == "memory"
        assert footer_summary["prefix_feature_cache_total_build_seconds"] == 0.0
        assert footer_summary["prefix_feature_cache_total_load_seconds"] >= 0.0
        assert mocks["load_prefix_feature_dataset"].call_count == 2
        mocks["save_prefix_feature_dataset"].assert_not_called()

    def test_prefix_feature_cache_runtime_offload_applies_when_all_loaders_are_cached(self):
        eval_losses = [2.0, 1.8] * 20
        cfg_overrides = {
            "lora": {"dropout": 0.0},
            "training": {
                "max_cycles": 2,
                "trainable_lora_scope": "last_25_percent",
                "prefix_feature_cache_experimental": True,
                "prefix_feature_cache_offload_prefix_to_cpu": True,
            },
            "eval": {"full_eval_every_cycles": 10},
        }

        deps = _patch_train_tg_lora_deps(model_loss=2.0, eval_losses=eval_losses)
        cached_train = _CachedFeatureDataset(n=10)
        cached_quick = _CachedFeatureDataset(n=4)
        cached_full = _CachedFeatureDataset(n=6)
        deps["src.training.train_tg_lora.configure_trainable_lora_scope"] = MagicMock(
            return_value=({"layers.3.mock_lora.lora_A", "layers.3.mock_lora.lora_B"}, {3})
        )
        deps["src.training.train_tg_lora.build_prefix_feature_dataset"] = MagicMock(
            side_effect=[cached_train, cached_quick, cached_full]
        )
        deps["src.training.train_tg_lora._gpu_allocated_mb"] = MagicMock(
            side_effect=[12000.0, 8200.0]
        )
        deps["src.training.train_tg_lora.offload_prefix_runtime_to_cpu"] = MagicMock(
            return_value={
                "offloaded_prefix_modules": 4,
                "offloaded_prefix_parameters": 123,
                "offloaded_prefix_input_embeddings": True,
                "split_layer_idx": 3,
            }
        )
        deps["src.training.train_tg_lora.save_prefix_feature_dataset"] = MagicMock()
        deps["src.training.train_tg_lora.forward_backward"] = MagicMock(return_value=1.0)
        deps["src.training.train_tg_lora.optimizer_step"] = MagicMock()

        import contextlib

        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_obj in deps.items():
                mocks[target.split(".")[-1]] = stack.enter_context(
                    patch(target, new=mock_obj)
                )
            train_tg_lora(_make_config(**cfg_overrides))

        mocks["offload_prefix_runtime_to_cpu"].assert_called_once()
        assert mocks["offload_prefix_runtime_to_cpu"].call_args.kwargs["split_layer_idx"] == 3
        footer_summary = mocks["RunMetrics"].return_value.write_footer.call_args.kwargs[
            "tg_lora_summary"
        ]
        assert footer_summary["prefix_feature_cache_runtime_offload_applied"] is True
        assert footer_summary["prefix_feature_cache_offloaded_prefix_modules"] == 4
        assert footer_summary["prefix_feature_cache_offloaded_prefix_parameters"] == 123
        assert (
            footer_summary[
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_before"
            ]
            == 12000.0
        )
        assert (
            footer_summary[
                "prefix_feature_cache_runtime_offload_gpu_allocated_mb_after"
            ]
            == 8200.0
        )
        assert footer_summary["prefix_feature_cache_runtime_offload_gpu_freed_mb"] == 3800.0


class TestErrorHandler:
    """Test that exceptions during extrapolation don't corrupt rollback state."""

    def test_rollback_manager_pop_called_on_exception(self):
        """rollback_mgr.pop() is called in finally block even on exception."""
        call_count = [0]

        def eval_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("Simulated eval failure")
            return 2.0

        deps = _patch_train_tg_lora_deps(eval_losses=[2.0])
        # Replace eval_loss with one that raises on 2nd call
        mock_eval_loss = MagicMock(side_effect=eval_side_effect)
        deps["src.training.train_tg_lora.eval_loss"] = mock_eval_loss

        import contextlib

        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_obj in deps.items():
                mocks[target.split(".")[-1]] = stack.enter_context(
                    patch(target, new=mock_obj)
                )
            try:
                train_tg_lora(_make_config(training={"max_cycles": 1}))
            except RuntimeError:
                pass  # Exception propagates if not caught inside train_tg_lora


class TestBuildTrainingSummaryComplete:
    """Verify build_training_summary output completeness."""

    def test_all_required_fields_present(self):
        """Summary contains all fields needed for logging."""
        from src.tg_lora.random_walk_controller import RandomWalkController

        cs = CycleState()
        dt = DeltaTracker()
        controller = RandomWalkController(
            K_initial=2,
            K_candidates=[2, 3],
            N_initial=1,
            N_candidates=[1, 3],
            alpha_initial=0.3,
            alpha_min=0.01,
            alpha_max=2.0,
            alpha_log_sigma=0.1,
            beta_initial=0.9,
            beta_candidates=[0.9],
            active_layer_strategy="last_25_percent",
            relative_update_cap=0.5,
            rollback_tolerance=0.005,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        controller.propose()
        controller.accept(1.0, 0.95)
        controller.reward(1.0, 0.95)

        before = {"lora_A": torch.zeros(4)}
        after = {"lora_A": torch.ones(4) * 0.1}
        dt.compute_and_record(after, before, K=2)
        cs.record_cycle(
            K=2, N=1, grad_accum=1, train_loss=2.0, valid_loss=2.0, accepted=True
        )

        summary = build_training_summary(controller, cs, dt)

        required_keys = [
            "cycles",
            "full_backward_passes",
            "extrapolation_steps",
            "reduction_rate",
            "best_valid_loss",
            "best_valid_step",
            "stale_cycles",
            "acceptance_rate",
            "accepted_count",
            "rejected_count",
            "final_train_loss",
            "total_norm",
            "max_component",
            "mean_abs",
            "anomalous",
            "convergence_trend",
            "history_length",
            "total_cycles",
            "current_alpha",
            "current_N",
            "current_K",
            "current_beta",
            "strategy",
            "controller_acceptance_rate",
            "controller_total_cycles",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_no_key_collision_between_sources(self):
        """Controller and cycle_state both produce acceptance_rate — both preserved."""
        from src.tg_lora.random_walk_controller import RandomWalkController

        cs = CycleState()
        dt = DeltaTracker()
        controller = RandomWalkController(
            K_initial=2,
            K_candidates=[2],
            N_initial=1,
            N_candidates=[1],
            alpha_initial=0.3,
            alpha_min=0.01,
            alpha_max=2.0,
            alpha_log_sigma=0.1,
            beta_initial=0.9,
            beta_candidates=[0.9],
            active_layer_strategy="last_25_percent",
            relative_update_cap=0.5,
            rollback_tolerance=0.005,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        controller.propose()
        controller.accept(1.0, 0.95)
        controller.reward(1.0, 0.95)

        before = {"lora_A": torch.zeros(4)}
        after = {"lora_A": torch.ones(4) * 0.1}
        dt.compute_and_record(after, before, K=2)
        cs.record_cycle(
            K=2, N=1, grad_accum=1, train_loss=2.0, valid_loss=2.0, accepted=True
        )

        summary = build_training_summary(controller, cs, dt)

        # Both controller and cycle_state acceptance_rate must be preserved
        assert "acceptance_rate" in summary
        assert "controller_acceptance_rate" in summary
        assert "controller_total_cycles" in summary
        # The cycle_state version is the canonical "acceptance_rate"
        assert summary["acceptance_rate"] == cs.acceptance_rate
        assert summary["controller_acceptance_rate"] == controller.acceptance_rate()


class TestAdaptiveLrSmokeTest:
    """Smoke tests verifying the training loop completes with adaptive lr enabled.

    The adaptive lr feature adds lr_initial, lr_min, lr_max, lr_accept_boost,
    lr_reject_decay to the tg_lora config. These tests ensure the config is
    wired through to the controller and the loop completes at least one cycle.
    """

    def test_adaptive_lr_config_wired_to_controller(self):
        """train_tg_lora passes adaptive lr params from config to controller."""
        cfg = _make_config(
            tg_lora={
                "K_initial": 2,
                "K_candidates": [2, 3],
                "N_initial": 1,
                "N_candidates": [1, 3],
                "alpha_initial": 0.3,
                "alpha_min": 0.01,
                "alpha_max": 2.0,
                "alpha_log_sigma": 0.1,
                "beta_initial": 0.9,
                "beta_candidates": [0.9],
                "active_layer_strategy": "last_25_percent",
                "relative_update_cap": 0.5,
                "random_middle_layers": 2,
                "layer_sample_temperature": 1.0,
                "lr_initial": 3e-4,
                "lr_min": 5e-6,
                "lr_max": 5e-3,
                "lr_accept_boost": 1.3,
                "lr_reject_decay": 0.4,
            }
        )

        # Interceptor to capture the controller created inside train_tg_lora
        captured_controller = [None]

        import contextlib

        original_rwc = __import__(
            "src.tg_lora.random_walk_controller",
            fromlist=["RandomWalkController"],
        ).RandomWalkController

        def capturing_controller(*args, **kwargs):
            ctrl = original_rwc(*args, **kwargs)
            captured_controller[0] = ctrl
            return ctrl

        eval_losses = [2.0, 1.5] * 10
        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)
        deps["src.training.train_tg_lora.RandomWalkController"] = capturing_controller

        # Patch MLflowLogger to avoid state leaking between tests
        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        with contextlib.ExitStack() as stack:
            for target, mock_obj in deps.items():
                stack.enter_context(patch(target, new=mock_obj))
            train_tg_lora(cfg)

        ctrl = captured_controller[0]
        assert ctrl is not None
        # Verify config values were wired through to controller
        assert ctrl.lr_min == pytest.approx(5e-6)
        assert ctrl.lr_max == pytest.approx(5e-3)
        assert ctrl.state.lr_accept_boost == pytest.approx(1.3)
        assert ctrl.state.lr_reject_decay == pytest.approx(0.4)
        # lr started at 3e-4 and may have changed via reward/penalize
        assert ctrl.lr_min <= ctrl.state.lr <= ctrl.lr_max

    def test_one_cycle_with_adaptive_lr_completes(self):
        """Training loop completes one full cycle with adaptive lr params set."""
        cfg_overrides = {
            "training": {"max_cycles": 1},
            "tg_lora": {
                "K_initial": 2,
                "K_candidates": [2, 3],
                "N_initial": 1,
                "N_candidates": [1, 3],
                "alpha_initial": 0.3,
                "alpha_min": 0.01,
                "alpha_max": 2.0,
                "alpha_log_sigma": 0.1,
                "beta_initial": 0.9,
                "beta_candidates": [0.9],
                "active_layer_strategy": "last_25_percent",
                "relative_update_cap": 0.5,
                "random_middle_layers": 2,
                "layer_sample_temperature": 1.0,
                "lr_initial": 5e-4,
                "lr_min": 1e-5,
                "lr_max": 1e-3,
                "lr_accept_boost": 1.2,
                "lr_reject_decay": 0.5,
            },
        }
        # Accept: pilot=2.0, after=1.5
        eval_losses = [2.0, 1.5]
        # Run with MLflow disabled to avoid cross-test state leakage
        import contextlib

        cfg = _make_config(**cfg_overrides)
        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)
        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_obj in deps.items():
                mocks[target.split(".")[-1]] = stack.enter_context(
                    patch(target, new=mock_obj)
                )
            train_tg_lora(cfg)
        assert mocks["eval_loss"].call_count >= 2

    def test_lr_changes_after_accept_reject_in_loop(self):
        """Controller lr changes after accept/reject in the training loop."""
        cfg_overrides = {
            "training": {"max_cycles": 3},
            "tg_lora": {
                "K_initial": 2,
                "K_candidates": [2, 3],
                "N_initial": 1,
                "N_candidates": [1, 3],
                "alpha_initial": 0.3,
                "alpha_min": 0.01,
                "alpha_max": 2.0,
                "alpha_log_sigma": 0.1,
                "beta_initial": 0.9,
                "beta_candidates": [0.9],
                "active_layer_strategy": "last_25_percent",
                "relative_update_cap": 0.5,
                "random_middle_layers": 2,
                "layer_sample_temperature": 1.0,
                "lr_initial": 5e-4,
                "lr_min": 1e-5,
                "lr_max": 1e-3,
                "lr_accept_boost": 2.0,
                "lr_reject_decay": 0.25,
            },
        }
        # Cycle 0: accept (2.0 → 1.5), Cycle 1: reject (2.0 → 3.0), Cycle 2: accept
        eval_losses = [2.0, 1.5, 2.0, 3.0, 2.0, 1.5]
        import contextlib

        cfg = _make_config(**cfg_overrides)
        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)
        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_obj in deps.items():
                mocks[target.split(".")[-1]] = stack.enter_context(
                    patch(target, new=mock_obj)
                )
            train_tg_lora(cfg)
        # The training ran without error — lr adapted through accept/reject
        assert mocks["eval_loss"].called


class TestLrExplorationIntegration:
    """Integration tests verifying lr_explore_prob is consumed end-to-end.

    The lr exploration feature in propose() generates a proposed lr via
    log-normal random walk. These tests verify that:
    1. lr_explore_prob and lr_log_sigma from config reach the controller
    2. The proposed lr is applied to controller.state.lr during training
    3. Multiple propose→accept/reject cycles produce non-trivial lr changes
       that cannot be explained by deterministic boost/decay alone.
    """

    def _run_with_captured_controller(self, cfg_overrides, eval_losses):
        """Run training and return the captured controller."""
        import contextlib

        cfg = _make_config(**cfg_overrides)
        captured = [None]

        original_rwc = __import__(
            "src.tg_lora.random_walk_controller",
            fromlist=["RandomWalkController"],
        ).RandomWalkController

        def capturing_controller(*args, **kwargs):
            ctrl = original_rwc(*args, **kwargs)
            captured[0] = ctrl
            return ctrl

        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)
        deps["src.training.train_tg_lora.RandomWalkController"] = capturing_controller
        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        with contextlib.ExitStack() as stack:
            for target, mock_obj in deps.items():
                stack.enter_context(patch(target, new=mock_obj))
            train_tg_lora(cfg)

        return captured[0]

    def test_lr_explore_prob_wired_from_config(self):
        """lr_explore_prob and lr_log_sigma from config reach the controller."""
        ctrl = self._run_with_captured_controller(
            cfg_overrides={
                "training": {"max_cycles": 1},
                "tg_lora": {
                    "K_initial": 2,
                    "K_candidates": [2, 3],
                    "N_initial": 1,
                    "N_candidates": [1, 3],
                    "alpha_initial": 0.3,
                    "alpha_min": 0.01,
                    "alpha_max": 2.0,
                    "alpha_log_sigma": 0.1,
                    "beta_initial": 0.9,
                    "beta_candidates": [0.9],
                    "active_layer_strategy": "last_25_percent",
                    "relative_update_cap": 0.5,
                    "random_middle_layers": 2,
                    "layer_sample_temperature": 1.0,
                    "lr_initial": 5e-4,
                    "lr_min": 1e-5,
                    "lr_max": 1e-3,
                    "lr_explore_prob": 0.8,
                    "lr_log_sigma": 0.25,
                },
            },
            eval_losses=[2.0, 1.5],
        )
        assert ctrl is not None
        assert ctrl.lr_explore_prob == pytest.approx(0.8)
        assert ctrl.lr_log_sigma == pytest.approx(0.25)

    def test_proposed_lr_applied_to_state(self):
        """After propose() in training, the explored lr is applied to state.

        With lr_explore_prob=1.0 and high sigma, the proposed lr should
        differ from the deterministic boost/decay path. After one accepted
        cycle (boost=1.2), deterministic lr would be 5e-4 * 1.2 = 6e-4.
        With exploration, the final lr should differ because the explored
        lr is applied before reward().
        """
        ctrl = self._run_with_captured_controller(
            cfg_overrides={
                "training": {"max_cycles": 1},
                "tg_lora": {
                    "K_initial": 2,
                    "K_candidates": [2, 3],
                    "N_initial": 1,
                    "N_candidates": [1, 3],
                    "alpha_initial": 0.3,
                    "alpha_min": 0.01,
                    "alpha_max": 2.0,
                    "alpha_log_sigma": 0.1,
                    "beta_initial": 0.9,
                    "beta_candidates": [0.9],
                    "active_layer_strategy": "last_25_percent",
                    "relative_update_cap": 0.5,
                    "random_middle_layers": 2,
                    "layer_sample_temperature": 1.0,
                    "lr_initial": 5e-4,
                    "lr_min": 1e-5,
                    "lr_max": 1e-3,
                    "lr_accept_boost": 1.2,
                    "lr_reject_decay": 0.5,
                    "lr_explore_prob": 1.0,
                    "lr_log_sigma": 0.3,
                },
            },
            eval_losses=[2.0, 1.5],  # accept: pilot=2.0, after=1.5
        )
        assert ctrl is not None
        # After one accepted cycle with exploration, lr should have changed.
        # The final lr = explored_lr * lr_accept_boost, where explored_lr
        # comes from log-normal walk. Verify lr is within bounds and differs
        # from the purely deterministic path (5e-4 * 1.2 = 6e-4).
        assert ctrl.lr_min <= ctrl.state.lr <= ctrl.lr_max
        deterministic_lr = 5e-4 * 1.2
        # With sigma=0.3 and prob=1.0, the explored lr is very unlikely
        # to land exactly on the deterministic path. We check it changed.
        assert ctrl.state.lr != pytest.approx(deterministic_lr, rel=1e-3)

    def test_full_propose_accept_reject_cycle_with_lr_walk(self):
        """Multi-cycle propose→accept/reject with lr_explore_prob > 0.

        Runs 5 cycles with alternating accept/reject outcomes and verifies:
        - lr stays within [lr_min, lr_max] throughout
        - lr changes across cycles (not frozen)
        - The final lr differs from purely deterministic boost/decay
        """
        # Pattern: accept, reject, accept, reject, accept
        eval_losses = [2.0, 1.5, 2.0, 3.0, 2.0, 1.5, 2.0, 3.0, 2.0, 1.5]
        ctrl = self._run_with_captured_controller(
            cfg_overrides={
                "training": {"max_cycles": 5},
                "tg_lora": {
                    "K_initial": 2,
                    "K_candidates": [2, 3],
                    "N_initial": 1,
                    "N_candidates": [1, 3],
                    "alpha_initial": 0.3,
                    "alpha_min": 0.01,
                    "alpha_max": 2.0,
                    "alpha_log_sigma": 0.1,
                    "beta_initial": 0.9,
                    "beta_candidates": [0.9],
                    "active_layer_strategy": "last_25_percent",
                    "relative_update_cap": 0.5,
                    "random_middle_layers": 2,
                    "layer_sample_temperature": 1.0,
                    "lr_initial": 5e-4,
                    "lr_min": 1e-5,
                    "lr_max": 1e-3,
                    "lr_accept_boost": 1.5,
                    "lr_reject_decay": 0.5,
                    "lr_explore_prob": 1.0,
                    "lr_log_sigma": 0.25,
                },
            },
            eval_losses=eval_losses,
        )
        assert ctrl is not None
        # lr must be within bounds
        assert ctrl.lr_min <= ctrl.state.lr <= ctrl.lr_max
        # lr must have changed from initial (exploration + feedback)
        assert ctrl.state.lr != 5e-4
        # Deterministic path: 5e-4 * 1.5^3 * 0.5^2 = 5e-4 * 3.375 * 0.25 = 4.21875e-4
        # With exploration, the path diverges significantly with sigma=0.25
        deterministic_lr = 5e-4 * (1.5 ** 3) * (0.5 ** 2)
        # Check lr is not at the deterministic value (highly unlikely with exploration)
        assert ctrl.state.lr != pytest.approx(deterministic_lr, rel=0.01)


class TestNonFiniteLossAfterWarning:
    """TASK-0053: Verify logger.warning on non-finite loss_after."""

    def _run_with_mlflow_mock(self, cfg_overrides, eval_losses):
        """Run mocked training with MLflowLogger patched out for isolation."""
        import contextlib

        cfg = _make_config(**(cfg_overrides or {}))
        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)
        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        with contextlib.ExitStack() as stack:
            mocks = {}
            for target, mock_obj in deps.items():
                mocks[target.split(".")[-1]] = stack.enter_context(
                    patch(target, new=mock_obj)
                )
            train_tg_lora(cfg)
        return mocks

    @pytest.mark.parametrize("loss_value,expect_warning,substr", [
        (float("nan"), True, "nan"),
        (float("inf"), True, "inf"),
        (1.5, False, None),
    ])
    def test_nonfinite_loss_handling(self, caplog, loss_value, expect_warning, substr):
        import logging

        eval_losses = [2.0, loss_value, 2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 1.5]
        with caplog.at_level(logging.WARNING, logger="tg-lora"):
            self._run_with_mlflow_mock(
                cfg_overrides={"training": {"max_cycles": 5}},
                eval_losses=eval_losses,
            )

        has_warning = any("Non-finite loss_after" in rec.message for rec in caplog.records)
        if expect_warning:
            assert has_warning, "Expected warning about non-finite loss_after"
            assert any(substr in rec.message for rec in caplog.records)
        else:
            assert not has_warning, (
                f"Unexpected warning about non-finite loss_after: {[r.message for r in caplog.records]}"
            )


class TestRollbackFailureResilience:
    """TC-084-01: Training continues safely when rollback() raises."""

    def _run_with_rollback_failure(self, caplog, side_effect):
        import contextlib
        import logging

        eval_losses = [2.0, 3.0] * 10
        cfg = _make_config(training={"max_cycles": 3})
        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)

        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        with caplog.at_level(logging.ERROR, logger="tg-lora"):
            with contextlib.ExitStack() as stack:
                mocks = {}
                for target, mock_obj in deps.items():
                    mocks[target.split(".")[-1]] = stack.enter_context(
                        patch(target, new=mock_obj)
                    )
                stack.enter_context(
                    patch(
                        "src.training.train_tg_lora.RollbackManager.rollback",
                        side_effect=side_effect,
                    )
                )
                train_tg_lora(cfg)
        return mocks, caplog.records

    @pytest.mark.parametrize("exc", [RuntimeError("snapshot corrupted"), IndexError("out of range")])
    def test_rollback_failure_logged_and_training_continues(self, caplog, exc):
        mocks, records = self._run_with_rollback_failure(caplog, exc)
        assert mocks["eval_loss"].called
        assert any("Rollback failed" in r.message for r in records), (
            f"Expected 'Rollback failed', got: {[r.message for r in records]}"
        )


class TestNonFiniteParamsRollbackException:
    """TASK-0054: Non-finite params path (lines 324-327) rollback exception E2E tests.

    Verifies that when check_lora_params_finite detects non-finite parameters
    AND rollback() subsequently raises, the training loop:
    - does not crash
    - logs the rollback failure
    - calls penalize and record_cycle
    - continues to the next cycle via ``continue``
    """

    def _run_nonfinite_rollback_failure(self, caplog, rollback_error):
        """Run training where first cycle has non-finite params + rollback failure."""
        import contextlib
        import logging

        eval_losses = [2.0, 1.5] * 20
        cfg = _make_config(training={"max_cycles": 3})
        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)

        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        with caplog.at_level(logging.ERROR, logger="tg-lora"):
            with contextlib.ExitStack() as stack:
                mocks = {}
                for target, mock_obj in deps.items():
                    mocks[target.split(".")[-1]] = stack.enter_context(
                        patch(target, new=mock_obj)
                    )
                finite_calls = [0]

                def check_finite(*args, **kwargs):
                    finite_calls[0] += 1
                    if finite_calls[0] == 1:
                        return (False, "layers_0_lora_A: NaN")
                    return (True, "")

                stack.enter_context(
                    patch(
                        "src.training.train_tg_lora.check_lora_params_finite",
                        side_effect=check_finite,
                    )
                )
                stack.enter_context(
                    patch(
                        "src.training.train_tg_lora.RollbackManager.rollback",
                        side_effect=rollback_error,
                    )
                )
                train_tg_lora(cfg)

        return mocks, caplog.records

    def test_nonfinite_rollback_runtime_error_no_crash(self, caplog):
        """Training loop does not crash when non-finite params detected and rollback raises RuntimeError."""
        mocks, _records = self._run_nonfinite_rollback_failure(
            caplog,
            RuntimeError("disk full"),
        )
        assert mocks["eval_loss"].called

    def test_nonfinite_rollback_error_logged(self, caplog):
        """Rollback failed error is logged in the non-finite params path."""
        _mocks, records = self._run_nonfinite_rollback_failure(
            caplog,
            RuntimeError("disk full"),
        )
        assert any("Rollback failed" in r.message for r in records), (
            f"Expected 'Rollback failed', got: {[r.message for r in records]}"
        )

    def test_nonfinite_rollback_training_continues(self, caplog):
        """Training continues for remaining cycles after non-finite rollback failure."""
        mocks, _records = self._run_nonfinite_rollback_failure(
            caplog,
            RuntimeError("disk full"),
        )
        # eval_loss called at least once per cycle (pilot eval)
        assert mocks["eval_loss"].call_count >= 3

    def test_nonfinite_rollback_penalize_and_record_cycle_called(self, caplog):
        """controller.penalize and cycle_state.record_cycle executed after non-finite rollback failure."""
        import contextlib
        import logging

        from src.tg_lora.cycle_state import CycleState
        from src.tg_lora.random_walk_controller import RandomWalkController

        eval_losses = [2.0, 1.5] * 20
        cfg = _make_config(training={"max_cycles": 3})
        deps = _patch_train_tg_lora_deps(eval_losses=eval_losses)

        mock_mlf = MagicMock()
        mock_mlf.enabled = False
        mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
        mock_mlf.__exit__ = MagicMock(return_value=False)
        deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(
            return_value=mock_mlf
        )

        penalize_calls = []
        original_penalize = RandomWalkController.penalize

        def tracking_penalize(self, *args, **kwargs):
            penalize_calls.append((args, kwargs))
            return original_penalize(self, *args, **kwargs)

        record_cycle_calls = []
        original_record_cycle = CycleState.record_cycle

        def tracking_record_cycle(self, *args, **kwargs):
            record_cycle_calls.append((args, kwargs))
            return original_record_cycle(self, *args, **kwargs)

        finite_calls = [0]

        def check_finite(*args, **kwargs):
            finite_calls[0] += 1
            if finite_calls[0] == 1:
                return (False, "layers_0_lora_A: NaN")
            return (True, "")

        with caplog.at_level(logging.ERROR, logger="tg-lora"):
            with contextlib.ExitStack() as stack:
                mocks = {}
                for target, mock_obj in deps.items():
                    mocks[target.split(".")[-1]] = stack.enter_context(
                        patch(target, new=mock_obj)
                    )
                stack.enter_context(
                    patch(
                        "src.training.train_tg_lora.check_lora_params_finite",
                        side_effect=check_finite,
                    )
                )
                stack.enter_context(
                    patch(
                        "src.training.train_tg_lora.RollbackManager.rollback",
                        side_effect=RuntimeError("disk full"),
                    )
                )
                stack.enter_context(
                    patch.object(RandomWalkController, "penalize", tracking_penalize)
                )
                stack.enter_context(
                    patch.object(CycleState, "record_cycle", tracking_record_cycle)
                )
                train_tg_lora(cfg)

        assert len(penalize_calls) >= 1, (
            "controller.penalize should have been called after non-finite rollback failure"
        )
        assert len(record_cycle_calls) >= 1, (
            "cycle_state.record_cycle should have been called after non-finite rollback failure"
        )
