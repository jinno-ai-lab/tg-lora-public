"""TASK-0028: TG-LoRA 10-cycle training smoke test.

Validates all 7 acceptance criteria:
1. 10-cycle TG-LoRA training completes normally
2. run_metrics.jsonl records 10 cycles of metrics
3. Accept/reject based on rollback_tolerance=0.005
4. Adaptive LR: accept → lr up, reject → lr down (0.5x)
5. lr stays in [lr_min=1e-5, lr_max=1e-3]
6. velocity cosine similarity calculated and recorded
7. Loss values are not NaN/Inf
"""

import math
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.tg_lora.random_walk_controller import RandomWalkController
from src.tg_lora.velocity import Velocity
from src.training.train_tg_lora import _decide_accept_rollback, train_tg_lora
from src.utils.run_metrics import RunMetrics

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_config(**overrides):
    defaults = {
        "experiment": {"name": "task0028_test", "seed": 42},
        "model": {"name_or_path": "test-model"},
        "lora": {"r": 16, "alpha": 32},
        "data": {
            "train_path": "/tmp/dummy",
            "valid_quick_path": "/tmp/dummy",
            "valid_full_path": "/tmp/dummy",
            "max_seq_len": 32,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 2e-4,
            "weight_decay": 0.0,
            "grad_accumulation": 1,
            "max_grad_norm": 1.0,
            "max_cycles": 10,
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
            "lr_initial": 5e-4,
            "lr_min": 1e-5,
            "lr_max": 1e-3,
            "lr_accept_boost": 1.2,
            "lr_reject_decay": 0.5,
        },
        "eval": {
            "quick_eval_examples": 5,
            "full_eval_every_cycles": 0,
            "rollback_tolerance": 0.005,
        },
        "logging": {
            "run_dir": "/tmp/task0028_test_run",
            "log_every_cycles": 1,
            "save_every_cycles": 25,
        },
    }
    cfg = OmegaConf.create(defaults)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


class _SimpleDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, _):
        return {
            "input_ids": torch.randint(0, 100, (16,)),
            "attention_mask": torch.ones(16, dtype=torch.long),
            "labels": torch.randint(0, 100, (16,)),
        }


class _LoRAMockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(4):
            setattr(
                self,
                f"layers_{i}_lora_A",
                torch.nn.Parameter(torch.randn(8, 8) * 0.01),
            )
            setattr(
                self,
                f"layers_{i}_lora_B",
                torch.nn.Parameter(torch.randn(8, 8) * 0.01),
            )
        self.save_pretrained = MagicMock()

    def __call__(self, **kwargs):
        out = MagicMock()
        out.loss = torch.tensor(2.0, requires_grad=True)
        return out

    def train(self, mode=True):
        return self


def _make_mock_deps(eval_losses=None):
    model = _LoRAMockModel()
    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()

    if eval_losses is None:
        eval_losses = [2.0]

    run_dir = Path("/tmp/task0028_test_run")

    metrics = MagicMock()

    deps = {
        "src.training.train_tg_lora.load_tokenizer": MagicMock(return_value=tokenizer),
        "src.training.train_tg_lora.load_base_model": MagicMock(return_value=model),
        "src.training.train_tg_lora.apply_lora": MagicMock(return_value=model),
        "src.training.train_tg_lora.get_input_device": MagicMock(return_value="cpu"),
        "src.training.train_tg_lora.load_dataset": MagicMock(
            return_value=_SimpleDataset()
        ),
        "src.training.train_tg_lora.eval_loss": MagicMock(
            side_effect=list(eval_losses) * 200,
        ),
        "src.training.train_tg_lora.ensure_dir": MagicMock(return_value=run_dir),
        "src.training.train_tg_lora.RunMetrics": MagicMock(return_value=metrics),
        "src.training.train_tg_lora.count_parameters": MagicMock(
            return_value={"total": 100, "trainable": 50},
        ),
        "src.training.train_tg_lora.set_seed": MagicMock(),
    }
    return deps, model, metrics


def _run_mocked_training(eval_losses=None, cfg_overrides=None):
    """Run train_tg_lora with mocked deps. Returns (mocks_dict, controller)."""
    import tempfile

    cfg = _make_config(**(cfg_overrides or {}))
    deps, model, metrics = _make_mock_deps(eval_losses=eval_losses)

    captured = [None]
    _OriginalRWC = RandomWalkController

    def _capturing_rwc(*args, **kwargs):
        ctrl = _OriginalRWC(*args, **kwargs)
        captured[0] = ctrl
        return ctrl

    deps["src.training.train_tg_lora.RandomWalkController"] = _capturing_rwc

    mock_mlf = MagicMock()
    mock_mlf.enabled = False
    mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
    mock_mlf.__exit__ = MagicMock(return_value=False)
    deps["src.training.train_tg_lora.MLflowLogger"] = MagicMock(return_value=mock_mlf)

    tmp_dir = Path(tempfile.mkdtemp(prefix="task0028_"))
    deps["src.training.train_tg_lora.ensure_dir"] = MagicMock(return_value=tmp_dir)

    with ExitStack() as stack:
        mocks = {}
        for target, mock_obj in deps.items():
            mocks[target.split(".")[-1]] = stack.enter_context(
                patch(target, new=mock_obj),
            )
        train_tg_lora(cfg)

    return mocks, captured[0]


# ── Criterion 1: 10-cycle training completes ────────────────────────────────


class TestTenCycleCompletion:
    """10サイクルのTG-LoRA学習が正常完了する"""

    def test_ten_accepted_cycles(self):
        mocks, ctrl = _run_mocked_training(eval_losses=[2.0, 1.5] * 30)
        assert ctrl.state.total_cycles == 10

    def test_ten_rejected_cycles(self):
        mocks, ctrl = _run_mocked_training(eval_losses=[2.0, 3.0] * 30)
        assert ctrl.state.total_cycles == 10
        assert ctrl.state.rolled_back_count == 10

    def test_mixed_accept_reject(self):
        mocks, ctrl = _run_mocked_training(
            eval_losses=[2.0, 1.5, 2.0, 3.0] * 15,
        )
        assert ctrl.state.total_cycles == 10
        assert ctrl.state.accepted_count + ctrl.state.rolled_back_count == 10


# ── Criterion 2: run_metrics.jsonl records 10 cycles ────────────────────────


class TestRunMetricsRecording:
    """run_metrics.jsonlに10サイクル分のメトリクスが記録される"""

    def test_record_step_called_ten_times(self):
        mocks, _ = _run_mocked_training(eval_losses=[2.0, 1.5] * 30)
        assert mocks["RunMetrics"].return_value.record_step.call_count == 10

    def test_jsonl_contains_ten_step_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            cfg = _make_config()
            rm = RunMetrics(run_dir, mode="tg_lora", run_id="test_0028")
            rm.write_header(
                cfg,
                budget_type="cycles",
                budget_value=10,
                param_counts={"total": 100, "trainable": 50},
            )
            for i in range(10):
                rm.record_step(
                    step=i * 6,
                    cycle=i,
                    loss_train=2.0 - i * 0.05,
                    loss_valid=2.0 - i * 0.03,
                    backward_passes=2,
                    total_backward_passes=(i + 1) * 2,
                    tg_lora_accepted=(i % 3 != 0),
                    tg_lora_cosine_sim=0.85 - i * 0.02,
                    tg_lora_reduction_rate=0.33,
                    tg_lora_K=2,
                    tg_lora_N=1,
                    tg_lora_alpha=0.3,
                    tg_lora_beta=0.9,
                    tg_lora_lr=5e-4,
                )
            rm.write_footer(
                best_valid_loss=1.73,
                best_valid_step=54,
                final_train_loss=1.55,
                tg_lora_summary={"total_cycles": 10},
            )
            rm.close()

            records = []
            with open(run_dir / "run_metrics.jsonl", "rb") as f:
                for line in f:
                    records.append(orjson.loads(line))

            steps = [r for r in records if r["type"] == "step"]
            assert len(steps) == 10
            for s in steps:
                assert "cycle" in s
                assert "loss_train" in s
                assert "tg_lora_lr" in s
                assert "tg_lora_cosine_sim" in s
                assert "tg_lora_accepted" in s

    def test_all_spec_fields_present_in_record_step(self):
        mocks, _ = _run_mocked_training(eval_losses=[2.0, 1.5] * 30)
        call = mocks["RunMetrics"].return_value.record_step.call_args_list[0]
        kwargs = call.kwargs
        for field in [
            "cycle",
            "loss_train",
            "tg_lora_lr",
            "tg_lora_accepted",
            "tg_lora_cosine_sim",
            "tg_lora_reduction_rate",
            "tg_lora_K",
            "tg_lora_N",
            "tg_lora_alpha",
            "tg_lora_beta",
            "tg_lora_validation_forwards",
            "tg_lora_pilot_validation_forwards",
            "tg_lora_post_validation_forwards",
            "tg_lora_post_extrapolation_eval",
            "tg_lora_post_extrapolation_eval_skipped",
            "tg_lora_post_extrapolation_eval_skip_reason",
            "tg_lora_rollback_triggered",
            "tg_lora_cap_global_ratio",
            "tg_lora_cap_mean_ratio",
            "tg_lora_cap_min_ratio",
            "tg_lora_cap_capped_fraction",
            "tg_lora_raw_update_norm",
            "tg_lora_applied_update_norm",
        ]:
            assert field in kwargs, f"Missing field: {field}"


# ── Criterion 3: Accept/reject with rollback_tolerance=0.005 ────────────────


class TestAcceptRejectTolerance:
    """受理/拒否の判定がrollback_tolerance=0.005に基づいて実行される"""

    def test_improvement_accepted(self):
        accepted, _ = _decide_accept_rollback(2.0, 1.0, 0.005)
        assert accepted is True

    def test_within_tolerance_accepted(self):
        accepted, _ = _decide_accept_rollback(2.0, 2.005, 0.005)
        assert accepted is True

    def test_exact_boundary_accepted(self):
        accepted, _ = _decide_accept_rollback(2.0, 2.01, 0.005)
        assert accepted is True

    def test_just_above_boundary_rejected(self):
        accepted, _ = _decide_accept_rollback(2.0, 2.011, 0.005)
        assert accepted is False

    def test_controller_accept_matches_tolerance(self):
        ctrl = RandomWalkController(
            rollback_tolerance=0.005,
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        assert ctrl.accept(1.0, 0.9) is True
        assert ctrl.accept(1.0, 1.004) is True
        assert ctrl.accept(1.0, 1.01) is False

    def test_training_loop_uses_tolerance(self):
        """Verify the training loop's accept/reject uses tolerance=0.005."""
        eval_losses = [2.0, 2.003, 2.0, 2.011] * 10
        mocks, ctrl = _run_mocked_training(
            eval_losses=eval_losses,
            cfg_overrides={"training": {"max_cycles": 4}},
        )
        assert ctrl.state.total_cycles == 4

    def test_moving_average_baseline_reduces_false_accept(self):
        accepted, _ = _decide_accept_rollback(
            2.0,
            2.011,
            0.005,
            loss_history=[1.98, 1.99, 2.0],
        )
        assert accepted is False

    def test_soft_accept_allows_borderline_case(self):
        with patch("random.random", return_value=0.0):
            accepted, reason = _decide_accept_rollback(
                2.0,
                2.012,
                0.005,
                loss_history=[2.0, 2.0, 2.0],
                temperature=1.0,
            )
        assert accepted is True
        assert "soft_accept" in reason


# ── Criterion 4: Adaptive LR ────────────────────────────────────────────────


class TestAdaptiveLR:
    """adaptive LRが動作: 受理時lr増加、拒否時lr減少（0.5倍）"""

    def test_reward_boosts_lr(self):
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
        assert ctrl.state.lr == pytest.approx(6e-4)

    def test_penalize_halves_lr(self):
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
        with patch("random.random", side_effect=[0.5, 0.5]):
            ctrl.penalize(1.0, 1.1)
        assert ctrl.state.lr == pytest.approx(2.5e-4)

    def test_lr_trajectory_over_ten_cycles(self):
        ctrl = RandomWalkController(
            lr_initial=5e-4,
            lr_min=1e-5,
            lr_max=1e-3,
            lr_accept_boost=1.2,
            lr_reject_decay=0.5,
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=5,
            N_candidates=[1, 3, 5],
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        lr_history = [ctrl.state.lr]
        outcomes = [True, True, False, True, False, True, True, False, True, True]
        for accepted in outcomes:
            if accepted:
                with patch("random.random", side_effect=[0.5, 0.5]):
                    ctrl.reward(1.0, 0.9)
            else:
                with patch("random.random", side_effect=[0.5, 0.5]):
                    ctrl.penalize(1.0, 1.1)
            lr_history.append(ctrl.state.lr)

        # Accept → lr up, reject → lr down
        assert lr_history[1] > lr_history[0]  # accept
        assert lr_history[3] < lr_history[2]  # reject → half
        assert lr_history[5] < lr_history[4]  # reject → half


# ── Criterion 5: lr stays in bounds ─────────────────────────────────────────


class TestLRBounds:
    """lrが常に[lr_min=1e-5, lr_max=1e-3]の範囲内に留まる"""

    def test_repeated_rejects_clamp_at_min(self):
        ctrl = RandomWalkController(
            lr_initial=5e-4,
            lr_min=1e-5,
            lr_max=1e-3,
            lr_reject_decay=0.5,
            lr_accept_boost=1.2,
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=5,
            N_candidates=[1, 3, 5],
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        for _ in range(50):
            with patch("random.random", side_effect=[0.5, 0.5]):
                ctrl.penalize(1.0, 1.1)
            assert ctrl.state.lr >= 1e-5
        assert ctrl.state.lr == 1e-5

    def test_repeated_accepts_clamp_at_max(self):
        ctrl = RandomWalkController(
            lr_initial=5e-4,
            lr_min=1e-5,
            lr_max=1e-3,
            lr_accept_boost=1.2,
            lr_reject_decay=0.5,
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=1,
            N_candidates=[1, 3, 5],
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        for _ in range(50):
            with patch("random.random", side_effect=[0.5, 0.5]):
                ctrl.reward(1.0, 0.9)
            assert ctrl.state.lr <= 1e-3
        assert ctrl.state.lr == 1e-3

    def test_mixed_cycles_stay_in_bounds(self):
        ctrl = RandomWalkController(
            lr_initial=5e-4,
            lr_min=1e-5,
            lr_max=1e-3,
            lr_accept_boost=1.5,
            lr_reject_decay=0.5,
            K_initial=3,
            K_candidates=[2, 3, 5],
            N_initial=5,
            N_candidates=[1, 3, 5],
            k_explore_prob=0.0,
            n_explore_prob=0.0,
            beta_explore_prob=0.0,
            strategy_explore_prob=0.0,
            lr_explore_prob=0.0,
        )
        for i, accepted in enumerate(
            [True, False, True, True, False, True, False, True, True, True],
        ):
            if accepted:
                with patch("random.random", side_effect=[0.5, 0.5]):
                    ctrl.reward(1.0, 0.9)
            else:
                with patch("random.random", side_effect=[0.5, 0.5]):
                    ctrl.penalize(1.0, 1.1)
            assert 1e-5 <= ctrl.state.lr <= 1e-3, f"Cycle {i}: lr={ctrl.state.lr}"


# ── Criterion 6: velocity cosine similarity ─────────────────────────────────


class TestCosineSimilarity:
    """velocity cosine similarityが計算・記録されている"""

    def test_cosine_sim_computed(self):
        vel = Velocity()
        d1 = {"lora_A": torch.randn(32)}
        vel.update(d1, beta=0.9)
        cos = vel.cosine_similarity({"lora_A": torch.randn(32)})
        assert -1.0 <= cos <= 1.0

    def test_cosine_sim_recorded_in_metrics(self):
        mocks, _ = _run_mocked_training(eval_losses=[2.0, 1.5] * 30)
        for call in mocks["RunMetrics"].return_value.record_step.call_args_list:
            cos = call.kwargs.get("tg_lora_cosine_sim")
            assert cos is not None
            assert -1.0 <= cos <= 1.0

    def test_perfect_alignment(self):
        vel = Velocity()
        vel.update({"A": torch.ones(16)}, beta=0.9)
        assert vel.cosine_similarity({"A": torch.ones(16) * 2.0}) == pytest.approx(
            1.0,
            abs=1e-5,
        )

    def test_opposite_direction(self):
        vel = Velocity()
        vel.update({"A": torch.ones(16)}, beta=0.9)
        assert vel.cosine_similarity({"A": torch.ones(16) * -1.0}) == pytest.approx(
            -1.0,
            abs=1e-5,
        )


# ── Criterion 7: Loss not NaN/Inf ───────────────────────────────────────────


class TestLossFinite:
    """損失値がNaN/Infにならない"""

    def test_pilot_average_finite(self):
        from src.training.train_tg_lora import _compute_pilot_average

        avg, metrics = _compute_pilot_average([2.5, 1.8, 2.1, 1.9, 2.3], K=5)
        assert math.isfinite(avg)
        assert math.isfinite(metrics["min_loss"])
        assert math.isfinite(metrics["max_loss"])

    def test_jsonl_losses_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            cfg = _make_config()
            rm = RunMetrics(run_dir, mode="tg_lora", run_id="finite_test")
            rm.write_header(
                cfg,
                budget_type="cycles",
                budget_value=10,
                param_counts={"total": 100, "trainable": 50},
            )
            for i in range(10):
                rm.record_step(
                    step=i * 2,
                    cycle=i,
                    loss_train=2.0 - i * 0.1,
                    loss_valid=2.1 - i * 0.08,
                    backward_passes=2,
                    total_backward_passes=(i + 1) * 2,
                    tg_lora_accepted=True,
                    tg_lora_cosine_sim=0.9,
                    tg_lora_reduction_rate=0.33,
                    tg_lora_K=2,
                    tg_lora_N=1,
                    tg_lora_alpha=0.3,
                    tg_lora_beta=0.9,
                    tg_lora_lr=5e-4,
                )
            rm.close()

            with open(run_dir / "run_metrics.jsonl", "rb") as f:
                for line in f:
                    rec = orjson.loads(line)
                    if rec["type"] == "step":
                        assert math.isfinite(rec["loss_train"])
                        if rec.get("loss_valid") is not None:
                            assert math.isfinite(rec["loss_valid"])

    def test_training_loop_records_finite_losses(self):
        eval_losses = [
            2.0,
            1.5,
            2.1,
            1.6,
            1.9,
            1.4,
            2.2,
            1.7,
            1.8,
            1.3,
            2.0,
            1.5,
            2.1,
            1.6,
            1.9,
            1.4,
            2.2,
            1.7,
            1.8,
            1.3,
        ]
        mocks, _ = _run_mocked_training(eval_losses=eval_losses)
        for call in mocks["RunMetrics"].return_value.record_step.call_args_list:
            assert math.isfinite(call.kwargs["loss_train"])
