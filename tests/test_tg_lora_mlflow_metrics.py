"""Tests for TASK-0058: TG-LoRA specialized metrics MLflow integration.

Verifies that velocity magnitude, delta tracker stats, acceptance/reduction
rates, and layer scores are all forwarded to MLflow during training cycles.
Uses the same mock-based training loop pattern as test_training_integration.py.
"""

from __future__ import annotations

import contextlib
import types
from unittest.mock import MagicMock, patch
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.training.train_tg_lora import train_tg_lora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_mlflow() -> types.ModuleType:
    """Create a fake ``mlflow`` module with the subset we use."""
    fake = types.ModuleType("mlflow")
    fake.start_run = MagicMock(return_value=MagicMock())
    fake.end_run = MagicMock()
    fake.log_params = MagicMock()
    fake.log_metrics = MagicMock()
    fake.log_artifact = MagicMock()
    fake.set_tracking_uri = MagicMock()
    fake.set_experiment = MagicMock()
    fake.set_tag = MagicMock()
    fake_pyfunc = types.ModuleType("mlflow.pyfunc")
    fake_pyfunc.log_model = MagicMock()
    fake.pyfunc = fake_pyfunc
    return fake


class _LoRALayer(torch.nn.Module):
    """Holds lora_A and lora_B as proper submodules for named_parameters."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01)


class _LoRAMockModel(torch.nn.Module):
    def __init__(self, num_layers: int = 4, hidden: int = 8):
        super().__init__()
        # Build layers so named_parameters contain "layers.{i}.lora_A" / "layers.{i}.lora_B"
        # which matches iter_lora_params and iter_lora_params_by_layer patterns.
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(_LoRALayer(hidden))
        self._loss_val = 2.0
        self.save_pretrained = MagicMock()

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


def _make_config(**overrides):
    defaults = {
        "experiment": {"name": "test_run", "seed": 42},
        "model": {
            "name_or_path": "dummy",
            "load_in_4bit": False,
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
            "max_cycles": 3,
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


def _run_training_with_mlflow_capture(eval_losses=None, cfg_overrides=None):
    """Run mocked training and return all captured MLflow log_metrics calls.

    Returns a list of (metrics_dict, step) tuples from every mlf.log_metrics call.
    """
    from torch.utils.data import Dataset

    class _SimpleDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 100, (16,)),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": torch.randint(0, 100, (16,)),
            }

    if eval_losses is None:
        eval_losses = [2.0, 1.5] * 20

    model = _LoRAMockModel()
    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()
    dataset = _SimpleDataset()
    import tempfile
    run_dir = Path(tempfile.mkdtemp(prefix="mlflow_test_"))
    metrics_mock = MagicMock()

    from src.eval.eval_loss import EvalLossResult

    mock_eval_loss = MagicMock(side_effect=list(eval_losses) * 100)
    mock_eval_loss_detailed = MagicMock(
        side_effect=[
            EvalLossResult(avg_loss=v, num_batches=1, min_loss=v, max_loss=v)
            for v in (list(eval_losses) * 100)
        ]
    )

    deps = {
        "src.training.train_tg_lora.load_tokenizer": MagicMock(return_value=tokenizer),
        "src.training.train_tg_lora.load_base_model": MagicMock(return_value=model),
        "src.training.train_tg_lora.apply_lora": MagicMock(return_value=model),
        "src.training.train_tg_lora.get_input_device": MagicMock(return_value="cpu"),
        "src.training.train_tg_lora.load_dataset": MagicMock(return_value=dataset),
        "src.training.train_tg_lora.eval_loss": mock_eval_loss,
        "src.training.train_tg_lora.eval_loss_detailed": mock_eval_loss_detailed,
        "src.training.train_tg_lora.ensure_dir": MagicMock(return_value=run_dir),
        "src.training.train_tg_lora.RunMetrics": MagicMock(return_value=metrics_mock),
        "src.training.train_tg_lora.count_parameters": MagicMock(
            return_value={"total": 100, "trainable": 50}
        ),
        "src.training.train_tg_lora.set_seed": MagicMock(),
    }

    # Use a real MLflowLogger backed by fake mlflow so we capture actual calls
    fake_mlflow = _make_fake_mlflow()

    overrides = dict(cfg_overrides or {})
    cfg = _make_config(**overrides)
    cfg.logging.run_dir = str(run_dir)

    with contextlib.ExitStack() as stack:
        for target, mock_obj in deps.items():
            stack.enter_context(patch(target, new=mock_obj))
        stack.enter_context(patch("src.utils.mlflow_logger._mlflow", fake_mlflow))
        train_tg_lora(cfg)

    # Collect all log_metrics calls: (metrics_dict, kwargs)
    calls = []
    for call in fake_mlflow.log_metrics.call_args_list:
        args, kwargs = call
        calls.append((args[0] if args else {}, kwargs))
    return calls


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestVelocityMagnitudeMlflow:
    """Acceptance criterion: velocity magnitude trend logged to MLflow."""

    def test_velocity_magnitude_trend_in_mlflow_metrics(self):
        calls = _run_training_with_mlflow_capture(
            eval_losses=[2.0, 1.5] * 20,
        )
        # Collect all metric keys across all log_metrics calls
        all_keys = set()
        for metrics_dict, _kwargs in calls:
            all_keys.update(metrics_dict.keys())
        assert "velocity_magnitude" in all_keys, (
            f"Expected 'velocity_magnitude' in MLflow metrics, got: {sorted(all_keys)}"
        )

    def test_velocity_magnitude_is_float(self):
        calls = _run_training_with_mlflow_capture(
            eval_losses=[2.0, 1.5] * 20,
        )
        for metrics_dict, _kwargs in calls:
            if "velocity_magnitude" in metrics_dict:
                assert isinstance(metrics_dict["velocity_magnitude"], float)


class TestDeltaTrackerStatsMlflow:
    """Acceptance criterion: DeltaTracker stats (total_norm, convergence_trend)."""

    def test_delta_total_norm_in_mlflow(self):
        calls = _run_training_with_mlflow_capture(
            eval_losses=[2.0, 1.5] * 20,
        )
        all_keys = set()
        for metrics_dict, _kwargs in calls:
            all_keys.update(metrics_dict.keys())
        assert "delta_total_norm" in all_keys, (
            f"Expected 'delta_total_norm' in MLflow metrics, got: {sorted(all_keys)}"
        )

    def test_convergence_trend_in_mlflow(self):
        calls = _run_training_with_mlflow_capture(
            eval_losses=[2.0, 1.5] * 20,
        )
        all_keys = set()
        for metrics_dict, _kwargs in calls:
            all_keys.update(metrics_dict.keys())
        assert "convergence_trend" in all_keys, (
            f"Expected 'convergence_trend' in MLflow metrics, got: {sorted(all_keys)}"
        )


class TestAcceptanceReductionRateMlflow:
    """Acceptance criterion: acceptance_rate, reduction_rate logged to MLflow."""

    def test_acceptance_rate_in_mlflow(self):
        calls = _run_training_with_mlflow_capture(
            eval_losses=[2.0, 1.5] * 20,
        )
        all_keys = set()
        for metrics_dict, _kwargs in calls:
            all_keys.update(metrics_dict.keys())
        assert "acceptance_rate" in all_keys, (
            f"Expected 'acceptance_rate' in MLflow metrics, got: {sorted(all_keys)}"
        )

    def test_reduction_rate_already_logged(self):
        """reduction_rate is already logged — verify it remains present."""
        calls = _run_training_with_mlflow_capture(
            eval_losses=[2.0, 1.5] * 20,
        )
        all_keys = set()
        for metrics_dict, _kwargs in calls:
            all_keys.update(metrics_dict.keys())
        assert "reduction_rate" in all_keys


class TestLayerScoresMlflow:
    """Acceptance criterion: layer_scores logged to MLflow."""

    def test_layer_scores_in_mlflow(self):
        calls = _run_training_with_mlflow_capture(
            eval_losses=[2.0, 1.5] * 20,
        )
        all_keys = set()
        for metrics_dict, _kwargs in calls:
            all_keys.update(metrics_dict.keys())
        # layer_scores should appear as individual metrics like layer_score_0, layer_score_1, etc.
        layer_score_keys = [k for k in all_keys if k.startswith("layer_score_")]
        # Or as a single "layer_scores" metric
        has_layer_scores = "layer_scores" in all_keys or len(layer_score_keys) > 0
        assert has_layer_scores, (
            f"Expected layer_score(s) in MLflow metrics, got: {sorted(all_keys)}"
        )


class TestMlflowDisabledSkipsMetrics:
    """When mlflow.enabled=False, no specialized metrics are sent."""

    def test_no_log_metrics_calls_when_disabled(self):
        """MLflow disabled → log_metrics never called."""
        from torch.utils.data import Dataset

        class _SimpleDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 100, (16,)),
                    "attention_mask": torch.ones(16, dtype=torch.long),
                    "labels": torch.randint(0, 100, (16,)),
                }

        model = _LoRAMockModel()
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()
        dataset = _SimpleDataset()
        import tempfile
        run_dir = Path(tempfile.mkdtemp(prefix="mlflow_disabled_"))
        metrics_mock = MagicMock()

        from src.eval.eval_loss import EvalLossResult

        eval_losses = [2.0, 1.5] * 20
        mock_eval_loss = MagicMock(side_effect=eval_losses)
        mock_eval_loss_detailed = MagicMock(
            side_effect=[
                EvalLossResult(avg_loss=v, num_batches=1, min_loss=v, max_loss=v)
                for v in eval_losses
            ]
        )

        fake_mlflow = _make_fake_mlflow()
        cfg = _make_config(
            logging={"run_dir": str(run_dir), "mlflow": {"enabled": False}},
        )

        deps = {
            "src.training.train_tg_lora.load_tokenizer": MagicMock(
                return_value=tokenizer
            ),
            "src.training.train_tg_lora.load_base_model": MagicMock(return_value=model),
            "src.training.train_tg_lora.apply_lora": MagicMock(return_value=model),
            "src.training.train_tg_lora.get_input_device": MagicMock(
                return_value="cpu"
            ),
            "src.training.train_tg_lora.load_dataset": MagicMock(return_value=dataset),
            "src.training.train_tg_lora.eval_loss": mock_eval_loss,
            "src.training.train_tg_lora.eval_loss_detailed": mock_eval_loss_detailed,
            "src.training.train_tg_lora.ensure_dir": MagicMock(return_value=run_dir),
            "src.training.train_tg_lora.RunMetrics": MagicMock(
                return_value=metrics_mock
            ),
            "src.training.train_tg_lora.count_parameters": MagicMock(
                return_value={"total": 100, "trainable": 50}
            ),
            "src.training.train_tg_lora.set_seed": MagicMock(),
        }

        with contextlib.ExitStack() as stack:
            for target, mock_obj in deps.items():
                stack.enter_context(patch(target, new=mock_obj))
            stack.enter_context(patch("src.utils.mlflow_logger._mlflow", fake_mlflow))
            train_tg_lora(cfg)

        fake_mlflow.log_metrics.assert_not_called()
        fake_mlflow.log_params.assert_not_called()
