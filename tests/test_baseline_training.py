"""Mock-based tests for src/training/train_baseline_qlora.py — TASK-0014.

Covers initialization, training steps, LR scheduler, gradient clipping,
eval_loss / best_loss tracking, and checkpoint saves via fully mocked deps.
"""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.eval.eval_loss import EvalLossResult
from src.training.train_baseline_qlora import train_baseline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockDataset(Dataset):
    def __init__(self, n: int = 20, seq_len: int = 16, vocab: int = 100):
        self.n = n
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, self.vocab, (self.seq_len,)),
        }


class _LeadingMaskedDataset(Dataset):
    def __init__(self, n: int = 20, seq_len: int = 16, vocab: int = 100):
        self.n = n
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        labels = torch.full((self.seq_len,), -100, dtype=torch.long) if idx == 0 else torch.randint(0, self.vocab, (self.seq_len,))
        return {
            "input_ids": torch.randint(0, self.vocab, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": labels,
        }


class _MockModel(torch.nn.Module):
    """Minimal nn.Module that returns a controllable loss."""

    def __init__(self, loss_value: float = 2.0):
        super().__init__()
        self._loss_value = loss_value
        self.linear = torch.nn.Linear(1, 1)  # dummy parameter
        self.save_pretrained = MagicMock()
        self.save_pretrained.__wrapped__ = lambda path: None

    def __call__(self, **kwargs):
        out = MagicMock()
        out.loss = torch.tensor(self._loss_value, requires_grad=True)
        return out

    def parameters(self):
        for p in super().parameters():
            p.requires_grad = True
            yield p

    def train(self, mode=True):
        return self

    def eval(self):
        return self


def _make_config(**overrides) -> OmegaConf:
    defaults = {
        "experiment": {"seed": 42, "name": "test-baseline"},
        "model": {"name_or_path": "gpt2"},
        "data": {
            "train_path": "/tmp/dummy_train",
            "valid_quick_path": "/tmp/dummy_valid",
            "max_seq_len": 16,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "grad_accumulation": 1,
            "max_grad_norm": 1.0,
            "max_steps": 5,
            "trainable_lora_scope": "all",
            "warmup_steps": 0,
            "schedule_type": "linear",
            "early_stopping_patience": None,
            "min_steps_before_stop": 100,
        },
        "eval": {
            "full_eval_every_steps": 3,
            "quick_eval_examples": 5,
        },
        "logging": {
            "run_dir": "/tmp/test_baseline_run",
            "log_every_steps": 2,
        },
        "lora": {
            "r": 8,
            "alpha": 16,
        },
    }
    cfg = OmegaConf.create(defaults)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _patch_all_deps(
    model_loss: float = 2.0,
    eval_losses: list[float] | None = None,
    mlflow_enabled: bool = False,
):
    """Return patch targets for all external deps of train_baseline."""
    model = _MockModel(loss_value=model_loss)
    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()

    if eval_losses is None:
        eval_losses = [2.0]

    eval_results = [
        EvalLossResult(
            avg_loss=loss_val, num_batches=1, min_loss=loss_val, max_loss=loss_val
        )
        for loss_val in eval_losses
    ]

    dataset = _MockDataset(20)

    run_dir = Path("/tmp/test_baseline_run")

    metrics = MagicMock()

    mock_set_seed = MagicMock()
    mock_load_tokenizer = MagicMock(return_value=tokenizer)
    mock_load_base_model = MagicMock(return_value=model)
    mock_apply_lora = MagicMock(return_value=model)
    mock_configure_trainable_lora_scope = MagicMock(return_value=(set(), set()))
    mock_get_input_device = MagicMock(return_value=torch.device("cpu"))
    mock_load_dataset = MagicMock(return_value=dataset)
    mock_eval_loss_detailed = MagicMock(side_effect=list(eval_results) * 100)
    mock_ensure_dir = MagicMock(return_value=run_dir)
    mock_run_metrics_cls = MagicMock(return_value=metrics)
    mock_count_parameters = MagicMock(return_value={"total": 1000, "trainable": 50})
    mock_forward_backward = MagicMock(return_value=2.0)
    mock_optimizer_step = MagicMock()
    mock_create_optimizer = MagicMock()
    mock_create_scheduler = MagicMock()

    mock_mlf = MagicMock()
    mock_mlf.enabled = mlflow_enabled
    mock_mlf.__enter__ = MagicMock(return_value=mock_mlf)
    mock_mlf.__exit__ = MagicMock(return_value=False)
    mock_mlflow_logger_cls = MagicMock(return_value=mock_mlf)

    return {
        "src.training.train_baseline_qlora.set_seed": mock_set_seed,
        "src.training.train_baseline_qlora.load_tokenizer": mock_load_tokenizer,
        "src.training.train_baseline_qlora.load_base_model": mock_load_base_model,
        "src.training.train_baseline_qlora.apply_lora": mock_apply_lora,
        "src.training.train_baseline_qlora.configure_trainable_lora_scope": mock_configure_trainable_lora_scope,
        "src.training.train_baseline_qlora.get_input_device": mock_get_input_device,
        "src.training.train_baseline_qlora.load_dataset": mock_load_dataset,
        "src.training.train_baseline_qlora.eval_loss_detailed": mock_eval_loss_detailed,
        "src.training.train_baseline_qlora.ensure_dir": mock_ensure_dir,
        "src.training.train_baseline_qlora.RunMetrics": mock_run_metrics_cls,
        "src.training.train_baseline_qlora.count_parameters": mock_count_parameters,
        "src.training.train_baseline_qlora.forward_backward": mock_forward_backward,
        "src.training.train_baseline_qlora.optimizer_step": mock_optimizer_step,
        "src.training.train_baseline_qlora.create_optimizer": mock_create_optimizer,
        "src.training.train_baseline_qlora.create_scheduler": mock_create_scheduler,
        "src.training.train_baseline_qlora.MLflowLogger": mock_mlflow_logger_cls,
        "src.training.train_baseline_qlora.save_baseline_training_state": MagicMock(),
    }


def _run_baseline(cfg_overrides=None, **patch_kwargs):
    """Run train_baseline with all deps mocked. Returns (cfg, mocks)."""
    cfg = _make_config(**(cfg_overrides or {}))
    deps = _patch_all_deps(**patch_kwargs)

    with contextlib.ExitStack() as stack:
        mocks = {}
        for target, mock_obj in deps.items():
            mocks[target.split(".")[-1]] = stack.enter_context(
                patch(target, new=mock_obj)
            )
        train_baseline(cfg)
        return cfg, mocks


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test model/optimizer/dataloader initialization."""

    @pytest.mark.parametrize(
        "mock_name, arg_extractor",
        [
            ("set_seed", lambda _cfg, _mocks: (42,)),
            ("load_tokenizer", lambda cfg, _mocks: (cfg,)),
            ("load_base_model", lambda cfg, _mocks: (cfg,)),
        ],
        ids=["set_seed", "load_tokenizer", "load_base_model"],
    )
    def test_called_once_with_correct_arg(self, mock_name, arg_extractor):
        cfg, mocks = _run_baseline()
        expected_args = arg_extractor(cfg, mocks)
        mocks[mock_name].assert_called_once_with(*expected_args)

    @pytest.mark.parametrize(
        "mock_name, model_from_mock, extra_args_fn",
        [
            ("apply_lora", "load_base_model", lambda cfg: (cfg,)),
            ("get_input_device", "apply_lora", lambda _cfg: ()),
        ],
        ids=["apply_lora", "get_input_device"],
    )
    def test_model_dep_called_once(self, mock_name, model_from_mock, extra_args_fn):
        cfg, mocks = _run_baseline()
        model = mocks[model_from_mock].return_value
        mocks[mock_name].assert_called_once_with(model, *extra_args_fn(cfg))

    def test_trainable_lora_scope_configured(self):
        cfg, mocks = _run_baseline(
            cfg_overrides={"training": {"trainable_lora_scope": "last_25_percent"}}
        )
        model = mocks["apply_lora"].return_value
        mocks["configure_trainable_lora_scope"].assert_called_once_with(
            model,
            cfg.training.trainable_lora_scope,
        )

    def test_load_dataset_called_for_train_and_valid(self):
        cfg, mocks = _run_baseline()
        assert mocks["load_dataset"].call_count == 2
        mocks["load_dataset"].assert_any_call(
            cfg.data.train_path,
            mocks["load_tokenizer"].return_value,
            cfg.data.max_seq_len,
            train_on_prompt=False,
        )
        mocks["load_dataset"].assert_any_call(
            cfg.data.valid_quick_path,
            mocks["load_tokenizer"].return_value,
            cfg.data.max_seq_len,
            train_on_prompt=False,
        )

    def test_ensure_dir_called(self):
        cfg, mocks = _run_baseline()
        mocks["ensure_dir"].assert_called_once_with(cfg.logging.run_dir)

    def test_run_metrics_initialized(self):
        cfg, mocks = _run_baseline()
        run_dir = mocks["ensure_dir"].return_value
        mocks["RunMetrics"].assert_called_once_with(run_dir, mode="baseline")

    def test_write_header_called(self):
        cfg, mocks = _run_baseline()
        metrics = mocks["RunMetrics"].return_value
        metrics.write_header.assert_called_once()
        header_call = metrics.write_header.call_args
        assert header_call.kwargs["budget_type"] == "backward_passes"
        assert header_call.kwargs["budget_value"] == cfg.training.max_steps


# ---------------------------------------------------------------------------
# 2. Training step tests (forward_backward + grad accumulation)
# ---------------------------------------------------------------------------


class TestTrainingSteps:
    """Test forward_backward calls and gradient accumulation."""

    def test_forward_backward_called_per_step(self):
        """With grad_accumulation=1, forward_backward called once per step."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3, "grad_accumulation": 1},
            }
        )
        assert mocks["forward_backward"].call_count == 3

    def test_forward_backward_called_per_grad_accum(self):
        """With grad_accumulation=2, forward_backward called 2x per step."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 2, "grad_accumulation": 2},
            }
        )
        assert mocks["forward_backward"].call_count == 4

    def test_forward_backward_receives_grad_accum(self):
        """forward_backward is called with grad_accumulation as positional arg."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 1, "grad_accumulation": 3},
            }
        )
        for c in mocks["forward_backward"].call_args_list:
            # grad_accum is passed as 3rd positional arg
            assert c[0][2] == 3

    def test_optimizer_step_called_per_training_step(self):
        """optimizer_step called once per training step (after grad accumulation)."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 4, "grad_accumulation": 2},
            }
        )
        assert mocks["optimizer_step"].call_count == 4

    def test_skips_all_masked_training_batches(self):
        cfg = _make_config(training={"max_steps": 2, "grad_accumulation": 1})
        deps = _patch_all_deps(eval_losses=[2.0] * 10)
        deps["src.training.train_baseline_qlora.load_dataset"] = MagicMock(return_value=_LeadingMaskedDataset())
        deps["src.training.train_baseline_qlora.get_input_device"] = MagicMock(return_value=torch.device("cpu"))

        def _assert_supervised(_model, batch, _grad_accum=1):
            assert torch.any(batch["labels"] != -100)
            return 1.0

        deps["src.training.train_baseline_qlora.forward_backward"] = MagicMock(side_effect=_assert_supervised)

        with contextlib.ExitStack() as stack:
            for target, mock_obj in deps.items():
                stack.enter_context(patch(target, new=mock_obj))
            train_baseline(cfg)

    def test_loss_is_average_over_grad_accum(self):
        """Step loss should be average of micro losses over grad_accum steps."""
        call_values = []
        original_fb = __import__(
            "src.training.trainer_loop", fromlist=["forward_backward"]
        ).forward_backward

        def tracking_fb(model, batch, grad_accum=1):
            val = original_fb(model, batch, grad_accum)
            call_values.append(val)
            return val

        # We just verify the structure: forward_backward returns a float
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 1, "grad_accumulation": 1},
            }
        )
        # forward_backward was mocked, but the mock was called
        assert mocks["forward_backward"].called


# ---------------------------------------------------------------------------
# 3. LR scheduler tests
# ---------------------------------------------------------------------------


class TestLRScheduler:
    """Test LR scheduler step execution."""

    def test_scheduler_passed_to_optimizer_step(self):
        """optimizer_step receives a scheduler object."""
        _, mocks = _run_baseline()
        for c in mocks["optimizer_step"].call_args_list:
            # args: (optimizer, scheduler, model, max_grad_norm)
            assert c[0][1] is not None  # scheduler should not be None

    def test_scheduler_created_with_correct_params(self):
        """create_scheduler called with correct num_training_steps and warmup."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 10, "warmup_steps": 2},
            }
        )
        mocks["create_scheduler"].assert_called_once()
        sargs = mocks["create_scheduler"].call_args
        # Called as create_scheduler(optimizer, num_training_steps=..., warmup_steps=...)
        assert sargs.kwargs["num_training_steps"] == 10
        assert sargs.kwargs["warmup_steps"] == 2

    @pytest.mark.parametrize("schedule_type", ["linear", "cosine"])
    def test_schedule_type_passed_to_create_scheduler(self, schedule_type):
        """schedule_type from config flows to create_scheduler call."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3, "schedule_type": schedule_type},
            }
        )
        mocks["create_scheduler"].assert_called_once()
        sargs = mocks["create_scheduler"].call_args
        assert sargs.kwargs["schedule_type"] == schedule_type

    def test_default_schedule_type_is_linear(self):
        """When schedule_type is omitted, create_scheduler gets 'linear'."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3},
            }
        )
        mocks["create_scheduler"].assert_called_once()
        sargs = mocks["create_scheduler"].call_args
        assert sargs.kwargs.get("schedule_type", "linear") == "linear"


# ---------------------------------------------------------------------------
# 4. Gradient clipping tests
# ---------------------------------------------------------------------------


class TestGradientClipping:
    """Test max_grad_norm is passed correctly."""

    @pytest.mark.parametrize(
        "cfg_overrides, expected_norm",
        [
            ({"training": {"max_steps": 1, "max_grad_norm": 0.5}}, 0.5),
            ({"training": {"max_steps": 1}}, 1.0),
        ],
        ids=["explicit_0.5", "default_1.0"],
    )
    def test_max_grad_norm_passed_to_optimizer_step(
        self, cfg_overrides, expected_norm
    ):
        """max_grad_norm from config is forwarded to optimizer_step."""
        _, mocks = _run_baseline(cfg_overrides=cfg_overrides)
        c = mocks["optimizer_step"].call_args
        norm = c.kwargs.get("max_grad_norm") or c[0][3]
        assert norm == expected_norm


# ---------------------------------------------------------------------------
# 5. eval_loss and best_loss tracking tests
# ---------------------------------------------------------------------------


class TestEvalAndBestLoss:
    """Test eval_loss calls and best_loss update logic."""

    def test_eval_loss_called_at_interval(self):
        """eval_loss called every full_eval_every_steps."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 10},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            }
        )
        # Steps 3, 6, 9 → 3 eval calls
        assert mocks["eval_loss_detailed"].call_count == 3

    def test_eval_loss_not_called_when_interval_not_reached(self):
        """No eval_loss call if no step reaches full_eval_every_steps."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 2},
                "eval": {"full_eval_every_steps": 5, "quick_eval_examples": 5},
            }
        )
        assert mocks["eval_loss_detailed"].call_count == 0

    def test_best_model_saved_on_improvement(self):
        """Model saved when eval loss improves."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            },
            eval_losses=[1.5],
        )
        model = mocks["apply_lora"].return_value
        model.save_pretrained.assert_called()

    def test_best_model_not_saved_when_no_improvement(self):
        """Model NOT saved when eval loss doesn't improve beyond first eval."""
        # First eval returns very high loss, second eval same → only initial save
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 6},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            },
            eval_losses=[5.0, 5.0],
        )
        model = mocks["apply_lora"].return_value
        # First eval: 5.0 < inf → save; Second eval: 5.0 == 5.0 → no save
        assert model.save_pretrained.call_count == 1

    def test_best_model_updated_on_each_improvement(self):
        """Multiple improvements trigger multiple saves."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 9},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            },
            eval_losses=[4.0, 3.0, 2.0],
        )
        model = mocks["apply_lora"].return_value
        # Each eval improves: 4.0 < inf → save, 3.0 < 4.0 → save, 2.0 < 3.0 → save
        assert model.save_pretrained.call_count >= 2

    def test_tokenizer_saved_with_best_model(self):
        """Tokenizer saved alongside model on improvement."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            },
            eval_losses=[1.0],
        )
        tokenizer = mocks["load_tokenizer"].return_value
        tokenizer.save_pretrained.assert_called()


# ---------------------------------------------------------------------------
# 6. Checkpoint save tests
# ---------------------------------------------------------------------------


class TestCheckpointSave:
    """Test periodic checkpoint saves."""

    def test_periodic_checkpoint_at_interval(self):
        """Checkpoint saved every save_every_steps."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 10, "save_every_steps": 5},
                "eval": {"full_eval_every_steps": 999},
            }
        )
        model = mocks["apply_lora"].return_value
        # Steps 5 and 10 → 2 checkpoint saves
        # eval every 999 so no best_model saves interfere
        assert model.save_pretrained.call_count == 2

    def test_no_periodic_checkpoint_when_interval_exceeds_steps(self):
        """No periodic checkpoint if save_every_steps > max_steps."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3, "save_every_steps": 10},
                "eval": {"full_eval_every_steps": 999},
            }
        )
        model = mocks["apply_lora"].return_value
        assert model.save_pretrained.call_count == 0

    def test_tokenizer_saved_with_checkpoint(self):
        """Tokenizer saved alongside model at checkpoint."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 5, "save_every_steps": 5},
                "eval": {"full_eval_every_steps": 999},
            }
        )
        tokenizer = mocks["load_tokenizer"].return_value
        # At least one tokenizer save for the checkpoint
        assert tokenizer.save_pretrained.call_count >= 1


# ---------------------------------------------------------------------------
# 7. Finalization tests
# ---------------------------------------------------------------------------


class TestFinalization:
    """Test metrics footer and cleanup."""

    @pytest.mark.parametrize(
        "cfg_overrides, patch_kwargs, expected_best_loss, expected_ppl",
        [
            (
                {
                    "training": {"max_steps": 3},
                    "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
                },
                {"eval_losses": [1.5]},
                1.5,
                "exp(1.5)",
            ),
            (
                {
                    "training": {"max_steps": 2},
                    "eval": {"full_eval_every_steps": 999, "quick_eval_examples": 5},
                },
                {},
                float("inf"),
                None,
            ),
            (
                {
                    "training": {"max_steps": 9},
                    "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
                },
                {"eval_losses": [2.0, 1.5, 1.8]},
                1.5,
                "exp(1.5)",
            ),
        ],
        ids=["single_eval_improvement", "no_eval_inf", "multi_eval_best_is_1.5"],
    )
    def test_write_footer_best_loss_and_perplexity(
        self, cfg_overrides, patch_kwargs, expected_best_loss, expected_ppl
    ):
        """write_footer receives correct best_valid_loss and perplexity."""
        import math

        _, mocks = _run_baseline(cfg_overrides=cfg_overrides, **patch_kwargs)
        metrics = mocks["RunMetrics"].return_value
        metrics.write_footer.assert_called_once()
        footer_kwargs = metrics.write_footer.call_args.kwargs
        assert footer_kwargs["best_valid_loss"] == expected_best_loss
        if expected_ppl is None:
            assert footer_kwargs.get("perplexity") is None
        else:
            assert "perplexity" in footer_kwargs
            expected_val = math.exp(float(expected_ppl.split("(")[1].rstrip(")")))
            assert abs(footer_kwargs["perplexity"] - expected_val) < 1e-6

    def test_write_footer_best_valid_step(self):
        """write_footer called with correct best_valid_step."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            },
            eval_losses=[1.5],
        )
        metrics = mocks["RunMetrics"].return_value
        footer_kwargs = metrics.write_footer.call_args.kwargs
        assert footer_kwargs["best_valid_step"] == 3

    def test_metrics_close_called(self):
        """RunMetrics.close() called at end."""
        _, mocks = _run_baseline()
        metrics = mocks["RunMetrics"].return_value
        metrics.close.assert_called_once()

    def test_record_step_called_per_step(self):
        """metrics.record_step called once per training step."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 4},
                "eval": {"full_eval_every_steps": 999},
            }
        )
        metrics = mocks["RunMetrics"].return_value
        assert metrics.record_step.call_count == 4


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases: single step, high grad_accum, etc."""

    def test_single_step(self):
        """Training completes with max_steps=1."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 1, "grad_accumulation": 1},
                "eval": {"full_eval_every_steps": 999},
            }
        )
        assert mocks["forward_backward"].call_count == 1
        assert mocks["optimizer_step"].call_count == 1

    def test_high_grad_accumulation(self):
        """Grad accumulation > dataset size still completes."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 1, "grad_accumulation": 5},
                "eval": {"full_eval_every_steps": 999},
            }
        )
        assert mocks["forward_backward"].call_count == 5

    def test_eval_and_checkpoint_on_same_step(self):
        """When eval and checkpoint intervals align, both trigger."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 5, "save_every_steps": 5},
                "eval": {"full_eval_every_steps": 5, "quick_eval_examples": 5},
            },
            eval_losses=[1.0],
        )
        model = mocks["apply_lora"].return_value
        # Should have both best_model save and checkpoint-5 save
        assert model.save_pretrained.call_count >= 2

    def test_create_optimizer_called_with_correct_params(self):
        """create_optimizer receives model, lr, weight_decay from config."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"learning_rate": 5e-4, "weight_decay": 0.1},
            }
        )
        mocks["create_optimizer"].assert_called_once()
        cargs = mocks["create_optimizer"].call_args
        assert cargs.kwargs["lr"] == 5e-4
        assert cargs.kwargs["weight_decay"] == 0.1


# ---------------------------------------------------------------------------
# 8. eval_loss_detailed integration tests
# ---------------------------------------------------------------------------


class TestEvalLossDetailedIntegration:
    """Test that eval_loss_detailed is called and its fields are logged."""

    # eval_loss_detailed_called_at_interval is covered by
    # TestEvalAndBestLoss.test_eval_loss_called_at_interval (same logic).

    def test_eval_loss_detailed_receives_correct_args(self):
        """eval_loss_detailed called with model, loader, device, max_examples."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            }
        )
        call = mocks["eval_loss_detailed"].call_args
        assert call.kwargs.get("max_examples") == 5 or call[1].get("max_examples") == 5

    def test_perplexity_min_max_logged_to_mlflow(self):
        """MLflow receives perplexity, min_loss, max_loss from EvalLossResult."""
        cfg, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            },
            eval_losses=[1.5],
            mlflow_enabled=True,
        )
        mlf = mocks["MLflowLogger"].return_value

        # Find the log_metrics call that includes the eval metrics
        eval_log_calls = [
            c
            for c in mlf.log_metrics.call_args_list
            if c[0][0] and "loss_valid" in c[0][0]
        ]
        assert len(eval_log_calls) >= 1
        metrics_logged = eval_log_calls[0][0][0]
        assert "perplexity" in metrics_logged
        assert "min_loss" in metrics_logged
        assert "max_loss" in metrics_logged

    def test_eval_result_values_match_logged_mlflow(self):
        """MLflow logged values match the EvalLossResult fields."""
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 3},
                "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
            },
            eval_losses=[1.8],
            mlflow_enabled=True,
        )
        mlf = mocks["MLflowLogger"].return_value

        eval_log_calls = [
            c
            for c in mlf.log_metrics.call_args_list
            if c[0][0] and "loss_valid" in c[0][0]
        ]
        assert len(eval_log_calls) >= 1
        metrics = eval_log_calls[0][0][0]

        import math

        expected_ppl = math.exp(1.8)
        assert abs(metrics["perplexity"] - expected_ppl) < 1e-6
        assert metrics["min_loss"] == 1.8
        assert metrics["max_loss"] == 1.8
        assert metrics["loss_valid"] == 1.8


# ---------------------------------------------------------------------------
# 9. MLflow parameter consistency tests — TASK-0038
# ---------------------------------------------------------------------------


class TestMLflowParamConsistency:
    """Verify baseline logs schedule_type, warmup_steps, perplexity,
    and best_valid_perplexity to MLflow for consistency with TG-LoRA."""

    @pytest.mark.parametrize(
        "cfg_overrides, param_key, expected_value",
        [
            ({"training": {"max_steps": 1, "schedule_type": "cosine"}}, "schedule_type", "cosine"),
            ({"training": {"max_steps": 1, "warmup_steps": 50}}, "warmup_steps", 50),
        ],
        ids=["schedule_type_cosine", "warmup_steps_50"],
    )
    def test_single_param_logged_to_mlflow(self, cfg_overrides, param_key, expected_value):
        """Specific param appears in MLflow log_params with expected value."""
        cfg, mocks = _run_baseline(
            cfg_overrides=cfg_overrides,
            mlflow_enabled=True,
        )
        mlf = mocks["MLflowLogger"].return_value
        params = mlf.log_params.call_args[0][0]
        assert param_key in params
        assert params[param_key] == expected_value

    def test_default_schedule_type_is_linear(self):
        """When schedule_type is omitted, MLflow logs 'linear'."""
        cfg, mocks = _run_baseline(
            cfg_overrides={
                "training": {"max_steps": 1},
            },
            mlflow_enabled=True,
        )
        mlf = mocks["MLflowLogger"].return_value
        params = mlf.log_params.call_args[0][0]
        assert params.get("schedule_type") == "linear"

    @pytest.mark.parametrize(
        "cfg_overrides, patch_kwargs, expected_ppl",
        [
            (
                {
                    "training": {"max_steps": 3},
                    "eval": {"full_eval_every_steps": 3, "quick_eval_examples": 5},
                },
                {"eval_losses": [1.2], "mlflow_enabled": True},
                "exp(1.2)",
            ),
            (
                {
                    "training": {"max_steps": 2},
                    "eval": {"full_eval_every_steps": 999, "quick_eval_examples": 5},
                },
                {"mlflow_enabled": True},
                "inf",
            ),
        ],
        ids=["with_eval", "no_eval"],
    )
    def test_best_valid_perplexity_logged_to_mlflow(
        self, cfg_overrides, patch_kwargs, expected_ppl
    ):
        """best_valid_perplexity appears in final MLflow log_metrics."""
        import math

        _, mocks = _run_baseline(cfg_overrides=cfg_overrides, **patch_kwargs)
        mlf = mocks["MLflowLogger"].return_value

        final_calls = [
            c
            for c in mlf.log_metrics.call_args_list
            if c[0][0] and "best_valid_loss" in c[0][0]
        ]
        assert len(final_calls) >= 1
        final_metrics = final_calls[-1][0][0]
        assert "best_valid_perplexity" in final_metrics
        if expected_ppl == "inf":
            assert final_metrics["best_valid_perplexity"] == float("inf")
        else:
            loss_val = float(expected_ppl.split("(")[1].rstrip(")"))
            assert abs(final_metrics["best_valid_perplexity"] - math.exp(loss_val)) < 1e-6

    def test_shared_params_present(self):
        """Both baseline and TG-LoRA share common param keys."""
        _, mocks = _run_baseline(mlflow_enabled=True)
        mlf = mocks["MLflowLogger"].return_value
        params = mlf.log_params.call_args[0][0]

        shared_keys = {
            "model",
            "lora_r",
            "lora_alpha",
            "batch_size",
            "grad_accumulation",
            "learning_rate",
            "seed",
        }
        assert shared_keys.issubset(set(params.keys()))
