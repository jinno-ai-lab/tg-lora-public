"""Tests for CLI entry points of train_tg_lora and train_baseline_qlora (TASK-0017).

Covers main() functions, argparse handling, StopIteration handling,
and __main__ guard paths using mocks to avoid GPU dependency.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from src.eval.eval_loss import EvalLossResult


def _full_yaml(tmp_path, name="test", seed=42):
    """Write a complete config YAML with valid data paths."""
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid_quick.jsonl"
    valid_full = tmp_path / "valid_full.jsonl"
    train.write_text("{}")
    valid.write_text("{}")
    valid_full.write_text("{}")
    return f"""
experiment:
  name: {name}
  seed: {seed}
model:
  name_or_path: test-model
  dtype: bfloat16
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  device_map: auto
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear
data:
  train_path: {train}
  valid_quick_path: {valid}
  valid_full_path: {valid_full}
  max_seq_len: 2048
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2.0e-4
  max_steps: 100
  warmup_steps: 0
  schedule_type: linear
eval:
  quick_eval_examples: 64
logging:
  run_dir: {tmp_path / "runs"}
"""


class TestTGLoRAMain:
    """Tests for src.training.train_tg_lora.main()."""

    def test_main_calls_train_with_config(self, tmp_path):
        config_path = tmp_path / "test.yaml"
        config_path.write_text(_full_yaml(tmp_path))

        with (
            patch("src.training.train_tg_lora.train_tg_lora") as mock_train,
            patch("src.utils.logging.setup_logging"),
        ):
            with patch("sys.argv", ["train_tg_lora.py", "--config", str(config_path)]):
                from src.training.train_tg_lora import main

                main()
                mock_train.assert_called_once()

    def test_main_missing_config_arg_exits(self):
        with patch("sys.argv", ["train_tg_lora.py"]):
            with pytest.raises(SystemExit):
                from src.training.train_tg_lora import main

                main()

    def test_main_passes_loaded_config(self, tmp_path):
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(_full_yaml(tmp_path, name="my_test", seed=7))

        with (
            patch("src.training.train_tg_lora.train_tg_lora") as mock_train,
            patch("src.utils.logging.setup_logging"),
        ):
            with patch("sys.argv", ["train_tg_lora.py", "--config", str(config_path)]):
                from src.training.train_tg_lora import main

                main()
                call_cfg = mock_train.call_args[0][0]
                assert call_cfg.experiment.name == "my_test"
                assert call_cfg.experiment.seed == 7

    def test_main_preflight_failure_exits(self, tmp_path):
        config_path = tmp_path / "test.yaml"
        config_path.write_text(_full_yaml(tmp_path))
        # Remove train data to trigger preflight failure
        (tmp_path / "train.jsonl").unlink()

        with (
            patch("src.training.train_tg_lora.train_tg_lora"),
            patch("src.utils.logging.setup_logging"),
        ):
            with patch("sys.argv", ["train_tg_lora.py", "--config", str(config_path)]):
                from src.training.train_tg_lora import main

                with pytest.raises(SystemExit, match="Preflight"):
                    main()


class TestBaselineMain:
    """Tests for src.training.train_baseline_qlora.main()."""

    def test_main_calls_train_with_config(self, tmp_path):
        config_path = tmp_path / "test.yaml"
        config_path.write_text(_full_yaml(tmp_path))

        with (
            patch("src.training.train_baseline_qlora.train_baseline") as mock_train,
            patch("src.utils.logging.setup_logging"),
        ):
            with patch(
                "sys.argv", ["train_baseline_qlora.py", "--config", str(config_path)]
            ):
                from src.training.train_baseline_qlora import main

                main()
                mock_train.assert_called_once()

    def test_main_missing_config_arg_exits(self):
        with patch("sys.argv", ["train_baseline_qlora.py"]):
            with pytest.raises(SystemExit):
                from src.training.train_baseline_qlora import main

                main()

    def test_main_passes_loaded_config(self, tmp_path):
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(_full_yaml(tmp_path, name="baseline_test", seed=99))

        with (
            patch("src.training.train_baseline_qlora.train_baseline") as mock_train,
            patch("src.utils.logging.setup_logging"),
        ):
            with patch(
                "sys.argv", ["train_baseline_qlora.py", "--config", str(config_path)]
            ):
                from src.training.train_baseline_qlora import main

                main()
                call_cfg = mock_train.call_args[0][0]
                assert call_cfg.experiment.name == "baseline_test"
                assert call_cfg.experiment.seed == 99

    def test_main_preflight_failure_exits(self, tmp_path):
        config_path = tmp_path / "test.yaml"
        config_path.write_text(_full_yaml(tmp_path))
        (tmp_path / "train.jsonl").unlink()

        with (
            patch("src.training.train_baseline_qlora.train_baseline"),
            patch("src.utils.logging.setup_logging"),
        ):
            with patch(
                "sys.argv", ["train_baseline_qlora.py", "--config", str(config_path)]
            ):
                from src.training.train_baseline_qlora import main

                with pytest.raises(SystemExit, match="Preflight"):
                    main()


class TestStopIterationHandling:
    """Test StopIteration handling in train_baseline_qlora (lines 79-83)."""

    def test_epoch_iter_resets_on_stop_iteration(self, tmp_path):
        """When DataLoader iterator is exhausted, it resets and continues."""
        from torch.utils.data import Dataset

        class _TinyDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 100, (16,)),
                    "attention_mask": torch.ones(16, dtype=torch.long),
                    "labels": torch.randint(0, 100, (16,)),
                }

        dataset = _TinyDataset()
        model = MagicMock()
        model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        model.save_pretrained = MagicMock()
        model.train = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=model)

        run_dir = tmp_path / "runs"
        run_dir.mkdir()
        metrics = MagicMock()

        cfg = OmegaConf.create(
            {
                "experiment": {"seed": 42, "name": "test-cli"},
                "model": {"name_or_path": "gpt2"},
                "data": {
                    "train_path": str(tmp_path / "train.jsonl"),
                    "valid_quick_path": str(tmp_path / "valid.jsonl"),
                    "max_seq_len": 16,
                },
                "training": {
                    "batch_size": 2,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "grad_accumulation": 1,
                    "max_grad_norm": 1.0,
                    "max_steps": 5,
                    "warmup_steps": 0,
                    "schedule_type": "linear",
                },
                "eval": {"full_eval_every_steps": 999, "quick_eval_examples": 5},
                "logging": {"run_dir": str(run_dir), "log_every_steps": 999},
                "lora": {"r": 8, "alpha": 16},
            }
        )

        with (
            patch("src.training.train_baseline_qlora.set_seed"),
            patch(
                "src.training.train_baseline_qlora.load_tokenizer",
                return_value=MagicMock(),
            ),
            patch(
                "src.training.train_baseline_qlora.load_base_model", return_value=model
            ),
            patch("src.training.train_baseline_qlora.apply_lora", return_value=model),
            patch(
                "src.training.train_baseline_qlora.get_input_device", return_value=torch.device("cpu")
            ),
            patch(
                "src.training.train_baseline_qlora.load_dataset", return_value=dataset
            ),
            patch(
                "src.training.train_baseline_qlora.eval_loss_detailed",
                return_value=EvalLossResult(
                    avg_loss=1.0, num_batches=1, min_loss=1.0, max_loss=1.0
                ),
            ),
            patch("src.training.train_baseline_qlora.ensure_dir", return_value=run_dir),
            patch("src.training.train_baseline_qlora.RunMetrics", return_value=metrics),
            patch(
                "src.training.train_baseline_qlora.count_parameters",
                return_value={"total": 100},
            ),
            patch(
                "src.training.train_baseline_qlora.forward_backward", return_value=1.0
            ),
            patch("src.training.train_baseline_qlora.optimizer_step"),
            patch("src.training.train_baseline_qlora.create_optimizer"),
            patch("src.training.train_baseline_qlora.create_scheduler"),
        ):
            from src.training.train_baseline_qlora import train_baseline

            train_baseline(cfg)

        metrics.write_footer.assert_called_once()


def _exec_main_guard(module):
    """Execute the __main__ guard of a module for in-process coverage tracking."""
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    guard = None
    for node in ast.iter_child_nodes(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            guard = node
            break
    assert guard is not None, f"__main__ guard not found in {module.__file__}"
    guard_module = ast.Module(body=[guard], type_ignores=[])
    code = compile(guard_module, module.__file__, "exec")
    orig = module.__dict__["__name__"]
    module.__dict__["__name__"] = "__main__"
    try:
        exec(code, module.__dict__)
    finally:
        module.__dict__["__name__"] = orig


class TestMainGuard:
    """In-process __main__ guard coverage via AST extraction."""

    def test_tg_lora_dunder_main(self, tmp_path):
        """__main__ guard in train_tg_lora.py invokes main()."""
        config_path = tmp_path / "guard.yaml"
        config_path.write_text(_full_yaml(tmp_path))
        with (
            patch("src.training.train_tg_lora.train_tg_lora") as mock_train,
            patch("src.utils.logging.setup_logging"),
            patch("sys.argv", ["train_tg_lora.py", "--config", str(config_path)]),
        ):
            import src.training.train_tg_lora as mod

            _exec_main_guard(mod)
            mock_train.assert_called_once()

    def test_baseline_dunder_main(self, tmp_path):
        """__main__ guard in train_baseline_qlora.py invokes main()."""
        config_path = tmp_path / "guard.yaml"
        config_path.write_text(_full_yaml(tmp_path))
        with (
            patch("src.training.train_baseline_qlora.train_baseline") as mock_train,
            patch("src.utils.logging.setup_logging"),
            patch(
                "sys.argv", ["train_baseline_qlora.py", "--config", str(config_path)]
            ),
        ):
            import src.training.train_baseline_qlora as mod

            _exec_main_guard(mod)
            mock_train.assert_called_once()
