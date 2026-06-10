"""E2E smoke tests for OptimizerLifecycleManager → RunMetrics header pipeline.

Verifies that optimizer_lifecycle policy configuration flows through
train_tg_lora → OptimizerLifecycleManager → RunMetrics.write_header
and appears correctly in run_metrics.jsonl header records.
"""

import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


from src.eval.eval_loss import EvalLossResult
from src.training.train_tg_lora import train_tg_lora
from tests.test_training_integration import (
    _LoRAMockModel,
    _SimpleDataset,
    _make_config,
)


def _run_e2e_training(tmp_path: Path, cfg_overrides: dict | None = None):
    """Run mock training with real RunMetrics. Returns parsed JSONL records."""
    overrides = {
        "model": {"name_or_path": "mock-model", "device": "cpu"},
        "lora": {"r": 8, "alpha": 16},
        "logging": {
            "run_dir": str(tmp_path),
            "log_every_cycles": 1,
            "save_every_cycles": 25,
            "mlflow": {"enabled": False},
        },
        "training": {"max_cycles": 2},
    }
    # Merge caller overrides
    if cfg_overrides:
        for key in ("training", "logging"):
            if key in cfg_overrides:
                overrides.setdefault(key, {}).update(cfg_overrides[key])
        for key, val in cfg_overrides.items():
            if key not in ("training", "logging"):
                overrides[key] = val

    cfg = _make_config(**overrides)

    model = _LoRAMockModel()
    model._loss_val = 2.0

    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()

    eval_losses = [2.0, 1.5] * 50

    dataset = _SimpleDataset(10)

    mock_load_tokenizer = MagicMock(return_value=tokenizer)
    mock_load_base_model = MagicMock(return_value=model)
    mock_apply_lora = MagicMock(return_value=model)
    mock_get_input_device = MagicMock(return_value="cpu")
    mock_load_dataset = MagicMock(return_value=dataset)
    mock_eval_loss = MagicMock(side_effect=eval_losses)
    mock_eval_loss_detailed = MagicMock(
        side_effect=[
            EvalLossResult(avg_loss=v, num_batches=1, min_loss=v, max_loss=v)
            for v in eval_losses
        ]
    )
    mock_set_seed = MagicMock()

    patches = {
        "src.training.train_tg_lora.load_tokenizer": mock_load_tokenizer,
        "src.training.train_tg_lora.load_base_model": mock_load_base_model,
        "src.training.train_tg_lora.apply_lora": mock_apply_lora,
        "src.training.train_tg_lora.get_input_device": mock_get_input_device,
        "src.training.train_tg_lora.load_dataset": mock_load_dataset,
        "src.training.train_tg_lora.eval_loss": mock_eval_loss,
        "src.training.train_tg_lora.eval_loss_detailed": mock_eval_loss_detailed,
        "src.training.train_tg_lora.set_seed": mock_set_seed,
    }

    with contextlib.ExitStack() as stack:
        for target, mock_obj in patches.items():
            stack.enter_context(patch(target, new=mock_obj))
        train_tg_lora(cfg)

    jsonl_path = tmp_path / "run_metrics.jsonl"
    assert jsonl_path.exists(), f"run_metrics.jsonl not found in {tmp_path}"

    return [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]


def test_recreate_policy_in_run_metrics_header(tmp_path):
    """recreate_per_cycle policy appears in run_metrics.jsonl header."""
    records = _run_e2e_training(
        tmp_path,
        cfg_overrides={"training": {"optimizer_lifecycle": "recreate_per_cycle"}},
    )

    header = records[0]
    assert header["type"] == "run_header"
    assert header["optimizer_lifecycle"] == "recreate_per_cycle"


def test_reuse_policy_in_run_metrics_header(tmp_path):
    """reuse_state_reset_experimental policy appears in run_metrics.jsonl header."""
    records = _run_e2e_training(
        tmp_path,
        cfg_overrides={
            "training": {"optimizer_lifecycle": "reuse_state_reset_experimental"}
        },
    )

    header = records[0]
    assert header["type"] == "run_header"
    assert header["optimizer_lifecycle"] == "reuse_state_reset_experimental"


def test_missing_field_in_run_metrics_header(tmp_path):
    """When optimizer_lifecycle field is absent, header contains null."""
    # _make_config() defaults do NOT include optimizer_lifecycle
    records = _run_e2e_training(tmp_path)

    header = records[0]
    assert header["type"] == "run_header"
    assert header["optimizer_lifecycle"] is None
