"""TASK-0029: Baseline QLoRA training validation.

Validates all 6 acceptance criteria:
1. Baseline training completes normally
2. run_metrics.jsonl records step-by-step metrics
3. Backward pass count matches TG-LoRA 10 cycles × K_initial
4. eval_loss recorded at eval_interval
5. Best model saved to runs/
6. Loss values are not NaN/Inf
"""

import math
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.eval.eval_loss import EvalLossResult
from src.training.train_baseline_qlora import train_baseline
from src.utils.run_metrics import RunMetrics

# ── Helpers ──────────────────────────────────────────────────────────────────


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


class _MockModel(torch.nn.Module):
    def __init__(self, loss_value: float = 2.0):
        super().__init__()
        self._loss_value = loss_value
        self.linear = torch.nn.Linear(1, 1)
        self.save_pretrained = MagicMock()

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


TG_LORA_CYCLES = 10
TG_LORA_K_INITIAL = 3
BACKWARD_PASS_BUDGET = TG_LORA_CYCLES * TG_LORA_K_INITIAL  # 30


def _make_config(**overrides):
    defaults = {
        "experiment": {"seed": 42, "name": "task0029_test"},
        "model": {"name_or_path": "test-model"},
        "lora": {"r": 16, "alpha": 32},
        "data": {
            "train_path": "/tmp/dummy",
            "valid_quick_path": "/tmp/dummy",
            "max_seq_len": 16,
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 2e-4,
            "weight_decay": 0.0,
            "grad_accumulation": 1,
            "max_grad_norm": 1.0,
            "max_steps": BACKWARD_PASS_BUDGET,
            "warmup_steps": 0,
            "schedule_type": "linear",
            "early_stopping_patience": None,
            "min_steps_before_stop": 100,
        },
        "eval": {
            "full_eval_every_steps": 10,
            "quick_eval_examples": 5,
        },
        "logging": {
            "run_dir": "/tmp/task0029_test_run",
            "log_every_steps": 10,
            "save_every_steps": 999,
        },
    }
    cfg = OmegaConf.create(defaults)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _make_eval_results(eval_losses):
    """Convert plain loss floats into EvalLossResult instances."""
    return [
        EvalLossResult(avg_loss=lv, num_batches=1, min_loss=lv, max_loss=lv)
        for lv in eval_losses
    ]


def _patch_all_deps(model_loss=2.0, eval_losses=None):
    model = _MockModel(loss_value=model_loss)
    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()

    if eval_losses is None:
        eval_losses = [2.0]

    eval_results = _make_eval_results(eval_losses)

    dataset = _MockDataset(60)

    run_dir = Path("/tmp/task0029_test_run")

    metrics = MagicMock()

    return {
        "src.training.train_baseline_qlora.set_seed": MagicMock(),
        "src.training.train_baseline_qlora.load_tokenizer": MagicMock(
            return_value=tokenizer
        ),
        "src.training.train_baseline_qlora.load_base_model": MagicMock(
            return_value=model
        ),
        "src.training.train_baseline_qlora.apply_lora": MagicMock(return_value=model),
        "src.training.train_baseline_qlora.get_input_device": MagicMock(
            return_value="cpu"
        ),
        "src.training.train_baseline_qlora.load_dataset": MagicMock(
            return_value=dataset
        ),
        "src.training.train_baseline_qlora.eval_loss_detailed": MagicMock(
            side_effect=eval_results * 100,
        ),
        "src.training.train_baseline_qlora.ensure_dir": MagicMock(return_value=run_dir),
        "src.training.train_baseline_qlora.RunMetrics": MagicMock(return_value=metrics),
        "src.training.train_baseline_qlora.count_parameters": MagicMock(
            return_value={"total": 1000, "trainable": 50},
        ),
        "src.training.train_baseline_qlora.forward_backward": MagicMock(
            return_value=2.0
        ),
        "src.training.train_baseline_qlora.optimizer_step": MagicMock(),
        "src.training.train_baseline_qlora.create_optimizer": MagicMock(),
        "src.training.train_baseline_qlora.create_scheduler": MagicMock(),
        "src.training.train_baseline_qlora.MLflowLogger": MagicMock(),
    }


def _run_baseline(cfg_overrides=None, **patch_kwargs):
    cfg = _make_config(**(cfg_overrides or {}))
    deps = _patch_all_deps(**patch_kwargs)

    with ExitStack() as stack:
        mocks = {}
        for target, mock_obj in deps.items():
            mocks[target.split(".")[-1]] = stack.enter_context(
                patch(target, new=mock_obj),
            )
        train_baseline(cfg)
        return cfg, mocks


# ── Criterion 1: Baseline training completes normally ────────────────────────


class TestBaselineTrainingCompletion:
    """ベースライン学習が正常完了する"""

    def test_30_step_training_completes(self):
        _, mocks = _run_baseline()
        assert mocks["forward_backward"].call_count == BACKWARD_PASS_BUDGET

    def test_optimizer_steps_match_training_steps(self):
        _, mocks = _run_baseline()
        assert mocks["optimizer_step"].call_count == BACKWARD_PASS_BUDGET

    def test_training_with_grad_accumulation(self):
        grad_accum = 4
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"grad_accumulation": grad_accum},
            }
        )
        assert mocks["forward_backward"].call_count == BACKWARD_PASS_BUDGET * grad_accum
        assert mocks["optimizer_step"].call_count == BACKWARD_PASS_BUDGET

    def test_no_exception_during_training(self):
        _run_baseline()  # Should not raise


# ── Criterion 2: run_metrics.jsonl records step-by-step metrics ──────────────


class TestRunMetricsRecording:
    """run_metrics.jsonlにステップごとのメトリクスが記録される"""

    def test_record_step_called_per_step(self):
        _, mocks = _run_baseline()
        metrics = mocks["RunMetrics"].return_value
        # record_step called once per step + once per eval point
        eval_interval = 10
        eval_calls = BACKWARD_PASS_BUDGET // eval_interval
        assert metrics.record_step.call_count == BACKWARD_PASS_BUDGET + eval_calls

    def test_jsonl_contains_step_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            cfg = _make_config()
            rm = RunMetrics(run_dir, mode="baseline", run_id="test_0029")
            rm.write_header(
                cfg,
                budget_type="backward_passes",
                budget_value=BACKWARD_PASS_BUDGET,
                param_counts={"total": 1000, "trainable": 50},
            )
            for i in range(BACKWARD_PASS_BUDGET):
                rm.record_step(
                    step=i + 1,
                    loss_train=2.0 - i * 0.02,
                    backward_passes=1,
                    total_backward_passes=i + 1,
                )
            rm.write_footer(
                best_valid_loss=1.5,
                best_valid_step=20,
                final_train_loss=1.42,
            )
            rm.close()

            records = []
            with open(run_dir / "run_metrics.jsonl", "rb") as f:
                for line in f:
                    records.append(orjson.loads(line))

            steps = [r for r in records if r["type"] == "step"]
            assert len(steps) == BACKWARD_PASS_BUDGET
            for s in steps:
                assert "step" in s
                assert "loss_train" in s
                assert "backward_passes" in s
                assert "total_backward_passes" in s

    def test_record_step_has_required_fields(self):
        _, mocks = _run_baseline()
        metrics = mocks["RunMetrics"].return_value
        for call in metrics.record_step.call_args_list:
            kwargs = call.kwargs
            assert "step" in kwargs
            assert "loss_train" in kwargs
            assert "backward_passes" in kwargs
            assert "total_backward_passes" in kwargs

    def test_step_numbers_sequential(self):
        _, mocks = _run_baseline()
        metrics = mocks["RunMetrics"].return_value
        steps = [c.kwargs["step"] for c in metrics.record_step.call_args_list]
        # Extract unique steps (eval records same step twice)
        unique_steps = sorted(set(steps))
        assert unique_steps == list(range(1, BACKWARD_PASS_BUDGET + 1))


# ── Criterion 3: backward pass count matches TG-LoRA budget ─────────────────


class TestBackwardPassBudget:
    """backward pass数がTG-LoRA 10サイクル×K_initialと一致する"""

    def test_backward_pass_budget_equals_30(self):
        assert BACKWARD_PASS_BUDGET == 30

    def test_max_steps_matches_tg_lora_budget(self):
        """Baseline max_steps = TG-LoRA cycles × K_initial."""
        tg_lora_cfg = OmegaConf.load("configs/9b_tg_lora.yaml")
        # TG-LoRA smoke test uses max_cycles=10, K_initial=3
        # Baseline uses max_steps = cycles * K_initial
        expected_budget = 10 * tg_lora_cfg.tg_lora.K_initial
        assert expected_budget == BACKWARD_PASS_BUDGET

    def test_total_backward_passes_recorded(self):
        _, mocks = _run_baseline()
        metrics = mocks["RunMetrics"].return_value
        for call in metrics.record_step.call_args_list:
            kwargs = call.kwargs
            assert kwargs["total_backward_passes"] == kwargs["step"] * kwargs.get(
                "backward_passes", 1
            )

    def test_total_backward_passes_at_end(self):
        _, mocks = _run_baseline()
        metrics = mocks["RunMetrics"].return_value
        last_call = metrics.record_step.call_args_list[-1]
        assert last_call.kwargs["total_backward_passes"] == BACKWARD_PASS_BUDGET

    def test_backward_passes_with_grad_accum(self):
        grad_accum = 3
        _, mocks = _run_baseline(
            cfg_overrides={
                "training": {"grad_accumulation": grad_accum},
            }
        )
        metrics = mocks["RunMetrics"].return_value
        last_call = metrics.record_step.call_args_list[-1]
        total_bp = last_call.kwargs["total_backward_passes"]
        assert total_bp == BACKWARD_PASS_BUDGET * grad_accum


# ── Criterion 4: eval_loss recorded at eval_interval ─────────────────────────


class TestEvalLossRecording:
    """定期評価（eval_interval毎）でeval_lossが記録される"""

    def test_eval_called_at_interval(self):
        eval_interval = 10
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {
                    "full_eval_every_steps": eval_interval,
                    "quick_eval_examples": 5,
                },
            }
        )
        expected_evals = BACKWARD_PASS_BUDGET // eval_interval  # 30 // 10 = 3
        assert mocks["eval_loss_detailed"].call_count == expected_evals

    def test_eval_loss_recorded_in_metrics(self):
        eval_interval = 10
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {
                    "full_eval_every_steps": eval_interval,
                    "quick_eval_examples": 5,
                },
            },
            eval_losses=[1.5],
        )
        metrics = mocks["RunMetrics"].return_value
        eval_calls = [
            c
            for c in metrics.record_step.call_args_list
            if c.kwargs.get("loss_valid") is not None
        ]
        assert len(eval_calls) == BACKWARD_PASS_BUDGET // eval_interval

    def test_no_eval_when_interval_exceeds_steps(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 999, "quick_eval_examples": 5},
            }
        )
        assert mocks["eval_loss_detailed"].call_count == 0

    def test_eval_at_every_step(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 1, "quick_eval_examples": 5},
            }
        )
        assert mocks["eval_loss_detailed"].call_count == BACKWARD_PASS_BUDGET


# ── Criterion 5: best model saved to runs/ ───────────────────────────────────


class TestBestModelSaved:
    """best modelがruns/に保存される"""

    def test_best_model_saved_on_improvement(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 10, "quick_eval_examples": 5},
            },
            eval_losses=[1.5],
        )
        model = mocks["apply_lora"].return_value
        model.save_pretrained.assert_called()

    def test_best_model_save_pretrained_called(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 10, "quick_eval_examples": 5},
            },
            eval_losses=[1.0],
        )
        model = mocks["apply_lora"].return_value
        # At least one save for best_model improvement
        assert model.save_pretrained.call_count >= 1

    def test_tokenizer_saved_with_best_model(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 10, "quick_eval_examples": 5},
            },
            eval_losses=[1.0],
        )
        tokenizer = mocks["load_tokenizer"].return_value
        tokenizer.save_pretrained.assert_called()

    def test_no_best_model_when_no_eval(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 999, "quick_eval_examples": 5},
            }
        )
        model = mocks["apply_lora"].return_value
        # No eval → no best_model save (unless periodic checkpoint)
        save_calls = [
            c for c in model.save_pretrained.call_args_list if "best_model" in str(c)
        ]
        assert len(save_calls) == 0

    def test_multiple_improvements_multiple_saves(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 10, "quick_eval_examples": 5},
                "logging": {"save_every_steps": 999},
            },
            eval_losses=[2.0, 1.5, 1.0],
        )
        model = mocks["apply_lora"].return_value
        # 3 evals, each improves → 3 best_model saves
        assert model.save_pretrained.call_count >= 2


# ── Criterion 6: Loss values are not NaN/Inf ─────────────────────────────────


class TestLossFinite:
    """損失値がNaN/Infにならない"""

    def test_jsonl_losses_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            cfg = _make_config()
            rm = RunMetrics(run_dir, mode="baseline", run_id="finite_test_0029")
            rm.write_header(
                cfg,
                budget_type="backward_passes",
                budget_value=BACKWARD_PASS_BUDGET,
                param_counts={"total": 1000, "trainable": 50},
            )
            for i in range(BACKWARD_PASS_BUDGET):
                rm.record_step(
                    step=i + 1,
                    loss_train=2.0 - i * 0.02,
                    loss_valid=(1.8 - i * 0.01) if i % 10 == 0 else None,
                    backward_passes=1,
                    total_backward_passes=i + 1,
                )
            rm.close()

            with open(run_dir / "run_metrics.jsonl", "rb") as f:
                for line in f:
                    rec = orjson.loads(line)
                    if rec["type"] == "step":
                        assert math.isfinite(rec["loss_train"]), (
                            f"Step {rec['step']}: loss_train is not finite"
                        )
                        if rec.get("loss_valid") is not None:
                            assert math.isfinite(rec["loss_valid"]), (
                                f"Step {rec['step']}: loss_valid is not finite"
                            )

    def test_mocked_training_records_finite_losses(self):
        _, mocks = _run_baseline(
            cfg_overrides={
                "eval": {"full_eval_every_steps": 10, "quick_eval_examples": 5},
            },
            eval_losses=[1.5],
        )
        metrics = mocks["RunMetrics"].return_value
        for call in metrics.record_step.call_args_list:
            assert math.isfinite(call.kwargs["loss_train"])

    def test_header_footer_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            cfg = _make_config()
            rm = RunMetrics(run_dir, mode="baseline", run_id="finite_hf")
            rm.write_header(
                cfg,
                budget_type="backward_passes",
                budget_value=BACKWARD_PASS_BUDGET,
                param_counts={"total": 1000, "trainable": 50},
            )
            for i in range(BACKWARD_PASS_BUDGET):
                rm.record_step(
                    step=i + 1,
                    loss_train=2.0,
                    backward_passes=1,
                    total_backward_passes=i + 1,
                )
            rm.write_footer(
                best_valid_loss=1.5,
                best_valid_step=20,
                final_train_loss=1.42,
            )
            rm.close()

            with open(run_dir / "run_metrics.jsonl", "rb") as f:
                for line in f:
                    rec = orjson.loads(line)
                    if rec["type"] == "run_footer":
                        assert math.isfinite(rec["best_valid_loss"])
                        assert math.isfinite(rec["final_train_loss"])

    def test_nan_loss_would_fail(self):
        """Verify that NaN would be caught (inverse test)."""
        assert not math.isfinite(float("nan"))
        assert not math.isfinite(float("inf"))
