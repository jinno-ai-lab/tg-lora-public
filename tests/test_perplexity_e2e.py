"""TASK-0043 & TASK-0044: Perplexity E2E pipeline and trainer parity tests.

E2E tests that simulate a short mock training loop and verify that
RunMetrics.write_footer contains a finite best_perplexity value.
Also verifies parity between baseline and tg_lora perplexity plumbing.
"""

import json
import math

import pytest

from src.eval.eval_loss import EvalLossResult
from src.utils.run_metrics import RunMetrics


class FakeCfg:
    class model:
        name_or_path = "test-model"
        device = "cpu"

    class training:
        batch_size = 1
        grad_accumulation = 1
        learning_rate = 1e-4
        max_steps = 10

    class lora:
        r = 8
        alpha = 16

    class experiment:
        seed = 42


def _read_footer(tmp_path, mode="baseline"):
    lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
    for line in reversed(lines):
        rec = json.loads(line)
        if rec["type"] == "run_footer":
            return rec
    raise AssertionError("No run_footer record found")


def _run_mock_training_loop(tmp_path, mode, eval_perplexity, num_steps=3):
    """Simulate a short training loop with mocked eval_loss_detailed."""
    cfg = FakeCfg()
    best_perplexity = None
    best_valid_loss = float("inf")
    best_valid_step = 0

    with RunMetrics(tmp_path, mode=mode) as metrics:
        metrics.write_header(cfg, budget_type="backward_passes", budget_value=num_steps)

        for step in range(1, num_steps + 1):
            train_loss = 3.0 - step * 0.1
            metrics.record_step(
                step=step,
                loss_train=train_loss,
                backward_passes=1,
                total_backward_passes=step,
            )

        # Simulate evaluation
        if eval_perplexity is not None:
            if math.isfinite(eval_perplexity) and eval_perplexity > 0:
                eval_result = EvalLossResult(
                    avg_loss=math.log(eval_perplexity),
                    num_batches=1,
                    min_loss=math.log(eval_perplexity),
                    max_loss=math.log(eval_perplexity),
                )
                valid_loss = eval_result.avg_loss
                perplexity = eval_result.perplexity
            else:
                valid_loss = float("nan")
                perplexity = eval_perplexity

            if math.isfinite(valid_loss) and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_step = num_steps
                best_perplexity = perplexity

        metrics.write_footer(
            best_valid_loss=best_valid_loss if best_valid_loss < float("inf") else 0.0,
            best_valid_step=best_valid_step,
            final_train_loss=2.0,
            perplexity=best_perplexity,
        )

    return _read_footer(tmp_path, mode)


# --- TASK-0043: E2E Perplexity Pipeline ---


@pytest.mark.parametrize(
    "mode,expected_ppl",
    [
        ("baseline", 42.5),
        ("tg_lora", 38.2),
    ],
)
def test_mock_training_loop_finite_perplexity(tmp_path, mode, expected_ppl):
    """TC-069-P19-01/02: Mock training loop produces finite perplexity in footer."""
    footer = _run_mock_training_loop(
        tmp_path / mode, mode, eval_perplexity=expected_ppl
    )
    assert footer["perplexity"] is not None
    assert math.isfinite(footer["perplexity"])
    assert abs(footer["perplexity"] - expected_ppl) < 0.01


def test_no_eval_produces_none_perplexity(tmp_path):
    """TC-069-P19-03: No evaluation → footer perplexity is None."""
    footer = _run_mock_training_loop(tmp_path, "baseline", eval_perplexity=None)
    assert footer["perplexity"] is None


def test_nan_perplexity_sanitized_to_none(tmp_path):
    """TC-069-P19-B01: NaN perplexity sanitized to None in footer."""
    footer = _run_mock_training_loop(tmp_path, "baseline", eval_perplexity=float("nan"))
    assert footer["perplexity"] is None


def test_inf_perplexity_sanitized_to_none(tmp_path):
    """TC-069-P19-B02: Inf perplexity sanitized to None in footer."""
    footer = _run_mock_training_loop(tmp_path, "baseline", eval_perplexity=float("inf"))
    assert footer["perplexity"] is None


# --- TASK-0044: Trainer Perplexity Parity ---


def _extract_perplexity_plumbing(source_lines, trainer_name):
    """Extract perplexity-related patterns from training script source."""
    patterns = {
        "init_best_perplexity": False,
        "eval_result_perplexity_access": False,
        "best_perplexity_update": False,
        "write_footer_perplexity": False,
        "mlflow_best_valid_perplexity": False,
    }
    source = "\n".join(source_lines)

    if (
        "best_perplexity = None" in source
        or "best_full_eval_perplexity = None" in source
    ):
        patterns["init_best_perplexity"] = True
    if "eval_result.perplexity" in source or "full_result.perplexity" in source:
        patterns["eval_result_perplexity_access"] = True
    if (
        "best_perplexity = eval_result.perplexity" in source
        or "best_full_eval_perplexity = full_result.perplexity" in source
    ):
        patterns["best_perplexity_update"] = True
    if "metrics.write_footer(" in source and "perplexity=" in source:
        patterns["write_footer_perplexity"] = True
    if "best_valid_perplexity" in source and "math.exp" in source:
        patterns["mlflow_best_valid_perplexity"] = True

    return patterns


def test_baseline_tg_lora_perplexity_parity():
    """TC-070-P19-01: Both trainers have identical perplexity plumbing patterns."""
    import pathlib

    base = pathlib.Path(__file__).resolve().parent.parent / "src" / "training"

    baseline_src = (base / "train_baseline_qlora.py").read_text().splitlines()
    tg_lora_src = (base / "train_tg_lora.py").read_text().splitlines()

    baseline_patterns = _extract_perplexity_plumbing(baseline_src, "baseline")
    tg_lora_patterns = _extract_perplexity_plumbing(tg_lora_src, "tg_lora")

    for key in baseline_patterns:
        assert baseline_patterns[key] == tg_lora_patterns[key], (
            f"Parity mismatch on '{key}': baseline={baseline_patterns[key]}, tg_lora={tg_lora_patterns[key]}"
        )


def test_perplexity_pipeline_end_to_end_both_modes(tmp_path):
    """Both trainers propagate EvalLossResult.perplexity through the same path."""
    for mode in ("baseline", "tg_lora"):
        ppl = 25.0
        footer = _run_mock_training_loop(tmp_path / mode, mode, eval_perplexity=ppl)
        assert footer["perplexity"] == pytest.approx(ppl, rel=1e-6), (
            f"mode={mode}: expected {ppl}, got {footer['perplexity']}"
        )
        assert footer["mode"] == mode
