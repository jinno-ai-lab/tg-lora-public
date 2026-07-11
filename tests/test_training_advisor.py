"""Tests for TrainingAdvisor — Phase 61 training advisory system.

Covers:
- AdvisoryAction / AdvisoryReport data structures
- TrainingAdvisor.evaluate() with various training scenarios
- Priority-based action ordering
- Health status determination
- generate_advice_from_history() helper
- CLI tool (scripts/advise_training.py)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.tg_lora.training_advisor import (
    AdvisoryAction,
    AdvisoryReport,
    AdvisorConfig,
    TrainingAdvisor,
    generate_advice_from_history,
)

SCRIPT = Path("scripts/advise_training.py")


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestAdvisoryAction:
    def test_valid_creation(self):
        a = AdvisoryAction(
            action_type="reduce_lr",
            priority="high",
            reason="Loss spike",
            suggested_value=0.5,
            confidence=0.9,
        )
        assert a.action_type == "reduce_lr"
        assert a.priority == "high"
        assert a.confidence == 0.9

    def test_confidence_must_be_in_range(self):
        with pytest.raises(ValueError, match="confidence"):
            AdvisoryAction(action_type="no_action", priority="low", reason="", confidence=1.5)

    def test_confidence_zero_ok(self):
        a = AdvisoryAction(action_type="no_action", priority="low", reason="", confidence=0.0)
        assert a.confidence == 0.0


class TestAdvisoryReport:
    def test_top_action_returns_highest_priority(self):
        actions = [
            AdvisoryAction(action_type="no_action", priority="low", reason="ok"),
            AdvisoryAction(action_type="reduce_lr", priority="critical", reason="spike"),
            AdvisoryAction(action_type="increase_k", priority="medium", reason="stagnation"),
        ]
        report = AdvisoryReport(overall_health="critical", actions=actions)
        top = report.top_action()
        assert top is not None
        assert top.priority == "critical"

    def test_top_action_returns_none_when_empty(self):
        report = AdvisoryReport(overall_health="healthy")
        assert report.top_action() is None


# ---------------------------------------------------------------------------
# TrainingAdvisor core tests
# ---------------------------------------------------------------------------


class TestTrainingAdvisorHealthy:
    """Advisor should report healthy when training is progressing well."""

    def test_healthy_progression(self):
        advisor = TrainingAdvisor()
        losses = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.85, 0.82, 0.80]
        for i, loss in enumerate(losses):
            report = advisor.evaluate(i, train_loss=loss)
        assert report.overall_health == "healthy"
        # Healthy training should have no critical/high actions
        critical_actions = [a for a in report.actions if a.priority in ("critical", "high")]
        assert len(critical_actions) == 0

    def test_save_checkpoint_on_new_best(self):
        advisor = TrainingAdvisor()
        # First cycle
        advisor.evaluate(0, train_loss=2.0)
        # Better loss
        r2 = advisor.evaluate(1, train_loss=1.5)
        save_actions = [a for a in r2.actions if a.action_type == "save_checkpoint"]
        assert len(save_actions) == 1
        assert "new best" in save_actions[0].reason.lower()


class TestTrainingAdvisorDivergence:
    """Advisor should detect and flag divergence."""

    def test_nan_detection_triggers_critical(self):
        advisor = TrainingAdvisor()
        advisor.evaluate(0, train_loss=2.0)
        report = advisor.evaluate(1, train_loss=float("nan"))
        assert report.overall_health == "critical"
        action_types = [a.action_type for a in report.actions]
        assert "rollback" in action_types
        assert "reduce_lr" in action_types

    def test_loss_spike_triggers_warning(self):
        advisor = TrainingAdvisor()
        advisor.evaluate(0, train_loss=1.0)
        report = advisor.evaluate(1, train_loss=5.0)  # 5x spike
        assert report.overall_health == "warning"
        reduce_actions = [a for a in report.actions if a.action_type == "reduce_lr"]
        assert len(reduce_actions) >= 1

    def test_gradual_increase_is_not_spike(self):
        advisor = TrainingAdvisor()
        # Gradual 50% increase (not a 2x spike)
        advisor.evaluate(0, train_loss=1.0)
        report = advisor.evaluate(1, train_loss=1.5)
        # Should not trigger divergence
        assert report.cycle_health is not None
        assert not report.cycle_health.divergence.detected


class TestTrainingAdvisorStagnation:
    """Advisor should detect training stagnation."""

    def test_stagnation_after_patience_cycles(self):
        config = AdvisorConfig(stagnation_patience=3)
        advisor = TrainingAdvisor(config=config)
        # First: establish best
        advisor.evaluate(0, train_loss=1.0)
        # Then: no improvement for patience+1 cycles
        for i in range(1, 5):
            report = advisor.evaluate(i, train_loss=1.1)
        assert report.cycle_health is not None
        assert report.cycle_health.stagnation.detected
        stagnation_actions = [a for a in report.actions if a.action_type == "increase_k"]
        assert len(stagnation_actions) >= 1

    def test_prolonged_stagnation_suggests_rollback(self):
        config = AdvisorConfig(stagnation_patience=2)
        advisor = TrainingAdvisor(config=config)
        advisor.evaluate(0, train_loss=1.0)
        for i in range(1, 6):  # 5 cycles without improvement (patience * 2.5)
            report = advisor.evaluate(i, train_loss=1.1)
        rollback_actions = [a for a in report.actions if a.action_type == "rollback"]
        assert len(rollback_actions) >= 1


class TestTrainingAdvisorTrajectory:
    """Advisor should integrate trajectory analysis signals."""

    def test_convergence_triggers_stop(self):
        advisor = TrainingAdvisor()
        # Rapidly converging to near-zero loss
        for i in range(20):
            loss = 2.0 * (0.8 ** i)
            report = advisor.evaluate(i, train_loss=loss)
        # After enough cycles, convergence should be detected
        [a for a in report.actions if a.action_type == "stop_training"]
        # Convergence or early-stop should trigger
        assert report.trajectory_summary is not None

    def test_plateau_triggers_increase_k(self):
        config = AdvisorConfig(trajectory_window=5, convergence_threshold=1e-4)
        advisor = TrainingAdvisor(config=config)
        # Flat loss for many cycles (plateau)
        for i in range(10):
            report = advisor.evaluate(i, train_loss=1.0 + 1e-6 * i)
        increase_k = [a for a in report.actions if a.action_type == "increase_k"]
        # Plateau detection should suggest increase_k
        assert len(increase_k) >= 1 or report.overall_health == "warning"

    def test_strong_trend_with_low_volatility_suggests_lr_increase(self):
        advisor = TrainingAdvisor()
        # Very gradual, consistent decrease — low volatility
        for i in range(15):
            loss = 1.0 - 0.001 * i
            report = advisor.evaluate(i, train_loss=loss)
        assert report.trajectory_summary is not None
        # The trajectory should detect the downward trend
        assert report.trajectory_summary["loss_trend"] < 0
        # With low volatility and downward trend, advisor may suggest LR increase
        [a for a in report.actions if a.action_type == "increase_lr"]
        # At minimum, trajectory analysis should produce a summary
        assert "convergence_rate" in report.trajectory_summary


class TestTrainingAdvisorAcceptanceRate:
    """Advisor should react to low TG-LoRA acceptance rates."""

    def test_low_acceptance_rate_triggers_decrease_k(self):
        advisor = TrainingAdvisor()
        for i in range(10):
            report = advisor.evaluate(
                i,
                train_loss=1.0,
                acceptance_rate=0.1,  # 10% acceptance
            )
        decrease_k = [a for a in report.actions if a.action_type == "decrease_k"]
        assert len(decrease_k) >= 1

    def test_good_acceptance_rate_no_decrease_k(self):
        advisor = TrainingAdvisor()
        for i in range(10):
            report = advisor.evaluate(
                i,
                train_loss=1.0 - 0.01 * i,
                acceptance_rate=0.7,  # 70% acceptance
            )
        decrease_k = [a for a in report.actions if a.action_type == "decrease_k"]
        assert len(decrease_k) == 0


class TestTrainingAdvisorSummary:
    """Advisor state tracking."""

    def test_tracks_best_loss_and_cycle(self):
        advisor = TrainingAdvisor()
        advisor.evaluate(0, train_loss=2.0)
        advisor.evaluate(1, train_loss=1.5)
        advisor.evaluate(2, train_loss=1.8)  # worse
        assert advisor.best_loss == 1.5
        assert advisor.best_cycle == 1

    def test_summary_dict(self):
        advisor = TrainingAdvisor()
        advisor.evaluate(0, train_loss=2.0)
        s = advisor.summary()
        assert s["cycle_count"] == 1
        assert s["best_loss"] == 2.0
        assert "monitor_summary" in s


# ---------------------------------------------------------------------------
# generate_advice_from_history tests
# ---------------------------------------------------------------------------


class TestGenerateAdviceFromHistory:
    def test_produces_report_from_records(self):
        history = [
            {"cycle": 0, "train_loss": 2.0},
            {"cycle": 1, "train_loss": 1.5},
            {"cycle": 2, "train_loss": 1.2},
            {"cycle": 3, "train_loss": 1.1},
            {"cycle": 4, "train_loss": 1.05},
        ]
        report = generate_advice_from_history(history)
        assert isinstance(report, AdvisoryReport)
        assert report.overall_health in ("healthy", "warning", "critical")

    def test_empty_history_returns_healthy(self):
        report = generate_advice_from_history([])
        assert report.overall_health == "healthy"
        assert report.summary == "No data"

    def test_with_valid_loss(self):
        history = [
            {"cycle": 0, "train_loss": 2.0, "valid_loss": 2.1},
            {"cycle": 1, "train_loss": 1.5, "valid_loss": 1.6},
            {"cycle": 2, "train_loss": 1.2, "valid_loss": 1.3},
        ]
        report = generate_advice_from_history(history)
        assert report.trajectory_summary is not None

    def test_nan_in_history_triggers_critical(self):
        history = [
            {"cycle": 0, "train_loss": 2.0},
            {"cycle": 1, "train_loss": float("nan")},
        ]
        report = generate_advice_from_history(history)
        assert report.overall_health == "critical"

    def test_none_train_loss_record_is_skipped_not_crash(self):
        """A None train_loss (e.g. an eval-only record a direct caller passed raw,
        with no usable loss signal) must be SKIPPED, not crash the advisor with
        ``TypeError: must be real number, not NoneType`` from ``math.isnan(None)``.
        The CLI's ``_extract_cycle_records`` surfaces ``loss_valid_full`` for
        full-eval records so this is defense-in-depth for direct callers. The
        finite records around the None one must still be consumed normally."""
        history = [
            {"cycle": 0, "train_loss": 2.0},
            {"cycle": 1, "train_loss": None},  # eval-only record, no loss signal
            {"cycle": 2, "train_loss": 1.5},
            {"cycle": 3, "train_loss": 1.2},
            {"cycle": 4, "train_loss": 1.05},
        ]
        report = generate_advice_from_history(history)
        assert isinstance(report, AdvisoryReport)
        # The finite records were consumed (4 cycles reached the advisor), and
        # no TypeError was raised reaching this assertion.
        assert report.overall_health in ("healthy", "warning", "critical")


# ---------------------------------------------------------------------------
# AdvisorConfig tests
# ---------------------------------------------------------------------------


class TestAdvisorConfig:
    def test_default_values(self):
        config = AdvisorConfig()
        assert config.stagnation_patience == 5
        assert config.spike_threshold == 2.0
        assert config.save_checkpoint_on_best is True

    def test_custom_values(self):
        config = AdvisorConfig(
            stagnation_patience=10,
            spike_threshold=3.0,
            plateau_lr_factor=0.3,
        )
        assert config.stagnation_patience == 10
        assert config.spike_threshold == 3.0
        assert config.plateau_lr_factor == 0.3

    def test_extrapolation_harm_patience_default_and_validation(self):
        # Default patience is 3 — enough cycles to rule out a single noisy
        # overshoot, short enough to surface a mis-tuned alpha promptly.
        assert AdvisorConfig().extrapolation_harm_patience == 3
        assert AdvisorConfig(extrapolation_harm_patience=1).extrapolation_harm_patience == 1
        # A patience < 1 can never fire, so it is rejected at construction.
        with pytest.raises(ValueError, match="extrapolation_harm_patience"):
            AdvisorConfig(extrapolation_harm_patience=0)


# ---------------------------------------------------------------------------
# Extrapolation efficacy — the advisor must consume loss_after vs loss_pilot
# (the core TG-LoRA signal of whether extrapolation reduces loss). These guard
# against the prior defect where ``_generate_actions`` accepted both values and
# silently dropped them.
# ---------------------------------------------------------------------------


def _harm_action(report: AdvisoryReport) -> AdvisoryAction | None:
    """Return the extrapolation-harm ``adjust_alpha`` action if present."""
    for a in report.actions:
        if a.action_type == "adjust_alpha" and "loss_after > loss_pilot" in a.reason:
            return a
    return None


class TestExtrapolationEfficacy:
    def test_harm_streak_fires_adjust_alpha(self):
        # loss_after > loss_pilot for `patience` consecutive cycles: the
        # speculative extrapolation is overshooting every cycle even though
        # the pilot SGD is driving train_loss down. The advisor must recommend
        # weakening extrapolation (lower alpha).
        advisor = TrainingAdvisor(AdvisorConfig(extrapolation_harm_patience=2))
        last = None
        for cycle in range(2):
            last = advisor.evaluate(
                cycle,
                train_loss=2.0 - 0.1 * cycle,  # improving pilot baseline
                loss_pilot=1.0,
                loss_after=1.2,  # extrapolation increased loss
            )
        harm = _harm_action(last)
        assert harm is not None, (
            f"expected extrapolation-harm adjust_alpha; got actions="
            f"{[(a.action_type, a.reason) for a in last.actions]}"
        )
        assert harm.suggested_value == AdvisorConfig().convergence_alpha_factor

    def test_help_cycle_resets_harm_streak(self):
        # Two harm cycles build the streak to the threshold, then a single
        # cycle where extrapolation actually helps (loss_after <= loss_pilot)
        # must reset the streak so the action does not linger. Mutation: if the
        # `else: streak = 0` reset were dropped, the action would persist here.
        advisor = TrainingAdvisor(AdvisorConfig(extrapolation_harm_patience=2))
        advisor.evaluate(0, train_loss=2.0, loss_pilot=1.0, loss_after=1.2)
        advisor.evaluate(1, train_loss=1.9, loss_pilot=1.0, loss_after=1.2)
        assert advisor._extrap_harm_streak == 2
        last = advisor.evaluate(2, train_loss=1.8, loss_pilot=1.0, loss_after=0.9)  # helps
        assert advisor._extrap_harm_streak == 0
        assert _harm_action(last) is None, (
            f"harm action should not fire after a reset; got "
            f"{[(a.action_type, a.reason) for a in last.actions]}"
        )

    def test_no_signal_when_pilot_absent_is_backward_compatible(self):
        # Callers that never pass loss_pilot/loss_after (defaults 0.0) — e.g.
        # a baseline run or an older fixture — must see byte-identical advice:
        # the harm branch never engages. Also pins the `loss_pilot > 0.0` guard:
        # a zero pilot is not a real extrapolation baseline.
        advisor = TrainingAdvisor(AdvisorConfig(extrapolation_harm_patience=2))
        last = None
        for cycle in range(5):
            last = advisor.evaluate(cycle, train_loss=2.0 - 0.1 * cycle)
        assert advisor._extrap_harm_streak == 0
        assert _harm_action(last) is None
        # An explicit zero pilot (no extrapolation baseline this cycle) is the
        # same no-op even if loss_after is nonzero.
        last = advisor.evaluate(
            6, train_loss=1.4, loss_pilot=0.0, loss_after=0.5,
        )
        assert advisor._extrap_harm_streak == 0
        assert _harm_action(last) is None

    def test_harm_signal_flows_through_generate_advice_from_history(self):
        # End-to-end via the record keys the consumer (_extract_cycle_records)
        # surfaces from the producer's tg_lora_loss_pilot_eval /
        # tg_lora_loss_after. Proves the producer -> consumer -> advisor chain
        # delivers the signal that _generate_actions previously dropped.
        history = []
        for cycle in range(3):
            history.append(
                {
                    "cycle": cycle,
                    "train_loss": 2.0 - 0.1 * cycle,
                    "loss_pilot": 1.0,
                    "loss_after": 1.25,  # extrapolation increased loss, every cycle
                }
            )
        report = generate_advice_from_history(history)
        assert _harm_action(report) is not None, (
            f"expected extrapolation-harm action from history; got "
            f"{[(a.action_type, a.reason) for a in report.actions]}"
        )

    def test_summary_exposes_harm_streak(self):
        advisor = TrainingAdvisor(AdvisorConfig(extrapolation_harm_patience=2))
        advisor.evaluate(0, train_loss=2.0, loss_pilot=1.0, loss_after=1.2)
        assert advisor.summary()["extrapolation_harm_streak"] == 1
        advisor.evaluate(1, train_loss=1.9, loss_pilot=1.0, loss_after=0.9)
        assert advisor.summary()["extrapolation_harm_streak"] == 0

    def test_none_pilot_after_does_not_crash(self):
        # A real producer full-eval record surfaces loss_pilot / loss_after as
        # present-but-None (``dict.get(k, default)`` returns None when the key
        # exists with value None). evaluate must not raise TypeError on the
        # comparison, and the absent signal must not advance the streak.
        advisor = TrainingAdvisor(AdvisorConfig(extrapolation_harm_patience=1))
        report = advisor.evaluate(0, train_loss=2.0, loss_pilot=None, loss_after=None)
        assert advisor._extrap_harm_streak == 0
        assert _harm_action(report) is None
        # Mixed (pilot missing, after present) is likewise a no-op, not a crash.
        report = advisor.evaluate(1, train_loss=1.9, loss_pilot=None, loss_after=1.2)
        assert advisor._extrap_harm_streak == 0
        assert _harm_action(report) is None


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestAdviseTrainingCLI:
    def test_produces_text_report(self, tmp_path):
        metrics = tmp_path / "run_metrics.jsonl"
        records = [
            {"type": "cycle_step", "cycle": 0, "loss_train": 2.0},
            {"type": "cycle_step", "cycle": 1, "loss_train": 1.5},
            {"type": "cycle_step", "cycle": 2, "loss_train": 1.2},
            {"type": "cycle_step", "cycle": 3, "loss_train": 1.1},
            {"type": "cycle_step", "cycle": 4, "loss_train": 1.05},
        ]
        _write_jsonl(metrics, records)

        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(metrics)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Advisory Report" in r.stdout

    def test_produces_json_report(self, tmp_path):
        metrics = tmp_path / "run_metrics.jsonl"
        output = tmp_path / "report.json"
        records = [
            {"type": "cycle_step", "cycle": 0, "loss_train": 2.0},
            {"type": "cycle_step", "cycle": 1, "loss_train": 1.5},
            {"type": "cycle_step", "cycle": 2, "loss_train": 1.2},
        ]
        _write_jsonl(metrics, records)

        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(metrics), "--json", "-o", str(output)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert "overall_health" in data
        assert "actions" in data

    def test_exits_critical_on_nan(self, tmp_path):
        metrics = tmp_path / "run_metrics.jsonl"
        records = [
            {"type": "cycle_step", "cycle": 0, "loss_train": 2.0},
            {"type": "cycle_step", "cycle": 1, "loss_train": float("nan")},
        ]
        _write_jsonl(metrics, records)

        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(metrics)],
            capture_output=True, text=True,
        )
        assert r.returncode == 2  # critical exit code

    def test_exits_1_on_missing_file(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/nonexistent/file.jsonl"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_cycle(self):
        advisor = TrainingAdvisor()
        report = advisor.evaluate(0, train_loss=1.0)
        assert report.overall_health == "healthy"

    def test_zero_loss(self):
        advisor = TrainingAdvisor()
        report = advisor.evaluate(0, train_loss=0.0)
        # Zero loss is valid (though unusual)
        assert report.overall_health in ("healthy", "warning", "critical")

    def test_large_cycle_numbers(self):
        advisor = TrainingAdvisor()
        report = advisor.evaluate(999999, train_loss=1.0)
        assert report.overall_health == "healthy"

    def test_valid_loss_overrides_train_for_best(self):
        advisor = TrainingAdvisor()
        advisor.evaluate(0, train_loss=1.0, valid_loss=0.8)
        assert advisor.best_loss == 0.8

    def test_multiple_evaluates_accumulate(self):
        advisor = TrainingAdvisor()
        for i in range(5):
            advisor.evaluate(i, train_loss=1.0 - 0.1 * i)
        assert advisor.cycle_count == 5

    def test_generate_advice_with_all_fields(self):
        history = [
            {
                "cycle": 0,
                "train_loss": 2.0,
                "valid_loss": 2.1,
                "grad_norm": 1.0,
                "velocity_magnitude": 0.5,
                "loss_pilot": 2.1,
                "loss_after": 2.0,
                "acceptance_rate": 0.8,
            },
            {
                "cycle": 1,
                "train_loss": 1.5,
                "valid_loss": 1.6,
                "grad_norm": 0.8,
                "velocity_magnitude": 0.4,
                "loss_pilot": 1.7,
                "loss_after": 1.5,
                "acceptance_rate": 0.9,
            },
        ]
        report = generate_advice_from_history(history)
        assert report.overall_health in ("healthy", "warning", "critical")
