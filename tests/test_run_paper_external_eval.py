"""Tests for scripts/run_paper_external_eval.py — TASK-0108 G3 Gate pipeline."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_LM_EVAL_AVAILABLE = importlib.util.find_spec("lm_eval") is not None


SCRIPT = Path("scripts/run_paper_external_eval.py")


def _make_summary(
    *,
    seeds: list[int] | None = None,
    tg_loss: list[float] | None = None,
    bl_loss: list[float] | None = None,
) -> dict:
    seeds = seeds or [42, 43, 44]
    len(seeds)
    tg_loss = tg_loss or [1.80, 1.82, 1.79]
    bl_loss = bl_loss or [1.90, 1.91, 1.89]

    import statistics

    per_seed = []
    for i, s in enumerate(seeds):
        per_seed.append({
            "seed": s,
            "warm_tg_best_valid_loss": tg_loss[i],
            "warm_baseline_best_valid_loss": bl_loss[i],
        })

    def _agg(vals):
        clean = [v for v in vals if v is not None]
        return {"values": clean, "mean": statistics.mean(clean) if clean else None, "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0}

    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "warm_tg_best_valid_loss": _agg(tg_loss),
            "warm_baseline_best_valid_loss": _agg(bl_loss),
        },
    }


def _write_summary(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "aggregate_summary.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


class TestFindBestModelPaths:
    """Best model identification from aggregate_summary."""

    def test_finds_seed_with_lowest_tg_loss(self, tmp_path):
        from scripts.run_paper_external_eval import find_best_model_paths
        summary = _make_summary(
            tg_loss=[1.80, 1.75, 1.79],
            bl_loss=[1.90, 1.88, 1.91],
        )
        summary_path = _write_summary(tmp_path, summary)

        # Create seed dirs with warm adapter checkpoints
        for seed in [42, 43, 44]:
            baseline_dir = tmp_path / f"seed_{seed}" / "coldwarm" / "warm" / "baseline" / "best_model"
            tg_dir = tmp_path / f"seed_{seed}" / "coldwarm" / "warm" / "tg_lora" / "best_model"
            baseline_dir.mkdir(parents=True)
            tg_dir.mkdir(parents=True)
            (baseline_dir / "adapter_model.safetensors").write_text("fake")
            (tg_dir / "adapter_model.safetensors").write_text("fake")

        result = find_best_model_paths(str(summary_path))
        assert result["tg_seed"] == 43
        assert result["baseline_seed"] == 43
        assert result["tg_adapter_path"].endswith("seed_43/coldwarm/warm/tg_lora/best_model")
        assert result["baseline_adapter_path"].endswith("seed_43/coldwarm/warm/baseline/best_model")

    def test_returns_none_when_no_seed_dirs(self, tmp_path):
        from scripts.run_paper_external_eval import find_best_model_paths
        summary = _make_summary()
        summary_path = _write_summary(tmp_path, summary)

        result = find_best_model_paths(str(summary_path))
        assert result["tg_model_path"] is None
        assert result["baseline_model_path"] is None


class TestEvaluateG3:
    """G3 gate evaluation logic."""

    def test_passes_when_all_tasks_within_threshold(self):
        from scripts.run_paper_external_eval import evaluate_g3
        tg_results = {"truthfulqa_mc2": 0.45, "arc_easy": 0.78, "hellaswag": 0.65}
        bl_results = {"truthfulqa_mc2": 0.453, "arc_easy": 0.785, "hellaswag": 0.655}

        result = evaluate_g3(tg_results, bl_results)
        assert result["passed"]

    def test_fails_when_single_task_drops_more_than_3_percent(self):
        from scripts.run_paper_external_eval import evaluate_g3

        # truthfulqa drops by ~6.6% relative
        tg_results = {"truthfulqa_mc2": 0.42, "arc_easy": 0.78, "hellaswag": 0.65}
        bl_results = {"truthfulqa_mc2": 0.45, "arc_easy": 0.785, "hellaswag": 0.655}

        result = evaluate_g3(tg_results, bl_results)
        assert not result["passed"]

    def test_fails_when_aggregate_drop_exceeds_1_percent(self):
        from scripts.run_paper_external_eval import evaluate_g3

        # All tasks drop ~4.4% relative → aggregate ~4.4%
        tg_results = {"truthfulqa_mc2": 0.43, "arc_easy": 0.75, "hellaswag": 0.62}
        bl_results = {"truthfulqa_mc2": 0.45, "arc_easy": 0.785, "hellaswag": 0.655}

        result = evaluate_g3(tg_results, bl_results)
        assert not result["passed"]

    def test_passes_at_exact_boundary(self):
        from scripts.run_paper_external_eval import evaluate_g3

        # All tasks drop exactly 0.99% relative → aggregate ~0.99% < 1%
        tg_results = {"truthfulqa_mc2": 0.9901, "arc_easy": 0.9901, "hellaswag": 0.9901}
        bl_results = {"truthfulqa_mc2": 1.0, "arc_easy": 1.0, "hellaswag": 1.0}

        result = evaluate_g3(tg_results, bl_results)
        assert result["passed"]

    def test_handles_zero_baseline_gracefully(self):
        from scripts.run_paper_external_eval import evaluate_g3
        tg_results = {"truthfulqa_mc2": 0.45, "arc_easy": 0.78, "hellaswag": 0.65}
        bl_results = {"truthfulqa_mc2": 0.0, "arc_easy": 0.785, "hellaswag": 0.655}

        result = evaluate_g3(tg_results, bl_results)
        # Zero baseline → can't compute relative drop for that task → skipped
        assert "truthfulqa_mc2" not in result.get("task_drops", {})

    def test_handles_missing_task_in_tg(self):
        from scripts.run_paper_external_eval import evaluate_g3
        tg_results = {"truthfulqa_mc2": 0.45, "arc_easy": 0.78}
        bl_results = {"truthfulqa_mc2": 0.453, "arc_easy": 0.785, "hellaswag": 0.655}

        result = evaluate_g3(tg_results, bl_results)
        # hellaswag missing from TG → not compared AND now surfaced as dropped
        # (loud), rather than vanishing silently from the report.
        assert "hellaswag" not in result.get("task_drops", {})
        assert "hellaswag" in result["dropped_tasks"]
        assert result["incomplete"] is True


class TestBuildExternalEvalResults:
    """external_eval_results.json structure validation."""

    def test_produces_valid_json_structure(self):
        from scripts.run_paper_external_eval import build_external_eval_results
        tg_results = {"truthfulqa_mc2": 0.45, "arc_easy": 0.78, "hellaswag": 0.65}
        bl_results = {"truthfulqa_mc2": 0.453, "arc_easy": 0.785, "hellaswag": 0.655}

        result = build_external_eval_results(
            tg_results=tg_results,
            baseline_results=bl_results,
            base_model="Qwen/Qwen3.5-9B",
            tg_adapter_path="/path/to/tg",
            baseline_adapter_path="/path/to/bl",
            tasks=["truthfulqa_mc2", "arc_easy", "hellaswag"],
        )

        assert "generated_at" in result
        assert result["tasks"] == ["truthfulqa_mc2", "arc_easy", "hellaswag"]
        assert result["models"]["tg"]["base_model"] == "Qwen/Qwen3.5-9B"
        assert result["models"]["tg"]["model_path"] == "/path/to/tg"
        assert result["models"]["tg"]["adapter_path"] == "/path/to/tg"
        assert result["models"]["tg"]["results"] == tg_results
        assert result["models"]["baseline"]["model_path"] == "/path/to/bl"
        assert result["models"]["baseline"]["adapter_path"] == "/path/to/bl"
        assert result["models"]["baseline"]["results"] == bl_results
        assert "comparison" in result
        assert "aggregate_relative_drop" in result["comparison"]
        assert "task_relative_drops" in result["comparison"]
        assert "g3_passed" in result["comparison"]


class TestSilentDropPrevention:
    """Regression guard for the silent task-drop bug class.

    Feedback directive: a requested eval task that fails to score must FAIL
    LOUD, not vanish from the report. Root cause: ``_run_lm_eval`` only
    recognized acc/acc_norm metrics, so a task whose lm-eval primary metric is
    neither (``truthfulqa_mc2`` -> ``mc2``, ``gsm8k`` -> ``exact_match``)
    silently produced no score, dropped out of the G3 comparison, and -- because
    ``G3.3`` checked the REQUESTED task list -- the gate's own "required tasks
    present" guard still passed. These tests pin both the source fix (metric
    recognition + drop surfacing) and the guard fix (check scored, not
    requested).
    """

    def test_extract_primary_metric_recognizes_truthfulqa_mc2(self):
        from scripts.run_paper_external_eval import _extract_primary_metric
        # truthfulqa_mc2 reports `mc2`, which the old acc-only list dropped.
        assert _extract_primary_metric("truthfulqa_mc2", {"mc2,none": 0.4321}) == 0.4321

    def test_extract_primary_metric_recognizes_gsm8k_exact_match(self):
        from scripts.run_paper_external_eval import _extract_primary_metric
        assert _extract_primary_metric("gsm8k", {"exact_match,none": 0.12}) == 0.12

    def test_extract_primary_metric_acc_norm_for_unmapped_task(self):
        from scripts.run_paper_external_eval import _extract_primary_metric
        assert _extract_primary_metric("some_custom_task", {"acc_norm,none": 0.7}) == 0.7

    def test_extract_primary_metric_returns_none_when_unrecognized(self):
        from scripts.run_paper_external_eval import _extract_primary_metric
        # No recognized key -> None (NOT silently omitted) so the caller records
        # the task as dropped instead of letting it vanish from the report.
        assert _extract_primary_metric("weird_task", {"foo,none": 0.5}) is None

    # Analog of the requested "extension-coverage" test, translated to this
    # repo's domain: every DEFAULT/REQUIRED G3 task must resolve to a non-None
    # primary metric against its real lm-eval result shape, so the next
    # unmapped default task fails LOUDLY at test time rather than vanishing
    # silently from a G3 report. truthfulqa_mc2 is the load-bearing case (its
    # real metric ``mc2`` is not acc/acc_norm -- drop its map entry and this RED).
    @pytest.mark.parametrize("task", ["truthfulqa_mc2", "arc_easy", "hellaswag"])
    def test_every_default_task_resolves_nonzero_metric(self, task):
        from scripts.run_paper_external_eval import _extract_primary_metric
        fixtures = {
            "truthfulqa_mc2": {"mc2,none": 0.4321, "mc2_stderr,none": 0.012},
            "arc_easy": {"acc,none": 0.6012, "acc_norm,none": 0.6503,
                         "acc_stderr,none": 0.004, "acc_norm_stderr,none": 0.004},
            "hellaswag": {"acc,none": 0.4010, "acc_norm,none": 0.5520,
                          "acc_stderr,none": 0.005, "acc_norm_stderr,none": 0.005},
        }
        score = _extract_primary_metric(task, fixtures[task])
        assert score is not None, f"{task} produced no recognized metric (would silently drop)"
        assert score > 0

    def test_evaluate_g3_surfaces_dropped_requested_task(self):
        from scripts.run_paper_external_eval import evaluate_g3
        # truthfulqa_mc2 vanished from BOTH sides (no recognized metric) -- the
        # truly silent case, undetectable without the requested-task list.
        tg = {"arc_easy": 0.78, "hellaswag": 0.65}
        bl = {"arc_easy": 0.785, "hellaswag": 0.655}
        result = evaluate_g3(
            tg, bl,
            requested_tasks=["truthfulqa_mc2", "arc_easy", "hellaswag"],
        )
        assert result["incomplete"] is True
        assert "truthfulqa_mc2" in result["dropped_tasks"]
        assert set(result["compared_tasks"]) == {"arc_easy", "hellaswag"}
        # The threshold verdict is still computed over the survivors ...
        assert result["passed"] is True
        # ... but the drop is named in the detail, not buried.
        assert "truthfulqa_mc2" in result["detail"]

    def test_evaluate_g3_no_drop_when_all_requested_scored(self):
        from scripts.run_paper_external_eval import evaluate_g3
        tg = {"truthfulqa_mc2": 0.45, "arc_easy": 0.78, "hellaswag": 0.65}
        bl = {"truthfulqa_mc2": 0.453, "arc_easy": 0.785, "hellaswag": 0.655}
        result = evaluate_g3(tg, bl, requested_tasks=list(tg))
        assert result["incomplete"] is False
        assert result["dropped_tasks"] == []

    def test_build_external_eval_results_records_scored_vs_dropped(self):
        from scripts.run_paper_external_eval import build_external_eval_results
        # Request 3, only 2 score -> the report must distinguish request from
        # reality so a drop can't hide behind the top-level `tasks` field.
        result = build_external_eval_results(
            tg_results={"arc_easy": 0.78, "hellaswag": 0.65},
            baseline_results={"arc_easy": 0.785, "hellaswag": 0.655},
            base_model="M",
            tg_adapter_path="/tg",
            baseline_adapter_path="/bl",
            tasks=["truthfulqa_mc2", "arc_easy", "hellaswag"],
        )
        cmp = result["comparison"]
        assert result["tasks"] == ["truthfulqa_mc2", "arc_easy", "hellaswag"]
        assert cmp["incomplete"] is True
        assert cmp["dropped_tasks"] == ["truthfulqa_mc2"]
        assert set(cmp["compared_tasks"]) == {"arc_easy", "hellaswag"}


@pytest.mark.skipif(not _LM_EVAL_AVAILABLE, reason="lm_eval not installed")
class TestRunLmEval:
    @patch("lm_eval.simple_evaluate")
    def test_uses_peft_model_args(self, mock_simple_evaluate):
        from scripts.run_paper_external_eval import _run_lm_eval

        mock_simple_evaluate.return_value = {
            "results": {
                "truthfulqa_mc2": {"acc,none": 0.5},
            }
        }

        results = _run_lm_eval(
            base_model="Qwen/Qwen3.5-9B",
            adapter_path="/tmp/adapter",
            tasks=["truthfulqa_mc2"],
            batch_size="8",
        )

        assert results == {"truthfulqa_mc2": 0.5}
        kwargs = mock_simple_evaluate.call_args.kwargs
        assert kwargs["model"] == "hf"
        assert kwargs["model_args"] == (
            "pretrained=Qwen/Qwen3.5-9B,"
            "peft=/tmp/adapter,"
            "dtype=float16,"
            "load_in_4bit=True"
        )

    @patch("lm_eval.simple_evaluate")
    @patch("scripts.run_paper_external_eval._build_preloaded_4bit_hflm")
    def test_retries_with_preloaded_4bit_model_on_typeerror(
        self,
        mock_build_preloaded_hflm,
        mock_simple_evaluate,
    ):
        from scripts.run_paper_external_eval import _run_lm_eval

        mock_build_preloaded_hflm.return_value = object()
        mock_simple_evaluate.side_effect = [
            TypeError("Qwen3_5ForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'"),
            {
                "results": {
                    "truthfulqa_mc2": {"acc,none": 0.51},
                }
            },
        ]

        results = _run_lm_eval(
            base_model="Qwen/Qwen3.5-9B",
            adapter_path="/tmp/adapter",
            tasks=["truthfulqa_mc2"],
            batch_size="8",
        )

        assert results == {"truthfulqa_mc2": 0.51}
        assert mock_simple_evaluate.call_count == 2
        first_call = mock_simple_evaluate.call_args_list[0].kwargs
        second_call = mock_simple_evaluate.call_args_list[1].kwargs
        mock_build_preloaded_hflm.assert_called_once_with(
            base_model="Qwen/Qwen3.5-9B",
            adapter_path="/tmp/adapter",
            batch_size="8",
        )
        assert first_call["model_args"] == (
            "pretrained=Qwen/Qwen3.5-9B,"
            "peft=/tmp/adapter,"
            "dtype=float16,"
            "load_in_4bit=True"
        )
        assert second_call["model"] is mock_build_preloaded_hflm.return_value


class TestCLIEndToEnd:
    """CLI smoke tests for run_paper_external_eval.py."""

    def test_analysis_mode_with_mock_results(self, tmp_path):
        """Run in analysis mode with pre-existing results, no GPU needed."""
        summary = _make_summary()
        _write_summary(tmp_path, summary)

        # Pre-create external eval results
        eval_results = {
            "generated_at": "2026-05-25T00:00:00+00:00",
            "tasks": ["truthfulqa_mc2", "arc_easy", "hellaswag"],
            "models": {
                "tg": {"model_path": "/fake/tg", "results": {"truthfulqa_mc2": 0.45, "arc_easy": 0.78, "hellaswag": 0.65}},
                "baseline": {"model_path": "/fake/bl", "results": {"truthfulqa_mc2": 0.453, "arc_easy": 0.785, "hellaswag": 0.655}},
            },
            "comparison": {"aggregate_relative_drop": 0.006, "g3_passed": True},
        }
        eval_path = tmp_path / "external_eval_results.json"
        eval_path.write_text(json.dumps(eval_results))

        output_path = tmp_path / "output" / "external_eval_results.json"

        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--analysis-mode",
             "--external-eval", str(eval_path),
             "--output", str(output_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert output_path.exists()

    def test_exit_1_on_missing_file(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--analysis-mode",
             "--external-eval", "/nonexistent/eval.json"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0


class TestG3GateIntegration:
    """Integration: _check_g3 in evaluate_paper_gates reads external_eval_results.json."""

    def test_g3_passes_with_valid_external_eval(self, tmp_path):
        from scripts.evaluate_paper_gates import _check_g3
        summary = _make_summary()

        external_eval = {
            "generated_at": "2026-05-25T00:00:00+00:00",
            "tasks": ["truthfulqa_mc2", "arc_easy", "hellaswag"],
            "comparison": {
                "aggregate_relative_drop": 0.005,
                "task_relative_drops": {
                    "truthfulqa_mc2": 0.006,
                    "arc_easy": 0.004,
                    "hellaswag": 0.005,
                },
                "g3_passed": True,
            },
        }
        eval_path = tmp_path / "external_eval_results.json"
        eval_path.write_text(json.dumps(external_eval))

        result = _check_g3(summary, external_eval_path=str(eval_path))
        assert result["passed"]

    def test_g3_fails_when_drop_too_large(self, tmp_path):
        from scripts.evaluate_paper_gates import _check_g3
        summary = _make_summary()

        external_eval = {
            "generated_at": "2026-05-25T00:00:00+00:00",
            "tasks": ["truthfulqa_mc2", "arc_easy", "hellaswag"],
            "comparison": {
                "aggregate_relative_drop": 0.05,
                "task_relative_drops": {
                    "truthfulqa_mc2": 0.07,
                    "arc_easy": 0.04,
                    "hellaswag": 0.04,
                },
                "g3_passed": False,
            },
        }
        eval_path = tmp_path / "external_eval_results.json"
        eval_path.write_text(json.dumps(external_eval))

        result = _check_g3(summary, external_eval_path=str(eval_path))
        assert not result["passed"]

    def test_g3_no_external_eval_is_informational(self):
        from scripts.evaluate_paper_gates import _check_g3
        summary = _make_summary()
        result = _check_g3(summary)
        assert not result["passed"]
        assert any("external eval" in c["detail"].lower() for c in result["checks"])

    def test_g3_auto_discover_external_eval(self, tmp_path):
        """Auto-discover external_eval_results.json next to aggregate_summary."""
        from scripts.evaluate_paper_gates import _check_g3
        summary = _make_summary()

        eval_data = {
            "generated_at": "2026-05-25T00:00:00+00:00",
            "tasks": ["truthfulqa_mc2", "arc_easy", "hellaswag"],
            "comparison": {
                "aggregate_relative_drop": 0.003,
                "task_relative_drops": {
                    "truthfulqa_mc2": 0.004,
                    "arc_easy": 0.002,
                    "hellaswag": 0.003,
                },
                "g3_passed": True,
            },
        }
        eval_path = tmp_path / "external_eval_results.json"
        eval_path.write_text(json.dumps(eval_data))

        # Call without explicit path — should auto-discover
        result = _check_g3(summary, external_eval_path=str(eval_path))
        assert result["passed"]

    def test_g3_fails_when_required_task_silently_dropped(self, tmp_path):
        """G3.3 must check SCORED tasks, not the REQUESTED list.

        Regression for the silent-drop bug class: a task that failed to score
        (absent from ``task_relative_drops``) used to fool G3.3 into PASSING,
        because the guard read the top-level ``tasks`` (request) field. With the
        fix, the gate fails and names the dropped required task. (Mutation
        check: revert G3.3 to checking ``tasks`` and this turns RED.)
        """
        from scripts.evaluate_paper_gates import _check_g3
        summary = _make_summary()
        external_eval = {
            "generated_at": "2026-05-25T00:00:00+00:00",
            "tasks": ["truthfulqa_mc2", "arc_easy", "hellaswag"],  # requested
            "comparison": {
                "aggregate_relative_drop": 0.004,
                "task_relative_drops": {
                    # truthfulqa_mc2 silently vanished -- only 2 of 3 compared
                    "arc_easy": 0.004,
                    "hellaswag": 0.005,
                },
                "g3_passed": True,
            },
        }
        eval_path = tmp_path / "external_eval_results.json"
        eval_path.write_text(json.dumps(external_eval))

        result = _check_g3(summary, external_eval_path=str(eval_path))
        # The gate must NOT pass while a required task is silently missing.
        assert not result["passed"]
        g33 = next(
            c for c in result["checks"]
            if c["check"] == "G3.3_required_tasks_present"
        )
        assert g33["pass"] is False
        assert "truthfulqa_mc2" in g33["detail"]
