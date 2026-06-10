"""Tests for run_query module — TASK-0060 query API."""

from pathlib import Path

import pytest

from src.utils.run_metrics import RunMetrics
from src.utils.run_query import (
    get_best_loss,
    get_best_perplexity,
    get_cycle_history,
    get_footer,
    list_runs,
    parse_jsonl,
)


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

    class logging:
        pass

    class eval:
        pass


def _write_run(
    run_dir: Path,
    mode: str = "baseline",
    run_id: str | None = None,
    steps: list[dict] | None = None,
    footer_kwargs: dict | None = None,
) -> Path:
    """Write a complete JSONL run file and return its path."""
    cfg = FakeCfg()
    m = RunMetrics(run_dir, mode=mode, run_id=run_id)
    m.write_header(cfg, budget_type="backward_passes", budget_value=10)
    for s in steps or []:
        m.record_step(**s)
    fk = footer_kwargs or {}
    m.write_footer(
        best_valid_loss=fk.get("best_valid_loss", 1.0),
        best_valid_step=fk.get("best_valid_step", 1),
        final_train_loss=fk.get("final_train_loss", 1.0),
        perplexity=fk.get("perplexity"),
        tg_lora_summary=fk.get("tg_lora_summary"),
    )
    m.close()
    return run_dir / "run_metrics.jsonl"


# --- parse_jsonl ---


def test_parse_jsonl_reads_all_records(tmp_path):
    p = _write_run(
        tmp_path,
        steps=[
            {
                "step": 1,
                "loss_train": 3.0,
                "backward_passes": 1,
                "total_backward_passes": 1,
            },
            {
                "step": 2,
                "loss_train": 2.5,
                "backward_passes": 1,
                "total_backward_passes": 2,
            },
        ],
    )
    records = parse_jsonl(p)
    assert len(records) == 4
    assert records[0]["type"] == "run_header"
    assert records[1]["type"] == "step"
    assert records[2]["type"] == "step"
    assert records[3]["type"] == "run_footer"


def test_parse_jsonl_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert parse_jsonl(p) == []


def test_parse_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_jsonl(tmp_path / "nonexistent.jsonl")


def test_parse_jsonl_corrupt_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"type": "run_header"}\nCORRUPT\n{"type": "run_footer"}\n')
    with pytest.raises(ValueError, match="line 2"):
        parse_jsonl(p)


# --- get_footer ---


def test_get_footer_returns_metadata(tmp_path):
    _write_run(
        tmp_path,
        footer_kwargs={
            "best_valid_loss": 1.5,
            "best_valid_step": 5,
            "final_train_loss": 1.2,
            "perplexity": 4.48,
        },
    )
    footer = get_footer(tmp_path / "run_metrics.jsonl")
    assert footer["type"] == "run_footer"
    assert footer["best_valid_loss"] == 1.5
    assert footer["best_valid_step"] == 5
    assert footer["perplexity"] == 4.48


def test_get_footer_no_footer(tmp_path):
    """File with header + steps but no footer should raise ValueError."""
    p = tmp_path / "run_metrics.jsonl"
    p.write_text('{"type": "run_header"}\n{"type": "step"}\n')
    with pytest.raises(ValueError, match="No run_footer"):
        get_footer(p)


# --- get_cycle_history ---


def test_get_cycle_history_returns_steps(tmp_path):
    _write_run(
        tmp_path,
        mode="tg_lora",
        run_id="tg_test",
        steps=[
            {
                "step": 1,
                "cycle": 0,
                "loss_train": 3.0,
                "loss_valid": 3.2,
                "backward_passes": 1,
                "total_backward_passes": 1,
                "tg_lora_accepted": True,
            },
            {
                "step": 2,
                "cycle": 0,
                "loss_train": 2.8,
                "loss_valid": 2.9,
                "backward_passes": 1,
                "total_backward_passes": 2,
                "tg_lora_accepted": False,
            },
            {
                "step": 3,
                "cycle": 1,
                "loss_train": 2.5,
                "loss_valid": 2.6,
                "backward_passes": 1,
                "total_backward_passes": 3,
                "tg_lora_accepted": True,
            },
        ],
    )
    history = get_cycle_history(tmp_path / "run_metrics.jsonl")
    assert len(history) == 3
    assert history[0]["cycle"] == 0
    assert history[1]["cycle"] == 0
    assert history[2]["cycle"] == 1
    assert history[0]["tg_lora_accepted"] is True


def test_get_cycle_history_filter_by_cycle(tmp_path):
    _write_run(
        tmp_path,
        steps=[
            {
                "step": 1,
                "cycle": 0,
                "loss_train": 3.0,
                "backward_passes": 1,
                "total_backward_passes": 1,
            },
            {
                "step": 2,
                "cycle": 1,
                "loss_train": 2.5,
                "backward_passes": 1,
                "total_backward_passes": 2,
            },
            {
                "step": 3,
                "cycle": 1,
                "loss_train": 2.0,
                "backward_passes": 1,
                "total_backward_passes": 3,
            },
        ],
    )
    history = get_cycle_history(tmp_path / "run_metrics.jsonl", cycle=1)
    assert len(history) == 2
    assert all(h["cycle"] == 1 for h in history)


def test_get_cycle_history_empty(tmp_path):
    _write_run(tmp_path)
    history = get_cycle_history(tmp_path / "run_metrics.jsonl")
    assert history == []


# --- get_best_loss ---


def test_get_best_loss(tmp_path):
    _write_run(
        tmp_path,
        footer_kwargs={
            "best_valid_loss": 1.23,
            "best_valid_step": 7,
            "final_train_loss": 1.0,
        },
    )
    result = get_best_loss(tmp_path / "run_metrics.jsonl")
    assert result["best_valid_loss"] == 1.23
    assert result["best_valid_step"] == 7


# --- get_best_perplexity ---


def test_get_best_perplexity(tmp_path):
    _write_run(
        tmp_path,
        footer_kwargs={
            "best_valid_loss": 1.5,
            "best_valid_step": 3,
            "final_train_loss": 1.2,
            "perplexity": 4.48,
        },
    )
    result = get_best_perplexity(tmp_path / "run_metrics.jsonl")
    assert result == 4.48


def test_get_best_perplexity_none(tmp_path):
    _write_run(
        tmp_path,
        footer_kwargs={
            "best_valid_loss": 1.5,
            "best_valid_step": 3,
            "final_train_loss": 1.2,
        },
    )
    assert get_best_perplexity(tmp_path / "run_metrics.jsonl") is None


# --- list_runs ---


def test_list_runs_single(tmp_path):
    _write_run(
        tmp_path / "run1",
        run_id="baseline_001",
        mode="baseline",
        steps=[
            {
                "step": 1,
                "loss_train": 3.0,
                "backward_passes": 1,
                "total_backward_passes": 1,
            },
        ],
        footer_kwargs={
            "best_valid_loss": 2.5,
            "best_valid_step": 1,
            "final_train_loss": 2.8,
        },
    )
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == "baseline_001"
    assert r["mode"] == "baseline"
    assert r["best_valid_loss"] == 2.5


def test_list_runs_multiple(tmp_path):
    for name, mode, bv in [("r1", "baseline", 2.5), ("r2", "tg_lora", 1.8)]:
        d = tmp_path / name
        _write_run(
            d,
            run_id=name,
            mode=mode,
            steps=[
                {
                    "step": 1,
                    "loss_train": 3.0,
                    "backward_passes": 1,
                    "total_backward_passes": 1,
                },
            ],
            footer_kwargs={
                "best_valid_loss": bv,
                "best_valid_step": 1,
                "final_train_loss": bv + 0.3,
            },
        )
    runs = list_runs(tmp_path)
    assert len(runs) == 2
    ids = {r["run_id"] for r in runs}
    assert ids == {"r1", "r2"}


def test_list_runs_empty_dir(tmp_path):
    assert list_runs(tmp_path) == []


def test_list_runs_skips_non_jsonl_dirs(tmp_path):
    (tmp_path / "not_a_run").mkdir()
    (tmp_path / "not_a_run" / "stuff.txt").write_text("hello")
    assert list_runs(tmp_path) == []


def test_list_run_includes_perplexity(tmp_path):
    _write_run(
        tmp_path / "run1",
        run_id="ppl_run",
        footer_kwargs={
            "best_valid_loss": 1.5,
            "best_valid_step": 1,
            "final_train_loss": 1.2,
            "perplexity": 4.48,
        },
    )
    runs = list_runs(tmp_path)
    assert runs[0]["perplexity"] == 4.48


def test_parse_jsonl_blank_lines_skipped(tmp_path):
    """Blank lines in JSONL should be silently skipped."""
    p = tmp_path / "blanks.jsonl"
    p.write_text(
        '{"type": "run_header"}\n\n{"type": "step"}\n  \n{"type": "run_footer"}\n'
    )
    records = parse_jsonl(p)
    assert len(records) == 3
    assert records[0]["type"] == "run_header"
    assert records[1]["type"] == "step"
    assert records[2]["type"] == "run_footer"


def test_list_runs_skips_corrupt_jsonl(tmp_path):
    """list_runs should silently skip runs with corrupt JSONL."""
    d = tmp_path / "corrupt_run"
    d.mkdir()
    (d / "run_metrics.jsonl").write_text('{"type": "run_header"}\nCORRUPT\n')
    runs = list_runs(tmp_path)
    assert runs == []


def test_list_runs_skips_no_header(tmp_path):
    """list_runs should skip JSONL files that have no run_header."""
    d = tmp_path / "no_header_run"
    d.mkdir()
    (d / "run_metrics.jsonl").write_text(
        '{"type": "step", "step": 1}\n{"type": "run_footer"}\n'
    )
    runs = list_runs(tmp_path)
    assert runs == []
