"""Query API for RunMetrics JSONL logs — TASK-0060."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


def parse_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    text = path.read_text()
    if not text.strip():
        return records
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(orjson.loads(line))
        except orjson.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {i}: {e}") from e
    return records


def get_footer(path: str | Path) -> dict[str, Any]:
    """Return the footer metadata dict from a JSONL run file."""
    for rec in reversed(parse_jsonl(path)):
        if rec.get("type") == "run_footer":
            return rec
    raise ValueError(f"No run_footer record in {path}")


def get_cycle_history(
    path: str | Path,
    cycle: int | None = None,
) -> list[dict[str, Any]]:
    """Return step records, optionally filtered by cycle number."""
    steps = [r for r in parse_jsonl(path) if r.get("type") == "step"]
    if cycle is not None:
        steps = [s for s in steps if s.get("cycle") == cycle]
    return steps


def get_best_loss(path: str | Path) -> dict[str, Any]:
    """Return best_valid_loss and best_valid_step from the footer."""
    footer = get_footer(path)
    return {
        "best_valid_loss": footer["best_valid_loss"],
        "best_valid_step": footer["best_valid_step"],
    }


def get_best_perplexity(path: str | Path) -> float | None:
    """Return perplexity from the footer, or None if not recorded."""
    return get_footer(path).get("perplexity")


def list_runs(run_dir: str | Path) -> list[dict[str, Any]]:
    """List summaries of all runs found under *run_dir*.

    Scans immediate subdirectories and the directory itself for
    ``run_metrics.jsonl`` files, returning one summary dict per run.
    """
    run_dir = Path(run_dir)
    summaries: list[dict[str, Any]] = []

    candidates = [run_dir]
    if run_dir.is_dir():
        candidates.extend(d for d in run_dir.iterdir() if d.is_dir())

    for candidate in candidates:
        jsonl = candidate / "run_metrics.jsonl"
        if not jsonl.exists():
            continue
        try:
            records = parse_jsonl(jsonl)
        except (ValueError, orjson.JSONDecodeError):
            continue

        # First header carries the run's original identity (run_id / started_at);
        # ``write_header`` is a no-op on an appended segment, so the header is
        # singular and first == last.
        header = next((r for r in records if r.get("type") == "run_header"), None)
        # LAST footer wins: a run resumed into the same dir (resume-after-
        # completion) appends a second ``run_footer``, and the latest is the run's
        # most recent completion state. Matches ``get_footer`` /
        # ``compare_runs.load_run`` / ``extract_best_valid_loss`` (all
        # last-footer); a first-footer read would feed ``find_best_run`` stale
        # ``best_valid_loss`` on the run-selection sweep.
        footer = next((r for r in reversed(records) if r.get("type") == "run_footer"), None)

        if header is None:
            continue

        summary: dict[str, Any] = {
            "run_id": header.get("run_id"),
            "mode": header.get("mode"),
            "model_name": header.get("model_name"),
            "started_at": header.get("started_at"),
            "accel_instability_lr_decay": header.get("accel_instability_lr_decay"),
            "accel_convergence_lr_boost": header.get("accel_convergence_lr_boost"),
            "comparison_reference_loss": (header.get("comparison_reference") or {}).get("loss"),
            "comparison_reference_kind": (header.get("comparison_reference") or {}).get("kind"),
            "_jsonl_path": str(jsonl),
        }
        if footer is not None:
            summary["best_valid_loss"] = footer.get("best_valid_loss")
            summary["best_valid_step"] = footer.get("best_valid_step")
            summary["final_train_loss"] = footer.get("final_train_loss")
            summary["perplexity"] = footer.get("perplexity")
            summary["total_wall_seconds"] = footer.get("total_wall_seconds")

        summaries.append(summary)

    return summaries
