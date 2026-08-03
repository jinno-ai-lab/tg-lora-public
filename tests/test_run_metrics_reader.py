"""Pin the fail-loud contract of ``scripts/run_metrics_reader.py``.

``run_metrics_reader.py`` is the helper ``scripts/run_best_config_eval.sh`` calls
in its FALLBACK run-directory matcher — the path taken when the primary glob
``tg_lora_9b_accel_*${BEST_RUN}*`` does not match a directory name. It scans each
candidate dir's ``run_metrics.jsonl`` for the ``run_header`` record (the schema
``src/utils/run_metrics.py`` writes first per run) and returns its ``run_id``.

It replaced an inline reader whose loop ran behind ``2>/dev/null`` and so
swallowed the *real* cause of a bad ``run_metrics.jsonl``: a corrupt line raised
``JSONDecodeError``, the traceback was eaten, ``run_id`` came back empty, every
comparison missed, and the operator saw only a misleading "Could not find run
directory" — never the corrupt file.

These tests pin the promotion to a loud, machine-distinguishable failure (the
same shape as ``test_best_run_reader.py``): distinct exit codes + a non-empty
stderr that names the cause. Each is invoked as a real subprocess — the exact
path the shell takes — so the contract is verified end-to-end. Reverting the
helper to the old silent-swallow turns ``test_malformed_line_does_not_die_silently``
red (empty stderr under the swallow), which is the mutation the guard exists to
prevent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "run_metrics_reader.py"


def _run(metrics_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), str(metrics_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    """Write a JSONL file from a list of already-serialized line payloads."""
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_HEADER = '{"type": "run_header", "run_id": "tg_lora_9b_accel_42"}'
_CYCLE = '{"type": "cycle_step", "cycle": 1, "loss": 0.5}'


def test_good_jsonl_prints_run_header_run_id(tmp_path: Path) -> None:
    metrics = _write(tmp_path, "run_metrics.jsonl", [_HEADER, _CYCLE])
    result = _run(metrics)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "tg_lora_9b_accel_42"


def test_first_run_header_wins(tmp_path: Path) -> None:
    # A resumed segment does NOT rewrite the header (run_metrics.py:178); the
    # first run_header is authoritative. Pin that scan order.
    header_a = '{"type": "run_header", "run_id": "first_authoritative"}'
    metrics = _write(tmp_path, "run_metrics.jsonl", [header_a, _CYCLE, _HEADER])
    result = _run(metrics)
    assert result.returncode == 0
    assert result.stdout.strip() == "first_authoritative"


def test_malformed_line_does_not_die_silently(tmp_path: Path) -> None:
    """The headline fix: a corrupt line in run_metrics.jsonl must NOT be
    swallowed. Under the old inline reader this raised ``JSONDecodeError``
    behind ``2>/dev/null`` and the shell continued with an empty run_id — the
    operator saw nothing. Now it must exit non-zero with a non-empty stderr
    that names the JSON failure."""
    metrics = _write(tmp_path, "run_metrics.jsonl", [_CYCLE, "garbage{not even json"])
    result = _run(metrics)
    assert result.returncode != 0
    assert result.stderr.strip(), "malformed run_metrics.jsonl died silently (empty stderr)"
    assert "JSON" in result.stderr


def test_trailing_blank_line_is_not_treated_as_corruption(tmp_path: Path) -> None:
    # The producer writes with "wb"/"ab"; a trailing newline (hence an empty
    # final splitline) is normal. Over-strict parsing that failed on it would
    # be a false-positive regression on every well-formed file.
    path = tmp_path / "run_metrics.jsonl"
    path.write_text(_HEADER + "\n\n", encoding="utf-8")
    result = _run(path)
    assert result.returncode == 0
    assert result.stdout.strip() == "tg_lora_9b_accel_42"


def test_non_header_records_are_scanned_past(tmp_path: Path) -> None:
    # cycle_step records precede the header only on an appended/resumed segment
    # where the header already exists earlier — but in general the scanner must
    # walk past non-matching records to find the run_header, not assume line 0.
    metrics = _write(tmp_path, "run_metrics.jsonl", [_CYCLE, _CYCLE, _HEADER])
    result = _run(metrics)
    assert result.returncode == 0
    assert result.stdout.strip() == "tg_lora_9b_accel_42"


def test_no_run_header_fails_loud(tmp_path: Path) -> None:
    metrics = _write(tmp_path, "run_metrics.jsonl", [_CYCLE, _CYCLE])
    result = _run(metrics)
    assert result.returncode == 4
    assert "run_header" in result.stderr


def test_empty_run_id_in_header_fails_loud(tmp_path: Path) -> None:
    header = '{"type": "run_header", "run_id": ""}'
    metrics = _write(tmp_path, "run_metrics.jsonl", [header])
    result = _run(metrics)
    assert result.returncode == 4
    assert "run_id" in result.stderr


def test_missing_file_fails_loud(tmp_path: Path) -> None:
    result = _run(tmp_path / "does_not_exist.jsonl")
    assert result.returncode == 2
    assert result.stderr.strip(), "missing run_metrics.jsonl died silently (empty stderr)"


def test_wrong_arg_count_exits_usage() -> None:
    result = subprocess.run(
        [sys.executable, str(HELPER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 64
    assert "usage" in result.stderr
