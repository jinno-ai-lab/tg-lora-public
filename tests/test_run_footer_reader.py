"""Pin the fail-loud contract of ``scripts/run_footer_reader.py``.

``run_footer_reader.py`` is the helper ``scripts/run_kstep_rollback_test.sh``
calls to print each run's end-of-run summary (best valid loss / final train
loss / perplexity / wall seconds) from the ``run_footer`` record the producer
appends last per run.

It replaced an inline reader whose loop ran behind ``2>/dev/null`` and an
``|| echo "{name}  (no footer yet)"`` and so swallowed the *real* cause of a
bad ``run_metrics.jsonl``: a corrupt line raised ``JSONDecodeError``, the
traceback was eaten, and the shell printed the SAME "(no footer yet)" it prints
for a run that is simply still in progress — masking a *corrupt* file as a
*normal transient*. An operator re-checking a finished sweep waited on a run
that had already died instead of being pointed at the corrupt file.

These tests pin the promotion to a loud, machine-distinguishable failure (the
same shape as ``test_run_metrics_reader.py`` / ``test_best_run_reader.py``):
distinct exit codes + a non-empty stderr that names the cause. Crucially,
corruption (exit 2 + stderr) is distinguished from a genuine NO-FOOTER-YET
(exit 3, NO stderr) — the two cases the old swallow collapsed into one silent
default. Each is invoked as a real subprocess — the exact path the shell takes
— so the contract is verified end-to-end. Reverting the helper to the old
silent-swallow turns ``test_malformed_line_does_not_die_silently`` red (empty
stderr under the swallow) and ``test_no_footer_record_is_transient_not_corrupt``
red (the old code returned "(no footer yet)" for BOTH, indistinguishable).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "run_footer_reader.py"


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


_STEP = '{"type": "step", "loss_train": 0.5, "total_backward_passes": 10}'
_FOOTER = (
    '{"type": "run_footer", "best_valid_loss": 1.0565, '
    '"final_train_loss": 0.98, "perplexity": 2.87, "total_wall_seconds": 123.4}'
)


def test_good_jsonl_prints_footer_summary(tmp_path: Path) -> None:
    metrics = _write(tmp_path, "run_metrics.jsonl", [_STEP, _STEP, _FOOTER])
    result = _run(metrics)
    assert result.returncode == 0
    assert result.stderr == ""
    # Shell prepends the run name; the reader prints the field line only.
    assert result.stdout.strip() == (
        "best_valid=1.0565  final_train=0.98  ppl=2.87  wall=123s"
    )


def test_missing_footer_fields_default_to_NA(tmp_path: Path) -> None:
    # A footer written before eval completed may omit best_valid_loss / ppl; the
    # producer defaults them to "N/A" in its own summary, and so does this reader
    # (mirroring the inline reader it replaced) rather than crashing.
    footer = '{"type": "run_footer", "total_wall_seconds": 50.0}'
    metrics = _write(tmp_path, "run_metrics.jsonl", [footer])
    result = _run(metrics)
    assert result.returncode == 0
    assert result.stdout.strip() == (
        "best_valid=N/A  final_train=N/A  ppl=N/A  wall=50s"
    )


def test_malformed_line_does_not_die_silently(tmp_path: Path) -> None:
    """The headline fix: a corrupt line in run_metrics.jsonl must NOT be
    swallowed. Under the old inline reader this raised ``JSONDecodeError``
    behind ``2>/dev/null`` and the shell printed "(no footer yet)" — identical
    to a run still in progress. Now it must exit non-zero with a non-empty
    stderr that names the JSON failure."""
    metrics = _write(tmp_path, "run_metrics.jsonl", [_STEP, "garbage{not even json"])
    result = _run(metrics)
    assert result.returncode != 0
    assert result.stderr.strip(), "malformed run_metrics.jsonl died silently (empty stderr)"
    assert "JSON" in result.stderr


def test_no_footer_record_is_transient_not_corrupt(tmp_path: Path) -> None:
    """The core distinction the old swallow erased: a valid JSONL with NO
    ``run_footer`` record is a run still in progress (the footer is appended
    LAST) — a normal transient, NOT corruption. It must exit 3 with NO stderr
    so the shell prints the normal "(no footer yet)" instead of a warning,
    while a corrupt file (exit 2 + stderr) is warned. Pinning both codes keeps
    them distinguishable."""
    metrics = _write(tmp_path, "run_metrics.jsonl", [_STEP, _STEP])
    result = _run(metrics)
    assert result.returncode == 3
    assert result.stderr == "", "no-footer transient must be silent, not warned"


def test_trailing_blank_line_is_not_treated_as_corruption(tmp_path: Path) -> None:
    # The producer writes with "wb"/"ab"; a trailing newline (hence an empty
    # final splitline) is normal. Over-strict parsing that failed on it would be
    # a false-positive regression on every well-formed file.
    path = tmp_path / "run_metrics.jsonl"
    path.write_text(_FOOTER + "\n\n", encoding="utf-8")
    result = _run(path)
    assert result.returncode == 0
    assert "wall=" in result.stdout


def test_non_footer_records_are_scanned_past(tmp_path: Path) -> None:
    # The footer is not guaranteed to be the only record type; the scanner must
    # walk past non-matching records to find run_footer, not assume line 0.
    metrics = _write(tmp_path, "run_metrics.jsonl", [_STEP, _FOOTER, _STEP])
    result = _run(metrics)
    assert result.returncode == 0
    assert "best_valid=1.0565" in result.stdout


def test_non_numeric_wall_does_not_crash(tmp_path: Path) -> None:
    # A hand-edited or malformed footer with a non-numeric total_wall_seconds
    # must not crash the whole summary on one bad field; it falls back to the
    # raw value rather than raising TypeError on ``:.0f``.
    footer = '{"type": "run_footer", "best_valid_loss": 1.0, "total_wall_seconds": null}'
    metrics = _write(tmp_path, "run_metrics.jsonl", [footer])
    result = _run(metrics)
    assert result.returncode == 0
    assert "wall=None" in result.stdout


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
