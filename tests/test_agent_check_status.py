"""Pin the fail-loud contract of ``scripts/agent_check_status.py``'s summary
reader.

``agent_check_status.py`` is the Makefile status target (``make`` ~line 664)
an operator runs to see dataset integrity, recent experiment runs, and a
"what to do next" recommendation. Stage 2 reads the most-recent suite's
``aggregate_summary.json``.

It used to load that summary behind a broad ``except Exception`` that printed a
buried ``[-]`` line and returned ``None`` — so a CORRUPT summary was
indistinguishable from a MISSING one, and stage 3 then told the operator to
"Run the 3-seed paper-memory suite" (``make paper-memory``): the suite had
already run, its summary was damaged, and the advice burned GPU re-running it
without ever naming the corrupt file. That is the same silent-death class the
sibling readers (``scripts/best_run_reader.py`` et al.) were promoted out of.

These tests pin the promotion to a loud, machine-distinguishable failure: a
corrupt summary must exit non-zero with a non-empty stderr that names the file
and the JSON cause, and must NOT emit the misleading "run the suite"
recommendation. Each is invoked as a real subprocess with ``cwd=tmp_path`` —
the exact CWD-globbing path the Makefile target takes — so the contract is
verified end-to-end. Reverting the reader to the old silent-swallow turns
``test_corrupt_summary_exits_loud_not_silent`` red (exit 0 + empty stderr under
the swallow, AND the misleading "run the suite" recommendation emitted), which
is the mutation the guard exists to prevent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "agent_check_status.py"

_SUITE = "runs/paper_memory_suite_test"
_SUMMARY = f"{_SUITE}/aggregate_summary.json"

# check_datasets() requires these files at-or-above these line counts before it
# reports data_ok=True; without them stage 3 short-circuits to "run data
# preparation" and never reaches the summary-dependent recommendation, so the
# silent-swallow's misleading "run the suite" path would be hidden. Seed them so
# the headline test exercises the real corruption -> wrong-recommendation path.
_DATA_MIN_LINES = {"train.jsonl": 4500, "valid_quick.jsonl": 450, "test.jsonl": 450}


def _run(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _make_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, n in _DATA_MIN_LINES.items():
        (data_dir / name).write_text("x\n" * n, encoding="utf-8")


def _make_suite(tmp_path: Path, summary_content: str) -> Path:
    summary = tmp_path / _SUMMARY
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summary_content, encoding="utf-8")
    return summary


def test_corrupt_summary_exits_loud_not_silent(tmp_path: Path) -> None:
    """The headline fix: a corrupt aggregate_summary.json must NOT degrade to
    the misleading 'run the suite' recommendation. Under the old broad
    ``except Exception`` this returned None -> (with data_ok=True) stage 3 said
    'Run the 3-seed paper-memory suite' with exit 0 and empty stderr. Now it
    must exit non-zero with a non-empty stderr that names the file + JSON cause,
    and must NOT emit the re-run advice."""
    _make_data(tmp_path)
    # Trailing comma — the exact 'Expecting property name enclosed in double
    # quotes' JSON failure mode (the prior iteration's judge_invalid_json).
    _make_suite(tmp_path, '{"seeds": 3,}')
    result = _run(tmp_path)
    assert result.returncode == 2, result.stderr
    assert result.stderr.strip(), "corrupt summary died silently (empty stderr)"
    assert "corrupt" in result.stderr.lower()
    assert "JSON" in result.stderr
    assert "aggregate_summary.json" in result.stderr
    # The misleading re-run recommendation must NOT be emitted: the suite ran,
    # the summary is damaged, not absent.
    assert "Run the 3-seed paper-memory suite" not in result.stdout


def test_valid_summary_loads_and_exits_zero(tmp_path: Path) -> None:
    _make_data(tmp_path)
    _make_suite(tmp_path, '{"seeds": 3, "best_valid_loss": 1.05}')
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "[+] Loaded aggregate_summary.json" in result.stdout


def test_missing_summary_does_not_crash(tmp_path: Path) -> None:
    # A suite dir with no aggregate_summary.json: the reader must fall through
    # to None (unchanged) and exit 0 — corruption is loud, absence is not.
    suite = tmp_path / _SUITE
    suite.mkdir(parents=True)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "aggregate_summary.json not found" in result.stdout
