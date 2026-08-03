"""Pin the fail-loud contract of ``scripts/lm_eval_results_reader.py``.

``lm_eval_results_reader.py`` is the helper ``scripts/run_best_config_eval.sh``
calls to display the per-task accuracy table once ``lm_eval`` has written its
``lm_eval_results.json`` artifact. It replaced an inline reader whose

.. code-block:: python

    if isinstance(data, dict) and 'results' in data:
        ...
    else:
        print(json.dumps(data, indent=2))

silently swallowed an unexpected shape: a valid-JSON-but-wrong-schema file (a
hand-edit, a schema drift, or a future lm-eval version wrapping the payload
differently) was pretty-printed as a blob with NO signal that the per-task table
had failed to build — the operator could mistake a broken parse for a successful
eval (the "false detection" the silent-default family exists to prevent).

These tests pin the promotion to a loud, machine-distinguishable signal (the same
shape as ``test_best_run_reader.py`` / ``test_run_metrics_reader.py``): the table
prints byte-identically on the happy path; any shape deviation emits a non-empty
stderr that names the cause + a distinct non-zero exit, while the raw payload is
still dumped to stdout so the data isn't hidden. Each is invoked as a real
subprocess — the exact path the shell takes — so the contract is verified
end-to-end, not just at the function level. Reverting the helper to the old
silent-swallow turns ``test_wrong_shape_does_not_die_silently`` red (empty stderr
under the swallow), which is the mutation the guard exists to prevent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "lm_eval_results_reader.py"


def _run(results_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), str(results_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_good_object_prints_per_task_table(tmp_path: Path) -> None:
    results = _write(
        tmp_path,
        "lm_eval_results.json",
        json.dumps(
            {
                "results": {
                    "arc_easy": {"acc,none": 0.75, "acc_norm,none": 0.78},
                    "hellaswag": {"acc,none": 0.6, "acc_norm,none": 0.62},
                }
            }
        ),
    )
    proc = _run(results)
    assert proc.returncode == 0
    assert proc.stderr == ""
    # Byte-identical to the inline reader's happy path — no regression in display.
    assert proc.stdout == "  arc_easy: 0.75\n  hellaswag: 0.6\n"


def test_acc_norm_fallback_when_acc_absent(tmp_path: Path) -> None:
    # gsm8k / truthfulqa expose only acc; arc exposes only acc_norm — the
    # `metrics.get('acc,none', metrics.get('acc_norm,none', 'N/A'))` precedence
    # must be preserved exactly.
    results = _write(
        tmp_path,
        "lm_eval_results.json",
        json.dumps(
            {"results": {"arc_challenge": {"acc_norm,none": 0.55}, "gsm8k": {"acc,none": 0.4}}}
        ),
    )
    proc = _run(results)
    assert proc.returncode == 0
    assert "arc_challenge: 0.55" in proc.stdout
    assert "gsm8k: 0.4" in proc.stdout


def test_wrong_shape_does_not_die_silently(tmp_path: Path) -> None:
    """The headline fix: a wrong-shape lm_eval_results.json must NOT be swallowed.
    Under the old inline reader the `else` branch pretty-printed this blob with
    NO stderr — the operator saw a mystery dump, never a signal that the parse
    failed. Now it must exit non-zero with a non-empty stderr that names the
    shape mismatch, while still dumping the raw payload so the data isn't
    hidden."""
    results = _write(tmp_path, "lm_eval_results.json", json.dumps(["arc_easy", "hellaswag"]))
    proc = _run(results)
    assert proc.returncode == 3
    assert proc.stderr.strip(), "wrong-shape lm_eval_results.json died silently (empty stderr)"
    assert "object" in proc.stderr
    assert "list" in proc.stderr
    # Raw payload still surfaced — the operator keeps the data.
    assert "arc_easy" in proc.stdout


def test_scalar_top_level_fails_loud_with_type(tmp_path: Path) -> None:
    results = _write(tmp_path, "lm_eval_results.json", json.dumps("just_a_string"))
    proc = _run(results)
    assert proc.returncode == 3
    assert "object" in proc.stderr
    assert "str" in proc.stderr


def test_malformed_json_does_not_die_silently(tmp_path: Path) -> None:
    # A truncated/interrupted lm_eval write — a real problem, not a display
    # nicety. Must surface the JSON failure rather than crashing the consuming
    # shell with a bare traceback.
    results = _write(tmp_path, "lm_eval_results.json", "garbage{not even json")
    proc = _run(results)
    assert proc.returncode != 0
    assert proc.stderr.strip(), "malformed lm_eval_results.json died silently (empty stderr)"
    assert "JSON" in proc.stderr


def test_missing_results_key_fails_loud(tmp_path: Path) -> None:
    # Dict but no 'results' — a schema drift (lm-eval wrapped the payload). The
    # old `else` would have silently dumped this; now it must name the missing
    # key, exactly the way best_run_reader names a missing 'best_run'.
    results = _write(tmp_path, "lm_eval_results.json", json.dumps({"metadata": {"version": "0.4"}}))
    proc = _run(results)
    assert proc.returncode == 4
    assert "results" in proc.stderr
    # Raw payload still surfaced.
    assert "metadata" in proc.stdout


def test_results_not_a_dict_fails_loud(tmp_path: Path) -> None:
    # `results` present but a list — `results.items()` would have raised the very
    # AttributeError the dict-guard family exists to prevent.
    results = _write(tmp_path, "lm_eval_results.json", json.dumps({"results": ["arc_easy"]}))
    proc = _run(results)
    assert proc.returncode == 4
    assert "results" in proc.stderr


def test_per_task_non_dict_metrics_does_not_crash(tmp_path: Path) -> None:
    # A weird per-task metrics value must not crash the table loop with an
    # AttributeError mid-print — print N/A and keep going (the dict-guard
    # spirit), rather than abort the whole display on one malformed task.
    results = _write(
        tmp_path,
        "lm_eval_results.json",
        json.dumps({"results": {"arc_easy": "not_a_dict", "hellaswag": {"acc,none": 0.6}}}),
    )
    proc = _run(results)
    assert proc.returncode == 0
    assert "arc_easy: N/A" in proc.stdout
    assert "hellaswag: 0.6" in proc.stdout


def test_missing_file_fails_loud(tmp_path: Path) -> None:
    proc = _run(tmp_path / "does_not_exist.json")
    assert proc.returncode == 2
    assert proc.stderr.strip(), "missing lm_eval_results.json died silently (empty stderr)"


def test_wrong_arg_count_exits_usage() -> None:
    proc = subprocess.run(
        [sys.executable, str(HELPER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 64
    assert "usage" in proc.stderr
