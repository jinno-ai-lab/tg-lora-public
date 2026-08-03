"""Pin the fail-loud contract of ``scripts/best_run_reader.py``.

``best_run_reader.py`` is the helper ``scripts/run_best_config_eval.sh`` calls to
extract the ``run_id`` to evaluate. It replaced an inline reader whose
``if not isinstance(data, dict): data = {}`` ran behind ``2>/dev/null`` and so
swallowed the *real* cause of a bad ``ranking.json``:

* a malformed file raised ``JSONDecodeError``; ``2>/dev/null`` ate the traceback
  and ``set -e`` killed the shell at the command substitution before the
  "could not determine best run" guard was reached — the operator saw *nothing*
  (silent death, exit 1, zero output); and
* a valid-JSON-but-non-object file (bare array/scalar/string) was silently
  coerced to ``{}`` and surfaced only as a misleading "could not determine
  best run".

These tests pin the promotion to a loud, machine-distinguishable failure:
distinct exit codes + a non-empty stderr that names the cause. Each is invoked
as a real subprocess — the exact path the shell takes — so the contract is
verified end-to-end, not just at the function level. Reverting the helper to the
old silent-swallow turns ``test_malformed_json_does_not_die_silently`` red
(empty stderr under the swallow), which is the mutation the guard exists to
prevent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "best_run_reader.py"


def _run(ranking_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), str(ranking_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_good_object_prints_run_id_and_exits_zero(tmp_path: Path) -> None:
    ranking = _write(
        tmp_path,
        "ranking.json",
        json.dumps({"best_run": {"run_id": "tg_lora_9b_accel_42"}}),
    )
    result = _run(ranking)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "tg_lora_9b_accel_42"


def test_non_dict_top_level_fails_loud_with_type(tmp_path: Path) -> None:
    # Valid JSON but a bare array — the exact shape the old `data = {}` silently
    # coerced. Must surface the type mismatch instead of pretending it's empty.
    ranking = _write(tmp_path, "ranking.json", json.dumps(["run_a", "run_b"]))
    result = _run(ranking)
    assert result.returncode == 3
    assert "object" in result.stderr
    assert "list" in result.stderr


def test_non_dict_scalar_top_level_fails_loud(tmp_path: Path) -> None:
    ranking = _write(tmp_path, "ranking.json", json.dumps("just_a_string"))
    result = _run(ranking)
    assert result.returncode == 3
    assert "object" in result.stderr
    assert "str" in result.stderr


def test_malformed_json_does_not_die_silently(tmp_path: Path) -> None:
    """The headline fix: a malformed ranking.json must NOT kill the consuming
    shell with zero output. Under the old inline reader this raised behind
    ``2>/dev/null`` and ``set -e`` aborted with NO stderr — silent death. Now it
    must exit non-zero with a non-empty, human-readable stderr cause."""
    ranking = _write(tmp_path, "ranking.json", "garbage{not even json")
    result = _run(ranking)
    assert result.returncode != 0
    assert result.stderr.strip(), "malformed ranking.json died silently (empty stderr)"
    assert "JSON" in result.stderr


def test_missing_best_run_fails_loud(tmp_path: Path) -> None:
    ranking = _write(tmp_path, "ranking.json", json.dumps({"runs": []}))
    result = _run(ranking)
    assert result.returncode == 4
    assert "best_run" in result.stderr


def test_best_run_not_a_dict_fails_loud(tmp_path: Path) -> None:
    # `best_run` present but a string — `.get('run_id')` would have raised the
    # very AttributeError the dict-guard family exists to prevent.
    ranking = _write(tmp_path, "ranking.json", json.dumps({"best_run": "a_string"}))
    result = _run(ranking)
    assert result.returncode == 4
    assert "best_run" in result.stderr


def test_empty_run_id_fails_loud(tmp_path: Path) -> None:
    ranking = _write(
        tmp_path, "ranking.json", json.dumps({"best_run": {"run_id": ""}})
    )
    result = _run(ranking)
    assert result.returncode == 4
    assert "run_id" in result.stderr


def test_missing_file_fails_loud(tmp_path: Path) -> None:
    result = _run(tmp_path / "does_not_exist.json")
    assert result.returncode == 2
    assert result.stderr.strip(), "missing ranking.json died silently (empty stderr)"


def test_wrong_arg_count_exits_usage() -> None:
    result = subprocess.run(
        [sys.executable, str(HELPER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 64
    assert "usage" in result.stderr
