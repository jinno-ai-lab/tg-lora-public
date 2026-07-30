"""Coverage-floor regression harness for the JSON-extraction headline metric.

Background
----------
AI-Hub feedback bullet 2 asked (in its phantom vocabulary) for a
"coverage-threshold + fidelity regression harness in CI" so silent regressions
on the headline quality surface can't recur on future changes. ``chunker`` /
``site-packages`` / ``chunk_id`` / ``labeling_contract_hash`` / ``merge_results``
are grep-empty in this public mirror (phantom AI-Hub infra), so the faithful
realization lands on the REAL headline metric: ``src/eval/eval_json_extraction.py``
— the TG-LoRA experiment's quality metric that replaced perplexity, wired into
the base-model go/no-go gate and both generation-eval wrappers.

Fidelity (behavioral correctness) is already pinned by
``tests/test_extraction_fidelity_delta.py`` (iter-39: balanced 9/9 vs greedy
5/9 ⇒ RESCUED=4 / REGRESSED=0). This file pins the complementary property
bullet 2 names explicitly — the **coverage floor**: the module's own test file
must leave *zero* executable lines uncovered, so any future change that adds
uncovered code to the scorer goes RED here instead of silently shipping a
dead/mis-wired branch (exactly the defect class the ``555cae4`` greedy-regex
fix addressed — a branch the suite *thought* was exercised but wasn't).

The two intentionally-uncovered lines are pragma-excluded in the module:
  * the ``_first_json_object`` non-dict fallthrough (dead under JSON object
    semantics — a ``{...}`` span always parses to ``dict``);
  * the ``if __name__ == "__main__"`` CLI self-test block.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_UNDER_TEST = "src/eval/eval_json_extraction.py"
OWN_TEST_FILE = "tests/test_eval_json_extraction.py"


def _module_missing_lines(tmp_path: Path) -> list[int]:
    """Run the module's own test file under coverage; return its uncovered
    executable line numbers (pragma-excluded lines already removed)."""
    data_file = tmp_path / "cov.data"
    # A fresh ``--data-file`` so this never clobbers or reads a parent
    # ``make test-cov`` run that may already be holding ``.coverage``.
    run = subprocess.run(
        [
            sys.executable, "-m", "coverage", "run",
            f"--data-file={data_file}",
            "--source=src/eval",
            "-m", "pytest", "-q", OWN_TEST_FILE,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, (
        f"coverage run of {OWN_TEST_FILE} failed (rc={run.returncode}):\n"
        f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    )
    json_path = tmp_path / "cov.json"
    rep = subprocess.run(
        [
            sys.executable, "-m", "coverage", "json",
            f"--data-file={data_file}", "-o", str(json_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert rep.returncode == 0, f"coverage json failed:\n{rep.stderr}"
    report = json.loads(json_path.read_text())
    files = report.get("files", {})
    entry = next(
        (v for k, v in files.items() if k.endswith("eval_json_extraction.py")),
        None,
    )
    assert entry is not None, (
        f"{MODULE_UNDER_TEST} missing from coverage report; keys={list(files)}"
    )
    return entry.get("missing_lines", [])


def test_eval_json_extraction_has_full_line_coverage(tmp_path):
    """The headline quality metric must have 0 uncovered executable lines from
    its own test file — a regression floor (see module docstring for the why)."""
    missing = _module_missing_lines(tmp_path)
    assert missing == [], (
        f"{MODULE_UNDER_TEST} has {len(missing)} uncovered executable line(s) "
        f"from {OWN_TEST_FILE}: {missing}. Cover them with a behavior test, or "
        f"if genuinely uncoverable, exclude with `# pragma: no cover`."
    )
