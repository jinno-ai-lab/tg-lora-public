"""Coverage-floor regression harness for the JSON-extraction headline metric.

Background
----------
AI-Hub feedback bullet 2 asked (in its phantom vocabulary) for a
"coverage-threshold + fidelity regression harness in CI" so silent regressions
on the headline quality surface can't recur on future changes. ``chunker`` /
``site-packages`` / ``chunk_id`` / ``labeling_contract_hash`` / ``merge_results``
are grep-empty in this public mirror (phantom AI-Hub infra), so the faithful
realization lands on the REAL headline metric: the TG-LoRA experiment's quality
metric that replaced perplexity.

This file pins bullet 2's complementary properties on the WHOLE
model→scored-headline-metric pipeline, not just the pure scorer:

  * the scorer  — ``src/eval/eval_json_extraction.py``  (fidelity pinned by
    ``test_extraction_fidelity_delta.py``; coverage floor here).
  * the two torch-dependent wrappers that *drive* it —
    ``src/eval/json_generation.py`` (batched) + ``src/eval/jsonex_generation.py``
    (per-record). A regression in either silently corrupts the headline metric
    (the gold-emitting contract is pinned by ``test_eval_json_generation.py``;
    the emitted-summary strict-JSON contract by ``test_emitted_json_integrity.py``).

The floor: each pipeline module must have *zero* uncovered executable lines
from the test file(s) that exercise it, so any future change that adds an
uncovered code path to the scorer or its wrappers goes RED here instead of
silently shipping a dead/mis-wired branch — exactly the defect class the
``555cae4`` greedy-regex fix addressed (a branch the suite *thought* was
exercised but wasn't).

Intentionally pragma-excluded lines (not counted as uncovered):
  * the ``_first_json_object`` non-dict fallthrough in the scorer (dead under
    JSON object semantics — a ``{...}`` span always parses to ``dict``);
  * the scorer's ``if __name__ == "__main__"`` CLI self-test;
  * ``json_generation.py``'s ``if __name__ == "__main__"`` CLI smoke.

Coverage topology note: the scorer is fully exercised by its OWN test file, but
``json_generation.py``'s ``format_score_summary`` is pinned by the emitted-JSON
integrity suite, so the wrapper floor runs BOTH complementary files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCORER_MODULE = "eval_json_extraction.py"
SCORER_TEST_FILES = ["tests/test_eval_json_extraction.py"]

WRAPPER_MODULES = ("json_generation.py", "jsonex_generation.py")
# The generation-behaviour branches live in the wrapper test file; the
# emitted-summary strict-JSON branch (format_score_summary) is pinned by the
# emitted-integrity suite. Both are needed for the wrappers' full coverage.
WRAPPER_TEST_FILES = [
    "tests/test_eval_json_generation.py",
    "tests/test_emitted_json_integrity.py",
]


def _missing_lines_for(
    tmp_path: Path, test_files: list[str], module_basename: str
) -> list[int]:
    """Run ``test_files`` together under coverage and return the uncovered
    executable line numbers of ``src/eval/<module_basename>``.

    pragma-excluded lines are already removed by coverage. A fresh per-module
    ``--data-file`` so this never clobbers or reads a parent ``make test-cov``
    run that may already be holding ``.coverage``.
    """
    tag = Path(module_basename).stem
    data_file = tmp_path / f"cov_{tag}.data"
    run = subprocess.run(
        [
            sys.executable, "-m", "coverage", "run",
            f"--data-file={data_file}",
            "--source=src/eval",
            "-m", "pytest", "-q", *test_files,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, (
        f"coverage run of {test_files} failed (rc={run.returncode}):\n"
        f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    )
    json_path = tmp_path / f"cov_{tag}.json"
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
        (v for k, v in files.items() if k.endswith(module_basename)),
        None,
    )
    assert entry is not None, (
        f"src/eval/{module_basename} missing from coverage report of "
        f"{test_files}; keys={list(files)}"
    )
    return entry.get("missing_lines", [])


def test_eval_json_extraction_has_full_line_coverage(tmp_path):
    """The pure scorer must have 0 uncovered executable lines from its own
    test file (see module docstring for the why)."""
    missing = _missing_lines_for(tmp_path, SCORER_TEST_FILES, SCORER_MODULE)
    assert missing == [], (
        f"src/eval/{SCORER_MODULE} has {len(missing)} uncovered executable "
        f"line(s) from {SCORER_TEST_FILES}: {missing}. Cover them with a "
        "behavior test, or if genuinely uncoverable, exclude with "
        "`# pragma: no cover`."
    )


def test_json_generation_wrappers_have_full_line_coverage(tmp_path):
    """The two wrappers that drive the scorer must also have 0 uncovered
    executable lines — completing the coverage floor over the whole
    model→scored-headline-metric pipeline. A regression in the generation,
    device-auto-detect, prompt-reconstruction, or max_examples path that the
    suite *thinks* is exercised but isn't would silently corrupt the metric."""
    for module in WRAPPER_MODULES:
        missing = _missing_lines_for(tmp_path, WRAPPER_TEST_FILES, module)
        assert missing == [], (
            f"src/eval/{module} has {len(missing)} uncovered executable "
            f"line(s) from {WRAPPER_TEST_FILES}: {missing}. Cover them with a "
            "behavior test, or if genuinely uncoverable, exclude with "
            "`# pragma: no cover`."
        )
