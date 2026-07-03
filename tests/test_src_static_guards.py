"""Static guard: the whole ``src/`` tree must be ``ruff``-clean (zero findings).

Extends the per-file pin in ``test_train_tg_lora_static_guards.py`` (which locks
``src/training/train_tg_lora.py`` to zero findings) to the entire ``src/`` tree.
The two guards share one rationale: the audit long described lint debt as
"pre-existing, not a regression" prose, and prose drifts. Commit a4d7c26 made
that drift measurable by pinning the training entry point; this test pins the
rest of the source tree the same way so a future ambiguous-name comprehension
variable, unused import, or re-introduced dead local fails CI here instead of
re-stating a stale lint count in PURPOSE.md.

The cleanup this locks in removed 11 ``src/`` findings, all behavior-preserving:
two unused imports (F401), three dead locals (F841 — computed-but-never-read,
the same dead-state class a4d7c26 removed from the trainer), and six ambiguous
``l`` loop variables (E741). It also surfaced and fixed a real latent
``NameError`` in ``scripts/`` (``incregments`` typo, F821) — covered by
``tests/test_analyze_trajectory_deltas.py`` — which is exactly the defect class
a clean ``src/`` tree keeps out of the core path.

Scope is ``src/`` only: ``scripts/`` and ``tests/`` still carry documented
pre-existing lint debt (``make lint`` checks all three and remains red on those
two). Pinning ``src/`` first keeps the highest-value slice — the imported core
— regression-proof without a high-churn bulk cleanup of the lower-priority
slices. Runs ``ruff`` via subprocess and skips when ``ruff`` is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src"


def _run_ruff(
    target: Path, select: Sequence[str] = ()
) -> subprocess.CompletedProcess[str]:
    ruff = shutil.which("ruff")
    cmd: list[str] = [ruff, "check"]
    if select:
        # ruff's default rule set is E4/E7/E9/F — it deliberately EXCLUDES the
        # pycodestyle W rules. ``select`` opts a specific rule back in so a guard
        # can police a defect class the default-clean check is blind to (W605).
        cmd += ["--select", ",".join(select)]
    cmd.append(str(target))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_src_tree_is_ruff_clean() -> None:
    """``ruff check src/`` must report zero findings across the default rules.

    Guards the ``src/`` lint-clean invariant: a non-zero count regresses it
    (re-introduced dead local, unused import, ambiguous name) and makes the
    audit's standing claim false. See the module docstring for the cleanup this
    locks in and the deliberate ``src/``-only scope.

    Scope note: this runs ruff's *default* rule set (E4/E7/E9/F), which excludes
    the pycodestyle ``W`` rules — so it does NOT catch invalid escape sequences
    (W605). Those are policed separately by
    :func:`test_src_tree_has_no_invalid_escape_sequences`, because a ``\\*`` in a
    docstring passes this default check while Python emits a ``DeprecationWarning``
    on every import (a ``SyntaxWarning`` from 3.12).
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the cleanliness guard")

    assert TARGET.is_dir(), f"src/ tree not found at {TARGET}"
    proc = _run_ruff(TARGET)
    assert proc.returncode == 0, (
        "src/ must be ruff-clean across the default rule set (E4/E7/E9/F) — a "
        "non-zero count regresses the src/ lint-clean invariant and makes the "
        "audit's standing claim false. ruff output:\n" + (proc.stdout + proc.stderr).strip()
    )


def test_src_tree_has_no_invalid_escape_sequences() -> None:
    """No ``src/`` file may contain an invalid escape sequence (ruff W605).

    Companion to :func:`test_src_tree_is_ruff_clean`. That test's default rule
    set excludes W605, so a ``\\*`` / ``\\d`` / ``\\m`` in a docstring passes it
    silently while Python emits a ``DeprecationWarning`` on every import (a hard
    ``SyntaxWarning`` from 3.12). One such site did exactly that for months — a
    ``L\\*`` in ``run_metrics.record_full_eval_loss``'s docstring rendered as the
    literal ``L\\*`` (not the intended ``L*``) and fired the warning on every
    test run, all while the default-clean guard stayed green. This test selects
    W605 explicitly to close that blind spot and keep the escape-sequence class
    out of the imported core for good.
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the escape-sequence guard")

    assert TARGET.is_dir(), f"src/ tree not found at {TARGET}"
    proc = _run_ruff(TARGET, select=("W605",))
    assert proc.returncode == 0, (
        "src/ must contain no invalid escape sequences (W605) — each one is a "
        "DeprecationWarning now and a SyntaxWarning from Python 3.12. "
        "ruff output:\n" + (proc.stdout + proc.stderr).strip()
    )
