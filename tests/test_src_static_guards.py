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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src"


def _run_ruff(target: Path) -> subprocess.CompletedProcess[str]:
    ruff = shutil.which("ruff")
    return subprocess.run(
        [ruff, "check", str(target)], capture_output=True, text=True, check=False
    )


def test_src_tree_is_ruff_clean() -> None:
    """``ruff check src/`` must report zero findings across all rules.

    Guards the ``src/`` lint-clean invariant: a non-zero count regresses it
    (re-introduced dead local, unused import, ambiguous name) and makes the
    audit's standing claim false. See the module docstring for the cleanup this
    locks in and the deliberate ``src/``-only scope.
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the cleanliness guard")

    assert TARGET.is_dir(), f"src/ tree not found at {TARGET}"
    proc = _run_ruff(TARGET)
    assert proc.returncode == 0, (
        "src/ must be ruff-clean (zero findings across all rules) — a non-zero "
        "count regresses the src/ lint-clean invariant and makes the audit's "
        "standing claim false. ruff output:\n" + (proc.stdout + proc.stderr).strip()
    )
