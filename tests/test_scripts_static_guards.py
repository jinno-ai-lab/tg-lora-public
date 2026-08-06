"""Static guard: no bare ``torch.save`` anywhere in ``scripts/``.

Companion to :func:`tests.test_src_static_guards.test_no_bare_torch_save_in_src`,
which pins the same invariant for ``src/``. Together the two guards close the
loop the run-feedback review named — "a static guard that forbids
``torch.save(<path>)`` outside that helper so no future or existing site
reintroduces the mid-save corruption risk" — across every artifact-writing slice
of source: ``src/`` *and* ``scripts/``.

The two ``scripts/`` sites this pins both write artifacts that must survive a
mid-save fault:

* ``scripts/recover.py`` writes ``training_state.pt`` (the sanitized,
  resume-critical run state — a torn write here is exactly the resume break the
  atomic-save helper exists to prevent), and
* ``scripts/collect_true_gradients.py`` writes ``gradient_step_*.pt`` (the
  effective-update tensors, each a full forward/backward cycle to recompute,
  feeding the trajectory-delta evaluation).

Both were migrated to :func:`src.utils.atomic_save._atomic_torch_save`
(PID-suffixed temp + ``os.replace``); this test pins that neither, nor any
future ``scripts/`` addition, regresses to a bare ``torch.save(blob, path)``
that would truncate the destination before writing and leave a torn file on an
OOM kill / SIGINT mid-dump.

Scope is ``scripts/`` only, mirroring the ``src/`` guard's deliberate split.
``tests/`` is EXCLUDED on purpose: test fixtures legitimately call
``torch.save`` to seed deliberately-corrupt files (see
``test_prefix_feature_shard.py``, ``test_recover.py``,
``test_prefix_feature_cache.py``) — those are inputs-under-test, not on-disk
artifacts a real run depends on, so they must not be policed by this invariant.

This is an AST scan, not a text grep, so docstring/comment mentions of
``torch.save`` do not false-positive — only real call expressions are counted.
Unrelated to lint state — it scans only for the ``torch.save`` call expression,
which a clean ``scripts/`` migration satisfies regardless of broader lint. The
companion ruff guard below (:func:`test_scripts_tree_real_bug_lint_clean`) now
pins ``scripts/`` clean across the pyflakes real-bug subset
(F401/F811/F821/F841/E9); only lower-priority style debt (E741 / F541 / E702)
keeps a bare ``ruff check scripts/`` red.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "scripts"


def test_no_bare_torch_save_in_scripts() -> None:
    """No ``scripts/`` file may call ``torch.save(<path>)`` directly.

    Every on-disk artifact written from ``scripts/`` must persist via
    :func:`src.utils.atomic_save._atomic_torch_save`. A bare
    ``torch.save(blob, path)`` truncates the destination before writing, so an
    interruption mid-dump leaves a torn file that breaks the next load — losing
    a run (``training_state.pt``) or forcing an expensive recompute
    (``gradient_step_*.pt``). See the module docstring for the full rationale
    and the deliberate ``tests/`` exclusion.
    """
    assert TARGET.is_dir(), f"scripts/ tree not found at {TARGET}"

    offenders: list[tuple[Path, int]] = []
    for src_file in sorted(TARGET.rglob("*.py")):
        tree = ast.parse(
            src_file.read_text(encoding="utf-8"), filename=str(src_file)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                offenders.append((src_file, node.lineno))

    assert not offenders, (
        "Every on-disk artifact written from scripts/ must persist via "
        "src.utils.atomic_save._atomic_torch_save — a bare "
        "torch.save(<path>) reintroduces the mid-save truncation hazard "
        "(torn training_state.pt / torn gradient_step_*.pt) the atomic-save "
        "helper exists to prevent. Offending sites:\n"
        + "\n".join(f"  {f}:{ln}" for f, ln in offenders)
    )


# ---------------------------------------------------------------------------
# Real-bug lint invariant: ``scripts/`` clean across the pyflakes + runtime
# subset (F401/F811/F821/F841/E9). Sibling to ``test_src_static_guards.
# test_src_tree_is_ruff_clean``, which pins the FULL canonical set (E4/E7/E9/F)
# on ``src/``. ``scripts/`` still carries documented lower-priority STYLE debt
# (F541 placeholder-less f-strings, E741 ambiguous names, E702 semicolons) that
# this guard deliberately does NOT police — pinning the crash/masking defect
# classes first keeps ``scripts/`` regression-proof without a high-churn bulk
# restyle. E4 is excluded on purpose: several scripts bootstrap ``sys.path``
# before their first ``src.*`` import (E402, by design).
# ---------------------------------------------------------------------------

# The pyflakes + runtime real-bug subset of ruff's canonical selection —
# INTENTIONALLY narrower than ``test_src_static_guards._CANONICAL_LINT_RULES``
# (E4/E7/E9/F): it drops the pure-style rules that remain as documented
# lower-priority debt in ``scripts/`` (E741 ambiguous name, E702 semicolons,
# F541 placeholder-less f-string) and E4 (E402 import-not-at-top, which scripts
# use deliberately for sys.path bootstrap; E401 multi-import). What remains is
# exactly the defect classes that crash or mask bugs:
#   F401 unused import, F811 redefined-while-unused, F821 undefined name,
#   F841 dead local, E9 runtime/syntax/IO error.
_REAL_BUG_LINT_RULES = ("F401", "F811", "F821", "F841", "E9")


def _run_ruff(
    target: Path, select: Sequence[str] = ()
) -> subprocess.CompletedProcess[str]:
    ruff = shutil.which("ruff")
    cmd: list[str] = [ruff, "check"]
    if select:
        # ``--select`` REPLACES ruff's default selection with exactly the
        # listed rules (it does not extend the default) — same rationale as
        # ``test_src_static_guards._run_ruff``.
        cmd += ["--select", ",".join(select)]
    cmd.append(str(target))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_scripts_tree_real_bug_lint_clean() -> None:
    """``ruff check scripts/`` must report zero findings across the real-bug
    lint subset (F401/F811/F821/F841/E9).

    Guards the ``scripts/`` real-bug-clean invariant. A non-zero count
    regresses it and makes the audit's standing claim false; the highest-value
    case is F821 undefined name — a latent ``NameError`` that runs silently
    until the line executes (the ``increments`` typo ``119e815`` fixed), the
    exact defect class a clean ``src/`` tree keeps out of the core path. See
    the module docstring for why this pins the real-bug subset rather than the
    full canonical set, and the deliberate ``scripts/``-only scope (``tests/``
    still carries 9 such findings — pure F401+F841, no F821 — a separate,
    smaller unit).

    The cleanup this locks in removed 37 ``scripts/`` findings, all
    behavior-preserving: 22 unused imports (F401) and 15 dead locals (F841 —
    computed-but-never-read statistics like ``surr_median`` / ``w_traj_mean`` /
    ``l_after``; redundant matplotlib return captures ``bars`` / ``leg2`` whose
    call already renders; a placeholder ``w_before_cycle = None``; and an
    unpacked-but-unused ``meta`` / ``prev_meta`` pair in
    ``analyze_trajectory_deltas.py``).
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the cleanliness guard")

    assert TARGET.is_dir(), f"scripts/ tree not found at {TARGET}"
    proc = _run_ruff(TARGET, select=_REAL_BUG_LINT_RULES)
    assert proc.returncode == 0, (
        "scripts/ must be ruff-clean across the real-bug lint set "
        "(F401/F811/F821/F841/E9) — a non-zero count regresses the scripts/ "
        "real-bug-clean invariant (unused import / dead local / undefined "
        "name / runtime error) and makes the audit's standing claim false. "
        "ruff output:\n" + (proc.stdout + proc.stderr).strip()
    )
