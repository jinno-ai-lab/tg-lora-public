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
The two existing ``src/``-only static guards are ruff-based and stay red on
``scripts/`` (documented pre-existing lint debt); this guard is unrelated to
that debt — it scans only for the ``torch.save`` call expression, which a clean
``scripts/`` migration satisfies regardless of broader lint state.
"""

from __future__ import annotations

import ast
from pathlib import Path

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
