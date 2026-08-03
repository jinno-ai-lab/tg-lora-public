#!/usr/bin/env python3
"""Read the ``run_id`` of the first ``run_header`` record in a
``run_metrics.jsonl`` — fail loud, never silent.

Invoked by ``scripts/run_best_config_eval.sh`` as the FALLBACK run-directory
matcher: when the primary glob ``tg_lora_9b_accel_*${BEST_RUN}*`` does not match
a directory name, the shell scans each candidate dir's ``run_metrics.jsonl`` for
the record whose ``type == 'run_header'`` (the schema
``src/utils/run_metrics.py`` writes first per run) and compares its ``run_id``.

Replaces the inline reader the shell used to embed, whose

.. code-block:: python

    for line in f:
        obj = json.loads(line)
        if isinstance(obj, dict) and obj.get('type') == 'run_header':
            print(obj.get('run_id', ''))
            break

ran behind a ``2>/dev/null`` and so swallowed the REAL cause of a bad
``run_metrics.jsonl``: a corrupt line raised ``JSONDecodeError``, the traceback
was eaten, ``run_id`` came back empty, every comparison missed, and the operator
saw only a misleading "Could not find run directory" — never the corrupt file.

This helper promotes that to a loud, machine-distinguishable failure, mirroring
:mod:`best_run_reader`: distinct exit codes + a one-line stderr cause, so an
operator (or the shell's ``||`` handler) knows *why* it failed instead of
guessing.

Exit contract:

* a JSONL whose first ``run_header`` record has a non-empty ``run_id``
  → print ``run_id``, exit 0
* file missing / unreadable, or a non-blank line that is not valid JSON
  → stderr cause, exit 2
* valid JSON throughout but no ``run_header`` record, or a ``run_header``
  whose ``run_id`` is missing/empty → stderr cause, exit 4
* wrong arg count → usage on stderr, exit 64 (``EX_USAGE``)

Blank/whitespace-only lines are skipped (a trailing newline is normal for the
producer's ``"wb"``/``"ab"`` append); only a non-blank line that fails to parse
is treated as corruption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn


def _fail(code: int, message: str) -> NoReturn:
    """Write *message* to stderr and exit with *code* (never return)."""
    sys.stderr.write(f"{message}\n")
    raise SystemExit(code)


def run_header_run_id(metrics_path: Path) -> str:
    """Return the ``run_id`` of the first ``run_header`` record in
    *metrics_path*, failing loud on any unreadable / malformed / headerless
    file (see the module docstring's exit contract)."""
    try:
        raw = metrics_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fail(2, f"run_metrics.jsonl: {metrics_path} not found")
    except OSError as exc:
        _fail(2, f"run_metrics.jsonl: {metrics_path} unreadable ({exc})")

    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue  # trailing newline / blank line — normal, not corruption
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(
                2,
                f"run_metrics.jsonl: {metrics_path} line {lineno} is not valid "
                f"JSON ({exc.msg})",
            )
        if isinstance(obj, dict) and obj.get("type") == "run_header":
            run_id = obj.get("run_id", "")
            if not isinstance(run_id, str) or not run_id:
                _fail(
                    4,
                    f"run_metrics.jsonl: {metrics_path} run_header record has "
                    f"a missing or empty run_id",
                )
            return run_id

    _fail(4, f"run_metrics.jsonl: {metrics_path} has no run_header record")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(f"usage: {argv[0]} <run_metrics.jsonl>\n")
        return 64
    print(run_header_run_id(Path(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
