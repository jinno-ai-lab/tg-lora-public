#!/usr/bin/env python3
"""Read the best ``run_id`` from a sweep's ``ranking.json`` — fail loud, never silent.

Invoked by ``scripts/run_best_config_eval.sh`` to extract the run to evaluate.
Replaces the inline reader it used to embed, whose

.. code-block:: python

    data = json.load(f)
    if not isinstance(data, dict):
        data = {}
    best = data.get("best_run", {})
    print(best.get("run_id", ""))

ran behind a ``2>/dev/null`` and so swallowed the REAL cause of a bad
``ranking.json``:

* a **malformed** file raised ``JSONDecodeError``; ``2>/dev/null`` ate the
  traceback and ``set -e`` then killed the shell at the command substitution
  before the "could not determine best run" guard was ever reached — the
  operator saw *nothing* (silent death, exit 1, zero output); and
* a **valid-JSON-but-non-object** file (a bare array/scalar/string — a
  hand-edited or wrong-format ranking) was silently coerced to ``{}``, so the
  same guard fired with a misleading "could not determine best run" instead of
  naming the shape mismatch.

This helper promotes both to a loud, machine-distinguishable failure: distinct
exit codes + a one-line stderr cause, so an operator (or the shell's ``||``
handler) knows *why* it failed instead of guessing.

Exit contract:

* valid object with a non-empty ``best_run.run_id`` → print ``run_id``, exit 0
* malformed JSON / unreadable file → stderr cause, exit 2
* valid JSON but top-level not an object → stderr cause, exit 3
* object missing ``best_run`` / ``best_run`` not an object / ``run_id`` empty
  → stderr cause, exit 4
* wrong arg count → usage on stderr, exit 64 (``EX_USAGE``)
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


def best_run_id(ranking_path: Path) -> str:
    """Return ``best_run.run_id`` from *ranking_path*, failing loud on any
    non-conformant shape (see the module docstring's exit contract)."""
    try:
        raw = ranking_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fail(2, f"ranking.json: {ranking_path} not found")
    except OSError as exc:
        _fail(2, f"ranking.json: {ranking_path} unreadable ({exc})")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(2, f"ranking.json: {ranking_path} is not valid JSON ({exc.msg})")

    if not isinstance(data, dict):
        _fail(
            3,
            f"ranking.json: {ranking_path} expected a JSON object at top level, "
            f"got {type(data).__name__}",
        )

    best = data.get("best_run")
    if not isinstance(best, dict):
        _fail(4, f"ranking.json: {ranking_path} 'best_run' missing or not an object")

    run_id = best.get("run_id", "")
    if not isinstance(run_id, str) or not run_id:
        _fail(4, f"ranking.json: {ranking_path} 'best_run.run_id' missing or empty")

    return run_id


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(f"usage: {argv[0]} <ranking.json>\n")
        return 64
    print(best_run_id(Path(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
