#!/usr/bin/env python3
"""Print the per-task accuracy table from an lm-eval ``lm_eval_results.json`` —
fail loud on a malformed or wrong-shape file, never silent.

Invoked by ``scripts/run_best_config_eval.sh`` to display the eval summary once
``lm_eval`` has written its artifact. Replaces the inline reader the shell used to
embed, whose

.. code-block:: python

    data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
        for task, metrics in results.items():
            acc = metrics.get('acc,none', metrics.get('acc_norm,none', 'N/A'))
            print(f'  {task}: {acc}')
    else:
        print(json.dumps(data, indent=2))

silently swallowed an unexpected shape: when ``lm_eval_results.json`` was valid
JSON but NOT a ``dict`` with a ``results`` key (a schema drift, a hand-edit, or a
future lm-eval version wrapping the payload differently), it pretty-printed the
whole blob with NO signal that the per-task table had failed to build — the
operator saw "eval done" plus a mystery JSON dump and could mistake a broken
parse for a successful eval.

This helper promotes that to a loud, machine-distinguishable signal, mirroring
:mod:`best_run_reader` / :mod:`run_metrics_reader`: the per-task table prints on
the happy path; on any shape deviation a one-line stderr cause names the
mismatch and the raw payload is still dumped to stdout (so the operator keeps the
data), with a distinct non-zero exit so the shell (or CI) can tell a broken
parse from a good one. The shell treats the non-zero as non-fatal — the eval
already succeeded and the artifact is durable — but the signal is no longer
silent.

Exit contract:

* a ``dict`` with a ``results`` object → print ``"  {task}: {acc}"`` per task,
  exit 0
* malformed JSON / unreadable file → stderr cause, exit 2
* valid JSON but top-level not an object → stderr cause + raw dump, exit 3
* object missing ``results`` / ``results`` not an object → stderr cause + raw
  dump, exit 4
* wrong arg count → usage on stderr, exit 64 (``EX_USAGE``)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn


def _fail(code: int, message: str, *, raw: Any = None) -> NoReturn:
    """Write *message* to stderr, optionally dump *raw* payload to stdout (so the
    operator keeps the data on a shape mismatch), then exit with *code*."""
    sys.stderr.write(f"{message}\n")
    if raw is not None:
        print(json.dumps(raw, indent=2, default=str))
    raise SystemExit(code)


def print_results(results_path: Path) -> None:
    """Print the per-task accuracy table from *results_path*, failing loud on any
    malformed or non-conformant shape (see the module docstring's exit
    contract)."""
    try:
        raw_text = results_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fail(2, f"lm_eval_results.json: {results_path} not found")
    except OSError as exc:
        _fail(2, f"lm_eval_results.json: {results_path} unreadable ({exc})")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _fail(2, f"lm_eval_results.json: {results_path} is not valid JSON ({exc.msg})")

    if not isinstance(data, dict):
        _fail(
            3,
            f"lm_eval_results.json: {results_path} expected a JSON object at top "
            f"level, got {type(data).__name__}",
            raw=data,
        )

    results = data.get("results")
    if not isinstance(results, dict):
        _fail(
            4,
            f"lm_eval_results.json: {results_path} 'results' missing or not an "
            f"object (got {type(results).__name__})",
            raw=data,
        )

    # Mirror the original display exactly on the happy path (byte-identical
    # per-task lines); guard only the AttributeError the dict-guard family exists
    # to prevent — a per-task ``metrics`` that isn't a dict would crash the loop
    # mid-table otherwise.
    for task, metrics in results.items():
        if isinstance(metrics, dict):
            acc = metrics.get("acc,none", metrics.get("acc_norm,none", "N/A"))
        else:
            acc = "N/A"
        print(f"  {task}: {acc}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(f"usage: {argv[0]} <lm_eval_results.json>\n")
        return 64
    print_results(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
