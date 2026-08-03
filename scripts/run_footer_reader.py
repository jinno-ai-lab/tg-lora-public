#!/usr/bin/env python3
"""Read the ``run_footer`` summary line of a ``run_metrics.jsonl`` — fail loud,
never silent.

Invoked by ``scripts/run_kstep_rollback_test.sh`` to print each run's
end-of-run summary (best valid loss / final train loss / perplexity / wall
seconds) from the ``run_footer`` record the producer
(:mod:`src.utils.run_metrics`) appends last per run.

Replaces the inline reader the shell used to embed, whose

.. code-block:: python

    for line in f:
        r = orjson.loads(line)
        if not isinstance(r, dict):
            continue
        if r['type'] == 'run_footer':
            print(f'{name}  best_valid={best}  ...  wall={wall:.0f}s')
            break

ran behind a ``2>/dev/null`` and an ``|| echo "{name}  (no footer yet)"`` and so
swallowed the REAL cause of a bad ``run_metrics.jsonl``: a corrupt line raised
``JSONDecodeError``, the traceback was eaten, and the shell printed the SAME
"(no footer yet)" it prints for a run that is simply still in progress — masking
a *corrupt* file as a *normal transient*. An operator re-checking a finished
sweep saw "(no footer yet)" and waited on a run that had already died, instead
of being pointed at the corrupt file.

This helper promotes that to a loud, machine-distinguishable failure, mirroring
:mod:`best_run_reader` / :mod:`run_metrics_reader`: distinct exit codes + a
one-line stderr cause, so the shell can tell CORRUPTION (warn, name the file)
apart from a genuine NO-FOOTER-YET (the normal mid-run state) — instead of
collapsing both into one silent default.

Exit contract:

* a JSONL whose ``run_footer`` record is present → print the formatted summary
  to stdout, exit 0
* file missing / unreadable, or a non-blank line that is not valid JSON
  → stderr cause, exit 2 (corrupt — the shell warns)
* valid JSON throughout but no ``run_footer`` record → exit 3, no stderr
  (transient — the run is still in progress; the shell prints "(no footer yet)")
* wrong arg count → usage on stderr, exit 64 (``EX_USAGE``)

Blank/whitespace-only lines are skipped (a trailing newline is normal for the
producer's ``"wb"``/``"ab"`` append); only a non-blank line that fails to parse
is treated as corruption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn


def _fail(code: int, message: str) -> NoReturn:
    """Write *message* to stderr and exit with *code* (never return)."""
    sys.stderr.write(f"{message}\n")
    raise SystemExit(code)


def _format_wall(value: Any) -> str:
    """Format ``total_wall_seconds`` as ``"<n>s"``, tolerating a non-numeric
    (hand-edited / malformed) value by falling back to its ``str()`` rather than
    crashing the whole summary on one bad footer field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{value}"
    return f"{float(value):.0f}s"


def footer_summary(metrics_path: Path) -> str:
    """Return the formatted ``run_footer`` summary line from *metrics_path*,
    failing loud on any unreadable / malformed file (see the module docstring's
    exit contract). A valid file with no ``run_footer`` record yet exits 3
    (transient) — distinguishable from corruption's exit 2."""
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
        if isinstance(obj, dict) and obj.get("type") == "run_footer":
            best = obj.get("best_valid_loss", "N/A")
            final = obj.get("final_train_loss", "N/A")
            ppl = obj.get("perplexity", "N/A")
            wall = _format_wall(obj.get("total_wall_seconds", 0))
            return f"best_valid={best}  final_train={final}  ppl={ppl}  wall={wall}"

    # Valid JSON throughout but no run_footer record: the run is still in
    # progress (the footer is appended LAST). Distinct exit code + NO stderr so
    # the shell prints the normal "(no footer yet)" instead of a warning.
    raise SystemExit(3)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(f"usage: {argv[0]} <run_metrics.jsonl>\n")
        return 64
    print(footer_summary(Path(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
