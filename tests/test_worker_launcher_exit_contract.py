"""Cross-module pin of the worker→launcher exit-code contract (GOAL §7).

``scripts/run_freeze_validloss_ci_9b.py`` (the worker) emits a small set of exit
codes that ``scripts/launch_freeze_ci_9b_full.py`` (the self-retrying launcher)
branches on via ``classify_exit_code``. The launcher cannot IMPORT those codes —
it is deliberately torch-free so it can poll a busy card without the worker's
GPU/torch dependency — so the two modules each define the five constants and the
contract between them is, without this test, enforced only by convention.

That is a silent-corruption class the per-commit honesty fixes (cost-axis /
G3-drop / atomic-write / OOM-tempfail / freeze-spec-reject / fingerprint-gate)
did not each cover in isolation: if the worker's emitted exit code drifts from
the launcher's expectation, BOTH modules pass on their own while the ASSEMBLED
contract breaks. The load-bearing case is exit 3 (IncompleteResumeError →
launcher RETRY): if the worker ever emitted a different code for it, the
launcher would classify the torn-ledger signal as FATAL, stop, and the
multi-hour full-budget verdict would lose the ledger-resume the ``--ledger``
exists to enable — a corrupt-but-green §4 verdict (GOAL §7).

The contract is pinned four ways so any drift fails loud:

* **value pins** — each constant equals its documented value (catches a
  coordinated value change on both sides at once — the one case cross-module
  equality alone cannot catch);
* **cross-module equality** — worker and launcher agree on every code (catches
  a unilateral change on one side);
* **classify mapping** — ``classify_exit_code`` routes each code to the
  documented ``Action`` (catches a launcher-side branch change);
* **source wiring** — ``main()`` returns the NAMED constant, not a bare literal,
  for each contract path (catches someone re-introducing ``return 3`` while the
  constant silently stays ``3``).
"""

from __future__ import annotations

from pathlib import Path

import scripts.run_freeze_validloss_ci_9b as worker
from scripts.launch_freeze_ci_9b_full import (
    EXIT_CUDA_DOWN,
    EXIT_DONE,
    EXIT_GPU_TEMPFAIL,
    EXIT_INCOMPLETE_RESUME,
    EXIT_OPERATOR_ERROR,
    EXIT_UNEXPECTED,
    Action,
    classify_exit_code,
)

WORKER_SOURCE = Path(worker.__file__).read_text(encoding="utf-8")

# (constant_on_each_module, documented_value, launcher_action, launcher_kind)
# — the single table both modules must agree on.
#
# The first five rows are the worker↔launcher retry contract (0/1/2/3/75). The
# sixth row is exit 78 (sysexits.h ``EX_CONFIG``): the operator-input error the
# producer emits for a missing config / malformed YAML / schema violation
# (TASK-0180/0181). The worker sources it from the leaf
# (``from src.utils.cli_errors import EXIT_OPERATOR_ERROR``), so
# ``worker.EXIT_OPERATOR_ERROR`` resolves and the same four-way pin applies.
_CONTRACT = [
    (worker.EXIT_DONE, EXIT_DONE, 0, Action.DONE, "success"),
    (worker.EXIT_GPU_TEMPFAIL, EXIT_GPU_TEMPFAIL, 75, Action.RETRY, "tempfail"),
    (worker.EXIT_INCOMPLETE_RESUME, EXIT_INCOMPLETE_RESUME, 3, Action.RETRY, "resume"),
    (worker.EXIT_CUDA_DOWN, EXIT_CUDA_DOWN, 2, Action.FATAL, "cuda_down"),
    (worker.EXIT_UNEXPECTED, EXIT_UNEXPECTED, 1, Action.FATAL, "unexpected"),
    (worker.EXIT_OPERATOR_ERROR, EXIT_OPERATOR_ERROR, 78, Action.FATAL, "operator_error"),
]


def test_contract_table_is_complete():
    """Every documented exit code is covered — a new code added to one module
    without a row here is an unguarded contract addition."""
    assert {c[0] for c in _CONTRACT} == {
        worker.EXIT_DONE, worker.EXIT_UNEXPECTED, worker.EXIT_CUDA_DOWN,
        worker.EXIT_INCOMPLETE_RESUME, worker.EXIT_GPU_TEMPFAIL,
        worker.EXIT_OPERATOR_ERROR,
    }


def test_worker_constants_pin_documented_values():
    """Value pin (worker side): each constant is the value the launcher's
    classify logic was written against. A coordinated change on both modules
    to, say, ``EXIT_INCOMPLETE_RESUME = 4`` fails here even though cross-module
    equality would still hold."""
    assert worker.EXIT_DONE == 0
    assert worker.EXIT_UNEXPECTED == 1
    assert worker.EXIT_CUDA_DOWN == 2
    assert worker.EXIT_INCOMPLETE_RESUME == 3
    assert worker.EXIT_GPU_TEMPFAIL == 75  # sysexits.h EX_TEMPFAIL
    assert worker.EXIT_OPERATOR_ERROR == 78  # sysexits.h EX_CONFIG (TASK-0182)


def test_launcher_constants_match_worker():
    """Cross-module equality: the launcher's mirror of each constant equals the
    worker's. A unilateral change on one side fails here."""
    assert EXIT_DONE == worker.EXIT_DONE
    assert EXIT_UNEXPECTED == worker.EXIT_UNEXPECTED
    assert EXIT_CUDA_DOWN == worker.EXIT_CUDA_DOWN
    assert EXIT_INCOMPLETE_RESUME == worker.EXIT_INCOMPLETE_RESUME
    assert EXIT_GPU_TEMPFAIL == worker.EXIT_GPU_TEMPFAIL
    assert EXIT_OPERATOR_ERROR == worker.EXIT_OPERATOR_ERROR


def test_classify_routes_each_code_to_its_documented_action():
    """Classify mapping: ``classify_exit_code`` routes the worker's emitted code
    to the ``Action`` the launcher's retry loop depends on. A launcher-side
    branch change fails here."""
    for _w, _l, value, action, kind in _CONTRACT:
        decision = classify_exit_code(value, resume_sleep=30, tempfail_sleep=120)
        assert decision.action is action, (
            f"exit {value} classifies as {decision.action.value}, expected "
            f"{action.value} — the launcher's retry policy no longer matches "
            "the worker's contract (GOAL §7)."
        )
        assert decision.kind == kind


def test_worker_main_returns_named_constants_not_bare_literals():
    """Source wiring: ``main()`` returns the named constant for each contract
    path, not a bare ``return 3`` etc. Without this, someone could re-introduce
    a bare literal that drifts from the (still-correct) constant, and the value
    pins above would not catch it. The IncompleteResumeError → RETRY path is the
    load-bearing one — a bare ``return 3`` there could silently become
    ``return 4`` and defeat the ledger-resume."""
    # Each contract return site must reference its named constant by name.
    assert "return EXIT_DONE" in WORKER_SOURCE, (
        "main()'s success path must `return EXIT_DONE`, not a bare `return 0` — "
        "a bare literal can drift from the constant the launcher pins."
    )
    assert "return EXIT_INCOMPLETE_RESUME" in WORKER_SOURCE, (
        "main()'s IncompleteResumeError path must `return EXIT_INCOMPLETE_RESUME`, "
        "not a bare `return 3` — drift here silently defeats the launcher's "
        "torn-ledger RETRY (GOAL §7)."
    )
    assert "return EXIT_CUDA_DOWN" in WORKER_SOURCE
    assert "return EXIT_UNEXPECTED" in WORKER_SOURCE
    assert "return EXIT_GPU_TEMPFAIL" in WORKER_SOURCE
    assert "return EXIT_OPERATOR_ERROR" in WORKER_SOURCE, (
        "main()'s operator-error path must `return EXIT_OPERATOR_ERROR`, not a "
        "bare `return 78` — a bare literal drifts from the 78 the launcher "
        "classifies as FATAL/operator_error (TASK-0182, GOAL §7)."
    )
    # And the bare literals the constants replaced must NOT remain on a contract
    # path. (`return 0` at end-of-main and the exception-handler returns are the
    # contract sites; their absence here confirms the named-constant wiring.)
    for bare in ("return 0\n", "return 1\n", "return 2\n", "return 3\n", "return 78\n"):
        assert bare not in WORKER_SOURCE, (
            f"bare `{bare.strip()}` remains in the worker — a contract exit code "
            "is emitted as a magic literal instead of its named constant; the "
            "launcher pin cannot see it (GOAL §7)."
        )
