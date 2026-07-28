"""CI-enforced FREEZE of PURPOSE.md 「次の一手」(next execution) section addendum growth.

CONTEXT
-------
AI-Hub feedback (TASK-0223..0226 cycle) bullet 3 states verbatim:

    "Document the contract-term lineage in PURPOSE.md once more, then freeze:
     with 6 terms the PURPOSE paragraph is becoming a changelog. ... stop
     adding new terms — the next move should be the operator decision, not
     term N+1."

Despite TASK-0223 declaring a "durable closeout", the loop still appended
addenda 追記(24)/(25)/(26)/(27) — i.e. a *prose* freeze was violated four
times. The recurring failure mode is that "stop adding terms" lives only in
prose, so each new iteration re-accretes an addendum. This module turns the
freeze into a CI-enforced structural invariant: the 次の一手 section's addendum
count is pinned at the frozen baseline, so a refactor/task that silently
appends term N+1 fails CI loudly instead of slipping through.

The §4 operator-decision axis this freeze guards is genuinely blocked:
recommendation == SHIP (machine-verifiable via
``scripts/section4_operator_decision.py``, exit 3 = awaiting operator call,
non-unilateral by REQ-403) and the only remaining open items — operator
``--land`` and private ``src.data`` absolute-loss (DATA/Cat-C) — are not
autonomously executable in this checkout. The canonical live state is the
machine-verifiable surface + the EARS registry ``specs/oper-decision-surface/``
(38 REQ, all 🔵); future status updates route THERE, not to a new prose
addendum here.

LEGITIMATE ESCAPE VALVE
-----------------------
A genuine NEW milestone (not §4-decision churn) may bump
``FROZEN_NEXT_STEPS_ADDENDUM_COUNT`` deliberately here and add its addendum.
That bump is a visible, reviewed act — exactly the opposite of silent
term-N+1 accretion.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PURPOSE_PATH = REPO_ROOT / "PURPOSE.md"

# Sentinel emitted by the FREEZE addendum (TASK-0227). Must appear exactly once
# in the 次の一手 section and must head the topmost (newest) addendum block.
FREEZE_SENTINEL = "OPER_DECISION_SURFACE_FROZEN_2026-07-28_T0227"

# An addendum "opener" line: a blockquote whose 【…】 bracket contains 追記.
# The FREEZE block itself uses "FREEZE" (not 追記) inside 【…】 so it is NOT
# counted — the frozen baseline counts only 追記 addenda.
ADDENDUM_OPENER_RE = re.compile(r">\s*\*\*【[^】]*追記")

# Any blockquote opener (【…】), used to locate the topmost block regardless of
# whether it is a 追記 or the FREEZE block.
ANY_OPENER_RE = re.compile(r">\s*\*\*【")

# Frozen baseline: number of 追記 addendum openers in the 次の一手 section as of
# the 2026-07-28 freeze (TASK-0227). Bump ONLY for a genuine new milestone.
FROZEN_NEXT_STEPS_ADDENDUM_COUNT = 97


def _next_steps_section() -> str:
    """Return the body of PURPOSE.md's 「次の一手（next execution）」section.

    The section runs from its ``## 次の一手`` heading to the next ``## ``
    heading or EOF. Computed dynamically so line shifts (e.g. prepending the
    FREEZE block) do not invalidate the guard.
    """
    text = PURPOSE_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## 次の一手"))
    rest = lines[start + 1 :]
    end = next((i for i, line in enumerate(rest) if line.startswith("## ")), len(rest))
    return "\n".join(rest[:end])


def _section_lines() -> list[str]:
    return _next_steps_section().splitlines()


class TestPurposeNextStepsFreeze:
    """Pin the 次の一手 addendum freeze as a CI-enforced structural invariant."""

    def test_freeze_sentinel_present_exactly_once(self) -> None:
        """The FREEZE sentinel must exist exactly once in the section.

        Removing or duplicating the freeze declaration breaks the freeze
        contract → RED.
        """
        section = _next_steps_section()
        assert section.count(FREEZE_SENTINEL) == 1, (
            f"FREEZE sentinel {FREEZE_SENTINEL!r} must appear exactly once in "
            f"the 次の一手 section; found {section.count(FREEZE_SENTINEL)}."
        )

    def test_freeze_block_is_topmost_addendum(self) -> None:
        """The FREEZE block must be the first (newest, topmost) blockquote opener.

        The section uses newest-at-top ordering, so the FREEZE declaration must
        sit above every 追記 addendum. A future iteration that prepends a new
        追記 above the freeze → RED (the freeze must stay the visible top
        declaration).
        """
        lines = _section_lines()
        openers = [line for line in lines if ANY_OPENER_RE.match(line)]
        assert openers, "次の一手 section has no blockquote addendum openers at all."
        first_opener = openers[0]
        assert FREEZE_SENTINEL in _next_steps_section(), "sentinel missing (see prior test)"
        # The sentinel lives on the FREEZE body line; the FREEZE *opener* is the
        # first opener and must carry the FREEZE marker (not 追記).
        assert "FREEZE" in first_opener, (
            "The topmost addendum opener must be the FREEZE block, but the first "
            f"opener is a 追記 (or non-FREEZE) block: {first_opener[:90]!r}."
        )

    def test_addendum_count_is_frozen(self) -> None:
        """The 追記 addendum count must equal the frozen baseline (anti-churn).

        This is the core anti-churn guard: appending OR prepending a new 追記
        (term N+1) raises the count above FROZEN_NEXT_STEPS_ADDENDUM_COUNT and
        fails CI. The §4 operator-decision status must update the
        ``specs/oper-decision-surface/`` registry + the machine-verifiable
        surface, NOT a new prose 追記 here.

        If this test is RED because you added a genuine NEW milestone (not
        §4-decision churn), bump FROZEN_NEXT_STEPS_ADDENDUM_COUNT deliberately
        in this file — that visible edit is the legitimate escape valve.
        """
        section = _next_steps_section()
        count = len(ADDENDUM_OPENER_RE.findall(section))
        assert count == FROZEN_NEXT_STEPS_ADDENDUM_COUNT, (
            f"次の一手 追記 addendum count drifted: expected frozen baseline "
            f"{FROZEN_NEXT_STEPS_ADDENDUM_COUNT}, got {count}. A new 追記 was "
            f"appended (term N+1 churn). Route §4 status to "
            f"specs/oper-decision-surface/ instead, or — for a genuine new "
            f"milestone — bump FROZEN_NEXT_STEPS_ADDENDUM_COUNT here."
        )

    def test_freeze_documents_canonical_surface_and_registry(self) -> None:
        """The FREEZE block must point readers to the canonical live state.

        So the freeze does not strand the next reader/executor: it must name
        the machine-verifiable surface (section4_operator_decision) and the EARS
        registry (specs/oper-decision-surface) as where live status lives.
        """
        section = _next_steps_section()
        assert "section4_operator_decision" in section, (
            "FREEZE block must reference the machine-verifiable surface "
            "(section4_operator_decision)."
        )
        assert "specs/oper-decision-surface" in section, (
            "FREEZE block must reference the EARS registry (specs/oper-decision-surface)."
        )

    def test_freeze_states_non_unilateral_blocking(self) -> None:
        """The FREEZE block must state the honest blocking condition.

        The freeze is honest only if it records that the §4 axis is blocked on
        a non-unilateral operator ``--land`` (not silently claiming completion).
        """
        section = _next_steps_section()
        assert "--land" in section, (
            "FREEZE block must name the operator --land decision as the blocking item."
        )
