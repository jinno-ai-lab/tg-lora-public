#!/usr/bin/env python3
"""Loop halt guard — makes "stop" structural for the §4 / MS-008 axis.

AI-Hub feedback rejected the 4th consecutive halt-doc (TASK-0236/0237/0238/0240)
with the diagnosis: *a genuine halt emits no commit, yet the halt-doc's own
existence contradicts the halt it claims*. The recurring peripheral-plumbing
churn this loop defaults to when GPU-blocked is exactly "emit another
timestamped awaiting-ratification markdown restating the same verdict".

This guard is the single machine-checkable artifact the loop (or operator)
consults **before emitting any axis commit**. When the axis is
``awaiting_ratification`` AND no external trigger has arrived, the loop MUST
produce nothing — not another halt-doc. Committing this guard does NOT
self-unblock: the guard's own files (this script, the tracker JSON, its tests)
are neither a 9B deposit, nor a closeout anchor, nor an operator axis-open
signal, so they trip none of the three triggers. That property is proven by
``test_guard_commit_does_not_self_unblock`` — which is how this commit, unlike a
halt-doc, is consistent with halting.

EXIT CODES
    0   PROCEED — the axis is NOT awaiting-ratification, OR a trigger fired.
    77  SKIP    — awaiting-ratification with no witnessed external trigger;
                  the loop should idle and emit no axis commit.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

EXIT_PROCEED = 0
EXIT_SKIP = 77

STATE_FILENAME = "loop_axis_state.json"
AWAITING = "awaiting_ratification"

# A 9B §4 deposit artifact (reduced- or full-budget deposit, or its run-log
# witness). The only honest way to grow this set is a real operator-side 9B run
# (Cat-C); a fabricated deposit trips the separate replay-honesty / sha256 gate.
NINE_B_DEPOSIT_GLOB = "freeze_validloss_ci_9b*.json"
# Operator-infra closeout anchor; absent on this mirror until the operator
# drafts + approves the MS-008 publishable-negative closeout.
CLOSEOUT_GLOBS = (
    "reports/close-the-loop/close_the_loop_funnel_go_nogo_*.json",
    "reports/close-the-loop/close_the_loop_funnel_go_nogo_*.md",
)


@dataclass(frozen=True)
class SkipDecision:
    """The guard's verdict on whether the loop should idle this iteration."""

    skip: bool
    reason: str
    status: str
    active_triggers: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def decide_skip(status: str, witnesses: dict[str, bool]) -> SkipDecision:
    """Pure decision core.

    ``witnesses`` maps each trigger name to whether it has fired. SKIP holds iff
    the axis is awaiting-ratification AND no witness has fired.
    """
    if status != AWAITING:
        return SkipDecision(
            skip=False,
            reason=f"status={status!r} (not {AWAITING})",
            status=status,
            active_triggers=[],
        )
    active = sorted(name for name, fired in witnesses.items() if fired)
    if active:
        return SkipDecision(
            skip=False,
            reason=f"trigger(s) witnessed: {active}",
            status=status,
            active_triggers=active,
        )
    return SkipDecision(
        skip=True,
        reason=f"{AWAITING}; no external trigger witnessed",
        status=status,
        active_triggers=active,
    )


def _count_nine_b_deposits(fixtures_dir: Path) -> int:
    if not fixtures_dir.is_dir():
        return 0
    return len(list(fixtures_dir.glob(NINE_B_DEPOSIT_GLOB)))


def _closeout_anchor_present(repo_root: Path) -> bool:
    return any(list(repo_root.glob(glob)) for glob in CLOSEOUT_GLOBS)


def compute_witnesses(repo_root: Path, state: dict) -> dict[str, bool]:
    """Map each external trigger to whether it has fired since entry.

    Each witness is a real, hard-to-forge signal — not agent-editable prose:
      * 9b_leg_fired       — a new committed 9B deposit artifact appeared
                             (count > the recorded baseline).
      * closeout_approved  — the operator closeout anchor now exists on mirror.
      * new_ms_axis_opened — the operator flipped the axis-open signal in the
                             tracker (option (b): open MS-009).
    """
    baseline = state.get("baseline", {})
    fixtures_dir = repo_root / "tests" / "fixtures"
    return {
        "9b_leg_fired": _count_nine_b_deposits(fixtures_dir)
        > int(baseline.get("nine_b_deposit_count", 0)),
        "closeout_approved": _closeout_anchor_present(repo_root),
        "new_ms_axis_opened": bool(
            state.get("operator_signals", {}).get("new_ms_axis_opened")
        ),
    }


def load_state(repo_root: Path) -> dict | None:
    tracker = repo_root / STATE_FILENAME
    if not tracker.is_file():
        return None
    return json.loads(tracker.read_text(encoding="utf-8"))


def should_skip(repo_root: Path) -> SkipDecision:
    """The repo-facing entry point. Permissive when no tracker is present."""
    state = load_state(repo_root)
    if state is None:
        return SkipDecision(
            skip=False,
            reason=f"no {STATE_FILENAME} — guard permissive",
            status="(none)",
            active_triggers=[],
        )
    return decide_skip(
        state.get("status", ""),
        compute_witnesses(repo_root, state),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Halt guard for the §4 / MS-008 axis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        type=Path,
        help="Repository root containing loop_axis_state.json (default: CWD).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the decision as JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    decision = should_skip(args.repo_root.resolve())
    if args.json:
        print(json.dumps(decision.as_dict(), indent=2, ensure_ascii=False))
    else:
        tag = "SKIP (halt — produce nothing)" if decision.skip else "PROCEED"
        print(f"[loop-halt-guard] {tag}")
        print(f"  status:          {decision.status}")
        print(f"  active_triggers: {decision.active_triggers or '(none)'}")
        print(f"  reason:          {decision.reason}")
    return EXIT_SKIP if decision.skip else EXIT_PROCEED


if __name__ == "__main__":
    sys.exit(main())
