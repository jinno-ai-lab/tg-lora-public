"""Pin ``docs/section4_terminal_verdict.md`` — the citable self-contained §7
terminal-verdict artifact — against LIVE re-derivation + committed fixtures.

The §4 research arc (does Progressive Freezing beat the random-order surrogate
at 9B target scale?) is **answered**: both citable full-budget legs re-derive to
faithful, evidence-intact **TIES**, and the operator decision **SHIP** is landed.
This is the terminal state the loop kept churning past. ``docs/section4_terminal_verdict.md``
formalizes that terminal state as a single citable artifact and — per GOAL §7's
discipline ("don't trust a predictor until it lands on loss") — this test refuses
to let the prose drift from reality.

It does NOT re-derive the verdict from scratch (that is the saturated decision
surface's job, pinned by ``tests/test_section4_operator_decision.py``). Instead it
cross-checks, GPU-free, that:

1. the report exists and states the structural conclusions (TIES both legs, SHIP
   landed, quality-vs-full-backprop SURPASSES, §7 self-contained, the (B)
   private-src.data hand-off, and the go/no-go);
2. the **live** ``assess_section4_decision()`` snapshot agrees — ``arc_complete``,
   both legs faithful + ``rederived_verdict == TIES`` + ``citable`` + landed
   ``ship`` + baseline ``SURPASSES``;
3. the report's stated means match the live snapshot's (so the prose cannot
   silently rewrite a number);
4. the **§7 self-containedness** claim is real: the offline ``--data-file`` rail
   deposited a real 9B A/B from a LOCAL dump (ledger ``data_file`` stamp), on the
   SAME model + SAME public dataset as the full verdicts, and the verdict worker
   is ``src.data``-free (the AST invariant test this report cites actually exists).

Mutation pins: each assertion names the single edit that turns it RED, so a future
"polish" pass cannot soften the terminal verdict back into prose without failing CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.section4_operator_decision import assess_section4_decision

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "section4_terminal_verdict.md"
FIX = REPO_ROOT / "tests" / "fixtures"
HOMOG_DEPOSIT = FIX / "freeze_validloss_ci_9b_full.json"
HETEROG_DEPOSIT = FIX / "freeze_validloss_ci_9b_full_heterogeneous.json"
DATAFILE_REPRO_DEPOSIT = FIX / "freeze_validloss_ci_9b_datafile_repro.json"
DATAFILE_REPRO_LEDGER = FIX / "freeze_validloss_ci_9b_datafile_repro_ledger.jsonl"
SELF_CONTAINED_TEST = REPO_ROOT / "tests" / "test_9b_verdict_producer_self_contained.py"


def _report() -> str:
    return REPORT.read_text(encoding="utf-8")


def test_report_exists_and_states_the_terminal_verdict() -> None:
    """The citable artifact exists and carries the structural conclusions — not
    a number a future edit can quietly drop. Mutation: deleting any cited
    conclusion marker from the report turns the matching assertion RED."""
    text = _report()
    # The verdict itself.
    assert "SHIP" in text                       # mutation: weaken to "INCONCLUSIVE" -> RED
    assert text.count("TIES") >= 2              # BOTH legs must read TIES (mutation: flip one -> RED)
    assert "SURPASSES" in text                  # P1 品質保持 vs full backprop (mutation: drop -> RED)
    # The §7 self-containedness claim is named (citable != reproducible, closed).
    assert "§7" in text and ("自己完結" in text or "self-contained" in text.lower())
    # The (B) hand-off + go/no-go are explicit (the terminal state, not a block).
    assert "(B)" in text and "private" in text.lower()           # private-src.data hand-off
    assert "go/no-go" in text.lower() or "Go/no-go" in text      # arc-continuation decision
    # Provenance points back to this test (the doc cites its own pin).
    assert "test_section4_terminal_verdict.py" in text


def test_live_snapshot_agrees_arc_complete_ship_landed() -> None:
    """The report's headline claim — arc complete, SHIP landed, both legs faithful
    TIES beating full-backprop on quality — is checked against the LIVE decision
    snapshot (GPU-free re-derivation of the deposited samples), not a frozen
    string. ``verdict_worker_status='executable'`` is the truthful architectural
    status (the worker imports no src.data; TASK-0239 fired it here), injected so
    the test never spawns an interpreter or depends on a torch install."""
    snap = assess_section4_decision(verdict_worker_status="executable")
    assert snap["arc_complete"] is True                      # mutation: unland/remove a leg -> RED
    assert snap["landed_decision"]["branch"] == "ship"       # mutation: re-land accept_null -> RED
    legs = {leg["label"]: leg for leg in snap["legs"]}
    for label in ("homogeneous", "heterogeneous"):
        leg = legs[label]
        assert leg["present"] is True
        assert leg["citable_as_full_section4_verdict"] is True   # mutation: mark reduced-budget -> RED
        assert leg["faithful"] is True                           # mutation: tamper a deposit byte -> RED
        assert leg["rederived_verdict"] == "TIES"                # mutation: edit deposited losses -> RED
        assert leg["recorded_verdict"] == "TIES"
        assert leg["proxy_scale"] is False                       # target scale, not proxy (mutation: proxy -> RED)
        # P1 品質保持: candidate SURPASSES the full-backprop baseline on both legs.
        assert leg["baseline_present"] is True
        assert leg["baseline_verdict"] == "SURPASSES"            # mutation: drop baseline arm -> RED


def test_report_stated_means_match_the_live_snapshot() -> None:
    """The report's four headline means (cand/surr × 2 legs) must equal the live
    snapshot's, formatted to 4 dp. Mutation: hand-editing any mean in the report
    prose (or in a deposit) makes the formatted string absent -> RED — the doc
    cannot drift from the re-derived numbers."""
    text = _report()
    snap = assess_section4_decision(verdict_worker_status="executable")
    legs = {leg["label"]: leg for leg in snap["legs"]}
    for label in ("homogeneous", "heterogeneous"):
        leg = legs[label]
        assert f"{leg['candidate_mean']:.4f}" in text, (label, leg["candidate_mean"])
        assert f"{leg['surrogate_mean']:.4f}" in text, (label, leg["surrogate_mean"])
    # The full-backprop baseline means are also cited (P1 品質保持 SURPASSES basis).
    assert f"{legs['homogeneous']['baseline_baseline_mean']:.4f}" in text
    assert f"{legs['heterogeneous']['baseline_baseline_mean']:.4f}" in text


def test_section7_offline_datafile_rail_is_proven_self_contained() -> None:
    """The §7 'citable != reproducible' gap is closed IN this public mirror: the
    offline ``--data-file`` rail deposited a real 9B A/B from a LOCAL Dolly dump
    (no network, no private src.data), on the SAME model + SAME public dataset as
    the full verdicts — so the verdict is offline-self-contained-reproducible.
    Mutation: regenerating the repro ledger via the streaming path drops the
    ``data_file`` fingerprint key -> RED."""
    # The fired offline-rail deposit exists and is real target-scale (not a stub).
    repro = json.loads(DATAFILE_REPRO_DEPOSIT.read_text())
    assert repro["citable_as_target_scale"] is True
    assert repro["proxy_scale"] is False
    assert repro["model"] == "Qwen/Qwen3.5-9B"
    # The ledger header fingerprints a LOCAL dump (offline-rail stamp). The
    # streaming-path ledgers carry NO ``data_file`` key — its presence is the
    # load-bearing proof the run ingested a local file, not a network stream.
    with DATAFILE_REPRO_LEDGER.open() as fh:
        header = json.loads(fh.readline())
    fp = header["fingerprint"]
    data_file = fp.get("data_file")
    assert isinstance(data_file, str) and data_file.endswith(".jsonl")  # mutation: streaming -> key absent -> RED
    # Same model + same public dataset as the full verdicts => the verdict is
    # reproducible via this same offline rail.
    for full_deposit in (HOMOG_DEPOSIT, HETEROG_DEPOSIT):
        full = json.loads(full_deposit.read_text())
        assert full["model"] == repro["model"]
        assert full["dataset"] == repro["dataset"] == "databricks/databricks-dolly-15k"
    # The verdict worker's src.data-freedom (architectural self-containedness)
    # is AST-pinned by the test this report cites — assert it exists so the
    # report's cross-reference is never left dangling.
    assert SELF_CONTAINED_TEST.is_file()


def test_report_foregrounds_level1_realized_cost_reduction_is_zero() -> None:
    """The §4 research question (GOAL §3.1) has TWO axes — quality (loss) and
    cost (realized backward reduction). The verdict cleanly answers quality
    (SURPASSES vs full backprop; TIES vs surrogate), but the COST axis returns
    realized 0.0 at the shipped Level-1 prod path: a Level-1 freeze stops the
    weight grad yet still propagates the activation gradient, so no backward
    traversal is elided (in-vivo-verified). This test refuses to let the citable
    artifact bury that null under a "cost-reduction win" headline — the §7 risk
    this repo exists to prevent ("don't let a predictor's promise outrun its
    measured landing", applied to the cost claim the SHIP framing implies).

    Cross-checked GPU-free against (1) the report prose stating the Level-1
    realized reduction is 0.0, (2) the LIVE deposited samples agreeing at 0.0,
    and (3) the in-vivo evidence test that PROVES ~0 existing (so the claim is
    never left as a bare assertion). Mutation: reverting §1 to a bare
    "品質を保持したコスト削減" headline while dropping the realized=0.0 caveat ->
    marker RED; editing a deposit's realized_reduction_rate off 0.0 -> live
    cross-check RED; deleting the cited in-vivo test -> provenance RED."""
    text = _report()
    # The verdict states the Level-1 realized backward reduction is 0.0 (a cost
    # caveat foregrounding the null), not a "cost-reduction win" headline that
    # outruns the in-vivo measurement.
    assert "実現 backward 削減 = 0.0" in text          # mutation: drop the cost-null caveat -> RED
    assert "realized_reduction_rate = 0.0" in text      # the deposit field the claim rests on
    # The LIVE deposited samples agree: both full-budget legs realize 0.0 at the
    # Level-1 prod path (the §4 SHIP target) — not a hand-stated prose number.
    for deposit_path in (HOMOG_DEPOSIT, HETEROG_DEPOSIT):
        deposit = json.loads(deposit_path.read_text())
        assert deposit["candidate_cost_reduction"]["realized_reduction_rate"] == 0.0
    # The in-vivo evidence that PROVES Level-1 realizes ~0 (not just asserts it)
    # is the test this report cites — assert it exists AND carries the assertion,
    # so the cross-reference is never dangling, exactly like the §7 self-contained
    # provenance check above.
    invivo = REPO_ROOT / "tests" / "test_progressive_freeze_invivo.py"
    assert invivo.is_file()
    assert "test_level1_freeze_only_cuts_no_backward_in_vivo" in invivo.read_text()
