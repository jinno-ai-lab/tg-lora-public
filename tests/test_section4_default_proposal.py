"""Pin ``specs/oper-decision-surface/default-change-proposal.md`` — the post-SHIP
operator hand-off for the ``progressive_freeze_enabled`` default — against LIVE
repo state.

The §4 verdict landed SHIP ("adopt as a quality-preservation method") on
2026-07-29 (``section4_landed_decision.json``), but that landing changed **no
default**: the schema default is still ``False`` and the mainline configs still
don't set the flag, so ``make train-tg-lora`` runs freeze-OFF. The proposal
document hands that residual product decision to the operator (non-unilateral,
per ``specs/oper-decision-surface/requirements.md`` REQ-403's discipline), and
this test refuses to let its factual basis drift:

1. the §1 "current default" claims are live-verified (schema line, mainline
   config key absence, landed SHIP record, in-vivo cost-null test existence);
2. the quoted §4 numbers match the LIVE ``assess_section4_decision()`` snapshot
   — the same live re-derivation ``tests/test_section4_terminal_verdict.py``
   pins the terminal-verdict doc against;
3. the proposal presents the D1/D2/D3 options with exact paths/commands and
   states the non-unilateral + loop-axis-orthogonality clauses.

Mutation pins: flipping the schema default, adding the flag to a mainline
config, or editing any quoted number WITHOUT updating the proposal turns the
matching assertion RED — a default change must carry its proposal update in the
same commit, never silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.section4_operator_decision import assess_section4_decision

REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = (
    REPO_ROOT / "specs" / "oper-decision-surface" / "default-change-proposal.md"
)
SCHEMA = REPO_ROOT / "src" / "training" / "config_schema.py"
MAINLINE_CONFIGS = [
    REPO_ROOT / "configs" / "9b_tg_lora.yaml",
    REPO_ROOT / "configs" / "9b_tg_lora_paper_poc.yaml",
]
LANDED = REPO_ROOT / "section4_landed_decision.json"
INVIVO_TEST = REPO_ROOT / "tests" / "test_progressive_freeze_invivo.py"


def _proposal() -> str:
    return PROPOSAL.read_text(encoding="utf-8")


def test_current_default_claims_match_live_state() -> None:
    """The proposal's §1 table states the live default: schema False, both
    mainline configs unset (freeze-OFF mainline). Mutation: flipping the schema
    default or adding the key to a mainline config without updating the
    proposal makes the corresponding assertion RED."""
    schema = SCHEMA.read_text(encoding="utf-8")
    assert "progressive_freeze_enabled: bool = False" in schema
    for cfg in MAINLINE_CONFIGS:
        assert "progressive_freeze_enabled" not in cfg.read_text(encoding="utf-8")
    text = _proposal()
    assert "progressive_freeze_enabled: bool = False" in text
    assert "src/training/config_schema.py" in text
    for cfg in MAINLINE_CONFIGS:
        assert f"configs/{cfg.name}" in text


def test_ship_landing_record_backs_the_proposal() -> None:
    """The proposal is the follow-through of a real landed SHIP record whose
    deposits exist. Mutation: deleting the landed record, a deposit fixture, or
    the proposal's citation of them -> RED."""
    record = json.loads(LANDED.read_text(encoding="utf-8"))
    assert record["branch"] == "ship"
    assert record["landed"] is True
    text = _proposal()
    assert "section4_landed_decision.json" in text
    for deposit in record["deposits"]:
        assert (REPO_ROOT / deposit).is_file(), deposit
        assert Path(deposit).name in text, deposit


def test_quoted_means_match_live_snapshot() -> None:
    """The proposal's §2 numbers (4 headline means + 2 baselines + both CI
    bounds) must equal the live snapshot's, formatted to 4 dp. Mutation:
    hand-editing any number in the proposal prose (or in a deposit) makes the
    formatted string absent -> RED."""
    text = _proposal()
    snap = assess_section4_decision(verdict_worker_status="executable")
    legs = {leg["label"]: leg for leg in snap["legs"]}
    for label in ("homogeneous", "heterogeneous"):
        leg = legs[label]
        assert f"{leg['candidate_mean']:.4f}" in text, (label, "candidate")
        assert f"{leg['surrogate_mean']:.4f}" in text, (label, "surrogate")
        assert f"{leg['baseline_baseline_mean']:.4f}" in text, (label, "baseline")
        assert f"{leg['ci_lower']:.4f}" in text, (label, "ci_lower")
        assert f"{leg['ci_upper']:.4f}" in text, (label, "ci_upper")


def test_cost_null_citation_is_real() -> None:
    """The proposal's cost-axis claim (realized Level-1 reduction = 0.0) cites
    the in-vivo test by name, and that test actually exists. Mutation: renaming
    the in-vivo test or dropping the citation -> RED."""
    text = _proposal()
    assert "0.0" in text
    assert "test_level1_freeze_only_cuts_no_backward_in_vivo" in text
    assert (
        "def test_level1_freeze_only_cuts_no_backward_in_vivo"
        in INVIVO_TEST.read_text(encoding="utf-8")
    )


def test_proposal_presents_three_landing_options_non_unilaterally() -> None:
    """The decision package is complete (D1/D2/D3 + exact surfaces + landing
    verification) and scopes itself as a proposal the operator lands — never a
    unilateral default flip, and orthogonal to the loop-axis ratification
    surface. Mutation: dropping an option, a cited gate, or either clause ->
    RED."""
    text = _proposal()
    for marker in ("D1", "D2", "D3"):
        assert marker in text
    assert "tests/test_config_launchability_gate.py" in text
    assert "freeze_mutual_exclusion" in text
    assert "non-unilateral" in text
    assert "loop_axis_state.json" in text
