"""Tests for ``scripts/section4_operator_decision.py`` — the machine-verifiable
§4 operator-decision surface.

The recurring AI-Hub feedback asks to "launch the 9B run and report its §4 TIES
verdict." That is stale: the verdict arc is already complete (both full-budget
deposits are citable faithful TIES) and the 9B run is architecturally
non-executable in this public mirror (``src.data`` is deliberately stripped).
This suite pins those two facts as machine-checked invariants and mutation-proves
the decision logic that consolidates them into a ship / accept-null / pivot call
— correcting the prior docs-only decision that framed PIVOT as a public-mirror
``src.data`` port when ``src.data`` is stripped here by design.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES

from scripts.section4_operator_decision import (
    HETEROGENEOUS_DEPOSIT,
    HOMOGENEOUS_DEPOSIT,
    REPO_ROOT,
    TRAIN_ENTRYPOINT_DATA_IMPORT,
    assess_section4_decision,
    main,
)


def _real_deposit(name: str) -> dict:
    with open(REPO_ROOT / name) as fh:
        return json.load(fh)


def _write_deposits(
    root: Path,
    *,
    homo: dict | None = None,
    hetero: dict | None = None,
) -> Path:
    """Materialise a fake ``tests/fixtures`` tree under ``root``.

    Defaults to byte-copies of the two real deposits so a mutation test only has
    to flip the one field it is exercising.
    """
    fixtures = root / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        (HOMOGENEOUS_DEPOSIT.split("/")[-1], homo if homo is not None else _real_deposit(HOMOGENEOUS_DEPOSIT)),
        (HETEROGENEOUS_DEPOSIT.split("/")[-1], hetero if hetero is not None else _real_deposit(HETEROGENEOUS_DEPOSIT)),
    ):
        (fixtures / name).write_text(json.dumps(payload))
    return fixtures


class TestSection4DecisionArc:
    def test_arc_complete_on_real_deposits(self):
        snap = assess_section4_decision()
        assert snap["arc_complete"] is True

    def test_both_legs_citable_faithful_ties(self):
        snap = assess_section4_decision()
        labels = {leg["label"]: leg for leg in snap["legs"]}
        for label in ("homogeneous", "heterogeneous"):
            leg = labels[label]
            assert leg["present"] is True
            assert leg["citable_as_full_section4_verdict"] is True
            assert leg["faithful"] is True
            # re-derived (not just read) — the stored floats earn TIES under the
            # deterministic bootstrap; the verdict is not painted on.
            assert leg["rederived_verdict"] == TIES
            assert leg["rederived_verdict"] == leg["recorded_verdict"]
            assert leg["seq_len"] == 1024

    def test_recommendation_is_ship_when_arc_complete_and_stripped(self):
        snap = assess_section4_decision()
        assert snap["recommendation"] == "SHIP"
        assert snap["run_executable_here"] is False


class TestPivotBranchCorrection:
    """PIVOT (absolute-loss via src.data) is private-repo-only here — the
    deliberate ``src.data`` strip makes it non-executable in this public mirror.
    This is the correction to the prior docs-only decision."""

    def test_pivot_is_private_repo_only_in_this_mirror(self):
        snap = assess_section4_decision()
        pivot = snap["branches"]["pivot"]
        assert pivot["executable_here"] is False
        assert pivot["private_repo_only"] is True

    def test_pivot_becomes_public_doable_when_src_data_present(self):
        # mutation: if src.data WERE present, PIVOT would be public-doable —
        # proves the branch keys off the strip invariant, not a hardcoded False.
        snap = assess_section4_decision(src_data_present=True)
        assert snap["branches"]["pivot"]["executable_here"] is True
        assert snap["branches"]["pivot"]["private_repo_only"] is False
        assert snap["run_executable_here"] is True
        # and the recommendation flips to FIRE_OR_EXTEND (arc complete + runnable)
        assert snap["recommendation"] == "FIRE_OR_EXTEND"

    def test_run_not_executable_due_to_deliberate_strip(self):
        snap = assess_section4_decision()
        assert snap["run_executable_here"] is False
        assert snap["src_data_status"] == "stripped_deliberate"


class TestArcIncompleteMutations:
    """Each load-bearing arc predicate is mutation-killed: breaking it must flip
    ``arc_complete`` to False and the recommendation to INCOMPLETE_ARC."""

    def test_empty_repo_root_is_incomplete(self, tmp_path):
        snap = assess_section4_decision(repo_root=tmp_path)
        assert snap["arc_complete"] is False
        assert snap["recommendation"] == "INCOMPLETE_ARC"
        assert all(not leg["present"] for leg in snap["legs"])

    def test_stale_recorded_verdict_breaks_arc(self, tmp_path):
        # faithful=false: stored verdict disagrees with the re-derived one.
        homo = _real_deposit(HOMOGENEOUS_DEPOSIT)
        homo["verdict"] = SURPASSES  # losses still re-derive to TIES → mismatch
        _write_deposits(tmp_path, homo=homo)
        snap = assess_section4_decision(repo_root=tmp_path)
        assert snap["arc_complete"] is False
        labels = {leg["label"]: leg for leg in snap["legs"]}
        assert labels["homogeneous"]["faithful"] is False
        assert labels["homogeneous"]["rederived_verdict"] == TIES
        assert labels["homogeneous"]["recorded_verdict"] == SURPASSES

    def test_non_citable_deposit_breaks_arc(self, tmp_path):
        # citable=false: a deposit that withholds the full-§4 citation claim.
        hetero = _real_deposit(HETEROGENEOUS_DEPOSIT)
        hetero["citable_as_full_section4_verdict"] = False
        _write_deposits(tmp_path, hetero=hetero)
        snap = assess_section4_decision(repo_root=tmp_path)
        assert snap["arc_complete"] is False
        labels = {leg["label"]: leg for leg in snap["legs"]}
        assert labels["heterogeneous"]["citable_as_full_section4_verdict"] is False
        # the homogeneous leg is still a clean citable faithful TIES on its own —
        # arc completeness requires BOTH legs, not just one.
        assert labels["homogeneous"]["faithful"] is True


class TestUnblockStepAndArchitecturalInvariant:
    def test_unblock_step_names_private_repo_and_deliberate_strip(self):
        snap = assess_section4_decision()
        step = snap["unblock_step"]
        assert "PRIVATE-REPO" in step
        assert "deliberately stripped" in step
        assert "scripts/prepare_data.py" in step
        assert "/home/jinno/tg-lora" in step
        assert "half-port" in step  # the explicit don't-break-the-boundary warning

    def test_unblock_step_short_form_when_src_data_present(self):
        snap = assess_section4_decision(src_data_present=True)
        assert "fire the run directly" in snap["unblock_step"]

    def test_src_data_is_deliberately_stripped_in_this_mirror(self):
        # The architectural invariant that makes the 9B run non-executable here.
        # Pinned (not fabricated): scripts/prepare_data.py documents the strip and
        # tests/test_filter_dataset.py + tests/test_dedup.py keep the interface
        # without its implementation. If this pin flips, the decision surface and
        # the PIVOT-branch logic above must be revisited together — do not just
        # delete this test.
        assert not (REPO_ROOT / "src" / "data").is_dir()
        assert not (REPO_ROOT / TRAIN_ENTRYPOINT_DATA_IMPORT).exists()
        # ...but the stripped interface is still documented in this mirror:
        assert (REPO_ROOT / "tests" / "test_filter_dataset.py").exists()
        assert (REPO_ROOT / "tests" / "test_dedup.py").exists()
        prepare = (REPO_ROOT / "scripts" / "prepare_data.py").read_text()
        assert "stripped from this public mirror" in prepare


class TestCLI:
    def test_json_snapshot(self, capsys):
        rc = main(["--json"])
        out = capsys.readouterr().out
        snap = json.loads(out)
        assert rc == 0
        assert snap["arc_complete"] is True
        assert snap["recommendation"] == "SHIP"

    def test_exit_code_tracks_arc(self, tmp_path):
        # arc complete (real repo) → exit 0; arc incomplete (empty root) → exit 2
        assert main([]) == 0
        assert main(["--repo-root", str(tmp_path)]) == 2

    def test_help_launches_as_module(self):
        # the canary every scripts.* CLI keeps: ``-m`` launch + --help works with
        # only the repo root on sys.path (no PYTHONPATH wrapper).
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.section4_operator_decision", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        assert proc.returncode == 0
        assert "ship" in proc.stdout
        assert "pivot" in proc.stdout

    def test_human_readable_mentions_all_three_branches(self, capsys):
        main([])
        out = capsys.readouterr().out
        for branch in ("ship", "accept_null", "pivot"):
            assert branch in out
        assert "RECOMMENDATION: SHIP" in out
