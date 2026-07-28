"""Tests for ``scripts/section4_operator_decision.py`` — the machine-verifiable
§4 operator-decision surface.

The recurring AI-Hub feedback asks to "launch the 9B run and report its §4 TIES
verdict." That is stale: the verdict arc is already complete (both full-budget
deposits are citable faithful TIES) AND the §4 verdict run uses
``run_freeze_validloss_ci_9b`` (public Dolly — NO ``src.data``), so it already
fired here and produced those deposits. This suite pins those facts as
machine-checked invariants and mutation-proves the decision logic that
consolidates them into a ship / accept-null / pivot call — including the
correction that ``run_executable_here`` keys off the verdict worker (public
Dolly) and NOT the recover.py ``--rerun`` / ``train_tg_lora`` path (src.data).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES

from scripts.section4_operator_decision import (
    EXIT_AWAITING_DECISION,
    EXIT_LAND_INVALID,
    HETEROGENEOUS_DEPOSIT,
    HOMOGENEOUS_DEPOSIT,
    LANDING_RECORD_REL,
    RECOVER_RERUN_ENTRYPOINT,
    REPO_ROOT,
    VALID_LAND_BRANCHES,
    VERDICT_WORKER_MODULE,
    _blocking_prompt,
    _land_decision,
    _load_landed_decision,
    _probe_verdict_worker,
    _quality_preservation_clause,
    assess_section4_decision,
    format_decision,
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
        (
            HOMOGENEOUS_DEPOSIT.split("/")[-1],
            homo if homo is not None else _real_deposit(HOMOGENEOUS_DEPOSIT),
        ),
        (
            HETEROGENEOUS_DEPOSIT.split("/")[-1],
            hetero if hetero is not None else _real_deposit(HETEROGENEOUS_DEPOSIT),
        ),
    ):
        (fixtures / name).write_text(json.dumps(payload))
    return fixtures


def _fake_proc(returncode: int, stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")


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

    def test_recommendation_is_ship_when_arc_complete(self):
        # A complete arc ⇒ the verdict is DONE ⇒ SHIP regardless of whether the
        # run could be re-fired (re-firing reproduces TIES).
        snap = assess_section4_decision()
        assert snap["recommendation"] == "SHIP"


class TestVerdictWorkerExecutability:
    """THE CORE CORRECTION: ``run_executable_here`` keys off the §4 verdict
    worker (``run_freeze_validloss_ci_9b``, public Dolly — no ``src.data``), NOT
    the recover.py ``--rerun`` / ``train_tg_lora`` path. The worker is executable
    here even though ``src.data`` is stripped; the prior surface conflated the
    two paths and so wrongly reported the verdict run as non-executable."""

    def test_run_executable_keys_off_verdict_worker_not_train_tg_lora(self):
        # The verdict worker is executable; src.data (the recover path's dep) is
        # stripped. If run_executable_here keyed off src.data (the old bug), this
        # would be False.
        snap = assess_section4_decision(verdict_worker_status="executable")
        assert snap["run_executable_here"] is True
        assert snap["verdict_worker_status"] == "executable"
        # ...and src.data IS stripped — but that blocks the recover path, a
        # SEPARATE field, not the verdict run:
        assert snap["recover_rerun_blocked_by_src_data"] is True
        assert snap["src_data_status"] == "stripped_deliberate"

    def test_real_checkout_worker_has_no_architectural_block(self):
        # On the real checkout the worker either imports (torch present) or fails
        # only on a transient factor (torch absent) — never on a stripped src.*
        # dep. (Pinned, not fabricated: the verdict worker imports no src.data.)
        snap = assess_section4_decision()
        assert snap["verdict_worker_status"] in {"executable", "transient_block"}
        assert snap["run_executable_here"] is True
        assert (REPO_ROOT / "scripts" / "run_freeze_validloss_ci_9b.py").exists()

    def test_architectural_block_makes_run_non_executable(self):
        # A stripped src.* dep is the only status that makes the verdict run
        # architecturally non-executable.
        snap = assess_section4_decision(verdict_worker_status="architectural_block")
        assert snap["run_executable_here"] is False
        assert snap["verdict_worker_status"] == "architectural_block"

    def test_transient_block_is_still_architecturally_executable(self):
        # A missing torch is a transient runtime factor, NOT an architectural
        # block — a torch-free probe must not masquerade as non-executable.
        snap = assess_section4_decision(verdict_worker_status="transient_block")
        assert snap["run_executable_here"] is True


class TestProbeClassification:
    """``_probe_verdict_worker`` classifies the subprocess import result so a
    transient factor (torch) cannot be misread as an architectural block."""

    def test_probe_executable_when_import_succeeds(self):
        status, reason = _probe_verdict_worker(runner=lambda cmd, **kw: _fake_proc(0))
        assert status == "executable"
        assert "public Dolly" in reason

    def test_probe_architectural_block_on_src_missing(self):
        status, _ = _probe_verdict_worker(
            runner=lambda cmd, **kw: _fake_proc(
                1, "ModuleNotFoundError: No module named 'src.data'"
            )
        )
        assert status == "architectural_block"

    def test_probe_transient_block_on_torch_missing(self):
        status, _ = _probe_verdict_worker(
            runner=lambda cmd, **kw: _fake_proc(
                1, "ModuleNotFoundError: No module named 'torch'"
            )
        )
        assert status == "transient_block"


class TestRecommendationLogic:
    """The recommendation keys arc-completeness FIRST: a complete arc ⇒ SHIP
    (verdict done) regardless of executability. The prior logic let
    executability flip a complete arc to FIRE_OR_EXTEND, which was wrong for a
    done verdict."""

    def test_arc_complete_ships_regardless_of_executability(self):
        # Even when the worker IS executable (the run could be re-fired), a
        # COMPLETE arc still ships — the verdict is already banked and re-firing
        # reproduces TIES. (mutation: under the prior logic arc_complete +
        # executable → FIRE_OR_EXTEND, NOT SHIP — so this assertion kills that.)
        snap = assess_section4_decision(verdict_worker_status="executable")
        assert snap["arc_complete"] is True
        assert snap["run_executable_here"] is True  # worker executable ...
        assert snap["recommendation"] == "SHIP"  # ... yet SHIP (verdict done)

    def test_arc_incomplete_and_executable_fires_or_extends(self, tmp_path):
        snap = assess_section4_decision(
            repo_root=str(tmp_path), verdict_worker_status="executable"
        )
        assert snap["arc_complete"] is False
        assert snap["recommendation"] == "FIRE_OR_EXTEND"
        assert "freeze-validloss-ci-9b-full" in snap["rationale"]

    def test_arc_incomplete_and_architecturally_blocked_is_incomplete(self, tmp_path):
        snap = assess_section4_decision(
            repo_root=str(tmp_path), verdict_worker_status="architectural_block"
        )
        assert snap["arc_complete"] is False
        assert snap["recommendation"] == "INCOMPLETE_ARC"

    def test_arc_incomplete_and_transient_block_still_fires_or_extends(self, tmp_path):
        # transient (torch) ≠ architectural: the operator could fire after
        # resolving the transient factor, so it's FIRE_OR_EXTEND, not INCOMPLETE.
        snap = assess_section4_decision(
            repo_root=str(tmp_path), verdict_worker_status="transient_block"
        )
        assert snap["recommendation"] == "FIRE_OR_EXTEND"
        assert "transient factor" in snap["rationale"]


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

    def test_recover_rerun_blocked_by_src_data_strip(self):
        # The recover.py --rerun / train_tg_lora path IS src.data-blocked here —
        # a real block, but on a DIFFERENT path than the (executable) verdict run.
        snap = assess_section4_decision()
        assert snap["recover_rerun_blocked_by_src_data"] is True
        assert snap["src_data_status"] == "stripped_deliberate"


class TestArcIncompleteMutations:
    """Each load-bearing arc predicate is mutation-killed: breaking it must flip
    ``arc_complete`` to False (the recommendation then depends on executability)."""

    def test_stale_recorded_verdict_breaks_arc(self, tmp_path):
        # faithful=false: stored verdict disagrees with the re-derived one.
        homo = _real_deposit(HOMOGENEOUS_DEPOSIT)
        homo["verdict"] = SURPASSES  # losses still re-derive to TIES → mismatch
        _write_deposits(tmp_path, homo=homo)
        snap = assess_section4_decision(repo_root=str(tmp_path))
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
        snap = assess_section4_decision(repo_root=str(tmp_path))
        assert snap["arc_complete"] is False
        labels = {leg["label"]: leg for leg in snap["legs"]}
        assert labels["heterogeneous"]["citable_as_full_section4_verdict"] is False
        # the homogeneous leg is still a clean citable faithful TIES on its own —
        # arc completeness requires BOTH legs, not just one.
        assert labels["homogeneous"]["faithful"] is True

    def test_missing_deposit_breaks_arc(self, tmp_path):
        snap = assess_section4_decision(repo_root=str(tmp_path))
        assert snap["arc_complete"] is False
        assert all(not leg["present"] for leg in snap["legs"])


class TestQualityPreservationAxis:
    """GOAL §4-247 / constitution P1 品質保持 — "does Progressive Freezing
    preserve (or surpass) full-backprop quality?"

    The committed homogeneous deposit already fired a full-backprop baseline
    arm (``n_baseline=3``) whose verdict is ``SURPASSES`` (candidate 1.6947 <
    baseline 1.8794): freezing does NOT cost quality — it beats full backprop.
    That is a MEASURED, public-Dolly, already-fired result DISTINCT from the
    PIVOT axis (absolute-loss vs the private-repo PRODUCTION baseline, Cat-C).
    Surfacing it — rather than burying it under a blanket "absolute-loss
    remains the only open axis" — is constitution P0 科学誠実性: the inverse of
    "don't conclude while unmeasured" is "don't leave MEASURED evidence
    unreported" (this repo's surface-don't-swallow kernel).
    """

    def test_homogeneous_baseline_verdict_is_surfaced_per_leg(self):
        snap = assess_section4_decision()
        homo = next(leg for leg in snap["legs"] if leg["label"] == "homogeneous")
        assert homo["baseline_present"] is True
        assert homo["baseline_verdict"] == SURPASSES
        assert homo["n_baseline"] == 3
        # candidate (progressive freeze) BEATS the full-backprop baseline ⇒ P1 met
        assert homo["baseline_candidate_mean"] < homo["baseline_baseline_mean"]

    def test_heterogeneous_baseline_verdict_is_surfaced_per_leg(self):
        snap = assess_section4_decision()
        hetero = next(leg for leg in snap["legs"] if leg["label"] == "heterogeneous")
        # The full-backprop baseline arm WAS fired for the heterogeneous leg too
        # (harvested 2026-07-27 from the full-12 re-run). candidate (progressive
        # freeze, output-first) BEATS the full-backprop baseline on the asymmetric
        # per-layer-rank adapter ⇒ P1 品質保持 met for BOTH legs, not just homogeneous.
        assert hetero["baseline_present"] is True
        assert hetero["n_baseline"] == 3
        assert hetero["baseline_verdict"] == SURPASSES
        assert hetero["baseline_candidate_mean"] < hetero["baseline_baseline_mean"]

    def test_quality_preservation_summary_is_keyed_per_leg(self):
        snap = assess_section4_decision()
        qp = snap["quality_preservation"]
        assert set(qp) == {"homogeneous", "heterogeneous"}
        assert qp["homogeneous"]["answered"] is True
        assert qp["homogeneous"]["verdict"] == SURPASSES
        assert qp["homogeneous"]["candidate_mean"] < qp["homogeneous"]["baseline_mean"]
        assert qp["heterogeneous"]["answered"] is True
        assert qp["heterogeneous"]["verdict"] == SURPASSES
        assert (
            qp["heterogeneous"]["candidate_mean"] < qp["heterogeneous"]["baseline_mean"]
        )

    def test_format_decision_renders_the_quality_preservation_axis(self):
        out = format_decision(assess_section4_decision())
        assert "GOAL §4-247" in out
        assert "P1 品質保持" in out
        # BOTH legs now answered (heterogeneous harvested 2026-07-27): the axis
        # renders two full-backprop SURPASSES verdicts and NO UNANSWERED gap.
        assert "SURPASSES" in out
        assert "UNANSWERED" not in out

    def test_extraction_reads_the_deposit_field_not_a_hardcoded_default(self, tmp_path):
        # Mutation guard: strip the baseline arm from a real homogeneous deposit
        # ⇒ the surface must report it UNANSWERED (proving it reads the actual
        # ``baseline`` field, not a hardcoded "homogeneous always SURPASSES").
        homo = _real_deposit(HOMOGENEOUS_DEPOSIT)
        homo["n_baseline"] = 0
        homo["baseline"] = None
        _write_deposits(tmp_path, homo=homo)
        snap = assess_section4_decision(
            repo_root=str(tmp_path), verdict_worker_status="executable"
        )
        homo_leg = next(leg for leg in snap["legs"] if leg["label"] == "homogeneous")
        assert homo_leg["baseline_present"] is False
        assert homo_leg["baseline_verdict"] is None
        assert snap["quality_preservation"]["homogeneous"]["answered"] is False

    def test_ship_rationale_disambiguates_the_two_absolute_loss_senses(self):
        # The old rationale claimed absolute-loss "remains the ONLY open axis",
        # burying the already-answered full-backprop comparison. The fix separates
        # PIVOT (private-repo production baseline, Cat-C) from the GOAL §4-247
        # full-backprop axis (surfaced in ``quality_preservation``).
        rationale = assess_section4_decision()["rationale"]
        assert "only open axis" not in rationale

    def test_ship_rationale_derives_quality_clause_from_surfaced_evidence(self):
        # P0 科学誠実性: the SHIP rationale's per-leg P1 品質保持 clause must be
        # DERIVED from the surfaced ``quality_preservation`` evidence, not
        # hardcoded — otherwise the recommendation could contradict the very
        # evidence it cites. After the 2026-07-27 heterogeneous baseline harvest,
        # BOTH legs are answered (SURPASSES); the derived clause tracks that
        # (it auto-corrected from the pre-harvest "heterogeneous unanswered").
        snap = assess_section4_decision()
        clause = _quality_preservation_clause(snap["quality_preservation"])
        assert "homogeneous SURPASSES" in clause
        assert "heterogeneous SURPASSES" in clause
        assert "unanswered" not in clause
        # the rationale embeds exactly this derived clause (no stale copy)
        assert clause in snap["rationale"]

    def test_quality_clause_tracks_answered_heterogeneous_not_hardcoded(self):
        # Forward-compat / mutation guard: when the in-flight full-budget
        # heterogeneous baseline arm lands + is harvested (n_baseline>0 +
        # baseline.verdict), ``quality_preservation`` flips heterogeneous to
        # answered. A HARDCODED "heterogeneous unanswered" rationale would then
        # directly contradict that surfaced block (P0 科学誠実性 break). The
        # derived clause must track the answered verdict instead.
        qp = {
            "homogeneous": {"answered": True, "verdict": SURPASSES},
            "heterogeneous": {"answered": True, "verdict": TIES},
        }
        clause = _quality_preservation_clause(qp)
        assert "heterogeneous TIES" in clause
        assert "heterogeneous unanswered" not in clause
        # mutation: a hardcoded "heterogeneous unanswered" survives this and
        # fails the line above — proving the clause reads the answered state.


class TestUnblockStepAndArchitecturalInvariant:
    def test_unblock_step_names_private_repo_and_deliberate_strip(self):
        snap = assess_section4_decision()
        step = snap["unblock_step"]
        assert "private repo" in step.lower()
        assert "deliberately stripped" in step
        assert "scripts/prepare_data.py" in step
        assert "/home/jinno/tg-lora" in step
        assert "half-port" in step  # the explicit don't-break-the-boundary warning

    def test_unblock_step_short_form_when_src_data_present(self):
        snap = assess_section4_decision(src_data_present=True)
        assert "fire the run directly" in snap["unblock_step"]

    def test_unblock_step_distinguishes_verdict_run_from_recover_path(self):
        # The honest correction: the unblock step states the verdict run already
        # fired (no re-fire needed) AND notes the recover path is a separate,
        # src.data-blocked path.
        step = assess_section4_decision()["unblock_step"]
        assert "no re-fire is needed" in step or "no re-fire" in step.lower()
        assert "recover.py" in step

    def test_src_data_is_deliberately_stripped_in_this_mirror(self):
        # The architectural invariant that blocks the recover.py --rerun path
        # (NOT the verdict run). Pinned (not fabricated): scripts/prepare_data.py
        # documents the strip and tests/test_filter_dataset.py + tests/test_dedup.py
        # keep the interface without its implementation. If this pin flips, the
        # PIVOT-branch + recover-path logic above must be revisited together — do
        # not just delete this test.
        assert not (REPO_ROOT / "src" / "data").is_dir()
        assert not (REPO_ROOT / RECOVER_RERUN_ENTRYPOINT).exists()
        # ...but the stripped interface is still documented in this mirror:
        assert (REPO_ROOT / "tests" / "test_filter_dataset.py").exists()
        assert (REPO_ROOT / "tests" / "test_dedup.py").exists()
        prepare = (REPO_ROOT / "scripts" / "prepare_data.py").read_text()
        assert "stripped from this public mirror" in prepare

    def test_verdict_worker_module_constant_points_at_the_worker(self):
        # Pins that VERDICT_WORKER_MODULE names the actual §4 verdict entry
        # (public Dolly), not train_tg_lora. A regression here would silently
        # re-conflate the two paths.
        assert VERDICT_WORKER_MODULE == "scripts.run_freeze_validloss_ci_9b"
        assert (REPO_ROOT / "scripts" / "run_freeze_validloss_ci_9b.py").exists()
        # and the worker genuinely imports no src.data:
        worker_src = (
            REPO_ROOT / "scripts" / "run_freeze_validloss_ci_9b.py"
        ).read_text()
        assert "from src.data" not in worker_src
        assert "import src.data" not in worker_src


class TestCLI:
    def test_json_snapshot(self, capsys):
        # real repo: arc complete, no decision landed yet → BLOCKING (exit 3). The
        # snapshot is still emitted, with arc_complete / recommendation intact and
        # the awaiting-decision state surfaced for machine consumers.
        rc = main(["--json"])
        out = capsys.readouterr().out
        snap = json.loads(out)
        assert rc == EXIT_AWAITING_DECISION
        assert snap["arc_complete"] is True
        assert snap["recommendation"] == "SHIP"
        assert snap["awaiting_operator_decision"] is True
        assert snap["landed_decision"] is None

    def test_exit_code_tracks_arc(self, tmp_path):
        # arc complete + UN-LANDED (real repo) → BLOCKING (exit 3); arc incomplete
        # (empty root) → exit 2. Landing a decision flips 3 → 0
        # (see TestLandedDecision).
        assert main([]) == EXIT_AWAITING_DECISION
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
        # the SINGLE blocking operator prompt is emitted while un-landed
        assert "ACTION REQUIRED" in out
        assert "--land" in out


class TestLandedDecision:
    """The decision-landing surface (feedback's highest-leverage move for this
    already-complete arc): BLOCK (exit 3) with ONE operator prompt until the
    operator lands a call via ``--land``, then exit 0 with no further variants.

    The tool does NOT make the call — landing ``accept_null`` does not mutate the
    evidence-based SHIP recommendation; the operator's recorded call and the
    tool's suggestion coexist. Mutation-proven: each guard's negation flips a
    test RED.
    """

    def test_no_record_awaits_with_exit_3(self, tmp_path):
        # arc complete, no landing record → BLOCKING. (mutation: if awaiting were
        # `landed is not None` this would be False; if exit were 0 this fails.)
        _write_deposits(tmp_path)
        snap = assess_section4_decision(repo_root=str(tmp_path))
        assert snap["arc_complete"] is True
        assert snap["landed_decision"] is None
        assert snap["awaiting_operator_decision"] is True
        assert main(["--repo-root", str(tmp_path), "--json"]) == EXIT_AWAITING_DECISION

    def test_land_writes_record_and_unblocks(self, tmp_path):
        # --land accept_null writes a committed record → awaiting clears → exit 0.
        # (mutation: drop the `if landed: return 0` in main → exit stays 3.)
        _write_deposits(tmp_path)
        rc, msg = _land_decision(
            str(tmp_path), "accept_null", "TIES is the honest null"
        )
        assert rc == 0
        assert "accept_null" in msg
        record_path = tmp_path / LANDING_RECORD_REL
        assert record_path.exists()
        record = json.loads(record_path.read_text())
        assert record["branch"] == "accept_null"
        assert record["landed"] is True
        snap = assess_section4_decision(repo_root=str(tmp_path))
        assert snap["awaiting_operator_decision"] is False
        assert snap["landed_decision"]["branch"] == "accept_null"
        assert main(["--repo-root", str(tmp_path)]) == 0

    def test_land_ship_or_accept_null_unblock(self, tmp_path):
        for branch in ("ship", "accept_null"):
            sub = tmp_path / branch
            sub.mkdir()
            _write_deposits(sub)
            rc, _ = _land_decision(str(sub), branch, "operator call")
            assert rc == 0, branch
            assert main(["--repo-root", str(sub)]) == 0, branch

    def test_land_rejects_invalid_branch(self, tmp_path):
        # (mutation: drop the branch check → rc becomes 0 and a record is written.)
        _write_deposits(tmp_path)
        rc, msg = _land_decision(str(tmp_path), "bogus", "x")
        assert rc == EXIT_LAND_INVALID
        assert "ship, accept_null, pivot" in msg
        assert not (tmp_path / LANDING_RECORD_REL).exists()

    def test_land_requires_basis(self, tmp_path):
        # (mutation: drop the basis check → rc becomes 0.)
        _write_deposits(tmp_path)
        assert _land_decision(str(tmp_path), "ship", None)[0] == EXIT_LAND_INVALID
        assert _land_decision(str(tmp_path), "ship", "   ")[0] == EXIT_LAND_INVALID
        assert not (tmp_path / LANDING_RECORD_REL).exists()

    def test_land_refused_when_arc_incomplete(self, tmp_path):
        # cannot land a call on an incomplete verdict. (mutation: drop the
        # arc_complete gate → rc becomes 0 and a record is written on an empty arc.)
        rc, msg = _land_decision(str(tmp_path), "ship", "x")
        assert rc == EXIT_LAND_INVALID
        assert "arc is incomplete" in msg
        assert not (tmp_path / LANDING_RECORD_REL).exists()

    def test_land_pivot_records_private_repo_only(self, tmp_path):
        # src.data stripped here → pivot is a private-repo action; the record says
        # so. (mutation: flip the `not executable_here` → pivot_private_repo_only
        # becomes False.)
        _write_deposits(tmp_path)
        rc, _ = _land_decision(str(tmp_path), "pivot", "go private")
        assert rc == 0
        record = json.loads((tmp_path / LANDING_RECORD_REL).read_text())
        assert record["branch"] == "pivot"
        assert record["pivot_private_repo_only"] is True

    def test_land_does_not_mutate_recommendation(self, tmp_path):
        # THE "not unilateral" PIN: landing accept_null does NOT flip the tool's
        # SHIP suggestion. The operator's call and the evidence-based
        # recommendation coexist; the tool surfaces both, it does not decide.
        # (mutation: if landing set recommendation=landed branch, this fails.)
        _write_deposits(tmp_path)
        _land_decision(str(tmp_path), "accept_null", "ties is the honest null")
        snap = assess_section4_decision(repo_root=str(tmp_path))
        assert snap["recommendation"] == "SHIP"
        assert snap["landed_decision"]["branch"] == "accept_null"

    def test_blocking_prompt_names_branches_and_land_command(self):
        # the single prompt names all three branches + the exact land command +
        # the exit-3 contract.
        prompt = _blocking_prompt()
        for branch in VALID_LAND_BRANCHES:
            assert branch in prompt
        assert "--land" in prompt
        assert "exit 3" in prompt

    def test_human_readable_blocks_when_awaiting(self, tmp_path, capsys):
        _write_deposits(tmp_path)
        rc = main(["--repo-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == EXIT_AWAITING_DECISION
        assert "ACTION REQUIRED" in out
        assert "--land" in out

    def test_human_readable_shows_landed_when_landed(self, tmp_path, capsys):
        _write_deposits(tmp_path)
        _land_decision(str(tmp_path), "accept_null", "ties is the honest null")
        rc = main(["--repo-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "LANDED OPERATOR DECISION" in out
        assert "accept_null" in out
        # no blocking prompt once a call is landed
        assert "ACTION REQUIRED" not in out

    def test_malformed_record_does_not_fake_landed(self, tmp_path):
        # a torn/malformed record cannot fake a landed call — treated as un-landed.
        # (mutation: drop the try/except in _load_landed_decision → raises.)
        _write_deposits(tmp_path)
        (tmp_path / LANDING_RECORD_REL).write_text("{not valid json")
        snap = assess_section4_decision(repo_root=str(tmp_path))
        assert snap["landed_decision"] is None
        assert snap["awaiting_operator_decision"] is True

    def test_non_branch_record_does_not_fake_landed(self, tmp_path):
        # a record naming a non-branch value cannot fake a landed call.
        # (mutation: drop the branch-membership check → this record would load.)
        _write_deposits(tmp_path)
        (tmp_path / LANDING_RECORD_REL).write_text(
            json.dumps({"branch": "definitely_ship", "landed": True})
        )
        snap = assess_section4_decision(repo_root=str(tmp_path))
        assert snap["landed_decision"] is None
        assert snap["awaiting_operator_decision"] is True

    def test_load_landed_decision_returns_none_when_absent(self, tmp_path):
        assert _load_landed_decision(str(tmp_path)) is None


class TestSnapshotFixtureIntegrity:
    """Pin the §4 operator-decision surface against the committed fixture.

    The committed ``tests/fixtures/section4_decision_snapshot_2026-07-28.json``
    captures a fixed audit artifact (TASK-0225). The §4 logic tests above cover
    the decision combinator, but a refactor that silently changes the surface
    output (new top-level predicate, leg label rename, additional branch, a
    reverted P1 品質保持 axis) would slip through them — the predicate set
    would compile fine and the existing tests would still pass. This class
    runs the REAL script via subprocess and asserts the *predicate set*
    matches the committed snapshot. Numerical means and CI bounds are
    intentionally excluded — they would require fixture refresh on a re-fire
    (Cat-C blocked in this mirror), and the predicate invariants are what the
    surface contracts on (a refactor that preserves verdicts but renames a
    leg is still a contract break even if numbers don't move).

    This is the real analog of feedback's "pin in CI rather than just in
    pytest discovery" for this checkout: the literal ``TestAuditCatalogCompleteness``
    name is phantom in this repo (grep 0 hit), but the SPIRIT — refactor-safety
    for a load-bearing surface — applies to the §4 decision surface. A
    mutation that breaks any pinned predicate flips a test RED.
    """

    FIXTURE_REL = "tests/fixtures/section4_decision_snapshot_2026-07-28.json"

    @pytest.fixture
    def fixture_path(self) -> Path:
        return REPO_ROOT / self.FIXTURE_REL

    @pytest.fixture
    def fixture_payload(self, fixture_path: Path) -> dict:
        return json.loads(fixture_path.read_text())

    @pytest.fixture
    def live_payload(self) -> dict:
        # Run the REAL script via subprocess — mirrors how an operator would
        # invoke it (``-m`` + cwd=REPO_ROOT so relative deposit paths resolve).
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.section4_operator_decision", "--json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        # On the real checkout the surface is AWAITING_DECISION (exit 3); a
        # different exit code means the fixture is stale relative to the
        # surface contract (e.g., the operator landed a call, or the arc
        # regressed). Either way the snapshot integrity claim is broken.
        assert proc.returncode == EXIT_AWAITING_DECISION, (
            f"surface should be AWAITING_DECISION (exit 3) on the real checkout; "
            f"got returncode={proc.returncode}, stderr={proc.stderr!r}"
        )
        return json.loads(proc.stdout)

    def test_fixture_exists_and_is_loadable(self, fixture_path: Path):
        # Guard the fixture itself — a delete / rename of the tracked audit
        # artifact would silently lose the §4 closeout trail.
        assert fixture_path.exists(), (
            f"tracked audit artifact missing: {fixture_path}. "
            f"Re-run TASK-0225 (drive surface + capture snapshot) to regenerate."
        )
        payload = json.loads(fixture_path.read_text())
        # A committed snapshot of an UNCOMPLETE arc would be incoherent —
        # a refactor that flips arc_complete without bumping the fixture
        # name is suspicious.
        assert payload["arc_complete"] is True
        assert payload["landed_decision"] is None

    def test_top_level_predicates_match_fixture(
        self, fixture_payload: dict, live_payload: dict
    ):
        # The top-level flags that gate operator action must match. A drift
        # here means the surface contract changed (e.g., new branch, new
        # status, recommendation flipped) without the snapshot being refreshed.
        #
        # run_executable_here / recover_rerun_blocked_by_src_data are DERIVED
        # from already-pinned inputs (verdict_worker_status / src_data_status),
        # but pinning their derived VALUES too closes the catalog-completeness
        # gap the feedback's "pin in CI" spirit names: a refactor that silently
        # changes the derivation (e.g. adds a condition) would flip the derived
        # predicate while leaving its input pinned, and so slip through unless
        # the derived value is itself pinned. recover_rerun_blocked_by_src_data
        # is load-bearing — it is the surface's whole point (docstring point 2:
        # the executable verdict run is distinguished from the src.data-blocked
        # recover path); a silent flip there is a scientific-honesty break.
        for key in (
            "arc_complete",
            "landed_decision",
            "awaiting_operator_decision",
            "recommendation",
            "verdict_worker_status",
            "src_data_status",
            "run_executable_here",
            "recover_rerun_blocked_by_src_data",
        ):
            assert live_payload[key] == fixture_payload[key], (
                f"top-level predicate {key!r} drifted: "
                f"fixture={fixture_payload[key]!r}, live={live_payload[key]!r}"
            )

    def test_legs_set_and_predicates_match(
        self, fixture_payload: dict, live_payload: dict
    ):
        # Same two legs (homogeneous + heterogeneous), same verdicts, same
        # faithfulness / citability / baseline flags. A new leg, a renamed
        # leg, a reverted verdict, or a dropped baseline all break here.
        live_by_label = {leg["label"]: leg for leg in live_payload["legs"]}
        fixture_by_label = {leg["label"]: leg for leg in fixture_payload["legs"]}
        assert set(live_by_label) == set(fixture_by_label), (
            f"leg labels drifted: fixture={set(fixture_by_label)}, "
            f"live={set(live_by_label)}"
        )
        for label in ("homogeneous", "heterogeneous"):
            live_leg = live_by_label[label]
            fix_leg = fixture_by_label[label]
            # proxy_scale=False is the load-bearing FULL-BUDGET claim of the
            # §4 arc — a silent flip to True would turn "citable full-budget
            # TIES" into "proxy TIES" (a scientific-honesty break). It is read
            # INDEPENDENTLY from the deposit (data.get("proxy_scale")), so none
            # of the other pinned predicates would catch its drift. architecture
            # likewise identifies each leg structurally (homogeneous=None /
            # heterogeneous="heterogeneous"). Both are pinned as VALUES here,
            # not just as catalog members (see test_leg_predicate_catalog_is_complete).
            for key in (
                "present",
                "recorded_verdict",
                "rederived_verdict",
                "citable_as_full_section4_verdict",
                "faithful",
                "baseline_present",
                "baseline_verdict",
                "proxy_scale",
                "architecture",
            ):
                assert live_leg[key] == fix_leg[key], (
                    f"leg={label}, predicate {key!r} drifted: "
                    f"fixture={fix_leg[key]!r}, live={live_leg[key]!r}"
                )

    def test_leg_predicate_catalog_is_complete(
        self, fixture_payload: dict, live_payload: dict
    ):
        """Leg-level catalog completeness — the nested analog of
        test_top_level_predicate_catalog_is_complete (TASK-0229).

        The sibling ``test_legs_set_and_predicates_match`` iterates a FIXED
        pinned key list per leg, so it catches a *value* drift on a known leg
        predicate and a KeyError if a pinned one is dropped — but NOT the
        *addition* of a new per-leg predicate: a refactor that introduces a
        fresh per-leg contract term (e.g. a new per-leg ``xxx_status`` gate)
        would land unpinned and slip through CI green. This is the residual
        "a refactor that silently adds a contract term blocks the PR, not slip
        through" gap (feedback bullet 3) at the LEG level — the per-leg
        verdicts are what drive the SHIP recommendation, so a leg predicate
        is load-bearing. This closes it by asserting each leg's key set is
        EXACTLY {pinned-leg-predicates} ∪ {allowlisted-non-load-bearing} for
        BOTH the live output and the committed fixture, so any added/removed
        leg key fails loudly and forces the author to consciously pin it
        (load-bearing) or allowlist it (identifier / numerical / informational).
        """
        # Per-leg predicates byte-pinned by test_legs_set_and_predicates_match —
        # the load-bearing verdict + structural flags (incl. proxy_scale = the
        # full-budget claim, and architecture = the leg's identity).
        PINNED_LEG = {
            "present",
            "recorded_verdict",
            "rederived_verdict",
            "citable_as_full_section4_verdict",
            "faithful",
            "baseline_present",
            "baseline_verdict",
            "proxy_scale",
            "architecture",
        }
        # Non-load-bearing leg keys deliberately NOT byte-pinned: two
        # identifiers (label / deposit path), the numerical verdict drivers
        # (means / CI bounds — intentionally excluded: they would need fixture
        # refresh on a re-fire, which is Cat-C blocked in this mirror), and two
        # informational ints (seq_len / n_baseline — n_baseline is already
        # value-pinned inside the quality_preservation sub-test). Each is
        # ENUMERATED (not a wildcard) so a new leg key cannot hide.
        ALLOWLISTED_LEG = {
            "label",
            "deposit",
            "candidate_mean",
            "surrogate_mean",
            "ci_lower",
            "ci_upper",
            "seq_len",
            "n_baseline",
            "baseline_candidate_mean",
            "baseline_baseline_mean",
        }
        catalog = PINNED_LEG | ALLOWLISTED_LEG
        live_by_label = {leg["label"]: leg for leg in live_payload["legs"]}
        fixture_by_label = {leg["label"]: leg for leg in fixture_payload["legs"]}
        for label in ("homogeneous", "heterogeneous"):
            for source, by_label in (
                ("live", live_by_label),
                ("fixture", fixture_by_label),
            ):
                leg_keys = set(by_label[label])
                assert leg_keys == catalog, (
                    f"§4 leg={label!r} ({source}) predicate catalog is "
                    "INCOMPLETE vs the pin — a refactor added or removed a "
                    "per-leg predicate. A NEW load-bearing leg predicate must "
                    "be pinned (add to PINNED_LEG and the value-pin tuple in "
                    "test_legs_set_and_predicates_match); a new non-load-bearing "
                    "leg key must be consciously allowlisted. "
                    f"missing_from_leg={catalog - leg_keys!r}, "
                    f"unpinned_new_in_leg={leg_keys - catalog!r}"
                )

    def test_quality_preservation_predicates_match(
        self, fixture_payload: dict, live_payload: dict
    ):
        # The P1 品質保持 axis (TASK-0213/0215/0216) is the ONLY leg verdict
        # driver and must match — a regression would silently undo the
        # auto-derivation that closed the heterogeneous full-budget baseline.
        assert set(live_payload["quality_preservation"]) == set(
            fixture_payload["quality_preservation"]
        )
        for label in ("homogeneous", "heterogeneous"):
            live_q = live_payload["quality_preservation"][label]
            fix_q = fixture_payload["quality_preservation"][label]
            for key in ("answered", "verdict", "n_baseline"):
                assert live_q[key] == fix_q[key], (
                    f"quality[{label}].{key!r} drifted: "
                    f"fixture={fix_q[key]!r}, live={live_q[key]!r}"
                )

    def test_branch_executability_matches(
        self, fixture_payload: dict, live_payload: dict
    ):
        # ship / accept_null / pivot branches — the ``executable_here`` flags
        # are the gate on operator action. pivot is private_repo_only=True in
        # this mirror (the only way it could flip to executable_here=True is
        # if src.data were ported — which is the deliberate boundary
        # ``src/data/*`` represents). The other two must remain locally
        # executable so the operator's land command is actually runnable.
        assert set(live_payload["branches"]) == set(fixture_payload["branches"]), (
            f"branch keys drifted: fixture={set(fixture_payload['branches'])}, "
            f"live={set(live_payload['branches'])}"
        )
        for branch in fixture_payload["branches"]:
            assert (
                live_payload["branches"][branch]["executable_here"]
                == fixture_payload["branches"][branch]["executable_here"]
            ), (
                f"branch {branch!r}.executable_here drifted: "
                f"fixture={fixture_payload['branches'][branch]['executable_here']!r}, "
                f"live={live_payload['branches'][branch]['executable_here']!r}"
            )

    def test_top_level_predicate_catalog_is_complete(
        self, fixture_payload: dict, live_payload: dict
    ):
        """Catalog completeness — feedback bullet 3 ("catalog completeness is
        now the load-bearing claim; a refactor that silently drops/ adds a
        contract term should block the PR, not slip through").

        The sibling ``test_top_level_predicates_match_fixture`` iterates a
        FIXED 8-key list, so it catches a *value* drift on a known predicate
        but NOT the *addition* of a new top-level predicate: a refactor that
        introduces a fresh contract term (e.g. a new ``xxx_status`` gate on
        operator action) would land unpinned and slip through CI green. This
        closes that residual gap by asserting the snapshot's top-level key set
        is EXACTLY {pinned} ∪ {allowlisted-non-load-bearing} — so any added or
        removed key fails loudly and forces the author to consciously pin it
        (load-bearing) or allowlist it (derived / prose / constant). This is
        the literal "catalog completeness" claim the feedback names, applied
        to the REAL §4 surface (``TestAuditCatalogCompleteness`` is phantom in
        this checkout — grep 0 hit; see TASK-0226/0228/0229).
        """
        # Keys byte-pinned by test_top_level_predicates_match_fixture — the
        # load-bearing operator-action gates (booleans / enums / the sole
        # executable-here verdict).
        PINNED_TOP_LEVEL = {
            "arc_complete",
            "landed_decision",
            "awaiting_operator_decision",
            "recommendation",
            "verdict_worker_status",
            "src_data_status",
            "run_executable_here",
            "recover_rerun_blocked_by_src_data",
        }
        # Keys pinned by their OWN dedicated sub-tests (leg / quality /
        # branch predicate sets), not by the top-level loop.
        PINNED_BY_SUBTEST = {"legs", "quality_preservation", "branches"}
        # Non-load-bearing keys deliberately NOT byte-pinned: a derived reason
        # string, an informational constant, and two prose fields (rationale /
        # unblock_step) whose wording may evolve without a verdict/state
        # change. Each is ENUMERATED (not a wildcard) so a new key cannot hide.
        ALLOWLISTED_NON_LOAD_BEARING = {
            "verdict_worker_reason",  # derived from verdict_worker_status
            "seq1024_full_budget_vram_floor_mib",  # informational constant (11_000)
            "rationale",  # prose, derived from state
            "unblock_step",  # prose, derived from state
        }
        catalog = PINNED_TOP_LEVEL | PINNED_BY_SUBTEST | ALLOWLISTED_NON_LOAD_BEARING
        live_keys = set(live_payload)
        fixture_keys = set(fixture_payload)
        # live == fixture key set is already implied by the sibling tests, but
        # assert it explicitly so a catalog drift is attributed correctly.
        assert live_keys == fixture_keys, (
            "live and fixture top-level key sets disagree before the catalog "
            "completeness check: "
            f"only_live={live_keys - fixture_keys!r}, "
            f"only_fixture={fixture_keys - live_keys!r}"
        )
        assert live_keys == catalog, (
            "§4 snapshot top-level predicate catalog is INCOMPLETE vs the pin — "
            "a refactor added or removed a top-level predicate. A NEW "
            "load-bearing predicate must be pinned (add to PINNED_TOP_LEVEL or "
            "a dedicated sub-test); a new non-load-bearing key must be "
            "consciously allowlisted. "
            f"missing_from_snapshot={catalog - live_keys!r}, "
            f"unpinned_new_in_snapshot={live_keys - catalog!r}"
        )
