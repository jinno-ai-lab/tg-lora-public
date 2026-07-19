"""Tests for ``scripts/replay_freeze_validloss_ci.py`` — the Category-C step
reduced to a concrete, GPU-free, executable command.

``run_freeze_validloss_ci`` trains the proxy and deposits real valid_loss
samples + its verdict to JSON. ``replay_freeze_validloss_ci`` re-judges that
recording through :func:`surrogate_valid_loss_ci` with no GPU, no model, and no
torch. The suite guards:

* **Import health + ``--help``** — the CLI launches as ``-m`` (the canary every
  ``scripts.run_*`` / ``scripts.replay_*`` CLI in this repo keeps).
* **Schema validation** — ``load_samples`` rejects a missing file, malformed
  JSON, and a file lacking the non-empty sample lists the judge needs.
* **Judge equivalence + determinism** — replaying stored samples is byte-identical
  to calling ``surrogate_valid_loss_ci`` directly with the same seed, and a
  clear candidate-vs-surrogate separation replays as ``SURPASSES``.
* **Faithfulness on real recorded evidence** — the committed fixture
  (``tests/fixtures/freeze_validloss_generalize_proxy.json``, a real RTX 3060
  ``--task generalize`` run) replays to the verdict it *recorded* (``TIES``):
  the stored floats earn the verdict under the deterministic bootstrap, the
  verdict is not painted on. This is the expected-output assertion that pins
  the recorded Category-C dataset.
* **Scale honesty + CLI assertion** — ``proxy_scale`` is surfaced from the
  file; ``--expected`` exits 0 on a match and 2 on a mismatch.
* **The target-scale drop-in + synthetic-provenance guard** — a committed
  ``proxy_scale: false`` plumbing fixture flips the replayed verdict's scale
  label to TARGET with no code change (the MS-PF2 Cat-C contract); the proxy and
  target recordings are distinguished by the file's ``proxy_scale`` flag alone.
  The fixture also carries ``synthetic: true``, so the replay withholds the
  citable "this verdict IS the §4 target-scale result" claim a genuine 9B run
  earns and warns instead — enforcing in code that synthetic plumbing is never
  cited as a real measurement. The genuine claim is covered on a constructed
  ``synthetic: false`` recording (the shape a real 9B run deposits).
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.tg_lora.freeze_surrogate_ci import surrogate_valid_loss_ci
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

from scripts.replay_freeze_validloss_ci import (
    format_replay,
    load_samples,
    main,
    replay_samples,
    replay_to_json,
)

# The committed real-GPU recording: a ``--task generalize`` run on the RTX
# 3060 (verdict TIES, candidate_mean≈2.529, surrogate_mean≈2.648,
# CI[95%]=[−0.067, +0.313]). Regenerate with ``make freeze-validloss-ci-generalize``.
FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_generalize_proxy.json"
)

# The committed PLUMBING fixture for the target-scale drop-in (proxy_scale=False).
# Synthetic floats (NOT a real 9B measurement) with a clear candidate-vs-surrogate
# separation that replays to SURPASSES — it exists to prove the replay judge
# upgrades a ``proxy_scale: false`` sample file to the TARGET_SCALE label with no
# code change (the MS-PF2 Cat-C contract): the branch the real proxy fixture
# (proxy_scale=True) above never exercises.
FIXTURE_TARGET = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_target_dropin_plumbing.json"
)

# The committed real-GPU recording for the DISCRIMINATING positive control:
# ``--architecture heterogeneous --task generalize`` on the RTX 3060 — the one
# regime where freeze order structurally CAN matter (per-layer rank rising toward
# the output, GOAL §1.5/§8 non-uniform per-layer cost) on a task order can move
# (held-out generalization). Every other leg (homogeneous, or the memorize task)
# is a TIES where order is structurally irrelevant or the task is unlearned; this
# leg's TIES is therefore the strongest proxy-scale evidence the apparatus cannot
# resolve order even where it should — consistent with the ratio=0.000
# order-sensitivity diagnostic, and the reason target-scale is proven necessary.
# Regenerate with ``make freeze-validloss-ci-heterogeneous-generalize``.
FIXTURE_HETEROGEN = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_heterogeneous_generalize_proxy.json"
)

# The committed real-GPU recording that exercises the SINGLE-CYCLE guard the two
# fixtures above only assert the *inverse* of: a thin arm (n=2/arm, below
# MIN_SAMPLE_FOR_BOOTSTRAP) on the same discriminating heterogeneous x generalize
# leg, where the candidate *appears* to win (a material point lead) but the gate
# refuses the significance call and flags it THIN_EVIDENCE instead — the one
# false-confidence guard (zero-median→TIES and baseline-less→unverified are
# already recorded) whose honest label had never been emitted on a real artifact.
# The n=2 losses are the deterministic first two seeds of FIXTURE_HETEROGEN
# (same base_seed), so this recording is a faithful thin truncation of the
# discriminating leg, not a different experiment. Regenerate with
# ``make freeze-validloss-ci-heterogeneous-generalize-thin``.
FIXTURE_THIN = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_heterogeneous_generalize_thin_proxy.json"
)

# The committed real-GPU recording that exercises the THIRD verdict label the
# recordings above never emit: UNDERSHOOTS. Every committed recording is TIES
# (the proxy order-signal is genuinely zero, so TIES is a true null — that is
# the research conclusion), and the only SURPASSES is a synthetic plumbing
# fixture. So no real recording had ever shown the gate CAN fire a non-TIES
# verdict on a measured signal. This negative control closes that gap: the
# candidate arm is deliberately under-trained (candidate_total=2 vs surrogate
# total=60) — an asymmetric budget UNRELATED to freeze order — so the candidate
# is reliably worse and the gate fires a real UNDERSHOOTS (CI entirely below
# zero). It is a sensitivity probe (proof the TIES recordings are a genuine
# null, not a broken always-TIES pipeline), tagged negative_control=True so the
# verdict is never misread as a §4 order result. Regenerate with
# ``make freeze-validloss-ci-negative-control``.
FIXTURE_NEGCTRL = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_negative_control_proxy.json"
)

# The committed real-GPU recording that exercises the THIRD verdict label's
# UPWARD direction on a real measurement: SURPASSES. Every committed recording
# is TIES (the proxy order-signal is genuinely zero), the only non-TIES real
# recording was the DOWNWARD UNDERSHOOTS (FIXTURE_NEGCTRL, candidate-degraded),
# and the only SURPASSES was a synthetic plumbing fixture — so the gate's
# ``lower > 0.0`` branch had never fired on gradient-trained samples. This
# SYMMETRIC negative control closes that gap: the SURROGATE arm is deliberately
# under-trained (surrogate_total=2 vs candidate total=60) — the same asymmetric
# non-order lever as FIXTURE_NEGCTRL, applied to the other arm — so the
# candidate looks better by construction and the gate fires a real SURPASSES
# (CI entirely above zero). It is the first real ``passes=True`` recording too
# (significant AND material), proving the UPWARD win path on a measurement, not
# hand-authored floats. It is a sensitivity probe, never a §4 order result — the
# ``negative_control`` provenance with ``negative_control_arm="surrogate"``
# enforces that. Regenerate with ``make freeze-validloss-ci-negative-control-surrogate``.
FIXTURE_NEGCTRL_SURROGATE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_negative_control_surrogate_proxy.json"
)

# The committed REAL 9B target-scale deposit — the move the loop's feedback asked
# for in place of another honesty guard: feed REAL numbers through the gate. Every
# other target-scale fixture above is synthetic plumbing (hand-authored floats);
# this one carries the genuine best_valid_loss of multi-seed 9B runs — candidate
# TG-LoRA (configs/9b_tg_lora.yaml) vs full-backprop baseline (9b_baseline.yaml),
# compute-matched at 94 optimizer steps, trained on the only config a 12GB RTX
# 3060 fits (seq_len=256 + eval_batch_size=1 + expandable_segments). It is the
# first recording to exercise ``citable_as_target_scale`` / the reduced-context
# guard on REAL data, not a single upstream candidate and not synthetic floats.
# The losses are harvested from upstream run_metrics.jsonl by
# ``scripts/form_freeze_validloss_deposit.py``; the ``verdict`` field is stamped
# from a replay so the faithfulness check is non-trivial. See TASK-0152 Tier-1.
FIXTURE_REAL_9B = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_validloss_9b_target.json"
)


# ---------------------------------------------------------------------------
# Import health + --help
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.replay_freeze_validloss_ci")
        for attr in (
            "main", "build_parser", "load_samples", "replay_samples",
            "format_replay", "replay_to_json", "EXPECTED_VERDICTS",
        ):
            assert hasattr(mod, attr), f"missing {attr}"

    def test_help_launches_as_module(self, tmp_path):
        # The canary contract: every scripts.* CLI launches via ``-m`` with a
        # working ``--help`` and exit 0 (the sys.path-bootstrap invariant).
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.replay_freeze_validloss_ci", "--help"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "surrogate_valid_loss_ci" in proc.stdout
        assert "--expected" in proc.stdout


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestLoadSamples:
    def test_rejects_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_samples("/nonexistent/path/to/samples.json")

    def test_rejects_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_samples(bad)

    def test_rejects_missing_sample_keys(self, tmp_path):
        bad = tmp_path / "no_samples.json"
        bad.write_text(json.dumps({"verdict": "TIES", "device": "cuda"}))
        with pytest.raises(ValueError, match="candidate_losses"):
            load_samples(bad)

    def test_rejects_empty_sample_list(self, tmp_path):
        bad = tmp_path / "empty.json"
        bad.write_text(
            json.dumps({"candidate_losses": [], "surrogate_losses": [1.0, 2.0]})
        )
        with pytest.raises(ValueError, match="candidate_losses"):
            load_samples(bad)

    def test_loads_committed_fixture(self):
        data = load_samples(FIXTURE)
        assert len(data["candidate_losses"]) == 5
        assert len(data["surrogate_losses"]) == 5
        assert data["proxy_scale"] is True
        assert data["device"] == "cuda"


# ---------------------------------------------------------------------------
# Judge equivalence + determinism
# ---------------------------------------------------------------------------


def _data(candidate, surrogate, *, base_seed=0, proxy_scale=True, synthetic=False,
          negative_control=False, full_context=True):
    # ``full_context`` is emitted only when False — a genuine full-context 9B
    # recording need not carry the field (the judge defaults it True), while a
    # reduced-context probe (the only thing a 12GB box can run) carries it
    # explicitly so it cannot be mis-cited as the full §4 verdict.
    d = {
        "candidate_losses": list(candidate),
        "surrogate_losses": list(surrogate),
        "base_seed": base_seed,
        "proxy_scale": proxy_scale,
        "synthetic": synthetic,
        "negative_control": negative_control,
    }
    if not full_context:
        d["full_context"] = False
    return d


class TestReplaySamples:
    def test_replay_equals_direct_judge_call(self):
        # Replay is a thin loader over the judge: same seed → identical object.
        cand = [2.7, 2.5, 2.4, 2.6]
        surr = [2.9, 2.6, 2.7, 2.5]
        data = _data(cand, surr, base_seed=7)
        replayed = replay_samples(data, seed=7)
        direct = surrogate_valid_loss_ci(cand, surr, seed=7)
        assert replayed == direct

    def test_seed_defaults_to_recorded_base_seed(self):
        # No --seed → the file's base_seed reproduces the recorded verdict.
        cand = [1.0, 1.0, 1.0, 1.0]
        surr = [2.0, 2.0, 2.0, 2.0]
        data = _data(cand, surr, base_seed=11)
        default = replay_samples(data)
        explicit = replay_samples(data, seed=11)
        assert default == explicit

    def test_clear_separation_replays_as_surpasses(self):
        # Candidate uniformly better than surrogate → CI entirely above 0.
        data = _data([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0])
        assert replay_samples(data).significance_verdict == SURPASSES


# ---------------------------------------------------------------------------
# Faithfulness on real recorded evidence (the expected-output assertion)
# ---------------------------------------------------------------------------


class TestFixtureFaithfulness:
    def test_replay_reproduces_recorded_ties(self):
        # The committed real-GPU recording replays to the verdict it recorded:
        # the stored floats earn TIES under the deterministic bootstrap.
        data = load_samples(FIXTURE)
        ci = replay_samples(data)
        assert ci.significance_verdict == data["verdict"] == TIES

    def test_replay_matches_recorded_statistics(self):
        # Means, the point improvement, and the CI bounds are reproduced from
        # the stored floats — the recording's numbers are not arbitrary.
        data = load_samples(FIXTURE)
        ci = replay_samples(data)
        assert ci.candidate_mean == pytest.approx(data["candidate_mean"])
        assert ci.surrogate_mean == pytest.approx(data["surrogate_mean"])
        assert ci.point_improvement == pytest.approx(data["point_improvement"])
        assert ci.lower == pytest.approx(data["lower"])
        assert ci.upper == pytest.approx(data["upper"])

    def test_replay_is_non_thin(self):
        # n=5/arm is above MIN_SAMPLE_FOR_BOOTSTRAP, so the verdict is not a
        # thin-evidence caveat — the recorded TIES is a real significance call.
        ci = replay_samples(load_samples(FIXTURE))
        assert not ci.is_thin_evidence


# ---------------------------------------------------------------------------
# The discriminating positive control (heterogeneous x generalize), run for real
# ---------------------------------------------------------------------------


class TestHeterogeneousGeneralizePositiveControl:
    """The leg where a §4 order win was most reachable, run end-to-end on a real
    GPU and locked by no-GPU replay.

    ``run_freeze_validloss_ci`` is the hardened §4 verdict gate: a bootstrap CI
    on the candidate-vs-surrogate valid_loss difference, graduated to
    SURPASSES/TIES/UNDERSHOOTS, with the false-confidence guards the loop's
    feedback asked whether they actually fire — thin-evidence (an arm below
    ``MIN_SAMPLE_FOR_BOOTSTRAP`` cannot anchor a significance call), materiality
    (a lead below the margin is not a win), and the proxy/target citation gate.
    This class runs that gate, through the no-GPU replay, on the *real* recording
    that settles the question a homogeneous or memorize TIES cannot: the only
    regime where order structurally CAN matter.

    The committed fixture is a genuine RTX 3060 ``--architecture heterogeneous
    --task generalize`` run (per-layer rank ``[1, 2, 4, 7, 13, 24]`` rising
    toward the output, held-out teacher-student task). On that stack a non-TIES
    verdict would be the evidence the apparatus is sensitive to order; the replay
    records what the gate actually emits on this real artifact and pins that the
    hardened guards surface honestly rather than dressing a proxy-scale tie up as
    a citable §4 win.
    """

    def test_fixture_is_the_discriminating_positive_control(self):
        # Provenance pins this as the heterogeneous x generalize leg — distinct
        # from FIXTURE (homogeneous generalize), whose TIES is the trivial one
        # (order structurally irrelevant on a uniform stack).
        data = load_samples(FIXTURE_HETEROGEN)
        assert data["architecture"] == "heterogeneous"
        assert data["task"] == "generalize"
        # Per-layer rank is non-uniform and rises toward the output — the GOAL
        # §1.5/§8 asymmetry that lets order matter (a uniform stack could not).
        assert len(set(data["ranks"])) > 1
        assert data["ranks"] == sorted(data["ranks"])

    def test_replay_reproduces_recorded_ties_faithfully(self):
        # The gate, re-run on the real stored floats with no GPU, emits the
        # verdict the recording stored — the floats earn the label under the
        # deterministic bootstrap; it is not painted on.
        data = load_samples(FIXTURE_HETEROGEN)
        ci = replay_samples(data)
        assert ci.significance_verdict == data["verdict"] == TIES
        assert ci.point_improvement == pytest.approx(data["point_improvement"])
        # An honest TIES: the CI straddles zero, so the lead is not significant.
        assert ci.lower < 0.0 < ci.upper

    def test_hardened_guards_surface_not_false_confidence(self):
        # The false-confidence guards the loop hardened the gate against, pinned
        # on the real recording: n=5/arm is non-thin, and a genuine proxy
        # recording is never citable as a target-scale §4 result.
        data = load_samples(FIXTURE_HETEROGEN)
        ci = replay_samples(data)
        assert not ci.is_thin_evidence  # n=5 >= MIN_SAMPLE_FOR_BOOTSTRAP
        out = replay_to_json(FIXTURE_HETEROGEN, data, ci)
        assert out["proxy_scale"] is True
        assert out["synthetic"] is False  # genuine run, not plumbing
        assert out["citable_as_target_scale"] is False
        assert out["faithful"] is True

    def test_expected_ties_exits_zero(self):
        # The recording is pinned to TIES: a replay gate asserting TIES passes,
        # so a future drift in the recorded floats (or the judge) fails loudly.
        assert main([str(FIXTURE_HETEROGEN), "--expected", TIES]) == 0


# ---------------------------------------------------------------------------
# The single-cycle guard, run end-to-end on a real (thin) artifact
# ---------------------------------------------------------------------------


class TestThinEvidenceGuardFires:
    """The false-confidence guard the loop's feedback named ("single-cycle"),
    run end-to-end on a real GPU artifact and locked by no-GPU replay.

    The hardened §4 verdict gate carries three false-confidence guards. Two were
    already recorded on real artifacts: the *zero-median* guard (a CI straddling
    zero honestly reads TIES — pinned by ``TestFixtureFaithfulness`` and
    ``TestHeterogeneousGeneralizePositiveControl``) and the *baseline-less* guard
    (``valid_loss_unverified=True`` until a GPU run deposits a quality number —
    pinned by the structural-gate suite). The *single-cycle* guard — an arm below
    ``MIN_SAMPLE_FOR_BOOTSTRAP`` cannot anchor a significance statement, so the
    verdict is flagged ``is_thin_evidence`` and the audit says "do not read as
    confirmed" — had only ever been asserted in its *inverse* (every committed
    recording is n=5/arm, ``not is_thin_evidence``). This class records that guard
    actually *firing* on a real thin run: the discriminating heterogeneous x
    generalize leg truncated to n=2/arm, where the candidate's point lead looks
    material yet the gate refuses the significance win and surfaces the
    THIN_EVIDENCE caveat rather than dressing a 2-seed anecdote up as a call.
    """

    def test_fixture_is_the_thin_discriminating_leg(self):
        # Provenance pins this as the same heterogeneous x generalize leg as
        # FIXTURE_HETEROGEN, but truncated to a thin n=2/arm sample — the exact
        # shape a researcher running too few seeds deposits.
        data = load_samples(FIXTURE_THIN)
        assert data["architecture"] == "heterogeneous"
        assert data["task"] == "generalize"
        assert len(data["candidate_losses"]) == 2
        assert len(data["surrogate_losses"]) == 2

    def test_thin_arm_is_the_first_two_seeds_of_the_n5_recording(self):
        # The n=2 run is a faithful thin truncation of the committed n=5
        # discriminating fixture (same base_seed → same per-seed init/data draw),
        # not a different experiment. This is what makes it a clean record of the
        # *guard* rather than a new result: the same leg, deliberately starved.
        thin = load_samples(FIXTURE_THIN)
        full = load_samples(FIXTURE_HETEROGEN)
        assert thin["candidate_losses"] == full["candidate_losses"][:2]
        assert thin["surrogate_losses"] == full["surrogate_losses"][:2]

    def test_single_cycle_guard_fires_on_real_thin_recording(self):
        # The guard the feedback asked whether it actually emits: a real thin arm
        # (n=2 < MIN_SAMPLE_FOR_BOOTSTRAP=3) flips is_thin_evidence True. The
        # recorded flag is not painted on — the replay re-derives it from the
        # stored float counts.
        data = load_samples(FIXTURE_THIN)
        ci = replay_samples(data)
        assert ci.is_thin_evidence
        assert ci.n_candidate == 2
        assert ci.n_surrogate == 2
        assert ci.is_thin_evidence is data["is_thin_evidence"]

    def test_thin_lead_is_refused_a_significance_win(self):
        # The false-confidence trap: the candidate *looks* materially better
        # (a positive point lead above the material margin) on a thin sample, yet
        # the wide thin-sample CI straddles zero so the verdict is TIES — never
        # SURPASSES, never passes — the guard doing exactly what it exists for.
        data = load_samples(FIXTURE_THIN)
        ci = replay_samples(data)
        assert ci.point_improvement > 0.0       # candidate appears better...
        assert ci.is_material                    # ...and the lead clears margin...
        assert ci.significance_verdict == TIES   # ...but it is NOT a significance win
        assert not ci.significant_surpasses
        assert not ci.passes
        assert ci.lower < 0.0 < ci.upper         # the CI straddles zero → honest TIES

    def test_replay_reproduces_recorded_ties_faithfully(self):
        # The gate, re-run on the real stored floats with no GPU, emits the
        # verdict the recording stored — the floats earn the label under the
        # deterministic bootstrap; it is not painted on.
        data = load_samples(FIXTURE_THIN)
        ci = replay_samples(data)
        assert ci.significance_verdict == data["verdict"] == TIES
        assert ci.point_improvement == pytest.approx(data["point_improvement"])
        assert ci.lower == pytest.approx(data["lower"])
        assert ci.upper == pytest.approx(data["upper"])

    def test_thin_evidence_note_is_emitted_not_hidden(self):
        # The honesty is in the rendered output, not just the boolean: the audit
        # says plainly the thin verdict must not be read as confirmed, so a
        # 2-seed anecdote can never masquerade as a settled §4 call.
        data = load_samples(FIXTURE_THIN)
        text = format_replay(FIXTURE_THIN, data, replay_samples(data))
        assert "THIN_EVIDENCE" in text
        assert "do not read this verdict as confirmed" in text

    def test_thin_recording_is_still_proxy_scale_and_not_citable(self):
        # Thin evidence does not relax the scale-honesty gate: a genuine proxy
        # recording is still not citable as a target-scale §4 result.
        data = load_samples(FIXTURE_THIN)
        out = replay_to_json(FIXTURE_THIN, data, replay_samples(data))
        assert out["proxy_scale"] is True
        assert out["synthetic"] is False  # genuine run
        assert out["citable_as_target_scale"] is False
        assert out["faithful"] is True

    def test_expected_ties_exits_zero(self):
        # The thin recording is pinned to TIES: asserting TIES passes, so a drift
        # in the recorded floats (or the judge) fails loudly even on the thin leg.
        assert main([str(FIXTURE_THIN), "--expected", TIES]) == 0


# ---------------------------------------------------------------------------
# The negative control (real UNDERSHOOTS): the third verdict label, finally
# recorded on a real artifact
# ---------------------------------------------------------------------------


class TestNegativeControlUndershoots:
    """The ``UNDERSHOOTS`` verdict recorded on a real GPU artifact.

    Every other committed recording is ``TIES`` (the proxy order-signal is
    genuinely zero, so ``TIES`` is a true null — the research conclusion), and
    the only ``SURPASSES`` is a synthetic plumbing fixture. So no real recording
    had ever shown the gate CAN fire a non-TIES verdict on a measured signal.
    This negative control closes that gap: a deliberately under-trained
    candidate (``candidate_total=2`` vs surrogate ``total=60`` — an asymmetric
    budget UNRELATED to freeze order) is reliably worse, so the gate fires a
    real ``UNDERSHOOTS`` (CI entirely below zero). It is a sensitivity probe,
    not a §4 order result — the ``negative_control`` provenance enforces that.
    """

    def test_fixture_is_the_negative_control(self):
        # The recording carries the provenance that makes it a sensitivity probe,
        # not an order result: candidate was under-trained (total=2 vs 60).
        data = load_samples(FIXTURE_NEGCTRL)
        assert data["negative_control"] is True
        assert data["candidate_total"] == 2
        assert data["total"] == 60
        assert data["candidate_total"] != data["total"]

    def test_real_undershoots_fires_on_real_recording(self):
        # The third verdict label, emitted on a real artifact: replay recomputes
        # UNDERSHOOTS from the stored floats (CI entirely below zero), matching
        # the verdict recorded at run time — the gate fires non-TIES on a
        # measured signal, so the order-experiment TIES recordings are a genuine
        # null, not a broken always-TIES pipeline.
        data = load_samples(FIXTURE_NEGCTRL)
        ci = replay_samples(data)
        assert ci.significance_verdict == UNDERSHOOTS == data["verdict"]
        assert ci.upper < 0.0  # CI entirely below zero ⇒ UNDERSHOOTS
        assert ci.n_candidate == 5 and ci.n_surrogate == 5

    def test_gap_is_from_undertraining_not_order(self):
        # The UNDERSHOOTS is earned by the under-trained candidate being reliably
        # WORSE (higher valid_loss), not by a low surrogate: candidate_mean sits
        # far above surrogate_mean, the injected non-order quality gap.
        data = load_samples(FIXTURE_NEGCTRL)
        ci = replay_samples(data)
        assert ci.candidate_mean > ci.surrogate_mean
        assert ci.point_improvement < 0.0

    def test_negative_control_is_not_thin(self):
        # n=5/arm is above MIN_SAMPLE_FOR_BOOTSTRAP: the UNDERSHOOTS is a real
        # significance call, not a thin-evidence caveat (distinct from FIXTURE_THIN).
        assert not replay_samples(load_samples(FIXTURE_NEGCTRL)).is_thin_evidence

    def test_negative_control_note_is_emitted_not_hidden(self):
        # The provenance surfaces in the human report so a recorded UNDERSHOOTS
        # cannot be misread as "the output-first order is worse than random".
        data = load_samples(FIXTURE_NEGCTRL)
        text = format_replay(FIXTURE_NEGCTRL, data, replay_samples(data))
        assert "NEGATIVE_CONTROL" in text
        assert "NOT a §4 order result" in text
        assert "do not read it as evidence" in text
        # The scale line also carries the flag for a reader scanning it.
        assert "negative_control=True" in text

    def test_negative_control_withholds_citable_claim(self):
        # A negative-control verdict is never citable as a §4 result, even though
        # it is a real (non-synthetic, proxy-scale) measurement — the gate refuses
        # it via the negative_control flag, mirroring the synthetic guard.
        data = load_samples(FIXTURE_NEGCTRL)
        out = replay_to_json(FIXTURE_NEGCTRL, data, replay_samples(data))
        assert out["negative_control"] is True
        assert out["synthetic"] is False
        assert out["citable_as_target_scale"] is False

    def test_expected_undershoots_exits_zero(self):
        # The recording is pinned to UNDERSHOOTS: the --expected gate passes, so a
        # drift in the recorded floats (or the judge) fails loudly on this leg.
        assert main([str(FIXTURE_NEGCTRL), "--expected", UNDERSHOOTS]) == 0

    def test_expected_ties_exits_nonzero(self):
        # Asserting the wrong verdict (TIES) on the UNDERSHOOTS recording exits
        # nonzero — the --expected gate distinguishes UNDERSHOOTS from TIES.
        assert main([str(FIXTURE_NEGCTRL), "--expected", TIES]) == 2


# ---------------------------------------------------------------------------
# The symmetric negative control (real SURPASSES): the upward real label,
# finally recorded on a real artifact
# ---------------------------------------------------------------------------


class TestNegativeControlSurpasses:
    """The ``SURPASSES`` verdict recorded on a real GPU artifact.

    The symmetric completion of ``TestNegativeControlUndershoots``. That class
    proved the gate fires the DOWNWARD real label (UNDERSHOOTS) by degrading the
    candidate; this proves the UPWARD real label (SURPASSES) by degrading the
    surrogate on the SAME non-order lever. Until this fixture, the gate's
    ``lower > 0.0`` branch had only ever fired on hand-authored synthetic floats
    (FIXTURE_TARGET) — so the apparatus had never demonstrated it resolves a
    genuine IMPROVEMENT, only a genuine degradation. A SURPASSES-only-on-
    synthetic gate could not distinguish "the candidate never truly wins at
    proxy scale" (the honest null) from "the SURPASSES branch is silently
    broken". This recording closes that gap: the under-trained surrogate
    (``surrogate_total=2`` vs candidate ``total=60``) is reliably worse, so the
    gate fires a real SURPASSES (CI entirely above zero) — and, as a bonus, the
    first real ``passes=True`` (significant AND material). It is a sensitivity
    probe, never a §4 order result — the ``negative_control_arm="surrogate"``
    provenance enforces that, and the note names the surrogate, not the
    candidate.
    """

    def test_fixture_is_the_surrogate_negative_control(self):
        # The recording carries the provenance that makes it a sensitivity probe
        # of the UPWARD direction: surrogate was under-trained (total=2 vs 60).
        data = load_samples(FIXTURE_NEGCTRL_SURROGATE)
        assert data["negative_control"] is True
        assert data["negative_control_arm"] == "surrogate"
        assert data["surrogate_total"] == 2
        assert data["total"] == 60
        assert data["surrogate_total"] != data["total"]
        assert data["candidate_total"] == data["total"]  # candidate symmetric

    def test_real_surpasses_fires_on_real_recording(self):
        # The upward verdict label, emitted on a real artifact: replay recomputes
        # SURPASSES from the stored floats (CI entirely above zero), matching the
        # verdict recorded at run time — the gate fires the UPWARD non-TIES label
        # on a measured signal, so the SURPASSES branch is proven on real samples,
        # not only the synthetic plumbing fixture.
        data = load_samples(FIXTURE_NEGCTRL_SURROGATE)
        ci = replay_samples(data)
        assert ci.significance_verdict == SURPASSES == data["verdict"]
        assert ci.lower > 0.0  # CI entirely above zero ⇒ SURPASSES
        assert ci.n_candidate == 5 and ci.n_surrogate == 5

    def test_first_real_passes_true_recording(self):
        # The TIES recordings have passes=False (not significant); the UNDERSHOOTS
        # negative control has passes=False; only the synthetic plumbing fixture
        # had passes=True. This is the first REAL recording clearing the §4 bar
        # (significant AND material) — proving the win path on a measurement.
        data = load_samples(FIXTURE_NEGCTRL_SURROGATE)
        ci = replay_samples(data)
        assert ci.passes is True
        assert ci.significant_surpasses is True
        assert ci.is_material is True

    def test_gap_is_from_surrogate_undertraining_not_order(self):
        # The SURPASSES is earned by the under-trained surrogate being reliably
        # WORSE (higher valid_loss), not by a strong candidate: candidate_mean
        # sits far below surrogate_mean, the injected non-order quality gap.
        data = load_samples(FIXTURE_NEGCTRL_SURROGATE)
        ci = replay_samples(data)
        assert ci.candidate_mean < ci.surrogate_mean
        assert ci.point_improvement > 0.0

    def test_surrogate_negative_control_is_not_thin(self):
        # n=5/arm is above MIN_SAMPLE_FOR_BOOTSTRAP: the SURPASSES is a real
        # significance call, not a thin-evidence caveat.
        assert not replay_samples(load_samples(FIXTURE_NEGCTRL_SURROGATE)).is_thin_evidence

    def test_surrogate_negative_control_note_names_surrogate(self):
        # The arm-aware provenance surfaces in the human report so a recorded
        # SURPASSES cannot be misread as "the output-first order beats random" —
        # the note names the SURROGATE as the degraded arm, not the candidate.
        data = load_samples(FIXTURE_NEGCTRL_SURROGATE)
        text = format_replay(FIXTURE_NEGCTRL_SURROGATE, data, replay_samples(data))
        assert "NEGATIVE_CONTROL" in text
        assert "surrogate arm was deliberately degraded" in text
        assert "candidate arm was deliberately degraded" not in text
        assert "NOT a §4 order result" in text
        assert "do not read it as evidence" in text
        assert "negative_control=True" in text

    def test_surrogate_negative_control_withholds_citable_claim(self):
        # A surrogate-side negative-control verdict is never citable as a §4
        # result, even though it is a real (non-synthetic, proxy-scale)
        # measurement — the gate refuses it via the negative_control flag,
        # mirroring the candidate-side and synthetic guards.
        data = load_samples(FIXTURE_NEGCTRL_SURROGATE)
        out = replay_to_json(FIXTURE_NEGCTRL_SURROGATE, data, replay_samples(data))
        assert out["negative_control"] is True
        assert out["negative_control_arm"] == "surrogate"
        assert out["synthetic"] is False
        assert out["citable_as_target_scale"] is False

    def test_expected_surpasses_exits_zero(self):
        # The recording is pinned to SURPASSES: the --expected gate passes, so a
        # drift in the recorded floats (or the judge) fails loudly on this leg.
        assert main([str(FIXTURE_NEGCTRL_SURROGATE), "--expected", SURPASSES]) == 0

    def test_expected_ties_exits_nonzero(self):
        # Asserting the wrong verdict (TIES) on the SURPASSES recording exits
        # nonzero — the --expected gate distinguishes SURPASSES from TIES.
        assert main([str(FIXTURE_NEGCTRL_SURROGATE), "--expected", TIES]) == 2


# ---------------------------------------------------------------------------
# Citation gate derives negative-control from the budget artifact, not just the
# stored flag — the operator-set label must not override machine-checkable reality
# ---------------------------------------------------------------------------


class TestUnflaggedBudgetDivergence:
    """A degraded-arm deposit whose ``negative_control`` flag is unset must NOT
    be citable — the gate derives negative-control from the budget artifact, not
    only from the stored boolean.

    The producer (``scripts.run_freeze_validloss_ci``) DERIVES the flag from
    per-arm budget divergence, so a deposit it writes cannot degrade an arm
    without flagging it. But until this fix the replay citation gate re-read the
    STORED boolean alone, so a hand-edited or externally-supplied deposit —
    divergent ``candidate_total`` / ``surrogate_total`` with the flag absent —
    would silently read as a citable target-scale §4 result: the operator-set
    label trusted over the machine-checkable artifact reality, the same class as
    TASK-0152's hand-typed ``best_valid_loss`` and the swapped-arm guard
    (``5ed3380``). This class closes that: budget divergence is authoritative at
    the gate whether the flag says so or not.
    """

    def _unflagged_divergent(self, *, arm="candidate"):
        # A target-scale deposit (proxy_scale=False, not synthetic) whose arm was
        # degraded but whose negative_control flag is ABSENT — the inconsistent
        # state no committed fixture carries (every committed divergent-budget
        # deposit sets negative_control=True). Built directly so the budget fields
        # are present without the _data helper's flag default.
        d = _data([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0],
                  proxy_scale=False, synthetic=False, negative_control=False)
        d["total"] = 60
        if arm == "candidate":
            d["candidate_total"] = 2
            d["surrogate_total"] = 60
        elif arm == "surrogate":
            d["candidate_total"] = 60
            d["surrogate_total"] = 2
        return d

    def test_divergent_budget_unflagged_withholds_target_scale(self):
        # PRIMARY mutation proof: before the fix the flag was trusted and this
        # read True; the gate now sees the divergent candidate_total and withholds.
        data = self._unflagged_divergent()
        out = replay_to_json("<test>", data, replay_samples(data))
        assert out["citable_as_target_scale"] is False

    def test_divergent_budget_unflagged_withholds_full_section4(self):
        # The full-§4 gate is target-scale AND full_context; withholding
        # target-scale withholds the full verdict too, even at full context.
        data = self._unflagged_divergent()
        assert data.get("full_context", True) is True  # default = full context
        out = replay_to_json("<test>", data, replay_samples(data))
        assert out["citable_as_full_section4_verdict"] is False

    def test_effective_negative_control_surfaced_machine_readable(self):
        # The machine output surfaces the EFFECTIVE negative_control (True), not
        # the stale stored flag, so a consumer sees how the gate treated it and
        # which arm diverged.
        data = self._unflagged_divergent()
        out = replay_to_json("<test>", data, replay_samples(data))
        assert out["negative_control"] is True
        assert out["negative_control_arm"] == "candidate"

    def test_explicit_false_flag_with_divergence_still_withholds(self):
        # An explicit negative_control: false must NOT override the budget
        # reality — the operator cannot defeat the gate by setting the flag false.
        data = self._unflagged_divergent()
        data["negative_control"] = False  # explicit, not merely absent
        out = replay_to_json("<test>", data, replay_samples(data))
        assert out["citable_as_target_scale"] is False
        assert out["negative_control"] is True  # effective, not stored

    def test_surrogate_divergence_unflagged_also_withholds(self):
        # Symmetric: a surrogate-degraded unflagged deposit is also withheld and
        # names the surrogate arm.
        data = self._unflagged_divergent(arm="surrogate")
        out = replay_to_json("<test>", data, replay_samples(data))
        assert out["citable_as_target_scale"] is False
        assert out["negative_control_arm"] == "surrogate"

    def test_unflagged_divergence_note_is_emitted(self):
        # The inconsistency the stored flag hid surfaces LOUD in the human report
        # — a dedicated note names the divergence and the unset flag.
        data = self._unflagged_divergent()
        text = format_replay("<test>", data, replay_samples(data))
        assert "BUDGET_DIVERGENCE_UNFLAGGED" in text
        assert "not asserted" in text
        # The stale flag value (False) is shown so the operator sees the gap.
        assert "negative_control is False" in text

    def test_unflagged_divergence_does_not_claim_deliberate(self):
        # The unflagged note must NOT assert the degradation was deliberate — the
        # gate detects divergence but cannot know intent, so it withholds without
        # claiming a deliberate probe (the regular NEGATIVE_CONTROL note reserves
        # "deliberately degraded" for the explicitly-flagged case).
        data = self._unflagged_divergent()
        text = format_replay("<test>", data, replay_samples(data))
        assert "deliberately degraded" not in text

    def test_symmetric_budget_unflagged_remains_citable(self):
        # NO false positive: a symmetric-budget target-scale deposit with the
        # flag absent stays citable at both levels — the fix only fires on
        # divergence. Mirrors the committed real-9B target deposit (total=15,
        # candidate_total=15, surrogate_total=15, negative_control=false) which
        # carries the budget fields precisely so the gate can verify no divergence.
        d = _data([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0],
                  proxy_scale=False, synthetic=False, negative_control=False)
        d["total"] = 60
        d["candidate_total"] = 60
        d["surrogate_total"] = 60
        out = replay_to_json("<test>", d, replay_samples(d))
        assert out["citable_as_target_scale"] is True
        assert out["citable_as_full_section4_verdict"] is True
        assert out["negative_control"] is False

    def test_committed_real_9b_target_deposit_unchanged(self):
        # The committed real-9B seq256 deposit (symmetric 15/15/15 budgets, flag
        # false) is byte-identical under the fix: still citable target-scale.
        data = load_samples(FIXTURE_REAL_9B)
        out = replay_to_json(FIXTURE_REAL_9B, data, replay_samples(data))
        assert out["negative_control"] is False
        assert out["citable_as_target_scale"] is True
        assert out["citable_as_full_section4_verdict"] is False  # seq256, not full

    def test_every_committed_deposit_effective_equals_stored(self):
        # INVARIANT: no committed deposit carries the corrupt divergent-but-
        # unflagged state, so the effective status equals the stored flag for
        # every committed fixture — the fix is byte-identical across the corpus.
        # A future fixture that legitimately diverges without the flag would trip
        # this and force a deliberate decision rather than a silent flip.
        fixture_dir = Path(__file__).resolve().parent / "fixtures"
        any_checked = False
        for fixture in sorted(fixture_dir.glob("freeze_validloss_*.json")):
            if "runlog" in fixture.name:
                # A loss-curve artifact, not a deposit (no candidate/surrogate
                # sample lists); load_samples is not its schema.
                continue
            data = load_samples(fixture)
            out = replay_to_json(fixture, data, replay_samples(data))
            stored = bool(data.get("negative_control", False))
            assert out["negative_control"] is stored, (
                f"{fixture.name}: effective negative_control "
                f"({out['negative_control']}) != stored flag ({stored}) — this "
                f"fixture carries a divergent-but-unflagged budget"
            )
            any_checked = True
        assert any_checked  # guard against a glob that matched nothing


# ---------------------------------------------------------------------------
# Scale honesty + CLI assertion
# ---------------------------------------------------------------------------


class TestScaleAndCLI:
    def test_proxy_scale_surfaced_from_fixture(self):
        data = load_samples(FIXTURE)
        out = replay_to_json(FIXTURE, data, replay_samples(data))
        assert out["proxy_scale"] is True
        assert out["recorded_verdict"] == TIES
        assert out["replayed_verdict"] == TIES
        assert out["faithful"] is True

    def test_expected_match_exits_zero(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps(_data([1.0] * 4, [2.0] * 4)))
        assert main([str(f), "--expected", SURPASSES]) == 0

    def test_expected_mismatch_exits_nonzero(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps(_data([1.0] * 4, [2.0] * 4)))
        # The samples replay as SURPASSES; asserting TIES must fail loudly.
        assert main([str(f), "--expected", TIES]) == 2

    def test_expected_ties_on_real_fixture_exits_zero(self):
        assert main([str(FIXTURE), "--expected", TIES]) == 0

    def test_json_output_is_faithful(self, tmp_path, capsys):
        rc = main([str(FIXTURE), "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert payload["faithful"] is True
        assert payload["proxy_scale"] is True
        assert payload["replayed_verdict"] == TIES


# ---------------------------------------------------------------------------
# The target-scale drop-in (proxy_scale=false -> TARGET_SCALE, no code change)
# ---------------------------------------------------------------------------


class TestTargetScaleDropIn:
    """The MS-PF2 Cat-C drop-in: a ``proxy_scale: false`` sample file flips the
    replayed verdict to the TARGET scale label with NO code change — but a
    *synthetic* plumbing recording is never presentable as a citable §4 result.

    The committed plumbing fixture (synthetic floats, ``proxy_scale: false``,
    ``synthetic: true``) is the first recording that exercises the TARGET branch
    of the replay scale label — the path the real proxy fixture
    (``proxy_scale: true``) never touches. The scale label must flip PROXY ->
    TARGET and the stored floats must faithfully re-earn their recorded SURPASSES
    verdict; but because the floats are tagged ``synthetic: true``, the rendered
    note withholds the "this verdict IS the §4 target-scale result" claim a
    genuine 9B recording would earn and instead warns "do not cite". This is the
    feedback's "every committed verdict is still proxy-scale and must not be cited
    as a §4 target-scale result" guard, enforced in the rendered output rather
    than left to the fixture's prose. The genuine target-scale note — the one a
    real 9B run produces — is covered on a constructed ``synthetic: false``
    recording below. Together with ``TestTargetScaleParam`` (the generator sets
    ``proxy_scale`` rather than hardcoding it), this closes the "same schema, no
    code change" contract that had been asserted by docstrings and PURPOSE but
    never demonstrated — now with a machine-readable barrier between plumbing and
    measurement.
    """

    def test_target_fixture_is_target_scale_and_synthetic(self):
        data = load_samples(FIXTURE_TARGET)
        assert data["proxy_scale"] is False
        assert data["synthetic"] is True

    def test_synthetic_recording_withholds_citable_target_claim(self):
        # The scale line still flips to TARGET (proxy_scale=False), but the
        # synthetic tag withholds the citable "this verdict IS the §4
        # target-scale result" claim and emits a SYNTHETIC do-not-cite note.
        data = load_samples(FIXTURE_TARGET)
        text = format_replay(FIXTURE_TARGET, data, replay_samples(data))
        assert "TARGET_SCALE" in text  # scale line shows the TARGET label
        assert "proxy_scale=False" in text
        assert "synthetic=True" in text
        assert "SYNTHETIC" in text  # the do-not-cite note fires
        assert "this verdict IS" not in text  # citable claim withheld
        assert "PROXY_SCALE" not in text

    def test_genuine_target_scale_renders_citable_note(self):
        # The note a REAL 9B run earns: proxy_scale=False with genuine (non-
        # synthetic) floats -> "this verdict IS the §4 target-scale result".
        # This is the branch the committed plumbing fixture (synthetic) cannot
        # reach; it is tested here on a constructed genuine recording — the exact
        # shape a 9B run deposits — so the genuine claim is covered without the
        # private src.data pipeline.
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False)  # synthetic=False
        text = format_replay("<genuine-9b>", data, replay_samples(data))
        assert "TARGET_SCALE" in text
        assert "this verdict IS" in text  # the citable claim a real run earns
        assert "PROXY_SCALE" not in text
        assert "SYNTHETIC" not in text  # no synthetic warning on genuine floats

    def test_target_fixture_replays_to_recorded_surpasses(self):
        # A clear separation (candidate ~1.0 << surrogate ~2.0) -> CI entirely
        # above 0 -> SURPASSES, non-thin. The TARGET path is not a stub:
        # it actually judges the stored floats (synthetic provenance does not
        # change the verdict — only whether it may be cited).
        data = load_samples(FIXTURE_TARGET)
        ci = replay_samples(data)
        assert ci.significance_verdict == SURPASSES == data["verdict"]
        assert not ci.is_thin_evidence

    def test_replay_to_json_carries_proxy_scale_false_and_synthetic_true(self):
        data = load_samples(FIXTURE_TARGET)
        out = replay_to_json(FIXTURE_TARGET, data, replay_samples(data))
        assert out["proxy_scale"] is False
        assert out["synthetic"] is True  # machine-readable provenance
        assert out["recorded_verdict"] == SURPASSES
        assert out["replayed_verdict"] == SURPASSES
        assert out["faithful"] is True

    def test_expected_surpasses_on_target_fixture_exits_zero(self):
        # Synthetic provenance does not block verdict assertion: the plumbing
        # floats still faithfully replay to SURPASSES, which is the point.
        assert main([str(FIXTURE_TARGET), "--expected", SURPASSES]) == 0

    def test_json_output_carries_proxy_scale_false_and_synthetic(self, capsys):
        rc = main([str(FIXTURE_TARGET), "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert payload["proxy_scale"] is False
        assert payload["synthetic"] is True
        assert payload["replayed_verdict"] == SURPASSES

    def test_real_proxy_fixture_replays_as_genuine(self):
        # The committed real-GPU recording carries no synthetic field, so the
        # replay treats it as genuine (synthetic defaults False) — the honest
        # inverse of the plumbing fixture's synthetic=True.
        data = load_samples(FIXTURE)
        out = replay_to_json(FIXTURE, data, replay_samples(data))
        assert out["synthetic"] is False
        assert out["proxy_scale"] is True

    def test_synthetic_defaults_false_when_absent(self, tmp_path):
        # A recording with no synthetic field is genuine by default.
        f = tmp_path / "genuine.json"
        f.write_text(json.dumps(_data([1.0] * 4, [2.0] * 4)))  # no synthetic key
        out = replay_to_json(f, load_samples(f), replay_samples(load_samples(f)))
        assert out["synthetic"] is False

    def test_proxy_and_target_labels_are_distinguishable(self):
        # The core drop-in switch: the same judge, two recordings that differ
        # only in proxy_scale -> two different scale labels. The label upgrade
        # is driven by the file's proxy_scale flag, not by code (the TARGET
        # label shows in the scale line even though the plumbing fixture's note
        # is the SYNTHETIC guard rather than the citable claim).
        proxy_text = format_replay(
            FIXTURE, load_samples(FIXTURE), replay_samples(load_samples(FIXTURE))
        )
        target_text = format_replay(
            FIXTURE_TARGET,
            load_samples(FIXTURE_TARGET),
            replay_samples(load_samples(FIXTURE_TARGET)),
        )
        assert "PROXY_SCALE" in proxy_text and "TARGET_SCALE" not in proxy_text
        assert "TARGET_SCALE" in target_text and "PROXY_SCALE" not in target_text


# ---------------------------------------------------------------------------
# Machine-readable citation gate (citable_as_target_scale)
# ---------------------------------------------------------------------------


class TestMachineCitationGate:
    """The citability contract on the machine-readable (JSON) path.

    The previous iteration gated the citable "this verdict IS the §4
    target-scale result" claim on the *human-readable* output (withheld for any
    proxy or synthetic recording). ``citable_as_target_scale`` mirrors that prose
    rule into a single boolean on the JSON path, so a downstream consumer does
    not have to infer citability from the raw ``proxy_scale`` / ``synthetic``
    flags — the feedback's "must not be cited as a §4 target-scale result"
    warning, enforced as a field rather than prose. These tests pin the gate for
    every recording type and prove it can never drift from the human claim.
    """

    def test_genuine_target_scale_recording_is_citable(self):
        # The one recording shape a real 9B run deposits: proxy_scale=False,
        # genuine floats (no synthetic). It alone earns the citation gate.
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False)  # synthetic=False
        out = replay_to_json("<genuine-9b>", data, replay_samples(data))
        assert out["citable_as_target_scale"] is True

    def test_committed_proxy_recording_is_not_citable(self):
        # The real-GPU proxy recording (proxy_scale=True) must never be citable
        # as a target-scale result — the core feedback constraint.
        data = load_samples(FIXTURE)
        out = replay_to_json(FIXTURE, data, replay_samples(data))
        assert out["proxy_scale"] is True
        assert out["citable_as_target_scale"] is False

    def test_synthetic_plumbing_is_not_citable_even_at_target_scale(self):
        # The drop-in plumbing fixture flips the scale label to TARGET
        # (proxy_scale=False) but its floats are synthetic — so the scale IS
        # target-scale yet the verdict is NOT citable. This is the exact shape
        # that could fool a consumer reading proxy_scale alone; the gate refuses
        # it via the synthetic flag.
        data = load_samples(FIXTURE_TARGET)
        out = replay_to_json(FIXTURE_TARGET, data, replay_samples(data))
        assert out["proxy_scale"] is False       # scale label IS target...
        assert out["synthetic"] is True
        assert out["citable_as_target_scale"] is False  # ...but not citable

    def test_machine_gate_matches_human_prose_across_all_recordings(self):
        # Two citation levels, each pinned to its prose claim so machine and prose
        # cannot drift: citable_as_target_scale == "samples are from a 9B run";
        # citable_as_full_section4_verdict == "this verdict IS" in prose.
        cases = [
            ("<genuine-9b-full>", _data([1.0] * 4, [2.0] * 4, proxy_scale=False)),
            # reduced-context 9B probe: real 9B floats, full_context=False.
            ("<genuine-9b-reduced>", _data(
                [1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=False)),
            (str(FIXTURE), load_samples(FIXTURE)),
            (str(FIXTURE_TARGET), load_samples(FIXTURE_TARGET)),
            # negative-control at target scale: a sensitivity probe, not citable.
            ("<negctrl-target>", _data(
                [2.0] * 4, [1.0] * 4, proxy_scale=False, negative_control=True)),
            (str(FIXTURE_NEGCTRL), load_samples(FIXTURE_NEGCTRL)),
            (str(FIXTURE_NEGCTRL_SURROGATE), load_samples(FIXTURE_NEGCTRL_SURROGATE)),
        ]
        for label, data in cases:
            ci = replay_samples(data)
            out = replay_to_json(label, data, ci)
            prose = format_replay(label, data, ci)
            assert out["citable_as_full_section4_verdict"] is ("this verdict IS" in prose)
            assert out["citable_as_target_scale"] is ("samples are from a 9B run" in prose)


# Reduced-context provenance guard (the seq_len=256 probe honesty axis)


class TestReducedContextProvenanceGuard:
    """A reduced-context (seq_len=256) 9B probe IS target-scale but NOT the full
    §4 seq_len=1024 verdict (TASK-0152 lines 86-97). The guard splits the two
    claims: judged faithfully and labeled target-scale, but the "this verdict IS
    the §4 target-scale result" claim is withheld until a full-context recording
    overwrites the deposit."""

    def test_reduced_context_probe_is_target_scale_but_not_full_verdict(self):
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=False)
        out = replay_to_json("<9b-reduced>", data, replay_samples(data))
        text = format_replay("<9b-reduced>", data, replay_samples(data))
        assert out["citable_as_target_scale"] is True   # context is independent of scale
        assert out["full_context"] is False
        assert out["citable_as_full_section4_verdict"] is False  # strong claim withheld
        assert "this verdict IS" not in text
        assert "REDUCED CONTEXT" in text
        assert "not the full" in text.lower()

    def test_full_context_probe_grants_full_verdict_claim(self):
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=True)
        out = replay_to_json("<9b-full>", data, replay_samples(data))
        text = format_replay("<9b-full>", data, replay_samples(data))
        assert out["citable_as_full_section4_verdict"] is True
        assert "this verdict IS" in text  # the citable claim a full run earns
        assert "REDUCED CONTEXT" not in text

    def test_reduced_context_defaults_to_full_when_field_absent(self):
        # Backward compat: a legacy deposit with no full_context field is treated
        # as full-context (existing fixtures omit it; they're non-citable anyway).
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False)
        data.pop("full_context", None)
        out = replay_to_json("<legacy-9b>", data, replay_samples(data))
        assert out["full_context"] is True
        assert out["citable_as_full_section4_verdict"] is True

    def test_reduced_context_note_names_seq_len_when_recorded(self):
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=False)
        data["seq_len"] = 256
        text = format_replay("<9b-256>", data, replay_samples(data))
        assert "seq_len=256" in text  # the caveat states the exact shortfall

    def test_reduced_context_does_not_block_verdict_recomputation(self):
        # The guard withholds the claim, never the verdict.
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=False)
        assert replay_samples(data).significance_verdict == SURPASSES


# Real 9B target-scale deposit (the feedback's "feed REAL numbers through it")


class TestRealTargetScale9BDeposit:
    """The committed REAL 9B target-scale deposit — the move the loop's feedback
    asked for in place of a fifth provenance guard: feed REAL numbers through the
    gate rather than leaving '9B §4 verdict pending' behind more guards.

    The deposit carries genuine ``best_valid_loss`` from multi-seed 9B runs
    (candidate TG-LoRA vs full-backprop baseline, seq_len=256 — the only config a
    12GB RTX 3060 fits), harvested from upstream ``run_metrics.jsonl`` by
    ``form_freeze_validloss_deposit``. Every prior target-scale fixture is
    synthetic plumbing; this is the first recording to exercise the
    ``citable_as_target_scale`` gate and the reduced-context guard on REAL data,
    not a single upstream candidate and not hand-authored floats.
    """

    def test_fixture_is_genuine_real_target_scale(self):
        data = load_samples(FIXTURE_REAL_9B)
        # Genuine target-scale recording — not proxy plumbing, not synthetic, not
        # a negative control. A real 9B run on the constitutional model.
        assert data["proxy_scale"] is False
        assert data["synthetic"] is False
        assert data["negative_control"] is False
        assert data["model"] == "Qwen/Qwen3.5-9B"

    def test_provenance_guard_fires_on_real_data(self):
        # The two-level citability gate exercised on REAL numbers: a 9B run IS
        # target-scale, but the seq_len=256 probe is NOT the full §4 verdict.
        data = load_samples(FIXTURE_REAL_9B)
        out = replay_to_json(FIXTURE_REAL_9B, data, replay_samples(data))
        assert out["citable_as_target_scale"] is True
        assert out["full_context"] is False
        assert out["seq_len"] == 256
        assert out["citable_as_full_section4_verdict"] is False

    def test_losses_are_real_9b_floats_multi_seed(self):
        data = load_samples(FIXTURE_REAL_9B)
        # Multi-seed on BOTH arms (not a single upstream candidate); real 9B
        # next-token CE losses, not synthetic [1.0, 1.1, 0.9, ...] plumbing.
        assert len(data["candidate_losses"]) >= 3
        assert len(data["surrogate_losses"]) >= 3
        seen = set()
        for v in data["candidate_losses"] + data["surrogate_losses"]:
            assert isinstance(v, float)
            assert 0.5 < v < 3.0  # sane 9B next-token CE-loss range
            seen.add(round(v, 4))
        # Real measurements are not the synthetic 0.1-spaced grid [1.0,1.1,...].
        assert seen != {1.0, 1.1, 0.9, 1.05, 0.95}

    def test_real_verdict_is_pinned_and_faithful(self):
        # The deposit's recorded verdict must reproduce under the deterministic
        # bootstrap — the stored REAL floats earn the verdict, it is not painted.
        data = load_samples(FIXTURE_REAL_9B)
        ci = replay_samples(data)
        out = replay_to_json(FIXTURE_REAL_9B, data, ci)
        assert ci.significance_verdict in (SURPASSES, TIES, UNDERSHOOTS)
        assert out["replayed_verdict"] == data["verdict"]
        assert out["faithful"] is True

    def test_real_verdict_pins_literal_ties_and_ci_bounds(self):
        # The faithfulness test above asserts the replayed verdict matches the
        # deposit's OWN ``verdict`` field — an internal-consistency check. It
        # cannot catch a coordinated change (losses re-harvested to a different
        # real measurement AND the ``verdict`` field repainted to match), which
        # would let the cited scientific result drift silently. Pin the ABSOLUTE
        # result instead: the real 9B seq256 verdict is TIES at these specific
        # magnitudes, with a CI that straddles zero. If the deposit floats ever
        # shift (an upstream re-run) or the bootstrap math changes, this forces a
        # deliberate update to the cited numbers — the checked-in scientific
        # claim becomes a durable regression invariant, not merely reproducible.
        data = load_samples(FIXTURE_REAL_9B)
        out = replay_to_json(FIXTURE_REAL_9B, data, replay_samples(data))
        # The verdict is the literal TIES (the recorded §0 quality-parity null),
        # not whatever the file happens to claim.
        assert out["replayed_verdict"] == TIES
        # The structural reason TIES holds: the bootstrap CI crosses zero.
        assert out["lower"] < 0.0 < out["upper"]
        # The exact magnitudes cited in the deposit commit / PURPOSE milestone
        # #10 — candidate TG-LoRA vs full-backprop baseline.
        assert round(out["candidate_mean"], 4) == 1.0510
        assert round(out["surrogate_mean"], 4) == 1.0438
        assert round(out["lower"], 4) == -0.0205
        assert round(out["upper"], 4) == 0.0018
        # Non-thin (n=3/3 on both arms) and not a material win — a parity null,
        # not noise and not a (small) TG-LoRA victory.
        assert out["is_thin_evidence"] is False
        assert out["is_material"] is False

