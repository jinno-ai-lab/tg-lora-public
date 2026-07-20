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


class TestFullContextSeqLenOverridesFlag:
    """The citation gate derives ``full_context`` from the artifact (``seq_len``)
    over the operator-set flag — the full-context sibling of
    ``_negative_control_active`` / ``9dff092``. A hand-edited or externally-
    supplied deposit (the private-``src.data`` 9B drop-in path this harness
    replays) that trained at reduced context must not be over-cited as the full
    §4 verdict because the ``full_context`` flag is absent (default True) or
    stale-True; ``seq_len`` is authoritative. Each test pins one branch of the
    derivation and is mutation-proven against "trust the stored boolean alone"."""

    def test_seq_len_overrides_absent_full_context_default_at_gate(self):
        # The corruption this guard closes: a genuine target-scale recording that
        # OMITS full_context (defaulting True) yet trained at seq_len=256. Trusting
        # the default would over-cite it as the FULL §4 verdict; the artifact must
        # withhold the claim.
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False)  # full_context absent
        assert "full_context" not in data  # self-check: the default path IS the gap
        data["seq_len"] = 256
        out = replay_to_json("<9b-gap>", data, replay_samples(data))
        assert out["full_context"] is False             # derived from seq_len, not default
        assert out["citable_as_target_scale"] is True   # genuine 9B is still target-scale
        assert out["citable_as_full_section4_verdict"] is False  # NOT the full verdict

    def test_seq_len_overrides_explicit_false_full_context_flag(self):
        # Symmetric: seq_len>=1024 overrides an explicit (stale) full_context=False,
        # so a full-context run cannot be UNDER-cited because of a stale label.
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=False)
        data["seq_len"] = 1024
        out = replay_to_json("<9b-overrides>", data, replay_samples(data))
        assert out["full_context"] is True              # derived from seq_len over the flag
        assert out["citable_as_full_section4_verdict"] is True

    def test_explicit_full_context_true_refuted_by_seq_len_surfaces_loud(self):
        # An operator over-claim (full_context=True) contradicted by seq_len=256:
        # the gate withholds from the artifact and surfaces the contradiction loud
        # (mirrors BUDGET_DIVERGENCE_UNFLAGGED), never silently trusting the label.
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=True)
        data["full_context"] = True  # explicitly assert the over-claim
        data["seq_len"] = 256
        out = replay_to_json("<9b-overclaim>", data, replay_samples(data))
        text = format_replay("<9b-overclaim>", data, replay_samples(data))
        assert out["full_context"] is False
        assert out["citable_as_full_section4_verdict"] is False
        assert "FULL_CONTEXT_FLAG_REFUTED" in text
        assert "seq_len=256" in text

    def test_stray_boolean_in_seq_len_does_not_count_as_context_length(self):
        # A stray True written into seq_len must fall back to the flag, not be read
        # as a count (bool is a subclass of int — the predicate excludes it).
        data = _data([1.0] * 4, [2.0] * 4, proxy_scale=False, full_context=True)
        data["seq_len"] = True
        out = replay_to_json("<9b-bool>", data, replay_samples(data))
        assert out["full_context"] is True  # falls back to the absent-flag default


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


# ---------------------------------------------------------------------------
# The 9B honesty-schema four-axis gate (budget / thin / regime), re-derived
# ---------------------------------------------------------------------------


def _9b_deposit_fixtures():
    """Every committed 9B §4 deposit (the runlog artifact is NOT a deposit)."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    return sorted(
        p for p in fixture_dir.glob("freeze_validloss_ci_9b_*.json")
        if "runlog" not in p.name
    )


class TestNineBHonestyGateReplay:
    """The replay's ``citable_as_full_section4_verdict`` must reproduce the
    PRODUCER's four-axis gate (target-scale + full-budget + non-thin +
    generalization) from a 9B deposit's artifacts — not the 2-axis (scale +
    context) gate it used before, which over-claimed every reduced-budget /
    non-generalization 9B deposit as the COMPLETE §4 verdict.

    Before this fix the replay gate honored only target-scale + full_context, so
    all five committed reduced-budget 9B deposits (``total_steps`` 20/96 ≪
    ``cfg_max_steps`` 1500) read ``citable_as_full_section4_verdict=True`` even
    though the producer stamped ``False`` — the silent over-claim this class
    closes. The fix re-derives the producer's budget / thin / regime axes from
    the deposit's stored artifacts via the shared ``freeze_verdict_honesty``
    leaf (the SAME logic the producer stamps), and a ``CITATION_LABEL_STALE``
    cross-check flags any deposit whose stored boolean disagrees with the
    artifact-rederived value. This is the citation-gate sibling of
    ``_negative_control_active`` (``9dff092``) and ``_full_context_effective``
    (``bbf6e68``): the gate trusts the machine-checkable artifact reality over
    the operator-set / stored label.
    """

    def test_reduced_budget_9b_deposits_are_not_citable_as_full_verdict(self):
        # PRIMARY mutation proof: the five reduced-budget 9B deposits (20/96
        # steps vs cfg_max_steps=1500) are NOT citable as the complete §4
        # verdict. Before the fix the replay gate honored only scale+context and
        # read these True; the budget axis (re-derived from total_steps vs
        # cfg_max_steps) now withholds — reverting the gate to the 2-axis form
        # flips these back to True and fails this test.
        reduced = [
            p for p in _9b_deposit_fixtures()
            if load_samples(p).get("reduced_budget") is True
        ]
        assert len(reduced) == 5, "expected exactly five reduced 9B deposits"
        for fixture in reduced:
            data = load_samples(fixture)
            out = replay_to_json(fixture, data, replay_samples(data))
            assert out["citable_as_full_section4_verdict"] is False, (
                f"{fixture.name}: reduced-budget deposit must NOT be citable as "
                "the complete §4 verdict (budget axis failed)"
            )
            assert "budget" in out["producer_honesty_axis_failures"], (
                f"{fixture.name}: the budget axis must be named as the failing axis"
            )

    def test_full_budget_generalizing_9b_deposits_remain_citable(self):
        # The two full-budget (1500==1500), non-thin (n=3/3), generalizing 9B
        # deposits stay citable — the fix is byte-identical on the genuine
        # complete-§4 results. (A regression that over-withholds would fail here.)
        full = [
            p for p in _9b_deposit_fixtures()
            if load_samples(p).get("reduced_budget") is False
        ]
        assert len(full) == 2, "expected exactly two full 9B deposits"
        for fixture in full:
            data = load_samples(fixture)
            out = replay_to_json(fixture, data, replay_samples(data))
            assert out["citable_as_full_section4_verdict"] is True, (
                f"{fixture.name}: full-budget generalizing deposit stays citable"
            )
            assert out["producer_honesty_axis_failures"] == []

    def test_every_committed_9b_deposit_effective_equals_stored(self):
        # INVARIANT: the replay's artifact-rederived verdict equals the producer's
        # STORED boolean for every committed 9B deposit, and no deposit carries a
        # stale label — the fix is byte-identical across the corpus. (Mirrors
        # test_every_committed_deposit_effective_equals_stored for the
        # negative-control axis.) A future deposit that legitimately diverges
        # would trip this and force a deliberate decision rather than a silent
        # over-claim — exactly the silent-corruption path this guard closes.
        checked = 0
        for fixture in _9b_deposit_fixtures():
            data = load_samples(fixture)
            out = replay_to_json(fixture, data, replay_samples(data))
            stored = data.get("citable_as_full_section4_verdict")
            assert stored is not None, f"{fixture.name}: 9B deposit must stamp the boolean"
            assert out["citable_as_full_section4_verdict"] is bool(stored), (
                f"{fixture.name}: effective ({out['citable_as_full_section4_verdict']}) "
                f"!= stored ({stored}) — the replay gate drifted from the producer"
            )
            assert out["citation_label_stale"] is False, (
                f"{fixture.name}: committed deposit must carry no stale label"
            )
            checked += 1
        assert checked >= 7  # all 7 committed 9B deposits

    def test_prose_invariant_holds_across_committed_9b_deposits(self):
        # The machine gate and the human prose claim cannot drift: the effective
        # verdict equals whether the prose contains "this verdict IS" for every
        # committed 9B deposit — including the new target-scale+full-context-but-
        # axis-failed branch (which withholds the strong claim and names the
        # failing axis instead). Extends test_machine_gate_matches_human_prose
        # to the 9B honesty-schema deposits it never iterated before.
        for fixture in _9b_deposit_fixtures():
            data = load_samples(fixture)
            ci = replay_samples(data)
            out = replay_to_json(fixture, data, ci)
            prose = format_replay(fixture, data, ci)
            assert out["citable_as_full_section4_verdict"] is (
                "this verdict IS" in prose
            ), f"{fixture.name}: machine gate != prose claim"

    def test_hand_edited_stale_true_label_overridden_and_flagged(self):
        # MUTATION PROOF (over-claim direction): a reduced-budget 9B deposit whose
        # stored boolean is hand-edited to True (a stale over-claim) must STILL
        # read effective=False — the gate derives the answer from the budget
        # artifact (total_steps < cfg_max_steps), not the stored label — and the
        # CITATION_LABEL_STALE cross-check fires in BOTH the machine field and the
        # prose note. Reverting the gate to trust the stored boolean makes
        # effective follow the lie (True); removing the cross-check drops the flag.
        data = load_samples(_9b_deposit_fixtures()[0])  # a reduced-budget deposit
        assert data.get("reduced_budget") is True
        data["citable_as_full_section4_verdict"] = True  # the stale over-claim
        ci = replay_samples(data)
        out = replay_to_json("<stale-true>", data, ci)
        assert out["citable_as_full_section4_verdict"] is False  # artifacts win
        assert out["citation_label_stale"] is True
        prose = format_replay("<stale-true>", data, ci)
        assert "CITATION_LABEL_STALE" in prose
        assert "NOT citable" in prose

    def test_hand_edited_stale_false_label_overridden_and_flagged(self):
        # MUTATION PROOF (under-claim direction, symmetric): a full-budget
        # generalizing 9B deposit whose stored boolean is hand-edited to False (a
        # stale under-claim) must STILL read effective=True — the artifacts
        # (full budget, non-thin, generalization) earn the verdict — and the
        # cross-check fires. Proves the gate derives from artifacts in BOTH
        # directions, not just the over-claim one.
        full = [
            p for p in _9b_deposit_fixtures()
            if load_samples(p).get("reduced_budget") is False
        ][0]
        data = load_samples(full)
        data["citable_as_full_section4_verdict"] = False  # the stale under-claim
        ci = replay_samples(data)
        out = replay_to_json("<stale-false>", data, ci)
        assert out["citable_as_full_section4_verdict"] is True  # artifacts win
        assert out["citation_label_stale"] is True
        assert "CITATION_LABEL_STALE" in format_replay("<stale-false>", data, ci)

    def test_failing_axis_named_in_prose_for_reduced_deposit(self):
        # A reduced 9B deposit is target-scale + full-context (seq_len=1024), so
        # it reaches the NEW prose branch: the strong "this verdict IS" claim is
        # withheld and the failing axis is named, so a reader sees WHY the
        # four-axis gate stayed closed on a recording that looks complete.
        data = load_samples(_9b_deposit_fixtures()[0])  # baseline: budget + regime
        prose = format_replay("<reduced-9b>", data, replay_samples(data))
        assert "this verdict IS" not in prose  # strong claim withheld
        assert "NOT citable as the COMPLETE §4 verdict" in prose
        assert "budget" in prose  # the failing axis is named

    def test_thin_9b_schema_deposit_withheld_on_thin_axis(self):
        # MUTATION PROOF (thin axis): a 9B-schema deposit that is full-budget and
        # generalizing but THIN (< MIN_SAMPLE_FOR_BOOTSTRAP seeds in an arm) is
        # withheld on the thin axis — a second producer axis the old 2-conjunct
        # gate could not see. Built by trimming a full deposit's samples to 2/arm.
        full = [
            p for p in _9b_deposit_fixtures()
            if load_samples(p).get("reduced_budget") is False
        ][0]
        data = load_samples(full)
        data["candidate_losses"] = data["candidate_losses"][:2]  # thin arm
        data["surrogate_losses"] = data["surrogate_losses"][:2]
        out = replay_to_json("<thin-9b>", data, replay_samples(data))
        assert out["citable_as_full_section4_verdict"] is False
        assert "thin" in out["producer_honesty_axis_failures"]
        assert out["is_thin_evidence"] is True

    def test_non_generalization_9b_schema_deposit_withheld_on_regime_axis(self):
        # MUTATION PROOF (regime axis): a 9B-schema deposit that is full-budget
        # and non-thin but whose candidate memorized (train CE ≈ 0) is withheld on
        # the regime axis — the third producer axis. The producer's
        # classify_regime (shared leaf) reads the artifact over any stored label.
        full = [
            p for p in _9b_deposit_fixtures()
            if load_samples(p).get("reduced_budget") is False
        ][0]
        data = load_samples(full)
        data["candidate_final_ce_train_loss_mean"] = 0.01  # memorized: CE ≈ 0
        out = replay_to_json("<memorized-9b>", data, replay_samples(data))
        assert out["citable_as_full_section4_verdict"] is False
        assert any(f.startswith("regime=") for f in out["producer_honesty_axis_failures"])

    def test_schema_marker_governs_when_axes_honored(self):
        # The producer axes are honored ONLY when the deposit carries the 9B
        # honesty schema (cfg_max_steps / candidate_final_ce_train_loss_mean /
        # regime). A target-scale + full-context recording WITHOUT the schema
        # (a proxy-style / legacy recording) uses the scale+context gate alone —
        # backward compatible, never over-withheld for lack of artifacts it
        # never carried. A recording WITH the schema but reduced IS withheld.
        from scripts.replay_freeze_validloss_ci import _carries_9b_honesty_schema

        no_schema = _data([1.0] * 4, [2.0] * 4, proxy_scale=False)  # no 9B fields
        assert _carries_9b_honesty_schema(no_schema) is False
        out_no = replay_to_json("<no-schema>", no_schema, replay_samples(no_schema))
        assert out_no["citable_as_full_section4_verdict"] is True  # old gate
        assert out_no["producer_honesty_axis_failures"] == []

        # Same recording shape but WITH a reduced 9B budget stamp -> withheld.
        with_schema = dict(no_schema)
        with_schema["cfg_max_steps"] = 1500
        with_schema["total_steps"] = 20  # reduced
        assert _carries_9b_honesty_schema(with_schema) is True
        out_yes = replay_to_json("<with-schema>", with_schema, replay_samples(with_schema))
        assert out_yes["citable_as_full_section4_verdict"] is False
        assert "budget" in out_yes["producer_honesty_axis_failures"]


class TestSubVerdictLabelStaleness:
    """The replay must re-derive a 9B deposit's ``direction`` / ``baseline``
    sub-verdicts from its stored per-arm losses (``control_losses`` /
    ``baseline_losses``) with the producer's seed — and flag a stored sub-verdict
    label that disagrees — rather than silently trusting the nested label.

    The producer (:func:`scripts.run_freeze_validloss_ci_9b.run_ci_9b`,
    ``run_freeze_validloss_ci_9b.py`` lines ~1715-1727) computes the
    direction-isolation CI (candidate vs input-contiguous control) and the
    full-backprop baseline CI (candidate vs no-freeze full-CE) with
    ``seed=base_seed`` and stamps each as ``direction.verdict`` /
    ``baseline.verdict``. Those losses are deposited alongside the label, so the
    verdict is *re-derivable* GPU-free. Before this fix the replay gate never
    re-derived them: an arbitrary (hand-edited or externally-supplied) deposit
    whose ``baseline.verdict`` lied about its ``baseline_losses`` passed
    completely silently — ``faithful: True`` (the lie is in the nested label, not
    the main ``verdict``), no stale field, no prose note. This is the SAME
    "stored-label-trusted-over-artifact-reality" class as ``_citation_label_stale``
    (``d734327``) and its budget / full-context siblings (``9dff092`` /
    ``bbf6e68``), now extended to the two §4 condition-(a)/(b) sub-verdicts.

    The fix re-derives each sub-verdict from the stored floats (the deterministic
    candidate-vs-arm bootstrap under ``base_seed``), cross-checks it against the
    stored label, surfaces ``direction_verdict_stale`` / ``baseline_verdict_stale``
    in :func:`replay_to_json`, and emits ``DIRECTION_VERDICT_STALE`` /
    ``BASELINE_VERDICT_STALE`` notes in :func:`format_replay`.
    """

    @staticmethod
    def _other_label(true_label):
        # A label provably different from the true one, so the lie can never
        # accidentally match the re-derived verdict.
        return TIES if true_label != TIES else UNDERSHOOTS

    def test_committed_deposits_sub_verdicts_match_losses(self):
        # BYTE-IDENTICAL invariant: for every committed 9B deposit, the replay's
        # artifact-rederived sub-verdict equals the producer's STORED label
        # wherever the arm ran, both stale flags are False, and neither stale
        # note appears in prose. A future deposit that legitimately diverged
        # would trip this and force a deliberate decision rather than a silent
        # over-claim — exactly the silent-corruption path this guard closes.
        # (Mirrors test_every_committed_9b_deposit_effective_equals_stored for the
        # sub-verdict axis.) Reverting the re-derivation (trusting the stored
        # label) makes rederived follow any lie, so this still holds — the
        # MUTATION resistance is the two hand-edited tests below.
        for fixture in _9b_deposit_fixtures():
            data = load_samples(fixture)
            ci = replay_samples(data)
            out = replay_to_json(fixture, data, ci)
            prose = format_replay(fixture, data, ci)
            for slot, lk in (
                ("direction", "control_losses"),
                ("baseline", "baseline_losses"),
            ):
                sub = data.get(slot)
                if not isinstance(sub, dict):
                    # Arm did not run (slot null): nothing to cross-check.
                    assert out[f"{slot}_verdict_rederived"] is None
                    assert out[f"{slot}_verdict_stale"] is False
                    assert f"{slot.upper()}_VERDICT_STALE" not in prose
                    continue
                # Arm ran: re-derived verdict reproduces the stored label exactly.
                assert out[f"{slot}_verdict_rederived"] == sub["verdict"], (
                    f"{fixture.name}: {slot} rederived "
                    f"({out[slot + '_verdict_rederived']!r}) != stored "
                    f"({sub['verdict']!r})"
                )
                assert out[f"{slot}_verdict_stale"] is False
                assert f"{slot.upper()}_VERDICT_STALE" not in prose

    def test_hand_edited_lying_baseline_verdict_caught_and_overridden(self):
        # MUTATION PROOF (baseline axis): a deposit whose stored ``baseline.verdict``
        # label is hand-edited to a lie must be caught — the replay re-derives the
        # verdict from ``baseline_losses`` (the same candidate-vs-baseline bootstrap
        # the producer computed the sub-CI with, under base_seed), reports the TRUE
        # verdict in ``baseline_verdict_rederived``, sets ``baseline_verdict_stale``
        # True, and emits the BASELINE_VERDICT_STALE prose note. Reverting the fix
        # (trust the stored label) makes rederived equal the lie and stale False.
        # Uses the committed full-budget deposit, which has a real baseline arm.
        data = load_samples(
            [p for p in _9b_deposit_fixtures() if p.name.endswith("_full.json")][0]
        )
        assert isinstance(data.get("baseline"), dict)  # the arm ran
        true = surrogate_valid_loss_ci(
            data["candidate_losses"], data["baseline_losses"],
            seed=int(data["base_seed"]),
        ).significance_verdict
        data["baseline"]["verdict"] = self._other_label(true)  # the lie
        ci = replay_samples(data)
        out = replay_to_json("<lie-baseline>", data, ci)
        prose = format_replay("<lie-baseline>", data, ci)
        assert out["baseline_verdict_rederived"] == true  # artifacts win
        assert out["baseline_verdict_stale"] is True
        assert "BASELINE_VERDICT_STALE" in prose
        assert f"{true!r}" in prose  # the true verdict is named in the note

    def test_hand_edited_lying_direction_verdict_caught_and_overridden(self):
        # MUTATION PROOF (direction axis, symmetric): a deposit whose stored
        # ``direction.verdict`` label is hand-edited to a lie must be caught on the
        # direction-isolation arm too — re-derived from ``control_losses``. Proves
        # the cross-check covers BOTH sub-verdicts, not just baseline. Uses the
        # committed direction deposit, which has a real control arm.
        data = load_samples(
            [p for p in _9b_deposit_fixtures() if "direction" in p.name][0]
        )
        assert isinstance(data.get("direction"), dict)  # the arm ran
        true = surrogate_valid_loss_ci(
            data["candidate_losses"], data["control_losses"],
            seed=int(data["base_seed"]),
        ).significance_verdict
        data["direction"]["verdict"] = self._other_label(true)  # the lie
        ci = replay_samples(data)
        out = replay_to_json("<lie-direction>", data, ci)
        prose = format_replay("<lie-direction>", data, ci)
        assert out["direction_verdict_rederived"] == true  # artifacts win
        assert out["direction_verdict_stale"] is True
        assert "DIRECTION_VERDICT_STALE" in prose

    def test_sub_verdict_rederivation_uses_producer_base_seed(self):
        # CONTRACT: the replay re-derives a sub-verdict with the deposit's
        # ``base_seed`` (the seed the producer computed the sub-CI with), NOT the
        # bootstrap default. Pins the seed so a deposit with a non-default seed
        # re-derives identically to how it was stamped.
        from scripts.replay_freeze_validloss_ci import _subverdict_rederived

        data = load_samples(
            [p for p in _9b_deposit_fixtures() if p.name.endswith("_full.json")][0]
        )
        seed = int(data["base_seed"])
        direct = surrogate_valid_loss_ci(
            data["candidate_losses"], data["baseline_losses"], seed=seed,
        ).significance_verdict
        assert _subverdict_rederived(data, losses_key="baseline_losses") == direct

    def test_sub_verdict_skipped_when_arm_did_not_run(self):
        # A deposit that did not run a control / baseline arm carries ``null`` for
        # that slot (the producer stamps None when n_control=0 / n_baseline=0).
        # The cross-check must read rederived=None / stale=False and emit NO note —
        # it must not false-positive on a legitimately-absent arm. full.json has
        # direction=None; direction.json has baseline=None.
        full = load_samples(
            [p for p in _9b_deposit_fixtures() if p.name.endswith("_full.json")][0]
        )
        assert full.get("direction") is None  # no control arm
        out_f = replay_to_json("<full>", full, replay_samples(full))
        assert out_f["direction_verdict_rederived"] is None
        assert out_f["direction_verdict_stale"] is False
        assert "DIRECTION_VERDICT_STALE" not in format_replay("<full>", full, replay_samples(full))

        direction = load_samples(
            [p for p in _9b_deposit_fixtures() if "direction" in p.name][0]
        )
        assert direction.get("baseline") is None  # no baseline arm
        out_d = replay_to_json("<direction>", direction, replay_samples(direction))
        assert out_d["baseline_verdict_rederived"] is None
        assert out_d["baseline_verdict_stale"] is False

    def test_prose_note_presence_matches_machine_stale_flag(self):
        # DRIFT invariant (mirrors test_prose_invariant_holds_across_committed_
        # 9b_deposits): the machine stale flag and the human prose note cannot
        # drift — for every committed deposit AND for a hand-edited lie on each
        # slot, ``<SLOT>_VERDICT_STALE in prose`` equals the JSON stale flag.
        cases = [("committed", load_samples(p)) for p in _9b_deposit_fixtures()]
        # Add the two mutated cases (one lie per slot).
        full = load_samples(
            [p for p in _9b_deposit_fixtures() if p.name.endswith("_full.json")][0]
        )
        full["baseline"]["verdict"] = self._other_label(
            surrogate_valid_loss_ci(
                full["candidate_losses"], full["baseline_losses"],
                seed=int(full["base_seed"]),
            ).significance_verdict
        )
        cases.append(("lie-baseline", full))
        direction = load_samples(
            [p for p in _9b_deposit_fixtures() if "direction" in p.name][0]
        )
        direction["direction"]["verdict"] = self._other_label(
            surrogate_valid_loss_ci(
                direction["candidate_losses"], direction["control_losses"],
                seed=int(direction["base_seed"]),
            ).significance_verdict
        )
        cases.append(("lie-direction", direction))
        for name, data in cases:
            ci = replay_samples(data)
            out = replay_to_json(f"<{name}>", data, ci)
            prose = format_replay(f"<{name}>", data, ci)
            for slot in ("direction", "baseline"):
                flag = out[f"{slot}_verdict_stale"]
                note = f"{slot.upper()}_VERDICT_STALE" in prose
                assert flag is note, (
                    f"{name}: {slot} stale flag ({flag}) != prose note ({note})"
                )


class TestLedgerBinding:
    """The deposit-vs-ledger binding: the replay re-derives the §4 verdict from a
    deposit's per-arm losses, but the committed ledger (``ledger_witness_path``)
    is the ground-truth record each arm's ``valid_loss`` was harvested from. The
    replay now cross-checks the deposit's cited losses AND its stamped witness
    hash against that ledger — the deposit-vs-ledger sibling of the intra-deposit
    label-vs-artifact guards (``371e934`` direction/baseline, ``d734327`` the four
    citation axes).

    The gap this closes: a hand-edited deposit that is *internally* self-consistent
    — its ``evidence_hash`` matches its edited fields, its recorded verdict matches
    its edited losses, every intra-deposit gate stays green — yet diverges from its
    committed ledger would re-derive a corrupt-but-green verdict with no stale
    flag, because the ledger is the one primary record the intra-deposit gates
    never consult. Proof of need (verified below): on the homogeneous full deposit
    (recorded TIES, already citable) editing ``candidate_losses[0]`` by −0.05 flips
    the replayed verdict to ``SURPASSES`` *while staying citable* — turning an
    honest null into a claimed §4 win that no prior gate flags; only the ledger
    binding catches it.
    """

    # The two citable full-budget deposits — the only recordings that carry a
    # committed ledger witness — each paired with its committed JSONL ledger.
    _FULL_PAIRS = [
        (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json",
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full_ledger.jsonl",
        ),
        (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full_heterogeneous.json",
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl",
        ),
    ]

    def test_both_ledgers_are_committed_alongside_their_deposits(self):
        # The witness ledger is a committed repo file — the binding cross-check
        # reads committed bytes, never a gitignored ``runs/`` path or a private
        # stable dir. Both full deposits carry a relative ``ledger_witness_path``
        # that resolves to a committed file next to the deposit.
        for deposit, ledger in self._FULL_PAIRS:
            assert ledger.exists(), (
                f"{ledger.name} missing — the citable full deposit's ledger "
                f"witness is not committed alongside it."
            )
            data = load_samples(str(deposit))
            wit = data["ledger_witness_path"]
            assert wit is not None and not Path(wit).is_absolute(), (
                f"{deposit.name}: ledger_witness_path must be relative, got {wit!r}"
            )

    def test_full_deposits_bind_cleanly_to_their_ledgers(self):
        # Byte-identical invariant: both citable full deposits' per-arm losses
        # reconstruct from their committed ledger EXACTLY (float ==) and the
        # stamped witness hash matches the re-derived one. A committed deposit
        # must never carry losses that diverge from its own ledger — this is the
        # deposit-vs-ledger analogue of the harvest-time
        # ``test_ledger_reconstructs_deposit_loss_vectors_exactly`` witness pin.
        for deposit, _ledger in self._FULL_PAIRS:
            data = load_samples(str(deposit))
            out = replay_to_json(str(deposit), data, replay_samples(data))
            assert out["ledger_losses_stale"] == [], (
                f"{deposit.name}: committed deposit's losses diverge from its "
                f"ledger on roles {out['ledger_losses_stale']!r}."
            )
            assert out["ledger_witness_stale"] is False, (
                f"{deposit.name}: stamped witness hash diverges from the ledger "
                f"re-derived from {data.get('ledger_witness_path')!r}."
            )

    def test_deposits_without_a_ledger_witness_skip_cleanly(self):
        # Backward-compat / skip discipline: a deposit that carries no committed
        # ledger (every proxy / synthetic / reduced-budget recording, and any full
        # deposit pre-harvest) must report CLEAN — the cross-check skips, it must
        # not false-positive on a deposit that simply has no ledger to bind
        # against. Every committed 9B deposit that is NOT one of the two full
        # ones carries no ledger witness (verified: baseline / direction /
        # generalization / heterogeneous_generalization / surrogate).
        full_names = {p[0].name for p in self._FULL_PAIRS}
        no_ledger = [
            p for p in _9b_deposit_fixtures() if p.name not in full_names
        ]
        assert no_ledger, "expected non-full 9B deposits to exercise the skip path"
        for deposit in no_ledger:
            data = load_samples(str(deposit))
            ci = replay_samples(data)
            out = replay_to_json(str(deposit), data, ci)
            prose = format_replay(str(deposit), data, ci)
            assert out["ledger_losses_stale"] == [], (
                f"{deposit.name}: a deposit without a ledger witness must skip, "
                f"got losses_stale={out['ledger_losses_stale']!r}"
            )
            assert out["ledger_witness_stale"] is False, (
                f"{deposit.name}: witness_stale must be False with no ledger"
            )
            assert "LEDGER_LOSSES_STALE" not in prose, (
                f"{deposit.name}: no prose note expected without a ledger"
            )
            assert "LEDGER_WITNESS_STALE" not in prose

    def test_hand_edited_candidate_loss_is_flagged(self):
        # PRIMARY mutation proof (the corrupt-but-green path). The homogeneous
        # full deposit records TIES and is citable; editing candidate_losses[0]
        # by -0.05 flips the replayed verdict to SURPASSES while STAYING citable
        # — an honest null turned into a claimed §4 win that every intra-deposit
        # gate (evidence_hash, recorded-verdict, direction/baseline, the four
        # citation axes) reads as green. Only the ledger binding names "candidate"
        # in ledger_losses_stale and emits the LEDGER_LOSSES_STALE prose note.
        deposit = self._FULL_PAIRS[0][0]
        data = load_samples(str(deposit))
        ci_clean = replay_samples(data)
        assert ci_clean.significance_verdict == TIES  # the recorded verdict
        # The lie: lower the candidate's first loss so the candidate "wins".
        data["candidate_losses"][0] -= 0.05
        ci_mut = replay_samples(data)
        assert ci_mut.significance_verdict == SURPASSES, (
            "mutation premise: the edited losses must flip TIES -> SURPASSES"
        )
        out = replay_to_json(str(deposit), data, ci_mut)
        # The binding is the ONLY gate that fires.
        assert out["ledger_losses_stale"] == ["candidate"], (
            f"got {out['ledger_losses_stale']!r}, expected ['candidate']"
        )
        assert "LEDGER_LOSSES_STALE" in format_replay(
            str(deposit), data, ci_mut
        )

    def test_hand_edited_surrogate_loss_is_flagged(self):
        # Mutation proof (surrogate arm): the sibling of the candidate proof — a
        # lie on the surrogate vector is named, not just the candidate.
        deposit = self._FULL_PAIRS[0][0]
        data = load_samples(str(deposit))
        data["surrogate_losses"][1] += 0.01
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert "surrogate" in out["ledger_losses_stale"], (
            f"got {out['ledger_losses_stale']!r}, expected to name 'surrogate'"
        )

    def test_dropped_arm_length_mismatch_is_flagged(self):
        # Mutation proof (shape): a deposit that silently dropped an arm's loss
        # (changing n_candidate and thus the verdict) diverges from the ledger in
        # LENGTH, not just value — the cross-check catches the shape mismatch too.
        deposit = self._FULL_PAIRS[0][0]
        data = load_samples(str(deposit))
        data["candidate_losses"] = data["candidate_losses"][:-1]
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert "candidate" in out["ledger_losses_stale"]

    def test_hand_edited_witness_hash_is_flagged(self):
        # Mutation proof (witness binding): a deposit pointed at the wrong ledger,
        # a ledger rewritten under the same path, or a hand-edited stamp — the
        # stamped ``ledger_witness_sha256`` no longer matches the re-derived one.
        # The losses still match (so losses_stale stays clean) but the content
        # binding is broken; witness_stale fires and the prose names it.
        deposit = self._FULL_PAIRS[0][0]
        data = load_samples(str(deposit))
        data["ledger_witness_sha256"] = "ZZZ_NEVER_REAL"
        ci = replay_samples(data)
        out = replay_to_json(str(deposit), data, ci)
        assert out["ledger_witness_stale"] is True
        assert out["ledger_losses_stale"] == [], (
            "a pure witness-hash lie must not also trip the losses check"
        )
        assert "LEDGER_WITNESS_STALE" in format_replay(str(deposit), data, ci)

    def test_prose_note_presence_matches_machine_flags(self):
        # DRIFT invariant (mirrors the direction/baseline drift test above): the
        # machine flags and the human prose notes cannot drift. Across a committed
        # deposit (clean) AND a hand-edited lie on each axis (candidate loss /
        # witness hash), ``LEDGER_LOSSES_STALE in prose`` == bool(losses_stale)
        # and ``LEDGER_WITNESS_STALE in prose`` == witness_stale.
        deposit = self._FULL_PAIRS[0][0]
        cases = [("committed", load_samples(str(deposit)))]
        lie_losses = load_samples(str(deposit))
        lie_losses["candidate_losses"][0] -= 0.05
        cases.append(("lie-losses", lie_losses))
        lie_wit = load_samples(str(deposit))
        lie_wit["ledger_witness_sha256"] = "ZZZ_NEVER_REAL"
        cases.append(("lie-witness", lie_wit))
        for name, data in cases:
            ci = replay_samples(data)
            out = replay_to_json(str(deposit), data, ci)
            prose = format_replay(str(deposit), data, ci)
            losses_note = "LEDGER_LOSSES_STALE" in prose
            witness_note = "LEDGER_WITNESS_STALE" in prose
            assert losses_note is bool(out["ledger_losses_stale"]), (
                f"{name}: losses prose ({losses_note}) != flag "
                f"({out['ledger_losses_stale']!r})"
            )
            assert witness_note is out["ledger_witness_stale"], (
                f"{name}: witness prose ({witness_note}) != flag "
                f"({out['ledger_witness_stale']!r})"
            )


class TestEvidenceHashBinding:
    """The deposit-vs-its-own-evidence binding: the producer stamps
    ``evidence_hash`` (a SHA-256 over the deposit's raw measurements +
    run-determining config — never the derived verdict/gate/regime labels) so a
    coordinated repaint that edits the floats + verdict + provenance TOGETHER
    (which passes every DERIVED check), or any accidental byte drift, moves the
    stamp. The replay now re-derives that hash from the SAME key list +
    canonicalization (the shared :mod:`src.tg_lora.freeze_evidence_hash` leaf) and
    flags a deposit whose stamp no longer matches — the deposit-vs-its-own-
    evidence sibling of :class:`TestLedgerBinding` (which binds the deposit to an
    EXTERNAL ledger by content hash; this binds it to its OWN evidence block).

    The gap this closes: the ledger binding reaches only the TWO citable
    full-budget deposits that carry a committed ledger witness. The OTHER FIVE
    committed 9B deposits (direction / baseline / surrogate / generalization /
    heterogeneous_generalization) carry an ``evidence_hash`` but NO ledger, so
    without this guard their committed bytes have no integrity binding at the
    torch-free replay chokepoint at all — a hand-edit to their freeze order,
    provenance, or run config passes every prior gate (faithful, the four
    citation axes, the direction/baseline sub-verdicts) silently. Proof of need
    (verified below): on the surrogate deposit (recorded SURPASSES, ledger-less)
    reversing ``candidate_order`` — the freeze order the §4 verdict is ABOUT —
    leaves the verdict faithful (losses untouched) and every ledger check clean
    (no ledger), yet the stamped ``evidence_hash`` is provably stale; only this
    guard catches it.
    """

    def test_committed_deposits_match_their_stamps(self):
        # Byte-identical invariant: every committed 9B deposit that carries an
        # ``evidence_hash`` must re-derive to ``evidence_hash_stale is False`` —
        # the replay reproduces the producer's stamp from the SAME leaf, so a
        # committed deposit never reads as stale. A future deposit whose
        # committed bytes drifted from its stamp (a botched harvest, a formatter)
        # trips this and the silent flip is prevented.
        for deposit in _9b_deposit_fixtures():
            data = load_samples(str(deposit))
            if "evidence_hash" not in data:
                continue  # the runlog artifact aside, all 7 carry a stamp
            ci = replay_samples(data)
            out = replay_to_json(str(deposit), data, ci)
            assert out["evidence_hash_stale"] is False, (
                f"{deposit.name}: committed evidence_hash is stale against its "
                f"own bytes — the deposit and its stamp have diverged."
            )
            assert "EVIDENCE_HASH_STALE" not in format_replay(str(deposit), data, ci)

    def test_deposits_without_a_stamp_skip_cleanly(self):
        # Skip / false-positive discipline: a proxy / synthetic / legacy
        # recording that carries no ``evidence_hash`` must report CLEAN — the
        # cross-check skips, it must not false-positive on a deposit that simply
        # has no stamp to bind. The committed proxy recording has none.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_generalize_proxy.json"
        )
        data = load_samples(str(deposit))
        assert "evidence_hash" not in data, "premise: the proxy deposit has no stamp"
        ci = replay_samples(data)
        out = replay_to_json(str(deposit), data, ci)
        assert out["evidence_hash_stale"] is False
        assert "EVIDENCE_HASH_STALE" not in format_replay(str(deposit), data, ci)

    def test_hand_edited_freeze_order_is_flagged_on_a_ledgerless_deposit(self):
        # PRIMARY mutation proof (the corrupt-but-green path the ledger binding
        # cannot reach). The surrogate deposit records SURPASSES and is
        # ledger-less, so no ledger gate can bind its bytes. Reversing
        # ``candidate_order`` — the freeze order the §4 verdict is ABOUT — leaves
        # the verdict faithful (losses untouched) and every ledger check clean,
        # yet the stamped evidence_hash is provably stale: the deposit's recorded
        # order no longer matches the order its verdict claims. ONLY this guard
        # names the corruption.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_surrogate.json"
        )
        data = load_samples(str(deposit))
        assert data.get("ledger_witness_path") in (None, ""), (
            "premise: the surrogate deposit is ledger-less"
        )
        ci_clean = replay_samples(data)
        assert ci_clean.significance_verdict == SURPASSES  # the recorded verdict
        # The lie: invert the recorded freeze order (an evidence key, not a loss).
        data["candidate_order"] = list(reversed(data["candidate_order"]))
        ci_mut = replay_samples(data)
        assert ci_mut.significance_verdict == SURPASSES, (
            "the order edit must not change the verdict (losses are untouched)"
        )
        out = replay_to_json(str(deposit), data, ci_mut)
        # The ledger binding is silent (no ledger); evidence_hash is the ONLY
        # gate that fires.
        assert out["ledger_losses_stale"] == []
        assert out["evidence_hash_stale"] is True, (
            "a reversed freeze order must stale the stamped evidence_hash"
        )
        assert "EVIDENCE_HASH_STALE" in format_replay(str(deposit), data, ci_mut)

    def test_hand_edited_run_config_is_flagged(self):
        # Mutation proof (run-config axis): a hand-edit to a run-determining
        # config field (``total_steps``) — which would misrepresent WHICH run
        # produced the measurements — stales the stamp. ``total_steps`` also
        # feeds the producer's budget axis, but the replay already re-derives
        # THAT axis from the artifact; the evidence_hash guard is the one that
        # flags the byte drift itself.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_direction.json"
        )
        data = load_samples(str(deposit))
        data["total_steps"] = int(data["total_steps"]) + 1
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert out["evidence_hash_stale"] is True

    def test_hand_edited_provenance_is_flagged(self):
        # Mutation proof (provenance axis): a hand-edit to a per-arm provenance
        # dict (which layers froze) — an evidence key no other replay check
        # binds — stales the stamp.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_baseline.json"
        )
        data = load_samples(str(deposit))
        data["candidate_provenance"] = {"ZZZ_NEVER_REAL": True}
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert out["evidence_hash_stale"] is True

    def test_hand_edited_stamp_is_flagged_but_bytes_stay_consistent(self):
        # Mutation proof (stamp lie): a forged / hand-edited stamp that no longer
        # matches the (unchanged) committed bytes — a deposit pointed at the
        # wrong harvest, or a stamp pasted from another run. The bytes are clean
        # (so no OTHER staleness axis fires); only the evidence_hash binding
        # catches the broken stamp.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_surrogate.json"
        )
        data = load_samples(str(deposit))
        data["evidence_hash"] = "ZZZ_NEVER_REAL"
        ci = replay_samples(data)
        out = replay_to_json(str(deposit), data, ci)
        assert out["evidence_hash_stale"] is True
        assert "EVIDENCE_HASH_STALE" in format_replay(str(deposit), data, ci)

    def test_stamp_covers_evidence_not_derived_labels(self):
        # Invariant (mirrors the producer's ``test_hash_is_over_evidence_not_
        # derived_labels``): the hash freezes EVIDENCE, never the derived labels,
        # so editing any derived label (verdict / regime / gate / CI statistics)
        # must NOT stale the stamp — otherwise the integrity check would be
        # circular. A label edit reads clean here; only an EVIDENCE edit stales.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_surrogate.json"
        )
        data = load_samples(str(deposit))
        for label_key in (
            "verdict", "passes", "significant_surpasses", "is_material",
            "citable_as_full_section4_verdict", "citable_as_target_scale",
            "regime", "point_improvement", "lower", "upper",
            "candidate_mean", "surrogate_mean",
        ):
            mutated = dict(data)
            current = data.get(label_key)
            mutated[label_key] = (
                "ZZZ_NEVER_REAL" if isinstance(current, str) else (not current)
            )
            out = replay_to_json(str(deposit), mutated, replay_samples(mutated))
            assert out["evidence_hash_stale"] is False, (
                f"evidence_hash leaked the derived label '{label_key}' — the "
                f"hash must cover only raw evidence, never labels."
            )

    def test_prose_note_presence_matches_machine_flag(self):
        # DRIFT invariant (mirrors TestLedgerBinding's): the machine flag and the
        # human prose note cannot drift. Across a committed deposit (clean) AND a
        # hand-edited lie on each axis (freeze order / run config / provenance /
        # stamp), ``EVIDENCE_HASH_STALE in prose`` == ``evidence_hash_stale``.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_surrogate.json"
        )
        base = load_samples(str(deposit))
        cases = [("committed", base)]
        for name, mutate in (
            ("lie-order", lambda d: d.__setitem__(
                "candidate_order", list(reversed(d["candidate_order"])))),
            ("lie-config", lambda d: d.__setitem__(
                "total_steps", int(d["total_steps"]) + 1)),
            ("lie-provenance", lambda d: d.__setitem__(
                "candidate_provenance", {"ZZZ": True})),
            ("lie-stamp", lambda d: d.__setitem__(
                "evidence_hash", "ZZZ_NEVER_REAL")),
        ):
            d = dict(base)
            mutate(d)
            cases.append((name, d))
        for name, data in cases:
            ci = replay_samples(data)
            out = replay_to_json(str(deposit), data, ci)
            prose = format_replay(str(deposit), data, ci)
            note = "EVIDENCE_HASH_STALE" in prose
            assert note is out["evidence_hash_stale"], (
                f"{name}: prose ({note}) != flag ({out['evidence_hash_stale']})"
            )


class TestCiStatsBinding:
    """The deposit-vs-its-own-derived-statistics binding: the producer stamps the
    main verdict's QUANTITATIVE backing — candidate / surrogate means, point
    improvement, bootstrap CI bounds, confidence, sample sizes, thin-evidence flag
    — straight from the ``ci`` it computed the verdict from
    (:func:`scripts.run_freeze_validloss_ci_9b.result_to_json` lines ~1995-2000).
    The replay re-derives each margin-invariant statistic from the SAME losses +
    seed and flags a deposit whose stored numbers no longer match the losses it
    cites — the stored-derived-statistic-trusted-over-artifact path this guard
    closes, a DISTINCT class from :class:`TestSubVerdictLabelStaleness` /
    ``371e934`` (the verdict LABEL), :class:`TestNineBHonestyGateReplay` /
    ``d734327`` (the citability BOOLEAN), and :class:`TestEvidenceHashBinding` /
    ``79577a5`` (the raw EVIDENCE hash).

    The gap this closes: ``faithful`` binds only the verdict LABEL and
    ``evidence_hash`` binds only the raw EVIDENCE bytes (losses + run config,
    DELIBERATELY not the derived statistics — see the producer's
    ``test_hash_is_over_evidence_not_derived_labels``), so a hand-edited deposit
    that repaints the CITED CI numbers while leaving the honest losses — and thus
    the honest verdict label — untouched passes every prior gate silently. Proof
    of need (verified below): on the homogeneous full deposit (recorded TIES,
    already citable) editing ``point_improvement``/``lower``/``upper`` to a
    confidently-positive result keeps ``faithful=True`` and
    ``evidence_hash_stale=False`` — every label / ledger / evidence gate green —
    yet the cited numbers are a lie; only this guard catches it.
    """

    def test_committed_deposits_stats_match_rederived(self):
        # Byte-identical invariant: every committed deposit that stores these §4
        # statistics must re-derive to ``ci_stats_stale == []`` — the producer
        # stamps each from the ``ci`` the deterministic bootstrap produced, so the
        # replay reproduces them bit-for-bit from the same losses + seed. A future
        # deposit whose stamped numbers drifted from its losses (a botched harvest,
        # a hand-edit) trips this and the silent repaint is prevented.
        seen = False
        for deposit in _9b_deposit_fixtures():
            data = load_samples(str(deposit))
            if not any(
                k in data for k in (
                    "candidate_mean", "point_improvement", "lower", "upper",
                )
            ):
                continue  # not a stats-stamping deposit
            seen = True
            ci = replay_samples(data)
            out = replay_to_json(str(deposit), data, ci)
            assert out["ci_stats_stale"] == [], (
                f"{deposit.name}: committed derived statistics diverge from the "
                f"re-derived ci on {out['ci_stats_stale']!r}."
            )
            assert "CI_STATS_STALE" not in format_replay(str(deposit), data, ci)
        assert seen, "expected at least one stats-stamping committed deposit"

    def test_recording_without_stats_skips_cleanly(self):
        # Skip / false-positive discipline: a recording that carries the sample
        # lists but stamps NONE of the derived statistics (a legacy / minimal
        # recording) must report CLEAN — the cross-check skips, it must not
        # false-positive on a deposit that simply has no statistics to bind.
        path = "synthetic-no-stats.json"
        data = {
            "candidate_losses": [1.0, 1.0, 1.0],
            "surrogate_losses": [2.0, 2.0, 2.0],
            "base_seed": 0,
        }
        ci = replay_samples(data)
        out = replay_to_json(path, data, ci)
        assert out["ci_stats_stale"] == [], (
            f"a stat-less recording must skip, got {out['ci_stats_stale']!r}"
        )
        assert "CI_STATS_STALE" not in format_replay(path, data, ci)

    def test_hand_edited_point_improvement_is_flagged(self):
        # PRIMARY mutation proof (the corrupt-but-green path). The homogeneous
        # full deposit records TIES and is citable; editing ``point_improvement``
        # to a large positive value — repainting the cited improvement the §4
        # result reports — keeps the verdict faithful (losses untouched) and the
        # evidence hash clean (derived statistics are not evidence), yet the
        # stored number no longer matches the losses. ONLY this guard names
        # "point_improvement" in ci_stats_stale and emits the CI_STATS_STALE note.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        ci_clean = replay_samples(data)
        assert ci_clean.significance_verdict == TIES  # the recorded verdict
        # The lie: claim a 5% improvement the stored losses do not support.
        data["point_improvement"] = 0.05
        ci_mut = replay_samples(data)
        assert ci_mut.significance_verdict == TIES, (
            "the stat edit must not change the verdict (losses are untouched)"
        )
        out = replay_to_json(str(deposit), data, ci_mut)
        assert out["ci_stats_stale"] == ["point_improvement"], (
            f"got {out['ci_stats_stale']!r}, expected ['point_improvement']"
        )
        assert "CI_STATS_STALE" in format_replay(str(deposit), data, ci_mut)

    def test_hand_edited_ci_bounds_are_flagged(self):
        # Mutation proof (CI bounds axis): repainting the bootstrap CI bounds —
        # the [lower, upper] interval cited beside the verdict — is caught
        # symmetrically; not just the point estimate.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        data["lower"] = 0.02
        data["upper"] = 0.08
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert set(out["ci_stats_stale"]) == {"lower", "upper"}, (
            f"got {out['ci_stats_stale']!r}"
        )

    def test_hand_edited_candidate_mean_is_flagged(self):
        # Mutation proof (means axis): repainting the candidate's valid-loss mean
        # — the absolute loss a paper cites — is caught. The candidate mean feeds
        # the point improvement, so it is named.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        data["candidate_mean"] = 1.50  # the stored losses re-derive ~1.695
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert "candidate_mean" in out["ci_stats_stale"], (
            f"got {out['ci_stats_stale']!r}, expected to name 'candidate_mean'"
        )

    def test_hand_edited_surrogate_mean_and_counts_are_flagged(self):
        # Mutation proof (surrogate mean + sample-size axes): repainting the
        # surrogate mean or the sample counts is caught — the binding is not
        # candidate-only, and covers the structural counts too.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        data["surrogate_mean"] = data["surrogate_mean"] + 0.1
        data["n_candidate"] = data["n_candidate"] + 1
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert "surrogate_mean" in out["ci_stats_stale"]
        assert "n_candidate" in out["ci_stats_stale"], (
            f"got {out['ci_stats_stale']!r}"
        )

    def test_is_material_is_not_bound(self):
        # SCOPE pin: ``is_material`` is the ONE margin-dependent statistic
        # (``point_improvement >= material_margin``), and the margin is NOT
        # stamped in the deposit — so binding it strictly would false-positive on
        # a producer run that used a non-zero margin. It is therefore
        # intentionally excluded from ``_CI_STAT_BINDINGS``: mutating it ALONE
        # must not trip ci_stats_stale. (Guards against a maintainer wrongly
        # "completing" the binding by adding the margin-dependent field back.)
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        data["is_material"] = not data["is_material"]
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert out["ci_stats_stale"] == [], (
            f"is_material is margin-dependent and must NOT be bound; got "
            f"{out['ci_stats_stale']!r}"
        )

    def test_corrupt_stats_pass_every_other_gate(self):
        # REACHABILITY proof (the silent path this guard closes). A deposit whose
        # stored CI numbers are repainted but whose losses (and thus verdict) stay
        # honest passes EVERY prior gate — faithful, evidence_hash, the citation
        # label, the ledger — and is caught ONLY by ci_stats_stale. This is the
        # corrupt-but-green §4 quantitative-claim path, made loud.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        data["point_improvement"] = 0.05
        data["lower"] = 0.02
        data["upper"] = 0.08
        ci = replay_samples(data)
        out = replay_to_json(str(deposit), data, ci)
        assert out["faithful"] is True, "losses are honest -> verdict still matches"
        assert out["evidence_hash_stale"] is False, (
            "derived statistics are not evidence bytes -> stamp stays clean"
        )
        assert out["citation_label_stale"] is False
        assert out["ledger_losses_stale"] == []
        assert out["ci_stats_stale"], (
            "the repainted CI numbers must be the ONE gate that fires"
        )

    def test_prose_note_presence_matches_machine_flag(self):
        # DRIFT invariant (mirrors TestLedgerBinding / TestEvidenceHashBinding):
        # the machine flag and the human prose note cannot drift. Across a
        # committed deposit (clean) AND a hand-edited lie on each axis (point
        # improvement / CI bounds / candidate mean / sample counts),
        # ``CI_STATS_STALE in prose`` == ``bool(ci_stats_stale)``.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        base = load_samples(str(deposit))
        cases = [("committed", base)]
        for name, mutate in (
            ("lie-point", lambda d: d.__setitem__("point_improvement", 0.05)),
            ("lie-bounds", lambda d: (
                d.__setitem__("lower", 0.02), d.__setitem__("upper", 0.08))),
            ("lie-mean", lambda d: d.__setitem__("candidate_mean", 1.50)),
            ("lie-counts", lambda d: d.__setitem__("n_candidate",
                                                   d["n_candidate"] + 1)),
        ):
            d = dict(base)
            mutate(d)
            cases.append((name, d))
        for name, data in cases:
            ci = replay_samples(data)
            out = replay_to_json(str(deposit), data, ci)
            prose = format_replay(str(deposit), data, ci)
            note = "CI_STATS_STALE" in prose
            assert note is bool(out["ci_stats_stale"]), (
                f"{name}: prose ({note}) != flag ({out['ci_stats_stale']!r})"
            )


class TestTargetScaleLabelBinding:
    """The deposit-vs-its-own-``proxy_scale`` Level-1 citation binding: the
    producer stamps ``citable_as_target_scale`` as a deterministic function of the
    deposit's own ``proxy_scale`` field — ``not result["proxy_scale"]``
    (:func:`scripts.run_freeze_validloss_ci_9b.result_to_json`, the 1-term Level-1
    citation contract). It is the Level-1 citation boolean a reader checks first
    ("is this a genuine target-scale 9B recording?"), the prerequisite the Level-2
    ``citable_as_full_section4_verdict`` gate composes on top of.

    The gap this closes: ``citation_label_stale`` / :class:`TestNineBHonestyGateReplay`
    / ``d734327`` binds the Level-2 ``citable_as_full_section4_verdict`` boolean by
    re-deriving the full four-conjunct gate — but that gate can be honestly
    ``False`` for reasons UNRELATED to target-scale (reduced context / thin /
    non-generalization), so it does NOT transitively bind this Level-1 boolean. And
    ``evidence_hash`` / :class:`TestEvidenceHashBinding` / ``79577a5`` deliberately
    never covers gate labels (only raw measurements + run config), so a hand-edited
    deposit that flips ``citable_as_target_scale`` — over-claiming target scale on a
    proxy recording (``stored=True`` while ``proxy_scale=True``), or under-claiming a
    genuine 9B run — while leaving ``proxy_scale`` honest passes every prior gate
    silently. Proof of need (verified below): on the homogeneous full deposit
    (recorded TIES, 9B, ``citable_as_target_scale=True``) flipping the boolean to
    ``False`` keeps ``faithful=True``, ``evidence_hash_stale=False``, and — critically
    — ``citation_label_stale=False`` (the Level-2 boolean is untouched), yet the
    deposit's Level-1 citation claim is a lie; only this guard catches it.
    """

    @staticmethod
    def _deposits_with_label():
        # Every committed fixture that stamps ``citable_as_target_scale`` — both
        # the 9B deposits (stored ``True``) and the proxy recordings (stored
        # ``False``) — so the byte-identical invariant is proven in BOTH
        # directions, not just the target-scale one.
        fixture_dir = Path(__file__).resolve().parent / "fixtures"
        out = []
        for p in sorted(fixture_dir.glob("freeze_validloss*.json")):
            if "runlog" in p.name:
                continue
            try:
                if "citable_as_target_scale" in json.loads(p.read_text()):
                    out.append(p)
            except Exception:
                continue
        return out

    def test_committed_deposits_label_matches_not_proxy_scale(self):
        # Byte-identical invariant: every committed deposit that stores the Level-1
        # citation boolean must re-derive to ``target_scale_label_stale is False`` —
        # the producer stamps it as ``not proxy_scale``, so the replay reproduces it
        # from the deposit's own ``proxy_scale`` bit-for-bit. A future deposit whose
        # stamped boolean drifted from its ``proxy_scale`` (a botched harvest, a
        # hand-edit) trips this and the silent Level-1 citation repaint is
        # prevented. Covers both stored-True (9B) and stored-False (proxy).
        assert self._deposits_with_label(), "expected committed labeled deposits"
        for deposit in self._deposits_with_label():
            data = load_samples(str(deposit))
            ci = replay_samples(data)
            out = replay_to_json(str(deposit), data, ci)
            assert out["target_scale_label_stale"] is False, (
                f"{deposit.name}: stored citable_as_target_scale="
                f"{data.get('citable_as_target_scale')!r} disagrees with not "
                f"proxy_scale={not bool(data.get('proxy_scale', True))!r}."
            )
            assert "TARGET_SCALE_LABEL_STALE" not in format_replay(
                str(deposit), data, ci
            )

    def test_recording_without_label_skips_cleanly(self):
        # Skip / false-positive discipline: a recording that carries the sample
        # lists but stamps NO ``citable_as_target_scale`` boolean (a legacy /
        # minimal recording that predates the field) must report CLEAN — the
        # cross-check skips, it must not false-positive on a deposit that simply
        # has no Level-1 label to bind.
        path = "legacy-no-target-scale-label.json"
        data = {
            "candidate_losses": [1.0, 1.0, 1.0],
            "surrogate_losses": [2.0, 2.0, 2.0],
            "base_seed": 0,
        }
        ci = replay_samples(data)
        out = replay_to_json(path, data, ci)
        assert out["target_scale_label_stale"] is False, (
            "a label-less recording must skip, got a stale flag"
        )
        assert "TARGET_SCALE_LABEL_STALE" not in format_replay(path, data, ci)

    def test_hand_edited_label_underclaim_is_flagged(self):
        # PRIMARY mutation proof (under-claim axis). The homogeneous full deposit
        # is a 9B run (``citable_as_target_scale=True``, ``proxy_scale=False``);
        # flipping the stored boolean to ``False`` under-claims a genuine target-
        # scale recording. The losses (and thus the verdict) are untouched, the
        # evidence hash stays clean (gate labels are not evidence), and the Level-2
        # citation label is untouched — yet the Level-1 boolean no longer matches
        # ``not proxy_scale``. ONLY this guard fires.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        ci = replay_samples(data)
        assert data["citable_as_target_scale"] is True
        assert data["proxy_scale"] is False
        data["citable_as_target_scale"] = False  # the lie: under-claim a 9B run
        out = replay_to_json(str(deposit), data, ci)
        assert out["target_scale_label_stale"] is True
        assert "TARGET_SCALE_LABEL_STALE" in format_replay(str(deposit), data, ci)

    def test_hand_edited_label_overclaim_is_flagged(self):
        # Mutation proof (over-claim axis). A proxy recording carries
        # ``citable_as_target_scale=False`` (``proxy_scale=True``); flipping the
        # stored boolean to ``True`` OVER-CLAIMS target scale on a proxy run — the
        # more dangerous direction (a proxy verdict dressed as a 9B result). Caught
        # symmetrically; the binding is not under-claim-only.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_heterogeneous_generalize_proxy.json"
        )
        data = load_samples(str(deposit))
        ci = replay_samples(data)
        assert data["citable_as_target_scale"] is False
        assert data["proxy_scale"] is True
        data["citable_as_target_scale"] = True  # the lie: dress a proxy run as 9B
        out = replay_to_json(str(deposit), data, ci)
        assert out["target_scale_label_stale"] is True
        assert "TARGET_SCALE_LABEL_STALE" in format_replay(str(deposit), data, ci)

    def test_hand_edited_proxy_scale_is_flagged(self):
        # Mutation proof (input axis). The deposit's ``proxy_scale`` is the input
        # the producer derives the label from; flipping ``proxy_scale`` while
        # leaving the stored label unchanged is the symmetric corruption (the input
        # moved, the label did not follow). Caught from the input side, not just the
        # label side — the binding reads ``proxy_scale`` directly.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        data["proxy_scale"] = True  # the lie: claim the 9B run was a proxy run
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert out["target_scale_label_stale"] is True, (
            "flipping proxy_scale (stored label now stale) must fire"
        )

    def test_independent_of_level2_citation_label(self):
        # DISTINCTION proof (not redundant with ``citation_label_stale``). The
        # Level-2 boolean ``citable_as_full_section4_verdict`` can be honestly
        # ``False`` for reasons unrelated to target-scale (reduced context / thin /
        # non-generalization), so it does NOT transitively bind the Level-1
        # boolean. A deposit whose Level-2 label is honestly ``False`` yet whose
        # Level-1 ``citable_as_target_scale`` was hand-edited passes
        # ``citation_label_stale`` (Level-2 stored == re-derived) but trips this
        # gate — proving the two are independent fields and this guard closes a
        # distinct, reachable path the Level-2 gate leaves open.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_heterogeneous_generalize_proxy.json"
        )
        data = load_samples(str(deposit))
        # The proxy recording's Level-2 label is honestly False; flip ONLY the
        # Level-1 boolean.
        data["citable_as_target_scale"] = not data["citable_as_target_scale"]
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert out["citation_label_stale"] is False, (
            "Level-2 label untouched -> the Level-2 gate must NOT fire"
        )
        assert out["target_scale_label_stale"] is True, (
            "the flipped Level-1 label is the distinct path this gate closes"
        )

    def test_corrupt_label_passes_every_other_gate(self):
        # REACHABILITY proof (the silent path this guard closes). A deposit whose
        # stored Level-1 boolean is flipped but whose losses / verdict / proxy_scale
        # / Level-2 label stay honest passes EVERY prior gate — faithful,
        # evidence_hash, the Level-2 citation label, the ledger, the sub-verdicts —
        # and is caught ONLY by target_scale_label_stale. This is the
        # corrupt-but-green Level-1 citation-claim path, made loud.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        data = load_samples(str(deposit))
        data["citable_as_target_scale"] = False  # flip Level-1 only
        out = replay_to_json(str(deposit), data, replay_samples(data))
        assert out["faithful"] is True, "losses untouched -> verdict still matches"
        assert out["evidence_hash_stale"] is False, (
            "gate labels are not evidence bytes -> stamp stays clean"
        )
        assert out["citation_label_stale"] is False, (
            "Level-2 boolean untouched -> Level-2 gate stays clean"
        )
        assert out["ledger_losses_stale"] == []
        assert out["direction_verdict_stale"] is False
        assert out["target_scale_label_stale"] is True, (
            "the flipped Level-1 label must be the ONE gate that fires"
        )

    def test_prose_note_presence_matches_machine_flag(self):
        # DRIFT invariant (mirrors TestCiStatsBinding / TestLedgerBinding /
        # TestEvidenceHashBinding): the machine flag and the human prose note cannot
        # drift. Across a committed deposit (clean) AND a hand-edited lie on each
        # axis (label under-claim / label over-claim / proxy_scale flip),
        # ``TARGET_SCALE_LABEL_STALE in prose`` == ``target_scale_label_stale``.
        deposit = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_ci_9b_full.json"
        )
        base = load_samples(str(deposit))
        proxy = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "freeze_validloss_heterogeneous_generalize_proxy.json"
        )
        proxy_base = load_samples(str(proxy))
        cases = [
            ("committed-9b", str(deposit), base),
            ("committed-proxy", str(proxy), proxy_base),
            ("lie-underclaim", str(deposit), {**base, "citable_as_target_scale": False}),
            ("lie-overclaim", str(proxy), {**proxy_base, "citable_as_target_scale": True}),
            ("lie-proxy-scale", str(deposit), {**base, "proxy_scale": True}),
        ]
        for name, path, data in cases:
            ci = replay_samples(data)
            out = replay_to_json(path, data, ci)
            prose = format_replay(path, data, ci)
            note = "TARGET_SCALE_LABEL_STALE" in prose
            assert note is out["target_scale_label_stale"], (
                f"{name}: prose ({note}) != flag "
                f"({out['target_scale_label_stale']!r})"
            )



