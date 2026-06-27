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
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES

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


def _data(candidate, surrogate, *, base_seed=0, proxy_scale=True, synthetic=False):
    return {
        "candidate_losses": list(candidate),
        "surrogate_losses": list(surrogate),
        "base_seed": base_seed,
        "proxy_scale": proxy_scale,
        "synthetic": synthetic,
    }


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
        # The boolean is provably consistent with the prose claim: it is True
        # exactly when the human-readable note grants the "this verdict IS the §4
        # target-scale result" claim. Any future edit to one path without the
        # other fails here.
        cases = [
            ("<genuine-9b>", _data([1.0] * 4, [2.0] * 4, proxy_scale=False)),
            (str(FIXTURE), load_samples(FIXTURE)),
            (str(FIXTURE_TARGET), load_samples(FIXTURE_TARGET)),
        ]
        for label, data in cases:
            ci = replay_samples(data)
            machine = replay_to_json(label, data, ci)["citable_as_target_scale"]
            human_grants_claim = "this verdict IS" in format_replay(label, data, ci)
            assert machine is human_grants_claim, (
                f"{label}: machine citable_as_target_scale={machine} but "
                f"human prose grants the citable claim={human_grants_claim}"
            )

