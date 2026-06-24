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


def _data(candidate, surrogate, *, base_seed=0, proxy_scale=True):
    return {
        "candidate_losses": list(candidate),
        "surrogate_losses": list(surrogate),
        "base_seed": base_seed,
        "proxy_scale": proxy_scale,
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
