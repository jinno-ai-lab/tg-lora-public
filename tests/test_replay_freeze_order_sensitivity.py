"""Tests for ``scripts/replay_freeze_order_sensitivity.py`` — the order-resolution
diagnostic reduced to a concrete, GPU-free, executable command.

``run_freeze_order_sensitivity`` trains 12 distinct freeze orders at a fixed
seed and 12 seeds at a fixed order, then reports ``Var(order)/Var(seed)``.
``replay_freeze_order_sensitivity`` re-derives that decomposition from the
recorded ``by_order`` / ``by_seed`` floats with no GPU, no model, and no torch.
The suite guards:

* **Import health + ``--help``** — the CLI launches as ``-m`` (the canary every
  ``scripts.replay_*`` CLI keeps).
* **Schema validation** — ``load_result`` rejects a missing file, malformed JSON,
  and a file lacking the >= 2-sample lists the decomposition needs.
* **Decomposition correctness** — the replayed variance matches the source
  diagnostic's ``_variance`` on shared inputs (torch-gated cross-check) and a
  known value (torch-free sanity); the ratio and outcome are recomputed from the
  stored floats, and the threshold is read *from the recording* (no parallel
  constant).
* **Faithfulness on real recorded evidence** — the committed fixture
  (``tests/fixtures/freeze_order_sensitivity_proxy.json``, a real RTX 3060 run)
  replays to the ratio it *recorded* (``0.000``) and outcome ``not_resolvable``:
  the stored floats earn the ratio under the deterministic decomposition, it is
  not painted on. This is the expected-output assertion that pins the recorded
  linchpin dataset.
* **The resolvable branch** — a hand-crafted recording where order moves the
  metric replays as ``resolvable``, proving the replay is not a dead
  always-``not_resolvable`` reader.
* **Scale honesty + CLI assertion** — ``proxy_scale`` is surfaced; the
  target-scale drop-in flips the scale label and ``citable_as_target_scale`` with
  no code change; ``--expected`` exits 0 on a match and 2 on a mismatch.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.replay_freeze_order_sensitivity import (
    DEFAULT_RESOLUTION_THRESHOLD,
    NOT_RESOLVABLE,
    RESOLVABLE,
    _sample_variance,
    format_replay,
    load_result,
    main,
    replay_order_sensitivity,
    replay_to_json,
)

# The committed real-GPU recording: a ``--task generalize`` homogeneous run on
# the RTX 3060 (ratio = 0.000 — all 12 distinct orders give identical
# valid_loss ≈ 2.7155; Var(seed) ≈ 0.0202). Regenerate with
# ``make freeze-order-sensitivity``.
FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "freeze_order_sensitivity_proxy.json"
)


# ---------------------------------------------------------------------------
# Import health + --help
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.replay_freeze_order_sensitivity")
        for attr in (
            "main", "build_parser", "load_result", "replay_order_sensitivity",
            "format_replay", "replay_to_json", "EXPECTED_OUTCOMES",
        ):
            assert hasattr(mod, attr), f"missing {attr}"

    def test_help_launches_as_module(self):
        # The canary contract: every scripts.* CLI launches via ``-m`` with a
        # working ``--help`` and exit 0 (the sys.path-bootstrap invariant).
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.replay_freeze_order_sensitivity", "--help"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Var(order)" in proc.stdout
        assert "--expected" in proc.stdout


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestLoadResult:
    def test_rejects_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_result("/nonexistent/path/to/order_sensitivity.json")

    def test_rejects_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_result(bad)

    def test_rejects_missing_sample_keys(self, tmp_path):
        bad = tmp_path / "no_samples.json"
        bad.write_text(json.dumps({"ratio": 0.0, "device": "cuda"}))
        with pytest.raises(ValueError, match="by_order"):
            load_result(bad)

    def test_rejects_too_short_sample_list(self, tmp_path):
        # Sample variance needs >= 2 observations; a single-sample list is not a
        # valid recording of the decomposition.
        bad = tmp_path / "short.json"
        bad.write_text(json.dumps({"by_order": [2.7], "by_seed": [2.7, 2.8]}))
        with pytest.raises(ValueError, match="by_order"):
            load_result(bad)

    def test_loads_committed_fixture(self):
        data = load_result(FIXTURE)
        assert len(data["by_order"]) == 12
        assert len(data["by_seed"]) == 12
        assert data["proxy_scale"] is True
        assert data["device"] == "cuda"


# ---------------------------------------------------------------------------
# Decomposition correctness + source cross-check
# ---------------------------------------------------------------------------


class TestSampleVariance:
    def test_known_value_torch_free(self):
        # The unbiased sample variance of 1..5 is exactly 2.5 — pins the formula
        # without importing the torch-bearing source module.
        assert _sample_variance([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(2.5)
        assert _sample_variance([2.7, 2.7, 2.7, 2.7]) == pytest.approx(0.0)

    def test_too_few_observations_is_zero(self):
        assert _sample_variance([1.0]) == 0.0
        assert _sample_variance([]) == 0.0

    def test_matches_source_diagnostic_variance(self):
        # The replay reimplements the source ``_variance`` locally (rather than
        # importing it) so it stays torch-free. Pin the two equal on shared
        # inputs so the local copy cannot drift from what recorded the fixture.
        pytest.importorskip("torch")
        from scripts.run_freeze_order_sensitivity import _variance as source_variance

        for values in (
            [2.7155, 2.7155, 2.7155, 2.7155],
            [2.5, 2.6, 2.7, 2.8, 2.9],
            [1.0, 1.0, 1.0, 5.0],
        ):
            assert _sample_variance(values) == pytest.approx(source_variance(values))


class TestReplayDecomposition:
    def test_zero_var_order_is_not_resolvable(self):
        # All identical orders (the proxy finding): Var(order)=0 => ratio=0 =>
        # not_resolvable, even against a real seed-noise floor.
        data = {
            "by_order": [2.7155] * 6,
            "by_seed": [2.60, 2.65, 2.70, 2.75, 2.80, 2.85],
            "resolution_threshold": DEFAULT_RESOLUTION_THRESHOLD,
        }
        out = replay_order_sensitivity(data)
        assert out["var_order"] == 0.0
        assert out["ratio"] == 0.0
        assert out["resolvable"] is False

    def test_moving_order_is_resolvable(self):
        # Order moves the metric well above the seed floor => ratio >= threshold.
        data = {
            "by_order": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],   # large order spread
            "by_seed": [2.0, 2.01, 2.02, 2.03, 2.04, 2.05],  # tiny seed floor
            "resolution_threshold": DEFAULT_RESOLUTION_THRESHOLD,
        }
        out = replay_order_sensitivity(data)
        assert out["ratio"] >= DEFAULT_RESOLUTION_THRESHOLD
        assert out["resolvable"] is True

    def test_threshold_read_from_recording(self):
        # The outcome follows the recording's own threshold, not a parallel
        # constant — a stricter threshold flips a borderline ratio's outcome.
        by_order = [2.0, 2.2]  # var_order = 0.02
        by_seed = [2.0, 2.4]   # var_seed  = 0.08  => ratio = 0.25
        loose = replay_order_sensitivity(
            {"by_order": by_order, "by_seed": by_seed, "resolution_threshold": 0.10}
        )
        strict = replay_order_sensitivity(
            {"by_order": by_order, "by_seed": by_seed, "resolution_threshold": 0.50}
        )
        assert loose["resolvable"] is True
        assert strict["resolvable"] is False
        assert loose["ratio"] == strict["ratio"] == pytest.approx(0.25)

    def test_fallback_threshold_matches_source(self):
        pytest.importorskip("torch")
        from scripts.run_freeze_order_sensitivity import RESOLUTION_THRESHOLD

        assert DEFAULT_RESOLUTION_THRESHOLD == RESOLUTION_THRESHOLD


# ---------------------------------------------------------------------------
# Faithfulness on the committed real-GPU recording
# ---------------------------------------------------------------------------


class TestCommittedFixtureFaithfulness:
    def test_replays_recorded_ratio_and_outcome(self):
        data = load_result(FIXTURE)
        out = replay_order_sensitivity(data)
        # The linchpin: ratio is exactly 0 (all 12 orders identical), so the
        # proxy apparatus cannot resolve order — target-scale is the only regime
        # that can. The replay must re-earn both the ratio and the outcome from
        # the stored floats.
        assert out["ratio"] == pytest.approx(data["ratio"])
        assert out["var_order"] == pytest.approx(0.0)
        assert out["var_seed"] == pytest.approx(data["var_seed"])
        assert out["resolvable"] is False
        assert _sample_variance(data["by_order"]) == pytest.approx(0.0)

    def test_machine_gate_not_citable_as_target_scale(self):
        data = load_result(FIXTURE)
        out = replay_order_sensitivity(data)
        js = replay_to_json(FIXTURE, data, out)
        assert js["faithful"] is True
        assert js["proxy_scale"] is True
        assert js["citable_as_target_scale"] is False
        assert js["replayed_outcome"] == NOT_RESOLVABLE

    def test_format_replay_reports_match(self):
        data = load_result(FIXTURE)
        out = replay_order_sensitivity(data)
        text = format_replay(FIXTURE, data, out)
        assert "PROXY_SCALE" in text
        assert "MATCHES recording" in text
        assert "NOT_RESOLVABLE" in text


# ---------------------------------------------------------------------------
# Scale honesty + CLI assertion (incl. target-scale drop-in)
# ---------------------------------------------------------------------------


class TestScaleHonestyAndCLI:
    def test_target_scale_dropin_flips_label_and_gate(self, tmp_path):
        # The same-schema drop-in: a recording tagged proxy_scale=False upgrades
        # the scale label to TARGET and the citable gate to True, with no code
        # change (the verdict replay's contract extended to this diagnostic).
        data = load_result(FIXTURE)
        data["proxy_scale"] = False
        out = replay_order_sensitivity(data)
        text = format_replay(tmp_path / "target.json", data, out)
        js = replay_to_json(tmp_path / "target.json", data, out)
        assert "TARGET_SCALE" in text
        assert "PROXY_SCALE" not in text
        assert js["proxy_scale"] is False
        assert js["citable_as_target_scale"] is True

    def test_expected_match_exits_zero(self):
        rc = main([str(FIXTURE), "--expected", NOT_RESOLVABLE])
        assert rc == 0

    def test_expected_mismatch_exits_nonzero(self):
        rc = main([str(FIXTURE), "--expected", RESOLVABLE])
        assert rc == 2

    def test_expected_resolvable_on_resolvable_recording(self, tmp_path):
        # A resolvable recording pinned to its outcome — proves ``--expected``
        # passes on the resolvable branch too (not hardwired to not_resolvable).
        f = tmp_path / "resolvable.json"
        f.write_text(json.dumps({
            "by_order": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "by_seed": [2.0, 2.01, 2.02, 2.03, 2.04, 2.05],
            "resolution_threshold": DEFAULT_RESOLUTION_THRESHOLD,
            "proxy_scale": True,
        }))
        rc = main([str(f), "--expected", RESOLVABLE])
        assert rc == 0
