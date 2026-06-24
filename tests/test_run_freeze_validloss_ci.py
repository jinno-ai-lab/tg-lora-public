"""Tests for ``scripts/run_freeze_validloss_ci.py`` — the Category-C attack.

This is the run that finally feeds :func:`surrogate_valid_loss_ci` numbers that
came out of an *actual* training run rather than constructed constants (see the
script docstring). The suite guards:

* **Import health + ``--help``** — the CLI is launchable as ``-m`` (the canary
  contract every ``scripts.run_*`` CLI in this repo keeps).
* **The arm is a real training run.** :func:`arm_valid_loss` returns a finite
  value below the uniform-init loss (the proxy genuinely learns, not a stub),
  is reproducible for a fixed ``(order, seed)`` (every RNG is locally seeded),
  and is not a constant across seeds.
* **The CI is wired correctly.** :func:`run_ci` deposits one real valid_loss
  sample per arm, the verdict is one of the three valid §4 labels, and the
  verdict is *self-consistent* with the bootstrap CI bounds (re-derived from
  ``lower``/``upper``) — proving the harness hands the samples to the right
  statistic rather than returning a label by another route.
* **Honest proxy-scale labeling.** The report and JSON carry the
  ``PROXY_SCALE`` caveat a reader must see before citing the verdict.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

# Tiny but non-thin (>= 3 seeds/arm, see MIN_SAMPLE_FOR_BOOTSTRAP) budget for a
# fast, deterministic CPU check. The make target / real GPU run use the larger
# converged-regime defaults; this only exercises the wiring.
_TINY_ARM = dict(total=15, warmup=4, depth=2)
_TINY = dict(total=15, warmup=4, depth=2, n_candidate=3, n_surrogate=3)
_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Import health + --help
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.run_freeze_validloss_ci")
        for attr in (
            "main",
            "build_parser",
            "run_ci",
            "arm_valid_loss",
            "format_report",
            "result_to_json",
            "resolve_device",
            "output_first_order",
        ):
            assert hasattr(mod, attr), f"missing {attr}"

    def test_output_first_order_descends_from_output_side(self):
        from scripts.run_freeze_validloss_ci import output_first_order

        assert output_first_order(6) == (5, 4, 3, 2, 1, 0)
        assert output_first_order(4) == (3, 2, 1, 0)


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_freeze_validloss_ci", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "valid_loss" in result.stdout.lower()
        # The §4 control (random-order surrogate) and the candidate must surface.
        assert "surrogate" in result.stdout.lower()
        assert "candidate" in result.stdout.lower()


# ---------------------------------------------------------------------------
# The arm is a real, reproducible training run
# ---------------------------------------------------------------------------


class TestArmIsRealTraining:
    def test_arm_returns_finite_learned_loss(self):
        from scripts.run_freeze_validloss_ci import arm_valid_loss, output_first_order

        v = arm_valid_loss(
            output_first_order(6), seed=0, device=_DEVICE, num_layers=6, **_TINY_ARM
        )
        # Finite, and the proxy genuinely learned: well below the ~log(32)=3.47
        # uniform-init loss (the invivo fixture descends to ~0.4 over a fuller
        # budget; this tiny budget still lands far below uniform).
        assert v == pytest.approx(v)  # finite (not nan/inf)
        assert v < 2.5, f"arm did not learn: valid_loss={v}"

    def test_arm_is_reproducible_for_fixed_order_and_seed(self):
        from scripts.run_freeze_validloss_ci import arm_valid_loss, output_first_order

        order = output_first_order(6)
        a = arm_valid_loss(order, seed=7, device=_DEVICE, num_layers=6, **_TINY_ARM)
        b = arm_valid_loss(order, seed=7, device=_DEVICE, num_layers=6, **_TINY_ARM)
        # Every RNG (torch init, batch generator, the CI's numpy seed) is locally
        # seeded, so an arm is bit-reproducible on a fixed device.
        assert a == b

    def test_arm_varies_across_seeds_not_a_constant(self):
        from scripts.run_freeze_validloss_ci import arm_valid_loss, output_first_order

        order = output_first_order(6)
        vals = [
            arm_valid_loss(order, seed=s, device=_DEVICE, num_layers=6, **_TINY_ARM)
            for s in (0, 1, 2)
        ]
        assert all(v == pytest.approx(v) for v in vals)  # all finite
        # The value is computed from a real run, not hardcoded: distinct seeds
        # produce a genuine spread (at least two differ).
        assert len(set(round(v, 6) for v in vals)) >= 2


# ---------------------------------------------------------------------------
# run_ci deposits real samples and wires them to the correct statistic
# ---------------------------------------------------------------------------


class TestRunCI:
    def test_run_ci_deposits_one_real_sample_per_arm_and_is_non_thin(self):
        from scripts.run_freeze_validloss_ci import run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        cand = r["candidate_losses"]
        surr = r["surrogate_losses"]
        assert len(cand) == _TINY["n_candidate"]
        assert len(surr) == _TINY["n_surrogate"]
        # Real samples: finite, and the arms actually learned (below uniform init).
        assert all(v == pytest.approx(v) for v in cand + surr)
        assert max(cand + surr) < 2.5
        # Non-thin: both arms meet MIN_SAMPLE_FOR_BOOTSTRAP (=3).
        ci = r["ci"]
        assert ci.is_thin_evidence is False

    def test_verdict_is_valid_label_and_self_consistent_with_ci_bounds(self):
        from scripts.run_freeze_validloss_ci import run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        ci = r["ci"]
        # The verdict is one of the three §4 labels the structural gate emits.
        assert ci.significance_verdict in {SURPASSES, TIES, UNDERSHOOTS}
        # Self-consistency — re-derive the verdict from the CI bounds: this proves
        # the harness handed the real samples to surrogate_valid_loss_ci (which
        # sets lower/upper) rather than producing the label another way.
        if ci.significance_verdict == SURPASSES:
            assert ci.lower > 0.0
        elif ci.significance_verdict == UNDERSHOOTS:
            assert ci.upper < 0.0
        else:  # TIES
            assert ci.lower <= 0.0 <= ci.upper
        # The point estimate is the observed difference of means, in valid_loss units.
        assert ci.point_improvement == pytest.approx(
            ci.surrogate_mean - ci.candidate_mean
        )

    def test_run_ci_is_reproducible_by_base_seed(self):
        from scripts.run_freeze_validloss_ci import run_ci

        a = run_ci(device=_DEVICE, base_seed=42, num_layers=6, **_TINY)
        b = run_ci(device=_DEVICE, base_seed=42, num_layers=6, **_TINY)
        assert a["candidate_losses"] == b["candidate_losses"]
        assert a["surrogate_losses"] == b["surrogate_losses"]
        assert a["ci"].significance_verdict == b["ci"].significance_verdict
        assert a["ci"].lower == b["ci"].lower and a["ci"].upper == b["ci"].upper


# ---------------------------------------------------------------------------
# Honest proxy-scale labeling in both renderings
# ---------------------------------------------------------------------------


class TestHonestProxyScaleLabeling:
    def test_report_carries_proxy_scale_caveat(self):
        from scripts.run_freeze_validloss_ci import format_report, run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        text = format_report(r)
        assert "PROXY_SCALE" in text
        # The caveat tells a reader not to cite it as a target-scale result.
        assert "9B target" in text
        assert "proxy_scale=True" in text

    def test_json_carries_proxy_scale_flag(self):
        from scripts.run_freeze_validloss_ci import result_to_json, run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        payload = result_to_json(r)
        assert payload["proxy_scale"] is True
        # The JSON is the evidence artifact a target-scale run would overwrite.
        parsed = json.loads(json.dumps(payload))
        assert parsed["verdict"] in {SURPASSES, TIES, UNDERSHOOTS}
        assert len(parsed["candidate_losses"]) == _TINY["n_candidate"]
        assert len(parsed["surrogate_losses"]) == _TINY["n_surrogate"]
