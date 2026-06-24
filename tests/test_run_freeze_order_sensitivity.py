"""Tests for ``scripts/run_freeze_order_sensitivity.py`` — the apparatus
order-resolution diagnostic.

The verdict runs (:mod:`scripts.run_freeze_validloss_ci`) report candidate-vs-
surrogate ``TIES`` at proxy scale, but a ``TIES`` is only conclusive if the
apparatus can *resolve* a freeze-order effect. This diagnostic answers that with
a variance decomposition: ``Var(order)`` (distinct orders at a fixed seed) vs
``Var(seed)`` (a fixed order across seeds). The suite guards:

* **Import health + ``--help``** — the CLI is launchable as ``-m`` (the canary
  contract every ``scripts.run_*`` CLI in this repo keeps).
* **The key finding is deterministic.** At a fixed seed, varying the freeze
  order leaves the final valid_loss essentially unchanged (``Var(order)`` ≈ 0)
  while varying the seed moves it (``Var(seed)`` > 0) — so the order signal is
  absent at proxy scale, not merely below a noisy floor. This is the property
  that makes the verdict ``TIES`` a genuine null rather than a power failure.
* **The ratio and ``resolvable`` flag are wired correctly** off the two
  variances (re-derived from the raw samples), and the distinct-order draw is
  collision-free and reproducible.
* **Honest proxy-scale labeling.** The report and JSON carry the
  ``PROXY_SCALE`` caveat a reader must see before citing the ratio.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

from scripts.run_freeze_order_sensitivity import (
    RESOLUTION_THRESHOLD,
    _variance,
    distinct_orders,
    format_report,
    order_sensitivity,
    result_to_json,
)

# Tiny but resolution-adequate budget for a fast, deterministic CPU check. Two
# arms of >= 2 draws each is the minimum the variances need; the make target /
# real GPU run use the larger converged-regime defaults. The order-insensitivity
# finding holds at this budget exactly (Var(order) == 0.0).
_TINY = dict(total=15, warmup=4, depth=2, n_orders=4, n_seeds=4)
_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Import health + --help
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.run_freeze_order_sensitivity")
        for attr in (
            "main",
            "build_parser",
            "order_sensitivity",
            "distinct_orders",
            "_variance",
            "format_report",
            "result_to_json",
            "RESOLUTION_THRESHOLD",
        ):
            assert hasattr(mod, attr), f"missing {attr}"

    def test_help_exits_zero(self):
        # ``-m`` launchability is the canary contract every scripts.run_* CLI
        # keeps; a clean --help exit is the cheapest proof the module imports
        # and the parser builds under the repo interpreter.
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.run_freeze_order_sensitivity", "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "order-resolution" in proc.stdout.lower() or "variance" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# distinct_orders helper
# ---------------------------------------------------------------------------


class TestDistinctOrders:
    def test_returns_n_distinct_permutations(self):
        orders = distinct_orders(6, 8, seed=12345)
        assert len(orders) == 8
        assert len({tuple(o) for o in orders}) == 8  # all distinct
        for o in orders:
            assert sorted(o) == [0, 1, 2, 3, 4, 5]  # each is a real permutation

    def test_reproducible(self):
        a = distinct_orders(6, 6, seed=999)
        b = distinct_orders(6, 6, seed=999)
        assert a == b

    def test_different_seeds_differ(self):
        # The draw is genuinely a function of the seed (not a constant order).
        a = distinct_orders(6, 6, seed=1)
        b = distinct_orders(6, 6, seed=2)
        assert a != b

    def test_empty_for_zero(self):
        assert distinct_orders(6, 0, seed=0) == []

    def test_oversubscribe_raises(self):
        import itertools

        total = len(list(itertools.permutations(range(6))))
        with pytest.raises(ValueError, match="permutations"):
            distinct_orders(6, total + 1, seed=0)


# ---------------------------------------------------------------------------
# _variance helper
# ---------------------------------------------------------------------------


class TestVariance:
    def test_constant_is_zero(self):
        assert _variance([2.5, 2.5, 2.5, 2.5]) == 0.0

    def test_under_two_is_zero(self):
        assert _variance([1.0]) == 0.0
        assert _variance([]) == 0.0

    def test_unbiased_estimator(self):
        # Sample variance (n-1): for [1, 2, 3] mean=2, sum sq dev=2, /2 = 1.0.
        assert _variance([1.0, 2.0, 3.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# order_sensitivity — the core diagnostic
# ---------------------------------------------------------------------------


class TestOrderSensitivity:
    def test_order_signal_is_absent_at_proxy_scale(self):
        # The headline finding: at a FIXED seed, varying the freeze order does
        # not move the final valid_loss (Var(order) ~= 0), while varying the
        # seed does (Var(seed) > 0). Order is unresolvable at proxy scale — a
        # genuine null, not a power failure.
        result = order_sensitivity(device=_DEVICE, **_TINY)
        assert result["var_seed"] > 0.0  # the apparatus is not a flat broken read
        assert result["var_order"] < 1e-6  # essentially zero to float precision
        # Robustly: order accounts for a negligible fraction of the seed floor.
        assert result["ratio"] < 0.01
        assert result["resolvable"] is False

    def test_ratio_derived_from_raw_samples(self):
        # The ratio is Var(order)/Var(seed), recomputable from the raw per-arm
        # samples the result exposes (the audit must be able to re-derive it
        # rather than trust a label).
        result = order_sensitivity(device=_DEVICE, **_TINY)
        expected = _variance(result["by_order"]) / _variance(result["by_seed"])
        assert result["ratio"] == pytest.approx(expected)
        assert result["var_order"] == pytest.approx(_variance(result["by_order"]))
        assert result["var_seed"] == pytest.approx(_variance(result["by_seed"]))

    def test_by_order_uses_distinct_orders(self):
        result = order_sensitivity(device=_DEVICE, **_TINY)
        orders = result["orders"]
        assert len(orders) == _TINY["n_orders"]
        assert len({tuple(o) for o in orders}) == _TINY["n_orders"]  # collision-free

    def test_reproducible_for_fixed_base_seed(self):
        # Every RNG is locally seeded, so a fixed base_seed reproduces the whole
        # decomposition bit-for-bit on a given device.
        a = order_sensitivity(device=_DEVICE, **_TINY)
        b = order_sensitivity(device=_DEVICE, **_TINY)
        assert a["by_order"] == b["by_order"]
        assert a["by_seed"] == b["by_seed"]
        assert a["ratio"] == b["ratio"]

    def test_resolvable_flag_uses_threshold(self):
        # resolvable is ratio >= RESOLUTION_THRESHOLD (not a separate verdict
        # route). With Var(order) ~= 0 the flag is False; the threshold constant
        # is the single knob.
        result = order_sensitivity(device=_DEVICE, **_TINY)
        assert result["resolvable"] == (result["ratio"] >= RESOLUTION_THRESHOLD)
        assert result["resolution_threshold"] == RESOLUTION_THRESHOLD

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="n_orders"):
            order_sensitivity(device=_DEVICE, **{**_TINY, "n_orders": 1})
        with pytest.raises(ValueError, match="n_seeds"):
            order_sensitivity(device=_DEVICE, **{**_TINY, "n_seeds": 1})
        with pytest.raises(ValueError, match="architecture"):
            order_sensitivity(device=_DEVICE, **{**_TINY, "architecture": "bogus"})
        with pytest.raises(ValueError, match="task"):
            order_sensitivity(device=_DEVICE, **{**_TINY, "task": "bogus"})

    def test_heterogeneous_stack_also_unresolvable(self):
        # The insensitivity is not a homogeneous-stack artifact: a per-layer
        # rank schedule (the verdict positive-control stack) is order-unresolvable
        # too, so the finding holds across the architectures the verdict runs use.
        result = order_sensitivity(device=_DEVICE, architecture="heterogeneous", **_TINY)
        assert result["var_order"] < 1e-6
        assert result["resolvable"] is False
        # Heterogeneous => a per-layer rank schedule, not the flat homogeneous stack.
        assert len(result["ranks"]) == result["num_layers"]
        assert len(set(result["ranks"])) > 1

    def test_proxy_scale_honesty_flag(self):
        result = order_sensitivity(device=_DEVICE, **_TINY)
        assert result["proxy_scale"] is True


# ---------------------------------------------------------------------------
# Report + JSON shape
# ---------------------------------------------------------------------------


class TestReportAndJson:
    def test_report_carries_verdict_and_ratio(self):
        result = order_sensitivity(device=_DEVICE, **_TINY)
        report = format_report(result)
        assert "freeze_order_sensitivity" in report
        assert "Var(order)" in report
        assert "Var(seed)" in report
        # The diagnostic's verdict is the resolvability call, surfaced plainly.
        assert "is NOT resolvable" in report
        assert "PROXY_SCALE" in report

    def test_json_round_trips_expected_keys(self):
        result = order_sensitivity(device=_DEVICE, **_TINY)
        payload = result_to_json(result)
        s = json.dumps(payload)  # serializable (no tensor/inf surprises)
        back = json.loads(s)
        for key in (
            "var_order", "var_seed", "ratio", "resolvable",
            "by_order", "by_seed", "orders", "proxy_scale",
            "architecture", "task", "resolution_threshold",
        ):
            assert key in back
        assert back["proxy_scale"] is True
        assert back["resolvable"] is False
