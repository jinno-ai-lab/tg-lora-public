"""Tests for ``scripts/run_freeze_frontier.py`` — the Phase 2 frontier CLI.

The CLI exposes ``src.tg_lora/freeze_frontier.frontier()`` from a bare
invocation so the Phase 2 FLOPs-reduction frontier (GOAL §3.1 Phase 2 / §4) can
be reproduced *before* any GPU run — the "verify the mechanism before trusting a
GPU run" discipline (GOAL §7). These tests guard import health, the ``--help``
surface, default construction, and the monotonicity guarantee that
``freeze_frontier`` itself promises (reduction non-decreasing in depth for a
fixed ``(policy, level)`` arm).
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Import health
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.run_freeze_frontier")
        for attr in (
            "main",
            "build_parser",
            "build_spec",
            "build_layer_costs",
            "format_table",
        ):
            assert hasattr(mod, attr), f"missing {attr}"

    def test_build_layer_costs_is_uniform_and_covers_active_set(self):
        from scripts.run_freeze_frontier import build_layer_costs

        costs = build_layer_costs(
            (0, 1, 2),
            weight_grad_flops=10.0,
            act_grad_flops=3.0,
            optim_state_bytes=100,
            act_grad_bytes=40,
        )
        assert set(costs) == {0, 1, 2}
        for c in costs.values():
            assert c.weight_grad_flops == 10.0
            assert c.act_grad_flops == 3.0
            assert c.optim_state_bytes == 100
            assert c.act_grad_bytes == 40


# ---------------------------------------------------------------------------
# --help CLI
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_freeze_frontier", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "frontier" in result.stdout.lower()
        # The Phase 2 degrees of freedom (GOAL §3.1) must be surfaced.
        assert "policies" in result.stdout
        assert "levels" in result.stdout


# ---------------------------------------------------------------------------
# build_spec defaults
# ---------------------------------------------------------------------------


class TestBuildSpecDefaults:
    def test_defaults_target_qwen9b_and_level1_only(self):
        from scripts.run_freeze_frontier import build_parser, build_spec

        args = build_parser().parse_args([])  # all defaults
        spec = build_spec(args)

        # Default homogeneous stack = Qwen3.5-9B's 32 blocks (SYSTEM_CONSTITUTION).
        assert spec.active_layer_indices == tuple(range(32))
        assert spec.num_epochs >= 1
        # Phase 2 sweeps all three ordering policies.
        assert set(spec.policies) == {"output_first", "convergence_order", "compromise"}
        # GOAL §1.6.3: the prod loop drives Level 1 only; Level 2 is opt-in.
        assert spec.levels == (1,)

    def test_active_layers_subset_overrides_num_layers(self):
        from scripts.run_freeze_frontier import build_parser, build_spec

        args = build_parser().parse_args(["--active-layers", "0,1,5"])
        spec = build_spec(args)
        assert spec.active_layer_indices == (0, 1, 5)
        assert set(spec.layer_costs) == {0, 1, 5}


# ---------------------------------------------------------------------------
# End-to-end via main()
# ---------------------------------------------------------------------------


class TestMainOutput:
    def test_table_output_has_columns_and_origin(self, capsys):
        from scripts.run_freeze_frontier import main

        rc = main(
            [
                "--num-layers",
                "6",
                "--num-epochs",
                "8",
                "--policies",
                "output_first",
                "--levels",
                "1",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "policy" in out.lower()
        assert "depth" in out.lower()
        assert "reduction" in out.lower()
        # depth-0 origin (0% reduction) is always emitted by frontier().
        assert "0.00%" in out

    def test_json_output_is_valid_and_monotonic(self, capsys):
        from scripts.run_freeze_frontier import main

        rc = main(
            [
                "--num-layers",
                "6",
                "--num-epochs",
                "8",
                "--policies",
                "output_first,compromise",
                "--levels",
                "1",
                "--json",
            ]
        )
        assert rc == 0
        pts = json.loads(capsys.readouterr().out)
        assert isinstance(pts, list) and pts

        # depth-0 origin: zero reduction on every (policy, level) arm.
        origins = [p for p in pts if p["depth"] == 0]
        assert origins
        assert all(abs(p["reduction_rate"]) < 1e-12 for p in origins)

        # Monotonicity guarantee (freeze_frontier docstring): for a fixed
        # (policy, level) arm, reduction_rate is non-decreasing in depth.
        by_arm: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
        for p in pts:
            by_arm[(p["policy"], p["level"])].append((p["depth"], p["reduction_rate"]))
        for arm, series in by_arm.items():
            series.sort()
            reductions = [r for _, r in series]
            assert reductions == sorted(reductions), (
                f"arm {arm} is not monotonic: {reductions}"
            )

    def test_convergence_order_policy_runs(self, capsys):
        from scripts.run_freeze_frontier import main

        # convergence_order needs a covering order; the CLI defaults it to the
        # active set in index order when --convergence-order is omitted.
        rc = main(
            [
                "--num-layers",
                "4",
                "--num-epochs",
                "8",
                "--policies",
                "convergence_order",
                "--levels",
                "1",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "convergence_order" in out

    def test_unknown_policy_exits_nonzero(self, capsys):
        from scripts.run_freeze_frontier import main

        rc = main(["--policies", "bogus", "--levels", "1"])
        assert rc != 0
