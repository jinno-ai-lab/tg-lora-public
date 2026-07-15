"""Cross-module single-source-of-truth guard for the heterogeneous LoRA rank schedule.

The geometric per-layer rank schedule (``base_rank ** (i / (n - 1))``) is the
load-bearing asymmetry that distinguishes the heterogeneous §4 verdict
(SURPASSES) from the homogeneous one (TIES): it injects GOAL §1.5/§8
non-uniform per-layer *capacity* so a freeze order can structurally matter. It
is consumed by THREE scripts — the proxy CI harness, the order-sensitivity
diagnosis, and the real-9B target-scale CI harness — and was, before
``geometric_rank_schedule``, duplicated verbatim in the two CI harnesses. A
silent drift between those copies would make the proxy *apparatus* verdict
(apparatus-TIES, the sensitivity null) and the 9B *target* verdict (SURPASSES)
test *different* architectures — the proxy would no longer validate the same
stack the target verdict claims.

These tests pin Constitution Rule #3 (single source of truth) two ways:

* **delegation** — both harness wrappers must call the one canonical
  :func:`src.model.lora_utils.geometric_rank_schedule`, so a formula edit can
  only happen in one place (mutated canonical ⇒ the frozen-literal schedule
  below flips RED while wrapper==canonical stays green, proving the wrappers
  delegate rather than re-implement);
* **non-divergence** — proxy, 9B, and canonical stay mutually byte-equal for
  every matching input, so a wrapper that re-implements the formula locally
  flips the cross-equality RED.

This is the cross-module byte-equality risk the AI-Hub feedback (Constitution
Rule #3) called out for the heterogeneous rank work; it lives here rather than
in a ``test_config.py`` because no such file exists and the duplication is
between two harness scripts, not config constants.
"""

from __future__ import annotations

from src.model.lora_utils import geometric_rank_schedule
from scripts.run_freeze_validloss_ci import HIDDEN, heterogeneous_ranks
from scripts.run_freeze_validloss_ci_9b import heterogeneous_ranks_9b

# Bases span both harness defaults: the proxy's ``HIDDEN`` (24) and the 9B
# config's homogeneous ``r`` (16), plus neighbours, so a rounding/floor drift
# surfaces at more than one magnitude.
BASES = (8, 16, 24, 32)


class TestCanonicalSchedule:
    def test_frozen_literal_schedule(self):
        # Authoritative geometric schedules, hand/independently verified and
        # frozen so ANY drift in the canonical formula (exponent, rounding, or
        # the max(1, ...) floor) flips this RED. The (8, 16) row is the same
        # schedule recorded in tests/test_run_freeze_validloss_ci_9b.py's
        # heterogeneous deposit comment ({24:1,25:1,26:2,27:3,28:5,29:7,30:11,
        # 31:16}), pinned here at the single source.
        assert geometric_rank_schedule(8, 16) == (1, 1, 2, 3, 5, 7, 11, 16)
        assert geometric_rank_schedule(6, 24) == (1, 2, 4, 7, 13, 24)
        assert geometric_rank_schedule(4, 8) == (1, 2, 4, 8)
        assert geometric_rank_schedule(12, 32) == (
            1, 1, 2, 3, 4, 5, 7, 9, 12, 17, 23, 32,
        )

    def test_edges(self):
        # Zero active layers ⇒ zero ranks (the 9B wrapper's empty-set guard);
        # one layer ⇒ the full base rank (geometric over a singleton collapses).
        assert geometric_rank_schedule(0, 16) == ()
        assert geometric_rank_schedule(1, 16) == (16,)
        assert geometric_rank_schedule(1, 24) == (24,)

    def test_output_capped_at_base_and_monotone(self):
        for n in range(2, 13):
            for base in BASES:
                ranks = geometric_rank_schedule(n, base)
                assert len(ranks) == n
                assert ranks[-1] == base  # output-most layer keeps full capacity
                assert ranks[0] >= 1  # max(1, ...) floor holds
                assert all(a <= b for a, b in zip(ranks, ranks[1:]))


class TestWrappersDelegateToCanonical:
    """Both harness wrappers must be thin delegates over the one canonical source.

    If a wrapper re-implements the geometric formula locally (the original
    duplication), wrapper != canonical for at least one input and these go RED.
    """

    def test_proxy_wrapper_equals_canonical(self):
        for n in range(1, 13):
            for base in BASES:
                assert heterogeneous_ranks(n, base) == geometric_rank_schedule(
                    n, base
                ), f"proxy diverged from canonical at n={n}, base={base}"

    def test_9b_wrapper_equals_canonical(self):
        for n in range(0, 13):
            for base in BASES:
                active = set(range(n))
                assert heterogeneous_ranks_9b(active, base) == (
                    geometric_rank_schedule(n, base)
                ), f"9B diverged from canonical at n={n}, base={base}"

    def test_proxy_at_hidden_matches_canonical(self):
        # The proxy's real call site is ``heterogeneous_ranks(num_layers,
        # HIDDEN)``; pin that exact production pairing at the canonical source.
        assert heterogeneous_ranks(6, HIDDEN) == geometric_rank_schedule(6, HIDDEN)


class TestCrossModuleByteEquality:
    """The feedback's Constitution-Rule-#3 ask: proxy == 9B for matching inputs.

    The proxy verdict validates the apparatus; the 9B verdict is the target-scale
    result. They are only comparable if they train the *same* heterogeneous stack,
    which reduces to these two wrappers emitting byte-identical rank tuples.
    """

    def test_proxy_equals_9b_across_shape_and_magnitude(self):
        for n in range(1, 13):
            for base in BASES:
                proxy = heterogeneous_ranks(n, base)
                target = heterogeneous_ranks_9b(set(range(n)), base)
                assert proxy == target, (
                    f"proxy/9B heterogeneous schedules diverged at n={n}, "
                    f"base={base}: {proxy} != {target}"
                )

    def test_9b_active_layer_order_invariant(self):
        # ``active_layers`` arrives as a set / arbitrary iterable from the model
        # introspection; the schedule must depend only on the *count*, not the
        # iteration order, so the same layers always map to the same ranks.
        layers = {31, 28, 29, 30}
        assert heterogeneous_ranks_9b(layers, 16) == heterogeneous_ranks_9b(
            sorted(layers), 16
        )

    def test_third_consumer_binds_to_proxy_delegate(self):
        # ``run_freeze_order_sensitivity`` imports the proxy wrapper, so the
        # order-sensitivity diagnosis also rides the canonical source. Pin that
        # binding: if someone copy-pastes a third copy into the
        # order-sensitivity script, this identity check flips RED.
        import scripts.run_freeze_order_sensitivity as order_sensitivity

        assert order_sensitivity.heterogeneous_ranks is heterogeneous_ranks
