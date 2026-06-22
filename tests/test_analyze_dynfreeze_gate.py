"""Integration tests for the §7 proxy speed-gate pipeline emission.

``freeze_cost`` is unit-tested to the bone (``tests/test_freeze_cost.py``): the
accountant, the width-extrapolation confidence (§6.1), the realizability
correction (§6.2), the variance-calibrated band (§6.3) and the graduated verdict
(§7) are each locked down against hand-computed values. What had *no* test was
the one piece that graduates all of that into the experiment's actual
``gate_decision.txt``: ``scripts.analyze_dynfreeze_experiment.py::
_proxy_speed_gate_section``.

That function is where the honest proxy path can silently break in a way no unit
test catches — it parses the guard controller's per-cycle ``block_layers`` log,
remaps *global* layer indices (e.g. L24..L31) to the local ``[0, num_layers)``
range the accountant speaks, derives the cycle count, builds the homogeneous
accountant, and decides whether the §6.3 band may be emitted at all. A bug in any
of those steps (a dropped remap, a wrong cycle count, a band emitted for a
Level-1 schedule that credits nothing) would print a wrong gate decision with
nothing failing. These tests are the missing integration-level guard for that
wiring — the same role the unit suite plays for the arithmetic.

The strongest check here is structural rather than value-pinned:
``_proxy_speed_gate_section`` must emit *exactly* what the canonical
``freeze_cost`` formatters produce for the *same* accountant — so the pipeline is
verified against the source of truth it wraps, not a hand-typed expectation that
could drift with the formatter. The behavioral cases then pin the graduation
(PASS / PROVISIONAL_PASS / REQUIRES_SCALE_MEASUREMENT / FAIL), the §6.3
"emit-the-band-only-when-something-is-credited" rule, the SKIP path, and the
layer-index remap.
"""

import pytest

from scripts.analyze_dynfreeze_experiment import (
    LOSS_TRIGGER_MARGIN,
    LOSS_VALID_FULL_KEY,
    _proxy_speed_gate_section,
    baseline_targets,
    extract_gold,
    extract_loss_and_time,
    format_honesty_contract_line,
    guard_stop_point,
    honesty_contract_status,
    load_run_metrics,
    write_gate_decision,
)
from src.tg_lora.freeze_cost import (
    PROXY_VALIDATED_MAX_WIDTH,
    ReductionSample,
    calibrate_reduction_band,
    compare_freeze_levels,
    format_level_comparison,
    format_reduction_band,
    format_speed_gate_verdict,
    frozen_at_epoch_from_freeze_log,
    level1_realization_record_from_measurements,
    per_cycle_realized_reductions,
    reproduction_record_from_ab_measurements,
    speed_gate_verdict,
    uniform_layer_accountant,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PROVISIONAL_PASS,
    VERDICT_REQUIRES_SCALE_MEASUREMENT,
)
from pathlib import Path

from src.utils.run_metrics import RunMetrics

# Eight output-side decoder blocks (the Qwen-9B target freezes a contiguous run
# of these). The pipeline speaks *global* indices in the run log; the accountant
# speaks local [0, 8). _proxy_speed_gate_section bridges the two.
LAYER_INDICES = [24, 25, 26, 27, 28, 29, 30, 31]

# A realistic observed schedule: the output side freezes progressively over
# cycles 1-4 (one more layer each cycle), with no freeze on the bookend cycles 0
# and 5. max(schedule) = 5 -> the pipeline derives num_cycles = 6. Each value is
# the {cycle: block_layers} shape the loader emits (global, comma-joined).
_FREEZE_BY_CYCLE = {
    1: [31],
    2: [30, 31],
    3: [29, 30, 31],
    4: [28, 29, 30, 31],
}

# Monotonic guard losses across the 6 cycles (used only for the num_cycles
# fallback when no schedule is observed; realistic shape keeps the fixture honest).
_GUARD_LOSSES = [2.0, 1.8, 1.6, 1.4, 1.3, 1.2]


def _schedule(freeze_by_cycle: dict[int, list[int]], *, num_cycles: int = 6) -> dict[int, dict]:
    """Build the {cycle: {block_size, block_layers}} log the pipeline consumes.

    Cycles with a freeze carry the global, comma-joined ``block_layers`` string
    (matching ``scripts/analyze_dynfreeze_experiment.py``'s loader); bookend
    cycles with no freeze carry an empty string so ``max(schedule)`` still pins
    ``num_cycles``.
    """
    schedule: dict[int, dict] = {}
    for cycle in range(num_cycles):
        layers = freeze_by_cycle.get(cycle, [])
        schedule[cycle] = {
            "block_size": len(layers),
            "block_layers": ",".join(str(g) for g in layers),
        }
    return schedule


def _expected_verdict(
    schedule: dict[int, dict],
    layer_indices: list[int],
    freeze_level: int,
    target_width: int,
    *,
    guard_losses: list[float] | None = None,
):
    """Re-derive the accountant + verdict the pipeline *should* build.

    Replicates ``_proxy_speed_gate_section``'s internal construction using the
    same public ``freeze_cost`` building blocks, so a test can assert the
    pipeline emits exactly the canonical formatter output for this verdict —
    catching any drift in parsing, remap, cycle-counting or accountant assembly.
    """
    block_log: dict[int, set[int]] = {}
    for cycle, info in schedule.items():
        parsed = {
            int(x) for x in str(info.get("block_layers", "")).split(",") if x.strip()
        }
        if parsed:
            block_log[cycle] = parsed
    global_to_local = {g: i for i, g in enumerate(layer_indices)}
    local_log = {
        cycle: {global_to_local[g] for g in layers if g in global_to_local}
        for cycle, layers in block_log.items()
    }
    frozen_at_epoch = frozen_at_epoch_from_freeze_log(local_log)
    num_cycles = (max(schedule) + 1) if schedule else len(guard_losses or [])
    accountant = uniform_layer_accountant(
        num_layers=len(layer_indices),
        num_epochs=num_cycles,
        frozen_at_epoch=frozen_at_epoch,
    )
    verdict = speed_gate_verdict(
        accountant, level=freeze_level, target_width=target_width
    )
    return accountant, verdict


def _emit(
    freeze_level: int,
    target_width: int,
    *,
    freeze_by_cycle: dict[int, list[int]] | None = None,
    layer_indices: list[int] | None = None,
    num_cycles: int = 6,
    level1_record=None,
    reproduction_record=None,
) -> list[str]:
    layer_indices = layer_indices or LAYER_INDICES
    # ``is None`` (not ``or``) so an explicitly-passed empty ``{}`` (no freeze
    # observed) stays empty instead of falling through to the default schedule —
    # otherwise the SKIP path these tests exercise would be silently masked.
    if freeze_by_cycle is None:
        freeze_by_cycle = _FREEZE_BY_CYCLE
    return _proxy_speed_gate_section(
        schedule=_schedule(freeze_by_cycle, num_cycles=num_cycles),
        layer_indices=layer_indices,
        guard_losses=_GUARD_LOSSES[:num_cycles],
        target_width=target_width,
        freeze_level=freeze_level,
        level1_record=level1_record,
        reproduction_record=reproduction_record,
    )


def _expected_comparison(
    schedule: dict[int, dict],
    layer_indices: list[int],
    target_width: int,
    *,
    level1_record=None,
    reproduction_record=None,
    guard_losses: list[float] | None = None,
):
    """Re-derive the Phase-3 comparison the pipeline *should* emit.

    Mirrors :func:`_expected_verdict` but builds :func:`compare_freeze_levels`
    over the same accountant the pipeline constructs, so a test can assert the
    pipeline's comparison block equals the canonical
    :func:`format_level_comparison` output — catching drift in the comparison
    wiring exactly as the verdict wiring is caught.
    """
    accountant, _ = _expected_verdict(
        schedule,
        layer_indices,
        freeze_level=2,
        target_width=target_width,
        guard_losses=guard_losses,
    )
    return compare_freeze_levels(
        accountant,
        target_width,
        level1_record=level1_record,
        reproduction_record=reproduction_record,
    )


# ---------------------------------------------------------------------------
# 1. The verdict graduates exactly as §6.1 + §6.2 dictate, at each width
# ---------------------------------------------------------------------------


class TestProxySpeedGateGraduation:
    """The CUDA-less verdict lands in the right category for each width/level."""

    @pytest.mark.parametrize(
        ("target_width", "expected"),
        [
            (PROXY_VALIDATED_MAX_WIDTH, VERDICT_PASS),            # h=2048, in envelope
            (PROXY_VALIDATED_MAX_WIDTH * 2, VERDICT_PROVISIONAL_PASS),  # 9B, 2x
            (PROXY_VALIDATED_MAX_WIDTH * 4, VERDICT_REQUIRES_SCALE_MEASUREMENT),  # 4x
        ],
    )
    def test_level2_verdict_graduates_with_width(self, target_width, expected):
        # Level-2 (trio) realizes a real reduction (~29% for this schedule), so
        # the verdict is purely a question of how far outside the validated
        # envelope the target sits: clean PASS in-envelope, provisional at 2x,
        # refused pending a real measurement at 4x.
        block = "\n".join(_emit(freeze_level=2, target_width=target_width))
        assert f"speed_gate_verdict: {expected}" in block

    def test_level1_fails_regardless_of_width(self):
        # Level-1 (freeze-only) realizes ~0 in vivo (§6.2), so it FAILs at every
        # width — a realizability failure, not a width one. The pipeline must
        # surface this rather than silently crediting the accountant's nonzero
        # arithmetic reduction.
        block = "\n".join(_emit(freeze_level=1, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert f"speed_gate_verdict: {VERDICT_FAIL}" in block
        assert "realized_reduction=0.0000" in block

    def test_section_header_and_provenance_present(self):
        # The audit block is self-describing: a labelled section and the
        # homogeneous-model provenance line (how many of the analyzed layers
        # froze, over how many cycles) so a reader of gate_decision.txt can see
        # exactly what fed the verdict.
        block = "\n".join(_emit(freeze_level=2, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert "Speed gate (proxy / CUDA-less" in block
        # 4 of the 8 layers froze, over the 6 observed cycles.
        assert "4/8 layers froze over 6 cycles" in block


# ---------------------------------------------------------------------------
# 2. The §6.3 band is emitted only when the gate credits something real
# ---------------------------------------------------------------------------


class TestProxySpeedGateBandEmission:
    """The band follows §6.3's rule: emit only for a credited (Level-2) reduction."""

    def test_level2_emits_calibrated_band(self):
        # Level-2 credits a real reduction, so its measured per-cycle spread is
        # recorded as a band. Six observed cycles >= MIN_SAMPLE_FOR_CONFIDENCE_BAND
        # (3), so the band is labelled "calibrated", not THIN_EVIDENCE.
        block = "\n".join(_emit(freeze_level=2, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert "reduction_band: empirical_envelope (calibrated)" in block
        assert "n=6" in block

    def test_level1_emits_no_band(self):
        # The §6.3 guard: a Level-1 schedule credits nothing realizable
        # (realized_reduction == 0), so presenting a band around zero would be
        # dishonest. The pipeline must emit the FAIL verdict with NO band block.
        block = "\n".join(_emit(freeze_level=1, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert "reduction_band:" not in block

    def test_band_emitted_even_when_verdict_is_requires_scale(self):
        # A 4x-width Level-2 run is refused a PASS pending a real measurement,
        # but its reduction is still real (just unvalidated at scale) — so the
        # band is still recorded. The width bound (§6.1) gates the verdict; it
        # does not suppress the uncertainty report (§6.3).
        block = "\n".join(
            _emit(freeze_level=2, target_width=PROXY_VALIDATED_MAX_WIDTH * 4)
        )
        assert f"speed_gate_verdict: {VERDICT_REQUIRES_SCALE_MEASUREMENT}" in block
        assert "reduction_band:" in block

    def test_thin_evidence_run_is_labelled_not_hidden(self):
        # A run that froze over only two cycles has fewer than
        # MIN_SAMPLE_FOR_CONFIDENCE_BAND observations: the statistics are still
        # computed for the audit, but the band is labelled THIN_EVIDENCE so it is
        # not dressed up as a calibrated confidence band (steering feedback:
        # "two reproductions of a median is thin evidence").
        freeze = {1: [31], 2: [30, 31]}  # 2 freeze cycles -> n=3? see below
        block = "\n".join(
            _emit(
                freeze_level=2,
                target_width=PROXY_VALIDATED_MAX_WIDTH,
                freeze_by_cycle=freeze,
                num_cycles=3,
            )
        )
        # num_cycles=3 -> per_cycle_realized_reductions yields 3 values ->
        # exactly at the MIN_SAMPLE_FOR_CONFIDENCE_BAND threshold, NOT thin.
        assert "reduction_band: empirical_envelope (calibrated)" in block
        # Now shrink below the threshold to exercise the thin-evidence label.
        block_thin = "\n".join(
            _emit(
                freeze_level=2,
                target_width=PROXY_VALIDATED_MAX_WIDTH,
                freeze_by_cycle=freeze,
                num_cycles=2,
            )
        )
        assert "reduction_band: empirical_envelope (THIN_EVIDENCE)" in block_thin


# ---------------------------------------------------------------------------
# 3. Empty / out-of-range schedules SKIP cleanly instead of fabricating a verdict
# ---------------------------------------------------------------------------


class TestProxySpeedGateSkipPaths:
    def test_no_freeze_schedule_observed_skips(self):
        # A run where no layer ever froze (every block_layers empty) has no
        # schedule to judge: the proxy path must SKIP with a stated reason, never
        # fabricate a verdict over an empty accountant.
        block = "\n".join(
            _emit(
                freeze_level=2,
                target_width=PROXY_VALIDATED_MAX_WIDTH,
                freeze_by_cycle={},
                num_cycles=4,
            )
        )
        assert "SKIP: no freeze schedule observed" in block
        assert "speed_gate_verdict:" not in block

    def test_frozen_layers_outside_analyzed_range_skips(self):
        # If the observed frozen layers are all outside the analyzed index range
        # (e.g. the run froze layers the gate was not asked about), there is
        # nothing local to credit — SKIP rather than emit a misleading verdict.
        block = "\n".join(
            _emit(
                freeze_level=2,
                target_width=PROXY_VALIDATED_MAX_WIDTH,
                freeze_by_cycle={1: [99], 2: [99, 100]},  # none in [24..31]
                num_cycles=4,
            )
        )
        assert "SKIP: observed frozen layers outside the analyzed range" in block
        assert "speed_gate_verdict:" not in block


# ---------------------------------------------------------------------------
# 4. The global -> local layer-index remap feeds the accountant correctly
# ---------------------------------------------------------------------------


class TestProxySpeedGateLayerRemap:
    def test_global_indices_remapped_to_local_before_accounting(self):
        # The run log carries global indices (L28..L31 here); the accountant is
        # built over local [0, 8). The emitted "froze M/N" count and the verdict
        # must reflect the *remapped* schedule, proving the bridge did not drop
        # or double-count a layer.
        accountant, verdict = _expected_verdict(
            _schedule(_FREEZE_BY_CYCLE),
            LAYER_INDICES,
            freeze_level=2,
            target_width=PROXY_VALIDATED_MAX_WIDTH,
        )
        # Local remap: globals 28,29,30,31 -> locals 4,5,6,7 all froze.
        assert set(accountant.frozen_at_epoch.keys()) == {4, 5, 6, 7}
        assert verdict.realized_reduction > 0.0
        block = "\n".join(_emit(freeze_level=2, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert "4/8 layers froze over 6 cycles" in block


# ---------------------------------------------------------------------------
# 5. The pipeline emits exactly what freeze_cost's formatters produce
#    (the wiring does not drift from the source of truth it wraps)
# ---------------------------------------------------------------------------


class TestProxySpeedGateMatchesFreezeCost:
    """The emitted verdict/band lines equal the canonical freeze_cost output."""

    @pytest.mark.parametrize("freeze_level", [1, 2])
    @pytest.mark.parametrize(
        "target_width",
        [
            PROXY_VALIDATED_MAX_WIDTH,
            PROXY_VALIDATED_MAX_WIDTH * 2,
            PROXY_VALIDATED_MAX_WIDTH * 4,
        ],
    )
    def test_verdict_block_matches_canonical_formatter(
        self, freeze_level, target_width
    ):
        # Whatever the pipeline prints for the verdict must be the indented
        # output of format_speed_gate_verdict over the *same* accountant — so a
        # change in parsing/remap/counting that desyncs the pipeline from
        # freeze_cost fails here, not silently in gate_decision.txt.
        schedule = _schedule(_FREEZE_BY_CYCLE)
        _, expected = _expected_verdict(
            schedule, LAYER_INDICES, freeze_level, target_width
        )
        block = "\n".join(_emit(freeze_level, target_width))
        for line in format_speed_gate_verdict(expected).split("\n"):
            assert "    " + line in block

    @pytest.mark.parametrize(
        "target_width",
        [
            PROXY_VALIDATED_MAX_WIDTH,
            PROXY_VALIDATED_MAX_WIDTH * 2,
            PROXY_VALIDATED_MAX_WIDTH * 4,
        ],
    )
    def test_band_block_matches_canonical_formatter(self, target_width):
        # Level-2: the emitted band must equal the indented output of
        # format_reduction_band over the same per-cycle series + accountant —
        # verifying the §6.3 wiring (series generation + calibration) end to end.
        schedule = _schedule(_FREEZE_BY_CYCLE)
        accountant, verdict = _expected_verdict(
            schedule, LAYER_INDICES, freeze_level=2, target_width=target_width
        )
        series = per_cycle_realized_reductions(accountant, level=2)
        band = calibrate_reduction_band(ReductionSample.from_values(series))
        block = "\n".join(_emit(freeze_level=2, target_width=target_width))
        for line in format_reduction_band(band).split("\n"):
            assert "    " + line in block


# ---------------------------------------------------------------------------
# 6. The Phase-3 Level-1-vs-Level-2 comparison reaches the real gate output
#    (GOAL §5 / Phase 3; §6.2 ceiling + §6.3 reproduction-bracket landing points)
# ---------------------------------------------------------------------------


class TestProxySpeedGateLevelComparison:
    """The Phase-3 Level-1-vs-Level-2 comparison is emitted into gate_decision.txt.

    ``compare_freeze_levels`` (GOAL §5 / Phase 3) shipped as a ``freeze_cost`` API
    exercised only in the unit suite; the real pipeline emitted just one level's
    verdict + band. This class pins that the pipeline now carries the cross-level
    comparison — both levels' verdicts over the *same* observed schedule, the
    marginal reduction the Level-2 suffix cut earns on top of the Level-1
    baseline (``additional_*_reduction``), and whether the suffix cut is what
    carries the gate (``additional_passes``) — emitted exactly as the canonical
    formatter produces. It also pins that the §6.2 ceiling and §6.3
    reproduction-bracket landing points are *reachable from the real pipeline
    path*: a supplied record moves the pipeline's output (a recovered Level-1
    verdict; a point headline thickened into a reproduction-counted bracket),
    where the default (no record) stays byte-identical.
    """

    def test_comparison_header_present(self):
        block = "\n".join(_emit(freeze_level=2, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert "Level comparison (Phase 3" in block

    @pytest.mark.parametrize(
        "target_width",
        [
            PROXY_VALIDATED_MAX_WIDTH,
            PROXY_VALIDATED_MAX_WIDTH * 2,
            PROXY_VALIDATED_MAX_WIDTH * 4,
        ],
    )
    def test_comparison_block_matches_canonical_formatter(self, target_width):
        # Whatever the pipeline prints for the comparison must equal the indented
        # output of format_level_comparison over the *same* accountant — so a
        # drift in the comparison wiring (a dropped ceiling, a wrong width) fails
        # here, not silently in gate_decision.txt.
        schedule = _schedule(_FREEZE_BY_CYCLE)
        comparison = _expected_comparison(
            schedule, LAYER_INDICES, target_width, guard_losses=_GUARD_LOSSES
        )
        block = "\n".join(_emit(freeze_level=2, target_width=target_width))
        for line in format_level_comparison(comparison).split("\n"):
            assert "    " + line in block

    def test_suffix_cut_carries_gate_where_level1_fails(self):
        # The Phase-3 headline, now in the real output: Level-1 realizes ~0 in
        # vivo (§6.2 ceiling) so it FAILs at every width, while Level-2's suffix
        # cut realizes the reduction and PASSes — the suffix cut is the sole
        # thing carrying the gate (additional_passes). The pipeline surfaces this
        # rather than reporting one level in isolation.
        block = "\n".join(_emit(freeze_level=2, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert "level1 (progressive freeze): FAIL" in block
        assert "level1 (progressive freeze): FAIL arith=" in block
        assert "realized=0.0000" in block  # §6.2 ceiling pins Level-1 at 0
        assert "level2 (suffix cut):         PASS" in block
        # The additional (level2 - level1) realized reduction clears the 10% bar.
        assert "additional (level2 - level1):" in block

    def test_no_evidence_leaves_point_headline_byte_identical(self):
        # Default (no record): the headline is a point estimate. No
        # reproduction_bracket line and no recovered-ceiling audit line appear,
        # so the comparison advances only when evidence is supplied.
        block = "\n".join(_emit(freeze_level=2, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert "reproduction_bracket" not in block
        assert "raised" not in block

    def test_reproduction_record_thickens_bracket_in_pipeline_output(self):
        # The §6.3 landing point reaches the real pipeline path: supply N=3 A/B
        # reproduction observations (through the same adapter a real run uses) and
        # the headline — a point by default — becomes a calibrated,
        # reproduction-counted bracket line in gate_decision.txt.
        baseline_backward = 1000.0
        # Three reproductions whose realized reductions wrap the accountant's
        # Level-2 headline (~0.2917 here) with a small honest spread.
        r = 0.2917
        reproduction_backwards = [
            baseline_backward * (1 - r),          # == headline
            baseline_backward * (1 - r * 0.95),   # slightly less reduction
            baseline_backward * (1 - r * 1.05),   # slightly more
        ]
        record = reproduction_record_from_ab_measurements(
            baseline_backward, reproduction_backwards
        )
        assert record.n == 3 and not record.is_thin_evidence
        block = "\n".join(
            _emit(
                freeze_level=2,
                target_width=PROXY_VALIDATED_MAX_WIDTH,
                reproduction_record=record,
            )
        )
        assert "reproduction_bracket" in block
        assert "(calibrated, n=3)" in block

    def test_level1_record_recovers_verdict_consistently_in_pipeline(self):
        # The §6.2 landing point reaches the real pipeline path AND stays
        # consistent across both blocks: a non-thin nonzero Level-1 realization
        # (the form a grad-ckpt run deposits through the same adapter) raises the
        # ceiling, flipping Level-1 from FAIL to PASS in BOTH the single-level
        # verdict block and the Phase-3 comparison block — a real verdict change
        # driven through the pipeline, not just the freeze_cost API.
        record = level1_realization_record_from_measurements(
            1000.0, [850.0, 845.0, 855.0], source="hypothetical_grad_ckpt"
        )
        without = "\n".join(_emit(freeze_level=1, target_width=PROXY_VALIDATED_MAX_WIDTH))
        assert f"speed_gate_verdict: {VERDICT_FAIL}" in without
        assert "level1 (progressive freeze): FAIL" in without
        with_record = "\n".join(
            _emit(
                freeze_level=1,
                target_width=PROXY_VALIDATED_MAX_WIDTH,
                level1_record=record,
            )
        )
        # Both the single-level verdict and the comparison recover to PASS.
        assert f"speed_gate_verdict: {VERDICT_PASS}" in with_record
        assert "level1 (progressive freeze): PASS" in with_record
        assert "level1 ceiling: raised" in with_record


class TestQualityGateDecision:
    """Direct coverage for the §5.2 two-stage stop decision functions.

    ``baseline_targets`` (§5.1: L* = best *valid_full* loss; G* = gold at L*)
    and ``guard_stop_point`` (§5.2: first cycle where valid_full <= L*+margin
    AND gold >= G*) are the Guard experiment's quality gate — the counterpart to
    the §7 speed gate the rest of this file covers — yet they had *zero* direct
    tests. Beyond pinning their contract, these lock the §5.1/§5.3 honesty rule
    the module docstring states but the code did not honour: the quality signal
    is the **full-eval loss** (``loss_valid_full``), never the cheap
    pilot/split-boundary proxy (``loss_valid``) every step also carries. The
    analyzer must read the full-eval loss when the trainer supplies it and fall
    back to the proxy transparently for legacy records (byte-identical) — the
    same pluggable-receiver pattern §6.2/§6.3 use for scale measurements.
    """

    # --- baseline_targets (§5.1) ---

    def test_baseline_targets_picks_best_loss_earliest_cycle_and_gold_there(self):
        # L* = minimum valid loss at the earliest cycle achieving it; G* = the
        # gold score recorded at or just before the L* cycle; a_total = run end.
        losses = [2.0, 1.5, 1.2, 1.2, 1.4]
        cycles = [0, 1, 2, 3, 4]
        times = [0.0, 10.0, 20.0, 30.0, 40.0]
        gold = [
            {"cycle": 0, "gold_combined": 0.10},
            {"cycle": 2, "gold_combined": 0.55},
            {"cycle": 4, "gold_combined": 0.60},
        ]
        l_star, l_star_cycle, g_star, a_total = baseline_targets(
            losses, cycles, times, gold
        )
        assert l_star == 1.2  # tie at cycles 2 & 3 -> earliest (cycle 2) wins
        assert l_star_cycle == 2
        assert g_star == 0.55  # gold at/before cycle 2
        assert a_total == 40.0

    def test_baseline_targets_empty_returns_all_none(self):
        result = baseline_targets([], [], [], [])
        assert result == (None, None, None, None)

    def test_baseline_targets_g_star_none_without_gold(self):
        l_star, _, g_star, _ = baseline_targets([2.0, 1.0], [0, 1], [0.0, 5.0], [])
        assert l_star == 1.0
        assert g_star is None

    def test_baseline_targets_g_star_falls_back_to_first_gold_when_l_star_precedes_gold(self):
        # L* lands before the first gold eval: G* falls back to the earliest
        # recorded gold (the closest available), not None.
        gold = [{"cycle": 5, "gold_combined": 0.40}]
        _, _, g_star, _ = baseline_targets([2.0, 1.0], [0, 1], [0.0, 5.0], gold)
        assert g_star == 0.40

    # --- guard_stop_point (§5.2) ---

    def test_guard_stop_point_first_cycle_meeting_both_conditions(self):
        # L*=1.0 -> threshold 1.02; G*=0.5. cycle 2 (loss ok, gold 0.40) fails
        # gold; cycle 4 (loss ok, gold 0.55) is the first to satisfy both.
        gold = [
            {"cycle": 2, "elapsed": 20.0, "loss": 1.01, "gold_combined": 0.40},
            {"cycle": 4, "elapsed": 40.0, "loss": 1.01, "gold_combined": 0.55},
        ]
        stop_cycle, stop_elapsed = guard_stop_point(gold, l_star=1.0, g_star=0.5)
        assert stop_cycle == 4
        assert stop_elapsed == 40.0

    def test_guard_stop_point_returns_row_elapsed(self):
        gold = [{"cycle": 3, "elapsed": 33.0, "loss": 1.0, "gold_combined": 0.6}]
        _, stop_elapsed = guard_stop_point(gold, l_star=1.0, g_star=0.5)
        assert stop_elapsed == 33.0

    def test_guard_stop_point_none_when_loss_outside_margin(self):
        gold = [{"cycle": 2, "loss": 1.05, "gold_combined": 0.60}]
        # loss 1.05 > L*+margin (1.02) -> trigger never fires.
        assert guard_stop_point(gold, l_star=1.0, g_star=0.5) == (None, None)

    def test_guard_stop_point_none_when_gold_below_g_star(self):
        gold = [{"cycle": 2, "loss": 1.01, "gold_combined": 0.40}]
        # loss ok, gold 0.40 < 0.5 -> never stops.
        assert guard_stop_point(gold, l_star=1.0, g_star=0.5) == (None, None)

    def test_guard_stop_point_none_when_targets_missing(self):
        # No baseline reference -> no stop decision possible.
        assert guard_stop_point([], l_star=1.0, g_star=0.5) == (None, None)
        assert guard_stop_point(
            [{"cycle": 0, "loss": 1.0, "gold_combined": 0.9}], l_star=None, g_star=0.5
        ) == (None, None)
        assert guard_stop_point(
            [{"cycle": 0, "loss": 1.0, "gold_combined": 0.9}], l_star=1.0, g_star=None
        ) == (None, None)

    def test_guard_stop_point_uses_configured_margin(self):
        # Pin LOSS_TRIGGER_MARGIN so a regression that hardcodes 0.02 is caught.
        assert LOSS_TRIGGER_MARGIN == 0.02

    # --- extract_gold honesty: prefer the full-eval loss (§5.1/§5.3) ---

    def test_extract_gold_prefers_full_eval_loss_over_pilot(self):
        # A cycle carrying BOTH the pilot proxy (loss_valid) and the honest
        # full-eval loss (loss_valid_full) must surface the full-eval value as
        # the row's §5.2 trigger loss.
        records = [
            {
                "type": "step",
                "cycle": 4,
                "elapsed_seconds": 40.0,
                "loss_valid": 0.90,
                "loss_valid_full": 1.05,
                "gold_combined": 0.5,
            }
        ]
        rows = extract_gold(records)
        assert len(rows) == 1
        assert rows[0]["loss"] == 1.05  # full-eval, not the 0.90 pilot

    def test_extract_gold_falls_back_to_pilot_when_full_eval_absent(self):
        # Legacy records (no loss_valid_full) -> pilot loss, byte-identical to
        # the pre-fix behavior.
        records = [
            {
                "type": "step",
                "cycle": 4,
                "elapsed_seconds": 40.0,
                "loss_valid": 0.90,
                "gold_combined": 0.5,
            }
        ]
        rows = extract_gold(records)
        assert rows[0]["loss"] == 0.90

    def test_extract_gold_skips_rows_without_gold(self):
        records = [
            {"type": "step", "cycle": 1, "loss_valid": 1.0},  # no gold
            {"type": "step", "cycle": 2, "loss_valid": 1.0, "gold_combined": 0.5},
        ]
        rows = extract_gold(records)
        assert [r["cycle"] for r in rows] == [2]

    # --- extract_loss_and_time: honest L* source (§5.1) ---

    def test_extract_loss_and_time_default_is_pilot_curve(self):
        records = [
            {"type": "step", "cycle": 0, "elapsed_seconds": 0.0, "loss_valid": 2.0},
            {"type": "step", "cycle": 1, "elapsed_seconds": 5.0, "loss_valid": 1.8},
        ]
        times, losses, cycles = extract_loss_and_time(records)
        assert losses == [2.0, 1.8]
        assert cycles == [0, 1]
        assert times == [0.0, 5.0]

    def test_extract_loss_and_time_full_eval_only_emits_full_eval_rows(self):
        # full_eval_only reads loss_valid_full and skips cycles without it — the
        # honest L* curve, not the every-cycle pilot proxy.
        records = [
            {
                "type": "step",
                "cycle": 0,
                "elapsed_seconds": 0.0,
                "loss_valid": 2.0,
                "loss_valid_full": 1.9,
            },
            {"type": "step", "cycle": 1, "elapsed_seconds": 5.0, "loss_valid": 1.8},
            {
                "type": "step",
                "cycle": 2,
                "elapsed_seconds": 10.0,
                "loss_valid": 1.6,
                "loss_valid_full": 1.55,
            },
        ]
        times, losses, cycles = extract_loss_and_time(records, full_eval_only=True)
        assert cycles == [0, 2]
        assert losses == [1.9, 1.55]
        assert times == [0.0, 10.0]

    def test_extract_loss_and_time_full_eval_only_empty_without_key(self):
        # No full-eval loss anywhere -> empty (caller falls back to the proxy
        # curve, byte-identical to legacy).
        records = [{"type": "step", "cycle": 0, "loss_valid": 2.0}]
        assert extract_loss_and_time(records, full_eval_only=True) == ([], [], [])

    # --- end-to-end: the fix changes the verdict, not just the plumbing ---

    def test_quality_gate_uses_full_eval_loss_not_pilot(self):
        # The same guard run scored on the pilot proxy vs the full-eval loss can
        # flip the verdict. Here the pilot loss (0.99) would trip the trigger
        # (<= L*+0.02 = 1.02) but the honest full-eval loss (1.08) does not, so
        # the §5.2 stop must NOT fire. Pre-fix (pilot) this stopped at cycle 2;
        # post-fix (full-eval) quality never reaches, exactly as §5.1 mandates.
        records = [
            {
                "type": "step",
                "cycle": 2,
                "elapsed_seconds": 20.0,
                "loss_valid": 0.99,
                "loss_valid_full": 1.08,
                "gold_combined": 0.60,
            }
        ]
        guard_gold = extract_gold(records)
        stop_cycle, _ = guard_stop_point(guard_gold, l_star=1.0, g_star=0.5)
        assert stop_cycle is None


class TestFullEvalLossHonestyE2E:
    """End-to-end proof the §5.2 honesty contract is NOT inert.

    ``TestQualityGateDecision`` above pins the decision functions against
    in-memory record dicts — necessary, but it cannot answer the question the
    §5.2 fix (commit 276991c) lives or dies by: *does the receiver actually wake
    up on a real run's persisted metrics?* The fix prefers ``loss_valid_full``
    over the pilot proxy ``loss_valid``, yet that preference is only as good as
    the records it reads. If no real run record ever carries ``loss_valid_full``
    — and the trainer does not yet persist it — the preference silently degrades
    to the byte-identical pilot proxy and the honesty contract never activates:
    the code *looks* fixed while the A/B comparison stays contaminated.

    These tests drive the FULL path the trainer uses —
    ``RunMetrics.record_step`` → ``run_metrics.jsonl`` → ``load_run_metrics`` →
    the §5.1/§5.2 decision functions — and assert, over an actual run's metrics
    file, that (1) the gate's L* is computed from the ``loss_valid_full``
    column, (2) a corrupt-input run where the pilot proxy and the full-eval loss
    disagree forces the full-eval verdict (the pilot alone would stop; the honest
    full-eval does not), and (3) when no record carries the key the receiver
    stays dormant and falls back to the pilot byte-identically. The corrupt-input
    shape (the test the feedback singled out) is what makes the honesty claim
    falsifiable rather than a happy-path assertion.
    """

    @staticmethod
    def _write_run_records(run_dir: Path, cycles: list[dict]) -> list[dict]:
        """Persist a run via the real ``RunMetrics.record_step`` path and reload.

        Each dict becomes one step record. ``loss_valid`` is the pilot proxy
        (every cycle); ``loss_valid_full`` the honest full-eval loss, emitted
        only on the cycles that carry it (its *presence* is what the analyzer
        keys on); ``gold_combined`` the §5.2 gold score. Absent keys are not
        emitted — exactly as the trainer does — so the on-disk records match a
        real run's, not a hand-built fixture.
        """
        m = RunMetrics(run_dir, mode="baseline")
        for c in cycles:
            kwargs: dict = {
                "step": c["cycle"],
                "cycle": c["cycle"],
                "loss_train": c.get("loss_train", 1.0),
                "loss_valid": c["loss_valid"],
                "backward_passes": 1,
                "total_backward_passes": c["cycle"],
            }
            if "loss_valid_full" in c:
                kwargs["loss_valid_full"] = c["loss_valid_full"]
            if "gold_combined" in c:
                kwargs["gold_combined"] = c["gold_combined"]
            m.record_step(**kwargs)
        m.close()
        return load_run_metrics(run_dir)

    def test_L_star_driven_by_full_eval_curve_over_real_metrics_file(self, tmp_path):
        # The pilot (loss_valid) and full-eval (loss_valid_full) curves
        # disagree: the pilot minimum (0.9 @ cycle 4) is NOT the full-eval
        # minimum (1.1 @ cycle 6). Over a REAL run_metrics.jsonl written by
        # RunMetrics and read back by load_run_metrics, L* must track the
        # full-eval curve — otherwise the receiver is silently using the proxy.
        records = self._write_run_records(
            tmp_path / "baseline",
            [
                {"cycle": 0, "loss_valid": 2.0, "loss_valid_full": 2.5, "gold_combined": 0.3},
                {"cycle": 1, "loss_valid": 1.6},
                {"cycle": 2, "loss_valid": 1.5, "loss_valid_full": 1.8},
                {"cycle": 3, "loss_valid": 1.4},
                {"cycle": 4, "loss_valid": 0.9, "loss_valid_full": 1.2, "gold_combined": 0.4},
                {"cycle": 5, "loss_valid": 0.85},
                {"cycle": 6, "loss_valid": 1.3, "loss_valid_full": 1.1, "gold_combined": 0.45},
            ],
        )
        # The dishonest (pilot) curve would pick 0.85 — the trap the fix closes.
        _, pilot_losses, _ = extract_loss_and_time(records)
        assert min(pilot_losses) == 0.85

        full_times, full_losses, full_cycles = extract_loss_and_time(
            records, full_eval_only=True
        )
        assert full_losses == [2.5, 1.8, 1.2, 1.1]
        assert full_cycles == [0, 2, 4, 6]

        gold = extract_gold(records)
        l_star, l_star_cycle, _, _ = baseline_targets(
            full_losses, full_cycles, full_times, gold
        )
        assert l_star == 1.1  # full-eval minimum, NOT the pilot's 0.9
        assert l_star_cycle == 6
        assert l_star != 0.9  # the receiver did not silently fall back to proxy

    def test_corrupt_input_pilot_would_stop_but_full_eval_does_not(self, tmp_path):
        # Corrupt-input: the SAME guard cycle carries a pilot proxy (0.99) that
        # WOULD trip the §5.2 trigger (<= L*+0.02 = 1.02) and a full-eval loss
        # (1.08) that does NOT. Over a real metrics file the gate must read the
        # honest full-eval column and decline to stop — proving the receiver is
        # not inert. (cf. the in-memory test_quality_gate_uses_full_eval_loss_not_pilot,
        # which proves the plumbing but not the persistence round-trip.)
        baseline_records = self._write_run_records(
            tmp_path / "baseline",
            [
                {"cycle": 0, "loss_valid": 1.6, "loss_valid_full": 1.5, "gold_combined": 0.3},
                {"cycle": 2, "loss_valid": 1.1, "loss_valid_full": 1.0, "gold_combined": 0.5},
                {"cycle": 4, "loss_valid": 1.3, "loss_valid_full": 1.2, "gold_combined": 0.55},
            ],
        )
        guard_records = self._write_run_records(
            tmp_path / "guard",
            [
                {"cycle": 5, "loss_valid": 0.99, "loss_valid_full": 1.08, "gold_combined": 0.60},
            ],
        )

        b_times, b_losses, b_cycles = extract_loss_and_time(
            baseline_records, full_eval_only=True
        )
        l_star, _, g_star, _ = baseline_targets(
            b_losses, b_cycles, b_times, extract_gold(baseline_records)
        )
        assert l_star == 1.0
        assert g_star == 0.5

        guard_gold = extract_gold(guard_records)
        # The receiver read the honest full-eval column, not the manipulable pilot.
        assert guard_gold[0]["loss"] == 1.08
        assert guard_gold[0]["loss"] != 0.99
        stop_cycle, _ = guard_stop_point(guard_gold, l_star=l_star, g_star=g_star)
        assert stop_cycle is None  # full-eval 1.08 > 1.02 → no stop

        # Concrete proof the disagreement is verdict-flipping, not cosmetic:
        # feed the pilot value a corrupt evaluator would have used and the
        # trigger DOES fire at cycle 5.
        pilot_row = {
            "cycle": 5,
            "elapsed": 0.0,
            "loss": 0.99,
            "gold_combined": 0.60,
        }
        corrupt_stop, _ = guard_stop_point([pilot_row], l_star=l_star, g_star=g_star)
        assert corrupt_stop == 5

    def test_byte_identical_fallback_when_no_full_eval_recorded(self, tmp_path):
        # The dormancy state of today's trainer output: no record carries the
        # honest key, so the receiver MUST stay dormant — the full-eval curve is
        # empty and the caller falls back to the pilot proxy, byte-identical to
        # the pre-fix behavior. The two tests above prove the receiver wakes the
        # moment honest data arrives; this one locks the safe fallback so the
        # dormant path can never silently change shape.
        records = self._write_run_records(
            tmp_path / "baseline",
            [
                {"cycle": 0, "loss_valid": 2.0},
                {"cycle": 1, "loss_valid": 1.5},
                {"cycle": 2, "loss_valid": 1.2, "gold_combined": 0.5},
            ],
        )
        assert all(LOSS_VALID_FULL_KEY not in r for r in records)
        assert extract_loss_and_time(records, full_eval_only=True) == ([], [], [])

        # The documented fallback: empty full-eval curve → use the pilot curve.
        times, losses, cycles = extract_loss_and_time(records)
        assert losses == [2.0, 1.5, 1.2]
        l_star, l_star_cycle, _, _ = baseline_targets(
            losses, cycles, times, extract_gold(records)
        )
        assert l_star == 1.2
        assert l_star_cycle == 2


class TestHonestyContractDormancyLoud:
    """The §5.2 honesty contract is DORMANT on every real run today: the trainer
    does not yet emit ``loss_valid_full`` (TASK-0141, GPU-only), so the analyzer
    SILENTLY falls back to the pilot proxy and L* is computed on the
    contaminated proxy — the exact "code looks fixed but the receiver is inert"
    state the §5.2 fix (276991c) exists to prevent. ``TestFullEvalLossHonestyE2E``
    proved the receiver is not inert in code; that is only honest if the
    dormancy is also VISIBLE and owned on every real analysis run, not buried in
    a byte-identical fallback. These corrupt-input tests pin that visibility over
    a real metrics file written by ``RunMetrics.record_step`` and read back by
    ``load_run_metrics`` — the same full round-trip the non-inert tests use.
    """

    def test_dormant_run_warning_names_the_unblocker(self, tmp_path):
        # Today's reality: a real run with NO loss_valid_full records. The
        # contract must report DORMANT and the diagnostic must name TASK-0141
        # (the trainer-emission task that flips it) so the gap is owned, not
        # buried in a silent byte-identical fallback.
        records = TestFullEvalLossHonestyE2E._write_run_records(
            tmp_path / "baseline",
            [
                {"cycle": 0, "loss_valid": 2.0, "gold_combined": 0.3},
                {"cycle": 1, "loss_valid": 1.5, "gold_combined": 0.4},
                {"cycle": 2, "loss_valid": 1.2, "gold_combined": 0.5},
            ],
        )
        status = honesty_contract_status(records)
        assert status["state"] == "dormant"
        assert status["full_eval_records"] == 0
        assert status["step_records"] == 3
        line = format_honesty_contract_line(status)
        assert "DORMANT" in line
        assert "TASK-0141" in line            # the gap is owned, not silent
        assert "pilot proxy" in line          # states L* is on the proxy
        assert LOSS_VALID_FULL_KEY in line

    def test_active_run_carries_no_unblocker_warning(self, tmp_path):
        # Corrupt-input inversion: the SAME run with loss_valid_full on the
        # full-eval cycles is ACTIVE — no TASK-0141 warning, and the partial
        # coverage (2/3) is surfaced so a half-wired trainer is visible too.
        records = TestFullEvalLossHonestyE2E._write_run_records(
            tmp_path / "baseline",
            [
                {"cycle": 0, "loss_valid": 2.0, "loss_valid_full": 2.2, "gold_combined": 0.3},
                {"cycle": 1, "loss_valid": 1.5},
                {"cycle": 2, "loss_valid": 1.2, "loss_valid_full": 1.3, "gold_combined": 0.5},
            ],
        )
        status = honesty_contract_status(records)
        assert status["state"] == "active"
        assert status["full_eval_records"] == 2
        assert status["step_records"] == 3
        line = format_honesty_contract_line(status)
        assert "ACTIVE" in line
        assert "TASK-0141" not in line        # no unblocker needed when active
        assert "2/3" in line                  # partial coverage surfaced

    def test_one_full_eval_record_flips_dormant_to_active(self, tmp_path):
        # Verdict-flip proof the diagnostic is driven by the records, not a
        # hardcoded string: adding a single loss_valid_full record flips
        # DORMANT→ACTIVE and removes the TASK-0141 warning.
        dormant = TestFullEvalLossHonestyE2E._write_run_records(
            tmp_path / "dormant",
            [{"cycle": 0, "loss_valid": 2.0}, {"cycle": 1, "loss_valid": 1.5}],
        )
        active = TestFullEvalLossHonestyE2E._write_run_records(
            tmp_path / "active",
            [
                {"cycle": 0, "loss_valid": 2.0, "loss_valid_full": 2.1},
                {"cycle": 1, "loss_valid": 1.5},
            ],
        )
        d_status = honesty_contract_status(dormant)
        a_status = honesty_contract_status(active)
        assert d_status["state"] == "dormant"
        assert a_status["state"] == "active"
        assert "TASK-0141" in format_honesty_contract_line(d_status)
        assert "TASK-0141" not in format_honesty_contract_line(a_status)

    def test_non_step_records_excluded_from_coverage(self, tmp_path):
        # Corrupt input: footer/header records must not inflate the step total
        # or be mistaken for honest data. Coverage counts step records only.
        records = TestFullEvalLossHonestyE2E._write_run_records(
            tmp_path / "baseline",
            [{"cycle": 0, "loss_valid": 2.0}, {"cycle": 1, "loss_valid": 1.5}],
        )
        # load_run_metrics may carry non-step records (footers); ensure they
        # don't corrupt the coverage denominator.
        records.append({"type": "footer", "best_valid_loss": 1.5})
        status = honesty_contract_status(records)
        assert status["step_records"] == 2
        assert status["state"] == "dormant"

    def test_dormancy_warning_persists_in_gate_decision_file(self, tmp_path):
        # User-visible artifact: a re-analysis of a dormant run must write the
        # warning into gate_decision.txt so the contaminated comparison cannot
        # ship silently. Drive write_gate_decision with a dormant honesty_status
        # over a real metrics file and read the artifact back.
        guard_records = TestFullEvalLossHonestyE2E._write_run_records(
            tmp_path / "guard",
            [
                {"cycle": 0, "loss_valid": 2.0, "gold_combined": 0.3},
                {"cycle": 1, "loss_valid": 1.5, "gold_combined": 0.5},
            ],
        )
        status = honesty_contract_status(guard_records)
        assert status["state"] == "dormant"
        out = tmp_path / "gate_decision.txt"
        write_gate_decision(
            baseline_dir=tmp_path / "baseline",
            guard_dir=tmp_path / "guard",
            baseline_losses=[2.0, 1.5],
            baseline_cycles=[0, 1],
            baseline_times=[0.0, 10.0],
            baseline_gold=[
                {"cycle": 0, "elapsed": 0.0, "loss": 2.0, "gold_combined": 0.3},
                {"cycle": 1, "elapsed": 10.0, "loss": 1.5, "gold_combined": 0.5},
            ],
            guard_losses=[2.0, 1.5],
            guard_times=[0.0, 8.0],
            guard_gold=extract_gold(guard_records),
            schedule={},
            layer_indices=[24, 25],
            target_width=4096,
            freeze_level=2,
            output_path=out,
            baseline_full_losses=[],      # dormant: no full-eval curve
            baseline_full_cycles=[],
            honesty_status=status,
        )
        text = out.read_text()
        assert "DORMANT" in text
        assert "TASK-0141" in text

    def test_active_status_omits_warning_from_gate_decision_file(self, tmp_path):
        # Corrupt-input inversion of the above: an ACTIVE run must NOT write the
        # DORMANT/TASK-0141 warning — proving the gate-decision line is driven by
        # the status, not unconditionally emitted.
        guard_records = TestFullEvalLossHonestyE2E._write_run_records(
            tmp_path / "guard",
            [
                {"cycle": 0, "loss_valid": 2.0, "loss_valid_full": 2.2, "gold_combined": 0.3},
                {"cycle": 1, "loss_valid": 1.5, "loss_valid_full": 1.6, "gold_combined": 0.5},
            ],
        )
        status = honesty_contract_status(guard_records)
        assert status["state"] == "active"
        out = tmp_path / "gate_decision.txt"
        write_gate_decision(
            baseline_dir=tmp_path / "baseline",
            guard_dir=tmp_path / "guard",
            baseline_losses=[2.0, 1.5],
            baseline_cycles=[0, 1],
            baseline_times=[0.0, 10.0],
            baseline_gold=[
                {"cycle": 0, "elapsed": 0.0, "loss": 2.0, "gold_combined": 0.3},
                {"cycle": 1, "elapsed": 10.0, "loss": 1.5, "gold_combined": 0.5},
            ],
            guard_losses=[2.0, 1.5],
            guard_times=[0.0, 8.0],
            guard_gold=extract_gold(guard_records),
            schedule={},
            layer_indices=[24, 25],
            target_width=4096,
            freeze_level=2,
            output_path=out,
            baseline_full_losses=[2.2, 1.6],   # active: full-eval curve present
            baseline_full_cycles=[0, 1],
            honesty_status=status,
        )
        text = out.read_text()
        assert "DORMANT" not in text
        assert "TASK-0141" not in text
        assert "ACTIVE" in text


class TestLossCurveSkipsNullValues:
    """§5.2 receiver honesty: ``extract_loss_and_time`` must skip records whose
    loss value is ``None``, in both the pilot (``loss_valid``) and full-eval
    (``loss_valid_full``) curves.

    A ``None`` in the curve poisons it and crashes L* — ``baseline_targets`` does
    ``min(losses)``, and ``min`` over a ``None``-containing list raises
    ``TypeError``. For ``loss_valid_full`` a null would *also* disagree with
    ``honesty_contract_status`` (which already skips nulls), splitting the
    receiver into two detectors that report different full-eval counts — the
    silent-mislead state §5.2 exists to prevent.

    The pilot path is not hypothetical: the baseline trainer persists
    ``loss_valid`` unconditionally (``run_metrics.record_step`` writes it even
    when ``None``), so every non-eval step in a real ``run_metrics.jsonl`` carries
    ``loss_valid: null``. Until the trainer emits ``loss_valid_full`` the
    analyzer is DORMANT and falls back to exactly that pilot curve — so without
    this skip the §5.2 L* computation crashes on the first real baseline run.
    """

    def test_pilot_curve_skips_null_loss_valid(self, tmp_path):
        # Realistic baseline shape: per-step records carry loss_valid=null.
        records = [
            {"type": "step", "cycle": 0, "elapsed_seconds": 0.0, "loss_valid": 2.0},
            {"type": "step", "cycle": 1, "elapsed_seconds": 5.0, "loss_valid": None},
            {"type": "step", "cycle": 2, "elapsed_seconds": 10.0,
             "loss_valid": 1.4, "gold_combined": 0.4},
        ]
        times, losses, cycles = extract_loss_and_time(records)
        assert losses == [2.0, 1.4]  # the null cycle 1 is dropped, not included
        assert cycles == [0, 2]
        # L* must compute, not crash on the null that would otherwise be in the list.
        l_star, l_star_cycle, _, _ = baseline_targets(
            losses, cycles, times, extract_gold(records)
        )
        assert l_star == 1.4
        assert l_star_cycle == 2

    def test_full_eval_curve_skips_null_loss_valid_full(self):
        # Corrupt input: a botched trainer wiring wrote loss_valid_full=null on a
        # non-full-eval cycle (key present, value null).
        records = [
            {"type": "step", "cycle": 0, "elapsed_seconds": 0.0,
             "loss_valid": 2.0, "loss_valid_full": 2.5},
            {"type": "step", "cycle": 1, "elapsed_seconds": 5.0,
             "loss_valid": 1.6, "loss_valid_full": None},
            {"type": "step", "cycle": 2, "elapsed_seconds": 10.0,
             "loss_valid": 1.4, "loss_valid_full": 1.8},
        ]
        times, losses, cycles = extract_loss_and_time(records, full_eval_only=True)
        assert losses == [2.5, 1.8]
        assert cycles == [0, 2]
        # No None reaches baseline_targets -> L* computes instead of crashing.
        l_star, _, _, _ = baseline_targets(losses, cycles, times, extract_gold(records))
        assert l_star == 1.8

    def test_null_loss_valid_full_keeps_both_detectors_consistent(self):
        # The two full-eval detectors must agree: a null loss_valid_full is NOT a
        # full-eval cycle for either. Before the fix, honesty_contract_status
        # reported 2 full-eval records while extract_loss_and_time returned 3
        # (including the null) and then crashed L*.
        records = [
            {"type": "step", "cycle": 0, "loss_valid": 2.0, "loss_valid_full": 2.5},
            {"type": "step", "cycle": 1, "loss_valid": 1.6, "loss_valid_full": None},
            {"type": "step", "cycle": 2, "loss_valid": 1.4, "loss_valid_full": 1.8},
        ]
        status = honesty_contract_status(records)
        _, losses, _ = extract_loss_and_time(records, full_eval_only=True)
        assert status["full_eval_records"] == len(losses) == 2
        assert None not in losses

    def test_real_persistence_path_skips_null_loss_valid(self, tmp_path):
        # End-to-end through the real writer: a record_step call that omits
        # loss_valid persists "loss_valid": null (unconditional write). The reader
        # must still skip it so the curve the analyzer builds is clean.
        m = RunMetrics(tmp_path / "run", mode="baseline")
        m.record_step(step=1, cycle=0, loss_train=2.0, loss_valid=2.0,
                      backward_passes=1, total_backward_passes=1)
        m.record_step(step=2, cycle=1, loss_train=1.9, backward_passes=1,
                      total_backward_passes=2)  # no loss_valid -> persisted null
        m.close()
        records = load_run_metrics(tmp_path / "run")
        null_records = [r for r in records
                        if r.get("type") == "step" and r.get("loss_valid") is None]
        assert null_records, "fixture sanity: a null loss_valid record was persisted"
        _, losses, _ = extract_loss_and_time(records)
        assert None not in losses
        assert losses == [2.0]
