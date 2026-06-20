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

from scripts.analyze_dynfreeze_experiment import _proxy_speed_gate_section
from src.tg_lora.freeze_cost import (
    PROXY_VALIDATED_MAX_WIDTH,
    ReductionSample,
    calibrate_reduction_band,
    format_reduction_band,
    format_speed_gate_verdict,
    frozen_at_epoch_from_freeze_log,
    per_cycle_realized_reductions,
    speed_gate_verdict,
    uniform_layer_accountant,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PROVISIONAL_PASS,
    VERDICT_REQUIRES_SCALE_MEASUREMENT,
)

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
