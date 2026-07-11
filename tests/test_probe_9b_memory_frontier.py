"""Unit tests for scripts/probe_9b_memory_frontier.py (pure/orchestrable core).

These do NOT require a GPU: the sweep/OOM-handling logic is exercised with a
fake ``step_runner`` so the OOM-catch is proven load-bearing without a live
CUDA context. The real GPU run is a separate, opt-in smoke (``make
probe-9b-memory``), not part of this unit suite.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch
from omegaconf import OmegaConf

from scripts.probe_9b_memory_frontier import (
    FrontierMeasurement,
    FrontierResult,
    PeakMemoryOOM,
    build_synthetic_batch,
    config_levers_from_cfg,
    format_report,
    full_section4_producible,
    gpu_cleanup,
    make_gpu_step_runner,
    max_fit_seq_len,
    parse_seq_lens,
    sweep_seq_lens,
)
from scripts import probe_9b_memory_frontier as probe_mod


# ── fixtures / helpers ──────────────────────────────────────────────────────


def _fake_step_factory(oom_above: int):
    """A step_runner that reports a synthetic peak and OOMs above a threshold."""

    def step(batch):
        sl = batch["seq_len"]
        if sl > oom_above:
            raise PeakMemoryOOM("fake cuda OOM")
        return (float(sl) * 10.0, float(sl) * 11.0)

    return step


def _bb(seq_len):
    return {"seq_len": seq_len}


# ── sweep_seq_lens + OOM handling ───────────────────────────────────────────


class TestSweepSeqLens:
    def test_fit_below_threshold_oom_above(self):
        ms = sweep_seq_lens([256, 512, 1024, 1536], _bb, _fake_step_factory(1024), None)
        assert [m.fit for m in ms] == [True, True, True, False]
        assert ms[-1].error is not None and "OOM" in ms[-1].error
        assert ms[-1].peak_alloc_mb is None

    def test_cleanup_invoked_per_seqlen(self):
        seen = []
        sweep_seq_lens(
            [256, 512],
            _bb,
            _fake_step_factory(10_000),
            cleanup=lambda sl: seen.append(sl),
        )
        assert seen == [256, 512]

    def test_oom_does_not_short_circuit_larger_seqlens(self):
        # 1536 OOMs but 2048 is still attempted+recorded (not silently skipped).
        ms = sweep_seq_lens([1024, 1536, 2048], _bb, _fake_step_factory(1024), None)
        assert [m.seq_len for m in ms] == [1024, 1536, 2048]
        assert ms[0].fit is True
        assert ms[1].fit is False
        assert ms[2].fit is False

    def test_non_oom_exception_propagates(self):
        def bad_step(batch):
            raise ValueError("not an OOM")

        with pytest.raises(ValueError, match="not an OOM"):
            sweep_seq_lens([256], _bb, bad_step, None)


class TestSweepOomCatchIsLoadBearing:
    """If sweep stopped catching PeakMemoryOOM, OOM would propagate instead of
    being recorded as fit=False. Mutation: drop the try/except -> test REDs."""

    def test_oom_recorded_not_raised(self):
        ms = sweep_seq_lens([2048], _bb, _fake_step_factory(1024), None)
        assert len(ms) == 1
        assert ms[0].fit is False
        assert ms[0].error is not None


# ── derived verdicts ────────────────────────────────────────────────────────


class TestMaxFitSeqLen:
    def test_returns_largest_fit(self):
        ms = sweep_seq_lens([256, 512, 1024, 1536], _bb, _fake_step_factory(1024), None)
        assert max_fit_seq_len(ms) == 1024

    def test_none_when_all_oom(self):
        ms = sweep_seq_lens([1536, 2048], _bb, _fake_step_factory(1024), None)
        assert max_fit_seq_len(ms) is None


class TestFullSection4Producible:
    def test_true_when_above_threshold_fits(self):
        ms = sweep_seq_lens([256, 512, 1024, 1536], _bb, _fake_step_factory(1536), None)
        assert full_section4_producible(ms, threshold=1024) is True

    def test_false_when_above_threshold_all_oom(self):
        ms = sweep_seq_lens([256, 1024, 1536], _bb, _fake_step_factory(768), None)
        assert full_section4_producible(ms, threshold=1024) is False

    def test_none_when_nothing_above_threshold_probed(self):
        ms = sweep_seq_lens([256, 512], _bb, _fake_step_factory(10_000), None)
        assert full_section4_producible(ms, threshold=1024) is None


# ── synthetic batch ─────────────────────────────────────────────────────────


class TestBuildSyntheticBatch:
    def test_shapes_and_full_supervision(self):
        batch = build_synthetic_batch(vocab_size=32000, pad_token_id=0,
                                      seq_len=128, batch_size=2,
                                      device=torch.device("cpu"))
        assert batch["input_ids"].shape == (2, 128)
        assert batch["attention_mask"].shape == (2, 128)
        assert torch.equal(batch["labels"], batch["input_ids"])
        assert torch.equal(batch["attention_mask"], torch.ones(2, 128, dtype=torch.long))

    def test_pad_token_excluded_from_inputs(self):
        # pad id 5 must not appear in input_ids (avoids degenerate all-pad).
        batch = build_synthetic_batch(vocab_size=10, pad_token_id=5,
                                      seq_len=512, batch_size=1,
                                      device=torch.device("cpu"))
        assert not torch.any(batch["input_ids"] == 5)

    def test_tokens_within_vocab(self):
        batch = build_synthetic_batch(vocab_size=100, pad_token_id=0,
                                      seq_len=64, batch_size=1,
                                      device=torch.device("cpu"))
        assert batch["input_ids"].max().item() < 100
        assert batch["input_ids"].min().item() >= 0


# ── CLI / parsing / config attribution ──────────────────────────────────────


class TestParseSeqLens:
    def test_basic(self):
        assert parse_seq_lens("256, 1024 ,2048") == [256, 1024, 2048]

    def test_single(self):
        assert parse_seq_lens("512") == [512]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_seq_lens(",,,")


class TestConfigLevers:
    def test_extracts_suffix_only_levers(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "name_or_path": "Qwen/Qwen3.5-9B",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "dtype": "bfloat16",
                },
                "training": {
                    "trainable_lora_scope": "last_25_percent",
                    "gradient_checkpointing": True,
                    "prefix_feature_cache_experimental": True,
                    "prefix_feature_cache_offload_prefix_to_cpu": True,
                },
            }
        )
        levers = config_levers_from_cfg(cfg)
        assert levers["trainable_lora_scope"] == "last_25_percent"
        assert levers["load_in_4bit"] is True
        assert levers["bnb_4bit_quant_type"] == "nf4"
        assert levers["prefix_feature_cache_offload_prefix_to_cpu"] is True
        # The probe never exercises prefix offload — attribution must say so.
        assert levers["prefix_feature_cache_offload_exercised"] is False

    def test_defaults_when_keys_absent(self):
        cfg = OmegaConf.create({"model": {}, "training": {}})
        levers = config_levers_from_cfg(cfg)
        assert levers["trainable_lora_scope"] == "all"
        assert levers["gradient_checkpointing"] is True


# ─_ report formatting ───────────────────────────────────────────────────────


class TestFormatReport:
    def _result(self, measurements, threshold=1024):
        return FrontierResult(
            gpu_name="RTX 3060",
            gpu_total_mb=12288.0,
            seq_lens=[m.seq_len for m in measurements],
            measurements=measurements,
            max_fit_seq_len=max_fit_seq_len(measurements),
            full_section4_producible=full_section4_producible(measurements, threshold),
            config_levers={
                "model": "Qwen/Qwen3.5-9B",
                "trainable_lora_scope": "last_25_percent",
                "load_in_4bit": True,
                "gradient_checkpointing": True,
            },
        )

    def test_report_contains_verdict_and_frontier(self):
        ms = sweep_seq_lens([256, 512, 1024, 1536], _bb, _fake_step_factory(1024), None)
        report = format_report(self._result(ms))
        assert "max fit seq_len: 1024" in report
        assert "full §4" in report
        assert "YES" in report  # 1024 fit -> producible

    def test_report_oom_verdict_when_blocked(self):
        ms = sweep_seq_lens([256, 512, 1024], _bb, _fake_step_factory(512), None)
        report = format_report(self._result(ms))
        assert "NO" in report
        # names the recovery lever (case-insensitive — report capitalizes "Prefix").
        assert "prefix-cache cpu offload" in report.lower()

    def test_report_unmeasured_when_below_threshold_only(self):
        ms = sweep_seq_lens([256, 512], _bb, _fake_step_factory(10_000), None)
        report = format_report(self._result(ms))
        assert "UNMEASURED" in report


# ─_ OOM detection helper + gpu_cleanup no-op-on-cpu ─────────────────────────


class TestIsOom:
    def test_torch_oom_class(self):
        assert probe_mod._is_oom(torch.cuda.OutOfMemoryError("x")) is True

    def test_runtime_oom_message(self):
        assert probe_mod._is_oom(RuntimeError("CUDA out of memory. Tried to allocate...")) is True

    def test_unrelated_runtime_error(self):
        assert probe_mod._is_oom(RuntimeError("shape mismatch")) is False

    def test_value_error(self):
        assert probe_mod._is_oom(ValueError("nope")) is False


class TestGpuCleanup:
    def test_safe_without_cuda(self, monkeypatch):
        # gpu_cleanup must not crash when CUDA is unavailable (it guards).
        monkeypatch.setattr(probe_mod.torch.cuda, "is_available", lambda: False)
        gpu_cleanup(1024)  # should be a no-op, not raise


# ─_ make_gpu_step_runner OOM wrapping (model-free) ──────────────────────────


class TestGpuStepRunnerOomWrapping:
    """The runner must translate a model forward that raises a CUDA OOM into a
    PeakMemoryOOM so the sweep records fit=False. We exercise the wrapping by
    monkeypatching reset_peak_memory_stats (no real CUDA step needed)."""

    def test_oom_from_model_forward_wrapped(self, monkeypatch):
        class _FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                raise torch.cuda.OutOfMemoryError("tried to allocate 4 GiB")

        monkeypatch.setattr(probe_mod.torch.cuda, "reset_peak_memory_stats", lambda: None)
        run_step, _opt = make_gpu_step_runner(_FakeModel(), lr=1e-4, grad_accumulation=1)
        with pytest.raises(PeakMemoryOOM):
            run_step({"input_ids": None, "attention_mask": None, "labels": None})

    def test_non_oom_forward_propagates(self, monkeypatch):
        class _FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                raise RuntimeError("some other forward error")

        monkeypatch.setattr(probe_mod.torch.cuda, "reset_peak_memory_stats", lambda: None)
        run_step, _opt = make_gpu_step_runner(_FakeModel(), lr=1e-4, grad_accumulation=1)
        with pytest.raises(RuntimeError, match="some other forward error"):
            run_step({"input_ids": None, "attention_mask": None, "labels": None})


# ─_ FrontierResult.to_dict round-trips for JSON deposit ─────────────────────


class TestFrontierResultSerialization:
    def test_to_dict_is_json_serializable(self):
        ms = sweep_seq_lens([256, 1024, 2048], _bb, _fake_step_factory(1024), None)
        result = FrontierResult(
            gpu_name="RTX 3060",
            gpu_total_mb=12288.0,
            seq_lens=[256, 1024, 2048],
            measurements=ms,
            max_fit_seq_len=max_fit_seq_len(ms),
            full_section4_producible=full_section4_producible(ms),
            config_levers={"model": "X"},
            note="probe",
        )
        blob = json.dumps(result.to_dict())
        back = json.loads(blob)
        assert back["max_fit_seq_len"] == 1024
        assert len(back["measurements"]) == 3
        assert back["measurements"][2]["fit"] is False


# ── Replay / faithfulness: re-derive the recorded verdict from the fixture ───
#
# The fixture (tests/fixtures/probe_9b_memory_frontier.json) is a checked-in
# snapshot of a *real* 9B GPU run. These tests reconstruct FrontierMeasurement
# lists from each arm's raw measurements and re-derive the verdict via the same
# pure functions the probe used at deposit time (max_fit_seq_len /
# full_section4_producible). If the recorded verdict ever drifts from what the
# pure functions would re-derive, these tests fail — a GPU-free guard that the
# fixture and the probe logic stay faithful to each other.

_FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parent / "fixtures" / "probe_9b_memory_frontier.json"
)


def _measurements_from_arm(arm: dict) -> list:
    """Rebuild FrontierMeasurement objects from a fixture arm's raw rows."""
    return [
        FrontierMeasurement(
            seq_len=row["seq_len"],
            peak_alloc_mb=row["peak_alloc_mb"],
            peak_reserved_mb=row["peak_reserved_mb"],
            fit=row["fit"],
            error=None,
        )
        for row in arm["measurements"]
    ]


class TestFixtureReplay:
    """Re-derive the recorded verdict from the fixture with no GPU."""

    @pytest.fixture(scope="class")
    def fixture(self):
        with open(_FIXTURE_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_fixture_is_present_and_well_formed(self, fixture):
        # Guards the deposit exists and names both arms — otherwise these
        # replay tests silently skip the whole point of the probe.
        assert fixture["probe"] == "9b_suffix_only_memory_frontier"
        assert "suffix_only_last25" in fixture["arms"]
        assert "standard_all" in fixture["arms"]
        for arm in fixture["arms"].values():
            assert "measurements" in arm and len(arm["measurements"]) >= 1

    def test_suffix_only_reders_full_section4_true(self, fixture):
        ms = _measurements_from_arm(fixture["arms"]["suffix_only_last25"])
        # The decisive claim: the suffix-only config is recorded as producing
        # the full §4 verdict (a seq>=1024 step fits). Re-derived, not asserted.
        assert max_fit_seq_len(ms) == 1024
        assert full_section4_producible(ms) is True

    def test_standard_reders_full_section4_false(self, fixture):
        ms = _measurements_from_arm(fixture["arms"]["standard_all"])
        # The controlled contrast: the standard config does NOT reach seq1024.
        assert max_fit_seq_len(ms) == 768
        assert full_section4_producible(ms) is False

    def test_controlled_contrast_seq1024(self, fixture):
        # The single lever that differs is trainable_lora_scope. Holding
        # model/GPU/quant fixed, only the suffix-only arm fits seq1024.
        suffix = _measurements_from_arm(fixture["arms"]["suffix_only_last25"])
        standard = _measurements_from_arm(fixture["arms"]["standard_all"])
        suffix_1024 = next(m for m in suffix if m.seq_len == 1024)
        standard_1024 = next(m for m in standard if m.seq_len == 1024)
        assert suffix_1024.fit is True and standard_1024.fit is False

    def test_recorded_verdict_matches_redersived_legs(self, fixture):
        # The fixture's own `verdict` string must be consistent with the legs
        # the pure functions re-derive (faithfulness between prose and data).
        suffix = _measurements_from_arm(fixture["arms"]["suffix_only_last25"])
        verdict = fixture["verdict"].lower()
        if full_section4_producible(suffix) is True:
            assert "memory block" in verdict and "removed" in verdict, (
                "suffix-only fits seq1024 but the verdict prose does not state "
                "the memory block is removed"
            )
