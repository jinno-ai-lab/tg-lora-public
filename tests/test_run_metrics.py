import json
import math
from unittest.mock import patch

import pytest
import torch

from src.eval.eval_loss import EvalLossResult, eval_loss_detailed
from src.utils.run_metrics import RunMetrics


class FakeCfg:
    class model:
        name_or_path = "test-model"
        device = "cpu"

    class training:
        batch_size = 1
        grad_accumulation = 1
        learning_rate = 1e-4
        max_steps = 10
        optimizer_lifecycle = "recreate_per_cycle"

    class lora:
        r = 8
        alpha = 16

    class experiment:
        seed = 42

    class logging:
        pass

    class eval:
        pass


def test_write_header_and_steps(tmp_path):
    cfg = FakeCfg()
    m = RunMetrics(tmp_path, mode="baseline")

    m.write_header(
        cfg,
        budget_type="backward_passes",
        budget_value=10,
        param_counts={"total": 100, "trainable": 10},
        comparison_keys={"epoch_batch_plan_key": "plan-123"},
    )

    m.record_step(step=1, loss_train=3.0, backward_passes=1, total_backward_passes=1)
    m.record_step(step=2, loss_train=2.5, backward_passes=1, total_backward_passes=2)

    m.write_footer(best_valid_loss=2.5, best_valid_step=2, final_train_loss=2.5)
    m.close()

    lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 4  # header + 2 steps + footer

    header = json.loads(lines[0])
    assert header["type"] == "run_header"
    assert header["mode"] == "baseline"
    assert header["model_name"] == "test-model"
    assert header["optimizer_lifecycle"] == "recreate_per_cycle"
    assert header["comparison_keys"]["epoch_batch_plan_key"] == "plan-123"

    step1 = json.loads(lines[1])
    assert step1["type"] == "step"
    assert step1["loss_train"] == 3.0
    assert step1["total_backward_passes"] == 1
    assert step1["tg_lora_accepted"] is None

    footer = json.loads(lines[3])
    assert footer["type"] == "run_footer"
    assert footer["best_valid_loss"] == 2.5


def test_loss_valid_full_is_emission_surface_not_default_key(tmp_path):
    # §5.1/§5.2 honesty contract: ``loss_valid_full`` is the full-eval loss the
    # analyzer prefers over the pilot proxy ``loss_valid``. It is recorded ONLY
    # on full-eval cycles, and the analyzer detects those cycles by *key
    # presence* (``LOSS_VALID_FULL_KEY in record``), not by value. So:
    #   - when supplied, the key is persisted verbatim (the emission surface the
    #     trainer uses to feed the receiver);
    #   - when omitted, the key is ABSENT entirely (not ``None``) so legacy /
    #     pilot-only records stay byte-identical and the receiver stays dormant
    #     until honest data arrives.
    m = RunMetrics(tmp_path, mode="baseline")
    m.record_step(
        step=1, cycle=1, loss_train=1.0, backward_passes=1, total_backward_passes=1
    )  # pilot-only cycle: no loss_valid_full
    m.record_step(
        step=2,
        cycle=2,
        loss_train=1.0,
        loss_valid=0.99,
        loss_valid_full=1.08,  # full-eval cycle: honest signal emitted
        backward_passes=1,
        total_backward_passes=2,
    )
    m.close()

    steps = [
        json.loads(line)
        for line in (tmp_path / "run_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pilot_only, full_eval = steps

    assert "loss_valid_full" not in pilot_only  # absence == not a full-eval cycle
    with pytest.raises(KeyError):
        pilot_only["loss_valid_full"]

    assert full_eval["loss_valid_full"] == 1.08  # presence == honest signal emitted
    assert full_eval["loss_valid"] == 0.99  # pilot proxy still carried alongside


def test_tg_lora_fields(tmp_path):
    cfg = FakeCfg()

    class TgCfg:
        K_initial = 3
        N_initial = 5
        alpha_initial = 0.3
        beta_initial = 0.8
        lr_initial = 5e-4
        accel_instability_lr_decay = 0.7
        accel_convergence_lr_boost = 1.1
        enable_random_walk = True
        enable_convergence_adaptation = True

    cfg.tg_lora = TgCfg()
    cfg.training.trainable_lora_scope = "last_25_percent"
    cfg.training.prefix_feature_cache_experimental = True
    cfg.training.prefix_feature_cache_train = True
    cfg.training.prefix_feature_cache_valid_quick = True
    cfg.training.prefix_feature_cache_valid_full = True
    cfg.training.prefix_feature_cache_mode = "reuse"
    cfg.training.prefix_feature_cache_share_across_seeds = True
    cfg.training.prefix_feature_cache_offload_prefix_to_cpu = True
    cfg.training.train_on_prompt = False
    m = RunMetrics(tmp_path, mode="tg_lora")

    m.write_header(cfg, budget_type="cycles", budget_value=100, param_counts=None)

    m.record_step(
        step=3,
        cycle=0,
        loss_train=2.0,
        loss_valid=2.1,
        backward_passes=3,
        total_backward_passes=3,
        tg_lora_accepted=True,
        tg_lora_cosine_sim=0.5,
        tg_lora_raw_delta_cosine_sim=0.1,
        tg_lora_predicted_consistency=0.5,
        tg_lora_short_long_norm_ratio=0.9,
        tg_lora_reduction_rate=0.625,
        tg_lora_K=3,
        tg_lora_N=5,
        tg_lora_proposed_N=10,
        tg_lora_alpha=0.3,
        tg_lora_beta=0.8,
        tg_lora_cache_built=True,
        tg_lora_cache_eligible=True,
        tg_lora_cache_hit=True,
        tg_lora_validation_forwards=2,
        tg_lora_pilot_validation_forwards=1,
        tg_lora_post_validation_forwards=1,
        tg_lora_post_extrapolation_eval=True,
        tg_lora_post_extrapolation_eval_skipped=False,
        tg_lora_post_extrapolation_eval_skip_reason="low_confidence",
        tg_lora_rollback_triggered=False,
        tg_lora_cap_global_ratio=0.75,
        tg_lora_cap_mean_ratio=0.8,
        tg_lora_cap_min_ratio=0.5,
        tg_lora_cap_capped_fraction=0.25,
        tg_lora_cap_capped_tensors=1,
        tg_lora_cap_tensors=4,
        tg_lora_raw_update_norm=2.0,
        tg_lora_applied_update_norm=1.5,
    )

    m.write_footer(
        best_valid_loss=2.1,
        best_valid_step=3,
        final_train_loss=2.0,
        tg_lora_summary={"total_cycles": 1, "accepted": 1},
    )
    m.close()

    lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
    header = json.loads(lines[0])
    step = json.loads(lines[1])
    assert header["accel_instability_lr_decay"] == 0.7
    assert header["accel_convergence_lr_boost"] == 1.1
    assert header["tg_lora_K_initial"] == 3
    assert header["trainable_lora_scope"] == "last_25_percent"
    assert header["prefix_feature_cache_experimental"] is True
    assert header["train_on_prompt"] is False
    assert step["tg_lora_accepted"] is True
    assert step["tg_lora_raw_delta_cosine_sim"] == 0.1
    assert step["tg_lora_predicted_consistency"] == 0.5
    assert step["tg_lora_short_long_norm_ratio"] == 0.9
    assert step["tg_lora_reduction_rate"] == 0.625
    assert step["tg_lora_proposed_N"] == 10
    assert step["cycle"] == 0
    assert step["tg_lora_cache_built"] is True
    assert step["tg_lora_cache_eligible"] is True
    assert step["tg_lora_cache_hit"] is True
    assert step["tg_lora_validation_forwards"] == 2
    assert step["tg_lora_pilot_validation_forwards"] == 1
    assert step["tg_lora_post_validation_forwards"] == 1
    assert step["tg_lora_post_extrapolation_eval"] is True
    assert step["tg_lora_post_extrapolation_eval_skipped"] is False
    assert step["tg_lora_post_extrapolation_eval_skip_reason"] == "low_confidence"
    assert step["tg_lora_rollback_triggered"] is False


def test_write_footer_preserves_cache_summary(tmp_path):
    m = RunMetrics(tmp_path, mode="tg_lora")
    m.write_footer(
        best_valid_loss=1.0,
        best_valid_step=3,
        final_train_loss=1.1,
        tg_lora_summary={
            "activation_cache_build_count": 3,
            "activation_cache_eligible_count": 2,
            "activation_cache_hit_count": 1,
            "activation_cache_miss_count": 1,
            "activation_cache_hit_rate": 0.5,
        },
    )
    m.close()

    footer = json.loads((tmp_path / "run_metrics.jsonl").read_text().strip())
    summary = footer["tg_lora_summary"]
    assert summary["activation_cache_build_count"] == 3
    assert summary["activation_cache_hit_count"] == 1
    assert summary["activation_cache_hit_rate"] == 0.5


def test_elapsed_seconds_increase(tmp_path):
    cfg = FakeCfg()
    m = RunMetrics(tmp_path, mode="baseline")

    m.write_header(
        cfg, budget_type="backward_passes", budget_value=2, param_counts=None
    )
    r1 = m.record_step(
        step=1, loss_train=3.0, backward_passes=1, total_backward_passes=1
    )

    import time

    time.sleep(0.05)

    r2 = m.record_step(
        step=2, loss_train=2.0, backward_passes=1, total_backward_passes=2
    )

    assert r2["elapsed_seconds"] > r1["elapsed_seconds"]
    m.close()


def test_run_id_property(tmp_path):
    m = RunMetrics(tmp_path, mode="baseline", run_id="custom-id-123")
    assert m.run_id == "custom-id-123"
    m.close()


def test_auto_run_id(tmp_path):
    m = RunMetrics(tmp_path, mode="tg_lora")
    assert m.run_id.startswith("tg_lora_")
    m.close()


def test_close_idempotent(tmp_path):
    m = RunMetrics(tmp_path, mode="baseline")
    m.close()
    m.close()  # should not raise


def test_record_step_returns_dict(tmp_path):
    m = RunMetrics(tmp_path, mode="baseline")
    result = m.record_step(
        step=1, loss_train=1.0, backward_passes=1, total_backward_passes=1
    )
    assert isinstance(result, dict)
    assert result["step"] == 1
    assert result["loss_train"] == 1.0
    m.close()


def test_gpu_peak_tracking(tmp_path):
    m = RunMetrics(tmp_path, mode="baseline")
    r1 = m.record_step(
        step=1, loss_train=3.0, backward_passes=1, total_backward_passes=1
    )
    assert r1["gpu_peak_mb"] == 0.0  # no GPU in CI
    m.close()


def test_context_manager(tmp_path):
    with RunMetrics(tmp_path, mode="baseline") as m:
        assert isinstance(m, RunMetrics)
        m.record_step(
            step=1, loss_train=1.0, backward_passes=1, total_backward_passes=1
        )
    # File should be closed after context manager exits
    assert m._file.closed


def test_context_manager_closes_on_exception(tmp_path):
    try:
        with RunMetrics(tmp_path, mode="baseline") as m:
            m.record_step(
                step=1, loss_train=1.0, backward_passes=1, total_backward_passes=1
            )
            raise ValueError("test error")
    except ValueError:
        pass
    assert m._file.closed


# --- GPU-path mock tests (TASK-0008) ---


def test_init_resets_peak_memory(tmp_path):
    with patch("src.utils.run_metrics.gpu_reset_peak_stats") as mock_reset:
        m = RunMetrics(tmp_path, mode="baseline")
        mock_reset.assert_called_once()
        m.close()


def test_write_header_with_gpu(tmp_path):
    cfg = FakeCfg()

    with (
        patch("src.utils.run_metrics.gpu_reset_peak_stats"),
        patch(
            "src.utils.run_metrics.gpu_info_dict",
            return_value={
                "name": "NVIDIA RTX 3060",
                "total_mb": 12288.0,
                "type": "cuda",
            },
        ),
    ):
        m = RunMetrics(tmp_path, mode="baseline")
        m.write_header(cfg, budget_type="backward_passes", budget_value=10)
        m.close()

        lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
        header = json.loads(lines[0])
        assert header["gpu_name"] == "NVIDIA RTX 3060"
        assert header["gpu_total_memory_mb"] == 12288.0


def test_write_header_optimizer_lifecycle_none_when_missing(tmp_path):
    """When cfg.training has no optimizer_lifecycle attr, header should record None."""

    class MinimalCfg:
        class model:
            name_or_path = "test"
            device = "cpu"

        class training:
            batch_size = 1
            grad_accumulation = 1
            learning_rate = 1e-4

        class lora:
            r = 8
            alpha = 16

        class experiment:
            seed = 42

    cfg = MinimalCfg()
    m = RunMetrics(tmp_path, mode="baseline")
    m.write_header(cfg, budget_type="backward_passes", budget_value=10)
    m.close()

    lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
    header = json.loads(lines[0])
    assert header["optimizer_lifecycle"] is None


def test_record_step_gpu_vram(tmp_path):
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.reset_peak_memory_stats"),
        patch(
            "src.utils.run_metrics.vram_usage_mb",
            return_value={
                "gpu0_allocated_mb": 1024.5,
                "gpu0_reserved_mb": 2048.3,
            },
        ),
        patch("torch.cuda.max_memory_allocated", return_value=3 * 1024**2),
    ):
        m = RunMetrics(tmp_path, mode="baseline")
        result = m.record_step(
            step=1, loss_train=1.0, backward_passes=1, total_backward_passes=1
        )
        m.close()

        assert result["gpu_allocated_mb"] == 1024.5
        assert result["gpu_reserved_mb"] == 2048.3


def test_record_step_peak_memory_update(tmp_path):
    with (
        patch("src.utils.run_metrics.gpu_reset_peak_stats"),
        patch(
            "src.utils.run_metrics.vram_usage_mb",
            return_value={
                "gpu0_allocated_mb": 100.0,
                "gpu0_reserved_mb": 200.0,
            },
        ),
        patch("src.utils.run_metrics.gpu_peak_memory_mb") as mock_peak,
    ):
        mock_peak.side_effect = [5.0, 3.0]  # 5 MB then 3 MB

        m = RunMetrics(tmp_path, mode="baseline")

        r1 = m.record_step(
            step=1, loss_train=1.0, backward_passes=1, total_backward_passes=1
        )
        assert r1["gpu_peak_mb"] == 5.0

        r2 = m.record_step(
            step=2, loss_train=0.5, backward_passes=1, total_backward_passes=2
        )
        assert r2["gpu_peak_mb"] == 5.0  # peak stays (3 < 5)
        m.close()


# --- TASK-0040: perplexity field in write_footer ---


def _footer_record(tmp_path, **footer_kwargs):
    """Write a footer with given kwargs and return the parsed JSONL footer record."""
    cfg = FakeCfg()
    m = RunMetrics(tmp_path, mode="baseline")
    m.write_header(cfg, budget_type="backward_passes", budget_value=10)
    m.write_footer(**footer_kwargs)
    m.close()
    lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
    return json.loads(lines[-1])


def test_write_footer_perplexity_normal(tmp_path):
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
        perplexity=7.389,
    )
    assert "perplexity" in footer
    assert footer["perplexity"] == 7.389


def test_write_footer_perplexity_none(tmp_path):
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
    )
    assert footer["perplexity"] is None


def test_write_footer_perplexity_nan(tmp_path):
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
        perplexity=float("nan"),
    )
    assert footer["perplexity"] is None


def test_write_footer_perplexity_inf(tmp_path):
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
        perplexity=float("inf"),
    )
    assert footer["perplexity"] is None


def test_write_footer_perplexity_neg_inf(tmp_path):
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
        perplexity=float("-inf"),
    )
    assert footer["perplexity"] is None


def test_write_footer_perplexity_negative(tmp_path):
    """Negative perplexity is physically impossible; should be sanitized to None."""
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
        perplexity=-1.5,
    )
    assert footer["perplexity"] is None


def test_write_footer_perplexity_zero(tmp_path):
    """Zero perplexity is physically impossible; should be sanitized to None."""
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
        perplexity=0.0,
    )
    assert footer["perplexity"] is None


def test_write_footer_perplexity_very_large(tmp_path):
    """Very large but finite perplexity is beyond meaningful range; sanitized to None."""
    large_ppl = 1e308
    footer = _footer_record(
        tmp_path,
        best_valid_loss=2.0,
        best_valid_step=5,
        final_train_loss=1.8,
        perplexity=large_ppl,
    )
    assert footer["perplexity"] is None


def test_e2e_eval_loss_result_to_run_metrics(tmp_path):
    """E2E: eval_loss_detailed result propagates perplexity through RunMetrics.write_footer."""
    from torch.utils.data import DataLoader

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, input_ids, attention_mask=None, labels=None):
            logits = self.linear(input_ids.float())
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
                return type("Out", (), {"loss": loss})()
            return type("Out", (), {"loss": torch.tensor(0.0)})()

    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self):
            self.input_ids = torch.randint(0, 10, (4, 4))
            self.attention_mask = torch.ones(4, 4, dtype=torch.long)
            self.labels = torch.randint(0, 2, (4,))

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    torch.manual_seed(42)
    model = _TinyModel()
    loader = DataLoader(_DS(), batch_size=2)

    result = eval_loss_detailed(model, loader, device="cpu")
    assert isinstance(result, EvalLossResult)
    assert result.perplexity > 1.0

    cfg = FakeCfg()
    m = RunMetrics(tmp_path, mode="baseline")
    m.write_header(cfg, budget_type="backward_passes", budget_value=10)
    m.write_footer(
        best_valid_loss=result.avg_loss,
        best_valid_step=1,
        final_train_loss=result.avg_loss,
        perplexity=result.perplexity,
    )
    m.close()

    lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
    footer = json.loads(lines[-1])
    assert footer["perplexity"] == result.perplexity
    assert footer["perplexity"] == math.exp(result.avg_loss)


# --- TASK-0097: constructor validation ---


class TestRunMetricsValidation:
    """TASK-0097: RunMetrics.__init__ rejects invalid parameter values."""

    def test_init_rejects_invalid_mode(self, tmp_path):
        with pytest.raises(ValueError, match="mode must be"):
            RunMetrics(tmp_path, mode="invalid")

    def test_init_rejects_empty_run_id(self, tmp_path):
        with pytest.raises(ValueError, match="run_id must be a non-empty string"):
            RunMetrics(tmp_path, mode="baseline", run_id="")

    def test_init_accepts_valid_baseline(self, tmp_path):
        m = RunMetrics(tmp_path, mode="baseline")
        assert m.run_id.startswith("baseline_")
        m.close()

    def test_init_accepts_valid_tg_lora(self, tmp_path):
        m = RunMetrics(tmp_path, mode="tg_lora")
        assert m.run_id.startswith("tg_lora_")
        m.close()

    def test_init_accepts_custom_run_id(self, tmp_path):
        m = RunMetrics(tmp_path, mode="baseline", run_id="custom-id")
        assert m.run_id == "custom-id"
        m.close()


def test_record_step_extra_fields_passthrough(tmp_path):
    """Dynamic extra fields (e.g. psa_lt_* per-layer-type metrics) are serialized."""
    m = RunMetrics(tmp_path, mode="tg_lora")
    result = m.record_step(
        step=5,
        cycle=2,
        loss_train=2.0,
        total_backward_passes=15,
        psa_lt_attention_out_amp_mean=1.23,
        psa_lt_attention_out_prior_stability=0.91,
        psa_lt_mlp_amp_mean=0.85,
        psa_lt_mlp_amp_std=0.12,
        custom_metric=42.0,
    )
    m.close()

    assert result["psa_lt_attention_out_amp_mean"] == 1.23
    assert result["psa_lt_attention_out_prior_stability"] == 0.91
    assert result["psa_lt_mlp_amp_mean"] == 0.85
    assert result["custom_metric"] == 42.0

    # Also verify serialized to file
    lines = (tmp_path / "run_metrics.jsonl").read_text().strip().split("\n")
    step_record = json.loads(lines[0])
    assert step_record["psa_lt_attention_out_amp_mean"] == 1.23


class TestAppendModeResumeContinuity:
    """Resume into the same run_dir must NOT truncate the metrics file.

    On-disk analog of the resume-state-loss axis: the per-cycle
    ``run_metrics.jsonl`` — read by the advisor / deposit / break-even gate —
    must survive a fault or periodic resume into the same ``run_dir``, just as
    the 12/12 caller-scoped accumulators (cycle / velocity / controller / psa /
    progressive_freeze) are restored. ``append=True`` opens the file in ``"ab"``
    and carries ``run_id`` + wall-clock forward so the resumed segment continues
    the same run instead of destroying its pre-resume trajectory.
    """

    @staticmethod
    def _records(path) -> list[dict]:
        lines = (path / "run_metrics.jsonl").read_text().splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]

    def test_fresh_construct_truncates_existing_file(self, tmp_path):
        """Default ``append=False`` keeps legacy truncate semantics — a fresh run
        into a populated run_dir starts clean (byte-identical to before this fix)."""
        m = RunMetrics(tmp_path, mode="baseline", run_id="first")
        m.record_step(step=1, loss_train=3.0, total_backward_passes=1)
        m.close()
        assert len(self._records(tmp_path)) == 1

        m2 = RunMetrics(tmp_path, mode="baseline", run_id="second")  # truncate
        m2.record_step(step=9, loss_train=9.0, total_backward_passes=9)
        m2.close()

        records = self._records(tmp_path)
        assert len(records) == 1
        assert records[0]["run_id"] == "second"
        assert records[0]["step"] == 9

    def test_append_preserves_and_extends_records(self, tmp_path):
        """Pre-resume records survive; new records are appended after them —
        nothing is lost on resume."""
        m = RunMetrics(tmp_path, mode="baseline", run_id="run-A")
        m.record_step(step=1, loss_train=3.0, total_backward_passes=1)
        m.record_step(step=2, loss_train=2.5, total_backward_passes=2)
        m.close()
        assert len(self._records(tmp_path)) == 2

        m2 = RunMetrics(tmp_path, mode="baseline", append=True)
        m2.record_step(step=3, loss_train=2.2, total_backward_passes=3)
        m2.close()

        records = self._records(tmp_path)
        # 2 pre-resume records preserved + 1 appended, none lost.
        assert [r["step"] for r in records] == [1, 2, 3]
        assert [r["loss_train"] for r in records] == [3.0, 2.5, 2.2]

    def test_append_reuses_existing_run_id(self, tmp_path):
        """``run_id`` is carried forward so the resumed segment shares the run's
        identity (not a fresh timestamp)."""
        m = RunMetrics(tmp_path, mode="tg_lora", run_id="shared-id")
        m.record_step(step=1, loss_train=3.0, total_backward_passes=1)
        m.close()

        m2 = RunMetrics(tmp_path, mode="tg_lora", append=True)
        assert m2.run_id == "shared-id"
        m2.record_step(step=2, loss_train=2.5, total_backward_passes=2)
        m2.close()

        records = self._records(tmp_path)
        assert {r["run_id"] for r in records} == {"shared-id"}

    def test_append_skips_duplicate_header(self, tmp_path):
        """``write_header`` is a no-op on an appended segment — the original
        header stays authoritative; no second ``run_header`` is interleaved
        mid-file."""
        cfg = FakeCfg()
        m = RunMetrics(tmp_path, mode="baseline", run_id="h-run")
        m.write_header(
            cfg,
            budget_type="backward_passes",
            budget_value=10,
            param_counts={"total": 100, "trainable": 10},
        )
        m.record_step(step=1, loss_train=3.0, total_backward_passes=1)
        m.close()

        m2 = RunMetrics(tmp_path, mode="baseline", append=True)
        m2.write_header(  # must be a no-op
            cfg,
            budget_type="backward_passes",
            budget_value=20,
            param_counts={"total": 200, "trainable": 20},
        )
        m2.record_step(step=2, loss_train=2.5, total_backward_passes=2)
        m2.close()

        records = self._records(tmp_path)
        headers = [r for r in records if r["type"] == "run_header"]
        assert len(headers) == 1  # no duplicate header
        # The surviving header is the ORIGINAL (budget_value 10, not the 20 the
        # no-op'd second call tried to write).
        assert headers[0]["compute_budget"]["budget_value"] == 10

    def test_append_continues_elapsed_monotonic(self, tmp_path):
        """Wall-clock ``elapsed_seconds`` continues from the pre-resume value
        rather than restarting at ~0 (footer ``total_wall_seconds`` likewise
        derives from the same carried-forward ``_start_time``)."""
        m = RunMetrics(tmp_path, mode="baseline", run_id="clk")
        m.record_step(step=1, loss_train=3.0, total_backward_passes=1)
        pre_elapsed = self._records(tmp_path)[-1]["elapsed_seconds"]
        m.close()

        m2 = RunMetrics(tmp_path, mode="baseline", append=True)
        m2.record_step(step=2, loss_train=2.5, total_backward_passes=2)
        m2.close()

        appended = self._records(tmp_path)[-1]
        # Continued, not restarted: an appended elapsed >= last pre-resume
        # elapsed. A restart would read ~0, far below ``pre_elapsed``.
        assert appended["elapsed_seconds"] >= pre_elapsed

    def test_append_on_nonexistent_file_behaves_like_fresh(self, tmp_path):
        """``append=True`` with no existing file degrades to a fresh write — the
        header is emitted and the run starts normally (no silent skip)."""
        fresh_dir = tmp_path / "never_used"
        m = RunMetrics(fresh_dir, mode="baseline", append=True)
        m.write_header(FakeCfg(), budget_type="backward_passes", budget_value=5)
        m.record_step(step=1, loss_train=3.0, total_backward_passes=1)
        m.close()

        records = self._records(fresh_dir)
        assert [r["type"] for r in records] == ["run_header", "step"]  # header kept

