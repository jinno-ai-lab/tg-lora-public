"""Tests for save_checkpoint, save/load_training_state, _sanitize_tensors."""

import logging
import os
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.dynamic_freeze import DynamicFreezeController, DynFreezeState
from src.tg_lora.random_walk_controller import ControllerState
from src.tg_lora.velocity import Velocity
from src.utils.checkpoint import (
    CheckpointSaveError,
    TrainingState,
    _atomic_publish_checkpoint_dir,
    _sanitize_tensors,
    load_training_state,
    save_checkpoint,
    save_training_state,
)


def _mock_model_and_tokenizer(tmp_path, file_count=3, save_pretrained_side_effect=None):
    """Create mock model/tokenizer that write files into *save_dir*."""
    model = MagicMock()

    def _fake_save_pretrained(save_dir):
        if save_pretrained_side_effect is not None:
            save_pretrained_side_effect(save_dir)
            return
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(file_count):
            (d / f"weight_{i}.bin").write_bytes(b"\x00")

    model.save_pretrained = _fake_save_pretrained

    tokenizer = MagicMock()

    def _fake_tok_save(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "tokenizer.json").write_text("{}")

    tokenizer.save_pretrained = _fake_tok_save
    return model, tokenizer


def _lawa_state_sample() -> dict:
    """A representative ``LAWAAverager.state_dict()`` with two snapshots, so the
    checkpoint round-trip exercises the buffer-tensor path (not just scalars)."""
    return {
        "window_size": 3,
        "start_cycle": 1,
        "cycle": 2,
        "recorded_count": 2,
        "buffer": [
            {"lora_A": torch.tensor([0.1, 0.2]), "lora_B": torch.tensor([0.3])},
            {"lora_A": torch.tensor([0.4, 0.5]), "lora_B": torch.tensor([0.6])},
        ],
    }


def _act_regime_state_sample() -> dict:
    """A representative ``ActivationFingerprintTracker.state_dict()`` with a
    populated regime inventory, so the checkpoint round-trip exercises the
    resume-persistent regime surface. The GOAL §4 ``activation_regime_inventory``
    / ``stable_fraction`` must survive resume or the run-end summary reflects
    only post-resume steps (sibling resume-state-loss axis to ``lawa_state``)."""
    return {
        "all_cosines": [0.97, 0.98, 0.96, 0.41, 0.97],
        "cosines": [0.96, 0.41, 0.97],
        "counts": {"stable": 3, "transition": 1, "chaotic": 1},
        "regime": "stable",
    }


def _efficiency_accounting_sample() -> dict:
    """A representative mid-run snapshot of the run-wide efficiency-accounting
    counters (GOAL §5 / P3), exercising the mixed int / float / list / dict
    counter types so the checkpoint round-trip exercises the full surface. These
    must survive resume or the run-end cost report (validation_forwards_total,
    cache hit-rate, subspace-ZO / alpha-line tallies, future-work projection
    mean) reflects only post-resume cycles (sibling resume-state-loss axis)."""
    return {
        "activation_cache_build_count": 2,
        "activation_cache_eligible_count": 40,
        "activation_cache_hit_count": 28,
        "activation_cache_miss_count": 12,
        "pilot_validation_forward_count": 37,
        "post_validation_forward_count": 19,
        "post_extrapolation_eval_count": 14,
        "post_extrapolation_eval_skipped_count": 5,
        "post_extrapolation_eval_skip_reasons": {"below_threshold": 3, "no_velocity": 2},
        "subspace_zo_attempted_steps_total": 8,
        "subspace_zo_accepted_steps_total": 5,
        "subspace_zo_rejected_steps_total": 3,
        "subspace_zo_forward_count_total": 16,
        "subspace_zo_dim1_steps_total": 6,
        "subspace_zo_dim2_steps_total": 2,
        "alpha_line_steps_total": 11,
        "alpha_line_base_recompute_total": 11,
        "alpha_line_v_update_wall_seconds_total": 0.42,
        "alpha_line_alpha_wall_seconds_total": 1.17,
        "future_work_projection_ratios": [0.31, 0.44, 0.52],
        "future_work_internal_pair_count": 9,
    }


def _psa_state_sample() -> dict:
    """A representative ``PSAPrior.state_dict()`` with a populated subspace-prior
    surface, so the checkpoint round-trip exercises the resume-persistent PSA
    state (per-step ``_delta_history`` ring buffer + extracted PC1 ``priors`` that
    drive amplification + ``_prev_priors`` blend anchor + ``_prior_cosines``
    stability series + ``_last_update_step``). These must survive resume or the
    run-end ``layer_delta_analysis`` (GOAL §4) is omitted on a short residual run
    and amplification is silently off until priors re-accumulate (sibling
    resume-state-loss axis to ``act_regime_state`` / ``efficiency_accounting``)."""
    return {
        "delta_history": [
            {"layers.0.lora_A.default.weight": torch.randn(2, 16)},
            {"layers.0.lora_A.default.weight": torch.randn(2, 16)},
        ],
        "priors": {"layers.0.lora_A.default.weight": _unit(torch.randn(32))},
        "prev_priors": {"layers.0.lora_A.default.weight": _unit(torch.randn(32))},
        "prior_cosines": {"layers.0.lora_A.default.weight": [0.91, 0.88]},
        "last_update_step": 10,
    }


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-12)


def _psa_regime_state_sample() -> dict:
    """A representative ``RegimeDetector.state_dict()`` with a populated
    run-wide surface (loss / velocity classification windows + current regime +
    transition count), so the checkpoint round-trip exercises the
    resume-persistent PSA regime state. These must survive resume or the
    per-cycle ``psa_regime_transitions`` (persisted to ``run_metrics.jsonl``)
    resets to 0 (sibling resume-state-loss axis to ``psa_state`` /
    ``act_regime_state``)."""
    return {
        "losses": [2.0, 1.9, 1.85, 1.82, 2.5],
        "velocities": [-0.1, -0.05, -0.03, 0.68],
        "regime": "transition",
        "transition_count": 3,
    }


class TestSaveCheckpointNormal:
    def test_creates_directory(self, tmp_path):
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=2)
        save_dir = tmp_path / "output"
        save_checkpoint(model, tokenizer, save_dir)
        assert save_dir.is_dir()

    def test_directory_contains_files(self, tmp_path):
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=3)
        save_dir = tmp_path / "output"
        save_checkpoint(model, tokenizer, save_dir)
        files = list(save_dir.iterdir())
        assert len(files) >= 1

    def test_creates_parent_directories(self, tmp_path):
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=1)
        save_dir = tmp_path / "deep" / "nested" / "dir"
        save_checkpoint(model, tokenizer, save_dir)
        assert save_dir.is_dir()


class TestSaveCheckpointReadbackVerification:
    def test_no_warning_on_successful_save(self, tmp_path, caplog):
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=3)
        save_dir = tmp_path / "output"
        with caplog.at_level(logging.WARNING, logger="src.utils.checkpoint"):
            save_checkpoint(model, tokenizer, save_dir)
        checkpoint_warnings = [
            r for r in caplog.records if "checkpoint" in r.message.lower()
        ]
        assert checkpoint_warnings == []

    def test_empty_save_pretrained_raises_and_publishes_nothing(self, tmp_path):
        """save_pretrained leaves an empty directory → fail loud, do NOT publish.

        A completed-but-empty save is the one path that still violated the
        atomic-publish contract ("destination never non-loadable"): publishing
        the empty temp over a prior good checkpoint would replace a loadable
        destination with an empty one. It must raise ``CheckpointSaveError``
        (after removing the orphan temp) rather than warn-and-publish.
        """

        def _save_empty(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        model = MagicMock()
        model.save_pretrained = _save_empty
        tokenizer = MagicMock()
        tokenizer.save_pretrained = lambda d: None

        save_dir = tmp_path / "empty_ckpt"
        with pytest.raises(CheckpointSaveError):
            save_checkpoint(model, tokenizer, save_dir)
        # Nothing published, and the orphan empty temp is removed.
        assert not save_dir.exists()
        assert not list(tmp_path.glob("empty_ckpt.tmp.*"))

    def test_missing_save_pretrained_raises_and_publishes_nothing(self, tmp_path):
        """save_pretrained does not create files → fail loud, do NOT publish."""
        model = MagicMock()
        model.save_pretrained = MagicMock()
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()

        save_dir = tmp_path / "missing_ckpt"
        with pytest.raises(CheckpointSaveError):
            save_checkpoint(model, tokenizer, save_dir)
        assert not save_dir.exists()
        assert not list(tmp_path.glob("missing_ckpt.tmp.*"))


class TestSaveCheckpointEmptyDoesNotClobberPrior:
    """The atomic-publish guarantee's last hole: a completed-but-empty save
    must not overwrite a prior loadable checkpoint with a non-loadable one.

    Symmetric in spirit to ``TestAtomicCheckpointDirPublish::
    test_overwrite_fault_restores_prior_checkpoint`` (a mid-swap *fault* leaves
    the prior checkpoint in place): here a save that *completes* but writes
    nothing must likewise leave the prior, still-loadable checkpoint untouched —
    the "resume loads cleanly" guarantee the feedback asks to confirm, at the
    Cat-A fault-injection level (the real SIGINT-mid-save GPU run is Cat-C).
    """

    def test_empty_save_leaves_prior_checkpoint_loadable(self, tmp_path):
        # 1) A prior good checkpoint is on disk.
        save_dir = tmp_path / "best_model"
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=2)
        save_checkpoint(model, tokenizer, save_dir)
        prior_files = sorted(p.name for p in save_dir.iterdir())
        assert prior_files  # non-empty prior

        # 2) A later save attempt whose save_pretrained writes nothing.
        def _save_empty(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        empty_model = MagicMock()
        empty_model.save_pretrained = _save_empty
        empty_tokenizer = MagicMock()
        empty_tokenizer.save_pretrained = lambda d: None

        with pytest.raises(CheckpointSaveError):
            save_checkpoint(empty_model, empty_tokenizer, save_dir)

        # 3) The prior checkpoint survives byte-for-byte — NOT clobbered, NOT
        #    empty, NOT a mix. This is the load-bearing assertion.
        assert save_dir.is_dir()
        assert sorted(p.name for p in save_dir.iterdir()) == prior_files
        # No orphan temp / backup litters the checkpoint directory.
        assert not list(tmp_path.glob("best_model.tmp.*"))
        assert not list(tmp_path.glob("best_model.old.*"))


class TestSaveCheckpointExistingDir:
    def test_works_with_preexisting_directory(self, tmp_path):
        save_dir = tmp_path / "existing"
        save_dir.mkdir()
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=1)
        save_checkpoint(model, tokenizer, save_dir)
        files = list(save_dir.iterdir())
        assert len(files) >= 1


class TestAtomicCheckpointDirPublish:
    """``save_checkpoint`` publishes the LoRA adapter dir atomically — a
    mid-save fault never leaves a torn ``adapter_model.safetensors`` for resume.

    Mirrors ``TestAtomicCheckpointSave`` (the ``torch.save`` sites) and
    ``test_atomic_save.py`` (SIGINT/SystemExit at the helper).
    ``save_pretrained`` → ``safetensors.save_file`` writes the costly weight
    bytes directly to its target with no temp+rename, so ``save_checkpoint``
    stages the full save in a PID-suffixed sibling temp dir and swaps it into
    place only once complete. The largest artifact to lose to a torn write (the
    LoRA weights) thus gets the same crash-atomicity guarantee the
    ``training_state.pt`` writers already have — closing the gap that sat
    outside the ``_atomic_torch_save`` axis.
    """

    @staticmethod
    def _populated_tmp(tmp_path, name="ckpt", files=("adapter_model.safetensors",)):
        tmp = tmp_path / f"{name}.tmp.{os.getpid()}"
        tmp.mkdir(parents=True)
        for f in files:
            (tmp / f).write_bytes(b"\x00" * 8)
        return tmp

    def test_successful_publish_moves_complete_dir_no_temp(self, tmp_path):
        save_dir = tmp_path / "ckpt"
        tmp = self._populated_tmp(tmp_path)
        _atomic_publish_checkpoint_dir(tmp, save_dir)

        assert save_dir.is_dir()
        assert (save_dir / "adapter_model.safetensors").exists()
        # the temp was renamed into place, not copied, so it is gone
        assert not tmp.exists()
        assert not list(tmp_path.glob("ckpt.tmp.*"))

    def test_fresh_publish_fault_creates_no_destination(self, tmp_path, monkeypatch):
        def _boom(_src, _dst):
            raise OSError("simulated mid-swap fault")

        monkeypatch.setattr(os, "replace", _boom)
        save_dir = tmp_path / "fresh"
        tmp = self._populated_tmp(tmp_path, name="fresh")

        with pytest.raises(OSError):
            _atomic_publish_checkpoint_dir(tmp, save_dir)

        # nothing published; the orphan temp holding the new bytes was removed
        assert not save_dir.exists()
        assert not list(tmp_path.glob("fresh.tmp.*"))

    def test_overwrite_fault_restores_prior_checkpoint(self, tmp_path, monkeypatch):
        # Prior complete checkpoint with a distinct (v1) weight file in place.
        save_dir = tmp_path / "best_model"
        save_dir.mkdir()
        (save_dir / "adapter_model.safetensors").write_bytes(b"V1-WEIGHTS")
        tmp = self._populated_tmp(tmp_path, name="best_model")

        calls = {"n": 0}
        orig_replace = os.replace

        def _fault_publish_only(src, dst):
            calls["n"] += 1
            # Fault ONLY the new-weights publish (tmp_dir -> save_dir): the
            # prior-aside move (save_dir -> backup) and the recovery restore
            # (backup -> save_dir) must stay functional so we observe that the
            # prior checkpoint is restored — exactly what would otherwise be lost
            # to a bare save_pretrained truncating the destination mid-dump.
            if Path(src) == tmp:
                raise OSError("simulated mid-swap fault")
            return orig_replace(src, dst)

        monkeypatch.setattr(os, "replace", _fault_publish_only)

        with pytest.raises(OSError):
            _atomic_publish_checkpoint_dir(tmp, save_dir)

        # The prior, still-loadable checkpoint is restored — NOT torn, NOT empty,
        # NOT the half-written new weights. This is the load-bearing assertion.
        assert (save_dir / "adapter_model.safetensors").read_bytes() == b"V1-WEIGHTS"
        # orphan temp and the backup are both cleaned
        assert not list(tmp_path.glob("best_model.tmp.*"))
        assert not list(tmp_path.glob("best_model.old.*"))

    def test_successful_overwrite_replaces_prior_and_cleans_backup(self, tmp_path):
        save_dir = tmp_path / "best_model"
        save_dir.mkdir()
        (save_dir / "adapter_model.safetensors").write_bytes(b"OLD")
        tmp = self._populated_tmp(tmp_path, name="best_model")

        _atomic_publish_checkpoint_dir(tmp, save_dir)

        # new weights land, prior backup removed
        assert (save_dir / "adapter_model.safetensors").read_bytes() == b"\x00" * 8
        assert not list(tmp_path.glob("best_model.old.*"))


class TestSaveCheckpointAtomicEndToEnd:
    """``save_checkpoint`` end-to-end: an interrupt mid-``save_pretrained`` never
    publishes a destination and never leaves an orphan temp."""

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, OSError])
    def test_mid_save_interrupt_publishes_nothing_and_cleans_temp(
        self, tmp_path, monkeypatch, interrupt
    ):
        save_dir = tmp_path / "cycle_ckpt"

        def _interrupt_mid_save(d):
            # Write a partial file then abort — mimics safetensors truncating
            # adapter_model.safetensors when a SIGINT lands mid-dump.
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "adapter_model.safetensors").write_bytes(b"\x80PARTIAL")
            raise interrupt("simulated interrupt mid-save")

        model = MagicMock()
        model.save_pretrained = _interrupt_mid_save
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()

        with pytest.raises(interrupt):
            save_checkpoint(model, tokenizer, save_dir)

        # No destination published, and the orphan temp (holding the torn bytes)
        # is removed so resume never sees a corrupt adapter_model.safetensors.
        assert not save_dir.exists()
        assert not list(tmp_path.glob("cycle_ckpt.tmp.*"))

    def test_successful_save_publishes_dir_and_leaves_no_temp(self, tmp_path):
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=2)
        save_dir = tmp_path / "ok_ckpt"

        save_checkpoint(model, tokenizer, save_dir)

        assert save_dir.is_dir()
        assert len(list(save_dir.iterdir())) >= 1
        assert not list(tmp_path.glob("ok_ckpt.tmp.*"))


class TestTrainingStateRoundtrip:
    """TC-209-01: save_training_state→load_training_state round-trip."""

    def _make_state(self) -> TrainingState:
        cs = CycleState(cycle=5, full_backward_passes=30, best_loss=2.5)
        ctrl = ControllerState(
            K=3,
            N=5,
            alpha=0.3,
            beta=0.8,
            lr=5e-4,
            active_layer_strategy="last_25_percent_plus_random_2",
            relative_update_cap=0.005,
        )
        vel = Velocity(max_history=100)
        vel.update({"lora_A": torch.tensor([1.0, 2.0])}, beta=0.8)
        vel.update({"lora_A": torch.tensor([1.0, 2.0])}, beta=0.8)
        dt = DeltaTracker(max_history=50)
        dt._history = [{"w": torch.tensor([0.1, 0.2])}]
        dt._norm_history = [0.224]
        return TrainingState(
            cycle_state=cs,
            controller_state=ctrl,
            velocity=vel,
            delta_tracker=dt,
            cycle_offset=3,
            train_batch_position=17,
            accepted_valid_history=[2.9, 2.5, 2.1],
            best_full_eval_loss=1.83,
            best_full_eval_perplexity=6.23,
            # Mid-production checkpoint: warmup already released, with a nonzero
            # consecutive-cosine count carried over from the release moment.
            warmup_released=True,
            warmup_cos_consecutive=3,
            # Mid-production LAWA window: two LoRA snapshots already recorded so
            # ``is_ready`` is True on resume and the LAWA comparison is NOT
            # silently skipped post-resume (sibling resume-state-loss axis).
            lawa_state=_lawa_state_sample(),
            # Mid-production best-LAWA-loss headline (the run-wide minimum of the
            # §3.3 mandatory-baseline comparison) so resume does not restart it at
            # inf and report a post-resume-only headline (sibling resume-state-loss).
            best_lawa_loss=1.17,
            # Mid-run linearity-budget state: two of the six target steps
            # (250/500/.../1500) already fired, so resume does not re-fire them
            # (redundant evals + duplicate is_step_aligned_full_eval records
            # corrupting the vs-baseline comparison dataset). Sibling resume-state-loss.
            triggered_target_steps=[250, 500],
            # Mid-production activation-regime inventory (GOAL §4): the tracker
            # has classified 5 steps (3 stable / 1 transition / 1 chaotic) so
            # resume does not rebuild it empty and the run-end summary's
            # activation_regime_inventory / stable_fraction reflect the full run,
            # not post-resume only (sibling resume-state-loss axis).
            act_regime_state=_act_regime_state_sample(),
            # Mid-run efficiency-accounting counters (GOAL §5 / P3): the run-wide
            # tallies have accumulated (cache hit-rate, validation_forwards_total,
            # subspace-ZO / alpha-line step counts, future-work projection ratios)
            # so resume does not rebuild them at zero and the run-end cost report
            # reflects the full run, not post-resume only (sibling resume-state-loss).
            efficiency_accounting=_efficiency_accounting_sample(),
            # Mid-production PSA subspace-prior accumulation (GOAL §1.5 / §3.3):
            # the prior has recorded 2 deltas and extracted priors once so resume
            # does not rebuild it empty (amplification silently off + the run-end
            # layer_delta_analysis omitted on a short residual run). Sibling
            # resume-state-loss axis.
            psa_state=_psa_state_sample(),
            # Mid-run PSA regime detector accumulation (GOAL §1.5 / §3.3): the
            # detector has classified 3 transitions so resume does not rebuild it
            # fresh and the per-cycle ``psa_regime_transitions`` (persisted to
            # ``run_metrics.jsonl``) does not reset to 0. Sibling resume-state-loss
            # axis to ``psa_state`` / ``act_regime_state``.
            psa_regime_state=_psa_regime_state_sample(),
            # Mid-run async-cache-swap state (GOAL §3.3 cache-ablation): both
            # loaders were swapped to the cached dataset at cycle 4 so resume
            # does not reset them to None and the run-end summary's
            # async_cache_swap_cycle_valid_quick/full fields are not silently
            # dropped (sibling resume-state-loss axis).
            swap_cycle_vq=4,
            swap_cycle_vf=4,
            # Mid-run progressive-freeze cumulative set (GOAL §1.6 / design §4.1):
            # layers 3 then 2 froze across earlier cycles, so resume does not
            # rebuild the set empty and (a) leave those layers silently
            # re-trainable — undoing the cost reduction that defines Progressive
            # Freezing — and (b) report only post-fault freezes in the run
            # footer's ``frozen_layers`` (the Tier-2 §4 order-verdict arm
            # provenance). Sibling resume-state-loss axis.
            progressive_freeze_state={
                "frozen_layers": [2, 3],
                "last_frozen_layer": 2,
            },
        )

    def test_roundtrip_preserves_values(self, tmp_path):
        state = self._make_state()
        path = tmp_path / "state.pt"
        save_training_state(state, path)
        loaded = load_training_state(path)

        assert loaded.cycle_offset == 3
        assert loaded.train_batch_position == 17
        assert loaded.accepted_valid_history == [2.9, 2.5, 2.1]
        assert loaded.cycle_state.cycle == 5
        assert loaded.cycle_state.best_loss == 2.5
        # Best-full-eval trackers must survive resume so the resumed save-best
        # gate compares against the genuine pre-fault best, not inf.
        assert loaded.best_full_eval_loss == 1.83
        assert loaded.best_full_eval_perplexity == 6.23
        # Warmup phase must survive resume so a mid-production checkpoint does
        # not silently drop back into the pilot-only warmup phase.
        assert loaded.warmup_released is True
        assert loaded.warmup_cos_consecutive == 3
        # LAWA window must survive resume: the snapshot buffer tensors round-trip
        # so a resumed averager is ``is_ready`` and does not silently skip the
        # LAWA comparison / LAWA-averaged JSON eval.
        assert loaded.lawa_state is not None
        assert loaded.lawa_state["recorded_count"] == 2
        assert len(loaded.lawa_state["buffer"]) == 2
        assert torch.equal(
            loaded.lawa_state["buffer"][1]["lora_A"], torch.tensor([0.4, 0.5])
        )
        # Best-LAWA-loss headline must survive resume so the run-end summary
        # reflects the genuine run-wide minimum, not an inf-restarted value.
        assert loaded.best_lawa_loss == 1.17
        # Linearity-budget target-step set must survive resume so a resumed run
        # does not re-fire already-crossed targets (redundant full evals +
        # duplicate is_step_aligned_full_eval records corrupting the
        # vs-baseline comparison dataset). Serialized sorted; round-trips as a
        # list the trainer converts back to a set.
        assert loaded.triggered_target_steps == [250, 500]
        # Activation-regime inventory (GOAL §4) must survive resume so the
        # run-end summary's activation_regime_inventory / stable_fraction reflect
        # the full run, not post-resume only. Round-trips as a plain dict.
        assert loaded.act_regime_state is not None
        assert loaded.act_regime_state["counts"] == {
            "stable": 3, "transition": 1, "chaotic": 1
        }
        assert loaded.act_regime_state["all_cosines"] == [0.97, 0.98, 0.96, 0.41, 0.97]
        assert loaded.act_regime_state["regime"] == "stable"
        # Efficiency-accounting counters (GOAL §5 / P3) must survive resume so
        # the run-end cost report (cache hit-rate, validation_forwards_total,
        # subspace-ZO / alpha-line tallies, future-work projection mean) reflects
        # the full run, not post-resume only. Round-trips as a plain dict.
        assert loaded.efficiency_accounting is not None
        assert loaded.efficiency_accounting == _efficiency_accounting_sample()
        # PSA subspace-prior accumulation (GOAL §1.5 / §3.3) must survive resume
        # so the run-end layer_delta_analysis (GOAL §4) is not omitted on a short
        # residual run and amplification is not silently off. Tensors round-trip
        # exactly; compare against the saved state (the sample is random).
        assert loaded.psa_state is not None
        assert loaded.psa_state["last_update_step"] == 10
        assert loaded.psa_state["prior_cosines"] == state.psa_state["prior_cosines"]
        _lora_name = "layers.0.lora_A.default.weight"
        assert torch.equal(
            loaded.psa_state["priors"][_lora_name],
            state.psa_state["priors"][_lora_name],
        )
        assert torch.equal(
            loaded.psa_state["delta_history"][1][_lora_name],
            state.psa_state["delta_history"][1][_lora_name],
        )
        # PSA regime detector accumulation (GOAL §1.5 / §3.3) must survive resume
        # so the per-cycle ``psa_regime_transitions`` (persisted to
        # ``run_metrics.jsonl``) does not reset to 0 (sibling resume-state-loss).
        assert loaded.psa_regime_state is not None
        assert loaded.psa_regime_state["transition_count"] == 3
        assert loaded.psa_regime_state["regime"] == "transition"
        assert loaded.psa_regime_state["losses"] == state.psa_regime_state["losses"]
        assert loaded.psa_regime_state["velocities"] == state.psa_regime_state[
            "velocities"
        ]
        # Async-cache-swap completion cycles (GOAL §3.3) must survive resume so
        # the run-end summary's async_cache_swap_cycle_valid_quick/full fields
        # are not silently dropped after a fault/periodic resume. Round-trip as
        # plain ints (caller-scoped scalars, None-safe).
        assert loaded.swap_cycle_vq == 4
        assert loaded.swap_cycle_vf == 4
        # Progressive-freeze cumulative frozen-layer set must survive resume so
        # the resumed controller re-applies requires_grad on the frozen layers
        # (safetensors does not carry it) and the run footer's frozen_layers
        # reflects the full run, not post-fault only (sibling resume-state-loss).
        assert loaded.progressive_freeze_state == {
            "frozen_layers": [2, 3],
            "last_frozen_layer": 2,
        }
        assert loaded.controller_state.K == 3
        assert loaded.controller_state.alpha == 0.3
        assert loaded.velocity._state is not None
        assert torch.allclose(
            loaded.velocity._state["lora_A"], torch.tensor([1.0, 2.0])
        )
        assert loaded.velocity.short_state is not None
        assert loaded.velocity.long_state is not None
        assert loaded.velocity.update_count == state.velocity.update_count
        assert (
            loaded.velocity.predicted_consistency()
            == state.velocity.predicted_consistency()
        )
        assert loaded.delta_tracker.norm_history == state.delta_tracker.norm_history

    def test_legacy_checkpoint_without_best_full_eval_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``best_full_eval_loss``/``_perplexity``;
        load must not break and must read as the safe 'no prior best' defaults
        (inf/None) so the resumed save-best gate never sees a fabricated low."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip both keys to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("best_full_eval_loss", None)
        blob.pop("best_full_eval_perplexity", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.best_full_eval_loss == float("inf")
        assert loaded.best_full_eval_perplexity is None

    def test_legacy_checkpoint_without_warmup_phase_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``warmup_released``/``warmup_cos_consecutive``;
        load must not break and must read as the safe 'not yet released' defaults
        (False/0). False is the only sane legacy reading: a True default would
        skip warmup on a checkpoint that was genuinely warming up, while False
        merely re-runs a brief warmup on an old mid-production checkpoint —
        backward-compatible with the pre-fix behavior."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip both keys to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("warmup_released", None)
        blob.pop("warmup_cos_consecutive", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.warmup_released is False
        assert loaded.warmup_cos_consecutive == 0

    def test_legacy_checkpoint_without_lawa_state_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``lawa_state``; load must not break and
        must read as the safe 'no prior window' default (None). None is the only
        sane legacy reading: the resume path treats a missing window as 'start
        fresh' (the pre-fix behavior), not a fabricated non-empty window."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("lawa_state", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.lawa_state is None

    def test_legacy_checkpoint_without_best_lawa_loss_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``best_lawa_loss``; load must not break and
        must read as the safe 'no prior best' default (inf). inf is the only sane
        legacy reading: the first post-resume LAWA comparison establishes a new
        run-wide minimum rather than comparing against a fabricated low — the
        pre-fix behavior (headline recomputed from post-resume cycles). Mirrors
        the ``best_full_eval_loss`` legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("best_lawa_loss", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.best_lawa_loss == float("inf")

    def test_legacy_checkpoint_without_triggered_target_steps_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``triggered_target_steps``; load must not
        break and must read as the safe 'no target yet fired' default (None).
        None is the only sane legacy reading: the resume path converts it to an
        empty set, so the loop re-fires targets from the resumed equivalent-step
        count (the pre-fix behavior) rather than fabricating an already-fired
        set. Mirrors the ``accepted_valid_history`` / ``best_full_eval_*``
        legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("triggered_target_steps", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.triggered_target_steps is None

    def test_legacy_checkpoint_without_act_regime_state_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``act_regime_state``; load must not break
        and must read as the safe 'no prior inventory' default (None). None is the
        only sane legacy reading: the resume path treats a missing inventory as
        'start empty' (the pre-fix behavior), not a fabricated non-empty one.
        Also covers an ``activation_regime_enabled: false`` run, which never
        serialized the tracker. Mirrors the ``lawa_state`` legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("act_regime_state", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.act_regime_state is None

    def test_legacy_checkpoint_without_efficiency_accounting_loads_clean(
        self, tmp_path
    ):
        """A pre-fix checkpoint omits ``efficiency_accounting``; load must not
        break and must read as the safe 'all counters at zero/empty init'
        default (None). None is the only sane legacy reading: the resume path
        treats a missing accounting bag as 'start every counter fresh' (the
        pre-fix behavior — no fabricated tallies), not a fabricated non-empty
        bag. Mirrors the lawa_state / act_regime_state / triggered_target_steps
        legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("efficiency_accounting", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.efficiency_accounting is None

    def test_legacy_checkpoint_without_psa_state_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``psa_state``; load must not break and must
        read as the safe 'no prior' default (None). None is the only sane legacy
        reading: the resume path treats a missing prior as 'start fresh' (the
        pre-fix behavior, no fabricated priors), not a fabricated non-empty one.
        Also covers an ``enable_psa: false`` run, which never serialized the
        prior. Mirrors the ``act_regime_state`` / ``efficiency_accounting`` /
        ``lawa_state`` legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("psa_state", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.psa_state is None

    def test_legacy_checkpoint_without_psa_regime_state_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``psa_regime_state``; load must not break
        and must read as the safe 'start fresh' default (None). None is the only
        sane legacy reading: the resume path treats a missing regime state as a
        fresh detector (the pre-fix behavior, no fabricated transitions), not a
        fabricated non-empty one. Also covers an ``enable_psa: false`` run, which
        never serialized the detector. Mirrors the ``psa_state`` /
        ``act_regime_state`` legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("psa_regime_state", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.psa_regime_state is None

    def test_legacy_checkpoint_without_swap_cycle_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``swap_cycle_vq``/``swap_cycle_vf``; load
        must not break and must read as the safe 'no swap yet' default (None).
        None is the only sane legacy reading: the resume path treats a missing
        cycle as 'the async cache swap had not fired' (the pre-fix behavior, no
        fabricated cycle), and the run-end summary simply omits the
        async_cache_swap_cycle_valid_quick/full fields. Also covers an
        async-cache-disabled run, which never sets either cycle. Mirrors the
        psa_state / act_regime_state legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy.pt"
        save_training_state(state, path)
        # Strip both keys to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("swap_cycle_vq", None)
        blob.pop("swap_cycle_vf", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.swap_cycle_vq is None
        assert loaded.swap_cycle_vf is None

    def test_legacy_checkpoint_without_progressive_freeze_state_loads_clean(
        self, tmp_path
    ):
        """A pre-fix checkpoint omits ``progressive_freeze_state``; load must not
        break and must read as the safe 'no frozen set' default (None).

        None is the only sane legacy reading: the resume path treats a missing
        frozen set as 'start empty' (the pre-fix behavior, no fabricated freezes)
        and skips the ``refreeze_loaded_layers`` step. Also covers a
        progressive-freeze-disabled run (every committed config sets
        ``progressive_freeze_enabled: false``), which never records a frozen set.
        Mirrors the psa_state / act_regime_state / swap_cycle legacy tolerance."""
        state = self._make_state()
        path = tmp_path / "legacy_pf.pt"
        save_training_state(state, path)
        # Strip the key to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob.pop("progressive_freeze_state", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.progressive_freeze_state is None
    """The §4 release-cooldown map (``released_at``) must survive a real
    ``save_training_state``→``load_training_state`` round-trip.

    The controller-level fix (``5a8bb7f``) added ``released_at`` to
    ``DynFreezeState`` and made ``DynamicFreezeController.load_state_dict``
    tolerant of its absence — but the checkpoint plumbing here never serialized
    it, so a run that checkpoints while a layer is mid-cooldown and resumes
    received ``released_at={}`` and §3 silently re-froze the just-released layer
    on its stale ``0.0`` r_A history: the §4 reversible release undone on resume.
    These pin both the data round-trip and the resumed *behavior* (the
    controller's own ``released_at`` unit test only checks the dict, never that a
    resumed ``decide_freeze`` actually respects it).
    """

    @staticmethod
    def _state_with_released_at() -> TrainingState:
        cs = CycleState(cycle=11, full_backward_passes=30, best_loss=2.5)
        ctrl = ControllerState(
            K=3,
            N=5,
            alpha=0.3,
            beta=0.8,
            lr=5e-4,
            active_layer_strategy="last_25_percent_plus_random_2",
            relative_update_cap=0.005,
        )
        # Block [5,4] after §4 released L3 at cycle 10; L3 is one cycle into its
        # window=4 cooldown. L3's restored r_A history is stale frozen-period 0.0.
        dyn = DynFreezeState(
            frozen_layer_indices=[5, 4],
            r_A_history={3: [0.0, 0.0, 0.0, 0.0]},
            frozen_since_cycle=10,
            released_at={3: 10},
        )
        return TrainingState(
            cycle_state=cs,
            controller_state=ctrl,
            velocity=Velocity(max_history=100),
            delta_tracker=DeltaTracker(max_history=50),
            dynfreeze_state=dyn,
        )

    def test_released_at_survives_checkpoint_roundtrip(self, tmp_path):
        """Data-level: ``released_at`` round-trips through the real checkpoint."""
        state = self._state_with_released_at()
        path = tmp_path / "state.pt"
        save_training_state(state, path)
        loaded = load_training_state(path)

        assert loaded.dynfreeze_state is not None
        assert loaded.dynfreeze_state.released_at == {3: 10}
        assert loaded.dynfreeze_state.frozen_layer_indices == [5, 4]

    def test_resumed_controller_honors_release_cooldown(self, tmp_path):
        """Behavioral: a controller rebuilt from the *loaded* state must NOT
        re-freeze a layer still in §4 cooldown, even though its r_A history reads
        quiet — and it MAY re-freeze once the cooldown expires (so the fix does
        not over-hold on resume either)."""
        path = tmp_path / "state.pt"
        save_training_state(self._state_with_released_at(), path)
        loaded = load_training_state(path)

        dfc = DynamicFreezeController(
            tau=0.02, window=4, all_layer_indices=list(range(6))
        )
        dfc.load_state_dict(loaded.dynfreeze_state)
        # Everything upstream of the released L3 is genuinely noisy, so the ONLY
        # re-freeze candidate is L3 (whose restored history is stale-quiet).
        for li in (2, 1, 0):
            dfc._r_A_history[li] = deque([0.05] * dfc._window, maxlen=dfc._window)

        # Mid-cooldown (11 - 10 = 1 < window 4): must NOT re-freeze the released
        # layer. Pre-fix released_at was {} here, so this returned [3].
        assert dfc.decide_freeze(11) == [], (
            "resumed controller re-froze released L3 mid-cooldown — "
            "§4 release undone on resume"
        )
        # Cooldown expired (14 - 10 = 4, not < 4): reversibility holds — L3 may
        # re-freeze. Pins the false-positive side so the fix cannot over-correct.
        assert dfc.decide_freeze(14) == [3]

    def test_legacy_checkpoint_without_released_at_loads_clean(self, tmp_path):
        """A pre-fix checkpoint omits ``released_at``; load must not break and
        must read as 'no active cooldown' (empty map)."""
        path = tmp_path / "legacy.pt"
        save_training_state(self._state_with_released_at(), path)
        # Strip the field to simulate a pre-fix checkpoint blob.
        blob = torch.load(path, weights_only=False)
        blob["dynfreeze_state"].pop("released_at", None)
        torch.save(blob, path)

        loaded = load_training_state(path)
        assert loaded.dynfreeze_state is not None
        assert loaded.dynfreeze_state.released_at == {}


class TestSanitizeTensors:
    """TC-209-02: _sanitize_tensors replaces NaN/Inf with zeros."""

    def test_sanitize_replaces_nan_and_inf(self):
        d = {
            "a": torch.tensor([1.0, float("nan"), 3.0]),
            "b": torch.tensor([float("inf"), 2.0, float("-inf")]),
            "c": torch.tensor([1.0, 2.0, 3.0]),
        }
        _sanitize_tensors(d, "test")
        assert torch.isfinite(d["a"]).all()
        assert torch.isfinite(d["b"]).all()
        assert d["c"].equal(torch.tensor([1.0, 2.0, 3.0]))

    def test_sanitize_logs_warning(self, caplog):
        d = {"x": torch.tensor([float("nan")])}
        with caplog.at_level(logging.WARNING, logger="src.utils.checkpoint"):
            _sanitize_tensors(d, "my_label")
        assert any("my_label" in r.message for r in caplog.records)


def _minimal_training_state() -> TrainingState:
    """A real, loadable TrainingState with only the required fields populated.

    The resume-state round-trip tests use the comprehensive fixture; the
    atomicity tests below only need an object ``save_training_state`` will
    actually serialize and reload, so a minimal state is enough.
    """
    return TrainingState(
        cycle_state=CycleState(cycle=2, full_backward_passes=10, best_loss=2.0),
        controller_state=ControllerState(
            K=3,
            N=5,
            alpha=0.3,
            beta=0.8,
            lr=5e-4,
            active_layer_strategy="last_25_percent_plus_random_2",
            relative_update_cap=0.005,
        ),
        velocity=Velocity(max_history=10),
        delta_tracker=DeltaTracker(max_history=10),
    )


class TestAtomicCheckpointSave:
    """training_state.pt is written atomically — a mid-commit fault never
    leaves a torn destination.

    ``os.replace`` is the sole publish point (the temp file is the only thing
    that ever holds the in-progress bytes). Monkeypatching ``os.replace`` to
    raise simulates a fault exactly at the commit boundary and locks the
    corruption-prevention contract at the behavior level:

    - a fresh save that faults publishes NO destination file,
    - a faulting overwrite leaves the prior, still-loadable checkpoint intact,
    - the orphaned temp is cleaned up either way.

    A regression to a bare ``torch.save(blob, path)`` would truncate the
    destination during serialization; the prior-checkpoint-intact check below
    would then fail (the prior file would be torn), so this test guards the
    atomic-write structure against that regression.
    """

    def test_successful_save_publishes_and_leaves_no_temp(self, tmp_path):
        state = _minimal_training_state()
        path = tmp_path / "state.pt"
        save_training_state(state, path)

        assert path.exists()
        # content round-trips (the published file is valid, not torn)
        assert load_training_state(path).cycle_state.cycle == 2
        # no PID-suffixed temp litters the directory on success
        assert not list(tmp_path.glob("state.pt.tmp.*"))

    def test_fresh_save_fault_creates_no_destination(self, tmp_path, monkeypatch):
        def _boom(src, dst):
            raise OSError("simulated mid-commit fault")

        monkeypatch.setattr(os, "replace", _boom)

        with pytest.raises(OSError):
            save_training_state(_minimal_training_state(), tmp_path / "state.pt")

        # no partial destination was published...
        assert not (tmp_path / "state.pt").exists()
        # ...and the orphaned temp was cleaned up
        assert not list(tmp_path.glob("state.pt.tmp.*"))

    def test_prior_checkpoint_survives_faulting_overwrite(self, tmp_path, monkeypatch):
        # Save a valid v1 checkpoint (real os.replace), then fault at the commit
        # boundary while attempting to overwrite it with a different v2 state.
        path = tmp_path / "state.pt"
        save_training_state(_minimal_training_state(), path)  # cycle == 2
        assert path.exists()

        def _boom(src, dst):
            raise OSError("simulated mid-commit fault")

        monkeypatch.setattr(os, "replace", _boom)

        v2 = _minimal_training_state()
        v2.cycle_state = CycleState(cycle=9, full_backward_passes=99, best_loss=0.5)
        with pytest.raises(OSError):
            save_training_state(v2, path)

        # The prior, still-loadable checkpoint is intact with the OLD value — the
        # torn-write (cycle 9) state was never published. A regressed bare
        # torch.save(path) would have truncated it and this would reload as cycle 9
        # or fail to load entirely.
        assert path.exists()
        assert load_training_state(path).cycle_state.cycle == 2
        assert not list(tmp_path.glob("state.pt.tmp.*"))
