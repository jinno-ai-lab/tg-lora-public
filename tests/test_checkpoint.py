"""Tests for save_checkpoint, save/load_training_state, _sanitize_tensors."""

import logging
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import torch

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.dynamic_freeze import DynamicFreezeController, DynFreezeState
from src.tg_lora.random_walk_controller import ControllerState
from src.tg_lora.velocity import Velocity
from src.utils.checkpoint import (
    TrainingState,
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

    def test_warning_when_directory_empty(self, tmp_path, caplog):
        """save_pretrained leaves an empty directory → warning logged."""

        def _save_empty(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        model = MagicMock()
        model.save_pretrained = _save_empty
        tokenizer = MagicMock()
        tokenizer.save_pretrained = lambda d: None

        save_dir = tmp_path / "empty_ckpt"
        with caplog.at_level(logging.WARNING, logger="src.utils.checkpoint"):
            save_checkpoint(model, tokenizer, save_dir)
        assert any("checkpoint" in r.message.lower() for r in caplog.records)

    def test_warning_when_directory_missing(self, tmp_path, caplog):
        """save_pretrained does not create directory → warning logged."""
        model = MagicMock()
        model.save_pretrained = MagicMock()
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()

        save_dir = tmp_path / "missing_ckpt"
        with caplog.at_level(logging.WARNING, logger="src.utils.checkpoint"):
            save_checkpoint(model, tokenizer, save_dir)
        assert any("checkpoint" in r.message.lower() for r in caplog.records)


class TestSaveCheckpointExistingDir:
    def test_works_with_preexisting_directory(self, tmp_path):
        save_dir = tmp_path / "existing"
        save_dir.mkdir()
        model, tokenizer = _mock_model_and_tokenizer(tmp_path, file_count=1)
        save_checkpoint(model, tokenizer, save_dir)
        files = list(save_dir.iterdir())
        assert len(files) >= 1


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


class TestDynFreezeStateRoundtrip:
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
