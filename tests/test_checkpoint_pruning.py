"""E2E test: keep_last_checkpoints bounds on-disk checkpoint growth.

Closes the M10.3 disk-death gap: the TG-LoRA periodic-save path writes a fresh
``checkpoint-cycle-<N>`` dir every ``save_every_cycles`` and never removes old
ones. With the M10 baseline saving every cycle for 120 cycles that is 120 dirs
of unbounded accumulation — the exact incident class this guard prevents.

With ``keep_last`` set, older ``checkpoint-cycle-*`` dirs must be *truly*
removed and the total checkpoint footprint must stay bounded across many saves,
not grow linearly with the cycle count.
"""

import ast
import shutil
from collections import namedtuple
from pathlib import Path

from omegaconf import OmegaConf

from src.training.config_schema import load_validate_and_build_config
from src.utils.checkpoint import (
    prune_checkpoint_cycles,
    prune_checkpoint_cycles_from_cfg,
    prune_step_checkpoints,
    prune_step_checkpoints_from_cfg,
    prune_trajectory_delta_artifacts,
    prune_trajectory_delta_artifacts_from_cfg,
    save_periodic_cycle_checkpoint,
)

CHECKPOINT_DIR_PREFIX = "checkpoint-cycle-"
PAYLOAD_BYTES = 1024 * 1024  # 1 MiB per checkpoint dir

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


def _save_checkpoint_dir(run_dir, cycle, payload_bytes=PAYLOAD_BYTES):
    """Stand-in for save_checkpoint: create the dir + a fixed-size payload file."""
    ckpt_dir = run_dir / f"{CHECKPOINT_DIR_PREFIX}{cycle}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"\x00" * payload_bytes)
    return ckpt_dir


def _checkpoint_dirs(run_dir):
    return sorted(
        (
            p
            for p in run_dir.iterdir()
            if p.is_dir() and p.name.startswith(CHECKPOINT_DIR_PREFIX)
        ),
        key=lambda p: p.name,
    )


def _cycle_of(path):
    return int(path.name[len(CHECKPOINT_DIR_PREFIX):])


def _checkpoint_total_bytes(run_dir):
    total = 0
    for d in _checkpoint_dirs(run_dir):
        for f in d.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


class TestKeepLastBoundsAccumulation:
    """With keep_last set, accumulation is bounded across many saves."""

    def test_older_dirs_truly_removed_after_many_saves(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3
        total_saves = 40

        for cycle in range(1, total_saves + 1):
            _save_checkpoint_dir(run_dir, cycle)
            prune_checkpoint_cycles(run_dir, keep_last=keep_last)

        remaining = _checkpoint_dirs(run_dir)
        assert [_cycle_of(p) for p in remaining] == [38, 39, 40]

        # Older dirs are truly gone, not hidden/renamed.
        for cycle in range(1, 38):
            assert not (run_dir / f"{CHECKPOINT_DIR_PREFIX}{cycle}").exists()

    def test_count_never_exceeds_keep_last_during_loop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3

        for cycle in range(1, 41):
            _save_checkpoint_dir(run_dir, cycle)
            prune_checkpoint_cycles(run_dir, keep_last=keep_last)
            # After every save+prune the live count is bounded.
            assert len(_checkpoint_dirs(run_dir)) <= keep_last

    def test_reclaimed_disk_stays_bounded_not_linear(self, tmp_path):
        """Footprint stays ~keep_last x payload, not saves x payload."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3
        peak_bytes = 0

        for cycle in range(1, 41):
            _save_checkpoint_dir(run_dir, cycle)
            prune_checkpoint_cycles(run_dir, keep_last=keep_last)
            peak_bytes = max(peak_bytes, _checkpoint_total_bytes(run_dir))

        final_bytes = _checkpoint_total_bytes(run_dir)
        # End state: at most keep_last payloads on disk.
        assert final_bytes <= keep_last * PAYLOAD_BYTES
        # Peak across the whole run is bounded by one save's worth of slack,
        # never the linear 40 payloads that the unguarded path would reach.
        assert peak_bytes <= (keep_last + 1) * PAYLOAD_BYTES
        assert peak_bytes < 40 * PAYLOAD_BYTES


class TestPruningContract:
    """Edge behaviour of the count bound."""

    def test_defaults_off_preserves_unbounded_behavior(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)

        removed = prune_checkpoint_cycles(run_dir)  # keep_last=0, min_free=0

        assert removed == []
        assert len(_checkpoint_dirs(run_dir)) == 5

    def test_keep_last_larger_than_count_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 3):
            _save_checkpoint_dir(run_dir, cycle)

        removed = prune_checkpoint_cycles(run_dir, keep_last=10)

        assert removed == []
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [1, 2]

    def test_no_dirs_is_safe_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        assert prune_checkpoint_cycles(run_dir, keep_last=3) == []

    def test_non_checkpoint_dirs_are_never_touched(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_dir = run_dir / "best_model"
        keep_dir.mkdir()
        (keep_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)

        prune_checkpoint_cycles(run_dir, keep_last=2)

        assert keep_dir.exists()
        assert (keep_dir / "adapter_model.safetensors").exists()


class TestMinFreeDiskFloor:
    """min_free_disk_gb prunes oldest-first until the floor is met."""

    def test_floor_met_stops_pruning(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)

        # Filesystem reports 10 GiB free; floor is 2 GiB -> already satisfied.
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda p: _DiskUsage(total=100, used=0, free=10 * 1024 ** 3),
        )

        removed = prune_checkpoint_cycles(
            run_dir, keep_last=0, min_free_disk_gb=2.0
        )

        assert removed == []
        assert len(_checkpoint_dirs(run_dir)) == 5

    def test_low_disk_prunes_oldest_until_floor_met(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)

        # Each dir is 1 MiB. Report free space that starts at 1 MiB and rises
        # 1 MiB per check, so a 3 MiB floor is satisfied once free >= 3 MiB.
        calls = {"n": 0}

        def fake_du(p):
            calls["n"] += 1
            free = (1 + (calls["n"] - 1)) * 1024 * 1024  # 1,2,3,... MiB
            return _DiskUsage(total=100 * 1024 ** 3, used=0, free=free)

        monkeypatch.setattr(shutil, "disk_usage", fake_du)

        removed = prune_checkpoint_cycles(
            run_dir, keep_last=0, min_free_disk_gb=3.0 / 1024  # 3 MiB floor
        )

        # Floor (3 MiB) is met at the 3rd check, so the two oldest dirs are
        # removed before the loop stops.
        assert [_cycle_of(p) for p in removed] == [1, 2]
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [3, 4, 5]

    def test_floor_never_deletes_the_last_checkpoint(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 4):
            _save_checkpoint_dir(run_dir, cycle)

        # Disk reports 0 free forever; floor unreachably high.
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda p: _DiskUsage(total=100, used=100, free=0),
        )

        removed = prune_checkpoint_cycles(
            run_dir, keep_last=0, min_free_disk_gb=1000.0
        )

        # Prunes oldest-first but always leaves the single newest checkpoint.
        assert [_cycle_of(p) for p in removed] == [1, 2]
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [3]

    def test_count_bound_and_disk_floor_compose(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 9):  # 8 dirs
            _save_checkpoint_dir(run_dir, cycle)

        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda p: _DiskUsage(total=100, used=100, free=0),
        )

        # keep_last=4 trims 1..4 away; disk floor then prunes survivors
        # oldest-first down to the single newest.
        removed = prune_checkpoint_cycles(run_dir, keep_last=4, min_free_disk_gb=1000.0)

        assert [_cycle_of(p) for p in removed] == [1, 2, 3, 4, 5, 6, 7]
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [8]


class TestConfigDrivenWiring:
    """The config->prune coupling wired into train_tg_lora's periodic-save path.

    These exercise prune_checkpoint_cycles_from_cfg — the unit-tested seam
    extracted from an inline read+call. They prove the knobs (read from
    cfg.logging) actually fire pruning, closing the "protection exists but
    never fires" gap an untested inline block left open: a renamed key or
    inverted guard would leave the disk-death protection inert while every
    isolated prune_checkpoint_cycles test still passed.
    """

    @staticmethod
    def _cfg(keep_last=0, min_free=0.0):
        return {"logging": {"keep_last_checkpoints": keep_last,
                            "min_free_disk_gb": min_free}}

    def test_m10_baseline_knobs_fire_count_bound(self, tmp_path):
        # The exact knobs shipped in the M10 baseline + guard configs.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 9):  # 8 dirs
            _save_checkpoint_dir(run_dir, cycle)

        removed = prune_checkpoint_cycles_from_cfg(self._cfg(3, 2.0), run_dir)

        assert [_cycle_of(p) for p in removed] == [1, 2, 3, 4, 5]
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [6, 7, 8]

    def test_defaults_off_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)

        assert prune_checkpoint_cycles_from_cfg(self._cfg(), run_dir) == []
        assert prune_checkpoint_cycles_from_cfg({}, run_dir) == []
        assert len(_checkpoint_dirs(run_dir)) == 5

    def test_missing_logging_section_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _save_checkpoint_dir(run_dir, 1)
        assert prune_checkpoint_cycles_from_cfg(
            {"experiment": {"name": "x"}}, run_dir
        ) == []
        assert prune_checkpoint_cycles_from_cfg(None, run_dir) == []

    def test_min_free_only_fires_from_cfg(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 4):
            _save_checkpoint_dir(run_dir, cycle)
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: _DiskUsage(total=100, used=100, free=0),
        )
        removed = prune_checkpoint_cycles_from_cfg(
            self._cfg(keep_last=0, min_free=1000.0), run_dir
        )
        assert [_cycle_of(p) for p in removed] == [1, 2]
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [3]

    def test_works_with_omegaconf_dictconfig(self, tmp_path):
        # Prod passes an OmegaConf DictConfig, not a plain dict.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)
        cfg = OmegaConf.create({"logging": {"keep_last_checkpoints": 2,
                                            "min_free_disk_gb": 0.0}})
        removed = prune_checkpoint_cycles_from_cfg(cfg, run_dir)
        assert [_cycle_of(p) for p in removed] == [1, 2, 3]
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [4, 5]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_M10_CONFIGS = [
    "configs/9b_tg_lora_m10_dynfreeze.yaml",
    "configs/9b_tg_lora_m10_dynfreeze_baseline.yaml",
]


class TestM10ConfigsShipPruningKnobs:
    """Guard feedback point (1) against regression.

    The M10 autonomous run configs must actually enable
    keep_last_checkpoints>0 and min_free_disk_gb>0 — both save every cycle
    (save_every_cycles: 1), so without these knobs either run accumulates one
    checkpoint-cycle-* dir per cycle forever (the M10.3 disk-death class).
    db3c08f enabled them on the baseline only; this asserts the TG-LoRA guard
    config stays symmetric. A future commit that zeroes/removes the knobs on
    either config fails here.
    """

    @staticmethod
    def _logging(rel):
        return OmegaConf.load(_REPO_ROOT / rel).logging

    def test_keep_last_checkpoints_enabled_in_both_m10_configs(self):
        for rel in _M10_CONFIGS:
            value = int(self._logging(rel).keep_last_checkpoints)
            assert value > 0, f"{rel}: keep_last_checkpoints must be > 0, got {value}"

    def test_min_free_disk_gb_enabled_in_both_m10_configs(self):
        for rel in _M10_CONFIGS:
            value = float(self._logging(rel).min_free_disk_gb)
            assert value > 0.0, f"{rel}: min_free_disk_gb must be > 0.0, got {value}"

    def test_m10_configs_save_every_cycle_so_guard_is_required(self):
        # Documents WHY the guard is mandatory here: a per-cycle save with no
        # prune is exactly the unbounded accumulation class.
        for rel in _M10_CONFIGS:
            assert int(self._logging(rel).save_every_cycles) == 1, rel


class TestM10ConfigsValidateThroughSchema:
    """The M10.3 guard knobs must survive the Pydantic schema gate at startup.

    db3c08f/3c2c9d7 shipped keep_last_checkpoints/min_free_disk_gb into the M10
    configs and proved the prune logic fires (TestKeepLastBoundsAccumulation) and
    that the YAML carries the knobs (TestM10ConfigsShipPruningKnobs). But
    LoggingConfig has extra="forbid", and the knobs were added to the YAML
    *without* being declared on the schema — so main()'s
    load_validate_and_build_config raised ValidationError and the guarded run
    could never start: the protection existed but the run died before training.

    TestM10ConfigsShipPruningKnobs loads via raw OmegaConf.load (bypassing the
    schema), which is exactly why it missed the crash. These load through the real
    startup path and close the loop the incident requires: config -> schema ->
    runtime prune seam -> bounded disk.
    """

    def test_both_m10_configs_load_through_schema_gate(self):
        # main() -> load_validate_and_build_config is the actual startup path;
        # it must not reject the guard knobs. The knobs must also survive the
        # Pydantic round-trip into the typed cfg prune_checkpoint_cycles_from_cfg
        # reads at runtime (validated.model_dump -> typed_cfg).
        for rel in _M10_CONFIGS:
            validated, typed_cfg = load_validate_and_build_config(_REPO_ROOT / rel)
            assert int(typed_cfg.logging.keep_last_checkpoints) > 0, rel
            assert float(typed_cfg.logging.min_free_disk_gb) > 0.0, rel

    def test_guarded_cfg_fires_prune_seam_end_to_end(self, tmp_path):
        # The validated cfg the trainer actually holds must fire pruning through
        # the runtime seam — not just carry the knobs. This is the closed loop:
        # a real M10 config file -> schema -> typed cfg -> bounded on-disk count.
        for rel in _M10_CONFIGS:
            _, typed_cfg = load_validate_and_build_config(_REPO_ROOT / rel)
            run_dir = tmp_path / rel.replace("/", "_")
            run_dir.mkdir()
            for cycle in range(1, 9):  # 8 dirs, keep_last=3 -> keep newest 3
                _save_checkpoint_dir(run_dir, cycle)
            removed = prune_checkpoint_cycles_from_cfg(typed_cfg, run_dir)
            assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [6, 7, 8]
            assert len(removed) == 5


class TestPeriodicSavePathBoundedDisk:
    """The training loop's periodic-save path bounds disk across many saves.

    Closes the residual gap after db3c08f/3c2c9d7/d2218ed: those commits proved
    prune_checkpoint_cycles bounds accumulation and that the M10 configs carry
    the knobs and survive the schema gate — but every test calls the prune
    function *directly*. None exercised the actual periodic-save block in
    train_tg_lora, whose inline save->prune call was an anonymous line in a
    multi-thousand-line loop. ``save_periodic_cycle_checkpoint`` is that block
    extracted into a testable seam; these tests drive the exact function the
    loop calls across many saves and assert bounded disk + per-save artifact
    logging — the real "the protection fires in the save path" proof, not the
    "the function works in isolation" proof the earlier tests give.
    """

    @staticmethod
    def _fake_save(model, tokenizer, save_dir):
        """Stand-in for save_checkpoint: create the dir + a fixed-size payload.

        The model-save internals are tested elsewhere (test_artifact_logging,
        resume E2E); here we test the save->artifact->prune orchestration that
        bounds disk, so a payload file stands in for the real safetensors write.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "adapter_model.safetensors").write_bytes(b"\x00" * PAYLOAD_BYTES)

    def test_loop_save_path_bounds_disk_across_many_saves(self, tmp_path, monkeypatch):
        import src.utils.checkpoint as ckpt

        monkeypatch.setattr(ckpt, "save_checkpoint", self._fake_save)
        monkeypatch.setattr(ckpt, "save_training_state", lambda state, path: None)

        # Real M10 config file -> schema -> typed cfg the trainer holds at runtime.
        _, typed_cfg = load_validate_and_build_config(
            _REPO_ROOT / "configs/9b_tg_lora_m10_dynfreeze.yaml"
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = int(typed_cfg.logging.keep_last_checkpoints)

        peak_bytes = 0
        for cycle in range(1, 41):
            checkpoint_dir = run_dir / f"{CHECKPOINT_DIR_PREFIX}{cycle}"
            save_periodic_cycle_checkpoint(
                model=None,
                tokenizer=None,
                checkpoint_dir=checkpoint_dir,
                run_dir=run_dir,
                cfg=typed_cfg,
                training_state=object(),
                log_artifact=lambda d, a: None,
            )
            peak_bytes = max(peak_bytes, _checkpoint_total_bytes(run_dir))

        # Only the newest keep_last survive; older dirs are truly gone.
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [38, 39, 40]
        # Footprint bounded to ~keep_last payloads, not the linear 40 the
        # unguarded path would reach.
        assert _checkpoint_total_bytes(run_dir) <= keep_last * PAYLOAD_BYTES
        assert peak_bytes < 40 * PAYLOAD_BYTES

    def test_loop_save_path_logs_artifact_each_save(self, tmp_path, monkeypatch):
        import src.utils.checkpoint as ckpt

        monkeypatch.setattr(ckpt, "save_checkpoint", self._fake_save)
        monkeypatch.setattr(ckpt, "save_training_state", lambda state, path: None)

        _, typed_cfg = load_validate_and_build_config(
            _REPO_ROOT / "configs/9b_tg_lora_m10_dynfreeze_baseline.yaml"
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        logged = []
        for cycle in range(1, 6):
            checkpoint_dir = run_dir / f"{CHECKPOINT_DIR_PREFIX}{cycle}"
            save_periodic_cycle_checkpoint(
                model=None,
                tokenizer=None,
                checkpoint_dir=checkpoint_dir,
                run_dir=run_dir,
                cfg=typed_cfg,
                training_state=object(),
                log_artifact=lambda d, a: logged.append((str(d), a)),
            )

        # Every save logged its checkpoint dir as a "checkpoints" artifact —
        # the artifact coupling is preserved by the extraction.
        assert len(logged) == 5
        assert all(artifact == "checkpoints" for _, artifact in logged)
        assert logged[-1] == (str(run_dir / f"{CHECKPOINT_DIR_PREFIX}5"), "checkpoints")

    def test_loop_save_path_defaults_off_keeps_every_dir(self, tmp_path, monkeypatch):
        # With the knobs off (a non-M10 config), the seam must NOT prune — the
        # default-off contract that leaves unrelated runs untouched.
        import src.utils.checkpoint as ckpt

        monkeypatch.setattr(ckpt, "save_checkpoint", self._fake_save)
        monkeypatch.setattr(ckpt, "save_training_state", lambda state, path: None)

        cfg = {"logging": {"keep_last_checkpoints": 0, "min_free_disk_gb": 0.0}}
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        for cycle in range(1, 6):
            checkpoint_dir = run_dir / f"{CHECKPOINT_DIR_PREFIX}{cycle}"
            removed = save_periodic_cycle_checkpoint(
                model=None,
                tokenizer=None,
                checkpoint_dir=checkpoint_dir,
                run_dir=run_dir,
                cfg=cfg,
                training_state=object(),
            )

        assert removed == []
        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [1, 2, 3, 4, 5]


class TestTrainLoopWiresTheGuard:
    """Structural guard: the periodic-save block actually calls the seam.

    The behavioral tests above prove ``save_periodic_cycle_checkpoint`` bounds
    disk; this pins that train_tg_lora's periodic-save path imports and invokes
    it, so a refactor that silently drops the call (re-opening the disk-death
    class) fails here rather than slipping past while every seam test stays
    green.
    """

    def test_periodic_save_block_imports_and_calls_the_seam(self):
        source = (_REPO_ROOT / "src/training/train_tg_lora.py").read_text()
        tree = ast.parse(source)

        imported = False
        call_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.utils.checkpoint":
                imported |= any(
                    a.name == "save_periodic_cycle_checkpoint" for a in node.names
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "save_periodic_cycle_checkpoint"
            ):
                call_count += 1

        assert imported, (
            "train_tg_lora.py must import save_periodic_cycle_checkpoint "
            "(the M10.3 disk-death guard seam)"
        )
        assert call_count >= 1, (
            "train_tg_lora.py periodic-save path must call "
            "save_periodic_cycle_checkpoint — dropping it re-opens unbounded "
            "checkpoint-cycle-* accumulation (M10.3 disk-death class)"
        )


# =============================================================================
# Trajectory-delta-artifact pruning — the M10.3 guard's second vector.
#
# db3c08f..43acef0 closed the checkpoint-cycle-* disk-death class and d2218ed /
# c96a5a0 made the knobs survive the schema gate repo-wide. But
# save_trajectory_delta_artifacts writes 1-2 .pt files per cycle into
# run_dir/trajectory_delta_artifacts/ and never removes old ones, and the cycle
# guard's regex (checkpoint-cycle-*) and min_free_disk_gb floor never scan that
# path — so the M10 runs (save_every_cycles: 1, save_trajectory_delta_artifacts:
# true, up to 120 cycles => ~240 delta-tensor files) accumulated this vector
# unbounded *while keep_last_checkpoints was already on*. These tests prove the
# same knobs now bound it, and that the trainer actually wires the seam.
# =============================================================================

ARTIFACT_SUBDIR = "trajectory_delta_artifacts"
_ARTIFACT_ANCHORS = ("after_pilot", "after_speculative_update")


def _save_artifact_files(
    run_dir, cycle, anchors=_ARTIFACT_ANCHORS, payload_bytes=PAYLOAD_BYTES
):
    """Stand-in for save_trajectory_delta_artifact.

    Writes .pt files matching the real artifact_file_name pattern
    (tg_lora_<anchor>_cycle_NNNNNN.pt), one per anchor for the given cycle. The
    model-save internals are tested elsewhere; here a fixed payload stands in for
    the real delta-tensor write so we can assert disk bounding.
    """
    art_dir = Path(run_dir) / ARTIFACT_SUBDIR
    art_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for anchor in anchors:
        f = art_dir / f"tg_lora_{anchor}_cycle_{cycle:06d}.pt"
        f.write_bytes(b"\x00" * payload_bytes)
        written.append(f)
    return written


def _artifact_files(run_dir):
    art_dir = Path(run_dir) / ARTIFACT_SUBDIR
    if not art_dir.is_dir():
        return []
    return sorted(
        (p for p in art_dir.iterdir() if p.is_file() and p.suffix == ".pt"),
        key=lambda p: p.name,
    )


def _artifact_cycle_of(path):
    # tg_lora_after_pilot_cycle_000003.pt -> 3
    return int(path.name.rsplit("_", 1)[1].split(".")[0])


def _artifact_total_bytes(run_dir):
    return sum(f.stat().st_size for f in _artifact_files(run_dir))


class TestTrajectoryArtifactBounding:
    """keep_last bounds the per-cycle .pt accumulation the cycle guard doesn't see."""

    def test_older_files_truly_removed_after_many_cycles(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3
        for cycle in range(1, 41):
            _save_artifact_files(run_dir, cycle)
            prune_trajectory_delta_artifacts(run_dir, keep_last=keep_last)

        remaining = _artifact_files(run_dir)
        assert sorted({_artifact_cycle_of(p) for p in remaining}) == [38, 39, 40]
        # All anchors for the surviving cycles are kept together.
        assert len(remaining) == keep_last * len(_ARTIFACT_ANCHORS)
        # Older cycles are truly gone.
        for cycle in range(1, 38):
            assert not any(_artifact_cycle_of(p) == cycle for p in remaining)

    def test_count_never_exceeds_keep_last_cycles_during_loop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3
        for cycle in range(1, 41):
            _save_artifact_files(run_dir, cycle)
            prune_trajectory_delta_artifacts(run_dir, keep_last=keep_last)
            cycles = {_artifact_cycle_of(p) for p in _artifact_files(run_dir)}
            assert len(cycles) <= keep_last

    def test_footprint_stays_bounded_not_linear(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3
        anchors = len(_ARTIFACT_ANCHORS)
        peak = 0
        for cycle in range(1, 41):
            _save_artifact_files(run_dir, cycle)
            prune_trajectory_delta_artifacts(run_dir, keep_last=keep_last)
            peak = max(peak, _artifact_total_bytes(run_dir))

        final = _artifact_total_bytes(run_dir)
        assert final <= keep_last * anchors * PAYLOAD_BYTES
        assert peak <= (keep_last + 1) * anchors * PAYLOAD_BYTES
        assert peak < 40 * anchors * PAYLOAD_BYTES


class TestTrajectoryArtifactContract:
    """Edge behaviour of the count bound, and isolation from the cycle guard."""

    def test_defaults_off_preserves_unbounded_behavior(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_artifact_files(run_dir, cycle)

        removed = prune_trajectory_delta_artifacts(run_dir)

        assert removed == []
        assert len(_artifact_files(run_dir)) == 5 * len(_ARTIFACT_ANCHORS)

    def test_keep_last_larger_than_count_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 3):
            _save_artifact_files(run_dir, cycle)

        removed = prune_trajectory_delta_artifacts(run_dir, keep_last=10)

        assert removed == []
        assert sorted({_artifact_cycle_of(p) for p in _artifact_files(run_dir)}) == [1, 2]

    def test_no_files_is_safe_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert prune_trajectory_delta_artifacts(run_dir, keep_last=3) == []
        # Even an absent subdir is safe.
        assert prune_trajectory_delta_artifacts(tmp_path / "never", keep_last=3) == []

    def test_non_pt_and_unsortable_files_are_never_touched(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        manifest = run_dir / ARTIFACT_SUBDIR / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        # No _cycle_N / _step_N suffix -> unsortable, so never pruned.
        anchorless = run_dir / ARTIFACT_SUBDIR / "tg_lora_summary.pt"
        anchorless.write_bytes(b"\x00")
        for cycle in range(1, 6):
            _save_artifact_files(run_dir, cycle)

        prune_trajectory_delta_artifacts(run_dir, keep_last=2)

        assert manifest.exists()
        assert anchorless.exists()

    def test_artifact_prune_never_touches_checkpoint_cycle_dirs(self, tmp_path):
        # The two regexes are disjoint: checkpoint-cycle-N carries the -cycle-
        # infix the artifact regex cannot match, and artifacts live in a subdir.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)
            _save_artifact_files(run_dir, cycle)

        prune_trajectory_delta_artifacts(run_dir, keep_last=1)

        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [1, 2, 3, 4, 5]
        assert sorted({_artifact_cycle_of(p) for p in _artifact_files(run_dir)}) == [5]

    def test_cycle_prune_never_touches_artifact_files(self, tmp_path):
        # Symmetric isolation in the other direction.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_checkpoint_dir(run_dir, cycle)
            _save_artifact_files(run_dir, cycle)

        prune_checkpoint_cycles(run_dir, keep_last=1)

        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [5]
        assert sorted({_artifact_cycle_of(p) for p in _artifact_files(run_dir)}) == [
            1, 2, 3, 4, 5,
        ]


class TestTrajectoryArtifactDiskFloor:
    """min_free_disk_gb prunes oldest cycle-keys until the floor is met."""

    def test_floor_met_stops_pruning(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_artifact_files(run_dir, cycle)

        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda p: _DiskUsage(total=100, used=0, free=10 * 1024 ** 3),
        )

        removed = prune_trajectory_delta_artifacts(
            run_dir, keep_last=0, min_free_disk_gb=2.0
        )

        assert removed == []
        assert len(_artifact_files(run_dir)) == 5 * len(_ARTIFACT_ANCHORS)

    def test_low_disk_prunes_oldest_keys_until_floor_met(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_artifact_files(run_dir, cycle)

        # Free space rises 1 MiB per check; a 3 MiB floor is met on the 3rd
        # check, so the two oldest cycle-keys (all their anchors) go first.
        calls = {"n": 0}

        def fake_du(p):
            calls["n"] += 1
            free = (1 + (calls["n"] - 1)) * 1024 * 1024  # 1,2,3,... MiB
            return _DiskUsage(total=100 * 1024 ** 3, used=0, free=free)

        monkeypatch.setattr(shutil, "disk_usage", fake_du)

        removed = prune_trajectory_delta_artifacts(
            run_dir, keep_last=0, min_free_disk_gb=3.0 / 1024  # 3 MiB floor
        )

        assert sorted({_artifact_cycle_of(p) for p in removed}) == [1, 2]
        assert sorted({_artifact_cycle_of(p) for p in _artifact_files(run_dir)}) == [3, 4, 5]

    def test_floor_never_deletes_the_last_key(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 4):
            _save_artifact_files(run_dir, cycle)

        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda p: _DiskUsage(total=100, used=100, free=0),
        )

        removed = prune_trajectory_delta_artifacts(
            run_dir, keep_last=0, min_free_disk_gb=1000.0
        )

        assert sorted({_artifact_cycle_of(p) for p in removed}) == [1, 2]
        assert sorted({_artifact_cycle_of(p) for p in _artifact_files(run_dir)}) == [3]


class TestTrajectoryArtifactConfigDrivenWiring:
    """prune_trajectory_delta_artifacts_from_cfg — the seam wired into the
    periodic-save path. Reads the SAME knobs as the cycle guard, so one opt-in
    bounds both vectors; the M10 configs already set them."""

    @staticmethod
    def _cfg(keep_last=0, min_free=0.0):
        return {"logging": {"keep_last_checkpoints": keep_last,
                            "min_free_disk_gb": min_free}}

    def test_knobs_fire_count_bound(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 9):
            _save_artifact_files(run_dir, cycle)

        removed = prune_trajectory_delta_artifacts_from_cfg(self._cfg(3, 2.0), run_dir)

        assert sorted({_artifact_cycle_of(p) for p in removed}) == [1, 2, 3, 4, 5]
        assert sorted({_artifact_cycle_of(p) for p in _artifact_files(run_dir)}) == [6, 7, 8]

    def test_defaults_off_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_artifact_files(run_dir, cycle)

        assert prune_trajectory_delta_artifacts_from_cfg(self._cfg(), run_dir) == []
        assert prune_trajectory_delta_artifacts_from_cfg({}, run_dir) == []

    def test_missing_logging_section_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _save_artifact_files(run_dir, 1)
        assert prune_trajectory_delta_artifacts_from_cfg(
            {"experiment": {"name": "x"}}, run_dir
        ) == []
        assert prune_trajectory_delta_artifacts_from_cfg(None, run_dir) == []

    def test_works_with_omegaconf_dictconfig(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for cycle in range(1, 6):
            _save_artifact_files(run_dir, cycle)
        cfg = OmegaConf.create({"logging": {"keep_last_checkpoints": 2,
                                            "min_free_disk_gb": 0.0}})

        removed = prune_trajectory_delta_artifacts_from_cfg(cfg, run_dir)

        assert sorted({_artifact_cycle_of(p) for p in removed}) == [1, 2, 3]
        assert sorted({_artifact_cycle_of(p) for p in _artifact_files(run_dir)}) == [4, 5]


class TestM10ConfigBoundsArtifactsEndToEnd:
    """Closed loop: a real M10 config file -> schema -> typed cfg -> bounded
    trajectory_delta_artifacts/ count.

    Both M10 configs ship save_trajectory_delta_artifacts: true AND
    keep_last_checkpoints=3. Before this guard the knobs were "on" but inert for
    artifacts — the cycle guard never scanned that path. This proves the same
    typed cfg the trainer holds at runtime now bounds the artifact dir too.
    """

    def test_both_m10_configs_bound_the_artifact_dir(self, tmp_path):
        for rel in _M10_CONFIGS:
            _, typed_cfg = load_validate_and_build_config(_REPO_ROOT / rel)
            run_dir = tmp_path / rel.replace("/", "_")
            run_dir.mkdir()
            for cycle in range(1, 9):  # 8 cycles, 2 anchors each = 16 files
                _save_artifact_files(run_dir, cycle)

            removed = prune_trajectory_delta_artifacts_from_cfg(typed_cfg, run_dir)

            assert sorted(
                {_artifact_cycle_of(p) for p in _artifact_files(run_dir)}
            ) == [6, 7, 8], rel
            assert len(removed) == 5 * len(_ARTIFACT_ANCHORS), rel

    def test_both_m10_configs_enable_artifact_saving_so_guard_is_required(self):
        # Documents WHY this guard is mandatory on the M10 runs: with
        # save_trajectory_delta_artifacts: true and interval default 1, both
        # write 1-2 .pt files every cycle that would otherwise accumulate forever.
        for rel in _M10_CONFIGS:
            cfg = OmegaConf.load(_REPO_ROOT / rel)
            assert bool(cfg.training.get("save_trajectory_delta_artifacts", False)), rel


class TestTrainLoopWiresTheArtifactGuard:
    """Structural guard: the periodic-save block actually calls the artifact seam.

    Mirrors TestTrainLoopWiresTheGuard — a refactor that silently drops the call
    re-opens the trajectory-artifact accumulation vector while every isolated seam
    test stays green.
    """

    def test_periodic_save_block_imports_and_calls_artifact_prune(self):
        source = (_REPO_ROOT / "src/training/train_tg_lora.py").read_text()
        tree = ast.parse(source)
        imported = False
        call_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.utils.checkpoint":
                imported |= any(
                    a.name == "prune_trajectory_delta_artifacts_from_cfg"
                    for a in node.names
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "prune_trajectory_delta_artifacts_from_cfg"
            ):
                call_count += 1

        assert imported, (
            "train_tg_lora.py must import prune_trajectory_delta_artifacts_from_cfg "
            "(the trajectory-artifact arm of the M10.3 disk-death guard)"
        )
        assert call_count >= 1, (
            "train_tg_lora.py periodic-save path must call "
            "prune_trajectory_delta_artifacts_from_cfg — dropping it re-opens "
            "unbounded trajectory_delta_artifacts/*.pt accumulation, the second "
            "M10.3 disk-death vector the cycle guard alone leaves open"
        )


# =============================================================================
# Step-checkpoint pruning — the M10.3 guard's third vector (baseline entrypoint).
#
# The cycle + artifact guards above closed the disk-death class on the TG-LoRA
# trainer. But the baseline QLoRA trainer (train_baseline_qlora, `make
# train-baseline`) writes checkpoint-<global_step> dirs every save_every_steps
# AND, when save_trajectory_delta_artifacts is on, .pt files into
# trajectory_delta_artifacts/ — and called NO pruner. The cycle guard's regex
# (^checkpoint-cycle-(\d+)$) deliberately does not match checkpoint-<step>, so
# those knobs were "on" but inert for the baseline code path: the exact
# "protection exists but doesn't fire" class. prune_step_checkpoints targets the
# step-naming scheme; the baseline trainer now calls BOTH it and the (existing)
# artifact pruner, reusing the SAME knobs. Default-off, so the shipped baseline
# configs keep today's behavior until they opt in.
# =============================================================================

STEP_CHECKPOINT_PREFIX = "checkpoint-"


def _save_step_checkpoint_dir(run_dir, step, payload_bytes=PAYLOAD_BYTES):
    """Stand-in for the baseline trainer's ``checkpoint-<global_step>`` save."""
    ckpt_dir = run_dir / f"{STEP_CHECKPOINT_PREFIX}{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"\x00" * payload_bytes)
    return ckpt_dir


def _step_checkpoint_dirs(run_dir):
    def _is_step_dir(name):
        # checkpoint-<digits> only; excludes checkpoint-cycle-* (rest not all
        # digits) and checkpoint-best / best_model / oom_checkpoint.
        rest = name[len(STEP_CHECKPOINT_PREFIX):]
        return name.startswith(STEP_CHECKPOINT_PREFIX) and rest.isdigit()

    return sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and _is_step_dir(p.name)),
        key=lambda p: int(p.name[len(STEP_CHECKPOINT_PREFIX):]),
    )


def _step_of(path):
    return int(path.name[len(STEP_CHECKPOINT_PREFIX):])


class TestStepCheckpointBounding:
    """keep_last bounds the baseline trainer's per-step checkpoint accumulation."""

    def test_older_step_dirs_truly_removed_after_many_saves(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3
        for step in range(250, 10001, 250):  # 40 step-saves, like max_steps=10000
            _save_step_checkpoint_dir(run_dir, step)
            prune_step_checkpoints(run_dir, keep_last=keep_last)

        remaining = _step_checkpoint_dirs(run_dir)
        assert [_step_of(p) for p in remaining] == [9250, 9500, 9750, 10000][-keep_last:]
        # Older steps are truly gone.
        for step in range(250, 9250, 250):
            assert not (run_dir / f"{STEP_CHECKPOINT_PREFIX}{step}").exists()

    def test_step_count_never_exceeds_keep_last_during_loop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        keep_last = 3
        for step in range(250, 10001, 250):
            _save_step_checkpoint_dir(run_dir, step)
            prune_step_checkpoints(run_dir, keep_last=keep_last)
            assert len(_step_checkpoint_dirs(run_dir)) <= keep_last


class TestStepCheckpointContract:
    """Edge behaviour of the step-checkpoint count bound + naming disjointness."""

    def test_defaults_off_preserves_unbounded_behavior(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500, 750):
            _save_step_checkpoint_dir(run_dir, step)
        assert prune_step_checkpoints(run_dir) == []  # keep_last=0, min_free=0
        assert len(_step_checkpoint_dirs(run_dir)) == 3

    def test_keep_last_larger_than_count_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500):
            _save_step_checkpoint_dir(run_dir, step)
        assert prune_step_checkpoints(run_dir, keep_last=10) == []
        assert len(_step_checkpoint_dirs(run_dir)) == 2

    def test_no_step_dirs_is_safe_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert prune_step_checkpoints(run_dir, keep_last=3) == []
        assert prune_step_checkpoints(tmp_path / "never", keep_last=3) == []

    def test_non_numeric_checkpoint_dirs_are_never_touched(self, tmp_path):
        # best_model / oom_checkpoint / checkpoint-best must survive: the
        # ^checkpoint-(\d+)$ regex only matches numeric step dirs.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500, 750):
            _save_step_checkpoint_dir(run_dir, step)
        (run_dir / "best_model").mkdir()
        (run_dir / "oom_checkpoint").mkdir()
        (run_dir / "checkpoint-best").mkdir()

        removed = prune_step_checkpoints(run_dir, keep_last=1)

        assert [_step_of(p) for p in _step_checkpoint_dirs(run_dir)] == [750]
        assert {p.name for p in removed} == {f"{STEP_CHECKPOINT_PREFIX}250",
                                             f"{STEP_CHECKPOINT_PREFIX}500"}
        assert (run_dir / "best_model").exists()
        assert (run_dir / "oom_checkpoint").exists()
        assert (run_dir / "checkpoint-best").exists()

    def test_step_prune_never_touches_cycle_dirs(self, tmp_path):
        # Forward disjointness: the step regex must not match checkpoint-cycle-*.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500, 750):
            _save_step_checkpoint_dir(run_dir, step)
        _save_checkpoint_dir(run_dir, 5)  # checkpoint-cycle-5

        prune_step_checkpoints(run_dir, keep_last=1)

        assert [_step_of(p) for p in _step_checkpoint_dirs(run_dir)] == [750]
        assert len(_checkpoint_dirs(run_dir)) == 1  # the cycle dir survived

    def test_cycle_prune_never_touches_step_dirs(self, tmp_path):
        # Reverse disjointness: the cycle regex must not match checkpoint-<step>.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500, 750):
            _save_step_checkpoint_dir(run_dir, step)
        _save_checkpoint_dir(run_dir, 5)  # checkpoint-cycle-5

        prune_checkpoint_cycles(run_dir, keep_last=1)

        assert [_cycle_of(p) for p in _checkpoint_dirs(run_dir)] == [5]
        assert len(_step_checkpoint_dirs(run_dir)) == 3  # all step dirs survived


class TestStepCheckpointDiskFloor:
    """min_free_disk_gb reclaims oldest step dirs first, never the newest step."""

    def test_floor_never_deletes_the_last_step(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500, 750):
            _save_step_checkpoint_dir(run_dir, step)

        # Impossible floor on a tiny disk -> reclaims everything it can but the
        # policy holds the single newest step as a recovery point.
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda path: _DiskUsage(total=100 * 1024 ** 3, used=99 * 1024 ** 3,
                                     free=0),
        )
        removed = prune_step_checkpoints(run_dir, keep_last=0, min_free_disk_gb=1000.0)

        assert sorted(_step_of(p) for p in removed) == [250, 500]
        assert [_step_of(p) for p in _step_checkpoint_dirs(run_dir)] == [750]


class TestStepCheckpointConfigDrivenWiring:
    """prune_step_checkpoints_from_cfg — the seam wired into the baseline
    periodic-save path. Reads the SAME knobs as the cycle/artifact guards."""

    @staticmethod
    def _cfg(keep_last=0, min_free=0.0):
        return {"logging": {"keep_last_checkpoints": keep_last,
                            "min_free_disk_gb": min_free}}

    def test_knobs_fire_count_bound(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500, 750, 1000, 1250):
            _save_step_checkpoint_dir(run_dir, step)

        removed = prune_step_checkpoints_from_cfg(self._cfg(2, 0.0), run_dir)

        assert sorted(_step_of(p) for p in removed) == [250, 500, 750]
        assert [_step_of(p) for p in _step_checkpoint_dirs(run_dir)] == [1000, 1250]

    def test_defaults_off_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500):
            _save_step_checkpoint_dir(run_dir, step)
        assert prune_step_checkpoints_from_cfg(self._cfg(), run_dir) == []
        assert prune_step_checkpoints_from_cfg({}, run_dir) == []

    def test_missing_logging_section_is_noop(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _save_step_checkpoint_dir(run_dir, 250)
        assert prune_step_checkpoints_from_cfg(
            {"experiment": {"name": "x"}}, run_dir
        ) == []
        assert prune_step_checkpoints_from_cfg(None, run_dir) == []

    def test_works_with_omegaconf_dictconfig(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (250, 500, 750):
            _save_step_checkpoint_dir(run_dir, step)
        cfg = OmegaConf.create({"logging": {"keep_last_checkpoints": 1,
                                            "min_free_disk_gb": 0.0}})

        removed = prune_step_checkpoints_from_cfg(cfg, run_dir)

        assert [_step_of(p) for p in removed] == [250, 500]
        assert [_step_of(p) for p in _step_checkpoint_dirs(run_dir)] == [750]


class TestBaselineConfigFiresStepPruneEndToEnd:
    """Closed loop: a real baseline config + schema + opt-in knobs -> bounded
    step-checkpoint count.

    No shipped baseline config sets the knobs (default-off preserves today's
    behavior for the comparison/paper runs). This flips them on through the real
    startup path (load_validate_and_build_config overrides -> Pydantic
    round-trip) to prove the knob survives the schema for a baseline config and
    that the typed cfg the baseline trainer holds at runtime fires pruning.
    """

    def test_baseline_config_with_knobs_bounds_step_dirs(self, tmp_path):
        _, typed_cfg = load_validate_and_build_config(
            _REPO_ROOT / "configs/9b_baseline.yaml",
            overrides=["logging.keep_last_checkpoints=3",
                       "logging.min_free_disk_gb=0.0"],
        )
        assert int(typed_cfg.logging.keep_last_checkpoints) == 3  # survived schema

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in range(250, 2001, 250):  # 8 step-saves, keep_last=3 -> keep newest 3
            _save_step_checkpoint_dir(run_dir, step)

        removed = prune_step_checkpoints_from_cfg(typed_cfg, run_dir)

        assert [_step_of(p) for p in _step_checkpoint_dirs(run_dir)] == [1500, 1750, 2000]
        assert len(removed) == 5


class TestBaselineLoopWiresTheGuard:
    """Structural guard: the baseline periodic-save block calls BOTH prune seams.

    Mirrors TestTrainLoopWiresTheArtifactGuard for the other training entrypoint.
    A refactor that silently drops either call re-opens unbounded accumulation
    on the baseline path while every isolated seam test stays green. ``make
    train-baseline`` saves checkpoint-<step> dirs every save_every_steps and,
    with save_trajectory_delta_artifacts: true, .pt files every step — both must
    be pruned through the cfg seams for the knobs to take effect there.
    """

    @staticmethod
    def _imports_and_calls(source, name):
        tree = ast.parse(source)
        imported = False
        call_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.utils.checkpoint":
                imported |= any(a.name == name for a in node.names)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
            ):
                call_count += 1
        return imported, call_count

    def test_periodic_save_block_imports_and_calls_step_prune(self):
        source = (_REPO_ROOT / "src/training/train_baseline_qlora.py").read_text()
        imported, call_count = self._imports_and_calls(
            source, "prune_step_checkpoints_from_cfg"
        )
        assert imported, (
            "train_baseline_qlora.py must import prune_step_checkpoints_from_cfg "
            "(the step-checkpoint arm of the M10.3 disk-death guard for the "
            "baseline entrypoint — the cycle regex never matches checkpoint-<step>)"
        )
        assert call_count >= 1, (
            "train_baseline_qlora.py periodic-save path must call "
            "prune_step_checkpoints_from_cfg — dropping it re-opens unbounded "
            "checkpoint-<step> accumulation on the baseline code path"
        )

    def test_periodic_save_block_imports_and_calls_artifact_prune(self):
        source = (_REPO_ROOT / "src/training/train_baseline_qlora.py").read_text()
        imported, call_count = self._imports_and_calls(
            source, "prune_trajectory_delta_artifacts_from_cfg"
        )
        assert imported, (
            "train_baseline_qlora.py must import "
            "prune_trajectory_delta_artifacts_from_cfg — the baseline saves "
            "trajectory_delta_artifacts/*.pt (save_trajectory_delta_artifacts: "
            "true) on the same path the TG-LoRA guard already covers"
        )
        assert call_count >= 1, (
            "train_baseline_qlora.py periodic-save path must call "
            "prune_trajectory_delta_artifacts_from_cfg — dropping it re-opens "
            "unbounded trajectory_delta_artifacts/*.pt accumulation on the "
            "baseline code path"
        )

