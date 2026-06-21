"""E2E test: keep_last_checkpoints bounds on-disk checkpoint growth.

Closes the M10.3 disk-death gap: the TG-LoRA periodic-save path writes a fresh
``checkpoint-cycle-<N>`` dir every ``save_every_cycles`` and never removes old
ones. With the M10 baseline saving every cycle for 120 cycles that is 120 dirs
of unbounded accumulation — the exact incident class this guard prevents.

With ``keep_last`` set, older ``checkpoint-cycle-*`` dirs must be *truly*
removed and the total checkpoint footprint must stay bounded across many saves,
not grow linearly with the cycle count.
"""

import shutil
from collections import namedtuple
from pathlib import Path

from omegaconf import OmegaConf

from src.training.config_schema import load_validate_and_build_config
from src.utils.checkpoint import prune_checkpoint_cycles, prune_checkpoint_cycles_from_cfg

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
