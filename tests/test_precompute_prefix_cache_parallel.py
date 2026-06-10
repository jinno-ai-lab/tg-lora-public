"""Tests for scripts/precompute_prefix_cache_parallel.py — pure function coverage.

Covers: _normalize_devices, _selected_labels, _dataset_path, _count_records,
_resolve_split_layer (validation guards), and _build_worker_configs.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.training.config_schema import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoRAConfig,
    LoggingConfig,
    ModelConfig,
    TGLoRAConfig,
    TGLoRAParams,
    TrainingConfig,
)


def _make_config(
    *,
    prefix_feature_cache_experimental: bool = True,
    prefix_feature_cache_train: bool = True,
    prefix_feature_cache_valid_quick: bool = True,
    prefix_feature_cache_valid_full: bool = False,
    trainable_lora_scope: str = "last_25_percent",
    lora_dropout: float = 0.0,
) -> TGLoRAConfig:
    """Build a minimal valid TGLoRAConfig for testing pure functions."""
    return TGLoRAConfig(
        experiment=ExperimentConfig(name="test", seed=42),
        model=ModelConfig(name_or_path="Qwen/Qwen3.5-9B"),
        lora=LoRAConfig(r=16, alpha=32, dropout=lora_dropout, target_modules="all-linear"),
        data=DataConfig(
            train_path="data/train.jsonl",
            valid_quick_path="data/valid_quick.jsonl",
            valid_full_path="data/valid_full.jsonl",
        ),
        training=TrainingConfig(
            batch_size=1,
            grad_accumulation=8,
            learning_rate=2e-4,
            max_cycles=10,
            trainable_lora_scope=trainable_lora_scope,
            prefix_feature_cache_experimental=prefix_feature_cache_experimental,
            prefix_feature_cache_train=prefix_feature_cache_train,
            prefix_feature_cache_valid_quick=prefix_feature_cache_valid_quick,
            prefix_feature_cache_valid_full=prefix_feature_cache_valid_full,
        ),
        eval=EvalConfig(),
        logging=LoggingConfig(run_dir="runs/test"),
        tg_lora=TGLoRAParams(
            K_initial=3,
            K_candidates=[3, 5, 7],
            N_initial=5,
            N_candidates=[5, 7],
            alpha_initial=0.3,
            alpha_min=0.1,
            alpha_max=1.0,
            beta_initial=0.8,
            beta_candidates=[0.5, 0.8, 0.9],
            relative_update_cap=0.5,
            active_layer_strategy="last_25_percent",
        ),
    )


# ---------------------------------------------------------------------------
# Import health
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.precompute_prefix_cache_parallel")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_normalize_devices")
        assert hasattr(mod, "_selected_labels")
        assert hasattr(mod, "_dataset_path")
        assert hasattr(mod, "_count_records")


# ---------------------------------------------------------------------------
# --help CLI
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.precompute_prefix_cache_parallel", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "precompute" in result.stdout.lower()


# ---------------------------------------------------------------------------
# _normalize_devices
# ---------------------------------------------------------------------------


class TestNormalizeDevices:
    def test_explicit_cuda_indices(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        result = _normalize_devices("cuda:0,cuda:1")
        assert result == ["cuda:0", "cuda:1"]

    def test_bare_numeric_indices(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        result = _normalize_devices("0,1,2")
        assert result == ["cuda:0", "cuda:1", "cuda:2"]

    def test_mixed_format(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        result = _normalize_devices("cuda:0,1")
        assert result == ["cuda:0", "cuda:1"]

    def test_single_device(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        result = _normalize_devices("cuda:3")
        assert result == ["cuda:3"]

    def test_whitespace_handling(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        result = _normalize_devices(" cuda:0 , cuda:1 ")
        assert result == ["cuda:0", "cuda:1"]

    def test_unsupported_token_raises(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        with pytest.raises(ValueError, match="Unsupported device token"):
            _normalize_devices("cpu")

    def test_empty_string_raises(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        with pytest.raises(ValueError, match="No CUDA devices"):
            _normalize_devices("")

    def test_commas_only_raises(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        with pytest.raises(ValueError, match="No CUDA devices"):
            _normalize_devices(",,")

    def test_auto_with_no_cuda_raises(self):
        from scripts.precompute_prefix_cache_parallel import _normalize_devices

        with patch("scripts.precompute_prefix_cache_parallel.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with pytest.raises(RuntimeError, match="CUDA is required"):
                _normalize_devices("auto")


# ---------------------------------------------------------------------------
# _selected_labels
# ---------------------------------------------------------------------------


class TestSelectedLabels:
    def test_auto_selects_enabled_labels(self):
        from scripts.precompute_prefix_cache_parallel import _selected_labels

        cfg = _make_config(
            prefix_feature_cache_train=True,
            prefix_feature_cache_valid_quick=True,
            prefix_feature_cache_valid_full=False,
        )
        result = _selected_labels(cfg, "auto")
        assert result == ["train", "valid_quick"]

    def test_auto_selects_all_enabled(self):
        from scripts.precompute_prefix_cache_parallel import _selected_labels

        cfg = _make_config(
            prefix_feature_cache_train=True,
            prefix_feature_cache_valid_quick=True,
            prefix_feature_cache_valid_full=True,
        )
        result = _selected_labels(cfg, "auto")
        assert "train" in result
        assert "valid_quick" in result
        assert "valid_full" in result

    def test_explicit_labels(self):
        from scripts.precompute_prefix_cache_parallel import _selected_labels

        cfg = _make_config()
        result = _selected_labels(cfg, "train,valid_full")
        assert result == ["train", "valid_full"]

    def test_invalid_label_raises(self):
        from scripts.precompute_prefix_cache_parallel import _selected_labels

        cfg = _make_config()
        with pytest.raises(ValueError, match="Unsupported dataset labels"):
            _selected_labels(cfg, "train,invalid_label")

    def test_whitespace_in_labels(self):
        from scripts.precompute_prefix_cache_parallel import _selected_labels

        cfg = _make_config()
        result = _selected_labels(cfg, " train , valid_quick ")
        assert result == ["train", "valid_quick"]


# ---------------------------------------------------------------------------
# _dataset_path
# ---------------------------------------------------------------------------


class TestDatasetPath:
    def test_train_label(self):
        from scripts.precompute_prefix_cache_parallel import _dataset_path

        cfg = _make_config()
        assert _dataset_path(cfg, "train") == "data/train.jsonl"

    def test_valid_quick_label(self):
        from scripts.precompute_prefix_cache_parallel import _dataset_path

        cfg = _make_config()
        assert _dataset_path(cfg, "valid_quick") == "data/valid_quick.jsonl"

    def test_valid_full_label(self):
        from scripts.precompute_prefix_cache_parallel import _dataset_path

        cfg = _make_config()
        assert _dataset_path(cfg, "valid_full") == "data/valid_full.jsonl"


# ---------------------------------------------------------------------------
# _count_records
# ---------------------------------------------------------------------------


class TestCountRecords:
    def test_counts_nonempty_lines(self, tmp_path):
        from scripts.precompute_prefix_cache_parallel import _count_records

        path = tmp_path / "data.jsonl"
        path.write_text('{"a":1}\n{"b":2}\n{"c":3}\n')
        assert _count_records(str(path)) == 3

    def test_empty_file(self, tmp_path):
        from scripts.precompute_prefix_cache_parallel import _count_records

        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert _count_records(str(path)) == 0

    def test_blank_lines_skipped(self, tmp_path):
        from scripts.precompute_prefix_cache_parallel import _count_records

        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a":1}\n\n\n{"b":2}\n')
        assert _count_records(str(path)) == 2

    def test_trailing_newline(self, tmp_path):
        from scripts.precompute_prefix_cache_parallel import _count_records

        path = tmp_path / "trailing.jsonl"
        path.write_text('{"a":1}\n{"b":2}\n\n')
        assert _count_records(str(path)) == 2

    def test_single_record(self, tmp_path):
        from scripts.precompute_prefix_cache_parallel import _count_records

        path = tmp_path / "one.jsonl"
        path.write_text('{"x":42}\n')
        assert _count_records(str(path)) == 1


# ---------------------------------------------------------------------------
# _resolve_split_layer — validation guard tests
# ---------------------------------------------------------------------------


class TestResolveSplitLayerValidation:
    def test_rejects_non_experimental(self):
        from scripts.precompute_prefix_cache_parallel import _resolve_split_layer

        cfg = _make_config(prefix_feature_cache_experimental=False)
        with pytest.raises(ValueError, match="prefix_feature_cache_experimental"):
            _resolve_split_layer(cfg)

    def test_rejects_non_last_25_percent_scope(self):
        from scripts.precompute_prefix_cache_parallel import _resolve_split_layer

        cfg = _make_config(trainable_lora_scope="all")
        with pytest.raises(ValueError, match="trainable_lora_scope"):
            _resolve_split_layer(cfg)

    def test_rejects_nonzero_dropout(self):
        from scripts.precompute_prefix_cache_parallel import _resolve_split_layer

        cfg = _make_config(lora_dropout=0.1)
        with pytest.raises(ValueError, match="dropout"):
            _resolve_split_layer(cfg)


# ---------------------------------------------------------------------------
# _build_worker_configs
# ---------------------------------------------------------------------------


def _mock_metadata(**overrides):
    base = {
        "format_version": 1,
        "dataset_path": "data/train.jsonl",
        "model_name": "Qwen/Qwen3.5-9B",
        "seed": 42,
        "max_seq_len": 2048,
        "split_layer_idx": 30,
    }
    base.update(overrides)
    return base


class TestBuildWorkerConfigs:
    """Tests for _build_worker_configs — orchestrates shard task creation."""

    @pytest.fixture()
    def _patch_deps(self):
        with patch("scripts.precompute_prefix_cache_parallel.build_prefix_feature_cache_metadata") as mock_meta, \
             patch("scripts.precompute_prefix_cache_parallel.get_prefix_feature_cache_path") as mock_path, \
             patch("scripts.precompute_prefix_cache_parallel._count_records") as mock_count:
            mock_meta.return_value = _mock_metadata()
            mock_path.return_value = Path("/cache/train_abc123.pt")
            mock_count.return_value = 100
            yield mock_meta, mock_path, mock_count

    def _call(self, cfg=None, devices=None, labels=None, force_rebuild=False, **kwargs):
        from scripts.precompute_prefix_cache_parallel import _build_worker_configs

        cfg = cfg or _make_config()
        devices = devices or ["cuda:0", "cuda:1"]
        labels = labels or ["train"]
        cache_dir = Path("/cache")
        shard_root = Path("/cache/.shards")
        return _build_worker_configs(
            cfg, Path("configs/test.yaml"), labels, devices,
            split_layer_idx=30, cache_dir=cache_dir, shard_root=shard_root,
            force_rebuild=force_rebuild,
        )

    def test_returns_worker_configs_and_dataset_plan(self, _patch_deps):
        worker_cfgs, dataset_plan = self._call()
        assert isinstance(worker_cfgs, list)
        assert isinstance(dataset_plan, dict)
        assert len(worker_cfgs) == 2

    def test_shard_tasks_split_across_devices(self, _patch_deps):
        worker_cfgs, _ = self._call()
        rank0_labels = [t.dataset_label for t in worker_cfgs[0].tasks]
        rank1_labels = [t.dataset_label for t in worker_cfgs[1].tasks]
        assert "train" in rank0_labels
        assert "train" in rank1_labels

    def test_shard_ranges_cover_all_records(self, _patch_deps):
        worker_cfgs, _ = self._call()
        ranges = [(t.start_idx, t.end_idx) for t in worker_cfgs[0].tasks + worker_cfgs[1].tasks]
        assert ranges[0] == (0, 50)
        assert ranges[1] == (50, 100)

    def test_multiple_labels_creates_tasks_per_rank(self, _patch_deps):
        mock_meta, _, _ = _patch_deps
        mock_meta.side_effect = [
            _mock_metadata(dataset_path="data/train.jsonl"),
            _mock_metadata(dataset_path="data/valid_quick.jsonl"),
        ]
        worker_cfgs, dataset_plan = self._call(labels=["train", "valid_quick"])
        assert len(dataset_plan) == 2
        assert "train" in dataset_plan
        assert "valid_quick" in dataset_plan
        for wc in worker_cfgs:
            assert len(wc.tasks) == 2

    def test_skip_existing_when_cache_present(self, _patch_deps, tmp_path):
        _, mock_path, _ = _patch_deps
        cache_path = tmp_path / "train_abc123.pt"
        cache_path.touch()
        mock_path.return_value = cache_path

        _, dataset_plan = self._call()
        assert dataset_plan["train"]["skip_existing"] is True

    def test_force_rebuild_ignores_existing_cache(self, _patch_deps, tmp_path):
        _, mock_path, _ = _patch_deps
        cache_path = tmp_path / "train_abc123.pt"
        cache_path.touch()
        mock_path.return_value = cache_path

        _, dataset_plan = self._call(force_rebuild=True)
        assert dataset_plan["train"]["skip_existing"] is False

    def test_worker_config_fields(self, _patch_deps):
        worker_cfgs, _ = self._call()
        wc = worker_cfgs[0]
        assert wc.batch_size == 1
        assert wc.max_seq_len == 2048
        assert wc.split_layer_idx == 30
        assert wc.device == "cuda:0"

    def test_dataset_plan_structure(self, _patch_deps):
        _, dataset_plan = self._call()
        plan = dataset_plan["train"]
        assert "dataset_path" in plan
        assert "metadata" in plan
        assert "cache_path" in plan
        assert "total_examples" in plan
        assert "skip_existing" in plan
        assert plan["total_examples"] == 100

    def test_single_device_all_records_in_one_shard(self, _patch_deps):
        worker_cfgs, _ = self._call(devices=["cuda:0"])
        assert len(worker_cfgs) == 1
        assert len(worker_cfgs[0].tasks) == 1
        task = worker_cfgs[0].tasks[0]
        assert task.start_idx == 0
        assert task.end_idx == 100

    def test_shard_path_per_rank(self, _patch_deps):
        worker_cfgs, _ = self._call()
        paths = [t.shard_path for t in worker_cfgs[0].tasks + worker_cfgs[1].tasks]
        assert any("rank_0" in p for p in paths)
        assert any("rank_1" in p for p in paths)
