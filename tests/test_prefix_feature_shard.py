"""Unit tests for compute_prefix_feature_shard_ranges and merge_prefix_feature_cache_shards.

Covers edge cases identified as gaps: empty input, single shard, uneven splits,
format version mismatch, inconsistent split_layer_idx, mixed position_ids presence.
"""


import pytest
import torch

from src.tg_lora.prefix_feature_cache import (
    PrefixFeatureDataset, PrefixFeatureExample,
    compute_prefix_feature_shard_ranges, merge_prefix_feature_cache_shards,
    save_prefix_feature_dataset)


def _make_dataset(n: int, split_layer_idx: int = 2, with_position_ids: bool = True):
    examples = []
    for i in range(n):
        examples.append(
            PrefixFeatureExample(
                hidden_states=torch.randn(4, 8),
                attention_mask=torch.ones(4, dtype=torch.long),
                labels=torch.full((4,), i, dtype=torch.long),
                split_layer_idx=split_layer_idx,
                position_ids=torch.arange(4, dtype=torch.long) if with_position_ids else None,
            )
        )
    return PrefixFeatureDataset(examples)


# ---------------------------------------------------------------------------
# compute_prefix_feature_shard_ranges
# ---------------------------------------------------------------------------


class TestComputeShardRanges:
    def test_zero_examples_returns_empty(self):
        assert compute_prefix_feature_shard_ranges(0, 4) == []

    def test_single_example_single_shard(self):
        result = compute_prefix_feature_shard_ranges(1, 1)
        assert result == [(0, 1)]

    def test_single_example_multiple_shards_capped(self):
        result = compute_prefix_feature_shard_ranges(1, 8)
        assert len(result) == 1
        assert result == [(0, 1)]

    def test_even_split(self):
        result = compute_prefix_feature_shard_ranges(12, 3)
        assert result == [(0, 4), (4, 8), (8, 12)]

    def test_uneven_split_distributes_remainder_to_first_shards(self):
        result = compute_prefix_feature_shard_ranges(10, 3)
        # 10 / 3 = 3 rem 1 → first shard gets +1
        assert result == [(0, 4), (4, 7), (7, 10)]

    def test_uneven_split_larger_remainder(self):
        result = compute_prefix_feature_shard_ranges(10, 4)
        # 10 / 4 = 2 rem 2 → first 2 shards get +1
        assert result == [(0, 3), (3, 6), (6, 8), (8, 10)]

    def test_total_coverage(self):
        for total, shards in [(17, 5), (1, 10), (100, 7), (3, 3)]:
            ranges = compute_prefix_feature_shard_ranges(total, shards)
            covered = sum(end - start for start, end in ranges)
            assert covered == total
            assert ranges[0][0] == 0
            assert ranges[-1][1] == total

    def test_no_gaps_or_overlaps(self):
        ranges = compute_prefix_feature_shard_ranges(23, 6)
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="total_examples must be >= 0"):
            compute_prefix_feature_shard_ranges(-1, 4)

    def test_zero_shard_count_raises(self):
        with pytest.raises(ValueError, match="shard_count must be >= 1"):
            compute_prefix_feature_shard_ranges(10, 0)

    def test_negative_shard_count_raises(self):
        with pytest.raises(ValueError, match="shard_count must be >= 1"):
            compute_prefix_feature_shard_ranges(10, -2)


# ---------------------------------------------------------------------------
# merge_prefix_feature_cache_shards
# ---------------------------------------------------------------------------


class TestMergeShards:
    def test_merge_two_shards(self, tmp_path):
        ds1 = _make_dataset(3, split_layer_idx=2)
        ds2 = _make_dataset(2, split_layer_idx=2)
        p1 = tmp_path / "shard1.pt"
        p2 = tmp_path / "shard2.pt"
        out = tmp_path / "merged.pt"

        save_prefix_feature_dataset(ds1, p1, metadata={"id": "s1"})
        save_prefix_feature_dataset(ds2, p2, metadata={"id": "s2"})

        merge_prefix_feature_cache_shards([p1, p2], out, metadata={"id": "merged"})

        blob = torch.load(out, map_location="cpu", weights_only=True)
        assert blob["hidden_states"].shape[0] == 5
        assert blob["metadata"]["id"] == "merged"
        assert blob["split_layer_idx"] == 2

    def test_merge_single_shard(self, tmp_path):
        ds = _make_dataset(4, split_layer_idx=3)
        p1 = tmp_path / "shard1.pt"
        out = tmp_path / "merged.pt"

        save_prefix_feature_dataset(ds, p1, metadata={"id": "s1"})
        merge_prefix_feature_cache_shards([p1], out, metadata={"id": "merged"})

        blob = torch.load(out, map_location="cpu", weights_only=True)
        assert blob["hidden_states"].shape[0] == 4

    def test_merge_preserves_position_ids(self, tmp_path):
        ds1 = _make_dataset(2, with_position_ids=True)
        ds2 = _make_dataset(3, with_position_ids=True)
        p1 = tmp_path / "shard1.pt"
        p2 = tmp_path / "shard2.pt"
        out = tmp_path / "merged.pt"

        save_prefix_feature_dataset(ds1, p1, metadata={})
        save_prefix_feature_dataset(ds2, p2, metadata={})

        merge_prefix_feature_cache_shards([p1, p2], out, metadata={})

        blob = torch.load(out, map_location="cpu", weights_only=True)
        assert blob["position_ids"] is not None
        assert blob["position_ids"].shape[0] == 5

    def test_merge_without_position_ids(self, tmp_path):
        ds1 = _make_dataset(2, with_position_ids=False)
        ds2 = _make_dataset(2, with_position_ids=False)
        p1 = tmp_path / "shard1.pt"
        p2 = tmp_path / "shard2.pt"
        out = tmp_path / "merged.pt"

        save_prefix_feature_dataset(ds1, p1, metadata={})
        save_prefix_feature_dataset(ds2, p2, metadata={})

        merge_prefix_feature_cache_shards([p1, p2], out, metadata={})

        blob = torch.load(out, map_location="cpu", weights_only=True)
        assert blob["position_ids"] is None

    def test_empty_shard_paths_raises(self, tmp_path):
        out = tmp_path / "merged.pt"
        with pytest.raises(ValueError, match="shard_paths must not be empty"):
            merge_prefix_feature_cache_shards([], out, metadata={})

    def test_format_version_mismatch_raises(self, tmp_path):
        ds = _make_dataset(2)
        good_path = tmp_path / "good.pt"
        save_prefix_feature_dataset(ds, good_path, metadata={})

        bad_path = tmp_path / "bad.pt"
        torch.save({"format_version": 999, "split_layer_idx": 2, "position_ids": None}, bad_path)

        out = tmp_path / "merged.pt"
        with pytest.raises(ValueError, match="Unsupported prefix feature cache format version"):
            merge_prefix_feature_cache_shards([good_path, bad_path], out, metadata={})

    def test_inconsistent_split_layer_idx_raises(self, tmp_path):
        ds1 = _make_dataset(2, split_layer_idx=2)
        ds2 = _make_dataset(2, split_layer_idx=5)
        p1 = tmp_path / "shard1.pt"
        p2 = tmp_path / "shard2.pt"
        out = tmp_path / "merged.pt"

        save_prefix_feature_dataset(ds1, p1, metadata={})
        save_prefix_feature_dataset(ds2, p2, metadata={})

        with pytest.raises(ValueError, match="same split_layer_idx"):
            merge_prefix_feature_cache_shards([p1, p2], out, metadata={})

    def test_mixed_position_ids_presence_raises(self, tmp_path):
        ds_with = _make_dataset(2, with_position_ids=True)
        ds_without = _make_dataset(2, with_position_ids=False)
        p1 = tmp_path / "shard1.pt"
        p2 = tmp_path / "shard2.pt"
        out = tmp_path / "merged.pt"

        save_prefix_feature_dataset(ds_with, p1, metadata={})
        save_prefix_feature_dataset(ds_without, p2, metadata={})

        with pytest.raises(ValueError, match="position_ids presence"):
            merge_prefix_feature_cache_shards([p1, p2], out, metadata={})

    def test_merge_three_shards_tensor_concatenation(self, tmp_path):
        ds1 = _make_dataset(2, split_layer_idx=2)
        ds2 = _make_dataset(3, split_layer_idx=2)
        ds3 = _make_dataset(1, split_layer_idx=2)
        paths = []
        for i, ds in enumerate([ds1, ds2, ds3], 1):
            p = tmp_path / f"shard{i}.pt"
            save_prefix_feature_dataset(ds, p, metadata={})
            paths.append(p)

        out = tmp_path / "merged.pt"
        merge_prefix_feature_cache_shards(paths, out, metadata={"merged": True})

        blob = torch.load(out, map_location="cpu", weights_only=True)
        assert blob["hidden_states"].shape[0] == 6
        assert blob["labels"].shape[0] == 6
        assert blob["attention_mask"].shape[0] == 6
        assert blob["position_ids"].shape[0] == 6
        assert blob["metadata"]["merged"] is True
