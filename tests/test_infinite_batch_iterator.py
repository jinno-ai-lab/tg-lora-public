"""Tests for InfiniteBatchIterator from src.training.batch_iter."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.batch_iter import InfiniteBatchIterator


class _DictDataset(Dataset):
    """Simple dataset yielding dicts with a single 'x' tensor."""

    def __init__(self, n: int):
        self.data = torch.arange(n, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"x": self.data[idx]}


class TestInfiniteBatchIteratorInit:
    def test_rejects_empty_dataset(self):
        loader = DataLoader(_DictDataset(0), batch_size=1)
        with pytest.raises(ValueError, match="empty dataset"):
            InfiniteBatchIterator(loader, device="cpu")


class TestInfiniteBatchIteratorIteration:
    def test_yields_dict_batches(self):
        loader = DataLoader(_DictDataset(4), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")
        batch = it.next()
        assert isinstance(batch, dict)
        assert "x" in batch
        assert batch["x"].shape == (2,)

    def test_wraps_around_after_epoch(self):
        loader = DataLoader(_DictDataset(4), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")

        batches = [it.next()["x"] for _ in range(4)]
        # 2 batches per epoch (4 items / batch_size 2), so 4 calls = 2 full epochs
        for b in batches:
            assert b.shape == (2,)

    def test_moves_tensors_to_cpu(self):
        loader = DataLoader(_DictDataset(4), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")
        batch = it.next()
        assert batch["x"].device.type == "cpu"

    def test_non_tensor_values_passed_through(self):
        class _MixedDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {"x": torch.tensor(idx, dtype=torch.float32), "label": "pos"}

        loader = DataLoader(_MixedDataset(), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")
        batch = it.next()

        assert isinstance(batch["x"], torch.Tensor)
        # DataLoader collates strings into a list
        assert batch["label"] == ["pos", "pos"]

    def test_infinite_iteration_stays_consistent(self):
        loader = DataLoader(_DictDataset(3), batch_size=1)
        it = InfiniteBatchIterator(loader, device="cpu")

        seen = []
        for _ in range(9):  # 3 epochs worth
            val = it.next()["x"].item()
            seen.append(val)

        # Pattern should repeat: 0, 1, 2, 0, 1, 2, 0, 1, 2
        expected = [0.0, 1.0, 2.0] * 3
        assert seen == expected

    def test_single_item_dataset(self):
        loader = DataLoader(_DictDataset(1), batch_size=1)
        it = InfiniteBatchIterator(loader, device="cpu")

        for _ in range(5):
            batch = it.next()
            assert batch["x"].item() == 0.0

    def test_stop_iteration_resets_iterator(self):
        """StopIteration 発生後に self._iter が新しく作成されていることを直接検証。"""
        loader = DataLoader(_DictDataset(3), batch_size=1)
        it = InfiniteBatchIterator(loader, device="cpu")

        old_iter = it._iter
        # Drain the epoch (3 items, batch_size=1 → 3 calls exhausts the iterator)
        for _ in range(3):
            it.next()
        # The 4th call triggers StopIteration internally and resets self._iter
        it.next()

        assert it._iter is not old_iter

    def test_first_batch_after_reset_matches_first_epoch(self):
        """StopIteration 後の最初のバッチが新エポックの先頭バッチと同一であることを検証。"""
        loader = DataLoader(_DictDataset(4), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")

        first_batch = it.next()  # epoch 0, batch 0
        # Drain remaining batches in epoch 0 (4 items / batch_size 2 = 2 batches)
        it.next()  # epoch 0, batch 1
        # Next call triggers StopIteration → reset → epoch 1, batch 0
        first_batch_new_epoch = it.next()

        assert torch.equal(first_batch["x"], first_batch_new_epoch["x"])

    def test_advance_skips_batches(self):
        loader = DataLoader(_DictDataset(5), batch_size=1)
        it = InfiniteBatchIterator(loader, device="cpu")

        it.advance(3)
        batch = it.next()

        assert batch["x"].item() == 3.0

    def test_advance_wraps_epochs(self):
        loader = DataLoader(_DictDataset(3), batch_size=1)
        it = InfiniteBatchIterator(loader, device="cpu")

        it.advance(5)
        batch = it.next()

        assert batch["x"].item() == 2.0

    def test_advance_rejects_negative_values(self):
        loader = DataLoader(_DictDataset(3), batch_size=1)
        it = InfiniteBatchIterator(loader, device="cpu")

        with pytest.raises(ValueError, match="non-negative"):
            it.advance(-1)


class TestSingleBatchDataloader:
    """Criterion 1: single-batch (batch_size == dataset_size) infinite iteration."""

    def test_single_batch_repeats_identically(self):
        """When dataset size == batch size, only 1 batch per epoch; it should repeat."""
        n = 4
        loader = DataLoader(_DictDataset(n), batch_size=n)
        it = InfiniteBatchIterator(loader, device="cpu")

        batches = [it.next()["x"] for _ in range(5)]
        for b in batches[1:]:
            assert torch.equal(batches[0], b)

    def test_single_batch_stopiteration_reset(self):
        """StopIteration→reset cycle works when there is exactly 1 batch per epoch."""
        loader = DataLoader(_DictDataset(4), batch_size=4)
        it = InfiniteBatchIterator(loader, device="cpu")

        old_iter = it._iter
        it.next()  # exhausts the only batch
        it.next()  # triggers StopIteration internally → reset
        assert it._iter is not old_iter


class TestDeviceCastEdgeCases:
    """Criterion 2: device cast with str and torch.device object."""

    def test_device_string_cast(self):
        loader = DataLoader(_DictDataset(4), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")
        batch = it.next()
        assert batch["x"].device.type == "cpu"

    def test_device_object_cast(self):
        loader = DataLoader(_DictDataset(4), batch_size=2)
        it = InfiniteBatchIterator(loader, device=torch.device("cpu"))
        batch = it.next()
        assert batch["x"].device.type == "cpu"

    def test_multi_key_device_cast(self):
        """All tensor values in a multi-key batch are moved to the target device."""

        class _MultiKeyDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([idx, idx + 1], dtype=torch.long),
                    "labels": torch.tensor(idx, dtype=torch.float32),
                }

        loader = DataLoader(_MultiKeyDataset(), batch_size=2)
        it = InfiniteBatchIterator(loader, device=torch.device("cpu"))
        batch = it.next()

        assert batch["input_ids"].device.type == "cpu"
        assert batch["labels"].device.type == "cpu"


class TestDtypePreservation:
    """Criterion 3: dtype is preserved through .to(device) cast."""

    def test_float16_dtype_preserved(self):

        class _F16Dataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {"x": torch.tensor(idx, dtype=torch.float16)}

        loader = DataLoader(_F16Dataset(), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")
        batch = it.next()
        assert batch["x"].dtype == torch.float16

    def test_int64_dtype_preserved(self):

        class _Int64Dataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {"ids": torch.tensor([idx, idx + 1], dtype=torch.int64)}

        loader = DataLoader(_Int64Dataset(), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")
        batch = it.next()
        assert batch["ids"].dtype == torch.int64

    def test_float32_dtype_preserved(self):

        class _F32Dataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {"x": torch.tensor(idx, dtype=torch.float32)}

        loader = DataLoader(_F32Dataset(), batch_size=2)
        it = InfiniteBatchIterator(loader, device="cpu")
        batch = it.next()
        assert batch["x"].dtype == torch.float32
