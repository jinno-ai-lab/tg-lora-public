import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.batch_iter import InfiniteBatchIterator


def _make_loader(n: int = 8) -> DataLoader:
    xs = torch.randn(n, 4)
    ys = torch.randint(0, 2, (n,))
    ds = TensorDataset(xs, ys)
    return DataLoader(ds, batch_size=2)


class TestInfiniteBatchIteratorValidation:
    def test_rejects_non_dataloader(self):
        with pytest.raises(TypeError, match="loader must be a DataLoader instance"):
            InfiniteBatchIterator([1, 2, 3], device="cpu")

    def test_rejects_invalid_device_string(self):
        loader = _make_loader()
        with pytest.raises(ValueError, match="Invalid device"):
            InfiniteBatchIterator(loader, device="not_a_real_device_xyz")

    def test_accepts_string_device(self):
        loader = _make_loader()
        it = InfiniteBatchIterator(loader, device="cpu")
        assert it.device == "cpu"

    def test_accepts_torch_device(self):
        loader = _make_loader()
        it = InfiniteBatchIterator(loader, device=torch.device("cpu"))
        assert it.device == torch.device("cpu")

    def test_rejects_empty_dataset(self):
        empty_ds = TensorDataset(torch.randn(0, 4))
        loader = DataLoader(empty_ds, batch_size=1)
        with pytest.raises(ValueError, match="empty dataset"):
            InfiniteBatchIterator(loader, device="cpu")
