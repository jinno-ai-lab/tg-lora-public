from collections.abc import Sized

import torch
from torch.utils.data import DataLoader


class InfiniteBatchIterator:
    """Wraps a DataLoader to yield batches forever, re-creating the iterator on epoch end."""

    def __init__(self, loader: DataLoader, device: torch.device | str) -> None:
        if not isinstance(loader, DataLoader):
            raise TypeError(
                f"loader must be a DataLoader instance, got {type(loader).__name__}"
            )
        dataset = loader.dataset
        if not isinstance(dataset, Sized) or len(dataset) == 0:
            raise ValueError("Cannot create InfiniteBatchIterator from empty dataset")
        try:
            torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid device: {device!r}") from exc
        self.loader = loader
        self.device = device
        self._iter = iter(loader)

    def next(self) -> dict[str, torch.Tensor]:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            batch = next(self._iter)
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def advance(self, batches: int) -> None:
        if batches < 0:
            raise ValueError(f"batches must be non-negative, got {batches}")
        for _ in range(batches):
            try:
                next(self._iter)
            except StopIteration:
                self._iter = iter(self.loader)
                next(self._iter)
