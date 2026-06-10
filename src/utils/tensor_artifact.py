from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_tensor_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    mmap: bool | None = None,
) -> Any:
    """Load a tensor-only torch artifact with PyTorch's safe loader mode.

    This should be used only for blobs composed of tensors plus builtin Python
    containers / primitives. It intentionally relies on ``weights_only=True``
    to avoid arbitrary object deserialization.
    """

    load_kwargs: dict[str, Any] = {
        "map_location": map_location,
        "weights_only": True,
    }
    if mmap is not None:
        load_kwargs["mmap"] = mmap
    return torch.load(Path(path), **load_kwargs)