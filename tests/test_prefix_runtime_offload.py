import pytest
import torch
import torch.nn as nn

from src.tg_lora.prefix_runtime_offload import offload_prefix_runtime_to_cpu


class _RecordingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2))
        self.moves: list[str] = []

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        self.moves.append(str(device))
        return super().to(*args, **kwargs)


class _FakeInner(nn.Module):
    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.embed_tokens = _RecordingModule()
        self.layers = nn.ModuleList(_RecordingModule() for _ in range(num_layers))
        self.norm = _RecordingModule()


class _FakeWrappedModel(nn.Module):
    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.model = _FakeInner(num_layers=num_layers)
        self.lm_head = _RecordingModule()


def test_offload_prefix_runtime_moves_embeddings_and_prefix_layers() -> None:
    model = _FakeWrappedModel(num_layers=4)

    summary = offload_prefix_runtime_to_cpu(model, split_layer_idx=2)

    assert model.model.model.embed_tokens.moves == ["cpu"]
    assert model.model.model.layers[0].moves == ["cpu"]
    assert model.model.model.layers[1].moves == ["cpu"]
    assert model.model.model.layers[2].moves == []
    assert model.model.model.layers[3].moves == []
    assert summary["offloaded_prefix_modules"] == 3
    assert summary["offloaded_prefix_input_embeddings"] is True
    assert summary["split_layer_idx"] == 2


def test_offload_prefix_runtime_rejects_invalid_split_layer() -> None:
    model = _FakeWrappedModel(num_layers=4)

    with pytest.raises(ValueError, match="split_layer_idx"):
        offload_prefix_runtime_to_cpu(model, split_layer_idx=0)