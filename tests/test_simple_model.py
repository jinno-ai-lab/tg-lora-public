"""Tests for the shared SimpleLoRAModel (scripts/simple_model.py)."""
from __future__ import annotations

import torch

from scripts.simple_model import SimpleLoRAModel


class TestSimpleLoRAModel:
    def test_forward_shape(self):
        model = SimpleLoRAModel(num_layers=4, dim=4)
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 4)

    def test_has_lora_params(self):
        model = SimpleLoRAModel(num_layers=4, dim=4)
        lora_params = [
            name for name, _ in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        assert len(lora_params) == 8

    def test_requires_grad(self):
        model = SimpleLoRAModel(num_layers=4, dim=4)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) == 8

    def test_output_is_finite(self):
        model = SimpleLoRAModel(num_layers=2, dim=8)
        x = torch.randn(4, 8)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_custom_dimensions(self):
        model = SimpleLoRAModel(num_layers=6, dim=16)
        x = torch.randn(3, 16)
        out = model(x)
        assert out.shape == (3, 16)
