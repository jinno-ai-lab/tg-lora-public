"""Simple LoRA model for testing and demonstration.

Shared between train_tg_lora.py and benchmark.py to avoid duplication.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLoRAModel(nn.Module):
    """Minimal model with LoRA-style parameters for testing.

    Each layer has a ``lora_A`` (trainable) and ``lora_B`` (trainable) pair
    whose product forms the weight matrix used in the forward pass.
    """

    def __init__(self, num_layers: int = 4, dim: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Module()
            layer.self_attn.q_proj.lora_A = nn.Parameter(torch.randn(dim, dim) * 0.01)
            layer.self_attn.q_proj.lora_B = nn.Parameter(torch.zeros(dim, dim))
            layer.self_attn.q_proj.lora_A.requires_grad_(True)
            layer.self_attn.q_proj.lora_B.requires_grad_(True)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            w = layer.self_attn.q_proj.lora_A @ layer.self_attn.q_proj.lora_B
            x = x @ w.T
        return x
