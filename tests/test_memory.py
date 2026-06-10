from unittest.mock import patch

import torch
import torch.nn as nn

from src.utils.memory import count_parameters, vram_usage_mb


class TestCountParameters:
    def test_all_trainable(self):
        model = nn.Linear(4, 2, bias=True)
        result = count_parameters(model)
        assert result["trainable"] == 10  # 4*2 weight + 2 bias
        assert result["total"] == 10

    def test_frozen_params(self):
        model = nn.Linear(4, 2, bias=True)
        model.weight.requires_grad = False
        result = count_parameters(model)
        assert result["trainable"] == 2  # only bias
        assert result["total"] == 10

    def test_no_params(self):
        model = nn.ReLU()
        result = count_parameters(model)
        assert result["total"] == 0
        assert result["trainable"] == 0


class TestVramUsageMb:
    def test_returns_dict(self):
        result = vram_usage_mb()
        assert isinstance(result, dict)

    @patch("src.utils.memory.detect_device", return_value=torch.device("cpu"))
    def test_cpu_returns_empty(self, _mock):
        assert vram_usage_mb() == {}

    @patch("src.utils.memory.gpu_memory_reserved_mb", return_value=100.0)
    @patch("src.utils.memory.gpu_memory_allocated_mb", return_value=50.0)
    @patch("src.utils.memory.detect_device", return_value=torch.device("mps"))
    @patch.object(torch.cuda, "device_count", return_value=0)
    def test_mps_returns_allocated(self, _cnt, _dev, _alloc, _res):
        result = vram_usage_mb()
        assert "gpu0_allocated_mb" in result
        assert result["gpu0_allocated_mb"] == 50.0

    @patch.object(torch.cuda, "memory_reserved", return_value=1024 * 1024 * 100)
    @patch.object(torch.cuda, "memory_allocated", return_value=1024 * 1024 * 50)
    @patch.object(torch.cuda, "device_count", return_value=1)
    @patch("src.utils.memory.detect_device", return_value=torch.device("cuda:0"))
    def test_single_gpu_cuda(self, _dev, _cnt, _alloc, _res):
        result = vram_usage_mb()
        assert "gpu0_allocated_mb" in result
        assert "gpu0_reserved_mb" in result
        assert result["gpu0_allocated_mb"] == 50.0
        assert result["gpu0_reserved_mb"] == 100.0
