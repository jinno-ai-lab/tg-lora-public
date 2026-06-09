import os

import pytest
import torch
import torch.nn as nn


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")
    config.addinivalue_line(
        "markers", "network: requires network access (downloads models)"
    )
    config.addinivalue_line(
        "markers", "slow: slow-running test (skipped in CI by default)"
    )
    # Suppress DeprecationWarnings from third-party internals (torch, faiss/swig)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:builtin type .* has no __module__ attribute:DeprecationWarning",
    )


def pytest_collection_modifyitems(config, items):
    is_ci = os.environ.get("CI") == "1"

    if is_ci:
        skip_gpu = pytest.mark.skip(reason="GPU not available in CI")
        skip_slow = pytest.mark.skip(reason="slow test skipped in CI")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not torch_cuda_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def torch_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared fixtures for AsyncCacheBuilder tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared fixtures for LoRA state / extrapolator / rollback tests
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Minimal LoRA-enabled linear layer for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)


class FakeLoRAModel(nn.Module):
    """Minimal model with one LoRA-enabled linear layer."""

    def __init__(self):
        super().__init__()
        self.linear = LoRALinear(4, 4)


@pytest.fixture()
def fake_lora_model():
    return FakeLoRAModel()
