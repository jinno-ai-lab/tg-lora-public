import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


class TinyModel(nn.Module):
    """Minimal transformer-like model whose `.layers` ModuleList is
    discoverable by `_get_decoder_layers`."""

    def __init__(self, vocab: int = 32, hidden: int = 16, layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        del kw
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return type("Out", (), {"loss": loss})()


class TokenDataset(Dataset):
    def __init__(self, n: int = 6, seq_len: int = 8, vocab: int = 32):
        self.input_ids = torch.randint(0, vocab, (n, seq_len))
        self.attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


@pytest.fixture()
def tiny_model():
    model = TinyModel()
    model.eval()
    return model


@pytest.fixture()
def tiny_model_factory():
    def factory(cfg):
        del cfg
        model = TinyModel()
        model.eval()
        return model, torch.device("cpu")

    return factory


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


@pytest.fixture(autouse=True)
def mock_transformers_tokenizer():
    """Autouse fixture to intercept GPT-2 tokenizer downloads and mock with a fast dummy."""
    import transformers
    from unittest.mock import patch

    orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

    class DummyTokenizer:
        def __init__(self):
            self.pad_token = "<|endoftext|>"
            self.eos_token = "<|endoftext|>"
            self.pad_token_id = 50256
            self.eos_token_id = 50256
            self.vocab_size = 50257

        def __call__(self, text, max_length=None, truncation=False, padding=False, return_tensors=None, return_offsets_mapping=False, **kwargs):
            import torch
            if isinstance(text, list):
                input_ids_list = []
                attention_mask_list = []
                offsets_list = []
                for t in text:
                    res_single = self._encode_single(t, max_length, truncation, padding, return_offsets_mapping)
                    input_ids_list.append(res_single["input_ids"])
                    attention_mask_list.append(res_single["attention_mask"])
                    if return_offsets_mapping:
                        offsets_list.append(res_single["offset_mapping"])
                
                res = {
                    "input_ids": torch.stack(input_ids_list) if return_tensors == "pt" else input_ids_list,
                    "attention_mask": torch.stack(attention_mask_list) if return_tensors == "pt" else attention_mask_list,
                }
                if return_offsets_mapping:
                    res["offset_mapping"] = torch.stack(offsets_list) if return_tensors == "pt" else offsets_list
                return res
            else:
                res_single = self._encode_single(text, max_length, truncation, padding, return_offsets_mapping)
                res = {
                    "input_ids": res_single["input_ids"].unsqueeze(0) if return_tensors == "pt" else res_single["input_ids"],
                    "attention_mask": res_single["attention_mask"].unsqueeze(0) if return_tensors == "pt" else res_single["attention_mask"],
                }
                if return_offsets_mapping:
                    res["offset_mapping"] = res_single["offset_mapping"].unsqueeze(0) if return_tensors == "pt" else res_single["offset_mapping"]
                return res

        def _encode_single(self, text, max_length, truncation, padding, return_offsets_mapping):
            import torch
            words = text.split()
            num_tokens = len(words)
            if max_length is not None and truncation and num_tokens > max_length:
                num_tokens = max_length
            
            input_ids = list(range(10, 10 + num_tokens))
            attention_mask = [1] * num_tokens
            
            if padding == "max_length" and max_length is not None:
                pad_len = max_length - num_tokens
                input_ids.extend([self.pad_token_id] * pad_len)
                attention_mask.extend([0] * pad_len)
                num_tokens = max_length
                
            res = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
            
            if return_offsets_mapping:
                offsets = []
                curr = 0
                for w in words:
                    offsets.append((curr, curr + len(w)))
                    curr += len(w) + 1
                offsets = offsets[:max_length]
                if len(offsets) < num_tokens:
                    offsets.extend([(0, 0)] * (num_tokens - len(offsets)))
                res["offset_mapping"] = torch.tensor(offsets)
            return res

        def decode(self, token_ids, skip_special_tokens=False, **kwargs):
            return "decoded_text"

    dummy_tok = DummyTokenizer()

    def mocked_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if "gpt2" in pretrained_model_name_or_path:
            return dummy_tok
        return orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=mocked_from_pretrained):
        yield
