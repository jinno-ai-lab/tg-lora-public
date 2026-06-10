from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from src.model.load_model import (_prepare_model_for_qlora_training,
                                  _resolve_dtype, apply_lora, build_bnb_config,
                                  get_device_map, get_input_device,
                                  load_base_model, load_tokenizer)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**model_overrides):
    base = {
        "model": {
            "name_or_path": "gpt2",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bf16",
        },
    }
    base["model"].update(model_overrides)
    return OmegaConf.create(base)


def _full_cfg(**model_overrides):
    """Build a config with model + training + lora sections."""
    model = {
        "name_or_path": "gpt2",
        "load_in_4bit": False,
        "dtype": "bf16",
    }
    model.update(model_overrides)
    return OmegaConf.create(
        {
            "model": model,
            "training": {
                "gradient_checkpointing": True,
            },
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
            },
        }
    )


# ---------------------------------------------------------------------------
# build_bnb_config
# ---------------------------------------------------------------------------


def test_build_bnb_config_returns_config_when_4bit_enabled():
    cfg = _cfg(load_in_4bit=True)
    bnb = build_bnb_config(cfg)
    assert bnb is not None
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_compute_dtype == torch.bfloat16
    assert bnb.bnb_4bit_use_double_quant is True


def test_build_bnb_config_returns_none_when_4bit_disabled():
    cfg = _cfg(load_in_4bit=False)
    assert build_bnb_config(cfg) is None


def test_build_bnb_config_default_values():
    cfg = OmegaConf.create({"model": {}})
    bnb = build_bnb_config(cfg)
    assert bnb is None  # load_in_4bit defaults to False


def test_build_bnb_config_fp16_compute_dtype():
    cfg = _cfg(bnb_4bit_compute_dtype="fp16")
    bnb = build_bnb_config(cfg)
    assert bnb.bnb_4bit_compute_dtype == torch.float16


def test_build_bnb_config_fp32_compute_dtype():
    cfg = _cfg(bnb_4bit_compute_dtype="fp32")
    bnb = build_bnb_config(cfg)
    assert bnb.bnb_4bit_compute_dtype == torch.float32


def test_build_bnb_config_quant_type():
    cfg = _cfg(bnb_4bit_quant_type="fp4")
    bnb = build_bnb_config(cfg)
    assert bnb.bnb_4bit_quant_type == "fp4"


# ---------------------------------------------------------------------------
# _resolve_dtype
# ---------------------------------------------------------------------------


def test_resolve_dtype_fp16():
    assert _resolve_dtype("fp16") == torch.float16


def test_resolve_dtype_float16_alias():
    assert _resolve_dtype("float16") == torch.float16


def test_resolve_dtype_bf16():
    assert _resolve_dtype("bf16") == torch.bfloat16


def test_resolve_dtype_bfloat16_alias():
    assert _resolve_dtype("bfloat16") == torch.bfloat16


def test_resolve_dtype_fp32():
    assert _resolve_dtype("fp32") == torch.float32


def test_resolve_dtype_float32_alias():
    assert _resolve_dtype("float32") == torch.float32


def test_resolve_dtype_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _resolve_dtype("unknown")


# ---------------------------------------------------------------------------
# get_device_map
# ---------------------------------------------------------------------------


def test_get_device_map_with_device_map_set():
    cfg = OmegaConf.create({"model": {"device_map": "auto"}})
    assert get_device_map(cfg) == "auto"


def test_get_device_map_without_device_map_returns_device():
    cfg = OmegaConf.create({"model": {"device": "cuda:1"}})
    assert get_device_map(cfg) == {"": "cuda:1"}


def test_get_device_map_defaults_to_auto():
    cfg = OmegaConf.create({"model": {}})
    from src.utils.device import detect_device
    detected = detect_device()
    if detected.type in ("cuda", "mps"):
        assert get_device_map(cfg) == {"": str(detected)}
    else:
        assert get_device_map(cfg) == "cpu"


# ---------------------------------------------------------------------------
# get_input_device
# ---------------------------------------------------------------------------


def test_get_input_device_returns_model_parameter_device():
    model = MagicMock()
    param = MagicMock()
    param.device = torch.device("cuda:0")
    model.parameters.return_value = iter([param])
    assert get_input_device(model) == torch.device("cuda:0")


def test_get_input_device_cpu():
    model = MagicMock()
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    assert get_input_device(model) == torch.device("cpu")


# ---------------------------------------------------------------------------
# load_tokenizer
# ---------------------------------------------------------------------------


@patch("src.model.load_model.AutoTokenizer")
def test_load_tokenizer_calls_from_pretrained(mock_auto_tok):
    mock_tok = MagicMock()
    mock_tok.pad_token = "<pad>"
    mock_auto_tok.from_pretrained.return_value = mock_tok

    cfg = OmegaConf.create({"model": {"name_or_path": "gpt2"}})
    result = load_tokenizer(cfg)

    mock_auto_tok.from_pretrained.assert_called_once_with("gpt2")
    assert result is mock_tok


@patch("src.model.load_model.AutoTokenizer")
def test_load_tokenizer_sets_pad_token_when_none(mock_auto_tok):
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "</s>"
    mock_auto_tok.from_pretrained.return_value = mock_tok

    cfg = OmegaConf.create({"model": {"name_or_path": "gpt2"}})
    result = load_tokenizer(cfg)

    assert result.pad_token == "</s>"


@patch("src.model.load_model.AutoTokenizer")
def test_load_tokenizer_preserves_existing_pad_token(mock_auto_tok):
    mock_tok = MagicMock()
    mock_tok.pad_token = "<pad>"
    mock_auto_tok.from_pretrained.return_value = mock_tok

    cfg = OmegaConf.create({"model": {"name_or_path": "gpt2"}})
    result = load_tokenizer(cfg)

    # pad_token should remain unchanged
    assert result.pad_token == "<pad>"
    # Should NOT have set it to eos_token
    mock_tok.__setattr__  # just confirm mock is intact


# ---------------------------------------------------------------------------
# apply_lora
# ---------------------------------------------------------------------------


@patch("src.model.load_model.get_peft_model")
@patch("src.model.load_model._prepare_model_for_qlora_training")
@patch("src.model.load_model.LoraConfig")
def test_apply_lora_creates_config_and_calls_get_peft_model(
    mock_lora_cfg_cls, mock_prepare_kbit, mock_get_peft
):
    mock_model = MagicMock()
    mock_prepare_kbit.return_value = mock_model
    mock_lora_model = MagicMock()
    mock_get_peft.return_value = mock_lora_model
    mock_lora_cfg_instance = MagicMock()
    mock_lora_cfg_cls.return_value = mock_lora_cfg_instance

    cfg = _full_cfg(load_in_4bit=True)
    result = apply_lora(mock_model, cfg)

    mock_prepare_kbit.assert_called_once_with(
        mock_model,
        use_gradient_checkpointing=True,
    )

    mock_lora_cfg_cls.assert_called_once_with(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    mock_get_peft.assert_called_once_with(mock_model, mock_lora_cfg_instance)
    mock_lora_model.print_trainable_parameters.assert_called_once()
    assert result is mock_lora_model


# ---------------------------------------------------------------------------
# load_base_model
# ---------------------------------------------------------------------------


@patch("src.model.load_model._get_model_class")
def test_load_base_model_basic_flow_no_4bit(mock_get_cls):
    mock_model = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model
    mock_get_cls.return_value = mock_cls

    from src.utils.device import detect_device, resolve_compute_dtype
    detected = detect_device()
    if detected.type in ("cuda", "mps"):
        expected_device_map = {"": str(detected)}
    else:
        expected_device_map = "cpu"
    expected_dtype = resolve_compute_dtype(detect_device(), "bf16")

    cfg = _full_cfg(load_in_4bit=False, dtype="bf16")
    result = load_base_model(cfg)

    mock_cls.from_pretrained.assert_called_once()
    call_kwargs = mock_cls.from_pretrained.call_args[1]
    assert call_kwargs["quantization_config"] is None
    assert call_kwargs["torch_dtype"] == expected_dtype
    assert call_kwargs["device_map"] == expected_device_map
    assert call_kwargs["low_cpu_mem_usage"] is True
    mock_model.gradient_checkpointing_enable.assert_called_once()
    assert result is mock_model


@patch("src.model.load_model._get_model_class")
def test_load_base_model_gradient_checkpointing_disabled(mock_get_cls):
    mock_model = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model
    mock_get_cls.return_value = mock_cls

    cfg = _full_cfg()
    # Override training section
    cfg.training.gradient_checkpointing = False
    result = load_base_model(cfg)

    mock_model.gradient_checkpointing_enable.assert_not_called()
    assert result is mock_model



def test_prepare_model_for_qlora_casts_norms_to_fp32():
    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4).to(dtype=torch.float16)
            self.embed = torch.nn.Embedding(8, 4).to(dtype=torch.float16)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return None

        def enable_input_require_grads(self):
            return None

    model = _TinyModel()
    model.is_loaded_in_4bit = True

    prepared = _prepare_model_for_qlora_training(
        model,
        use_gradient_checkpointing=True,
    )

    assert prepared.norm.weight.dtype == torch.float32
    assert prepared.norm.bias.dtype == torch.float32


@patch("src.model.load_model._optional_cast_budget_bytes", return_value=1024)
def test_prepare_model_for_qlora_skips_large_optional_fp32_cast(_mock_budget):
    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4).to(dtype=torch.float16)
            self.embed = torch.nn.Embedding(8, 4).to(dtype=torch.float16)
            self.lm_head = torch.nn.Linear(4, 8, bias=False).to(dtype=torch.float16)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

        def enable_input_require_grads(self):
            return None

    model = _TinyModel()
    model.is_loaded_in_4bit = True
    model.embed.weight = torch.nn.Parameter(torch.zeros((1024, 1024), dtype=torch.float16, device="cpu"))
    model.lm_head.weight = torch.nn.Parameter(torch.zeros((1024, 1024), dtype=torch.float16, device="cpu"))

    prepared = _prepare_model_for_qlora_training(
        model,
        use_gradient_checkpointing=True,
    )

    assert prepared.embed.weight.dtype == torch.float16
    assert prepared.lm_head.weight.dtype == torch.float16


def test_prepare_model_for_qlora_freezes_base_parameters():
    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4).to(dtype=torch.float16)

        def get_input_embeddings(self):
            return None

        def get_output_embeddings(self):
            return None

        def enable_input_require_grads(self):
            return None

    model = _TinyModel()
    model.is_loaded_in_4bit = True

    _prepare_model_for_qlora_training(model, use_gradient_checkpointing=False)

    assert all(not param.requires_grad for param in model.parameters())
