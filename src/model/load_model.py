import logging

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from src.utils.device import detect_device, resolve_compute_dtype

logger = logging.getLogger("tg-lora")

_OPTIONAL_FP32_CAST_MAX_BYTES = 512 * 1024**2
_OPTIONAL_FP32_CAST_FREE_FRACTION = 0.1

# Map multimodal architecture names to their text-only causal LM equivalents.
_TEXT_ONLY_CLASS_MAP = {
    "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",
}


def _get_model_class(config):
    """Return the text-only causal LM class for the given config."""
    architectures = getattr(config, "architectures", []) or []
    for arch in architectures:
        if arch in _TEXT_ONLY_CLASS_MAP:
            target = _TEXT_ONLY_CLASS_MAP[arch]
            import transformers

            return getattr(transformers, target)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def build_bnb_config(cfg: DictConfig) -> BitsAndBytesConfig | None:
    if not cfg.model.get("load_in_4bit", False):
        return None
    compute_dtype = _resolve_dtype(cfg.model.get("bnb_4bit_compute_dtype", "bf16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def get_device_map(cfg: DictConfig) -> str | dict:
    device = cfg.model.get("device", None)
    if device is not None:
        return {"": str(device)}
    device_map = cfg.model.get("device_map", None)
    if device_map is not None:
        return device_map
    detected = detect_device()
    if detected.type in ("cuda", "mps"):
        return {"": str(detected)}
    return "cpu"


def get_input_device(model) -> torch.device:
    return next(model.parameters()).device


def _is_quantized_parameter(param: torch.nn.Parameter) -> bool:
    return param.__class__.__name__ == "Params4bit"


def _is_fp32_cast_candidate(param: torch.nn.Parameter) -> bool:
    return (
        param.dtype in (torch.float16, torch.bfloat16)
        and not _is_quantized_parameter(param)
    )


def _optional_cast_budget_bytes(param: torch.nn.Parameter) -> int:
    budget = _OPTIONAL_FP32_CAST_MAX_BYTES
    if param.device.type != "cuda":
        return budget
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(param.device.index or 0)
    except RuntimeError:
        return budget
    return min(budget, int(free_bytes * _OPTIONAL_FP32_CAST_FREE_FRACTION))


def _cast_parameter_to_fp32(
    param: torch.nn.Parameter,
    *,
    param_name: str,
    optional: bool,
) -> bool:
    if not _is_fp32_cast_candidate(param):
        return False

    required_bytes = param.numel() * 4
    if optional:
        budget = _optional_cast_budget_bytes(param)
        if required_bytes > budget:
            logger.info(
                "Skipping fp32 cast for %s (needs %.1f MiB, optional budget %.1f MiB)",
                param_name,
                required_bytes / 1024**2,
                budget / 1024**2,
            )
            return False

    try:
        param.data = param.data.to(torch.float32)
    except RuntimeError as exc:
        logger.info("Skipping fp32 cast for %s (%s)", param_name, exc)
        return False
    return True


def _is_norm_module(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return isinstance(module, torch.nn.LayerNorm) or "norm" in class_name


def _prepare_model_for_qlora_training(
    model,
    *,
    use_gradient_checkpointing: bool,
):
    """Prepare a quantized model for QLoRA without blanket fp32 casting.

    PEFT's stock helper casts every non-4bit fp16/bf16 tensor to fp32. On
    large-vocab models such as Qwen3.5-9B, this can upcast multi-gigabyte
    embedding / lm_head tensors on GPU and fail before training starts.

    For single-GPU QLoRA we only need:
    - base parameters frozen
    - numerically sensitive norm layers in fp32
    - optional fp32 upcast for small input/output embeddings when affordable
    - input grads enabled for gradient checkpointing
    """

    for param in model.parameters():
        param.requires_grad = False

    for module_name, module in model.named_modules():
        if not _is_norm_module(module):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            _cast_parameter_to_fp32(
                param,
                param_name=(
                    f"{module_name}.{param_name}" if module_name else param_name
                ),
                optional=False,
            )

    input_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        for param_name, param in input_embeddings.named_parameters(recurse=False):
            _cast_parameter_to_fp32(
                param,
                param_name=f"input_embeddings.{param_name}",
                optional=True,
            )

    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
    if output_embeddings is None and hasattr(model, "lm_head"):
        output_embeddings = model.lm_head
    if output_embeddings is not None:
        for param_name, param in output_embeddings.named_parameters(recurse=False):
            _cast_parameter_to_fp32(
                param,
                param_name=f"output_embeddings.{param_name}",
                optional=True,
            )

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif input_embeddings is not None:
            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)

            input_embeddings.register_forward_hook(make_inputs_require_grad)

    return model


def load_base_model(cfg: DictConfig):
    bnb_config = build_bnb_config(cfg)
    device = detect_device()
    dtype = resolve_compute_dtype(device, cfg.model.get("dtype", "bf16"))
    device_map = get_device_map(cfg)

    # Resolve to text-only causal LM class (skip visual encoder entirely).
    config = AutoConfig.from_pretrained(cfg.model.name_or_path)
    model_cls = _get_model_class(config)

    # For multimodal models, extract the nested text config.
    if hasattr(config, "text_config"):
        text_config = config.text_config
    else:
        text_config = config

    # Monkey-patch _init_weights to skip expensive float32 initialization.
    # All weights come from the checkpoint, so random init is unnecessary.
    original_init_weights = getattr(model_cls, "_init_weights", None)
    model_cls._init_weights = lambda self, module: None

    try:
        model = model_cls.from_pretrained(
            cfg.model.name_or_path,
            config=text_config,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    finally:
        if original_init_weights is not None:
            model_cls._init_weights = original_init_weights

    if cfg.training.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    return model


def load_tokenizer(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_lora(
    model, cfg: DictConfig, *, rank_pattern: dict | None = None,
    alpha_pattern: dict | None = None,
):
    if cfg.model.get("load_in_4bit", False):
        use_gradient_checkpointing = bool(
            cfg.get("training", {}).get("gradient_checkpointing", True)
        )
        model = _prepare_model_for_qlora_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    # rank_pattern / alpha_pattern realize the heterogeneous (per-layer asymmetric
    # rank) architecture: a regex->{rank} and regex->{alpha} dict that PEFT
    # matches against each target module name and uses to size that layer's LoRA
    # adapter. Empty (the default, ``None``) is PEFT's own default — every layer
    # gets ``cfg.lora.r`` / ``cfg.lora.alpha`` — so the production path
    # (``train_tg_lora`` / ``train_baseline``, which never passes a pattern) is
    # byte-identical to before. ``alpha_pattern`` is set alongside ``rank_pattern``
    # to hold ``alpha / rank`` constant, so the only thing that varies across
    # layers is adapter *capacity* (rank), not the LoRA scaling magnitude.
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        rank_pattern=rank_pattern or {},
        lora_alpha=cfg.lora.alpha,
        alpha_pattern=alpha_pattern or {},
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def _resolve_dtype(name: str) -> torch.dtype:
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(name).lower()
    if key not in aliases:
        raise ValueError(f"Unsupported dtype: {name}")
    return aliases[key]
