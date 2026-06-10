"""Unified layer type classification for consistent metric naming.

Used by both psa.py and layer_delta_analysis.py so that metric keys
(psa_lt_*), analysis reports, and gain mapping all agree on the same
layer type names.
"""

from enum import Enum


class LayerType(str, Enum):
    ATTENTION_OUT = "attention_out"
    ATTENTION_V = "attention_v"
    ATTENTION_OTHER = "attention_other"
    DELTANET = "deltanet"
    MLP = "mlp"
    UNKNOWN = "unknown"


def classify_layer_type(tensor_name: str) -> LayerType:
    """Classify a tensor name into a layer type based on naming conventions.

    Qwen3.5-9B / Qwen3.6-35B naming:
    - DeltaNet layers: model.layers.N.dt_*.lora_A/B
    - Attention layers: model.layers.N.self_attn.{out_proj,v_proj,...}.lora_A/B
    - MLP/FFN: model.layers.N.mlp.*.lora_A/B
    - Mamba/SSM layers: model.layers.N.{mamba,ssm}.*.lora_A/B
    """
    lower = tensor_name.lower()
    if "out_proj" in lower:
        return LayerType.ATTENTION_OUT
    if "v_proj" in lower:
        return LayerType.ATTENTION_V
    if "self_attn" in lower or "attention" in lower:
        return LayerType.ATTENTION_OTHER
    if (
        "dt_" in lower
        or "deltanet" in lower
        or "gdn" in lower
        or "mamba" in lower
        or "ssm" in lower
        or "a_log" in lower
        or "linear_attn" in lower
    ):
        return LayerType.DELTANET
    if "mlp" in lower or "ffn" in lower or "feed_forward" in lower:
        return LayerType.MLP
    return LayerType.UNKNOWN
