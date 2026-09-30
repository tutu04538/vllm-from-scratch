"""Attention 边界（对应 vLLM `vllm/attention/` 与 `v1/attention/` 的子集）。

    forward_context.py      一次 forward 内的"层名 → AttentionMetadata"
    metadata.py             AttentionMetadata 与它的 builder
    layer.py                Attention：模型里的边界，只转发
    backends/torch_sdpa.py  教学后端：写 KV + 按绝对位置 causal attention
"""

from .forward_context import ForwardContext, get_forward_context, set_forward_context
from .layer import Attention
from .metadata import AttentionMetadata, AttentionMetadataBuilder

__all__ = ["Attention", "AttentionMetadata", "AttentionMetadataBuilder",
           "ForwardContext", "get_forward_context", "set_forward_context"]
