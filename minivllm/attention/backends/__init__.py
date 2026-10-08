"""Attention 后端：模型只表达"这里做 attention"，块表怎么排布由后端决定。

对应 vLLM `v1/attention/backends/`。本关只有一个教学后端（逐请求 gather + Torch 数学）。
"""

from .torch_sdpa import (TorchAttentionBackend, TorchAttentionImpl,
                         TorchAttentionMetadataBuilder)

__all__ = ["TorchAttentionBackend", "TorchAttentionImpl",
           "TorchAttentionMetadataBuilder"]
