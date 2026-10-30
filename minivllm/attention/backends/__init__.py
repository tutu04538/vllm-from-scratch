"""Attention 后端注册：默认 Torch SDPA；可选 Triton paged。"""

from .torch_sdpa import (TorchAttentionBackend, TorchAttentionImpl,
                         TorchAttentionMetadataBuilder)


def get_attention_backend(name: str):
    name = str(name).lower()
    if name in ("torch", "torch_sdpa"):
        return TorchAttentionBackend
    if name in ("triton", "triton_attn"):
        from .triton_attn import TritonAttentionBackend

        return TritonAttentionBackend
    raise ValueError(
        f"未知 attention_backend={name!r}（只支持 torch / triton / triton_attn）")


__all__ = ["TorchAttentionBackend", "TorchAttentionImpl",
           "TorchAttentionMetadataBuilder", "get_attention_backend"]
