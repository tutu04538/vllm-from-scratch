"""Attention 后端注册：默认 Torch SDPA；可选 Triton paged。"""

from .torch_sdpa import (TorchAttentionBackend, TorchAttentionImpl,
                         TorchAttentionMetadataBuilder)


def get_attention_backend(name: str):
    name = str(name).lower()
    if name in ("torch", "torch_sdpa"):
        return TorchAttentionBackend
    if name == "triton":
        from .triton_paged import TritonPagedAttentionBackend

        return TritonPagedAttentionBackend
    raise ValueError(f"未知 attention_backend={name!r}（只支持 torch / triton）")


__all__ = ["TorchAttentionBackend", "TorchAttentionImpl",
           "TorchAttentionMetadataBuilder", "get_attention_backend"]
