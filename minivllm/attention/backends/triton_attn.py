"""固定场景的 Triton paged attention 后端。

固定条件：

    Qwen3 target / block_size=16 / head_dim in {16,32,64,128}
    causal / GQA / paged KV / fp16·bf16·fp32
    query 行按 query_start_loc 打包，prefill/decode/混合批都支持
    无量化 / 无 sliding / 无 cascade / 无 sinks / 无 MM prefix / 无 RSWA

一个 program 负责一条请求的一个 query block 的一个 query head；query block 之间并行，
不再像旧 `triton_paged` 那样“一行 query 一个 program”。block4/head4 回退旧 kernel。
本后端不是完整上游 TRITON_ATTN；配置名 `triton_attn`，`triton` 是别名。
"""

from __future__ import annotations

import torch

from ..backend import AttentionCGSupport
from ..metadata import AttentionMetadata
from ..ops.fixed_attention import fixed_paged_attention
from .triton_paged import (TritonPagedAttentionBackend,
                           TritonPagedAttentionImpl,
                           TritonPagedAttentionMetadataBuilder)


class TritonAttentionMetadataBuilder(TritonPagedAttentionMetadataBuilder):
    """与 Triton paged builder 同字段；能力 ALWAYS。"""

    _cudagraph_support = AttentionCGSupport.ALWAYS


class TritonAttentionImpl(TritonPagedAttentionImpl):
    """图内优先走上游 unified_attention；小 head/小块走本地逐行 fallback。"""

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache,
                attn_metadata: AttentionMetadata):
        self.write_kv(key, value, kv_cache, attn_metadata)
        if self._graph_dispatched():
            bs = attn_metadata.block_size
            if self.head_size >= 16 and bs >= 16:
                return self._forward_unified(query, kv_cache, attn_metadata)
            return self._forward_paged(query, kv_cache, attn_metadata)
        return self._forward_generic(query, kv_cache, attn_metadata)

    def _forward_unified(self, query: torch.Tensor, kv_cache,
                         attn_metadata: AttentionMetadata) -> torch.Tensor:
        num_tokens = query.shape[0]
        num_heads, head_size = self.num_heads, self.head_size
        q = query.reshape(num_tokens, num_heads, head_size)
        # 固定场景要求 block16、head_dim>=16；调用方已 gate。
        out = torch.zeros_like(q)
        fixed_paged_attention(
            q, out, kv_cache[0], kv_cache[1],
            attn_metadata.block_table[: attn_metadata.num_reqs],
            attn_metadata.seq_lens[: attn_metadata.num_reqs],
            attn_metadata.query_start_loc[: attn_metadata.num_reqs + 1],
            block_size=attn_metadata.block_size,
            scale=self.scale,
        )
        return out.reshape(num_tokens, num_heads * head_size)


class TritonAttentionBackend(TritonPagedAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_FIXED"

    @staticmethod
    def get_builder_cls():
        return TritonAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return TritonAttentionImpl
