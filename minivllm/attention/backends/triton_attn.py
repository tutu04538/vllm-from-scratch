"""上游 `TRITON_ATTN` 的核心子集：vendored `unified_attention` kernel。

与本地 `triton_paged.py` 的区别：

    triton_paged    本仓库 step22/37 的逐行 paged kernel；作为 block_size<16 或
                    head_dim<16 的 fallback。
    triton_attn     上游 `triton_unified_attention.py` 的 2D unified kernel；
                    使用与上游同形的 `max_query_len/max_seq_len/qsl/seq_lens/block_table`
                    参数和 KV 布局 `[num_blocks, block_size, Hk, D]`。

本子集只支持：

    fp16 / bf16 / fp32，block_size >= 16，head_dim >= 16
    causal / 非 causal、GQA、varlen、混合 prefill/decode

明确拒绝（由 builder/impl 或上游 kernel 本身报错）：

    量化 KV、sliding window、cascade、sinks、MM prefix、RSWA、tensor descriptor
"""

from __future__ import annotations

import torch

from ..backend import AttentionCGSupport
from ..metadata import AttentionMetadata
from ..ops.kv_quant import KVQuantMode
from ..ops.triton_unified_attention import unified_attention
from .torch_sdpa import TorchAttentionImpl
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
        out = torch.empty_like(q)

        num_reqs = attn_metadata.num_reqs
        block_table = attn_metadata.block_table[:num_reqs]
        # max_seqlen_q/k 必须是 Python int（图内不能 .item()）；这里用固定上界：
        #   q 上界 = 本轮总行数
        #   k 上界 = 块表宽度 × block_size（= 上下文容量）
        # 2D kernel 只用它们算 grid/tile，不会按最大值真扫满。
        max_seqlen_q = num_tokens
        max_seqlen_k = block_table.shape[1] * attn_metadata.block_size

        unified_attention(
            q=q,
            k=kv_cache[0],
            v=kv_cache[1],
            out=out,
            cu_seqlens_q=attn_metadata.query_start_loc[: num_reqs + 1],
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.seq_lens[:num_reqs],
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=None,
            num_par_softmax_segments=None,
            softmax_segm_output=None,
            softmax_segm_max=None,
            softmax_segm_expsum=None,
            alibi_slopes=None,
            output_scale=None,
            qq_bias=None,
            sinks=None,
            mm_prefix_range=None,
            rswa_prefix_lens=None,
            rswa_window=None,
            use_alibi_sqrt=False,
            kv_quant_mode=KVQuantMode.NONE,
            k_scale_cache=None,
            v_scale_cache=None,
            chunk_lookback=-1,
            use_td=False,
            mm_prefix_clamp_sliding_window=False,
        )
        return out.reshape(num_tokens, num_heads * head_size)


class TritonAttentionBackend(TritonPagedAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_builder_cls():
        return TritonAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return TritonAttentionImpl
