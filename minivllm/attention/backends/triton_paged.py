"""Triton paged attention 后端（FlashAttention 风格的 varlen/paged 实现）。

本文件不是从零发明 kernel：主体复用本仓库第 22/37 关的 Triton paged attention
（`_paged_attention_kernel` / `tiled_paged_attention`，见 `docs/step22_triton_attention.md`
与 `docs/step37_query_tiled_prefill_attention.md`），并补上 vllm V2 需要的：

    * `AttentionCGSupport.ALWAYS`：每请求 query 行数任意，支持混合 prefill/decode；
    * 图内路径只读 GPU 上的 query_start_loc/seq_lens/block_table，不取 `.item()`；
    * 输出可以直接写进 `Attention._output_buffer`（breakable PIECEWISE 用）。

与上游 FlashAttentionImpl 的差异：上游用 FlashAttention/FlashInfer 的 fused varlen kernel；
本实现是**本仓库历史的 Triton 分页 kernel**，语义对齐 paged causal attention，性能不等价。
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only import path
    triton = None
    tl = None

from ..backend import AttentionCGSupport
from ..metadata import AttentionMetadata
from .torch_sdpa import (TorchAttentionBackend, TorchAttentionImpl,
                         TorchAttentionMetadataBuilder)


@triton.jit
def _paged_attention_kernel(
    q_ptr, out_ptr, k_pool_ptr, v_pool_ptr,
    block_tables_ptr, seq_lens_ptr, token_to_req_ptr, query_pos_ptr,
    stride_qn, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btn, stride_btb,
    stride_on, stride_oh, stride_od,
    SM_SCALE: tl.constexpr, S: tl.constexpr, S_POW2: tl.constexpr,
    D: tl.constexpr, D_POW2: tl.constexpr, GROUP_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # 一个 program = 一个打包 query 行的一个 head。
    # 形状由 grid 固定；每条请求的 query 长度、seq_len、块表都只从 GPU 张量读。
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP_SIZE

    req = tl.load(token_to_req_ptr + row)
    qpos = tl.load(query_pos_ptr + row)
    seq_len = tl.load(seq_lens_ptr + req)

    offs_d = tl.arange(0, D_POW2)
    mask_d = offs_d < D
    offs_s = tl.arange(0, S_POW2)

    q = tl.load(q_ptr + row * stride_qn + q_head * stride_qh + offs_d * stride_qd,
                mask=mask_d, other=0.0).to(tl.float32)

    m_i = float("-inf")
    z_i = 0.0
    acc = tl.zeros([D_POW2], dtype=tl.float32)

    for blk in range(0, tl.cdiv(seq_len, S)):
        phys = tl.load(block_tables_ptr + req * stride_btn + blk * stride_btb)
        valid = (offs_s < S) & (offs_s < seq_len - blk * S)
        if CAUSAL:
            valid = valid & ((blk * S + offs_s) <= qpos)

        kv_offsets = (phys * stride_kb + offs_s[:, None] * stride_ks
                      + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_pool_ptr + kv_offsets,
                    mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        score = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        score = tl.where(valid, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=0))
        scale = tl.exp(m_i - m_new)
        p = tl.where(valid, tl.exp(score - m_new), 0.0)

        v_offsets = (phys * stride_vb + offs_s[:, None] * stride_vs
                     + kv_head * stride_vh + offs_d[None, :] * stride_vd)
        v = tl.load(v_pool_ptr + v_offsets,
                    mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        z_i = scale * z_i + tl.sum(p, axis=0)
        acc = scale * acc + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # seq_len=0 的 padding 请求会走空循环（z_i=0）；不能做 0/0，否则 NaN 会进入
    # padding 行的 hidden。真实行不受影响，但保持 padding 行有限更安全。
    out = tl.where(z_i > 0.0, acc / z_i, 0.0)
    tl.store(out_ptr + row * stride_on + q_head * stride_oh + offs_d * stride_od,
             out.to(out_ptr.dtype.element_ty), mask=mask_d)


def paged_attention(q: torch.Tensor, out: torch.Tensor, k_pool: torch.Tensor,
                    v_pool: torch.Tensor, block_tables: torch.Tensor,
                    seq_lens: torch.Tensor, token_to_req: torch.Tensor,
                    query_pos: torch.Tensor, group_size: int,
                    scale: float, causal: bool) -> torch.Tensor:
    """q/out [N,H,D]；k_pool/v_pool [num_blocks, block_size, Hk, D]。"""
    if triton is None:
        raise RuntimeError("Triton 不可用，不能使用 Triton paged attention 后端")
    num_rows, num_q_heads, head_dim = q.shape
    block_size = k_pool.shape[1]
    _paged_attention_kernel[(num_rows, num_q_heads)](
        q, out, k_pool, v_pool,
        block_tables, seq_lens, token_to_req, query_pos,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        SM_SCALE=scale, S=block_size,
        S_POW2=triton.next_power_of_2(block_size),
        D=head_dim, D_POW2=max(16, triton.next_power_of_2(head_dim)),
        GROUP_SIZE=group_size, CAUSAL=bool(causal),
    )
    return out


class TritonPagedAttentionMetadataBuilder(TorchAttentionMetadataBuilder):
    """与 Torch builder 同字段；能力声明为 ALWAYS（任意 query 长度 / 混合批）。"""

    _cudagraph_support = AttentionCGSupport.ALWAYS


class TritonPagedAttentionImpl(TorchAttentionImpl):
    """图内走 Triton paged kernel；eager/PIECEWISE 仍走父类的通用参考路径。"""

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache,
                attn_metadata: AttentionMetadata):
        self.write_kv(key, value, kv_cache, attn_metadata)
        if self._graph_dispatched():
            return self._forward_paged(query, kv_cache, attn_metadata)
        return self._forward_generic(query, kv_cache, attn_metadata)

    def _forward_paged(self, query: torch.Tensor, kv_cache,
                       attn_metadata: AttentionMetadata) -> torch.Tensor:
        num_tokens = query.shape[0]
        num_heads, head_size = self.num_heads, self.head_size
        num_reqs = attn_metadata.num_reqs
        q = query.reshape(num_tokens, num_heads, head_size)
        out = torch.empty_like(q)

        qsl = attn_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = attn_metadata.seq_lens[:num_reqs]
        # 行 -> 请求：qsl 是单调前缀和，用比较+求和得到（无 .item()、无 Python 循环）。
        row = torch.arange(num_tokens, device=query.device, dtype=torch.int32)
        token_to_req = (row[:, None] >= qsl[None, :-1]).sum(dim=1) - 1
        token_to_req = token_to_req.clamp_(0, num_reqs - 1).to(torch.int32)
        q_lens = qsl[1:] - qsl[:-1]
        start = qsl[token_to_req]
        query_pos = (seq_lens[token_to_req] - q_lens[token_to_req]
                     + (row - start)).to(torch.int32)

        paged_attention(
            q, out, kv_cache[0], kv_cache[1],
            attn_metadata.block_table, seq_lens, token_to_req, query_pos,
            group_size=self.num_queries_per_kv, scale=self.scale,
            causal=attn_metadata.causal,
        )
        return out.reshape(num_tokens, num_heads * head_size)


class TritonPagedAttentionBackend(TorchAttentionBackend):
    """后端入口：builder 声明 ALWAYS，impl 走 Triton paged kernel。"""

    @staticmethod
    def get_name() -> str:
        return "TRITON_PAGED"

    @staticmethod
    def get_builder_cls():
        return TritonPagedAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return TritonPagedAttentionImpl

    @staticmethod
    def supports_device_cpu_query_lens_mismatch() -> bool:
        # query_pos/qsl/seq_lens 全部从 device 读，没有按 CPU query 长度做规划。
        return True
