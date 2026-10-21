"""Phase 2：图安全 varlen decode 注意力（Torch 稠密实现）。"""

from __future__ import annotations

import pytest
import torch

from spec74_helpers import requires_cuda
from minivllm.attention.backends.torch_sdpa import TorchAttentionImpl
from minivllm.attention.metadata import AttentionMetadata
from minivllm.config import CUDAGraphMode
from minivllm.forward_context import BatchDescriptor, set_forward_context


def _case(*, q_lens=(3, 2), seq_lens=(5, 6), max_q=3, causal=True):
    torch.manual_seed(0)
    heads, head_size, kv_heads = 4, 4, 2
    block_size, num_blocks = 2, 4
    impl = TorchAttentionImpl(heads, head_size, 1.0 / head_size ** 0.5,
                              kv_heads)
    kv_cache = torch.randn(2, num_blocks, block_size, kv_heads, head_size,
                           device="cuda")
    block_table = torch.tensor(
        [[0, 1, 2, 3], [2, 1, 0, 3]], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, *q_lens], dtype=torch.int32, device="cuda")
    qsl = torch.cumsum(qsl, 0).to(torch.int32)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    num_tokens = int(sum(q_lens))
    query = torch.randn(num_tokens, heads * head_size, device="cuda")
    meta = AttentionMetadata(
        query_start_loc=qsl, seq_lens=seq, block_table=block_table,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device="cuda"),
        block_size=block_size, num_reqs=len(q_lens), max_query_len=max_q,
        causal=causal)
    return impl, kv_cache, query, meta


@requires_cuda
@pytest.mark.parametrize("causal", [True, False])
def test_varlen_attention_matches_generic(causal):
    impl, kv_cache, query, meta = _case(causal=causal)
    want = impl._forward_generic(query, kv_cache, meta)
    with set_forward_context(
            {}, num_tokens=query.shape[0],
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=8, num_reqs=2)):
        got = impl._forward_varlen_padded(query, kv_cache, meta)
    assert torch.equal(got, want), (got - want).abs().max()


@requires_cuda
def test_varlen_attention_handles_padding_rows_and_requests():
    """第二个请求 q_len=0（只有 padding 行）时，前 3 行仍与 generic 相同。"""
    impl, kv_cache, query, meta = _case(q_lens=(3, 0), seq_lens=(5, 0), max_q=3)
    want = impl._forward_generic(query, kv_cache, meta)
    with set_forward_context(
            {}, num_tokens=query.shape[0],
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=6, num_reqs=2)):
        got = impl._forward_varlen_padded(query, kv_cache, meta)
    assert torch.equal(got[:3], want[:3])
    assert torch.isfinite(got[3:]).all()


@requires_cuda
def test_varlen_attention_does_not_require_cpu_sync(monkeypatch):
    """图内路径不能把 GPU 张量取成 Python int：禁止 `.item()` / `int(tensor)`。"""
    impl, kv_cache, query, meta = _case()
    def boom(*args, **kwargs):
        raise AssertionError("varlen 图路径不允许 host 同步")
    monkeypatch.setattr(torch.Tensor, "item", boom)
    # `int(attn_metadata.max_query_len)` 是 Python int，不触发 `.item()`。
    with set_forward_context(
            {}, num_tokens=query.shape[0],
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=8, num_reqs=2)):
        impl._forward_varlen_padded(query, kv_cache, meta)
