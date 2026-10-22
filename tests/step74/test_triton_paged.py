"""Triton paged attention 后端：数值对照 + ALWAYS 能力。"""

from __future__ import annotations

import pytest
import torch

from spec74_helpers import requires_cuda
from minivllm.attention.backend import AttentionCGSupport
from minivllm.attention.backends.triton_paged import (
    TritonPagedAttentionBackend, TritonPagedAttentionImpl)
from minivllm.attention.backends.torch_sdpa import TorchAttentionImpl
from minivllm.attention.metadata import AttentionMetadata
from minivllm.config import CUDAGraphMode
from minivllm.forward_context import BatchDescriptor, set_forward_context


def _case(*, q_lens=(3, 2), seq_lens=(5, 6), causal=True, dtype=torch.float16):
    torch.manual_seed(0)
    heads, head_size, kv_heads = 4, 4, 2
    block_size, num_blocks = 2, 8
    kv_cache = torch.randn(2, num_blocks, block_size, kv_heads, head_size,
                           device="cuda", dtype=dtype)
    block_table = torch.tensor(
        [[0, 1, 2, 3], [2, 1, 0, 3]], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, *q_lens], dtype=torch.int32, device="cuda").cumsum(0)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    num_tokens = int(sum(q_lens))
    query = torch.randn(num_tokens, heads * head_size, device="cuda", dtype=dtype)
    meta = AttentionMetadata(
        query_start_loc=qsl, seq_lens=seq, block_table=block_table,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device="cuda"),
        block_size=block_size, num_reqs=len(q_lens), max_query_len=3,
        causal=causal)
    return kv_cache, query, meta


def test_triton_backend_declares_always():
    assert (TritonPagedAttentionBackend.get_builder_cls()
            ._cudagraph_support == AttentionCGSupport.ALWAYS)


@requires_cuda
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_triton_paged_matches_torch_generic(causal, dtype):
    kv_cache, query, meta = _case(causal=causal, dtype=dtype)
    got_impl = TritonPagedAttentionImpl(4, 4, 1.0 / 2.0, 2)
    want_impl = TorchAttentionImpl(4, 4, 1.0 / 2.0, 2)
    want = want_impl._forward_generic(query, kv_cache, meta)
    with set_forward_context(
            {}, num_tokens=query.shape[0],
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=8, num_reqs=2)):
        got = got_impl._forward_paged(query, kv_cache, meta)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_triton_paged_handles_padding_request_and_tail_rows():
    kv_cache, query, meta = _case(q_lens=(3, 2), seq_lens=(5, 0))
    # 第二个请求 seq_len=0、q_len=2：它的两行会走空 K/V 循环。
    impl = TritonPagedAttentionImpl(4, 4, 1.0 / 2.0, 2)
    ref = TorchAttentionImpl(4, 4, 1.0 / 2.0, 2)._forward_generic(
        query, kv_cache, meta)
    with set_forward_context(
            {}, num_tokens=query.shape[0],
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=8, num_reqs=2)):
        got = impl._forward_paged(query, kv_cache, meta)
    torch.testing.assert_close(got[:3], ref[:3], rtol=1e-3, atol=1e-3)
    assert torch.isfinite(got).all()


@requires_cuda
def test_fixed_kernel_query_tiled_prefill_matches_generic():
    """query block 并行路径：一条请求 20 行 + 一条 7 行，覆盖 >BLOCK_M 的 prefill。"""
    from minivllm.attention.ops.fixed_attention import fixed_paged_attention

    torch.manual_seed(0)
    heads, head_size, kv_heads = 4, 16, 2
    block_size, num_blocks = 16, 8
    kv_cache = torch.randn(2, num_blocks, block_size, kv_heads, head_size,
                           device="cuda", dtype=torch.float16)
    block_table = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7]],
        dtype=torch.int32, device="cuda")
    q_lens = (20, 7)
    qsl = torch.tensor([0, q_lens[0], sum(q_lens)],
                       dtype=torch.int32, device="cuda")
    seq = torch.tensor([25, 10], dtype=torch.int32, device="cuda")
    num_tokens = sum(q_lens)
    query = torch.randn(num_tokens, heads * head_size,
                        device="cuda", dtype=torch.float16)
    meta = AttentionMetadata(
        query_start_loc=qsl, seq_lens=seq, block_table=block_table,
        slot_mapping=torch.arange(num_tokens, device="cuda"),
        block_size=block_size, num_reqs=2, max_query_len=max(q_lens))
    want = TorchAttentionImpl(heads, head_size, 1.0 / 4.0, kv_heads)\
        ._forward_generic(query, kv_cache, meta)
    out = torch.zeros(num_tokens, heads, head_size,
                      device="cuda", dtype=torch.float16)
    fixed_paged_attention(
        query.view(num_tokens, heads, head_size), out,
        kv_cache[0], kv_cache[1], block_table, seq, qsl,
        block_size=block_size, scale=1.0 / 4.0)
    torch.testing.assert_close(out.view(num_tokens, -1), want,
                               rtol=1e-3, atol=1e-3)
