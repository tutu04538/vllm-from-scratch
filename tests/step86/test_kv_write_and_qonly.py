"""086：Triton KV 写入、Q-only 只读和后端边界。"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from spec86_helpers import requires_cuda
from minivllm.attention.backends.torch_sdpa import TorchAttentionImpl
from minivllm.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionWorkspacePool,
)
from minivllm.attention.metadata import AttentionMetadata
from minivllm.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from minivllm.attention.backends import get_attention_backend


def _cache_pairs(num_blocks, block_size, num_kv_heads, head_size,
                 dtype=torch.float16, pad=0):
    storage = torch.zeros(
        2, num_blocks, block_size + pad, num_kv_heads, head_size,
        device="cuda", dtype=dtype)
    cache = storage[:, :, :block_size]
    return storage, cache


@requires_cuda
def test_negative_slots_are_not_written():
    _, cache = _cache_pairs(4, 4, 1, 8)
    k = torch.randn(4, 1, 8, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    slots = torch.tensor([0, 5, -1, 15], device="cuda", dtype=torch.int64)
    before = cache.clone()
    triton_reshape_and_cache_flash(k, v, cache[0], cache[1], slots)
    torch.testing.assert_close(cache[0, 0, 0], k[0])
    torch.testing.assert_close(cache[0, 1, 1], k[1])
    torch.testing.assert_close(cache[0, 3, 3], k[3])
    # slot=-1 对应 block 0 slot 0 之外没有目标；直接检查所有未命中槽位未变。
    untouched = torch.ones_like(before[0], dtype=torch.bool)
    untouched[0, 0] = False
    untouched[1, 1] = False
    untouched[3, 3] = False
    assert torch.equal(cache[0][untouched], before[0][untouched])
    assert torch.equal(cache[1][untouched], before[1][untouched])


@requires_cuda
def test_packed_cache_stride_is_respected():
    storage, cache = _cache_pairs(4, 4, 2, 8, pad=3)
    assert not cache.is_contiguous()
    k = torch.randn(3, 2, 8, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    slots = torch.tensor([3, 10, 12], device="cuda", dtype=torch.int64)
    triton_reshape_and_cache_flash(k, v, cache[0], cache[1], slots)
    for src, slot in enumerate(slots.tolist()):
        blk, off = divmod(slot, 4)
        torch.testing.assert_close(cache[0, blk, off], k[src])
        torch.testing.assert_close(cache[1, blk, off], v[src])


@requires_cuda
def test_write_rejects_bad_shape_or_dtype():
    _, cache = _cache_pairs(4, 4, 1, 8)
    k = torch.randn(1, 1, 8, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="dtype"):
        triton_reshape_and_cache_flash(
            k, v, cache[0], cache[1],
            torch.tensor([0], device="cuda", dtype=torch.int64))
    k2 = torch.randn(1, 2, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="kv_heads/head_size"):
        triton_reshape_and_cache_flash(
            k2, k2, cache[0], cache[1],
            torch.tensor([0], device="cuda", dtype=torch.int64))


def _qonly_case(*, window=None, seq_len=9, heads=4, head_size=16,
                kv_heads=1, block_size=4):
    torch.manual_seed(0)
    q = torch.randn(1, heads, head_size, device="cuda", dtype=torch.float32)
    _, cache = _cache_pairs(3, block_size, kv_heads, head_size,
                            dtype=torch.float32)
    cache.normal_()
    qsl = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seq = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    bt = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    meta = AttentionMetadata(
        query_start_loc=qsl, seq_lens=seq, block_table=bt,
        slot_mapping=torch.zeros(1, device="cuda", dtype=torch.int64),
        block_size=block_size, num_reqs=1, max_query_len=1,
        sliding_window=window, causal=True)
    # 087：3D workspace 的所有权在 metadata builder；直接调 Impl 时显式给一个池。
    meta.workspace_pool = TritonAttentionWorkspacePool()
    return q, cache, meta


@requires_cuda
def test_qonly_reads_shared_cache_without_writing():
    q, cache, meta = _qonly_case()
    impl = TritonAttentionImpl(4, 16, 1.0, 1)
    layer = SimpleNamespace(per_layer_sliding_window=None,
                            _output_buffer=None)
    before = cache.clone()
    got = impl.forward_q_only(layer, q.reshape(1, -1), cache, meta)
    assert torch.equal(cache, before), "Q-only 绝不能写源层 cache"
    want = TorchAttentionImpl(4, 16, 1.0, 1)._forward_generic(
        q.reshape(1, -1), cache, meta)
    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)


@requires_cuda
def test_qonly_respects_sliding_window_and_output_buffer():
    q, cache, meta = _qonly_case(window=4, seq_len=9)
    impl = TritonAttentionImpl(4, 16, 1.0, 1)
    buf = torch.zeros(1, 4 * 16, device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(per_layer_sliding_window=4,
                            _output_buffer=buf)
    got = impl.forward_q_only(layer, q.reshape(1, -1), cache, meta)
    assert got.data_ptr() == buf.data_ptr(), "应直接写现有 output buffer"
    ref = TorchAttentionImpl(4, 16, 1.0, 1)._forward_generic(
        q.reshape(1, -1), cache, meta)
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)


@requires_cuda
def test_do_kv_cache_update_uses_the_triton_write_path():
    _, cache = _cache_pairs(3, 4, 1, 8)
    k = torch.randn(1, 1, 8, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    impl = TritonAttentionImpl(4, 8, 1.0, 1)
    impl.do_kv_cache_update(
        None, k.reshape(1, -1), v.reshape(1, -1), cache,
        torch.tensor([6], device="cuda", dtype=torch.int64))
    torch.testing.assert_close(cache[0, 1, 2, 0], k[0, 0])
    torch.testing.assert_close(cache[1, 1, 2, 0], v[0, 0])


@requires_cuda
def test_forward_writes_kv_and_returns_output_buffer_alias():
    torch.manual_seed(1)
    heads, head_size, kv_heads = 4, 16, 1
    block_size = 4
    _, cache = _cache_pairs(3, block_size, kv_heads, head_size,
                            dtype=torch.float32)
    q = torch.randn(1, heads * head_size, device="cuda")
    k = torch.randn(1, kv_heads * head_size, device="cuda")
    v = torch.randn(1, kv_heads * head_size, device="cuda")
    qsl = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seq = torch.tensor([1], device="cuda", dtype=torch.int32)
    bt = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    meta = AttentionMetadata(
        query_start_loc=qsl, seq_lens=seq, block_table=bt,
        slot_mapping=torch.tensor([5], device="cuda", dtype=torch.int64),
        block_size=block_size, num_reqs=1, max_query_len=1, causal=True)
    meta.workspace_pool = TritonAttentionWorkspacePool()
    impl = TritonAttentionImpl(heads, head_size, 1.0, kv_heads)
    buf = torch.zeros(1, heads * head_size, device="cuda")
    layer = SimpleNamespace(per_layer_sliding_window=None,
                            _output_buffer=buf)
    got = impl.forward(layer, q, k, v, cache, meta)
    assert got.data_ptr() == buf.data_ptr()
    torch.testing.assert_close(cache[0, 1, 1], k.view(kv_heads, head_size))
    torch.testing.assert_close(cache[1, 1, 1], v.view(kv_heads, head_size))


def test_deleted_fallbacks_are_gone_and_config_rejects_old_name():
    for module in (
            "minivllm.attention.backends.triton_paged",
            "minivllm.attention.ops.fixed_attention"):
        assert importlib.util.find_spec(module) is None, f"{module} 应已删除"
    with pytest.raises(ValueError, match="attention_backend"):
        get_attention_backend("triton_paged")
    assert TritonAttentionBackend.get_name() == "TRITON_ATTN"
    assert TritonAttentionBackend.get_builder_cls()._cudagraph_support.name == "ALWAYS"
    assert not hasattr(TritonAttentionImpl, "_forward_generic")
