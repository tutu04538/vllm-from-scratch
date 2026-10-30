"""087：3D 工作区所有权、metadata 池复用与 Graph 内有效长度。

这一组不跑引擎，只钉住三条最容易回归的边界：

    W1  builder 持有池，所有 metadata 拿到同一个池；没有进程级缓存；
    W2  shutdown 释放后 tensor 真能被回收（弱引用为 None）；
    W3  CUDA Graph 重放时 kernel 从设备上的 `seq_lens` 读长度，而不是捕获时的标量。

W3 用 2D/3D 两条 kernel 路径各录一张图：先按 seq_len=4/5 捕获，再把同一块
`seq_lens` 改成 8/5 重放，输出必须等于用新长度现场算的 eager 结果。
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from spec87_helpers import requires_cuda

from minivllm.attention.backends.triton_attn import (
    TritonAttentionMetadataBuilder,
    TritonAttentionWorkspacePool,
)
from minivllm.attention.ops.triton_unified_attention import (
    make_3d_workspace,
    unified_attention,
)


def _metadata(builder):
    device = torch.device("cpu")
    return builder.build(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int64),
        num_reqs=1,
        uniform_query_len=3,
        max_query_len=3,
    )


def test_builder_owns_one_pool_for_all_metadata():
    builder = TritonAttentionMetadataBuilder(4)
    first = _metadata(builder)
    second = _metadata(builder)
    assert first.workspace_pool is builder.workspace_pool
    assert second.workspace_pool is builder.workspace_pool
    # 同一 builder 的每一步/每层都借同一个池；池本身可以还没有分配 tensor。
    assert isinstance(builder.workspace_pool, TritonAttentionWorkspacePool)
    assert len(builder.workspace_pool) == 0


def test_workspace_bucket_reuses_address_and_release_frees_tensors():
    builder = TritonAttentionMetadataBuilder(4)
    pool = builder.workspace_pool
    workspace = pool.get("cpu", 4, 16, 2, 16)
    again = pool.get("cpu", 4, 16, 2, 16)
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(workspace, again))
    assert len(pool) == 1

    refs = [weakref.ref(t) for t in workspace]
    del workspace, again
    builder.release()
    assert len(pool) == 0
    gc.collect()
    assert all(ref() is None for ref in refs), "release 后工作区 tensor 仍被引用"


def test_no_process_global_workspace_cache():
    import minivllm.attention.backends.triton_attn as backend

    assert not hasattr(backend, "_WORKSPACE_CACHE")
    assert not hasattr(backend, "get_3d_workspace")


def _case(*, q_len, seq_len, heads=4, head_size=16, block_size=4):
    torch.manual_seed(7)
    num_blocks = 2
    q = torch.randn(q_len, heads, head_size, device="cuda")
    cache = torch.randn(
        2, num_blocks, block_size, 1, head_size, device="cuda")
    block_table = torch.tensor(
        [[0, 1]], device="cuda", dtype=torch.int32)
    qsl = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    seq = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    return q, cache, block_table, qsl, seq, out


def _run_2d(q, cache, block_table, qsl, seq, out):
    unified_attention(
        q, cache[0], cache[1], out, qsl, seq, block_table,
        softmax_scale=1.0, causal=True, window_size=(-1, -1),
        max_seqlen_q=q.shape[0])


def _run_3d(q, cache, block_table, qsl, seq, out, workspace):
    unified_attention(
        q, cache[0], cache[1], out, qsl, seq, block_table,
        softmax_scale=1.0, causal=True, window_size=(-1, -1),
        max_seqlen_q=1, seq_threshold_3D=1,
        num_par_softmax_segments=16,
        softmax_segm_output=workspace[0],
        softmax_segm_max=workspace[1],
        softmax_segm_expsum=workspace[2])


@requires_cuda
def test_graph_2d_replay_reads_updated_seq_lens():
    q, cache, block_table, qsl, seq, out = _case(q_len=3, seq_len=4)
    _run_2d(q, cache, block_table, qsl, seq, out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_2d(q, cache, block_table, qsl, seq, out)
    torch.cuda.synchronize()

    for new_len in (8, 5):
        seq.fill_(new_len)
        graph.replay()
        torch.cuda.synchronize()
        want = torch.empty_like(out)
        _run_2d(q, cache, block_table, qsl, seq, want)
        torch.testing.assert_close(out, want, rtol=2e-5, atol=2e-5)


@requires_cuda
def test_graph_3d_replay_reads_updated_seq_lens():
    q, cache, block_table, qsl, seq, out = _case(
        q_len=1, seq_len=5, heads=8, head_size=256)
    workspace = make_3d_workspace("cuda", 8, 256, 1, 16)
    _run_3d(q, cache, block_table, qsl, seq, out, workspace)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_3d(q, cache, block_table, qsl, seq, out, workspace)
    torch.cuda.synchronize()

    seq.fill_(8)  # cache 只有 2 个 block × 4 slot = 8；不能越界测“有效长度”
    graph.replay()
    torch.cuda.synchronize()
    want = torch.empty_like(out)
    _run_3d(q, cache, block_table, qsl, seq, want, workspace)
    torch.testing.assert_close(out, want, rtol=2e-5, atol=2e-5)
