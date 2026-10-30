"""087：真实 Gemma4 target/assistant 的 eager vs FULL Graph。

验收不以“最终 token 一样”为唯一证据：同时在 `sample()` 入口抓取 target 的 hidden 与
logits（Graph 重放写静态缓冲，必须在本轮 clone）；K=2 还检查 assistant 的 Q-only
metadata 组确实切到 Triton，且 target/assistant 的 3D 工作区按预期分桶。

真实权重路径与 086 固定 revision 相同；没有权重时整组 skip。
"""

from __future__ import annotations

import gc
import weakref

import torch

from spec87_helpers import (
    REAL_PROMPT, REAL_SHORT_PROMPT, CaptureTargetHiddens, close_engine,
    make_real_engine,
    requires_cuda, requires_models, run_engine,
)

from minivllm import SamplingParams
from minivllm.attention.backends.triton_attn import (
    TritonAttentionMetadataBuilder,
)


def _assert_records_close(eager, graph):
    # K=2 用例在 greedy 对照之后还会追加一条随机请求制造拒绝；这里只比较
    # greedy 那段（graph 的前缀），不能把随机轮的记录当成同一 workload。
    assert len(graph.records) >= len(eager.records), (
        f"graph 记录少于 eager：eager={len(eager.records)} "
        f"graph={len(graph.records)}")
    for step, (a, b) in enumerate(zip(eager.records, graph.records)):
        assert a["hidden"].shape == b["hidden"].shape, f"step {step} hidden 形状不同"
        torch.testing.assert_close(
            a["hidden"], b["hidden"], rtol=2e-2, atol=2e-2)
        assert a["logits"].shape == b["logits"].shape
        torch.testing.assert_close(
            a["logits"], b["logits"], rtol=2e-2, atol=2e-2)
        assert (a["logits"].argmax(dim=-1)
                == b["logits"].argmax(dim=-1)).all(), f"step {step} argmax 不同"


def _snapshot_null_block(runner):
    """0 号块是留白/垃圾桶；087 的 Triton 写入应跳过 padding 的 -1 slot。"""
    return {
        name: cache[:, 0].detach().clone()
        for name, cache in runner.kv_caches.items()
        if cache is not None
    }


def _assert_null_block_unchanged(runner, before):
    for name, want in before.items():
        got = runner.kv_caches[name][:, 0]
        assert torch.equal(got, want), f"0 号留白块被写：{name}"


@requires_cuda
@requires_models
def test_real_target_k0_full_graph_matches_eager():
    eager, eager_runner = make_real_engine(k=0, mode="none")
    eager_capture = CaptureTargetHiddens(eager_runner)
    want = run_engine(eager, [("r", REAL_SHORT_PROMPT)], max_tokens=4)
    close_engine(eager)

    graph, runner = make_real_engine(
        k=0, mode="full", capture_sizes=(1, 2, 3, 4, 8))
    null_before = _snapshot_null_block(runner)
    graph_capture = CaptureTargetHiddens(runner)
    got = run_engine(graph, [("r", REAL_SHORT_PROMPT)], max_tokens=4)

    manager = runner.target_cudagraph_manager
    assert manager is not None and manager.graphs
    assert any(d.cg_mode.name == "FULL" for d in manager.graphs)
    pool = runner.attn_metadata_builder.workspace_pool
    # K=0 单 token decode 走 3D；target 有 sliding 256 / full 512 两个桶。
    assert len(pool) == 2, f"3D 工作区桶数不对：{pool.keys()}"
    workspace = next(iter(pool._workspaces.values()))
    refs = [weakref.ref(t) for t in workspace]
    del workspace
    try:
        _assert_records_close(eager_capture, graph_capture)
        assert got == want
        _assert_null_block_unchanged(runner, null_before)
    finally:
        close_engine(graph)
    gc.collect()
    assert all(ref() is None for ref in refs), "引擎 shutdown 后工作区仍未释放"
    assert len(pool) == 0
    assert runner.attn_metadata_builder is None
    assert runner.target_cudagraph_manager is None


@requires_cuda
@requires_models
def test_real_target_and_assistant_k2_graph_matches_eager():
    eager, eager_runner = make_real_engine(k=2, mode="none")
    eager_capture = CaptureTargetHiddens(eager_runner)
    want = run_engine(eager, [("r", REAL_SHORT_PROMPT)], max_tokens=4)
    assert len(eager_runner.attn_metadata_builder.workspace_pool) == 0, (
        "K=2 target 的 eager 3 行 decode 走 2D，不应有 3D 工作区")
    close_engine(eager)

    graph, runner = make_real_engine(
        k=2, mode="full", capture_sizes=(1, 2, 3, 4, 8))
    null_before = _snapshot_null_block(runner)
    graph_capture = CaptureTargetHiddens(runner)
    got = run_engine(graph, [("r", REAL_SHORT_PROMPT)], max_tokens=4)

    manager = runner.target_cudagraph_manager
    assert manager is not None and manager.graphs
    speculator = runner.speculator
    assert speculator is not None
    assert speculator.prefill_cudagraph_manager.num_replays > 0
    assert speculator.decode_cudagraph_manager.num_replays > 0
    # K=2 的真实 target 本轮 query 是 3 行，实际 decode 选择的是 3 行 bucket；
    # 图捕获同时会录 1/2/4/8 行的 mixed bucket，其中 1 行 graph 走 3D，所以
    # **target builder 里出现工作区并不等于 decode 用了 3D**。这里钉住 greedy
    # 这段真正运行到的 target bucket 全是 3 行。
    assert manager.selections
    assert all(sel["desc_num_tokens"] == 3 for sel in manager.selections)

    # 随机采样制造拒绝：验证 Graph 重放之后，下一轮的 rejected 计数/有效长度
    # 仍由设备状态修正；只要求统计里真的出现拒绝，不规定接受率。
    graph.add_request(
        "rand", list(REAL_PROMPT),
        SamplingParams(max_tokens=8, temperature=1.0, seed=0,
                       ignore_eos=True))
    drafted = accepted = 0
    for _ in range(50):
        if not graph.has_unfinished_requests():
            break
        for _out in graph.step():
            pass
        stats = getattr(graph, "last_scheduler_stats", None)
        sd = getattr(stats, "spec_decoding_stats", None)
        if sd is not None:
            drafted += sd.num_draft_tokens
            accepted += sd.num_accepted_tokens
    assert drafted > 0
    assert accepted < drafted, (
        f"Graph 路径没有触发拒绝（drafted={drafted}, accepted={accepted}）")

    # assistant 的 metadata 组也必须跟随 target 的 Triton 后端；Gemma4 assistant
    # 有 sliding/full 两个 KV group，因此是两个组、两个 builder。
    groups = speculator.attn_groups[0]
    assert len(groups) == 2, f"assistant KV group 数不对：{len(groups)}"
    assert all(group.backend.get_name() == "TRITON_ATTN"
               for group in groups)
    assert all(isinstance(group.get_metadata_builder(),
                          TritonAttentionMetadataBuilder)
               for group in groups)
    assistant_layers = [
        m for _, m in speculator.model.named_modules()
        if hasattr(m, "impl") and getattr(m, "q_only", False)
    ]
    assert len(assistant_layers) == 4
    assert all(type(m.impl).__name__ == "TritonAttentionImpl"
               for m in assistant_layers)

    # K=2 的真实 target 本轮 query 是 3 行；assistant 的 1 行 decode 才用 3D。
    assistant_pools = [group.get_metadata_builder().workspace_pool
                       for group in groups]
    assert [len(pool) for pool in assistant_pools] == [1, 1], (
        f"assistant 的 sliding/full 应各有一个 3D 桶："
        f"{[pool.keys() for pool in assistant_pools]}")
    refs = []
    for pool in assistant_pools:
        workspace = next(iter(pool._workspaces.values()))
        refs.extend(weakref.ref(t) for t in workspace)
    del workspace

    try:
        _assert_records_close(eager_capture, graph_capture)
        assert got == want
        _assert_null_block_unchanged(runner, null_before)
    finally:
        close_engine(graph)
    gc.collect()
    assert all(ref() is None for ref in refs), "assistant 工作区未随引擎释放"
    assert all(len(pool) == 0 for pool in assistant_pools)
    assert runner.attn_metadata_builder is None
    assert runner.speculator is None
