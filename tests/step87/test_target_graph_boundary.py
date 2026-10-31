"""087：target Graph 的桶、有效长度与容量边界（tiny Qwen3/EAGLE3）。

覆盖需求 §3/§4/§6 里不依赖真实 Gemma4 的部分：

    T1  同一形状 bucket 连续换不同 prompt/seq_len：图不重捕，内容由常驻缓冲更新；
    T2  捕获 metadata 的 seq_lens/query_start_loc/block_table/slot_mapping 都是常驻
        缓冲的视图（地址稳定）；
    T3  KV 容量 64 vs 256 个 block 不改变图键/输出；
    T4  K=4 的 5 行 query 仍走 2D，eager/Graph 输出一致。
"""

from __future__ import annotations

import pytest
import torch

from spec87_helpers import (
    make_tiny_engine, requires_cuda, run_engine,
)

from minivllm.config import CUDAGraphMode
from minivllm.worker.gpu.cudagraph_utils import prepare_inputs_to_capture


def _one(engine, prompt, *, max_tokens=2):
    return run_engine(engine, [("r", list(prompt))], max_tokens=max_tokens)["r"]


@requires_cuda
def test_same_bucket_swaps_seq_len_without_recapture():
    prompt_a = [1, 2, 3, 4]
    prompt_b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    eager, _ = make_tiny_engine(mode="none", spec_k=2)
    want_a = _one(eager, prompt_a)
    want_b = _one(eager, prompt_b)
    eager.shutdown()

    graph, runner = make_tiny_engine(
        mode="full", spec_k=2, capture_sizes=(1, 2, 3, 4, 8, 16))
    manager = runner.target_cudagraph_manager
    assert manager is not None
    before_graphs = set(manager.graphs)
    before_replays = manager.num_replays

    got_a = _one(graph, prompt_a)
    got_b = _one(graph, prompt_b)
    try:
        assert got_a == want_a
        assert got_b == want_b
        # 两个 prompt 的 decode 都是 3 行：同一个 bucket，不能因为 seq_len 不同重捕。
        assert set(manager.graphs) == before_graphs
        assert manager.num_replays > before_replays
        decode_sels = [s for s in manager.selections
                       if s["mode"] == "FULL" and s["desc_num_tokens"] == 3]
        assert len(decode_sels) >= 2
        assert all(s["replayed"] for s in decode_sels)
    finally:
        graph.shutdown()


@requires_cuda
def test_capture_metadata_references_persistent_buffers():
    engine, runner = make_tiny_engine(mode="full", spec_k=2)
    try:
        state = prepare_inputs_to_capture(
            num_reqs=1,
            num_tokens=3,
            input_buffers=runner.input_buffers,
            block_tables=runner.block_tables,
            build_attn_metadata=runner._build_target_capture_attn_metadata,
            full_cudagraph=True,
        )
        metadata = next(iter(state.attn_metadata.values()))
        assert metadata.seq_lens.data_ptr() == runner.input_buffers.seq_lens.data_ptr()
        assert (metadata.query_start_loc.data_ptr()
                == runner.input_buffers.query_start_loc.data_ptr())
        assert (metadata.block_table.data_ptr()
                == runner.block_tables.input_block_tables[0].data_ptr())
        assert (metadata.slot_mapping.data_ptr()
                == runner.block_tables.slot_mappings.data_ptr())
        # 087：池也必须是 builder 的那一个，不能在 Impl 里再建。
        assert metadata.workspace_pool is runner.attn_metadata_builder.workspace_pool
        # 图键不包含 seq_len；它只有模式/token 数/请求数/query 形状。
        desc = runner.target_cudagraph_manager.dispatch(
            1, 3, 3, max_query_len=3)
        assert desc.cg_mode == CUDAGraphMode.FULL  # 形状已捕获
        for captured in runner.target_cudagraph_manager._capture_descs.get(
                CUDAGraphMode.FULL, []):
            assert not hasattr(captured, "seq_len")
    finally:
        engine.shutdown()


@requires_cuda
def test_capacity_change_does_not_change_graph_or_output():
    prompt = [1, 2, 3, 4, 5]
    # 块表宽度由 max_model_len / block_size 决定；物理块数与之配套。
    small, small_runner = make_tiny_engine(
        mode="full", spec_k=2, blocks=64, max_model_len=32,
        capture_sizes=(1, 2, 3, 4, 8, 16))
    large, large_runner = make_tiny_engine(
        mode="full", spec_k=2, blocks=128, max_model_len=64,
        capture_sizes=(1, 2, 3, 4, 8, 16))
    try:
        small_blocks = small_runner.block_tables.input_block_tables[0].shape[1]
        large_blocks = large_runner.block_tables.input_block_tables[0].shape[1]
        assert large_blocks > small_blocks, "容量测试没有真的改变块表宽度"
        assert _one(small, prompt) == _one(large, prompt)
        small_graphs = set(small_runner.target_cudagraph_manager.graphs)
        large_graphs = set(large_runner.target_cudagraph_manager.graphs)
        assert small_graphs == large_graphs
    finally:
        small.shutdown()
        large.shutdown()


@requires_cuda
def test_k4_target_graph_matches_eager():
    prompt = [1, 2, 3, 4]
    eager, _ = make_tiny_engine(mode="none", spec_k=4)
    want = _one(eager, prompt)
    eager.shutdown()

    graph, runner = make_tiny_engine(
        mode="full", spec_k=4, capture_sizes=(1, 2, 3, 4, 5, 8, 16))
    got = _one(graph, prompt)
    try:
        assert got == want
        manager = runner.target_cudagraph_manager
        assert manager is not None
        # K=4 的 target 每请求 5 行，只能走 2D；图键里应有 5 行 bucket。
        assert any(desc.uniform_token_count in (None, 5)
                   for desc in manager.graphs)
        assert any(s["desc_num_tokens"] == 5 for s in manager.selections)
    finally:
        graph.shutdown()


@requires_cuda
def test_draft_round_state_matches_eager_across_prefill_and_decode_graphs():
    """目标图 + 草稿两张图都重放时，逐轮草稿状态与 eager 相同。

    首次 propose 走 draft prefill 图（每请求 1+K 行），后续走 draft decode 图
    （每请求 1 行）；Qwen3/EAGLE3 的草稿层也跟随 target 的 Triton backend，这里
    同时验证图生命周期/输入缓冲与后端切换没有改坏逐轮状态。
    """
    from spec87_helpers import DraftRoundRecorder

    prompt = [1, 2, 3]
    eager, eager_runner = make_tiny_engine(mode="none", spec_k=2)
    eager_rec = DraftRoundRecorder(eager_runner)
    want = run_engine(eager, [("r", prompt)], max_tokens=4)
    eager_rounds = eager_rec.rounds
    eager.shutdown()

    graph, graph_runner = make_tiny_engine(
        mode="full", spec_k=2, capture_sizes=(1, 2, 3, 4, 8, 16))
    graph_rec = DraftRoundRecorder(graph_runner)
    got = run_engine(graph, [("r", prompt)], max_tokens=4)
    graph_rounds = graph_rec.rounds
    try:
        assert len(graph_rounds) == len(eager_rounds) > 0
        for idx, (a, b) in enumerate(zip(eager_rounds, graph_rounds)):
            assert a["current_step"] == b["current_step"], f"round {idx} step"
            # 物理块号由各引擎自己的 BlockPool 分配，不跨引擎比较；这里比较
            # 由输入/位置决定的草稿状态，slot_mapping 的正确性由上面的 kernel
            # 差分和既有 step74 测试覆盖。
            for key in ("draft_tokens", "positions", "seq_lens"):
                assert torch.equal(a[key], b[key]), f"round {idx} {key} 不同"
        assert got == want
        # 首轮 propose 吃 target prefill 的行；按现有策略 prefill 批不硬塞
        # 草稿 prefill 图（uniform_token_count 在 has_prefill 时为 None），走 eager。
        # 后续 decode 轮才让草稿 prefill 图（每轮位置 0）和 decode 图（K−1 步）
        # 都真正重放；这是明确边界，不是“Graph 不支持 prefill”。
        assert graph_rounds[0]["has_prefill"] is True
        assert graph_rounds[0]["prefill_replays"] == 0
        later = [r for r in graph_rounds if not r["has_prefill"]]
        assert later, "没有任何后续草稿 decode 轮"
        assert all(r["prefill_replays"] > 0 for r in later)
        assert all(r["decode_replays"] > 0 for r in later)
        manipulator = graph_runner.speculator
        assert manipulator.prefill_cudagraph_manager.num_replays > 0
        assert manipulator.decode_cudagraph_manager.num_replays > 0
    finally:
        graph.shutdown()


def test_attention_metadata_path_has_no_d2h_sync_tokens():
    """087 的静态边界：metadata/输入准备/Impl 入口不得重新引入 D2H 取值。

    采样提交结果时的 D2H 是合法边界，不在这几个函数里；这里只检查 attention
    热路径。`seq_lens.numel()` 是 host 侧形状，不是读设备值，允许。
    """
    import inspect

    from minivllm.attention.backends.triton_attn import TritonAttentionImpl
    from minivllm.worker.gpu.cudagraph_utils import prepare_inputs_to_capture
    from minivllm.worker.gpu.model_runner import GPUModelRunner

    funcs = [
        TritonAttentionImpl._run,
        TritonAttentionImpl.forward,
        TritonAttentionImpl.forward_q_only,
        TritonAttentionImpl.do_kv_cache_update,
        GPUModelRunner._build_attn_metadata,
        prepare_inputs_to_capture,
    ]
    banned = (".item(", ".cpu(", ".tolist(")
    for func in funcs:
        source = inspect.getsource(func)
        for token in banned:
            assert token not in source, (
                f"{func.__qualname__} 里出现了 {token}：会把设备值拉回 CPU，"
                "破坏 Graph/metadata 边界")
