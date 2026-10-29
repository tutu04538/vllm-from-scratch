"""Triton paged 后端的 V2 端到端：FULL 混合批 + 投机 target 图。

目标不是性能，而是验证 `AttentionCGSupport.ALWAYS` 真的兑现：
混合 prefill/decode 也走一张 FULL 图，且输出与 Torch eager 逐 token 相同。
"""

from __future__ import annotations

from spec74_helpers import make_engine, requires_cuda
from minivllm import SamplingParams
from minivllm.config import CUDAGraphMode

EOS = 99


def _run_mixed(engine, *, shutdown=True):
    engine.add_request("a", [1, 2, 3, 4], SamplingParams(
        max_tokens=5, temperature=0.0, eos_token_id=EOS))
    outputs = {}
    for step in range(30):
        if step == 1:
            engine.add_request("b", [7, 8, 9, 10], SamplingParams(
                max_tokens=3, temperature=0.0, eos_token_id=EOS))
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    if shutdown:
        engine.shutdown()
    return outputs


def _run_decode(engine, *, shutdown=True):
    engine.add_request("a", [1, 2, 3, 4], SamplingParams(
        max_tokens=4, temperature=0.0, eos_token_id=EOS))
    engine.add_request("b", [5, 6, 7], SamplingParams(
        max_tokens=4, temperature=0.0, eos_token_id=EOS))
    outputs = {}
    for _ in range(30):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    if shutdown:
        engine.shutdown()
    return outputs


@requires_cuda
def test_triton_full_mixed_batch_matches_torch_eager():
    eager, _, _ = make_engine(spec_k=None, mode="none",
                              capture_sizes=[4, 8, 12], max_num_seqs=2)
    want = _run_mixed(eager)

    graph, _, runner = make_engine(
        spec_k=None, mode="full", capture_sizes=[4, 8, 12], max_num_seqs=2,
        attention_backend="triton")
    got = _run_mixed(graph, shutdown=False)
    try:
        assert got == want
        assert runner.attention_backend_name == "triton"
        assert runner.cudagraph_mode == CUDAGraphMode.FULL
        manager = runner.target_cudagraph_manager
        assert manager is not None
        mixed = [d for d in manager.graphs
                 if d.uniform_token_count is None and d.max_query_len is None]
        assert mixed, "Triton ALWAYS 后端应捕获混合批 FULL 图"
        assert any(s["mode"] == "FULL" and s["desc_num_tokens"] >= s["num_tokens"]
                   for s in manager.selections)
    finally:
        graph.shutdown()


@requires_cuda
def test_triton_spec_target_graph_matches_torch_eager():
    eager, _, _ = make_engine(
        spec_k=3, mode="none", capture_sizes=[4, 8, 12, 16], max_num_seqs=2)
    want = _run_decode(eager)

    graph, _, runner = make_engine(
        spec_k=3, mode="full", capture_sizes=[4, 8, 12, 16], max_num_seqs=2,
        attention_backend="triton")
    got = _run_decode(graph, shutdown=False)
    try:
        assert got == want
        assert runner.attention_backend_name == "triton"
        assert runner.cudagraph_mode == CUDAGraphMode.FULL
        assert runner.target_cudagraph_manager is not None
        assert runner.target_cudagraph_manager.graphs
    finally:
        graph.shutdown()


@requires_cuda
def test_triton_attn_upstream_kernel_block16_matches_torch():
    """上游 unified_attention 子集：block_size=16、head_dim=16 的真实 kernel。"""
    eager, _, _ = make_engine(
        spec_k=None, mode="none", capture_sizes=[4, 8, 12], max_num_seqs=2,
        block_size=16)
    want = _run_mixed(eager)

    graph, _, runner = make_engine(
        spec_k=None, mode="full", capture_sizes=[4, 8, 12], max_num_seqs=2,
        block_size=16, attention_backend="triton_attn")
    got = _run_mixed(graph, shutdown=False)
    try:
        assert got == want
        assert runner.attention_backend_name == "triton_attn"
        assert runner.attention_backend.get_name() == "TRITON_ATTN_FIXED"
        assert runner.cudagraph_mode == CUDAGraphMode.FULL
        assert runner.target_cudagraph_manager is not None
        assert runner.target_cudagraph_manager.graphs
    finally:
        graph.shutdown()
