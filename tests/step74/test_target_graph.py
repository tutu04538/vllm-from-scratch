"""Phase 1：V2 target 的 FULL_DECODE_ONLY CUDA Graph。

这些用例只覆盖统一 decode 批（每请求 1+K 行）；混合 prefill/decode 在 Phase 1 明确回退
eager，varlen decode 属 Phase 2，PIECEWISE 属 Phase 3。
"""

from __future__ import annotations

import pytest
import torch

from spec74_helpers import make_engine, requires_cuda
from minivllm import SamplingParams

EOS = 99


@pytest.fixture(autouse=True)
def _no_leaked_breakable_capture():
    from minivllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    yield
    assert BreakableCUDAGraphCapture.current() is None, "测试结束后仍有活跃 breakable capture"


def _run(engine, prompts, *, max_tokens=4, shutdown=True):
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt), SamplingParams(
            max_tokens=max_tokens, temperature=0.0, eos_token_id=EOS))
    outputs: dict[str, list[int]] = {}
    for _ in range(100):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    if shutdown:
        engine.shutdown()
    return outputs


@requires_cuda
def test_target_full_decode_graph_matches_eager():
    prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7])]
    eager, _, _ = make_engine(mode="none", capture_sizes=[4, 8], max_num_seqs=2)
    eager_out = _run(eager, prompts)

    graph, _, runner = make_engine(
        mode="full", capture_sizes=[4, 8], max_num_seqs=2)
    graph_out = _run(graph, prompts, shutdown=False)
    try:
        manager = runner.target_cudagraph_manager
        assert manager is not None
        assert graph_out == eager_out
        assert manager.graphs
        assert all(d.cg_mode.name == "FULL" for d in manager.graphs)
        assert any(sel["mode"] == "FULL" and sel["replayed"]
                   for sel in manager.selections)
    finally:
        graph.shutdown()


@requires_cuda
def test_target_graph_pads_real_smaller_batch():
    """真实 1 请求 × 2 行命中捕获的 2 请求 × 2 行图；padding 行不能改变输出。"""
    prompts = [("a", [1, 2, 3])]
    eager, _, _ = make_engine(
        spec_k=None, mode="none", capture_sizes=[2], max_num_seqs=2)
    eager_out = _run(eager, prompts)

    graph, _, runner = make_engine(
        spec_k=None, mode="full", capture_sizes=[2], max_num_seqs=2)
    graph_out = _run(graph, prompts, shutdown=False)
    try:
        manager = runner.target_cudagraph_manager
        assert manager is not None
        assert graph_out == eager_out
        padded = [s for s in manager.selections if s["mode"] == "FULL"]
        assert padded
        assert any(s["num_tokens"] == 1 and s["desc_num_tokens"] == 2
                   and s["num_reqs"] == 1 and s["desc_num_reqs"] == 2
                   for s in padded)
    finally:
        graph.shutdown()


@requires_cuda
def test_target_graph_dispatches_piecewise_for_mixed_batch():
    """混合 prefill/decode 没有 uniform_token_count：Phase 3 起命中 PIECEWISE。"""
    engine, _, runner = make_engine(
        mode="full", capture_sizes=[4, 8], max_num_seqs=2)
    try:
        manager = runner.target_cudagraph_manager
        assert manager is not None
        desc = manager.dispatch(
            num_reqs=2, num_tokens=5, uniform_token_count=None,
            max_query_len=3)
        assert desc.cg_mode.name == "PIECEWISE"
        assert desc.num_reqs is None
    finally:
        engine.shutdown()


@requires_cuda
def test_target_output_buffer_is_static_and_capture_precedes_speculator():
    engine, _, runner = make_engine(
        mode="full", capture_sizes=[4, 8], max_num_seqs=2)
    try:
        manager = runner.target_cudagraph_manager
        assert manager is not None
        assert manager.hidden_states is not None
        ptr = manager.hidden_states.data_ptr()
        stats = runner.cudagraph_capture_stats
        assert stats["target"]["captured"] > 0
        assert stats["graphs"]["prefill"]["captured"] > 0
        _run(engine, [("a", [1, 2, 3, 4]), ("b", [5, 6, 7])], shutdown=False)
        assert manager.hidden_states.data_ptr() == ptr
    finally:
        engine.shutdown()


@requires_cuda
def test_target_graph_without_speculator():
    """非投机 decode 也走同一套 target 图（decode_query_len=1）。"""
    prompts = [("a", [1, 2, 3])]
    eager, _, _ = make_engine(spec_k=None, mode="none", capture_sizes=[2],
                              max_num_seqs=2)
    eager_out = _run(eager, prompts)
    graph, _, runner = make_engine(spec_k=None, mode="full", capture_sizes=[2],
                                   max_num_seqs=2)
    graph_out = _run(graph, prompts, shutdown=False)
    try:
        assert graph_out == eager_out
        assert runner.target_cudagraph_manager is not None
        assert runner.target_cudagraph_manager.graphs
    finally:
        graph.shutdown()


def _run_mixed(engine, runner, *, shutdown=True):
    """先跑一条 decode，再插入一条新 prefill，制造混合批。"""
    engine.add_request("a", [1, 2, 3, 4, 5, 6], SamplingParams(
        max_tokens=6, temperature=0.0, eos_token_id=EOS))
    outputs: dict[str, list[int]] = {}
    for step in range(30):
        if step == 1:
            engine.add_request("b", [7, 8, 9, 10], SamplingParams(
                max_tokens=4, temperature=0.0, eos_token_id=EOS))
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    if shutdown:
        engine.shutdown()
    return outputs


@requires_cuda
def test_target_breakable_piecewise_mixed_batch_matches_eager():
    """Phase 3：混合 prefill/decode 批走 breakable PIECEWISE，输出与 eager 相同。"""
    eager, _, _ = make_engine(mode="none", capture_sizes=[4, 8, 12, 16],
                              max_num_seqs=2, budget=64)
    eager_out = _run_mixed(eager, None)

    graph, _, runner = make_engine(mode="full", capture_sizes=[4, 8, 12, 16],
                                   max_num_seqs=2, budget=64)
    graph_out = _run_mixed(graph, runner, shutdown=False)
    try:
        assert graph_out == eager_out
        manager = runner.target_cudagraph_manager
        assert manager is not None and manager.pw_model is not None
        assert manager.pw_model.entries
        assert manager.pw_model.num_replays > 0
        pw = [s for s in manager.selections if s["mode"] == "PIECEWISE"]
        assert pw
        assert all(s["desc_num_reqs"] is None for s in pw)
    finally:
        graph.shutdown()
