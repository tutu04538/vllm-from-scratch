"""086：TRITON_ATTN 的 V2 eager / FULL 图与“无 Torch fallback”契约。"""

from __future__ import annotations

import pytest
import torch

from spec86_helpers import make_engine, requires_cuda, run_engine
from minivllm.attention.backends.torch_sdpa import TorchAttentionImpl


PROMPTS = [("a", [1, 2, 3, 4]), ("b", [7, 8, 9, 10])]


@requires_cuda
def test_tiny_eager_and_full_graph_match_torch():
    torch_engine, _ = make_engine(
        attention_backend="torch", mode="none", spec_k=None)
    want = run_engine(torch_engine, PROMPTS)
    torch_engine.shutdown()

    eager, _ = make_engine(
        attention_backend="triton", mode="none", spec_k=None)
    got_eager = run_engine(eager, PROMPTS)
    eager.shutdown()

    graph, runner = make_engine(
        attention_backend="triton", mode="full", spec_k=None,
        capture_sizes=(1, 2, 3, 4, 8))
    got_graph = run_engine(graph, PROMPTS)
    try:
        assert got_eager == want
        assert got_graph == want
        assert runner.attention_backend.get_name() == "TRITON_ATTN"
        manager = runner.target_cudagraph_manager
        assert manager is not None and manager.graphs
        # 混合 prefill/decode 也应该走 FULL 图（ALWAYS 能力）。
        assert any(d.uniform_token_count is None for d in manager.graphs)
    finally:
        graph.shutdown()


@requires_cuda
def test_triton_run_never_calls_torch_generic(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("TRITON_ATTN 不应调用 Torch _forward_generic")

    monkeypatch.setattr(TorchAttentionImpl, "_forward_generic", _boom)
    engine, runner = make_engine(
        attention_backend="triton", mode="none", spec_k=None)
    try:
        out = run_engine(engine, PROMPTS[:1], max_tokens=4)
        assert out["a"]
        assert runner.attention_backend.get_name() == "TRITON_ATTN"
    finally:
        engine.shutdown()


@requires_cuda
def test_triton_spec_decode_graph_matches_eager():
    # EAGLE3 tiny draft 的草稿层仍是 Torch；target 的验证 attention
    # 走 unified。这里只要求 target 图/eager 的最终 token 相同。
    eager, _ = make_engine(
        attention_backend="triton", mode="none", spec_k=2,
        capture_sizes=(1, 2, 3, 4, 8))
    want = run_engine(eager, PROMPTS[:1], max_tokens=5)
    eager.shutdown()

    graph, runner = make_engine(
        attention_backend="triton", mode="full_decode_only", spec_k=2,
        capture_sizes=(1, 2, 3, 4, 8))
    got = run_engine(graph, PROMPTS[:1], max_tokens=5)
    try:
        assert got == want
        assert runner.attention_backend.get_name() == "TRITON_ATTN"
    finally:
        graph.shutdown()
