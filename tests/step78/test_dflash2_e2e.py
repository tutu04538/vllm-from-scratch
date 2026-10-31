"""step78：DFlash2 的 V2 端到端（tiny 模型，greedy 无损性 + 图/eager 一致 + slot 复用）。

为什么这里不做"真实权重"的结论：本机没有 DFlash2 checkpoint（需求 078 §5 要求把 tiny 与
真实权重分开报告）。所以本文件只证明**机制**：
  - 投机输出与不投机的 greedy 输出逐 token 相同（拒绝采样无损）
  - FULL 图与 eager 的逐轮草稿相同（候选选择 + walk + 落 q 都在图里）
  - slot 被复用（请求数 > max_num_seqs）时输出仍与对照相同（旧候选列清理生效）
"""

import pytest
import torch

from dflash2_helpers import make_baseline_engine, make_engine, run_prompts

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DFlash2 的 V2 通路需要 CUDA（Triton 内核 + UVA），只在 CUDA 上执行")

PROMPTS = [("a", [1, 2, 3, 4]), ("b", [5, 6])]


@requires_cuda
def test_greedy_speculation_is_lossless():
    """DFlash2 的 greedy 输出必须与同 target 的**无投机** greedy 逐 token 相同。"""
    baseline_engine, _, _ = make_baseline_engine()
    baseline = run_prompts(baseline_engine, PROMPTS, max_tokens=6, temperature=0.0)

    engine, _, runner = make_engine(k=3, draft_sample_method="greedy")
    assert runner.speculator.__class__.__name__ == "DFlash2Speculator"
    speculative = run_prompts(engine, PROMPTS, max_tokens=6, temperature=0.0)
    assert speculative == baseline, (speculative, baseline)


@requires_cuda
def test_graph_and_eager_produce_the_same_drafts():
    """FULL 图里录的是"主干 + 候选 + walk + 落 q"：逐轮草稿必须与 eager 相同。"""
    prompts = PROMPTS
    rounds, finals = {}, {}
    for mode in ("none", "full_decode_only"):
        engine, _core, runner = make_engine(
            k=3, draft_sample_method="probabilistic", mode=mode,
            capture_sizes=(8, 16))
        try:
            for req_id, prompt in prompts:
                engine.add_request(req_id, list(prompt), _sampling_params())
            snapshot, final = [], {}
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    final[out.request_id] = list(out.token_ids)
                snapshot.append(runner.req_states.draft_tokens[:2].clone())
            if mode == "full_decode_only":
                stats = runner.cudagraph_capture_stats
                selections = list(runner.speculator.query_cudagraph_manager.selections)
        finally:
            engine.shutdown()
        rounds[mode] = snapshot
        finals[mode] = final
        if mode == "full_decode_only":
            assert stats and stats.get("captured", 0) >= 1, stats
            assert selections

    assert finals["none"] == finals["full_decode_only"]
    assert len(rounds["none"]) == len(rounds["full_decode_only"]) > 1
    for index, (eager, graph) in enumerate(zip(rounds["none"], rounds["full_decode_only"])):
        assert torch.equal(eager, graph), f"第 {index} 轮的草稿 token 不同"


def _sampling_params():
    from minivllm import SamplingParams

    return SamplingParams(max_tokens=6, temperature=0.7, seed=1234, eos_token_id=99)


@requires_cuda
def test_greedy_and_probabilistic_agree_at_temperature_zero():
    """温度 0 时两条草稿路径都退化成 argmax：最终 token 必须相同（无损性）。"""
    outputs = {}
    for method in ("greedy", "probabilistic"):
        engine, _, runner = make_engine(k=3, draft_sample_method=method)
        assert (runner.speculator.draft_logits is not None) == (method == "probabilistic")
        outputs[method] = run_prompts(engine, PROMPTS, max_tokens=6, temperature=0.0)
    assert outputs["greedy"] == outputs["probabilistic"]


@requires_cuda
def test_slot_reuse_keeps_the_output_identical():
    """`max_num_seqs=1` + 三条请求：slot 0 被反复复用，旧候选列必须被清掉。

    如果 `_cached_candidate_ids` 的清理失效，q 里会残留上一请求的候选列，拒绝采样会按
    错误的 proposal 分布工作——这里用"与不投机 greedy 相同"来盯住。
    """
    prompts = [("r0", [1, 2]), ("r1", [3, 4]), ("r2", [5, 6])]
    baseline_engine, _, _ = make_baseline_engine(max_num_seqs=1)
    baseline = run_prompts(baseline_engine, prompts, max_tokens=5, temperature=0.0)

    engine, _core, runner = make_engine(k=2, draft_sample_method="probabilistic",
                                        max_num_seqs=1)
    speculative = run_prompts(engine, prompts, max_tokens=5, temperature=0.0,
                              shutdown=False)
    try:
        assert speculative == baseline, (speculative, baseline)
        # slot 0 的缓存候选 id 必须在最后一轮之后仍然只有 k 列非哨兵值
        assert runner.speculator._cached_candidate_ids[0].numel() == 2 * 3
    finally:
        engine.shutdown()


@requires_cuda
def test_dflash2_draft_follows_target_triton_and_graph_matches_eager():
    """087 收口：DFlash2 草稿层也跟随 target 的 Triton backend。"""
    from minivllm.attention.backends.triton_attn import TritonAttentionImpl

    prompts = [("a", [1, 2, 3, 4])]
    eager_engine, _, eager_runner = make_engine(
        k=2, draft_sample_method="greedy", mode="none",
        attention_backend="triton")
    eager = run_prompts(eager_engine, prompts, max_tokens=6, shutdown=False)
    eager_engine.shutdown()

    graph_engine, _, graph_runner = make_engine(
        k=2, draft_sample_method="greedy", mode="full_decode_only",
        capture_sizes=(3, 6), attention_backend="triton")
    graph = run_prompts(graph_engine, prompts, max_tokens=6, shutdown=False)
    try:
        assert graph == eager
        assert graph_runner.attention_backend.get_name() == "TRITON_ATTN"
        attn_layers = [
            m for _, m in graph_runner.speculator.model.named_modules()
            if hasattr(m, "impl")
        ]
        assert attn_layers
        assert all(isinstance(m.impl, TritonAttentionImpl) for m in attn_layers)
        assert graph_runner.speculator.metadata_builder.__class__.__name__ == (
            "TritonAttentionMetadataBuilder")
        manager = graph_runner.speculator.query_cudagraph_manager
        assert manager is not None and any(e["replayed"] for e in manager.selections)
    finally:
        graph_engine.shutdown()
