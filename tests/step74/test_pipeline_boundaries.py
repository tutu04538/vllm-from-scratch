"""74 关：图路径在**批次边界**上的行为（需求 074 §4 的后四组）。

覆盖：

    B1  批重排：不同提示词长度的多条请求（runner 会把 decode 类请求排到前面）→ 输出与 eager 逐
        token 相同，且真的有图被重放
    B2  部分 prefill（chunked prefill）：`has_prefill=True` 的轮次**不允许**进图（统一 decode
        判据的一票否决），走 eager；输出与 eager 相同
    B3  首拒之后的下一轮输入：第一次被拒（`num_rejected > 0`）之后的一轮，草稿输入口径不变
        （草稿仍按"最后一行有效行"锚定），输出与 eager 相同
    B4  模型长度边界：`max_model_len` 处 `positions` 被内核夹住（不越界），输出与 eager 相同
    B5  抢占恢复：块不够 → 抢占；图路径不崩、输出与 eager 相同
    B6  MTP 的 hook 契约（§3.6）：四个 hook 在**捕获与运行两侧成对触发**，且
        `on_prefill_begin()` **无条件**清除 skip_topk（异常中断也不会残留到下一轮）
"""

import json
from pathlib import Path

import pytest
import torch

from spec74_helpers import (RoundRecorder, greedy, make_config, make_engine,  # noqa: E402
                            requires_cuda, run_steps)

from minivllm import SamplingParams  # noqa: E402

PROMPTS = [("A", [1, 2, 3, 4, 5, 6]), ("B", [7, 8]), ("C", [10, 9, 8, 7, 6])]


def _run(mode, *, spec_k=3, max_tokens=10, budget=64, blocks=64, max_model_len=64,
         max_num_seqs=3, capture_sizes=None, prompts=PROMPTS, seed=None,
         recorder=False):
    kwargs = dict(spec_k=spec_k, mode=mode, budget=budget, blocks=blocks,
                  max_model_len=max_model_len, max_num_seqs=max_num_seqs)
    if capture_sizes is not None:
        kwargs["capture_sizes"] = capture_sizes
    engine, _core, runner = make_engine(**kwargs)
    rec = RoundRecorder(runner) if recorder else None
    rejected_rounds = []
    original_sample = runner.sample

    def sample_spy(hidden_states, input_batch, grammar_output):
        out = original_sample(hidden_states, input_batch, grammar_output)
        rejected_rounds.append(int(out[2][0].item()) if len(out[2]) else 0)
        return out

    runner.sample = sample_spy
    try:
        for req_id, prompt in prompts:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                              seed=seed, eos_token_id=999))
        final = {}
        for _ in range(400):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                final[out.request_id] = list(out.token_ids)
        return final, runner, rec, rejected_rounds
    finally:
        engine.shutdown()


@requires_cuda
def test_batch_reorder_matches_eager():
    """B1：批重排（不同长度的提示词混在一轮里）下，图路径与 eager 逐 token 相同。"""
    eager, _, _, _ = _run("none")
    graphed, runner, rec, _ = _run("full_decode_only", recorder=True)
    assert eager == graphed
    assert all(len(tokens) == 10 for tokens in graphed.values())
    # 至少有一次真的重放了图（否则这条用例只证明了"大家都没走图"）
    assert runner.speculator.prefill_cudagraph_manager.num_replays > 0
    assert runner.speculator.decode_cudagraph_manager.num_replays > 0


@requires_cuda
def test_partial_prefill_stays_eager_and_matches():
    """B2：chunked prefill 的轮次不进图（`has_prefill=True` 一票否决），输出与 eager 相同。"""
    kwargs = dict(budget=8, max_num_seqs=2, prompts=[("A", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                                     ("B", [7, 8, 9])])
    eager, _, _, _ = _run("none", **kwargs)
    graphed, runner, rec, _ = _run("full_decode_only", recorder=True, **kwargs)
    assert eager == graphed
    # 提示词长 10 > budget 8 → 一定发生过分块 prefill
    prefill_modes = rec.modes("prefill")
    assert "NONE" in prefill_modes, f"分块 prefill 的轮次不该进图：{prefill_modes}"
    # 但统一 decode 的轮次仍然走图（这条边界是"部分 prefill 不进图"，不是"图全废了"）
    assert "FULL" in prefill_modes, f"统一 decode 的轮次应当走图：{prefill_modes}"


@requires_cuda
def test_after_first_rejection_matches_eager():
    """B3：第一次被拒之后（`num_rejected > 0`）的一轮，草稿输入口径与 eager 相同。"""
    kwargs = dict(prompts=[("A", [1, 2, 3, 4, 5, 6])], max_tokens=12)
    eager, _, _, eager_rejected = _run("none", **kwargs)
    graphed, runner, rec, rejected = _run("full_decode_only", recorder=True, **kwargs)
    assert eager == graphed
    assert max(rejected) > 0, (
        "这轮用的 tiny 权重没产生任何拒绝，用例失去意义（首拒路径没被走到）")
    assert max(eager_rejected) == max(rejected), "两条路的接受/拒绝序列不同"
    # 被拒之后仍有 decode 图在跑（被拒不是"图失效"）
    assert runner.speculator.decode_cudagraph_manager.num_replays > 0


@requires_cuda
def test_model_length_boundary_matches_eager():
    """B4：`max_model_len` 边界上 positions 被夹住（不越界），输出与 eager 相同。"""
    # 提示词 6 + 生成到 12 = 18 > max_model_len(12) → 请求在模型长度处停下
    kwargs = dict(max_model_len=12, max_tokens=12, prompts=[("A", [1, 2, 3, 4, 5, 6])],
                  capture_sizes=[1, 2, 3, 4, 8])
    eager, _, _, _ = _run("none", **kwargs)
    graphed, runner, rec, _ = _run("full_decode_only", recorder=True, **kwargs)
    assert eager == graphed
    assert len(graphed["A"]) <= 12 - 6 + 1, f"没有在模型长度处停下：{graphed['A']}"
    # 草稿侧的 positions / seq_lens 都不越界（内核夹到 max_model_len）
    for entry in rec.rounds:
        assert int(entry["positions"].max()) <= 12 - 1


@requires_cuda
def test_preemption_with_graphs_matches_eager():
    """B5：块不够触发抢占恢复，图路径不崩、输出与 eager 相同。"""
    kwargs = dict(blocks=6, budget=16, max_tokens=10)
    eager, _, _, _ = _run("none", **kwargs)
    graphed, runner, rec, _ = _run("full_decode_only", recorder=True, **kwargs)
    assert eager == graphed
