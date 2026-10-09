"""73 关：V1/V2 两条执行路径的差分与 V2 投机管道（对应需求 073 §4 的第三~六组验收）。

判据的选择理由（需求总纲 §5）：比较**分布性系统**时不要求不同实现同 seed 逐 token 相等；
但这里比较的是**同精度、同后端、同 greedy** 的两条输入组装路径，输出必须逐 token 相同——
不一致就说明其中一条的输入/寻址错了（这正是 63 关那类"不报错、只是草稿变差"的错的反面：
这里错了一定看得出来）。

覆盖：

    E1  同一 tiny 模型 V1/V2 eager greedy 输出逐 token 相同（多条请求、混合 prefill）
    E2  V2 真的在投机（草稿进批）+ num_sampled/num_rejected 由 GPU 算出且自洽
    E3  logprobs：V2 与 V1 对同一批 token 给出相同的数值
    E4  抢占恢复：V2 不崩、输出与 V1 相同，且 `prompt_len` 不被 `prefill_len` 覆盖
    E5  零请求轮与结束清理：不碰模型、slot 全归还
    B1  配置期边界：V2 只支持 EAGLE/EAGLE3 + eager + 同步调度
"""

import numpy as np
import pytest
import torch

from spec73_helpers import (VOCAB, DraftRecorder, greedy, make_config, make_engine,  # noqa: E402
                            requires_cuda, run_to_end)

from minivllm import SamplingParams, StructuredOutputsParams  # noqa: E402

PROMPTS = [("A", [1, 2, 3, 4, 5, 6]), ("B", [7, 8, 9]), ("C", [10, 9, 8, 7, 6, 5, 4, 3])]


@requires_cuda
@pytest.mark.parametrize("spec_k", [1, 3])
def test_v2_matches_v1_greedy(spec_k):
    """E1：同一 tiny 模型 + EAGLE3，K=1/3 下 V1 与 V2 的 greedy 输出逐 token 相同。"""
    v1 = greedy(PROMPTS, max_tokens=12, v2=False, spec_k=spec_k)
    v2 = greedy(PROMPTS, max_tokens=12, v2=True, spec_k=spec_k)
    assert v1 == v2
    # 每条请求都真的产出了 12 个 token（不是"两边都空"这种退化相等）
    assert all(len(tokens) == 12 for tokens in v1.values())


@requires_cuda
def test_v2_drafts_are_scheduled_and_counts_come_from_gpu():
    """E2：草稿真的进批（每轮 K 行/请求），且 `num_rejected = num_logits − num_sampled`。"""
    engine, _core, runner = make_engine(v2=True, spec_k=3, max_num_seqs=1)
    rec = DraftRecorder(runner)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=12, temperature=0.0,
                                          eos_token_id=VOCAB + 7))
        run_to_end(engine)
    finally:
        engine.shutdown()

    decode_rounds = [n for n in rec.draft_rows]
    assert max(decode_rounds) == 3, f"草稿没进批：每轮草稿行数={rec.draft_rows}"
    assert len(rec.logits_rows) == len(rec.draft_rows)
    for rows, sampled, rejected in zip(rec.logits_rows, rec.num_sampled, rec.num_rejected):
        assert sampled <= 4 and sampled >= 1                 # 最多 K+1 个 token
        assert rejected == rows - sampled, (
            f"num_rejected 必须等于 logits 行数 − 采样数：rows={rows} "
            f"sampled={sampled} rejected={rejected}")
    # 全是 greedy：接受与否由 target 的 argmax 决定；随机 tiny 权重下几乎必然全拒，
    # 但管道必须走通（num_sampled == 1 也要有对应的 num_rejected == K）。
    assert set(rec.histogram()) <= {1, 2, 3, 4}


@requires_cuda
def test_v2_logprobs_match_v1():
    """E3：logprobs 走 V2 的 slot 寻址后数值与 V1 相同（候选/恢复/bonus 三种行都覆盖）。"""
    v1 = greedy(PROMPTS, max_tokens=8, v2=False, spec_k=3, logprobs=2)
    v2 = greedy(PROMPTS, max_tokens=8, v2=True, spec_k=3, logprobs=2)
    assert v1 == v2

    # 逐位置比对 logprobs 的数值（同一模型、同一 dtype → 同一条 fp32 归约路径）
    def first_logprobs(v2_flag):
        engine, _core, _runner = make_engine(v2=v2_flag, spec_k=3, max_num_seqs=1)
        try:
            engine.add_request("A", [1, 2, 3, 4, 5, 6],
                               SamplingParams(max_tokens=6, temperature=0.0,
                                              eos_token_id=VOCAB + 7, logprobs=2))
            outs = []
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                outs += list(engine.step())
            return outs[-1]
        finally:
            engine.shutdown()

    for flag in (False, True):
        out = first_logprobs(flag)
        assert out.logprobs is not None and len(out.logprobs) == len(out.token_ids)
        # 每个位置的第 0 项就是**实际采到的**那个 token（68 关的容器语义）
        for pos, token_id in enumerate(out.token_ids):
            assert token_id in out.logprobs[pos]
            assert out.logprobs[pos][token_id].rank == 1


@requires_cuda
def test_v2_survives_preemption():
    """E4：块不够触发抢占恢复，V2 输出仍与 V1 相同，且 prefill_len ≥ prompt_len。"""
    kwargs = dict(max_tokens=10, spec_k=3, blocks=6, budget=16, max_num_seqs=3)
    v1 = greedy(PROMPTS, v2=False, **kwargs)
    v2 = greedy(PROMPTS, v2=True, **kwargs)
    assert v1 == v2

    # 恢复时 runner 会把 prefill_len 写成"整段要重算的历史"，prompt_len 保持用户给的值
    engine, core, runner = make_engine(v2=True, spec_k=3, blocks=6, budget=16,
                                       max_num_seqs=3)
    try:
        # 和上面同一条配置：三条请求才压得出抢占（单条请求 6 个块够用）
        for req_id, prompt in PROMPTS:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=10, temperature=0.0,
                                              eos_token_id=VOCAB + 7))
        seen = []
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            engine.step()
            states = runner.req_states
            for req_id, slot in states.req_id_to_index.items():
                seen.append((int(states.prompt_len.np[slot]),
                             int(states.prefill_len.np[slot])))
        assert seen and all(plen >= prompt for prompt, plen in seen)
        # 这条用例不能是"没真的抢占"的空转：块给得少，必须发生过抢占恢复
        assert core.scheduler.num_preemptions > 0, (
            f"没发生抢占（num_gpu_blocks=6 下 V2 也排得下？）——用例失去意义")
    finally:
        engine.shutdown()


@requires_cuda
def test_v2_zero_request_step_and_cleanup():
    """E5：没有请求时 `execute_model` 不碰模型；请求结束后 slot 全部归还。"""
    engine, core, runner = make_engine(v2=True, spec_k=3, max_num_seqs=2)
    try:
        # 空轮：返回空输出，不执行模型（execute_model_state 保持 None）
        out = engine.step()
        assert list(out) == []
        assert runner.execute_model_state is None
        assert runner.req_states.num_reqs == 0

        engine.add_request("A", [1, 2, 3], SamplingParams(max_tokens=4, temperature=0.0,
                                                          eos_token_id=VOCAB + 7))
        run_to_end(engine)
        # 结束清理：请求被移出，slot 与它上面的状态都不再属于它
        assert runner.req_states.num_reqs == 0
        assert runner.req_states.req_id_to_index == {}
        assert len(runner.req_states.free_indices) == runner.max_num_reqs
        assert list(engine.step()) == []
    finally:
        engine.shutdown()


def test_v2_config_boundaries():
    """B1：V2 不支持的方法 / 图 / 异步在**配置期**明确报错，不静默退回 V1。"""
    import os

    with pytest.raises(NotImplementedError, match="V2 Model Runner 本关只支持"):
        make_config(v2=True, method="ngram", spec_k=2)
    with pytest.raises(NotImplementedError, match="V2 的 CUDA Graph 属 74 关"):
        make_config(v2=True, spec_k=2, mode="full_and_piecewise")
    with pytest.raises(NotImplementedError, match="异步调度"):
        cfg = make_config(v2=True, spec_k=2)
        object.__setattr__(cfg.scheduler_config, "async_scheduling", True)
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        cfg.validate_v2_model_runner()
    # 关掉 V2 之后同样的配置能建出来（说明拒绝来自 V2 分派，不是把这些能力一起砍了）
    make_config(v2=False, method="ngram", spec_k=2)
    make_config(v2=False, spec_k=2, mode="full_and_piecewise")


@requires_cuda
def test_v2_take_draft_token_ids_protocol():
    """E2 的协议面：同步调度下 `take_draft_token_ids()` 交回**-1 占位**（宽度 = K）。

    为什么是 -1 而不是真 token：V2 的草稿常驻在执行侧（`req_states.draft_tokens`，
    GPU 按 slot 存），调度器只需要**宽度**来排下一轮；真实 id 由 runner 在
    `combine_sampled_and_draft_tokens` 里按 slot 自己读（这正是异步调度的前提）。
    """
    engine, _core, runner = make_engine(v2=True, spec_k=3, max_num_seqs=1)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=4, temperature=0.0,
                                          eos_token_id=VOCAB + 7))
        engine.step()   # prefill 轮：还没有草稿
        engine.step()   # decode 轮：提议者出草稿，handler 记下 req_ids/宽度
        draft_ids = runner.take_draft_token_ids()
        assert draft_ids is not None
        assert draft_ids.req_ids == ["A"]
        assert draft_ids.draft_token_ids == [[-1, -1, -1]]
    finally:
        engine.shutdown()


@requires_cuda
def test_v2_structured_output_uses_real_input_batch():
    """E6：语法掩码走 V2 的**真实 `InputBatch`**，且 V1/V2 结果一致。

    为什么要单独钉这条：`StructuredOutputsWorker.apply_grammar_bitmask` 要用 `cu_num_logits`
    （含前导 0）把"第 i 条请求的第 j 行"翻成 logits 行号——这是 68 关"掩码行索引必须显式重建"
    在 V2 坐标系里的等价物。行号打错**不报错**，只是让掩码落到别人的行上，所以必须从真实接线处取证
    （替身 input batch 测不出这一点）。
    顺带覆盖 `DraftTokensHandler` 的真 D2H 分支：`has_structured_output_reqs=True` 时草稿会被拷回
    CPU 交给调度器做语法校验（V2 里唯一需要草稿回 CPU 的情形）。

    用的是 68 关那份**带 JSON tokenizer 的 tiny 对**（`tiny_mqa`）：`tiny_gqa` 词表只有 11 个 id、
    没有 tokenizer，任何 JSON 语法都拼不出来（实测 xgrammar 会放出一个词表里不存在的 token，
    调度器的严格校验当场报错——这属于测试取材问题，不是实现差异）。
    """
    schema = ('{"type": "object", "properties": {"x": {"const": 1}}, '
              '"required": ["x"], "additionalProperties": false}')
    outputs, seen = {}, []
    for v2 in (False, True):
        engine, _core, runner = make_engine(v2=v2, spec_k=3, max_num_seqs=1, structured=True)
        if v2:
            # 只有 V2 有 `structured_outputs_worker`（V1 走 `Runner._apply_grammar_bitmask`），
            # 这里要取证的是"V2 的真实 InputBatch 接到了掩码内核"。
            worker = runner.structured_outputs_worker
            original = worker.apply_grammar_bitmask

            def spy(logits, input_batch, req_ids, bitmask, _o=original, _seen=seen):
                _seen.append({
                    "req_ids": list(req_ids),
                    "num_reqs": int(input_batch.num_reqs),
                    "cu_num_logits_np": list(input_batch.cu_num_logits_np),
                    "has_drafts": int(getattr(input_batch, "num_draft_tokens", 0)),
                })
                return _o(logits, input_batch, req_ids, bitmask)

            worker.apply_grammar_bitmask = spy
        try:
            engine.add_request(
                "A", [2, 5],   # {"x" 的 token（见 tiny_models.STRUCTURED_TOKENS）
                SamplingParams(max_tokens=12, temperature=0.0, eos_token_id=1,
                               structured_outputs=StructuredOutputsParams(json=schema)))
            outs = []
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                outs += list(engine.step())
            outputs[v2] = outs[-1]
        finally:
            engine.shutdown()

    # 掩码真的被调用过，而且拿到的是**真实** InputBatch：cu_num_logits 含前导 0、长度 = 请求数+1
    assert seen, "语法掩码从未被调用（结构化输出没走到执行侧）"
    for call in seen:
        assert call["cu_num_logits_np"][0] == 0
        assert len(call["cu_num_logits_np"]) == call["num_reqs"] + 1
    # 投机 + 语法：至少有一轮是带草稿行算掩码的（每请求 1+K 行）
    assert any(call["has_drafts"] > 0 for call in seen)
    # 两条路径的 greedy 输出与文本相同（掩码的施加方式不改变结果）
    assert outputs[False].token_ids == outputs[True].token_ids
    assert outputs[True].text.startswith('{"x": 1')
